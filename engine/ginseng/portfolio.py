"""Fractional sell-only lots and asset-level backstop diagnostics.

Tax costs conservatively price positive lot gains only. No loss rebate,
wash-sale handling, exact tax return, or whole-share execution is assumed.
"""

from datetime import date

import numpy as np
from scipy.optimize import linprog

from ginseng.risk import probabilities
from ginseng.simulate import _joint_history, cash_paths
from ginseng.withdrawals import is_long_term, LONG_TERM_CAPITAL_GAINS_RATE, ORDINARY_INCOME_RATE


def aligned_asset_returns(state) -> tuple[list[str], np.ndarray] | None:
    symbols = [holding.symbol for holding in state.taxable_portfolio]
    histories = {item.symbol: dict(item.daily_returns) for item in state.asset_daily_returns}
    if not symbols or len(histories) != len(state.asset_daily_returns) or any(s not in histories for s in symbols):
        return None
    dates = [ts.date() for ts in _joint_history(state).index]
    for item in state.asset_daily_returns:
        if len(histories[item.symbol]) != len(item.daily_returns):
            return None
    if any(d not in histories[s] for s in symbols for d in dates):
        return None
    matrix = np.array([[histories[s][d] for s in symbols] for d in dates], dtype=float)
    if not np.all(np.isfinite(matrix)) or np.any(matrix <= -1):
        return None
    return symbols, matrix


def ledoit_wolf_covariance(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit–Wolf spherical shrinkage, centered ML covariance (1/n).

    beta estimates the sampling variance of covariance; delta is the
    squared distance from mu*I. The ratio, clipped to [0,1], is the
    shrinkage intensity. Equivalent to the standard centered LW estimator.
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError("Covariance needs aligned, finite asset returns.")
    x = x - x.mean(axis=0)
    n, p = x.shape
    covariance = x.T @ x / n
    mu = np.trace(covariance) / p
    delta = float(np.sum((covariance - mu * np.eye(p))**2))
    beta = float((np.mean(np.sum(x*x, axis=1)**2) - np.sum(covariance**2)) / n)
    shrinkage = float(np.clip(beta / delta, 0, 1)) if delta > 0 else 0.0
    return (1 - shrinkage) * covariance + shrinkage * mu * np.eye(p), shrinkage


def lot_catalog(holdings, as_of: date, long_rate: float, short_rate: float) -> list[dict]:
    lots = []
    for holding in holdings:
        for lot in holding.tax_lots:
            if holding.current_price <= 0 or lot.quantity <= 0:
                continue
            rate = long_rate if is_long_term(lot.purchase_date, as_of) else short_rate
            gain_fraction = 1 - lot.cost_basis_per_share / holding.current_price
            lots.append({"lot_id": lot.lot_id, "symbol": holding.symbol, "price": holding.current_price,
                         "value": lot.market_value(holding.current_price), "gain_fraction": gain_fraction,
                         "tax_per_dollar": max(0, gain_fraction) * rate, "assumed_rate": rate})
    return lots


def sell_only_lp(lots: list[dict], target: float, *, symbol_targets: dict | None = None) -> np.ndarray:
    if not np.isfinite(target) or target < 0 or target > sum(l["value"] for l in lots) + 1e-6:
        raise ValueError("Sale target exceeds available taxable holdings.")
    if not lots:
        return np.zeros(0)
    rows, amounts = [np.ones(len(lots))], [target]
    if symbol_targets is not None:
        for symbol, amount in symbol_targets.items():
            rows.append(np.array([float(l["symbol"] == symbol) for l in lots]))
            amounts.append(amount)
    result = linprog([l["tax_per_dollar"] for l in lots], A_eq=np.array(rows), b_eq=np.array(amounts),
                     bounds=[(0, l["value"]) for l in lots], method="highs",
                     options={"time_limit": 5.0})
    if not result.success or result.x is None or not np.all(np.isfinite(result.x)):
        raise ValueError("Lot allocation could not be solved.")
    if not np.allclose(np.array(rows) @ result.x, amounts, atol=1e-5):
        raise ValueError("Lot allocation failed the proceeds check.")
    return np.maximum(0, result.x)


def portfolio_lab(state, bundle, obligations, target: float, weights=None, long_rate=LONG_TERM_CAPITAL_GAINS_RATE, short_rate=ORDINARY_INCOME_RATE) -> dict:
    aligned = aligned_asset_returns(state)
    if aligned is None:
        return {"status": "unavailable", "message": "Aligned daily returns for every taxable holding are required. Missing returns are not treated as zero."}
    symbols, history = aligned
    covariance, shrinkage = ledoit_wolf_covariance(history)
    values = np.array([h.market_value for h in state.taxable_portfolio])
    if target > values.sum() + 1e-6:
        return {"status": "unavailable", "message": "The requested sale exceeds available taxable holdings."}
    w = probabilities(bundle.n_paths, weights)
    terminal = np.prod(1 + history[bundle.index_matrix], axis=1) - 1
    under_buffer = (state.immediate_funding + cash_paths(state, bundle, obligations)).min(axis=1) < state.operating_buffer
    mass = float(w[under_buffer].sum())
    average = w @ terminal
    conditional = (w[under_buffer] @ terminal[under_buffer]) / mass if mass > 0 else None
    gap = average - conditional if conditional is not None else None

    def risk(remaining):
        total = float(remaining.sum())
        allocation = remaining / total if total > 1e-8 else np.zeros_like(remaining)
        variance = float(allocation @ covariance @ allocation)
        return {"remaining_value": total, "daily_volatility": float(np.sqrt(max(0, variance))),
                "conditional_underperformance": float(allocation @ gap) if gap is not None and total > 1e-8 else None}

    total = values.sum()
    allocation = values / total
    variance = float(allocation @ covariance @ allocation)
    contributions = allocation * (covariance @ allocation) / variance if variance > 0 else np.zeros_like(values)
    lots = lot_catalog(state.taxable_portfolio, state.as_of, long_rate, short_rate)
    variants = []
    if target > 0:
        sales = {
            "Proportional": np.array([target * l["value"] / total for l in lots]),
            "Tax-conscious": sell_only_lp(lots, target),
        }
        if gap is not None:
            remaining = values.copy()
            amount_left = target
            # Re-evaluate the remaining portfolio for every proposed slice.
            # Minimize conditional underperformance, breaking ties by current
            # variance. This is a heuristic, never called globally optimal.
            while amount_left > 1e-6:
                candidates = []
                step = min(target / 100, amount_left, float(remaining[remaining > 1e-8].min()))
                for i, value in enumerate(remaining):
                    if value <= 1e-8:
                        continue
                    amount = min(step, value)
                    after = remaining.copy(); after[i] -= amount
                    outcome = risk(after)
                    candidates.append(((outcome["conditional_underperformance"] or 0, outcome["daily_volatility"]), i, amount))
                _, index, amount = min(candidates)
                remaining[index] -= amount
                amount_left -= amount
            sales["Conditional backstop heuristic"] = sell_only_lp(lots, target, symbol_targets=dict(zip(symbols, values - remaining)))
        for label, dollars in sales.items():
            sold_by_asset = np.array([sum(x for l, x in zip(lots, dollars) if l["symbol"] == symbol) for symbol in symbols])
            variants.append({"label": label, "proceeds": float(dollars.sum()),
                "realized_gain_loss": float(sum(x * l["gain_fraction"] for l, x in zip(lots, dollars))),
                "estimated_positive_gain_tax": float(sum(x * l["tax_per_dollar"] for l, x in zip(lots, dollars))),
                "after": risk(np.maximum(0, values - sold_by_asset)),
                "lots": [{"lot_id": l["lot_id"], "symbol": l["symbol"], "dollars": float(x), "shares": float(x / l["price"]),
                          "realized_gain_loss": float(x * l["gain_fraction"])} for l, x in zip(lots, dollars) if x > 1e-6]})
    return {"status": "ready", "source": "Synthetic aligned asset returns", "history_days": len(history),
            "shrinkage": shrinkage, "covariance": covariance.tolist(), "before": risk(values),
            "cash_pressure_probability": mass, "target_proceeds": target, "variants": variants,
            "tax_assumptions": {"long_term_rate": long_rate, "short_term_rate": short_rate,
                                "label": "Assumed positive-gain cost; not final tax liability. Losses do not create cash rebates."},
            "assets": [{"symbol": symbol, "value": float(values[i]), "risk_contribution": float(contributions[i]),
                        "return_all_paths": float(average[i]), "return_under_cash_pressure": float(conditional[i]) if conditional is not None else None}
                       for i, symbol in enumerate(symbols)]}
