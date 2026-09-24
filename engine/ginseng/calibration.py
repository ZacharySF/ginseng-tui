"""Walk-forward evidence, using only history available before each forecast.

Fixed monthly flows are inferred from repeated, same-day-of-month training
transactions. Realized liquidity includes those flow types and the same three
variable series as the forecast. One-off user scenarios are not backtested.
"""

from collections import defaultdict
from dataclasses import replace
from datetime import timedelta

import numpy as np
from scipy.special import xlogy
from scipy.stats import binomtest, chi2

from ginseng.metrics import required_liquidity_per_path, required_liquidity_reserve
from ginseng.risk import cvar, quantile
from ginseng.simulate import cash_paths, draw_bundle
from ginseng.state import FinancialState, Obligation, TransactionType as T

MIN_TRAINING_DAYS = 365
VARIABLE_TYPES = (T.INCOME_VARIABLE, T.EXPENSE_ESSENTIAL_VARIABLE, T.EXPENSE_DISCRETIONARY_VARIABLE)
FIXED_TYPES = (T.INCOME_FIXED, T.EXPENSE_FIXED, T.CREDIT_PAYMENT)


def randomized_pit(predictions: np.ndarray, realized: float, rng: np.random.Generator) -> float:
    below = float(np.mean(predictions < realized))
    through = float(np.mean(predictions <= realized))
    return float(rng.uniform(below, through))


def pinball_loss(predicted_quantile: float, realized: float, q: float) -> float:
    error = realized - predicted_quantile
    return float(max(q * error, (q - 1) * error))


def empirical_crps(predictions: np.ndarray, realized: float) -> float:
    x = np.sort(predictions)
    n = len(x)
    # E|X-y| - 0.5 E|X-X'| without an n-by-n matrix.
    return float(np.mean(np.abs(x - realized)) - np.dot(2 * np.arange(1, n + 1) - n - 1, x) / n**2)


def training_state(state: FinancialState, cutoff, horizon: int) -> FinancialState:
    transactions = tuple(t for t in state.transactions if t.txn_date <= cutoff)
    groups = defaultdict(list)
    for txn in transactions:
        if txn.transaction_type in FIXED_TYPES:
            groups[(txn.transaction_type, txn.label)].append(txn)
    incomes, bills = [], []
    for (kind, label), values in groups.items():
        values.sort(key=lambda t: t.txn_date)
        # Require repeated monthly evidence; never read a future payment to
        # reconstruct what supposedly was known at this historical cutoff.
        recent = values[-6:]
        if len(recent) < 3 or len({t.txn_date.day for t in recent}) != 1:
            continue
        for day in range(1, horizon + 1):
            if (cutoff + timedelta(days=day)).day == recent[-1].txn_date.day:
                item = Obligation(f"{label}-{day}", label, recent[-1].amount, day, kind)
                (incomes if kind == T.INCOME_FIXED else bills).append(item)
    return replace(state, as_of=cutoff, transactions=transactions,
                   fixed_income_schedule=tuple(incomes), fixed_obligations=tuple(bills),
                   holdings=(), credit_accounts=(), planned_discretionary_events=(),
                   portfolio_daily_returns=(), asset_daily_returns=(), forecast_horizon=horizon,
                   history_end=cutoff if state.history_end is not None else None)


def expected_shortfall_test(rows: list[dict], predictions: list[np.ndarray], q: float, seed: int) -> dict:
    """Acerbi–Székely Z2 with the paper's non-continuous I' correction.

    Losses here are positive, so Z2 = 1 - mean(loss * I' / ES) / alpha.
    The conditional null is simulated from the archived forecast distributions,
    assuming independent held-out windows. No fixed regulatory cutoff is used.
    """
    alpha = 1 - q
    n = len(rows)
    if alpha <= 0 or n * alpha < 20 or sum(not r["covered"] for r in rows) < 20:
        return {"status": "unavailable", "reason": "Expected-shortfall test unavailable: fewer than 20 expected or observed held-out tail events."}
    if len(predictions) != n:
        return {"status": "unavailable", "reason": "Expected-shortfall test requires the archived predictive distributions."}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7919]))
    null_size = 1000
    simulated_sum = np.zeros(null_size)
    observed_sum = 0.0
    for row, predicted in zip(rows, predictions):
        cutoff = quantile(predicted, q)
        es = cvar(predicted, q)
        if es <= 0:
            return {"status": "unavailable", "reason": "Expected-shortfall normalization is undefined for a zero predicted tail loss."}
        mass_above = float(np.mean(predicted > cutoff))
        mass_at = float(np.mean(predicted == cutoff))
        boundary_fraction = float(np.clip((alpha - mass_above) / mass_at, 0, 1))
        def tail_indicator(values):
            return (values > cutoff).astype(float) + boundary_fraction * (values == cutoff)
        realized = np.array([row["realized_required"]])
        observed_sum += float((realized * tail_indicator(realized) / es)[0])
        simulated = rng.choice(predicted, size=null_size)
        simulated_sum += simulated * tail_indicator(simulated) / es
    observed = 1 - observed_sum / (n * alpha)
    null = 1 - simulated_sum / (n * alpha)
    return {"status": "computed", "statistic": float(observed),
            "p_value": float((1 + np.sum(null <= observed)) / (null_size + 1)),
            "method": "Acerbi–Székely Z2 with fractional boundary atoms",
            "null_simulations": null_size,
            "caveat": "One-sided underestimation test using 1,000 conditional model simulations; assumes independent windows. No PASS verdict."}


def formal_tests(rows: list[dict], q: float, predictions: list[np.ndarray] | None = None, seed: int = 0) -> dict:
    n = len(rows)
    expected = n * (1 - q)
    unavailable = {"status": "unavailable", "reason": "Formal tail test unavailable: insufficient independent observations."}
    tests = {name: dict(unavailable) for name in ("kupiec", "christoffersen", "acerbi_szekely")}
    if expected < 5 or n * q < 5:
        return tests
    failures = np.array([not row["covered"] for row in rows], dtype=int)
    count = int(failures.sum())
    likelihood = xlogy(count, 1 - q) + xlogy(n - count, q)
    fitted = xlogy(count, count / n) + xlogy(n - count, 1 - count / n)
    tests["kupiec"] = {"status": "computed", "statistic": float(2 * (fitted - likelihood)),
                       "p_value": float(chi2.sf(2 * (fitted - likelihood), 1)),
                       "caveat": "Assumes independent windows; no PASS verdict. Ties can make nominal coverage conservative."}
    transitions = np.zeros((2, 2), dtype=int)
    for a, b in zip(failures[:-1], failures[1:]):
        transitions[a, b] += 1
    if np.min(transitions) >= 5:
        transition_p = transitions[:, 1] / transitions.sum(axis=1)
        pooled_p = transitions[:, 1].sum() / transitions.sum()
        independent = np.sum(xlogy(transitions[:, 1], pooled_p) + xlogy(transitions[:, 0], 1 - pooled_p))
        markov = np.sum(xlogy(transitions[:, 1], transition_p) + xlogy(transitions[:, 0], 1 - transition_p))
        tests["christoffersen"] = {"status": "computed", "statistic": float(2 * (markov - independent)),
                                   "p_value": float(chi2.sf(2 * (markov - independent), 1)),
                                   "caveat": "Non-overlapping windows only; no PASS verdict."}
    tests["acerbi_szekely"] = expected_shortfall_test(rows, predictions or [], q, seed)
    return tests


def walk_forward(state: FinancialState, horizon: int, paths: int, seed: int, q: float, buffer: float,
                 *, training_days=MIN_TRAINING_DAYS, source="demo", max_windows=None) -> dict:
    start = state.history_start or min(t.txn_date for t in state.transactions)
    end = state.history_end or state.as_of
    history_days = (end - start).days + 1
    def bounded(values):
        values = list(values)
        if max_windows is not None and len(values) > max_windows:
            return [values[i] for i in np.linspace(0, len(values) - 1, max_windows, dtype=int)]
        return values
    primary_starts = bounded(range(training_days, history_days - horizon + 1, horizon))
    spacings = sorted({max(1, horizon // 2), horizon, 2 * horizon})
    starts_by_spacing = {s: bounded(range(training_days, history_days - horizon + 1, s)) for s in spacings}
    descriptive_starts = bounded(range(training_days, history_days - horizon + 1, 7))
    starts = sorted(set(descriptive_starts).union(*(set(v) for v in starts_by_spacing.values())))
    realized_daily = np.zeros(history_days)
    for kind in (*VARIABLE_TYPES, *FIXED_TYPES):
        sign = 1 if kind in (T.INCOME_VARIABLE, T.INCOME_FIXED) else -1
        realized_daily += sign * state.daily_series(kind, start, end).to_numpy()
    by_start = {}
    primary_predictions = []
    for offset in starts:
        cutoff = start + timedelta(days=offset - 1)
        training = training_state(state, cutoff, horizon)
        draw_seed, pit_seed = np.random.SeedSequence([seed, offset, 719]).generate_state(2)
        bundle = draw_bundle(training, horizon, paths, int(draw_seed))
        predicted = required_liquidity_per_path(cash_paths(training, bundle), buffer)
        if offset in primary_starts:
            primary_predictions.append(predicted)
        realized = float(required_liquidity_per_path(np.cumsum(realized_daily[offset:offset + horizon])[None, :], buffer)[0])
        reserve = required_liquidity_reserve(predicted, q)
        by_start[offset] = {
            "training_cutoff": cutoff.isoformat(),
            "start": (cutoff + timedelta(days=1)).isoformat(),
            "end": (cutoff + timedelta(days=horizon)).isoformat(),
            "predicted_reserve": reserve, "realized_required": realized,
            "covered": realized <= reserve, "pit": randomized_pit(predicted, realized, np.random.default_rng(int(pit_seed))),
            "pinball_loss": pinball_loss(reserve, realized, q), "crps": empirical_crps(predicted, realized),
            "predicted_expected_shortfall": cvar(predicted, q),
            "in_primary_sample": offset in primary_starts,
        }

    def sample(offsets, spacing):
        rows = [by_start[o] for o in offsets]
        n = len(rows)
        covered = sum(row["covered"] for row in rows)
        interval = binomtest(covered, n).proportion_ci() if n and spacing >= horizon else None
        return {"spacing_days": spacing, "overlapping": spacing < horizon, "windows": n, "covered": covered,
                "observed_coverage": covered / n if n else None,
                "interval": {"low": interval.low, "high": interval.high} if interval else None,
                "mean_pinball_loss": float(np.mean([r["pinball_loss"] for r in rows])) if n else None,
                "mean_crps": float(np.mean([r["crps"] for r in rows])) if n else None}

    primary = sample(primary_starts, horizon)
    primary_rows = [by_start[o] for o in primary_starts]
    pit_counts, pit_edges = np.histogram([r["pit"] for r in primary_rows], bins=np.linspace(0, 1, 11))
    return {
        "status": "ready" if primary_starts else "insufficient_history",
        "source": ("Synthetic history; this does not validate a real household." if source == "demo"
                   else "Retrospective evaluation using current classified records filtered by effective date; historical arrival and revision times are unavailable."),
        "information_timing": "retrospective_current_records",
        "target": "Starting cash required to preserve the buffer at each end of day, including routine variable flows and fixed monthly flows inferred from prior records.",
        "excluded": "Irregular expenses, investment transactions, transfers, and today's added scenario events are outside this historical target.",
        "history_days": history_days, "training_days_minimum": training_days,
        "horizon_days": horizon, "paths_per_forecast": paths, "nominal_coverage": q,
        "primary": primary, "spacing_results": [sample(v, s) for s, v in starts_by_spacing.items()],
        "expected_tail_failures": len(primary_rows) * (1 - q),
        "interval_note": "95% exact binomial reference interval assumes independent windows. Non-overlap removes shared days, but serial dependence may remain.",
        "formal_tests": formal_tests(primary_rows, q, primary_predictions, seed),
        "pit": {"counts": pit_counts.tolist(), "edges": pit_edges.tolist(), "method": "Randomized PIT, non-overlapping windows only; descriptive at this sample size."},
        "windows": list(by_start.values()), "descriptive_windows": len(by_start),
    }
