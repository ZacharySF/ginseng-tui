"""Liquidity metrics (spec sections 21-27, 32, 34, 69).

All quantities are computed from `X_{j,t}`, the cumulative future net cash
flow per simulated path (spec 21), and `B_{j,t} = C + X_{j,t}`, simulated
available cash. `R_j`, the per-path required liquidity, always uses the
*running* trajectory over the whole horizon rather than the terminal value,
so a path that dips negative mid-horizon and recovers still registers a
requirement (spec 23).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from ginseng.risk import MONEY_TOLERANCE, probabilities, quantile, quantiles

from ginseng.simulate import (
    DrawBundle,
    PathBundle,
    cash_paths as _compute_cash_paths,
    known_flows,
    portfolio_value_paths,
)
from ginseng.state import FinancialState, Obligation
from ginseng.execution import execution_scope, current_context, prepare_scenario


def required_liquidity_per_path(cash_matrix: np.ndarray, operating_buffer: float) -> np.ndarray:
    """`R_j = max(0, max_t(b - X_{j,t}))` (spec 23), using the running
    maximum over the full horizon rather than only the terminal day."""
    return np.maximum(0.0, operating_buffer - np.min(cash_matrix, axis=1))


def required_liquidity_reserve(required_per_path: np.ndarray, coverage_target: float, weights: np.ndarray | None = None) -> float:
    """`RLR_q = Q_q(R)` (spec 24)."""
    return quantile(required_per_path, coverage_target, weights)


def funding_gap(reserve: float, immediate_funding: float) -> float:
    """`Gap_q = max(0, RLR_q - C)` (spec 26): the hero number."""
    return max(0.0, reserve - immediate_funding)


def coverage_at_funding(required_per_path: np.ndarray, funding: float, weights: np.ndarray | None = None) -> float:
    """Probability that `funding` dollars of immediate funding would have
    been enough to keep every simulated path above the operating buffer."""
    w = probabilities(len(required_per_path), weights)
    covered = required_per_path <= funding
    if np.all(covered[w > 0]):
        return 1.0
    return float(np.clip(np.dot(w, covered), 0, 1))


def severity_metrics(
    cash_matrix: np.ndarray, immediate_funding: float, operating_buffer: float, weights: np.ndarray | None = None
) -> dict:
    """Spec 27: cash-shortfall probability (hard zero floor), average
    deficit conditional on being short, and dollar-days below the operating
    buffer."""
    available_cash = immediate_funding + cash_matrix  # B_{j,t}
    min_cash_per_path = np.min(available_cash, axis=1)
    w = probabilities(len(cash_matrix), weights)
    cash_shortfall_probability = float(w[min_cash_per_path < -MONEY_TOLERANCE].sum() / w.sum())

    max_deficit_per_path = np.maximum(0.0, -min_cash_per_path)  # H_j (spec 27.2)
    short_mask = max_deficit_per_path > MONEY_TOLERANCE
    avg_cash_deficit_when_short = (
        float(np.dot(w[short_mask], max_deficit_per_path[short_mask]) / w[short_mask].sum()) if cash_shortfall_probability > 0 else 0.0
    )

    below_buffer = np.maximum(0.0, operating_buffer - available_cash)
    dollar_days_below_buffer = float(np.dot(w, np.sum(below_buffer, axis=1)))

    return {
        "cash_shortfall_probability": cash_shortfall_probability,
        "avg_cash_deficit_when_short": avg_cash_deficit_when_short,
        "expected_max_cash_deficit": float(np.dot(w, max_deficit_per_path)),
        "dollar_days_below_buffer": dollar_days_below_buffer,
    }


def coverage_curve(
    required_per_path: np.ndarray, immediate_funding: float, n_points: int = 41, weights: np.ndarray | None = None
) -> list[dict]:
    """Liquidity coverage curve (spec 32): immediate funding on the x-axis,
    probability of staying above the operating buffer on the y-axis."""
    max_required = float(np.max(required_per_path)) if required_per_path.size else 0.0
    upper = max(max_required, immediate_funding, 1.0) * 1.2
    grid = np.linspace(0.0, upper, n_points)
    return [
        {"funding": float(f), "coverage": coverage_at_funding(required_per_path, float(f), weights)}
        for f in grid
    ]


def reserve_buffer_curve(
    cash_matrix: np.ndarray, coverage_target: float, active_buffer: float, n_points: int = 41, weights: np.ndarray | None = None, *, minima=None
) -> list[dict]:
    """Reserve-vs-buffer curve (spec 69): operating buffer on the x-axis,
    required liquidity reserve on the y-axis, swept over the same cash
    matrix and coverage target as the displayed reserve. The sweep spans
    zero to `max(2000, 2 * active_buffer)` and contains the active buffer
    exactly, so the curve always passes through the displayed point."""
    minima = np.min(cash_matrix, axis=1) if minima is None else minima
    upper = max(2000.0, active_buffer * 2.0)
    grid = np.unique(np.append(np.linspace(0.0, upper, n_points), active_buffer))
    return [
        {
            "operating_buffer": float(buffer),
            "required_liquidity_reserve": required_liquidity_reserve(
                np.maximum(0.0, float(buffer) - minima), coverage_target, weights
            ),
        }
        for buffer in grid
    ]


def wrong_way_risk(
    cash_matrix: np.ndarray,
    portfolio_value_matrix: np.ndarray,
    immediate_funding: float,
    operating_buffer: float,
    initial_portfolio_value: float,
    weights: np.ndarray | None = None,
) -> dict:
    """Spec 8.4: whether the portfolio underperforms in exactly the paths
    where cash runs short - the conditioning that makes liquidating into a
    downturn expensive precisely when it is needed.

    A path is *forced* when available cash `B_{j,t} = C + X_{j,t}` dips
    below the operating buffer on any day; the metric compares the mean
    terminal portfolio return across all paths against the same mean over
    only the forced paths. `portfolio_return_when_forced` is None when no
    path is ever forced.
    """
    available = immediate_funding + cash_matrix
    forced_mask = np.any(available < operating_buffer, axis=1)
    terminal_return = (
        (portfolio_value_matrix[:, -1] - initial_portfolio_value)
        / max(initial_portfolio_value, 1e-9)
    )
    w = probabilities(len(cash_matrix), weights)
    forced_mass = float(w[forced_mask].sum())
    avg_all = float(np.dot(w, terminal_return))
    avg_forced = float(np.dot(w[forced_mask], terminal_return[forced_mask]) / forced_mass) if forced_mass > 0 else None
    return {
        "fraction_forced_to_sell": forced_mass,
        "portfolio_return_all_paths": avg_all,
        "portfolio_return_when_forced": avg_forced,
        "wrong_way_risk_present": bool(avg_forced is not None and avg_forced < avg_all),
    }


def shortfall_distribution(
    cash_matrix: np.ndarray, immediate_funding: float, n_bins: int = 30, weights: np.ndarray | None = None, *, minimum_balance=None
) -> dict:
    """Distribution of path-minimum cash positions (spec 34)."""
    min_cash_per_path = np.min(immediate_funding + cash_matrix, axis=1) if minimum_balance is None else minimum_balance
    counts, edges = np.histogram(min_cash_per_path, bins=n_bins)
    mass, _ = np.histogram(min_cash_per_path, bins=edges, weights=probabilities(len(cash_matrix), weights))
    return {"bin_edges": edges.tolist(), "counts": counts.tolist(), "probabilities": mass.tolist()}


def percentile_cash_paths(cash_matrix: np.ndarray, immediate_funding: float, weights: np.ndarray | None = None) -> dict:
    """Per-day p10/p50/p90 of simulated available cash `B_{j,t}`, for the
    coverage-curve and cash-path charts."""
    from ginseng.execution import current_context, EvaluationContext
    context = current_context()
    if context is not None:
        return context.charts(cash_matrix, immediate_funding, weights)
    with EvaluationContext() as context:
        return context.charts(cash_matrix, immediate_funding, weights)



@dataclass(frozen=True)
class ScenarioMetrics:
    """The metrics portion of the `/scenario` response contract."""

    required_liquidity_reserve: float
    funding_gap: float
    coverage_at_current_funding: float
    severity: dict
    coverage_curve: list
    reserve_buffer_curve: list
    cash_paths: dict
    shortfall_distribution: dict
    wrong_way_risk: dict | None = None  # None when no market history / marketable assets


@execution_scope
def compute_scenario_metrics(
    state: FinancialState,
    bundle: DrawBundle | PathBundle,
    obligations: Sequence[Obligation],
    coverage_target: float,
    operating_buffer: float,
    weights: np.ndarray | None = None,
) -> ScenarioMetrics:
    """Compute every metric in the `/scenario` contract for one draw bundle
    and one set of obligations. Paired plan comparisons reuse the same
    `bundle` and change only `obligations` (spec 20)."""
    context = current_context()
    prepared = prepare_scenario(state, bundle, obligations)
    execution = context.evaluate(prepared, state.immediate_funding, operating_buffer)
    matrix = execution.paths
    stats = execution.statistics
    context.publish_paths(prepared, matrix)
    weights = context.weights(bundle.n_paths, weights)
    required_per_path = np.maximum(0., operating_buffer - stats.minima)
    immediate_funding = state.immediate_funding
    reserve = required_liquidity_reserve(required_per_path, coverage_target, weights)

    pv_matrix = portfolio_value_paths(state, bundle)
    wwr = (
        wrong_way_risk(
            matrix, pv_matrix, immediate_funding,
            operating_buffer, state.marketable_backup_capital, weights,
        )
        if pv_matrix is not None
        else None
    )

    known_income, known_obligations = known_flows(
        state,
        obligations,
        bundle.horizon_days,
        bundle if isinstance(bundle, PathBundle) else None,
    )
    paths = {
        "days": list(range(1, bundle.horizon_days + 1)),
        **percentile_cash_paths(matrix, immediate_funding, weights),
        "known_income": known_income.tolist(),
        "known_obligations": known_obligations.tolist(),
    }

    return ScenarioMetrics(
        required_liquidity_reserve=reserve,
        funding_gap=funding_gap(reserve, immediate_funding),
        coverage_at_current_funding=coverage_at_funding(required_per_path, immediate_funding, weights),
        severity=severity_from_statistics(stats, weights),
        coverage_curve=coverage_curve(required_per_path, immediate_funding, weights=weights),
        reserve_buffer_curve=reserve_buffer_curve(matrix, coverage_target, operating_buffer, weights=weights, minima=stats.minima),
        cash_paths=paths,
        shortfall_distribution=shortfall_distribution(matrix, immediate_funding, weights=weights, minimum_balance=stats.minimum_balance),
        wrong_way_risk=wwr,
    )


def cash_risk_summary(cash_matrix, opening_cash, operating_buffer, coverage_target, weights=None):
    """Minimal end-of-day reduction shared by every numerical sampler."""
    x = np.asarray(cash_matrix, dtype=float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.all(np.isfinite(x)):
        raise ValueError("Cash paths must be a finite nonempty paths-by-days matrix.")
    if not np.all(np.isfinite([opening_cash, operating_buffer, coverage_target])) or not 0 <= coverage_target <= 1:
        raise ValueError("Cash and buffer must be finite; coverage must be in [0, 1].")
    w = probabilities(len(x), weights)
    minima = x.min(axis=1)
    required = np.maximum(0, operating_buffer - minima)
    deficits = np.maximum(0, -(opening_cash + minima))
    reserve = quantile(required, coverage_target, w)
    failure = float(np.clip(w[deficits > 0].sum(), 0.0, 1.0))
    mean = float(np.dot(w, deficits))
    return {"required_liquidity_reserve": reserve,
            "cash_shortfall_probability": failure,
            "expected_max_cash_deficit": mean,
            "avg_cash_deficit_when_short": mean / failure if failure else 0.0,
            "funding_gap": max(0.0, reserve - opening_cash)}


def severity_from_statistics(stats, weights):
    """App contract keeps its established micro-dollar solver tolerance."""
    short = stats.maximum_deficit > MONEY_TOLERANCE
    mass = float(weights[short].sum() / weights.sum())
    return dict(cash_shortfall_probability=mass,
        avg_cash_deficit_when_short=float(weights[short] @ stats.maximum_deficit[short] / weights[short].sum()) if mass else 0.,
        expected_max_cash_deficit=float(weights @ stats.maximum_deficit),
        dollar_days_below_buffer=float(weights @ stats.buffer_dollar_days))
