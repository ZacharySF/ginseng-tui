"""Statistical honesty layer: estimate uncertainty and stress testing
(spec sections 18-19, 29-31, 46).

This module answers a question the stochastic forecast in `simulate.py`
and `metrics.py` cannot: *how much does the Required Liquidity Reserve
itself wobble because we only observed a finite, 24-month history?* That
is answered here with an **outer dependent bootstrap** (spec 29-30) and a
**persistence sensitivity table** (spec 18). It also answers a distinct
question the historical resampler structurally cannot: *what happens in a
state historical resampling can never produce?* That is answered with a
**stress test** (spec 19, 46) that is explicitly not probability
calibrated.

Terminology (spec 31 — "Avoid 'Confidence' Collision"):

* the user's chosen quantile `q` is always called the **coverage target**
  (the `coverage_target` parameter/field, matching `metrics.py`);
* statistical estimation uncertainty from finite history is always called
  the **Model-Estimate Range** (a.k.a. Estimate Uncertainty Band) — never
  "confidence," and never the same word used for the coverage target.

Every function here is a pure, deterministic function of an explicit
`seed`, and reuses `simulate.draw_bundle` / `simulate.cash_paths` /
`metrics.*` for every draw and every reserve computation. Nothing in this
module reimplements the Stationary Bootstrap recursion or the RLR/gap
math; see `References/research/bootstrap.md` sections 6 and 8 for the
measured construction this follows.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd

from ginseng.generate import DEFAULT_SEED
from ginseng.metrics import (
    funding_gap,
    required_liquidity_per_path,
    required_liquidity_reserve,
    severity_metrics,
)
from ginseng.simulate import cash_paths, draw_bundle
from ginseng.state import FinancialState, Obligation, Transaction, TransactionType

# ---- Defaults -------------------------------------------------------------
#
# horizon_days / n_paths mirror the rest of the engine's defaults
# (`api.py`'s `ScenarioRequest`, `generate.py`'s `FORECAST_HORIZON`).
#
# n_outer=50 (not the research file's raw-kernel default of 100) because
# this implementation deliberately reuses the full `draw_bundle`/
# `cash_paths` pipeline per outer replicate rather than the minimal numpy
# kernel bootstrap.md measured (so every replicate also re-estimates the
# block length through `arch.optimal_block_length` and rebuilds a
# `FinancialState`). bootstrap.md section 6 documents exactly this
# trade-off: "If latency tightens, cut B_OUTER to 50 before cutting inner
# paths" — inner paths (`n_paths`) stay at 2000 because bootstrap.md's
# measurement shows MC noise is already ~7x smaller than estimate
# uncertainty at that path count, so cutting B_OUTER is the correct lever.
DEFAULT_HORIZON_DAYS = 30
DEFAULT_N_PATHS = 2000
DEFAULT_N_OUTER = 50
DEFAULT_LOW_PERCENTILE = 0.10
DEFAULT_HIGH_PERCENTILE = 0.90
DEFAULT_DROUGHT_DAYS = 45

# Spec 18's two honest verdicts, verbatim.
INSENSITIVE_VERDICT = "Reserve estimate is relatively insensitive to the persistence assumption."
SENSITIVE_VERDICT = "Reserve estimate is sensitive to income-persistence assumptions."

# Documented threshold for `stability_verdict`: relative spread
# (max - min) / mean across the four sensitivity rows. This is a
# hackathon-stage, explicitly stated threshold (not derived from a formal
# hypothesis test) — chosen so that a swing of more than roughly one
# reserve-dollar-in-seven across block-length assumptions is flagged
# sensitive rather than smoothed over. Spec 18: "Do not hide this."
PERSISTENCE_SENSITIVITY_THRESHOLD = 0.15
SENSITIVITY_FIXED_BLOCKS: tuple[int, ...] = (7, 14, 21)

# Spec 46's mandated stress-test label, verbatim.
STRESS_LABEL = "Stress scenario — not probability calibrated"
STRESS_PROBABILITY = "Not estimated"

_VARIABLE_TYPES = frozenset(
    {
        TransactionType.INCOME_VARIABLE,
        TransactionType.EXPENSE_ESSENTIAL_VARIABLE,
        TransactionType.EXPENSE_DISCRETIONARY_VARIABLE,
    }
)


@dataclass(frozen=True)
class EstimateBand:
    """Spec 29-31 payload: `point` is the ordinary Required Liquidity
    Reserve estimate (identical to what `/scenario` reports for the same
    `seed`); `low`/`high` are the outer-dependent-bootstrap percentile
    band around it — the **Model-Estimate Range**. `coverage_target` is
    the user's `q` (spec 31 terminology); it is never called "confidence,"
    and neither is the range.
    """

    point: float
    low: float
    high: float
    coverage_target: float
    mean_block_length: int
    n_outer: int
    n_paths: int
    horizon_days: int
    low_percentile: float
    high_percentile: float
    label: str = "Model-Estimate Range"


@dataclass(frozen=True)
class SensitivityRow:
    """One row of the spec 18 persistence-sensitivity table."""

    block_label: str
    mean_block_length: int
    required_liquidity_reserve: float
    is_estimated: bool
    was_clipped: bool


def _joint_arrays(state: FinancialState) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild the joint history `simulate._joint_history` computes,
    using only `FinancialState.daily_series` (spec 7's public API) so this
    module never depends on `simulate`'s private helpers — only its public
    `draw_bundle`/`cash_paths` entry points."""
    start = state.history_start or min(t.txn_date for t in state.transactions)
    end = state.history_end or state.as_of
    income = state.daily_series(TransactionType.INCOME_VARIABLE, start, end)
    essential = state.daily_series(TransactionType.EXPENSE_ESSENTIAL_VARIABLE, start, end)
    discretionary = state.daily_series(TransactionType.EXPENSE_DISCRETIONARY_VARIABLE, start, end)
    return income.index, income.to_numpy(), essential.to_numpy(), discretionary.to_numpy()


def _synthetic_transactions(
    dates: pd.DatetimeIndex, income: np.ndarray, essential: np.ndarray, discretionary: np.ndarray
) -> tuple[Transaction, ...]:
    """Rebuild only the three variable-series transactions so a resampled
    joint history round-trips back through `daily_series` unchanged.
    Zero-magnitude days are omitted (harmless: `daily_series` zero-fills
    missing days), keeping the resampled ledger's size proportional to the
    real transaction density rather than always `3 * history_days`."""
    label = "outer-bootstrap-resample"
    txns: list[Transaction] = []
    for day, inc, ess, disc in zip(dates, income, essential, discretionary):
        txn_date = day.date()
        if inc != 0.0:
            txns.append(Transaction(txn_date, TransactionType.INCOME_VARIABLE, float(inc), label))
        if ess != 0.0:
            txns.append(Transaction(txn_date, TransactionType.EXPENSE_ESSENTIAL_VARIABLE, float(ess), label))
        if disc != 0.0:
            txns.append(Transaction(txn_date, TransactionType.EXPENSE_DISCRETIONARY_VARIABLE, float(disc), label))
    return tuple(txns)


def _resampled_state(
    state: FinancialState,
    dates: pd.DatetimeIndex,
    income_hist: np.ndarray,
    essential_hist: np.ndarray,
    discretionary_hist: np.ndarray,
    idx_row: np.ndarray,
) -> FinancialState:
    """Spec 29 step 1: "create a resampled historical dataset." Only the
    three stochastic variable series are resampled (fixed income/
    obligations are never resampled, spec 12/20); every other transaction
    (opening balance, fixed income/expenses, credit activity, investment
    lots) is carried over unchanged so `immediate_funding` and every other
    derived `FinancialState` property still mean what they say."""
    kept = tuple(t for t in state.transactions if t.transaction_type not in _VARIABLE_TYPES)
    synthetic = _synthetic_transactions(
        dates, income_hist[idx_row], essential_hist[idx_row], discretionary_hist[idx_row]
    )
    return replace(state, transactions=kept + synthetic)


def estimate_band(
    state: FinancialState,
    obligations: Sequence[Obligation],
    coverage_target: float,
    operating_buffer: float,
    point_estimate: float,
    point_mean_block_length: int,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    n_paths: int = DEFAULT_N_PATHS,
    n_outer: int = DEFAULT_N_OUTER,
    seed: int = DEFAULT_SEED,
    *,
    low_percentile: float = DEFAULT_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_HIGH_PERCENTILE,
) -> EstimateBand:
    """Outer dependent bootstrap for estimate uncertainty (spec 29-31).

    The stochastic forecast (`simulate`/`metrics`) already describes
    uncertainty in *future outcomes* at the chosen `coverage_target`. This
    describes a second, distinct kind of uncertainty: that a 24-month
    history is a finite sample, so the Required Liquidity Reserve *itself*
    is an estimate with a range, not an exact number (spec 29).

    Construction, following bootstrap.md section 6 exactly:

    1. Receive the ordinary point estimate and its data-estimated mean block
       length from the caller's already-rendered scenario bundle. The band is
       therefore anchored to the exact hero number, not to a second Monte
       Carlo draw with the same seed.
    2. Reuse `draw_bundle` pointed at the *history* itself: calling it with
       `horizon_days=len(history)` and `n_paths=n_outer` makes its
       `index_matrix` an `(n_outer, history_length)` block-bootstrap
       resampling of history days — exactly the outer resample spec 29 step 1
       asks for, with zero new sampler code.
    3. For each outer world: rebuild a resampled `FinancialState`, then
       re-estimate the block length and evaluate that world's RLR.
    4. The band is the `[low_percentile, high_percentile]` percentile of
       outer RLR values, widened if necessary to contain the displayed point
       estimate.

    Deterministic in `seed`: every downstream draw uses an integer child
    seed derived from a single `np.random.default_rng(seed)` stream.
    """
    dates, income_hist, essential_hist, discretionary_hist = _joint_arrays(state)
    t_obs = len(income_hist)

    seed_rng = np.random.default_rng(seed)
    outer_seed = int(seed_rng.integers(1, 2**31 - 1))
    inner_seeds = seed_rng.integers(1, 2**31 - 1, size=n_outer)

    # Reuse draw_bundle itself to get an (n_outer, t_obs) block-bootstrap
    # index matrix over *history days* (spec 29 step 1). The displayed
    # scenario's data-estimated persistence is the outer sampler's L0.
    outer_bundle = draw_bundle(
        state, horizon_days=t_obs, n_paths=n_outer, seed=outer_seed, mean_block_length=point_mean_block_length
    )

    rlrs = np.empty(n_outer, dtype=float)
    for b in range(n_outer):
        idx_row = outer_bundle.index_matrix[b]
        resampled_state = _resampled_state(state, dates, income_hist, essential_hist, discretionary_hist, idx_row)
        # mean_block_length=None re-estimates the block structure on this
        # resample (spec 29 step 2).
        inner_bundle = draw_bundle(resampled_state, horizon_days, n_paths, seed=int(inner_seeds[b]))
        inner_matrix = cash_paths(resampled_state, inner_bundle, obligations)
        inner_required = required_liquidity_per_path(inner_matrix, operating_buffer)
        rlrs[b] = required_liquidity_reserve(inner_required, coverage_target)

    band_low, band_high = np.quantile(rlrs, [low_percentile, high_percentile])
    low = float(min(band_low, point_estimate))
    high = float(max(band_high, point_estimate))

    return EstimateBand(
        point=point_estimate,
        low=low,
        high=high,
        coverage_target=coverage_target,
        mean_block_length=point_mean_block_length,
        n_outer=n_outer,
        n_paths=n_paths,
        horizon_days=horizon_days,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )


def persistence_sensitivity(
    state: FinancialState,
    obligations: Sequence[Obligation],
    coverage_target: float,
    operating_buffer: float,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    n_paths: int = DEFAULT_N_PATHS,
    seed: int = DEFAULT_SEED,
) -> list[SensitivityRow]:
    """Persistence sensitivity test (spec 18): the Required Liquidity
    Reserve at fixed mean block lengths of 7, 14, and 21 days, plus the
    data-estimated mean block length — exactly the table spec 18
    prescribes.

    Every row reuses the *same* `seed`, so (per the common-random-numbers
    discipline of spec 20, applied here to isolate one assumption) every
    row draws identical underlying uniforms/restart indices from
    `draw_bundle`'s RNG; only the block-length assumption differs. Any
    difference between rows is therefore attributable to the persistence
    assumption itself, not to independent resampling noise.
    """
    rows: list[SensitivityRow] = []
    for block_length in SENSITIVITY_FIXED_BLOCKS:
        bundle = draw_bundle(state, horizon_days, n_paths, seed=seed, mean_block_length=block_length)
        matrix = cash_paths(state, bundle, obligations)
        required = required_liquidity_per_path(matrix, operating_buffer)
        reserve = required_liquidity_reserve(required, coverage_target)
        rows.append(
            SensitivityRow(
                block_label=f"{block_length}d",
                mean_block_length=block_length,
                required_liquidity_reserve=reserve,
                is_estimated=False,
                was_clipped=False,
            )
        )

    estimated_bundle = draw_bundle(state, horizon_days, n_paths, seed=seed, mean_block_length=None)
    estimated_matrix = cash_paths(state, estimated_bundle, obligations)
    estimated_required = required_liquidity_per_path(estimated_matrix, operating_buffer)
    estimated_reserve = required_liquidity_reserve(estimated_required, coverage_target)
    estimated_suffix = " (capped)" if estimated_bundle.mean_block_length_was_clipped else ""
    rows.append(
        SensitivityRow(
            block_label=f"Estimated {estimated_bundle.mean_block_length}d{estimated_suffix}",
            mean_block_length=estimated_bundle.mean_block_length,
            required_liquidity_reserve=estimated_reserve,
            is_estimated=True,
            was_clipped=estimated_bundle.mean_block_length_was_clipped,
        )
    )
    return rows


def stability_verdict(rows: Sequence[SensitivityRow]) -> str:
    """Spec 18's honest verdict: compare the relative spread of
    `required_liquidity_reserve` across `rows` — `(max - min) / mean` — to
    the documented `PERSISTENCE_SENSITIVITY_THRESHOLD`. Above threshold
    returns the *sensitive* verdict; at or below returns the *insensitive*
    verdict. The sensitive verdict is never suppressed or softened (spec
    18: "Do not hide this.").
    """
    reserves = np.array([row.required_liquidity_reserve for row in rows], dtype=float)
    reserve_range = float(np.max(reserves) - np.min(reserves))
    reserve_mean = float(np.mean(reserves))
    if reserve_mean > 0.0:
        relative_spread = reserve_range / reserve_mean
    else:
        relative_spread = 0.0 if reserve_range == 0.0 else float("inf")
    if relative_spread > PERSISTENCE_SENSITIVITY_THRESHOLD:
        return SENSITIVE_VERDICT
    return INSENSITIVE_VERDICT


def income_drought_stress(
    state: FinancialState,
    obligations: Sequence[Obligation],
    coverage_target: float,
    operating_buffer: float,
    horizon_days: int,
    seed: int = DEFAULT_SEED,
    drought_days: int = DEFAULT_DROUGHT_DAYS,
) -> dict:
    """Stress test (spec 19, 46): the canonical 45-day variable-income
    drought. Variable income is forced to exactly zero for the first
    `drought_days` forecast days; essential and discretionary spending are
    left stochastic and unchanged.

    Spec 14's horizon rule, applied to the scenario itself: the evaluation
    horizon is `max(horizon_days, drought_days, latest obligation landing
    day)` — the full span the scenario creates, never clipped to the
    visible window. The visible chart may stay centered on the first
    `horizon_days` days, but a 45-day drought requested against a 30-day
    chart is evaluated across all 45 days (`applied_drought_days` equals
    the requested `drought_days`), and an obligation due on day 38 does
    not become free by falling outside the chart. No trailing buffer is
    appended past the last touched day: the reserve and severity metrics
    below take running extremes over the horizon, so the funding trough is
    already fully captured on the last day anything material happens. The
    payload reports both windows (`horizon_days` vs
    `evaluation_horizon_days`) so a caller can render the visible chart
    without mistaking it for the evaluated span.

    Spec 19 draws a hard line between a *probabilistic forecast*
    ("what happens if the historical process broadly continues," which
    historical resampling can answer) and a *stress test* ("deliberately
    adverse and not probability calibrated," which historical resampling
    structurally cannot produce if the drought never occurred in
    history). This is the latter: the payload carries
    `probability_calibrated=False` and the verbatim spec 46 UI label so no
    caller can present it as a probability-weighted outcome.

    Implementation reuses `draw_bundle`/`cash_paths` for the baseline
    forecast unmodified; the only new arithmetic is the horizon extension
    above and subtracting the cumulative income contribution during the
    drought window (using the same `index_matrix` `draw_bundle` produced,
    so the removed income is bit-for-bit the same draws `cash_paths` used
    to build the baseline). Reserve/gap/severity still come from
    `metrics.py`, not reimplemented.
    """
    _dates, income_hist, _essential_hist, _discretionary_hist = _joint_arrays(state)

    # Spec 14: max(requested horizon, the full span the scenario creates)
    # plus any trailing days the passed obligations need. `max(1, ...)`
    # mirrors `simulate._deterministic_daily_flow`'s landing-day rule
    # exactly, so the evaluated horizon covers precisely the days those
    # obligations can land on.
    latest_obligation_day = max(
        (max(1, obligation.due_in_days) for obligation in obligations), default=0
    )
    evaluation_horizon = max(horizon_days, drought_days, latest_obligation_day)

    bundle = draw_bundle(state, evaluation_horizon, DEFAULT_N_PATHS, seed=seed)
    baseline_matrix = cash_paths(state, bundle, obligations)

    window_days = max(0, min(drought_days, evaluation_horizon))
    income_draws = income_hist[bundle.index_matrix]  # same gather cash_paths performed internally
    income_removed = np.zeros_like(income_draws)
    income_removed[:, :window_days] = income_draws[:, :window_days]
    stressed_matrix = baseline_matrix - np.cumsum(income_removed, axis=1)

    immediate_funding = state.immediate_funding
    required = required_liquidity_per_path(stressed_matrix, operating_buffer)
    reserve = required_liquidity_reserve(required, coverage_target)
    gap = funding_gap(reserve, immediate_funding)
    severity = severity_metrics(stressed_matrix, immediate_funding, operating_buffer)

    return {
        "scenario": f"{drought_days}-day variable-income drought",
        "variable_income_shock": f"0 for {window_days} days",
        "drought_days": drought_days,
        "applied_drought_days": window_days,
        "horizon_days": horizon_days,
        "evaluation_horizon_days": evaluation_horizon,
        "coverage_target": coverage_target,
        "required_liquidity_reserve": reserve,
        "funding_gap": gap,
        "severity": severity,
        "probability": STRESS_PROBABILITY,
        "probability_calibrated": False,
        "label": STRESS_LABEL,
    }
