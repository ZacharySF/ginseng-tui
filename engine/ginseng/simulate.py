"""Stationary block bootstrap over the joint income/spending series
(spec sections 15-17, 20).

The joint series
    Y_t = [variable income_t, essential variable spending_t, discretionary
    spending_t]
is resampled with shared time indices so cross-series dependence between
uncertain income and spending survives the resample (spec 15). Fixed known
flows (fixed income, fixed obligations, inserted future obligations) are
added deterministically and are never resampled (spec 12, 20).

Every consumer of the stochastic forecast takes a `DrawBundle` so that
paired plan comparisons can later reuse identical draws under common random
numbers (spec 20).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ginseng.state import FinancialState, Obligation, TransactionType

MIN_MEAN_BLOCK_LENGTH = 7
MAX_MEAN_BLOCK_LENGTH = 28


@dataclass(frozen=True)
class DrawBundle:
    """A reusable set of bootstrap draws (spec section 20).

    `index_matrix` has shape `(n_paths, horizon_days)`; entry `[j, t]` is the
    historical day index resampled for path `j` at forecast day `t + 1`.
    `bootstrap_draw_id` is a stable hex digest of `index_matrix`, so any two
    consumers holding the same bundle are provably evaluating identical
    stochastic realizations.
    """

    seed: int
    horizon_days: int
    n_paths: int
    mean_block_length: int
    mean_block_length_was_clipped: bool
    history_length: int
    index_matrix: np.ndarray
    bootstrap_draw_id: str
    sampler: str = "legacy_mc"
    material_indices: np.ndarray | None = None
    sampling_metadata: tuple = ()
    requested_mean_block_length: int | None = None
    initial_block_lengths: np.ndarray | None = None

    def __post_init__(self):
        # Immutable bytes backing also prevents callers re-enabling writes.
        for name in ("index_matrix", "material_indices", "initial_block_lengths"):
            value = getattr(self, name)
            if value is not None:
                from ginseng.execution import snapshot
                if np.asarray(value).dtype != np.dtype('int64'):
                    raise ValueError('Draw indices must use native int64 without implicit conversion.')
                object.__setattr__(self, name, snapshot(value, '<i8'))

@dataclass(frozen=True)
class PathBundle:
    """Precomputed prospective paths for models without a historical bootstrap.

    Assumption-based forecasts generate their stochastic paths prospectively
    and scheduled forecasts are deterministic.  Both retain one reusable
    bundle so baseline, preview, funding, and optimizer calculations share
    exactly the same draws without inventing a ledger history.  Arrays carry
    a longer material horizon and are sliced by ``horizon_days``; funding can
    therefore evaluate real settlement/payment dates beyond the visible chart
    without drawing a second world.
    """

    source: str
    seed: int
    horizon_days: int
    n_paths: int
    bootstrap_draw_id: str
    daily_cash_flows: np.ndarray
    known_income_daily: np.ndarray
    known_obligation_daily: np.ndarray
    discretionary_daily: np.ndarray
    portfolio_values: np.ndarray | None = None
    mean_block_length: int = 0
    mean_block_length_was_clipped: bool = False
    history_length: int = 0

    def __post_init__(self) -> None:
        full_horizon = self.daily_cash_flows.shape[1] if self.daily_cash_flows.ndim == 2 else 0
        expected_paths = self.daily_cash_flows.shape[0] if self.daily_cash_flows.ndim == 2 else 0
        if (
            self.horizon_days < 1
            or self.horizon_days > full_horizon
            or expected_paths != self.n_paths
            or self.known_income_daily.shape != (full_horizon,)
            or self.known_obligation_daily.shape != (full_horizon,)
            or self.discretionary_daily.shape != (self.n_paths, full_horizon)
        ):
            raise ValueError("Path bundle arrays do not match their declared horizon.")
        if self.portfolio_values is not None and self.portfolio_values.shape != (
            self.n_paths,
            full_horizon,
        ):
            raise ValueError("Portfolio paths do not match the cash-path bundle.")

        from ginseng.execution import snapshot
        for name in ('daily_cash_flows','known_income_daily','known_obligation_daily','discretionary_daily','portfolio_values'):
            value = getattr(self,name)
            if value is not None:
                object.__setattr__(self,name,snapshot(value))

    @property
    def available_horizon_days(self) -> int:
        return self.daily_cash_flows.shape[1]


def direct_path_draw_id(source: str, seed: int, n_paths: int, horizon_days: int) -> str:
    """Stable CRN identity for direct prospective path generators.

    Deterministic schedule changes deliberately do not enter the digest:
    paired what-if comparisons differ only in their known flows while using
    the same generated innovations.
    """

    material = f"{source}:{seed}:{n_paths}:{horizon_days}".encode()
    return hashlib.sha256(material).hexdigest()


def _uncached_joint_history(state: FinancialState) -> pd.DataFrame:
    """The daily joint series Y_t (spec 15) over the full recorded ledger
    window. Irregular expenses are excluded, per spec 10."""
    if not state.transactions and state.history_start is None:
        raise ValueError("Classified history is required to draw a bootstrap bundle.")
    start = state.history_start or min(t.txn_date for t in state.transactions)
    end = state.history_end or state.as_of
    if start > end:
        raise ValueError("Classified history cannot begin after its recorded end date.")
    return pd.DataFrame(
        {
            "variable_income": state.daily_series(TransactionType.INCOME_VARIABLE, start, end),
            "essential_variable_spending": state.daily_series(
                TransactionType.EXPENSE_ESSENTIAL_VARIABLE, start, end
            ),
            "discretionary_spending": state.daily_series(
                TransactionType.EXPENSE_DISCRETIONARY_VARIABLE, start, end
            ),
        }
    )


def _joint_history(state: FinancialState) -> pd.DataFrame:
    from ginseng.execution import current_context
    from ginseng.provenance import digest
    from dataclasses import asdict
    context = current_context()
    if context is None:
        return _uncached_joint_history(state)
    # Includes reconciliation transactions and dates; opening-cash edits that
    # change history cannot accidentally hit a cash-only dependency key.
    key = ('history', digest((tuple(asdict(t) for t in state.transactions), state.history_start, state.history_end, state.as_of)))
    def create():
        context.counters['history_preparations'] += 1
        return _uncached_joint_history(state)
    return context.remember(key, create, lambda frame: int(frame.memory_usage(deep=True).sum()))


def _fallback_block_length(z: np.ndarray) -> float:
    """Documented fallback mean block length when `arch`'s Politis-White
    estimator is unavailable: convert the lag-1 autocorrelation of the
    composite net-flow series into an expected geometric block length,
    `1 / (1 - |rho_1|)`, the mean run length of an AR(1)-equivalent process
    with that autocorrelation.
    """
    centered = z - z.mean()
    denom = float(np.sum(centered * centered))
    if denom <= 0.0:
        return 14.0
    rho1 = float(np.sum(centered[:-1] * centered[1:]) / denom)
    rho1 = min(max(rho1, -0.95), 0.95)
    return 1.0 / (1.0 - abs(rho1))


def _estimate_mean_block_length(z: np.ndarray) -> tuple[int, bool]:
    """Return the usable mean block length and whether the data estimate
    exceeded Ginseng's supported [7, 28]-day window."""
    if len(z) < 2 or np.ptp(z) == 0:
        return 14, False
    try:
        from arch.bootstrap import optimal_block_length

        block_lengths = optimal_block_length(z)
        stationary_column = "stationary" if "stationary" in block_lengths.columns else "b_sb"
        estimate = float(block_lengths[stationary_column].iloc[0])
    except (ImportError, ValueError, FloatingPointError):
        estimate = _fallback_block_length(z)
    if not np.isfinite(estimate) or estimate <= 0:
        estimate = 14.0
    rounded = round(estimate)
    resolved = int(np.clip(rounded, MIN_MEAN_BLOCK_LENGTH, MAX_MEAN_BLOCK_LENGTH))
    return resolved, resolved != rounded


def estimate_mean_block_length(z: np.ndarray) -> int:
    """Estimate `L` (spec 17): the Politis-White optimal Stationary Bootstrap
    block length for the composite net-flow series `Z_t`, clipped to
    `[7, 28]` for hackathon stability."""
    return _estimate_mean_block_length(z)[0]


def _stationary_bootstrap_indices(
    rng: np.random.Generator,
    n_hist: int,
    n_paths: int,
    horizon_days: int,
    mean_block_length: int,
) -> np.ndarray:
    """Vectorized Stationary Bootstrap (spec 16): geometric block lengths
    with mean `mean_block_length`, wrapping circularly through the `n_hist`
    historical days so every path stays fully defined."""
    continuation_probability = 1.0 - 1.0 / mean_block_length
    index_matrix = np.empty((n_paths, horizon_days), dtype=np.int64)
    index_matrix[:, 0] = rng.integers(0, n_hist, size=n_paths)
    continue_draws = rng.random((n_paths, horizon_days))
    restart_indices = rng.integers(0, n_hist, size=(n_paths, horizon_days))
    for t in range(1, horizon_days):
        continues = continue_draws[:, t] < continuation_probability
        index_matrix[:, t] = np.where(
            continues, (index_matrix[:, t - 1] + 1) % n_hist, restart_indices[:, t]
        )
    return index_matrix


def _compute_draw_id(index_matrix: np.ndarray) -> str:
    payload = np.ascontiguousarray(index_matrix, dtype=np.int64).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _uncached_draw_bundle(
    state: FinancialState,
    horizon_days: int,
    n_paths: int,
    seed: int,
    mean_block_length: int | None = None,
) -> DrawBundle:
    """Draw a reusable set of joint bootstrap indices for `state`."""
    joint = _joint_history(state)
    n_hist = len(joint)
    if mean_block_length is None:
        z = (
            joint["variable_income"]
            - joint["essential_variable_spending"]
            - joint["discretionary_spending"]
        ).to_numpy()
        resolved_block_length, mean_block_length_was_clipped = _estimate_mean_block_length(z)
    else:
        resolved_block_length = int(
            np.clip(mean_block_length, MIN_MEAN_BLOCK_LENGTH, MAX_MEAN_BLOCK_LENGTH)
        )
        mean_block_length_was_clipped = resolved_block_length != mean_block_length

    rng = np.random.default_rng(seed)
    index_matrix = _stationary_bootstrap_indices(rng, n_hist, n_paths, horizon_days, resolved_block_length)
    return DrawBundle(
        requested_mean_block_length=mean_block_length,
        seed=seed,
        horizon_days=horizon_days,
        n_paths=n_paths,
        mean_block_length=resolved_block_length,
        mean_block_length_was_clipped=mean_block_length_was_clipped,
        history_length=n_hist,
        index_matrix=index_matrix,
        bootstrap_draw_id=_compute_draw_id(index_matrix),
    )


def draw_bundle(state, horizon_days, n_paths, seed, mean_block_length=None):
    from ginseng.execution import current_context, array_id
    context = current_context()
    if context is None:
        return _uncached_draw_bundle(state,horizon_days,n_paths,seed,mean_block_length)
    joint = _joint_history(state).to_numpy()
    key = ('legacy_draws',array_id(joint),horizon_days,n_paths,seed,mean_block_length)
    if key in context.cache:
        context.counters['cache_hits'] += 1
        return context.cache[key]
    context.check(n_paths*horizon_days*8*4)
    def create():
        context.counters['draw_generations'] += 1
        return _uncached_draw_bundle(state,horizon_days,n_paths,seed,mean_block_length)
    return context.remember(key,create,lambda b:b.index_matrix.nbytes)


def _recurring_days(item: Obligation, horizon_days: int) -> list[int]:
    """Forecast-day offsets (1-indexed) on which a scheduled item lands
    within `[1, horizon_days]`."""
    days: list[int] = []
    day = max(1, item.due_in_days)
    while day <= horizon_days:
        days.append(day)
        if item.recurrence_days is None or item.recurrence_days <= 0:
            break
        day += item.recurrence_days
    return days


def _future_obligation_daily(obligations: Sequence[Obligation], horizon_days: int) -> np.ndarray:
    """Positive outflows for one-off scenario obligations."""
    obligation_daily = np.zeros(horizon_days, dtype=float)
    for obligation in obligations:
        day = max(1, obligation.due_in_days)
        if day <= horizon_days:
            obligation_daily[day - 1] += abs(obligation.amount)
    return obligation_daily


def _deterministic_daily_flow(
    state: FinancialState, obligations: Sequence[Obligation], horizon_days: int
) -> np.ndarray:
    """The deterministic component of daily net flow (spec 12, 20): fixed
    income, fixed obligations, and any inserted future obligations. Never
    resampled."""
    flow = np.zeros(horizon_days, dtype=float)
    for item in state.fixed_income_schedule:
        for day in _recurring_days(item, horizon_days):
            flow[day - 1] += abs(item.amount)
    for item in state.fixed_obligations:
        for day in _recurring_days(item, horizon_days):
            flow[day - 1] -= abs(item.amount)
    return flow - _future_obligation_daily(obligations, horizon_days)


def known_flows(
    state: FinancialState,
    obligations: Sequence[Obligation],
    horizon_days: int,
    bundle: PathBundle | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative deterministic income and obligation series for charting.

    Direct prospective bundles carry their known schedule separately from
    stochastic/direct cash paths.  Bootstrap callers retain the historical
    state-derived behavior exactly.
    """
    if bundle is not None:
        income_daily = bundle.known_income_daily[:horizon_days]
        obligation_daily = bundle.known_obligation_daily[:horizon_days]
        additional = _future_obligation_daily(obligations, horizon_days)
        if np.any(additional):
            obligation_daily = obligation_daily + additional
        return np.cumsum(income_daily), np.cumsum(obligation_daily)

    income_daily = np.zeros(horizon_days, dtype=float)
    for item in state.fixed_income_schedule:
        for day in _recurring_days(item, horizon_days):
            income_daily[day - 1] += abs(item.amount)

    obligation_daily = np.zeros(horizon_days, dtype=float)
    for item in state.fixed_obligations:
        for day in _recurring_days(item, horizon_days):
            obligation_daily[day - 1] += abs(item.amount)
    obligation_daily += _future_obligation_daily(obligations, horizon_days)

    return np.cumsum(income_daily), np.cumsum(obligation_daily)


def _uncached_discretionary_resampled_paths(state: FinancialState, bundle: DrawBundle | PathBundle) -> np.ndarray:
    """Per-path discretionary spending from the shared forecast bundle."""
    if isinstance(bundle, PathBundle):
        return bundle.discretionary_daily[:, : bundle.horizon_days]
    disc = _joint_history(state)["discretionary_spending"].to_numpy()
    return disc[bundle.index_matrix]


def _uncached_portfolio_value_paths(
    state: FinancialState, bundle: DrawBundle | PathBundle
) -> np.ndarray | None:
    """Per-path market value on the same paths as the cash forecast."""
    if isinstance(bundle, PathBundle):
        if bundle.portfolio_values is None:
            return None
        return bundle.portfolio_values[:, : bundle.horizon_days]

    initial_value = state.marketable_backup_capital
    if state.asset_daily_returns and initial_value > 0:
        from ginseng.portfolio import aligned_asset_returns
        aligned = aligned_asset_returns(state)
        if aligned is not None:
            _, history = aligned
            values = np.array([h.market_value for h in state.taxable_portfolio])
            return np.sum(values * np.cumprod(1 + history[bundle.index_matrix], axis=1), axis=2)
    if not state.portfolio_daily_returns or initial_value <= 0.0:
        return None
    joint = _joint_history(state)
    returns_by_date = dict(state.portfolio_daily_returns)
    if len(returns_by_date) != len(state.portfolio_daily_returns) or any(ts.date() not in returns_by_date for ts in joint.index):
        return None
    market_history = np.array(
        [returns_by_date[ts.date()] for ts in joint.index]
    )
    if not np.all(np.isfinite(market_history)) or np.any(market_history <= -1):
        return None
    daily_returns = market_history[bundle.index_matrix]
    return initial_value * np.cumprod(1.0 + daily_returns, axis=1)


def cash_paths(
    state: FinancialState, bundle: DrawBundle | PathBundle, obligations: Sequence[Obligation] = (),
    *, prepared_history: np.ndarray | None = None
) -> np.ndarray:
    """`X_{j,t}` cumulative future net cash flow, excluding opening cash."""
    from ginseng.execution import current_context, prepare_scenario
    context = current_context()
    if context is not None:
        return context.cash_paths(prepare_scenario(state, bundle, obligations, prepared_history))
    if isinstance(bundle, PathBundle):
        daily_net = bundle.daily_cash_flows[:, : bundle.horizon_days]
        additional = _future_obligation_daily(obligations, bundle.horizon_days)
        if np.any(additional):
            daily_net = daily_net - additional[np.newaxis, :]
        return np.cumsum(daily_net, axis=1)

    joint = _joint_history(state).to_numpy() if prepared_history is None else prepared_history
    if joint.shape != (bundle.history_length, 3):
        raise ValueError("Prepared history does not match bundle.")
    income, essential, discretionary = joint.T

    idx = bundle.index_matrix
    stochastic_daily = income[idx] - essential[idx] - discretionary[idx]
    deterministic_daily = _deterministic_daily_flow(state, obligations, bundle.horizon_days)

    daily_net = stochastic_daily + deterministic_daily[np.newaxis, :]
    return np.cumsum(daily_net, axis=1)


def discretionary_resampled_paths(state, bundle):
    from ginseng.execution import current_context, array_id, snapshot
    from ginseng.provenance import digest
    from dataclasses import asdict
    context = current_context()
    if context is None:
        return _uncached_discretionary_resampled_paths(state, bundle)
    if isinstance(bundle, PathBundle):
        arrays = (bundle.discretionary_daily, bundle.portfolio_values)
        draw_key = tuple(array_id(a) if a is not None else None for a in arrays)
    else:
        draw_key = array_id(bundle.index_matrix)
    key = ("discretionary_resampled_paths", digest(asdict(state)), draw_key, bundle.horizon_days)
    def create():
        value = _uncached_discretionary_resampled_paths(state, bundle)
        return None if value is None else snapshot(value)
    return context.remember(key, create, lambda a: 0 if a is None else a.nbytes)


def portfolio_value_paths(state, bundle):
    from ginseng.execution import current_context, array_id, snapshot
    from ginseng.provenance import digest
    from dataclasses import asdict
    context = current_context()
    if context is None:
        return _uncached_portfolio_value_paths(state, bundle)
    if isinstance(bundle, PathBundle):
        arrays = (bundle.discretionary_daily, bundle.portfolio_values)
        draw_key = tuple(array_id(a) if a is not None else None for a in arrays)
    else:
        draw_key = array_id(bundle.index_matrix)
    key = ("portfolio_value_paths", digest(asdict(state)), draw_key, bundle.horizon_days)
    def create():
        value = _uncached_portfolio_value_paths(state, bundle)
        return None if value is None else snapshot(value)
    return context.remember(key, create, lambda a: 0 if a is None else a.nbytes)
