"""Funding-plan candidate generation and evaluation (spec sections 8, 14, 38-41).

Every candidate plan solves the *same* liability: the funding gap computed
by `ginseng.metrics.funding_gap` (spec 26). Plans differ only in which
funding class (spec 8) they draw on and how they time the resulting cash
flows. `evaluate_plan` is a pure function of `(state, bundle, obligations,
spec)` — every number a plan needs (settlement timing, credit-account
choice, lot-selection method, ...) is baked into its `PlanSpec` by
`build_candidates`. Call `comparison_draw_bundle` before evaluating a set
of candidates so every plan shares the full evaluation window and draws
(spec 20, common random numbers).
"""

from __future__ import annotations

import hashlib
from ginseng.execution import execution_scope

from dataclasses import dataclass, replace
from datetime import date, timedelta
from enum import Enum
from typing import Sequence

import numpy as np

from ginseng.metrics import severity_metrics
from ginseng.risk import probabilities, weight_hash, balance_risk
from ginseng.simulate import (
    DrawBundle,
    PathBundle,
    cash_paths,
    discretionary_resampled_paths,
)
from ginseng.state import CreditAccount, FinancialState, Holding, Obligation, TaxLot
from ginseng.withdrawals import (
    DEFAULT_ASSUMPTIONS, WithdrawalAssumptions, WithdrawalAllocation, AccountWithdrawal,
    quote_withdrawals, withdrawal_units,
    SALE_SETTLEMENT_DAYS, EXTERNAL_TRANSFER_DAYS,
)


class PlanKind(str, Enum):
    """The four spec-38 funding plans."""

    CREDIT = "credit"
    LIQUIDATE = "liquidate"
    HYBRID = "hybrid"
    PROTECTIVE = "protective"


@dataclass(frozen=True)
class FundingConfig:
    """Knobs for candidate construction and settlement timing.

    `settlement_days` + `external_transfer_days` approximate sale settlement
    and brokerage-to-bank transfer as calendar-day offsets. The three-day
    default does not implement a business-day or exchange holiday calendar.
    `hybrid_*_fraction` values must sum to 1.0 across liquidation, credit,
    and deferral; existing cash has already reduced the funding gap.
    """

    settlement_days: int = SALE_SETTLEMENT_DAYS
    external_transfer_days: int = EXTERNAL_TRANSFER_DAYS
    trailing_days: int = 3
    protective_spending_reduction: float = 0.30
    protective_spending_days: int | None = None
    hybrid_cash_fraction: float = 0.0
    hybrid_liquidation_fraction: float = 0.60
    hybrid_credit_fraction: float = 0.25
    hybrid_deferral_fraction: float = 0.15
    lot_selection: str = "fifo"
    specific_lot_ids: tuple[str, ...] = ()
    use_business_days: bool = False
    max_credit_utilization: float | None = None

    def __post_init__(self):
        fractions = (self.hybrid_cash_fraction, self.hybrid_liquidation_fraction,
                     self.hybrid_credit_fraction, self.hybrid_deferral_fraction)
        if any(not np.isfinite(f) or f < 0 for f in fractions) or not np.isclose(sum(fractions), 1.0):
            raise ValueError("Hybrid funding fractions must be nonnegative and sum to one.")
        if self.hybrid_cash_fraction != 0:
            raise ValueError("Existing cash is already included in the funding gap and cannot fund it twice.")


@dataclass(frozen=True)
class PlanSpec:
    """A fully-parameterized candidate plan, ready for ``evaluate_plan``."""

    id: str
    label: str
    kind: PlanKind
    credit_account_id: str | None = None
    credit_draw: float = 0.0
    pay_in_full: bool = True
    liquidation_target: float = 0.0
    settlement_days: int = SALE_SETTLEMENT_DAYS
    external_transfer_days: int = EXTERNAL_TRANSFER_DAYS
    lot_selection: str = "fifo"
    specific_lot_ids: tuple[str, ...] = ()
    discretionary_reduction_fraction: float = 0.0
    discretionary_reduction_days: int | None = None
    trailing_days: int = 3
    use_business_days: bool = False
    unfunded_cash_amount: float = 0.0
    withdrawal_allocations: tuple[WithdrawalAllocation, ...] | None = None
    withdrawal_assumptions: WithdrawalAssumptions = DEFAULT_ASSUMPTIONS


@dataclass(frozen=True)
class PlanResult:
    """Every spec-60 objective for one evaluated plan, plus the feasibility
    and Pareto bookkeeping `policy.py` fills in."""

    id: str
    label: str
    kind: PlanKind
    evaluation_horizon_days: int
    evaluation_draw_id: str
    cash_shortfall_probability: float
    avg_cash_deficit_when_short: float
    new_debt: float
    interest_exposure: float
    investment_sold: float
    realized_gain_loss: float
    deferred_spending: float
    credit_utilization: float
    feasible: bool
    infeasibility_reason: str | None = None
    dominated: bool = False
    dominated_by: str | None = None
    evaluation_weight_hash: str = ""
    overdraft_interest_exposure: float = 0.0
    withdrawal_tax_reserve: float = 0.0
    withdrawal_penalty_reserve: float = 0.0
    withdrawal_net_cash: float = 0.0
    withdrawal_accounts: tuple[AccountWithdrawal, ...] = ()
    buffer_breach_probability: float = 0.0
    dollar_days_below_buffer: float = 0.0
    tail_deficit: float = 0.0
    verification: dict | None = None


@dataclass(frozen=True)
class PlanEvaluation:
    result: PlanResult
    cash_matrix: np.ndarray
    spending_reduction: np.ndarray


# --------------------------------------------------------------------------
# Horizon extension (spec 14)
# --------------------------------------------------------------------------


def extend_draw_bundle(
    bundle: DrawBundle | PathBundle, target_horizon_days: int
) -> DrawBundle | PathBundle:
    """Extend one shared path bundle without replacing its random world."""
    if target_horizon_days <= bundle.horizon_days:
        return bundle
    if isinstance(bundle, PathBundle):
        if target_horizon_days > bundle.available_horizon_days:
            raise ValueError(
                "The prospective path bundle does not cover the plan's latest settlement or payment date."
            )
        return replace(bundle, horizon_days=target_horizon_days)

    if bundle.sampler != "legacy_mc":
        if bundle.material_indices is None or target_horizon_days > bundle.material_indices.shape[1]:
            raise ValueError("Requested horizon exceeds declared material horizon; create a new sampling plan.")
        indices = bundle.material_indices[:, :target_horizon_days]
        return replace(bundle, horizon_days=target_horizon_days, index_matrix=indices,
                       bootstrap_draw_id=hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest())

    extra_days = target_horizon_days - bundle.horizon_days
    n_hist = bundle.history_length
    n_paths = bundle.n_paths
    continuation_probability = 1.0 - 1.0 / bundle.mean_block_length

    sub_seed_material = (
        f"{bundle.seed}:{bundle.bootstrap_draw_id}:{bundle.horizon_days}:{target_horizon_days}"
    ).encode()
    sub_seed = int(hashlib.sha256(sub_seed_material).hexdigest()[:16], 16)
    rng = np.random.default_rng(sub_seed)
    continue_draws = rng.random((n_paths, extra_days))
    restart_indices = rng.integers(0, n_hist, size=(n_paths, extra_days))

    extension = np.empty((n_paths, extra_days), dtype=np.int64)
    previous = bundle.index_matrix[:, -1]
    for t in range(extra_days):
        continues = continue_draws[:, t] < continuation_probability
        previous = np.where(continues, (previous + 1) % n_hist, restart_indices[:, t])
        extension[:, t] = previous

    full_index_matrix = np.concatenate([bundle.index_matrix, extension], axis=1)
    draw_id = hashlib.sha256(
        np.ascontiguousarray(full_index_matrix, dtype=np.int64).tobytes()
    ).hexdigest()
    return DrawBundle(
        sampling_metadata=bundle.sampling_metadata,
        requested_mean_block_length=bundle.requested_mean_block_length,
        seed=bundle.seed,
        horizon_days=target_horizon_days,
        n_paths=n_paths,
        mean_block_length=bundle.mean_block_length,
        mean_block_length_was_clipped=bundle.mean_block_length_was_clipped,
        history_length=n_hist,
        index_matrix=full_index_matrix,
        bootstrap_draw_id=draw_id,
    )


# --------------------------------------------------------------------------
# Credit-card modeling (spec 39; CFPB grace-period rules, finance-sources #2)
# --------------------------------------------------------------------------


def _day_of_month_offset(as_of: date, day_of_month: int) -> int:
    """Days from `as_of` (0 = `as_of` itself) to the next date whose
    day-of-month is `day_of_month`. Mirrors `generate.py`'s scheduling
    convention: `CreditAccount.statement_close_day` / `payment_due_day`
    are day-of-month integers, not day offsets."""
    if not 1 <= day_of_month <= 31:
        raise ValueError("Credit calendar days must be between 1 and 31.")
    # A day such as the 31st may be absent from the current month.
    for offset in range(0, 63):
        if (as_of + timedelta(days=offset)).day == day_of_month:
            return offset
    raise ValueError(f"No calendar occurrence of day {day_of_month} follows {as_of}.")


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def _next_day_of_month(as_of: date, day_of_month: int) -> date:
    """Return the first valid configured day on or after ``as_of``."""
    candidate = date(as_of.year, as_of.month, day_of_month)
    if candidate >= as_of:
        return candidate
    year, month = _next_month(as_of.year, as_of.month)
    return date(year, month, day_of_month)

def _next_charge_payment_offset(
    as_of: date,
    account: CreditAccount,
    *,
    one_indexed: bool = False,
) -> int:
    """Forecast payment day for a charge made at the opening date."""
    close_offset = _day_of_month_offset(as_of, account.statement_close_day)
    for offset in range(close_offset + 1, close_offset + 63):
        if (as_of + timedelta(days=offset)).day == account.payment_due_day:
            return offset + 1 if one_indexed else offset
    raise ValueError("No valid credit payment date follows statement close.")


def settlement_forecast_day(
    as_of: date,
    settlement_days: int,
    external_transfer_days: int,
    *,
    use_business_days: bool = False,
) -> int:
    """Return the one-indexed day when sale proceeds become available."""
    total_days = max(0, settlement_days) + max(0, external_transfer_days)
    if not use_business_days:
        return max(1, total_days)
    cursor = as_of
    remaining = total_days
    while remaining:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            remaining -= 1
    return (cursor - as_of).days + 1


def _select_primary_credit_account(state: FinancialState, max_utilization: float | None = None) -> CreditAccount | None:
    if not state.credit_accounts:
        return None
    return max(state.credit_accounts, key=lambda account: min(account.available_credit,
        max(0, account.credit_limit * max_utilization - account.current_balance))
        if max_utilization is not None else account.available_credit)


def plan_evaluation_horizon(
    state: FinancialState,
    bundle: DrawBundle | PathBundle,
    spec: PlanSpec,
) -> int:
    """Return the full comparison horizon required by one candidate plan."""
    material_days = [bundle.horizon_days]
    if spec.credit_draw > 0 and spec.credit_account_id is not None:
        account = next(
            (item for item in state.credit_accounts if item.account_id == spec.credit_account_id),
            None,
        )
        if account is not None:
            material_days.append(
                _next_charge_payment_offset(
                    state.as_of,
                    account,
                    one_indexed=spec.use_business_days,
                )
            )
    if spec.liquidation_target > 0 or spec.withdrawal_allocations:
        material_days.append(
            settlement_forecast_day(
                state.as_of,
                spec.settlement_days,
                spec.external_transfer_days,
                use_business_days=spec.use_business_days,
            )
        )
    latest_material_day = max(material_days)
    return (
        latest_material_day
        if latest_material_day <= bundle.horizon_days
        else latest_material_day + spec.trailing_days
    )


# --------------------------------------------------------------------------
# Tax-lot liquidation (spec 40, 41; SEC T+1, IRS Tax Topic 409)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LiquidationResult:
    proceeds: float
    cost_basis_disposed: float
    shortfall: float
    allocations: tuple[WithdrawalAllocation, ...] = ()

    @property
    def realized_gain_loss(self) -> float:
        return self.proceeds - self.cost_basis_disposed


@dataclass(frozen=True)
class LiquidationBasisSegment:
    """One exact affine branch of a pro-rata taxable liquidation."""

    lower_proceeds: float
    upper_proceeds: float
    cost_basis_intercept: float
    cost_basis_slope: float


def _disposal_order(
    lots: Sequence[TaxLot], lot_selection: str, specific_lot_ids: Sequence[str]
) -> list[TaxLot]:
    """Order selectable tax lots under the user's declared disposal policy."""
    if lot_selection == "hifo":
        return sorted(
            lots,
            key=lambda lot: (lot.cost_basis_per_share, lot.purchase_date),
            reverse=True,
        )
    if lot_selection == "specific" and specific_lot_ids:
        by_id = {lot.lot_id: lot for lot in lots}
        ordered = [by_id[lot_id] for lot_id in specific_lot_ids if lot_id in by_id]
        remaining = sorted(
            (lot for lot in lots if lot.lot_id not in specific_lot_ids),
            key=lambda lot: lot.purchase_date,
        )
        return ordered + remaining
    return sorted(lots, key=lambda lot: lot.purchase_date)


def liquidation_cost_basis(
    holdings: Sequence[Holding],
    target_proceeds: float,
    lot_selection: str = "fifo",
    specific_lot_ids: Sequence[str] = (),
) -> float:
    """Return the exact basis disposed by :func:`_liquidate`'s sale rule."""
    total_market_value = sum(holding.market_value for holding in holdings)
    if total_market_value <= 0.0 or target_proceeds <= 0.0:
        return 0.0

    to_raise = min(target_proceeds, total_market_value)
    cost_basis_disposed = 0.0
    for holding in holdings:
        remaining_allocation = to_raise * (holding.market_value / total_market_value)
        for lot in _disposal_order(holding.tax_lots, lot_selection, specific_lot_ids):
            if remaining_allocation <= 1e-9:
                break
            lot_value = lot.market_value(holding.current_price)
            if lot_value <= remaining_allocation:
                cost_basis_disposed += lot.cost_basis
                remaining_allocation -= lot_value
            else:
                cost_basis_disposed += lot.cost_basis * (remaining_allocation / lot_value)
                remaining_allocation = 0.0
    return cost_basis_disposed


def liquidation_basis_segments(
    holdings: Sequence[Holding],
    lot_selection: str = "fifo",
    specific_lot_ids: Sequence[str] = (),
    *,
    max_segments: int | None = None,
) -> tuple[LiquidationBasisSegment, ...] | None:
    """Return every exact affine basis interval for a pro-rata sale.

    Each holding receives the same global sale fraction, while its own lots
    are consumed in the selected order.  The union of those per-holding lot
    boundaries therefore makes disposed basis affine on every returned
    interval.  ``None`` means the caller's explicit resource bound was
    exceeded before a complete partition could be built.
    """
    total_market_value = sum(holding.market_value for holding in holdings)
    if total_market_value <= 0.0:
        return ()

    boundaries = {0.0, total_market_value}
    for holding in holdings:
        holding_market_value = holding.market_value
        if holding_market_value <= 0.0:
            continue
        cumulative_lot_value = 0.0
        for lot in _disposal_order(holding.tax_lots, lot_selection, specific_lot_ids):
            lot_value = lot.market_value(holding.current_price)
            if lot_value <= 0.0:
                continue
            cumulative_lot_value += lot_value
            breakpoint = min(
                total_market_value,
                cumulative_lot_value * total_market_value / holding_market_value,
            )
            if 0.0 < breakpoint < total_market_value:
                boundaries.add(breakpoint)
                if max_segments is not None and len(boundaries) - 1 > max_segments:
                    return None

    ordered_boundaries = sorted(boundaries)
    segments: list[LiquidationBasisSegment] = []
    for lower_proceeds, upper_proceeds in zip(ordered_boundaries, ordered_boundaries[1:]):
        width = upper_proceeds - lower_proceeds
        if width <= 0.0:
            continue
        lower_basis = liquidation_cost_basis(
            holdings,
            lower_proceeds,
            lot_selection,
            specific_lot_ids,
        )
        upper_basis = liquidation_cost_basis(
            holdings,
            upper_proceeds,
            lot_selection,
            specific_lot_ids,
        )
        slope = (upper_basis - lower_basis) / width
        segments.append(
            LiquidationBasisSegment(
                lower_proceeds=lower_proceeds,
                upper_proceeds=upper_proceeds,
                cost_basis_intercept=lower_basis - slope * lower_proceeds,
                cost_basis_slope=slope,
            )
        )
    return tuple(segments)


def _liquidate(
    holdings: Sequence[Holding],
    target_proceeds: float,
    lot_selection: str = "fifo",
    specific_lot_ids: Sequence[str] = (),
) -> LiquidationResult:
    """Sell up to `target_proceeds` dollars of `holdings` (spec 41):
    allocated across symbols in proportion to market value, then FIFO (or
    specific-lot override) by tax lot within each symbol.
    `realized_gain_loss = proceeds - disposed cost basis` (IRS Tax Topic
    409)."""
    total_market_value = sum(h.market_value for h in holdings)
    if total_market_value <= 0.0 or target_proceeds <= 0.0:
        return LiquidationResult(0.0, 0.0, max(0.0, target_proceeds))

    to_raise = min(target_proceeds, total_market_value)
    proceeds = 0.0
    cost_basis_disposed = 0.0
    allocations = []
    for holding in holdings:
        allocation = to_raise * (holding.market_value / total_market_value)
        if lot_selection == "proportional":
            proceeds += allocation
            cost_basis_disposed += holding.cost_basis * allocation / holding.market_value if holding.market_value > 0 else 0.0
            allocations.extend(WithdrawalAllocation(f"taxable:{lot.lot_id}",
                to_raise * lot.market_value(holding.current_price) / total_market_value)
                for lot in holding.tax_lots if lot.quantity > 0)
            continue
        remaining_allocation = allocation
        for lot in _disposal_order(holding.tax_lots, lot_selection, specific_lot_ids):
            if remaining_allocation <= 1e-9:
                break
            lot_value = lot.market_value(holding.current_price)
            if lot_value <= remaining_allocation:
                allocations.append(WithdrawalAllocation(f"taxable:{lot.lot_id}", lot_value))
                proceeds += lot_value
                cost_basis_disposed += lot.cost_basis
                remaining_allocation -= lot_value
            else:
                allocations.append(WithdrawalAllocation(f"taxable:{lot.lot_id}", remaining_allocation))
                fraction = remaining_allocation / lot_value
                proceeds += remaining_allocation
                cost_basis_disposed += lot.cost_basis * fraction
                remaining_allocation = 0.0

    shortfall = max(0.0, target_proceeds - proceeds)
    return LiquidationResult(proceeds, cost_basis_disposed, shortfall, tuple(allocations))


def _taxable_gross_for_net(state, target, config, assumptions):
    """Gross up a named taxable sale so its tax reserve does not underfund it."""
    if target <= 0:
        return 0.0
    maximum = state.marketable_backup_capital
    def net(gross):
        disposal = _liquidate(state.taxable_portfolio, gross, config.lot_selection, config.specific_lot_ids)
        return quote_withdrawals(state, disposal.allocations, assumptions).net_cash
    maximum_net = net(maximum)
    if target > maximum_net:
        return maximum + target - maximum_net  # evaluator reports the unavailable amount
    if net(target) >= target - 1e-9:
        return target
    low, high = target, maximum
    for _ in range(40):
        midpoint = (low + high) / 2
        if net(midpoint) >= target:
            high = midpoint
        else:
            low = midpoint
    return high


# --------------------------------------------------------------------------
# Protective spending / hybrid deferral (spec 12, 38)
# --------------------------------------------------------------------------


def _avg_daily_discretionary_spend(state: FinancialState) -> float:
    """Historical average daily discretionary spending, used only to size
    the hybrid plan's discretionary-reduction fraction against its target
    dollar contribution to the gap (spec 38 Plan C). The actual evaluated
    savings always come from `simulate.discretionary_resampled_paths`."""
    if not state.transactions:
        return 0.0
    start = state.history_start or min(t.txn_date for t in state.transactions)
    end = state.history_end or state.as_of
    total_days = (end - start).days + 1
    if total_days <= 0:
        return 0.0
    total_discretionary = sum(
        t.amount for t in state.discretionary_spending_history if start <= t.txn_date <= end
    )
    return total_discretionary / total_days


def _hybrid_deferral_fraction(
    state: FinancialState,
    gap: float,
    config: FundingConfig,
    average_daily_discretionary_spending: float | None = None,
) -> float:
    target = gap * config.hybrid_deferral_fraction
    avg_daily = (
        average_daily_discretionary_spending
        if average_daily_discretionary_spending is not None
        else _avg_daily_discretionary_spend(state)
    )
    horizon = state.forecast_horizon
    if avg_daily <= 0.0 or horizon <= 0:
        return 0.0
    return float(np.clip(target / (avg_daily * horizon), 0.0, 1.0))


# --------------------------------------------------------------------------
# Candidate generation (spec 38)
# --------------------------------------------------------------------------


def build_candidates(
    state: FinancialState,
    obligations: Sequence[Obligation],
    gap: float,
    config: FundingConfig = FundingConfig(),
    withdrawal_assumptions: WithdrawalAssumptions = DEFAULT_ASSUMPTIONS,
    *,
    average_daily_discretionary_spending: float | None = None,
) -> list[PlanSpec]:
    """Generate actual funding alternatives for the computed gap."""
    del obligations
    gap = max(0.0, gap)
    account = _select_primary_credit_account(state, config.max_credit_utilization)
    account_id = account.account_id if account is not None else None
    timing = {
        "settlement_days": config.settlement_days,
        "external_transfer_days": config.external_transfer_days,
        "trailing_days": config.trailing_days,
        "use_business_days": config.use_business_days,
    }

    return [
        PlanSpec(
            id="credit",
            label="Credit Bridge",
            kind=PlanKind.CREDIT,
            credit_account_id=account_id,
            credit_draw=gap / (1 - account.cash_advance_fee_pct) if account else gap,
            pay_in_full=True,
            **timing,
        ),
        PlanSpec(
            id="liquidate",
            label="Taxable Liquidation",
            kind=PlanKind.LIQUIDATE,
            credit_account_id=account_id,
            liquidation_target=_taxable_gross_for_net(state, gap, config, withdrawal_assumptions),
            withdrawal_assumptions=withdrawal_assumptions,
            lot_selection=config.lot_selection,
            specific_lot_ids=config.specific_lot_ids,
            **timing,
        ),
        PlanSpec(
            id="hybrid",
            label="Hybrid",
            kind=PlanKind.HYBRID,
            credit_account_id=account_id,
            credit_draw=gap * config.hybrid_credit_fraction / (1 - account.cash_advance_fee_pct) if account else gap * config.hybrid_credit_fraction,
            pay_in_full=True,
            liquidation_target=_taxable_gross_for_net(state, gap * config.hybrid_liquidation_fraction, config, withdrawal_assumptions),
            withdrawal_assumptions=withdrawal_assumptions,
            lot_selection=config.lot_selection,
            specific_lot_ids=config.specific_lot_ids,
            discretionary_reduction_fraction=_hybrid_deferral_fraction(
                state,
                gap,
                config,
                average_daily_discretionary_spending,
            ),
            **timing,
        ),
        PlanSpec(
            id="protective",
            label="Protective Spending",
            kind=PlanKind.PROTECTIVE,
            credit_account_id=account_id,
            discretionary_reduction_fraction=config.protective_spending_reduction,
            discretionary_reduction_days=config.protective_spending_days,
            **timing,
        ),
    ]


# --------------------------------------------------------------------------
# Evaluation (spec 42, 60)
# --------------------------------------------------------------------------


def required_plan_horizon(
    state: FinancialState, bundle: DrawBundle | PathBundle, spec: PlanSpec
) -> int:
    """Compatibility name for the canonical plan comparison horizon."""
    return plan_evaluation_horizon(state, bundle, spec)


def comparison_draw_bundle(
    state: FinancialState, bundle: DrawBundle, specs: Sequence[PlanSpec]
) -> DrawBundle:
    """Extend once to the longest candidate horizon, preserving the chart's draws.

    Use the returned bundle for every candidate and the optimizer. Plans
    with no late payment still face ordinary cash flows throughout this
    common window; their risk cannot benefit from a shorter evaluation.
    """
    horizon = max(
        (required_plan_horizon(state, bundle, spec) for spec in specs),
        default=bundle.horizon_days,
    )
    return extend_draw_bundle(bundle, horizon)


def optimizer_comparison_bundle(state: FinancialState, bundle: DrawBundle | PathBundle, specs: Sequence[PlanSpec],
                                config: FundingConfig = FundingConfig()) -> DrawBundle | PathBundle:
    account = _select_primary_credit_account(state, config.max_credit_utilization)
    available_levers = PlanSpec("bounds", "Available funding", PlanKind.HYBRID,
        credit_account_id=account.account_id if account else None,
        credit_draw=account.available_credit if account else 0,
        liquidation_target=sum(u.capacity for u in withdrawal_units(state)),
        settlement_days=config.settlement_days, external_transfer_days=config.external_transfer_days,
        trailing_days=config.trailing_days, use_business_days=config.use_business_days)
    return comparison_draw_bundle(state, bundle, [*specs, available_levers])


@execution_scope
def evaluate_plan_paths(
    state: FinancialState,
    bundle: DrawBundle | PathBundle,
    obligations: Sequence[Obligation],
    spec: PlanSpec,
    weights: np.ndarray | None = None,
    *,
    operating_buffer: float | None = None,
    overdraft_apr: float = 0.0,
    evaluation_horizon_days: int | None = None,
    decision_horizon_days: int | None = None,
) -> PlanEvaluation:
    """Evaluate one `PlanSpec` on `bundle` (spec 20: the same bundle every
    candidate plan in a comparison must share) and return every spec-60
    objective. For comparisons, first prepare `comparison_draw_bundle`.
    Standalone evaluation still extends to include the plan's own liabilities.
    """
    reasons: list[str] = []
    if spec.unfunded_cash_amount > 0:
        reasons.append("Existing cash is already in the forecast; this plan declares an unfunded cash contribution")

    credit_account = None
    if spec.credit_account_id is not None:
        credit_account = next(
            (a for a in state.credit_accounts if a.account_id == spec.credit_account_id), None
        )
    if spec.credit_draw > 0 and credit_account is None:
        reasons.append("no actual credit account is available for this draw")

    new_debt = 0.0
    interest_exposure = 0.0
    decision_horizon = decision_horizon_days or bundle.horizon_days
    overdraft_interest_exposure = 0.0
    credit_utilization = 0.0
    credit_due_day = 0
    credit_payment_due_amount = 0.0

    if credit_account is not None:
        credit_utilization = (
            (credit_account.current_balance + spec.credit_draw) / credit_account.credit_limit
            if credit_account.credit_limit > 0
            else 0.0
        )
        if spec.credit_draw > 0:
            if spec.credit_draw > credit_account.available_credit:
                reasons.append(
                    f"${spec.credit_draw:,.2f} credit draw exceeds ${credit_account.available_credit:,.2f} "
                    f"available on {credit_account.account_id}"
                )
            new_debt = spec.credit_draw
            credit_due_day = _next_charge_payment_offset(
                state.as_of,
                credit_account,
                one_indexed=spec.use_business_days,
            )
            # CFPB: grace applies only when the card offers one, the
            # cardholder is not already carrying a balance, and the plan
            # intends to pay the new statement balance in full by the due
            # date (finance-sources.md section 2). Otherwise interest
            # accrues on the unpaid portion from the transaction date.
            grace_applies = (
                credit_account.grace_period_eligible
                and credit_account.current_balance <= 0.0
                and spec.pay_in_full
            )
            if grace_applies:
                credit_payment_due_amount = spec.credit_draw
            else:
                daily_rate = credit_account.purchase_apr / 365.0
                elapsed_days = credit_due_day - 1 if spec.use_business_days else credit_due_day
                interest_exposure = spec.credit_draw * daily_rate * elapsed_days
                owed = spec.credit_draw + interest_exposure
                credit_payment_due_amount = (
                    owed if spec.pay_in_full else min(credit_account.minimum_payment, owed)
                )
            interest_exposure += spec.credit_draw * credit_account.cash_advance_fee_pct

    investment_sold = 0.0
    realized_gain_loss = 0.0
    settlement_day = 1
    withdrawal_quote = quote_withdrawals(state, (), spec.withdrawal_assumptions)
    if spec.withdrawal_allocations is not None:
        try:
            withdrawal_quote = quote_withdrawals(state, spec.withdrawal_allocations, spec.withdrawal_assumptions)
        except ValueError as error:
            reasons.append(str(error))
        investment_sold = withdrawal_quote.gross
        realized_gain_loss = withdrawal_quote.realized_taxable_gain
        settlement_day = settlement_forecast_day(
            state.as_of,
            spec.settlement_days,
            spec.external_transfer_days,
            use_business_days=spec.use_business_days,
        )
    elif spec.liquidation_target > 0:
        disposal = _liquidate(
            state.taxable_portfolio, spec.liquidation_target, spec.lot_selection, spec.specific_lot_ids
        )
        investment_sold = disposal.proceeds
        realized_gain_loss = disposal.realized_gain_loss
        withdrawal_quote = quote_withdrawals(state, disposal.allocations, spec.withdrawal_assumptions)
        # T+1 settlement (SEC Rule 15c6-1) plus a configurable external
        # transfer delay (finance-sources.md section 3): proceeds are not
        # spendable cash on the trade date.
        settlement_day = settlement_forecast_day(
            state.as_of,
            spec.settlement_days,
            spec.external_transfer_days,
            use_business_days=spec.use_business_days,
        )
        if disposal.shortfall > 1e-6:
            reasons.append(
                f"only ${disposal.proceeds:,.2f} of marketable backup capital available toward a "
                f"${spec.liquidation_target:,.2f} liquidation target"
            )

    plan_horizon = required_plan_horizon(state, bundle, spec)
    evaluation_horizon = max(plan_horizon, evaluation_horizon_days or plan_horizon)
    try:
        eval_bundle = extend_draw_bundle(bundle, evaluation_horizon)
    except ValueError as error:
        reasons.append(str(error))
        eval_bundle = bundle

    cash_matrix = cash_paths(state, eval_bundle, obligations)
    n_paths, horizon_days = cash_matrix.shape
    adjustment = np.zeros(horizon_days, dtype=float)
    if spec.credit_draw > 0 and credit_account is not None:
        if credit_due_day > horizon_days:
            reasons.append("the credit repayment date falls beyond the available evaluation paths")
        else:
            adjustment[0] += spec.credit_draw * (1 - credit_account.cash_advance_fee_pct)
            adjustment[credit_due_day - 1] -= credit_payment_due_amount

    per_path_adjustment = np.broadcast_to(adjustment, (n_paths, horizon_days)).copy()

    if investment_sold > 0:
        # The modeled sale executes today at Holding.current_price.
        # Settlement and transfer delay cash availability, not execution:
        # subsequent market returns cannot reprice shares already sold.
        # The assumed tax and penalty reserve is earmarked from proceeds.
        # It is unavailable to pay bills, even before the final tax due date.
        per_path_adjustment[:, settlement_day - 1] += withdrawal_quote.net_cash

    deferred_spending = 0.0
    spending_reduction = np.zeros(n_paths)
    if spec.discretionary_reduction_fraction > 0:
        reduction_days = min(spec.discretionary_reduction_days or decision_horizon, horizon_days)
        discretionary = discretionary_resampled_paths(state, eval_bundle)
        savings = spec.discretionary_reduction_fraction * discretionary[:, :reduction_days]
        per_path_adjustment[:, :reduction_days] += savings
        spending_reduction = np.sum(savings, axis=1)
        deferred_spending = float(np.dot(probabilities(n_paths, weights), spending_reduction))

    adjusted_cash = cash_matrix + np.cumsum(per_path_adjustment, axis=1)
    resolved_operating_buffer = (
        state.operating_buffer if operating_buffer is None else operating_buffer
    )
    severity = severity_metrics(
        adjusted_cash,
        state.immediate_funding,
        resolved_operating_buffer,
        weights,
    )
    available_cash = state.immediate_funding + adjusted_cash
    risk = balance_risk(available_cash, resolved_operating_buffer, state.coverage_target, weights)
    overdraft_dollar_days = np.sum(np.maximum(0.0, -available_cash), axis=1)
    overdraft_interest_exposure = (
        float(np.dot(probabilities(n_paths, weights), overdraft_dollar_days))
        * max(0.0, overdraft_apr)
        / 365.0
    )

    result = PlanResult(
        id=spec.id,
        label=spec.label,
        kind=spec.kind,
        evaluation_horizon_days=horizon_days,
        evaluation_draw_id=eval_bundle.bootstrap_draw_id,
        cash_shortfall_probability=risk["cash_shortfall_probability"],
        avg_cash_deficit_when_short=severity["avg_cash_deficit_when_short"],
        new_debt=new_debt,
        interest_exposure=interest_exposure,
        investment_sold=investment_sold,
        realized_gain_loss=realized_gain_loss,
        overdraft_interest_exposure=overdraft_interest_exposure,
        deferred_spending=deferred_spending,
        credit_utilization=credit_utilization,
        feasible=not reasons,
        infeasibility_reason="; ".join(reasons) if reasons else None,
        evaluation_weight_hash=weight_hash(probabilities(n_paths, weights)),
        withdrawal_tax_reserve=withdrawal_quote.tax_reserve,
        withdrawal_penalty_reserve=withdrawal_quote.penalty_reserve,
        withdrawal_net_cash=withdrawal_quote.net_cash,
        withdrawal_accounts=withdrawal_quote.accounts,
        buffer_breach_probability=risk["buffer_breach_probability"],
        dollar_days_below_buffer=risk["dollar_days_below_buffer"],
        tail_deficit=risk["tail_deficit"],
    )
    return PlanEvaluation(result, adjusted_cash, spending_reduction)


def evaluate_plan(
    state: FinancialState,
    bundle: DrawBundle | PathBundle,
    obligations: Sequence[Obligation],
    spec: PlanSpec,
    weights: np.ndarray | None = None,
    *,
    operating_buffer: float | None = None,
    overdraft_apr: float = 0.0,
    evaluation_horizon_days: int | None = None,
    decision_horizon_days: int | None = None,
) -> PlanResult:
    return evaluate_plan_paths(
        state,
        bundle,
        obligations,
        spec,
        weights,
        operating_buffer=operating_buffer,
        overdraft_apr=overdraft_apr,
        evaluation_horizon_days=evaluation_horizon_days,
        decision_horizon_days=decision_horizon_days,
    ).result
