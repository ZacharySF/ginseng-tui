"""Canonical personal-finance forecast construction and evaluation.

This module turns a saved :class:`FinanceWorkspace` into the existing shared
scenario engine inputs.  It never calls ``generate_persona`` and never
manufactures transaction history for assumption mode: scheduled paths are
known-flow paths, assumptions create prospective paths directly, and history
mode bootstraps only classified user transactions.
"""

from __future__ import annotations

import calendar
from ginseng.execution import execution_scope

from dataclasses import dataclass, replace
from datetime import date, timedelta
from math import ceil, exp, log1p, sqrt
from typing import Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ginseng.finance_models import (
    EventRule,
    FinanceInputs,
    FinanceWorkspace,
    HistoricalTransaction,
    ScenarioOverrides,
)
from ginseng.funding import FundingConfig
from ginseng.policy import FundingPolicy
from ginseng.metrics import compute_scenario_metrics
from ginseng.scenario_service import ScenarioResponse, evaluate_scenario
from ginseng.uncertainty import SensitivityRow, stability_verdict
from ginseng.simulate import (
    PathBundle,
    cash_paths,
    direct_path_draw_id,
    draw_bundle,
    known_flows,
)
from ginseng.state import (
    CreditAccount,
    FinancialState,
    Holding,
    Obligation,
    TaxLot,
    Transaction,
    TransactionType,
)

SUPPORTED_HORIZONS = frozenset((14, 30, 60))
DEFAULT_FORECAST_SEED = 42
DEFAULT_FORECAST_PATHS = 2_000
MAX_FORECAST_PATHS = 4_000
MIN_COMPLETE_HISTORY_DAYS = 90
# Funding can need a statement payment after the visible 60-day chart.  Direct
# path bundles retain enough prospective days for real settlement evaluation.
MAX_PERSONAL_EVALUATION_HORIZON = 400
DAYS_PER_MONTH = 365.2425 / 12.0
ASSUMPTION_PERSISTENCE_SENSITIVITY_DAYS = (7, 14, 28)


class DataRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    label: str
    section: Literal["cash", "income", "history", "assumptions", "credit", "investments", "policy"]


class ForecastAlert(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    severity: Literal["info", "warning", "critical"]
    title: str
    detail: str


class ScenarioChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    before: str
    after: str


class BacktestWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_date: date
    end_date: date
    realized_required_cents: int
    predicted_reserve_cents: int
    covered: bool


class BacktestSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    periods: int
    observed_coverage: float
    mean_absolute_error_cents: int
    windows: list[BacktestWindow]
    warning: str | None
    calibration: dict | None = None
    information_timing: str = "retrospective_current_records"


class ForecastRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "needs-input"]
    model_mode: Literal["scheduled", "assumptions", "history"]
    input_revision: int
    horizon_days: Literal[14, 30, 60]
    as_of: date | None
    result: ScenarioResponse | None
    requirements: list[DataRequirement] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    alerts: list[ForecastAlert] = Field(default_factory=list)
    accuracy: BacktestSummary | None = None


class ForecastResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline: ForecastRun
    preview: ForecastRun | None
    changes: list[ScenarioChange] = Field(default_factory=list)


@dataclass(frozen=True)
class _Schedule:
    income: tuple[Obligation, ...]
    obligations: tuple[Obligation, ...]
    unresolved_preopening_bills: tuple[str, ...]


_CATEGORY_TYPES: dict[str, TransactionType] = {
    "income_fixed": TransactionType.INCOME_FIXED,
    "income_variable": TransactionType.INCOME_VARIABLE,
    "expense_fixed": TransactionType.EXPENSE_FIXED,
    "expense_essential_variable": TransactionType.EXPENSE_ESSENTIAL_VARIABLE,
    "expense_discretionary_variable": TransactionType.EXPENSE_DISCRETIONARY_VARIABLE,
    "expense_irregular": TransactionType.EXPENSE_IRREGULAR,
    "transfer": TransactionType.TRANSFER,
    "credit_purchase": TransactionType.CREDIT_PURCHASE,
    "credit_payment": TransactionType.CREDIT_PAYMENT,
    "investment_buy": TransactionType.INVESTMENT_BUY,
    "investment_sell": TransactionType.INVESTMENT_SELL,
}


def apply_scenario_overrides(
    workspace: FinanceWorkspace,
    overrides: ScenarioOverrides | None,
) -> FinanceWorkspace:
    """Return one validated read-only scenario replacement of a saved snapshot."""
    if overrides is None:
        return workspace

    payload = workspace.model_dump(mode="python")
    inputs = dict(payload["inputs"])
    if overrides.bills is not None:
        payload["bills"] = overrides.bills
    for field in ("income_events", "event_rules", "assumptions", "policy", "mode"):
        value = getattr(overrides, field)
        if value is not None:
            inputs[field] = value
    payload["inputs"] = inputs
    return FinanceWorkspace.model_validate(payload)


def _last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _add_months_clipped(anchor: date, months: int) -> date:
    total = anchor.year * 12 + (anchor.month - 1) + months
    year, month_index = divmod(total, 12)
    month = month_index + 1
    return date(year, month, min(anchor.day, _last_day_of_month(year, month)))


def _add_years_clipped(anchor: date, years: int) -> date:
    year = anchor.year + years
    return date(year, anchor.month, min(anchor.day, _last_day_of_month(year, anchor.month)))


def _occurrence_dates(
    anchor: date,
    recurrence: str,
    end_date: date | None,
    start: date,
    stop: date,
) -> list[date]:
    """Generate anchored occurrences with month-end and leap-year clipping."""
    limit = min(stop, end_date) if end_date is not None else stop
    if anchor > limit:
        return []
    if recurrence == "none":
        return [anchor] if start <= anchor <= limit else []

    dates: list[date] = []
    if recurrence in {"weekly", "biweekly"}:
        interval = 7 if recurrence == "weekly" else 14
        offset = max(0, ceil((start - anchor).days / interval))
        occurrence = anchor + timedelta(days=offset * interval)
        while occurrence <= limit:
            dates.append(occurrence)
            occurrence += timedelta(days=interval)
        return dates

    if recurrence == "monthly":
        months = max(0, (start.year - anchor.year) * 12 + start.month - anchor.month - 1)
        occurrence = _add_months_clipped(anchor, months)
        while occurrence < start:
            months += 1
            occurrence = _add_months_clipped(anchor, months)
        while occurrence <= limit:
            dates.append(occurrence)
            months += 1
            occurrence = _add_months_clipped(anchor, months)
        return dates

    years = max(0, start.year - anchor.year - 1)
    occurrence = _add_years_clipped(anchor, years)
    while occurrence < start:
        years += 1
        occurrence = _add_years_clipped(anchor, years)
    while occurrence <= limit:
        dates.append(occurrence)
        years += 1
        occurrence = _add_years_clipped(anchor, years)
    return dates


def _rule_map(inputs: FinanceInputs) -> dict[object, EventRule]:
    return {rule.event_id: rule for rule in inputs.event_rules}


def _schedule_for_workspace(workspace: FinanceWorkspace, horizon_days: int) -> _Schedule:
    as_of = workspace.as_of
    if as_of is None:
        return _Schedule((), (), ())
    stop = as_of + timedelta(days=horizon_days - 1)
    by_id = _rule_map(workspace.inputs)
    income: list[Obligation] = []
    obligations: list[Obligation] = []
    unresolved: list[str] = []

    def add_event(event: object, is_income: bool) -> None:
        event_id = getattr(event, "id")
        label = getattr(event, "label")
        amount = getattr(event, "amount_cents") / 100.0
        anchor = getattr(event, "due_date")
        rule = by_id.get(event_id)
        recurrence = rule.recurrence if rule is not None else "none"
        end_date = rule.end_date if rule is not None else None
        settlements = {item.due_date: item for item in rule.settlements} if rule is not None else {}

        # A pre-opening one-off bill is neither pushed to day one nor quietly
        # treated as paid. The user must reconcile it explicitly.
        if not is_income and recurrence == "none" and anchor < as_of and anchor not in settlements:
            unresolved.append(label)

        landings: list[tuple[date, date]] = []
        for due_date, settlement in settlements.items():
            if settlement.status == "settled" and settlement.settled_on is not None:
                if as_of <= settlement.settled_on <= stop:
                    landings.append((settlement.settled_on, due_date))
            # Skipped and already-settled events are intentionally absent.

        for due_date in _occurrence_dates(anchor, recurrence, end_date, as_of, stop):
            if due_date not in settlements:
                landings.append((due_date, due_date))

        target = income if is_income else obligations
        transaction_type = TransactionType.INCOME_FIXED if is_income else TransactionType.EXPENSE_FIXED
        for landing, due_date in sorted(landings):
            target.append(
                Obligation(
                    id=f"{event_id}:{due_date.isoformat()}",
                    label=label,
                    amount=amount,
                    due_in_days=(landing - as_of).days + 1,
                    transaction_type=transaction_type,
                )
            )

    for event in workspace.inputs.income_events:
        add_event(event, True)
    for event in workspace.bills:
        add_event(event, False)

    for account in workspace.inputs.credit_accounts:
        if account.current_balance_cents <= 0 or account.minimum_payment_cents <= 0:
            continue
        due_date = date(as_of.year, as_of.month, account.payment_due_day)
        if due_date < as_of:
            due_date = _add_months_clipped(due_date, 1)
        remaining_cents = account.current_balance_cents
        while due_date <= stop and remaining_cents > 0:
            payment_cents = min(remaining_cents, account.minimum_payment_cents)
            obligations.append(
                Obligation(
                    id=f"credit-minimum:{account.id}:{due_date.isoformat()}",
                    label=f"{account.name} minimum payment",
                    amount=payment_cents / 100.0,
                    due_in_days=(due_date - as_of).days + 1,
                    transaction_type=TransactionType.CREDIT_PAYMENT,
                )
            )
            remaining_cents -= payment_cents
            due_date = _add_months_clipped(due_date, 1)

    return _Schedule(tuple(income), tuple(obligations), tuple(unresolved))


def _base_requirements(workspace: FinanceWorkspace) -> list[DataRequirement]:
    requirements: list[DataRequirement] = []
    if workspace.as_of is None:
        requirements.append(
            DataRequirement(
                code="opening-date",
                label="Save the opening-of-day cash date before forecasting.",
                section="cash",
            )
        )
    if not workspace.accounts:
        requirements.append(
            DataRequirement(
                code="cash-accounts",
                label="Add at least one cash account and its opening balance.",
                section="cash",
            )
        )
    return requirements

def _credit_requirements(inputs: FinanceInputs) -> list[DataRequirement]:
    missing_minimums = [
        account.name
        for account in inputs.credit_accounts
        if account.current_balance_cents > 0 and account.minimum_payment_cents <= 0
    ]
    if not missing_minimums:
        return []
    shown = ", ".join(missing_minimums[:3])
    suffix = "" if len(missing_minimums) <= 3 else " and other accounts"
    return [
        DataRequirement(
            code="credit-minimum-payment",
            label=f"Enter a current minimum payment for {shown}{suffix} before forecasting cash settlement.",
            section="credit",
        )
    ]

def _investment_requirements(inputs: FinanceInputs) -> list[DataRequirement]:
    missing_lots = [holding.symbol for holding in inputs.holdings if not holding.tax_lots]
    if not missing_lots:
        return []
    shown = ", ".join(missing_lots[:3])
    suffix = "" if len(missing_lots) <= 3 else " and other holdings"
    return [
        DataRequirement(
            code="holding-lots",
            label=f"Enter quantity and tax lots for {shown}{suffix} before using its value in funding plans.",
            section="investments",
        )
    ]


def _history_requirements(workspace: FinanceWorkspace) -> list[DataRequirement]:
    inputs = workspace.inputs
    as_of = workspace.as_of
    requirements: list[DataRequirement] = []
    if not inputs.history_complete:
        requirements.append(
            DataRequirement(
                code="complete-history",
                label="Mark the imported classified history complete before using historical bootstrap paths.",
                section="history",
            )
        )
    if inputs.history_start is None or inputs.history_end is None:
        requirements.append(
            DataRequirement(
                code="history-range",
                label="Provide a complete history start and end date (at least 90 contiguous days recommended).",
                section="history",
            )
        )
        return requirements
    history_days = (inputs.history_end - inputs.history_start).days + 1
    if history_days < MIN_COMPLETE_HISTORY_DAYS:
        requirements.append(
            DataRequirement(
                code="history-length",
                label=f"Provide at least {MIN_COMPLETE_HISTORY_DAYS} contiguous days of classified history; {history_days} are saved.",
                section="history",
            )
        )
    if as_of is not None and inputs.history_end >= as_of:
        requirements.append(
            DataRequirement(
                code="history-opening-boundary",
                label="History must end before the opening-of-day cash date so the forecast does not learn future-day activity.",
                section="history",
            )
        )
    return requirements


def _assumption_requirements(inputs: FinanceInputs) -> list[DataRequirement]:
    assumptions = inputs.assumptions
    if any(
        (
            assumptions.monthly_variable_income_cents,
            assumptions.monthly_essential_spending_cents,
            assumptions.monthly_discretionary_spending_cents,
        )
    ):
        return []
    return [
        DataRequirement(
            code="assumption-streams",
            label="Enter at least one reviewed variable-income or spending assumption before using prospective paths.",
            section="assumptions",
        )
    ]


def _unresolved_bill_requirements(schedule: _Schedule) -> list[DataRequirement]:
    if not schedule.unresolved_preopening_bills:
        return []
    shown = ", ".join(schedule.unresolved_preopening_bills[:3])
    suffix = "" if len(schedule.unresolved_preopening_bills) <= 3 else " and other bills"
    return [
        DataRequirement(
            code="resolve-preopening-bills",
            label=(
                f"Resolve one-time bills dated before the opening balance ({shown}{suffix}); "
                "they are not moved to today."
            ),
            section="cash",
        )
    ]


def _personal_credit_accounts(inputs: FinanceInputs) -> tuple[CreditAccount, ...]:
    return tuple(
        CreditAccount(
            account_id=str(account.id),
            credit_limit=account.credit_limit_cents / 100.0,
            current_balance=account.current_balance_cents / 100.0,
            purchase_apr=float(account.cash_advance_apr),
            statement_close_day=account.statement_close_day,
            payment_due_day=account.payment_due_day,
            grace_period_eligible=False,
            minimum_payment=account.minimum_payment_cents / 100.0,
            cash_advance_limit=account.cash_advance_limit_cents / 100.0,
            cash_advance_fee_pct=float(account.cash_advance_fee_pct),
        )
        for account in inputs.credit_accounts
    )


def _personal_holdings(inputs: FinanceInputs) -> tuple[Holding, ...]:
    return tuple(
        Holding(
            symbol=holding.symbol,
            account=holding.account,
            current_price=holding.current_price_cents / 100.0,
            tax_lots=tuple(
                TaxLot(
                    lot_id=str(lot.id),
                    symbol=holding.symbol,
                    quantity=float(lot.quantity),
                    cost_basis_per_share=lot.cost_basis_per_share_cents / 100.0,
                    purchase_date=lot.purchase_date,
                )
                for lot in holding.tax_lots
            ),
        )
        for holding in inputs.holdings
    )




def _history_portfolio_returns(inputs: FinanceInputs) -> tuple[tuple[date, float], ...]:
    if inputs.history_start is None or inputs.history_end is None or not inputs.portfolio_returns:
        return ()
    by_date = {entry.date: float(entry.return_decimal) for entry in inputs.portfolio_returns}
    cursor = inputs.history_start
    values: list[tuple[date, float]] = []
    while cursor <= inputs.history_end:
        value = by_date.get(cursor)
        if value is None:
            return ()
        values.append((cursor, value))
        cursor += timedelta(days=1)
    return tuple(values)


def _state_for_workspace(
    workspace: FinanceWorkspace,
    schedule: _Schedule,
    *,
    history: bool,
) -> FinancialState:
    as_of = workspace.as_of
    if as_of is None:
        raise ValueError("An opening cash date is required.")
    inputs = workspace.inputs
    transactions: tuple[Transaction, ...]
    history_start: date | None = None
    history_end: date | None = None
    portfolio_returns: tuple[tuple[date, float], ...] = ()
    if history:
        history_start = inputs.history_start
        history_end = inputs.history_end
        if history_start is None or history_end is None:
            raise ValueError("Complete history dates are required.")
        # Keep the declared training boundary even when the opening balance
        # date is later.  ``_joint_history`` consumes these exact dates, so
        # unobserved days between history end and opening are never zero-filled.
        transactions = _ledger_with_opening_reconciliation(workspace, inputs.transactions, as_of)
        portfolio_returns = _history_portfolio_returns(inputs)
    else:
        opening_cash = sum(account.balance_cents for account in workspace.accounts) / 100.0
        transactions = (
            Transaction(
                txn_date=as_of,
                transaction_type=TransactionType.TRANSFER,
                amount=opening_cash,
                label="Opening balance reconciliation",
            ),
        )

    return FinancialState(
        as_of=as_of,
        transactions=transactions,
        fixed_income_schedule=schedule.income,
        fixed_obligations=schedule.obligations,
        planned_discretionary_events=(),
        credit_accounts=_personal_credit_accounts(inputs),
        holdings=_personal_holdings(inputs),
        operating_buffer=inputs.policy.operating_buffer_cents / 100.0,
        coverage_target=float(inputs.policy.coverage_target),
        forecast_horizon=0,  # replaced by the caller's visible horizon
        portfolio_daily_returns=portfolio_returns,
        history_start=history_start,
        history_end=history_end,
        roth_contribution_basis=inputs.roth_contribution_basis_cents / 100.0,
    )


def _ledger_with_opening_reconciliation(
    workspace: FinanceWorkspace,
    records: Sequence[HistoricalTransaction],
    as_of: date,
) -> tuple[Transaction, ...]:
    transactions: list[Transaction] = []
    for record in records:
        transaction_type = _CATEGORY_TYPES[record.category]
        amount = record.amount_cents / 100.0
        if transaction_type is not TransactionType.TRANSFER:
            amount = abs(amount)
        transactions.append(
            Transaction(
                txn_date=record.date,
                transaction_type=transaction_type,
                amount=amount,
                label=record.description,
            )
        )
    opening_cash = sum(account.balance_cents for account in workspace.accounts) / 100.0
    reconciliation = opening_cash - sum(transaction.cash_effect for transaction in transactions)
    transactions.append(
        Transaction(
            txn_date=as_of,
            transaction_type=TransactionType.TRANSFER,
            amount=reconciliation,
            label="Opening balance reconciliation",
        )
    )
    return tuple(transactions)


def _with_horizon(state: FinancialState, horizon_days: int) -> FinancialState:
    return replace(state, forecast_horizon=horizon_days)


def _known_daily(state: FinancialState, horizon_days: int) -> tuple[np.ndarray, np.ndarray]:
    income, obligations = known_flows(state, (), horizon_days)
    income_daily = np.diff(np.concatenate((np.array([0.0]), income)))
    obligation_daily = np.diff(np.concatenate((np.array([0.0]), obligations)))
    return income_daily, obligation_daily


def _scheduled_bundle(state: FinancialState, seed: int) -> PathBundle:
    known_income, known_obligations = _known_daily(state, MAX_PERSONAL_EVALUATION_HORIZON)
    daily = (known_income - known_obligations)[np.newaxis, :]
    return PathBundle(
        source="scheduled",
        seed=seed,
        horizon_days=state.forecast_horizon,
        n_paths=1,
        bootstrap_draw_id=direct_path_draw_id(
            "scheduled", seed, 1, MAX_PERSONAL_EVALUATION_HORIZON
        ),
        daily_cash_flows=daily,
        known_income_daily=known_income,
        known_obligation_daily=known_obligations,
        discretionary_daily=np.zeros_like(daily),
    )


def _persistence_phi(persistence_days: int) -> float:
    """Return the lag-one autocorrelation for the selected e-folding horizon."""
    return exp(-1.0 / max(1, persistence_days))


def _persistent_normals(rng: np.random.Generator, paths: int, days: int, persistence_days: int) -> np.ndarray:
    """Draw stationary unit-variance Gaussian AR(1) shocks.

    The same seeded generator consumes the same underlying standard-normal
    innovations for every persistence choice; only the AR(1) transform
    changes.  That preserves common random numbers in persistence stress rows.
    """
    phi = _persistence_phi(persistence_days)
    innovation_scale = sqrt(max(0.0, 1.0 - phi * phi))
    values = np.empty((paths, days), dtype=float)
    values[:, 0] = rng.standard_normal(paths)
    for day_index in range(1, days):
        values[:, day_index] = (
            phi * values[:, day_index - 1] + innovation_scale * rng.standard_normal(paths)
        )
    return values


def _mean_preserving_lognormal_multiplier(noise: np.ndarray, variability: float) -> np.ndarray:
    """Map a unit Gaussian shock to a nonnegative multiplier with mean one.

    ``variability`` is the user-supplied coefficient of variation as a decimal
    fraction.  The lognormal transform retains that marginal coefficient of
    variation without the upward mean bias caused by clipping linear shocks at
    zero.
    """
    sigma = sqrt(log1p(variability * variability))
    return np.exp(sigma * noise - 0.5 * sigma * sigma)


def _ar1_cumulative_variance(days: int, persistence_days: int) -> float:
    """Return ``Var(sum(z_t))`` for stationary unit-variance AR(1) shocks."""
    phi = _persistence_phi(persistence_days)
    remaining_days = days - 1
    denominator = 1.0 - phi
    sum_autocorrelation = phi * (1.0 - phi**remaining_days) / denominator
    weighted_autocorrelation = (
        phi
        * (1.0 - days * phi**remaining_days + remaining_days * phi**days)
        / (denominator * denominator)
    )
    return days + 2.0 * (days * sum_autocorrelation - weighted_autocorrelation)


def _uncached_assumption_components(state: FinancialState, inputs: FinanceInputs, seed: int, paths: int):
    """Draw prospective paths from reviewed assumptions, never invented history.

    Income/spending and income/market correlations apply to the latent
    standardized Gaussian AR(1) shocks.  They are not direct correlations of
    the nonlinear dollar paths after the lognormal cash transform.
    """
    assumptions = inputs.assumptions
    days = MAX_PERSONAL_EVALUATION_HORIZON
    rng = np.random.default_rng(seed)
    income_noise = _persistent_normals(rng, paths, days, assumptions.persistence_days)
    independent_spend_noise = _persistent_normals(rng, paths, days, assumptions.persistence_days)
    correlation = float(assumptions.income_spending_correlation)
    spending_noise = correlation * income_noise + sqrt(1.0 - correlation * correlation) * independent_spend_noise

    income_base = assumptions.monthly_variable_income_cents / 100.0 / DAYS_PER_MONTH
    essential_base = assumptions.monthly_essential_spending_cents / 100.0 / DAYS_PER_MONTH
    discretionary_base = assumptions.monthly_discretionary_spending_cents / 100.0 / DAYS_PER_MONTH
    income_multiplier = _mean_preserving_lognormal_multiplier(
        income_noise,
        float(assumptions.income_variability_pct),
    )
    spending_multiplier = _mean_preserving_lognormal_multiplier(
        spending_noise,
        float(assumptions.spending_variability_pct),
    )
    # A payment-arrival hurdle, independent of conditional payment-size shocks.
    # Scaling by p preserves expected monthly income while retaining zero-pay days.
    probability = float(assumptions.income_payments_per_month) / DAYS_PER_MONTH
    arrival_rng = np.random.default_rng(np.random.SeedSequence([seed, 71841]))
    arrivals = arrival_rng.random((paths, days)) < probability
    income = income_base * income_multiplier * arrivals / probability
    essential = essential_base * spending_multiplier
    discretionary = discretionary_base * spending_multiplier


    portfolio_values: np.ndarray | None = None
    if assumptions.market_assumptions_enabled and state.marketable_backup_capital > 0.0:
        market_independent = _persistent_normals(rng, paths, days, assumptions.persistence_days)
        market_correlation = float(assumptions.income_market_correlation)
        market_noise = (
            market_correlation * income_noise
            + sqrt(1.0 - market_correlation * market_correlation) * market_independent
        )
        annual_return = float(assumptions.expected_annual_return_pct)
        annual_volatility = float(assumptions.annual_return_volatility_pct)
        annual_log_growth = log1p(annual_return)
        # Calibrate the entire 365-day log return, not a daily independent
        # shock: AR(1) persistence changes the variance of this sum.
        annual_noise_variance = _ar1_cumulative_variance(365, assumptions.persistence_days)
        # The drift makes the expected 365-day simple return equal to the
        # supplied annual return after the lognormal volatility correction.
        daily_log_drift = (annual_log_growth - 0.5 * annual_volatility * annual_volatility) / 365.0
        daily_log_noise_scale = annual_volatility / sqrt(annual_noise_variance)
        cumulative_log_returns = np.cumsum(
            daily_log_drift + daily_log_noise_scale * market_noise,
            axis=1,
        )
        portfolio_values = state.marketable_backup_capital * np.exp(cumulative_log_returns)

    from ginseng.execution import snapshot
    return (snapshot(income - essential - discretionary), snapshot(discretionary),
            None if portfolio_values is None else snapshot(portfolio_values))


def _assumption_bundle(state: FinancialState, inputs: FinanceInputs, seed: int, paths: int) -> PathBundle:
    from ginseng.execution import current_context
    from ginseng.provenance import digest
    context = current_context()
    days = MAX_PERSONAL_EVALUATION_HORIZON
    key = ('assumption_components',digest(inputs.assumptions.model_dump(mode='json')),seed,paths,days,state.marketable_backup_capital)
    def create():
        if context:
            context.counters['assumption_preparations'] += 1
        return _uncached_assumption_components(state,inputs,seed,paths)
    if context:
        if key not in context.cache:
            context.check(paths*days*8*16)
        stochastic,discretionary,portfolio_values = context.remember(key,create,lambda arrays:sum(a.nbytes for a in arrays if a is not None))
    else:
        stochastic,discretionary,portfolio_values = create()
    known_income, known_obligations = _known_daily(state,days)
    daily_cash = stochastic + known_income[np.newaxis,:] - known_obligations[np.newaxis,:]
    return PathBundle(
        source="assumptions",
        seed=seed,
        horizon_days=state.forecast_horizon,
        n_paths=paths,
        bootstrap_draw_id=direct_path_draw_id("assumptions", seed, paths, days),
        daily_cash_flows=daily_cash,
        known_income_daily=known_income,
        known_obligation_daily=known_obligations,
        discretionary_daily=discretionary,
        portfolio_values=portfolio_values,
    )


def _assumption_persistence_sensitivity(
    state: FinancialState,
    inputs: FinanceInputs,
    baseline_bundle: PathBundle,
    *,
    seed: int,
    paths: int,
) -> tuple[list[dict[str, object]], str]:
    """Evaluate direct-model persistence stress rows without funding re-solves."""
    assumptions = inputs.assumptions
    persistence_days = list(ASSUMPTION_PERSISTENCE_SENSITIVITY_DAYS)
    if assumptions.persistence_days not in persistence_days:
        persistence_days.append(assumptions.persistence_days)

    rows: list[dict[str, object]] = []
    verdict_rows: list[SensitivityRow] = []
    for days in persistence_days:
        if days == assumptions.persistence_days:
            stressed_bundle = baseline_bundle
        else:
            stressed_assumptions = assumptions.model_copy(update={"persistence_days": days})
            stressed_inputs = inputs.model_copy(update={"assumptions": stressed_assumptions})
            # Resetting to the same seed deliberately reuses every latent
            # standard-normal innovation; changing ``days`` changes only AR(1)
            # persistence, never the sampled random stream or a funding decision.
            stressed_bundle = _assumption_bundle(state, stressed_inputs, seed, paths)
        metrics = compute_scenario_metrics(
            state,
            stressed_bundle,
            (),
            state.coverage_target,
            state.operating_buffer,
        )
        is_current = (
            days == assumptions.persistence_days
            and days not in ASSUMPTION_PERSISTENCE_SENSITIVITY_DAYS
        )
        row = {
            "block_label": f"Current {days}d" if is_current else f"{days}d",
            # Preserve the established response field while carrying the
            # direct-model name as well; these are not bootstrap blocks.
            "mean_block_length": days,
            "persistence_days": days,
            "required_liquidity_reserve": metrics.required_liquidity_reserve,
            "funding_gap": metrics.funding_gap,
            "cash_shortfall_probability": metrics.severity["cash_shortfall_probability"],
            "is_estimated": False,
            "was_clipped": False,
        }
        rows.append(row)
        verdict_rows.append(
            SensitivityRow(
                block_label=str(row["block_label"]),
                mean_block_length=days,
                required_liquidity_reserve=metrics.required_liquidity_reserve,
                is_estimated=False,
                was_clipped=False,
            )
        )
    return rows, stability_verdict(verdict_rows)


def _funding_config(inputs: FinanceInputs) -> FundingConfig:
    policy = inputs.policy
    return FundingConfig(
        settlement_days=policy.settlement_days,
        external_transfer_days=policy.external_transfer_days,
        lot_selection=policy.lot_selection,
        use_business_days=True,
    )


def _funding_policy(inputs: FinanceInputs) -> FundingPolicy:
    policy = inputs.policy
    return FundingPolicy(
        max_cash_shortfall_probability=max(0.0, 1.0 - float(policy.coverage_target)),
        max_credit_utilization=float(policy.max_credit_utilization),
        priorities=tuple(policy.priorities),
        capital_gains_rate=float(policy.capital_gains_rate),
        overdraft_apr=float(policy.overdraft_apr),
    )


def _alerts_for_run(
    workspace: FinanceWorkspace,
    run_result: ScenarioResponse,
    *,
    mode: str,
) -> list[ForecastAlert]:
    preferences = workspace.inputs.alerts
    if not preferences.enabled:
        return []
    alerts: list[ForecastAlert] = []
    shortfall = run_result.severity.cash_shortfall_probability
    if shortfall > 0.0:
        if mode == "scheduled":
            alerts.append(
                ForecastAlert(
                    id="deterministic-shortfall",
                    severity="critical",
                    title="Known cash shortfall",
                    detail="Saved scheduled cash flows cross below zero in this deterministic forecast.",
                )
            )
        elif shortfall >= preferences.shortfall_probability:
            alerts.append(
                ForecastAlert(
                    id="modeled-shortfall",
                    severity="warning",
                    title="Modeled cash-shortfall threshold exceeded",
                    detail=(
                        f"The current forecast shortfall rate ({shortfall:.1%}) exceeds your in-app alert "
                        f"threshold ({preferences.shortfall_probability:.1%})."
                    ),
                )
            )
    if run_result.funding_gap > 0.0:
        alerts.append(
            ForecastAlert(
                id="reserve-gap",
                severity="warning",
                title="Reserve coverage gap",
                detail="Current settled cash is below the reserve indicated by this forecast's policy settings.",
            )
        )
    if mode == "history" and workspace.inputs.history_end is not None and workspace.as_of is not None:
        stale_days = (workspace.as_of - workspace.inputs.history_end).days
        if stale_days > preferences.stale_after_days:
            alerts.append(
                ForecastAlert(
                    id="stale-history",
                    severity="info",
                    title="Recorded history is older than your freshness preference",
                    detail=(
                        f"History ends {stale_days} days before the opening balance date; "
                        "this is an in-app forecast freshness notice, not background monitoring."
                    ),
                )
            )
    return alerts


def _warnings_for_mode(
    workspace: FinanceWorkspace,
    state: FinancialState,
    *,
    mode: str,
    optimizer_reason: str | None,
) -> list[str]:
    warnings: list[str] = []
    if mode == "scheduled":
        warnings.append(
            "This is a deterministic schedule of saved income and bill dates; it has no percentile uncertainty."
        )
        if state.marketable_backup_capital > 0.0:
            warnings.append(
                "Wrong-way risk is unavailable for a deterministic schedule without reviewed market paths."
            )
    elif mode == "assumptions":
        warnings.append("Prospective paths use your reviewed assumptions directly; they are not historical observations.")
        if state.marketable_backup_capital > 0.0 and not workspace.inputs.assumptions.market_assumptions_enabled:
            warnings.append("Wrong-way risk is unavailable until reviewed market assumptions are enabled for taxable holdings.")
    else:
        inputs = workspace.inputs
        assert inputs.history_start is not None and inputs.history_end is not None
        warnings.append(
            f"Bootstrap paths use only classified history from {inputs.history_start.isoformat()} through "
            f"{inputs.history_end.isoformat()}; days after that end date are not treated as zero activity."
        )
        if state.marketable_backup_capital > 0.0 and not state.portfolio_daily_returns:
            warnings.append(
                "Wrong-way risk is unavailable because complete daily market returns were not supplied for the recorded history window."
            )
    if optimizer_reason is not None:
        warnings.append(optimizer_reason)
    return warnings


@execution_scope
def evaluate_personal_forecast(
    workspace: FinanceWorkspace,
    horizon_days: int,
    seed: int = DEFAULT_FORECAST_SEED,
    paths: int = DEFAULT_FORECAST_PATHS,
) -> ForecastRun:
    """Purely evaluate one saved or in-memory canonical finance workspace."""
    if horizon_days not in SUPPORTED_HORIZONS:
        raise ValueError("Forecast horizon must be 14, 30, or 60 days.")
    if paths < 1 or paths > MAX_FORECAST_PATHS:
        raise ValueError(f"Forecast paths must be between 1 and {MAX_FORECAST_PATHS}.")

    mode = workspace.inputs.mode
    requirements = _base_requirements(workspace)
    requirements.extend(_credit_requirements(workspace.inputs))
    requirements.extend(_investment_requirements(workspace.inputs))
    schedule = _schedule_for_workspace(workspace, MAX_PERSONAL_EVALUATION_HORIZON)
    requirements.extend(_unresolved_bill_requirements(schedule))
    if mode == "history":
        requirements.extend(_history_requirements(workspace))
    elif mode == "assumptions":
        requirements.extend(_assumption_requirements(workspace.inputs))

    if requirements:
        return ForecastRun(
            status="needs-input",
            model_mode=mode,
            input_revision=workspace.revision,
            horizon_days=horizon_days,
            as_of=workspace.as_of,
            result=None,
            requirements=requirements,
        )

    state = _with_horizon(
        _state_for_workspace(workspace, schedule, history=mode == "history"),
        horizon_days,
    )
    if mode == "history":
        bundle = draw_bundle(state, horizon_days=horizon_days, n_paths=paths, seed=seed)
        outer_paths = min(20, max(8, 80_000 // paths))
        evaluation = evaluate_scenario(
            state,
            bundle,
            (),
            coverage_target=state.coverage_target,
            operating_buffer=state.operating_buffer,
            model_source=mode, input_config=workspace.inputs.model_dump(mode="json", exclude={"transactions", "scenarios"}),
            funding_config=_funding_config(workspace.inputs),
            funding_policy=_funding_policy(workspace.inputs),
            overdraft_apr=float(workspace.inputs.policy.overdraft_apr),
            buffer_tolerance_dollar_days=workspace.inputs.policy.buffer_tolerance_dollar_days,
            capital_gains_rate=float(workspace.inputs.policy.capital_gains_rate),
            include_reserve_uncertainty=True,
            include_persistence_sensitivity=True,
            uncertainty_outer_paths=outer_paths,
        )
    elif mode == "assumptions":
        bundle = _assumption_bundle(state, workspace.inputs, seed, paths)
        evaluation = evaluate_scenario(
            state,
            bundle,
            (),
            coverage_target=state.coverage_target,
            operating_buffer=state.operating_buffer,
            model_source=mode, input_config=workspace.inputs.model_dump(mode="json", exclude={"transactions", "scenarios"}),
            funding_config=_funding_config(workspace.inputs),
            funding_policy=_funding_policy(workspace.inputs),
            overdraft_apr=float(workspace.inputs.policy.overdraft_apr),
            buffer_tolerance_dollar_days=workspace.inputs.policy.buffer_tolerance_dollar_days,
            capital_gains_rate=float(workspace.inputs.policy.capital_gains_rate),
            include_reserve_uncertainty=False,
            include_persistence_sensitivity=False,
        )
    else:
        bundle = _scheduled_bundle(state, seed)
        evaluation = evaluate_scenario(
            state,
            bundle,
            (),
            coverage_target=state.coverage_target,
            operating_buffer=state.operating_buffer,
            model_source=mode, input_config=workspace.inputs.model_dump(mode="json", exclude={"transactions", "scenarios"}),
            funding_config=_funding_config(workspace.inputs),
            funding_policy=_funding_policy(workspace.inputs),
            overdraft_apr=float(workspace.inputs.policy.overdraft_apr),
            buffer_tolerance_dollar_days=workspace.inputs.policy.buffer_tolerance_dollar_days,
            capital_gains_rate=float(workspace.inputs.policy.capital_gains_rate),
            include_reserve_uncertainty=False,
            include_persistence_sensitivity=False,
            deterministic=True,
        )
    result = evaluation.response
    if mode == "assumptions":
        sensitivity, sensitivity_verdict = _assumption_persistence_sensitivity(
            state,
            workspace.inputs,
            bundle,
            seed=seed,
            paths=paths,
        )
        result = result.model_copy(
            update={
                "sensitivity": sensitivity,
                "sensitivity_verdict": sensitivity_verdict,
            }
        )


    return ForecastRun(
        status="ready",
        model_mode=mode,
        input_revision=workspace.revision,
        horizon_days=horizon_days,
        as_of=workspace.as_of,
        result=result,
        warnings=_warnings_for_mode(
            workspace,
            state,
            mode=mode,
            optimizer_reason=evaluation.optimizer_reason,
        ),
        alerts=_alerts_for_run(workspace, result, mode=mode),
    )


def describe_scenario_changes(
    workspace: FinanceWorkspace,
    overrides: ScenarioOverrides,
) -> list[ScenarioChange]:
    """Describe only explicit user-provided replacement inputs for a preview."""
    changes: list[ScenarioChange] = []
    inputs = workspace.inputs
    if overrides.mode is not None and overrides.mode != inputs.mode:
        changes.append(ScenarioChange(label="Forecast source", before=inputs.mode, after=overrides.mode))
    if overrides.bills is not None:
        changes.append(
            ScenarioChange(
                label="Scheduled bills",
                before=f"{len(workspace.bills)} saved bill(s)",
                after=f"{len(overrides.bills)} replacement bill(s)",
            )
        )
    if overrides.income_events is not None:
        changes.append(
            ScenarioChange(
                label="Scheduled income",
                before=f"{len(inputs.income_events)} saved income event(s)",
                after=f"{len(overrides.income_events)} replacement income event(s)",
            )
        )
    if overrides.event_rules is not None:
        changes.append(
            ScenarioChange(
                label="Recurring event rules",
                before=f"{len(inputs.event_rules)} saved rule(s)",
                after=f"{len(overrides.event_rules)} replacement rule(s)",
            )
        )
    if overrides.assumptions is not None:
        changes.extend(
            [
                ScenarioChange(
                    label="Monthly variable income assumption",
                    before=f"${inputs.assumptions.monthly_variable_income_cents / 100:,.2f}",
                    after=f"${overrides.assumptions.monthly_variable_income_cents / 100:,.2f}",
                ),
                ScenarioChange(
                    label="Monthly essential spending assumption",
                    before=f"${inputs.assumptions.monthly_essential_spending_cents / 100:,.2f}",
                    after=f"${overrides.assumptions.monthly_essential_spending_cents / 100:,.2f}",
                ),
                ScenarioChange(
                    label="Monthly discretionary spending assumption",
                    before=f"${inputs.assumptions.monthly_discretionary_spending_cents / 100:,.2f}",
                    after=f"${overrides.assumptions.monthly_discretionary_spending_cents / 100:,.2f}",
                ),
            ]
        )
    if overrides.policy is not None:
        changes.extend(
            [
                ScenarioChange(
                    label="Operating buffer",
                    before=f"${inputs.policy.operating_buffer_cents / 100:,.2f}",
                    after=f"${overrides.policy.operating_buffer_cents / 100:,.2f}",
                ),
                ScenarioChange(
                    label="Reserve coverage target",
                    before=f"{inputs.policy.coverage_target:.0%}",
                    after=f"{overrides.policy.coverage_target:.0%}",
                ),
                ScenarioChange(
                    label="Maximum credit utilization",
                    before=f"{inputs.policy.max_credit_utilization:.0%}",
                    after=f"{overrides.policy.max_credit_utilization:.0%}",
                ),
            ]
        )
    return [change for change in changes if change.before != change.after]


def _backtest_requirements(workspace: FinanceWorkspace, horizon_days: int) -> tuple[date, date]:
    inputs = workspace.inputs
    if not inputs.history_complete or inputs.history_start is None or inputs.history_end is None:
        raise ValueError("Complete classified history with a saved start and end date is required for backtesting.")
    if (inputs.history_end - inputs.history_start).days + 1 < MIN_COMPLETE_HISTORY_DAYS + horizon_days:
        raise ValueError(
            f"Backtesting a {horizon_days}-day horizon needs at least {MIN_COMPLETE_HISTORY_DAYS + horizon_days} contiguous history days."
        )
    return inputs.history_start, inputs.history_end


def _variable_transactions_until(
    records: Sequence[HistoricalTransaction],
    end: date,
) -> tuple[Transaction, ...]:
    variable = {
        "income_variable": TransactionType.INCOME_VARIABLE,
        "expense_essential_variable": TransactionType.EXPENSE_ESSENTIAL_VARIABLE,
        "expense_discretionary_variable": TransactionType.EXPENSE_DISCRETIONARY_VARIABLE,
    }
    return tuple(
        Transaction(
            txn_date=record.date,
            transaction_type=variable[record.category],
            amount=abs(record.amount_cents) / 100.0,
            label=record.description,
        )
        for record in records
        if record.date <= end and record.category in variable
    )


def _actual_variable_change_cents(
    records: Sequence[HistoricalTransaction], start: date, end: date
) -> int:
    total = 0
    for record in records:
        if not start <= record.date <= end:
            continue
        if record.category == "income_variable":
            total += record.amount_cents
        elif record.category in {
            "expense_essential_variable",
            "expense_discretionary_variable",
        }:
            total += record.amount_cents
    return total


def backtest_personal_history(
    workspace: FinanceWorkspace, horizon_days: int, *, seed: int = DEFAULT_FORECAST_SEED, paths: int = 500,
) -> BacktestSummary:
    """Validate the reserve target on non-overlapping held-out personal history."""
    from ginseng.calibration import walk_forward
    if horizon_days not in SUPPORTED_HORIZONS:
        raise ValueError("Backtest horizon must be 14, 30, or 60 days.")
    _backtest_requirements(workspace, horizon_days)
    state = _with_horizon(_state_for_workspace(workspace,
        _schedule_for_workspace(workspace, MAX_PERSONAL_EVALUATION_HORIZON), history=True), horizon_days)
    report = walk_forward(state, horizon_days, min(paths, 500), seed, state.coverage_target,
                          state.operating_buffer, training_days=MIN_COMPLETE_HISTORY_DAYS,
                          source="history", max_windows=48)
    rows = [row for row in report["windows"] if row["in_primary_sample"]]
    periods = len(rows)
    windows = [BacktestWindow(start_date=row["start"], end_date=row["end"],
        realized_required_cents=round(row["realized_required"] * 100),
        predicted_reserve_cents=round(row["predicted_reserve"] * 100), covered=row["covered"]) for row in rows]
    warning = (f"Only {periods} non-overlapping windows; non-overlap does not prove independence. "
               + ("Formal tail tests are underpowered at this coverage target." if report["expected_tail_failures"] < 5
                  else "Use the reported uncertainty interval and test limitations."))
    return BacktestSummary(periods=periods,
        observed_coverage=report["primary"]["observed_coverage"] or 0,
        mean_absolute_error_cents=round(sum(abs(w.realized_required_cents-w.predicted_reserve_cents) for w in windows) / periods) if periods else 0,
        windows=windows, warning=warning, calibration=report)


def historical_numerical_case(workspace: FinanceWorkspace, horizon_days: int):
    """Use the same canonical schedule and history requirements as the forecast."""
    from ginseng.inputs import InputCase
    if workspace.inputs.mode != 'history':
        raise ValueError('Risk exploration requires complete classified history; scheduled and assumption forecasts are not supported.')
    requirements = _base_requirements(workspace) + _history_requirements(workspace)
    schedule = _schedule_for_workspace(workspace, MAX_PERSONAL_EVALUATION_HORIZON)
    requirements.extend(_unresolved_bill_requirements(schedule))
    if requirements:
        raise ValueError(' '.join(item.label for item in requirements))
    state = _with_horizon(_state_for_workspace(workspace,schedule,history=True),horizon_days)
    return InputCase('personal-history',state,())
