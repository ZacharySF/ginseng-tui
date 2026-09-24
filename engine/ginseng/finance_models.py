"""Canonical contracts for the supplemental personal-finance workspace data.

Cash accounts and bills remain in :mod:`ginseng.workspace`.  This module owns
only the additional, owner-scoped information that is saved alongside that
cash snapshot through the finance RPC.
"""

from __future__ import annotations

import calendar
from decimal import Decimal
from typing import Annotated, Literal, Sequence
from uuid import UUID

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    model_serializer,
    model_validator,
)

from ginseng.workspace import (
    MAX_ABS_BALANCE_CENTS,
    MAX_BILL_CENTS,
    MAX_EXPECTED_REVISION,
    MAX_WORKSPACE_REVISION,
    CashAccount,
    CashBill,
    CashWorkspace,
    StrictInt,
    WorkspaceDate,
)

MAX_FINANCE_TRANSACTIONS = 20_000
MAX_FINANCE_INCOME_EVENTS = 200
MAX_EVENT_RULES = 400
MAX_SETTLEMENTS_PER_EVENT = 10_000
MAX_CREDIT_ACCOUNTS = 100
MAX_HOLDINGS = 500
MAX_TAX_LOTS_PER_HOLDING = 1_000
MAX_PORTFOLIO_RETURNS = 20_000
MAX_FINANCE_INPUT_BYTES = 4_000_000
MAX_NAMED_SCENARIOS = 20
MAX_DESCRIPTION_LENGTH = 500
MAX_SOURCE_KEY_LENGTH = 200
MAX_SYMBOL_LENGTH = 32
MAX_FINANCE_NAME_LENGTH = 100
MAX_TRANSACTION_CENTS = MAX_ABS_BALANCE_CENTS
MAX_QUANTITY = 1_000_000_000_000
MAX_AGGREGATE_HOLDING_CENTS = MAX_ABS_BALANCE_CENTS
MAX_BUFFER_TOLERANCE_DOLLAR_DAYS = 1_000_000_000_000_000.0
_SCENARIO_NAME_CASE = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")

FinanceMode = Literal["scheduled", "assumptions", "history"]
EventRecurrence = Literal["none", "weekly", "biweekly", "monthly", "yearly"]
SettlementStatus = Literal["settled", "skipped"]
TransactionCategory = Literal[
    "income_fixed",
    "income_variable",
    "expense_fixed",
    "expense_essential_variable",
    "expense_discretionary_variable",
    "expense_irregular",
    "transfer",
    "credit_purchase",
    "credit_payment",
    "investment_buy",
    "investment_sell",
]
HoldingAccount = Literal["taxable", "traditional", "roth", "retirement"]
PlanningPriority = Literal[
    "avoid_interest_bearing_debt",
    "minimize_taxable_sales",
    "minimize_deferred_spending",
]
LotSelection = Literal["fifo", "hifo"]


def _require_json_number(value: object) -> object:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Financial numbers must be JSON numbers.")
    return value


FiniteNumber = Annotated[
    float,
    BeforeValidator(_require_json_number),
    Field(allow_inf_nan=False),
]


class _FinanceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _require_trimmed(value: str, field_name: str) -> str:
    if value != value.strip():
        raise ValueError(f"{field_name} cannot start or end with whitespace.")
    return value


def _require_unique_ids(items: Sequence[object], label: str, id_field: str = "id") -> None:
    ids = [getattr(item, id_field) for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} must have unique IDs.")


class EventSettlement(_FinanceModel):
    due_date: WorkspaceDate
    status: SettlementStatus
    settled_on: WorkspaceDate | None

    @model_validator(mode="after")
    def settlement_state_is_complete(self) -> "EventSettlement":
        if self.status == "settled" and self.settled_on is None:
            raise ValueError("Settled occurrences require their settlement date.")
        if self.status == "skipped" and self.settled_on is not None:
            raise ValueError("Skipped occurrences cannot have a settlement date.")
        return self


class EventRule(_FinanceModel):
    event_id: UUID
    recurrence: EventRecurrence
    end_date: WorkspaceDate | None
    settlements: list[EventSettlement] = Field(default_factory=list, max_length=MAX_SETTLEMENTS_PER_EVENT)

    @model_validator(mode="after")
    def settlement_dates_are_unique(self) -> "EventRule":
        due_dates = [settlement.due_date for settlement in self.settlements]
        if len(set(due_dates)) != len(due_dates):
            raise ValueError("Each event occurrence can be settled only once.")
        if self.recurrence == "none" and self.end_date is not None:
            raise ValueError("One-time events cannot have a recurrence end date.")
        return self


class HistoricalTransaction(_FinanceModel):
    id: UUID
    date: WorkspaceDate
    description: Annotated[str, Field(min_length=1, max_length=MAX_DESCRIPTION_LENGTH)]
    amount_cents: Annotated[StrictInt, Field(ge=-MAX_TRANSACTION_CENTS, le=MAX_TRANSACTION_CENTS)]
    category: TransactionCategory
    source_key: Annotated[str, Field(min_length=1, max_length=MAX_SOURCE_KEY_LENGTH)] | None

    @model_validator(mode="after")
    def amount_matches_category(self) -> "HistoricalTransaction":
        positive_categories = {"income_fixed", "income_variable", "investment_sell"}
        negative_categories = {
            "expense_fixed",
            "expense_essential_variable",
            "expense_discretionary_variable",
            "expense_irregular",
            "credit_purchase",
            "credit_payment",
            "investment_buy",
        }
        if self.amount_cents == 0:
            raise ValueError("Historical transactions cannot have a zero amount.")
        if self.category in positive_categories and self.amount_cents <= 0:
            raise ValueError("Income and investment sales must use positive cents.")
        if self.category in negative_categories and self.amount_cents >= 0:
            raise ValueError("Expenses, payments, and investment purchases must use negative cents.")
        return self

    @model_validator(mode="after")
    def text_is_trimmed(self) -> "HistoricalTransaction":
        _require_trimmed(self.description, "Transaction descriptions")
        if self.source_key is not None:
            _require_trimmed(self.source_key, "Transaction source keys")
        return self


class ModelAssumptions(_FinanceModel):
    income_payments_per_month: Annotated[FiniteNumber, Field(gt=0, le=30)] = 2.0
    monthly_variable_income_cents: Annotated[StrictInt, Field(ge=0, le=MAX_BILL_CENTS)] = 0
    monthly_essential_spending_cents: Annotated[StrictInt, Field(ge=0, le=MAX_BILL_CENTS)] = 0
    monthly_discretionary_spending_cents: Annotated[StrictInt, Field(ge=0, le=MAX_BILL_CENTS)] = 0
    income_variability_pct: Annotated[FiniteNumber, Field(ge=0, le=2)] = 0.0
    spending_variability_pct: Annotated[FiniteNumber, Field(ge=0, le=2)] = 0.0
    persistence_days: Annotated[StrictInt, Field(ge=1, le=30)] = 7
    income_spending_correlation: Annotated[FiniteNumber, Field(ge=-0.95, le=0.95)] = 0.0
    market_assumptions_enabled: StrictBool = False
    expected_annual_return_pct: Annotated[FiniteNumber, Field(ge=-0.99, le=2)] = 0.0
    annual_return_volatility_pct: Annotated[FiniteNumber, Field(ge=0, le=2)] = 0.0
    income_market_correlation: Annotated[FiniteNumber, Field(ge=-0.95, le=0.95)] = 0.0


class PersonalCreditAccount(_FinanceModel):
    id: UUID
    name: Annotated[str, Field(min_length=1, max_length=MAX_FINANCE_NAME_LENGTH)]
    credit_limit_cents: Annotated[StrictInt, Field(ge=0, le=MAX_ABS_BALANCE_CENTS)]
    current_balance_cents: Annotated[StrictInt, Field(ge=0, le=MAX_ABS_BALANCE_CENTS)]
    purchase_apr: Annotated[FiniteNumber, Field(ge=0, le=1)]
    statement_close_day: Annotated[StrictInt, Field(ge=1, le=28)]
    payment_due_day: Annotated[StrictInt, Field(ge=1, le=28)]
    grace_period_eligible: StrictBool
    minimum_payment_cents: Annotated[StrictInt, Field(ge=0, le=MAX_ABS_BALANCE_CENTS)]
    cash_advance_limit_cents: Annotated[StrictInt, Field(ge=0, le=MAX_ABS_BALANCE_CENTS)] = 0
    cash_advance_apr: Annotated[FiniteNumber, Field(ge=0, le=1)] = 0.0
    cash_advance_fee_pct: Annotated[FiniteNumber, Field(ge=0, le=0.5)] = 0.0

    @model_validator(mode="after")
    def name_is_trimmed(self) -> "PersonalCreditAccount":
        _require_trimmed(self.name, "Credit account names")
        return self


class PersonalTaxLot(_FinanceModel):
    id: UUID
    quantity: Annotated[FiniteNumber, Field(gt=0, le=MAX_QUANTITY)]
    cost_basis_per_share_cents: Annotated[StrictInt, Field(ge=0, le=MAX_BILL_CENTS)]
    purchase_date: WorkspaceDate


class PersonalHolding(_FinanceModel):
    id: UUID
    symbol: Annotated[str, Field(min_length=1, max_length=MAX_SYMBOL_LENGTH)]
    account: HoldingAccount
    current_price_cents: Annotated[StrictInt, Field(gt=0, le=MAX_BILL_CENTS)]
    tax_lots: list[PersonalTaxLot] = Field(default_factory=list, max_length=MAX_TAX_LOTS_PER_HOLDING)

    @model_validator(mode="after")
    def holding_fields_are_consistent(self) -> "PersonalHolding":
        _require_trimmed(self.symbol, "Holding symbols")
        _require_unique_ids(self.tax_lots, "Tax lots")
        return self


class PortfolioReturn(_FinanceModel):
    date: WorkspaceDate
    return_decimal: Annotated[FiniteNumber, Field(ge=-1, le=100)]


class PlanningPolicy(_FinanceModel):
    operating_buffer_cents: Annotated[StrictInt, Field(ge=0, le=MAX_ABS_BALANCE_CENTS)] = 0
    coverage_target: Annotated[FiniteNumber, Field(gt=0, le=1)] = 0.95
    max_credit_utilization: Annotated[FiniteNumber, Field(ge=0, le=1)] = 0.30
    priorities: list[PlanningPriority] = Field(
        default_factory=lambda: [
            "avoid_interest_bearing_debt",
            "minimize_taxable_sales",
            "minimize_deferred_spending",
        ],
        min_length=1,
        max_length=3,
    )
    settlement_days: Annotated[StrictInt, Field(ge=0, le=30)] = 1
    external_transfer_days: Annotated[StrictInt, Field(ge=0, le=30)] = 2
    capital_gains_rate: Annotated[FiniteNumber, Field(ge=0, le=1)] = 0.15
    overdraft_apr: Annotated[FiniteNumber, Field(ge=0, le=1)] = 0.18
    buffer_tolerance_dollar_days: Annotated[
        FiniteNumber,
        Field(ge=0, le=MAX_BUFFER_TOLERANCE_DOLLAR_DAYS),
    ] | None = None
    lot_selection: LotSelection = "fifo"

    @model_validator(mode="after")
    def priorities_are_unique(self) -> "PlanningPolicy":
        if len(set(self.priorities)) != len(self.priorities):
            raise ValueError("Planning priorities cannot repeat.")
        return self


class ScenarioOverrides(_FinanceModel):
    """Optional complete replacements applied to the currently saved inputs."""

    bills: list[CashBill] | None = Field(default=None, max_length=200)
    income_events: list[CashBill] | None = Field(default=None, max_length=MAX_FINANCE_INCOME_EVENTS)
    event_rules: list[EventRule] | None = Field(default=None, max_length=MAX_EVENT_RULES)
    assumptions: ModelAssumptions | None = None
    policy: PlanningPolicy | None = None
    mode: FinanceMode | None = None

    @model_validator(mode="after")
    def replacement_ids_are_unique(self) -> "ScenarioOverrides":
        for field_name in ("bills", "income_events", "event_rules", "assumptions", "policy", "mode"):
            if field_name in self.model_fields_set and getattr(self, field_name) is None:
                raise ValueError(f"Scenario override '{field_name}' must be omitted, not null.")
        if self.bills is not None:
            _require_unique_ids(self.bills, "Scenario bills")
        if self.income_events is not None:
            _require_unique_ids(self.income_events, "Scenario income events")
        if self.event_rules is not None:
            _require_unique_ids(self.event_rules, "Scenario event rules", "event_id")
        return self

    @model_serializer(mode="wrap")
    def serialize_only_present_values(self, handler):
        payload = handler(self)
        if not isinstance(payload, dict):
            return payload
        return {key: value for key, value in payload.items() if value is not None}


class NamedScenario(_FinanceModel):
    id: UUID
    name: Annotated[str, Field(min_length=1, max_length=MAX_FINANCE_NAME_LENGTH)]
    base_revision: Annotated[StrictInt, Field(ge=0, le=MAX_WORKSPACE_REVISION)]
    overrides: ScenarioOverrides

    @model_validator(mode="after")
    def name_is_trimmed(self) -> "NamedScenario":
        _require_trimmed(self.name, "Scenario names")
        return self


class AlertPreferences(_FinanceModel):
    enabled: StrictBool = True
    shortfall_probability: Annotated[FiniteNumber, Field(ge=0, le=1)] = 0.05
    stale_after_days: Annotated[StrictInt, Field(ge=1, le=3_650)] = 30


def _last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _is_recurrence_occurrence(
    anchor: date,
    recurrence: EventRecurrence,
    occurrence: date,
) -> bool:
    if occurrence < anchor:
        return False
    if recurrence == "none":
        return occurrence == anchor
    if recurrence == "weekly":
        return (occurrence - anchor).days % 7 == 0
    if recurrence == "biweekly":
        return (occurrence - anchor).days % 14 == 0
    if recurrence == "monthly":
        month_offset = (occurrence.year - anchor.year) * 12 + occurrence.month - anchor.month
        expected_day = min(anchor.day, _last_day_of_month(occurrence.year, occurrence.month))
        return occurrence.day == expected_day and month_offset >= 0
    year_offset = occurrence.year - anchor.year
    expected_day = min(anchor.day, _last_day_of_month(occurrence.year, anchor.month))
    return (
        year_offset >= 0
        and occurrence.month == anchor.month
        and occurrence.day == expected_day
    )


def _validate_event_rules(
    cash_bills: Sequence[CashBill],
    income_events: Sequence[CashBill],
    event_rules: Sequence[EventRule],
    *,
    scope: str,
) -> None:
    _require_unique_ids(cash_bills, f"{scope} bills")
    _require_unique_ids(income_events, f"{scope} income events")
    all_events = [*cash_bills, *income_events]
    _require_unique_ids(all_events, f"{scope} cash-flow events")
    _require_unique_ids(event_rules, f"{scope} event rules", "event_id")

    events_by_id = {event.id: event for event in all_events}
    for rule in event_rules:
        event = events_by_id.get(rule.event_id)
        if event is None:
            raise ValueError(f"{scope} event rules must refer to a saved bill or income event.")
        if rule.end_date is not None and rule.end_date < event.due_date:
            raise ValueError(f"{scope} recurrence end dates cannot precede their event date.")
        for settlement in rule.settlements:
            if rule.end_date is not None and settlement.due_date > rule.end_date:
                raise ValueError(f"{scope} settlements cannot be after the recurrence end date.")
            if not _is_recurrence_occurrence(event.due_date, rule.recurrence, settlement.due_date):
                raise ValueError(
                    f"{scope} settlement dates must be occurrences anchored to the original event date."
                )


class FinanceInputs(_FinanceModel):
    roth_contribution_basis_cents: Annotated[StrictInt, Field(ge=0, le=MAX_ABS_BALANCE_CENTS)] = 0
    mode: FinanceMode = "scheduled"
    income_events: list[CashBill] = Field(default_factory=list, max_length=MAX_FINANCE_INCOME_EVENTS)
    event_rules: list[EventRule] = Field(default_factory=list, max_length=MAX_EVENT_RULES)
    transactions: list[HistoricalTransaction] = Field(
        default_factory=list,
        max_length=MAX_FINANCE_TRANSACTIONS,
    )
    history_start: WorkspaceDate | None = None
    history_end: WorkspaceDate | None = None
    history_complete: StrictBool = False
    assumptions: ModelAssumptions = Field(default_factory=ModelAssumptions)
    credit_accounts: list[PersonalCreditAccount] = Field(default_factory=list, max_length=MAX_CREDIT_ACCOUNTS)
    holdings: list[PersonalHolding] = Field(default_factory=list, max_length=MAX_HOLDINGS)
    portfolio_returns: list[PortfolioReturn] = Field(default_factory=list, max_length=MAX_PORTFOLIO_RETURNS)
    policy: PlanningPolicy = Field(default_factory=PlanningPolicy)
    scenarios: list[NamedScenario] = Field(default_factory=list, max_length=MAX_NAMED_SCENARIOS)
    alerts: AlertPreferences = Field(default_factory=AlertPreferences)

    @model_validator(mode="after")
    def fields_are_consistent(self) -> "FinanceInputs":
        _require_unique_ids(self.income_events, "Income events")
        _require_unique_ids(self.event_rules, "Event rules", "event_id")
        _require_unique_ids(self.transactions, "Historical transactions")
        _require_unique_ids(self.credit_accounts, "Credit accounts")
        _require_unique_ids(self.holdings, "Holdings")
        _require_unique_ids(self.scenarios, "Named scenarios")

        lot_ids = [lot.id for holding in self.holdings for lot in holding.tax_lots]
        if len(set(lot_ids)) != len(lot_ids):
            raise ValueError("Tax lot IDs must be unique across holdings.")
        aggregate_market_value_cents = sum(
            (
                Decimal(str(lot.quantity)) * holding.current_price_cents
                for holding in self.holdings
                for lot in holding.tax_lots
            ),
            Decimal(),
        )
        aggregate_cost_basis_cents = sum(
            (
                Decimal(str(lot.quantity)) * lot.cost_basis_per_share_cents
                for holding in self.holdings
                for lot in holding.tax_lots
            ),
            Decimal(),
        )
        if (
            aggregate_market_value_cents > MAX_AGGREGATE_HOLDING_CENTS
            or aggregate_cost_basis_cents > MAX_AGGREGATE_HOLDING_CENTS
        ):
            raise ValueError("Aggregate holding market value and cost basis must remain within supported cents.")


        return_dates = [record.date for record in self.portfolio_returns]
        if len(set(return_dates)) != len(return_dates):
            raise ValueError("Portfolio return dates cannot repeat.")
        if len(self.model_dump_json().encode("utf-8")) > MAX_FINANCE_INPUT_BYTES:
            raise ValueError("Supplemental finance inputs are too large to save.")


        # Pin ASCII case matching to SQL; Unicode casing varies by DB locale.
        scenario_names = [scenario.name.translate(_SCENARIO_NAME_CASE) for scenario in self.scenarios]
        if len(set(scenario_names)) != len(scenario_names):
            raise ValueError("Named scenario names cannot repeat.")

        if (self.history_start is None) != (self.history_end is None):
            raise ValueError("History start and end dates must be saved together.")
        if self.history_start is not None and self.history_end is not None:
            if self.history_start > self.history_end:
                raise ValueError("History start cannot be after history end.")
            outside_history = [
                transaction
                for transaction in self.transactions
                if not self.history_start <= transaction.date <= self.history_end
            ]
            if outside_history:
                raise ValueError("Historical transactions must fall inside the saved history coverage.")
        if self.history_complete and self.history_start is None:
            raise ValueError("Complete history requires start and end dates.")
        return self

    def validate_against_cash_bills(self, cash_bills: Sequence[CashBill]) -> None:
        """Validate event references after a complete cash snapshot is available."""

        _validate_event_rules(
            cash_bills,
            self.income_events,
            self.event_rules,
            scope="Saved",
        )
        for scenario in self.scenarios:
            overrides = scenario.overrides
            scenario_bills = cash_bills if overrides.bills is None else overrides.bills
            scenario_income = self.income_events if overrides.income_events is None else overrides.income_events
            scenario_rules = self.event_rules if overrides.event_rules is None else overrides.event_rules
            _validate_event_rules(
                scenario_bills,
                scenario_income,
                scenario_rules,
                scope=f"Scenario '{scenario.name}'",
            )


def default_finance_inputs() -> FinanceInputs:
    """Return a fresh scheduled-mode default without synthetic user data."""

    return FinanceInputs()


class FinanceWorkspace(CashWorkspace):
    """Cash workspace plus its supplemental owner-scoped financial inputs."""

    inputs: FinanceInputs = Field(default_factory=default_finance_inputs)

    @model_validator(mode="after")
    def validates_complete_finance_snapshot(self) -> "FinanceWorkspace":
        _require_unique_ids(self.accounts, "Cash accounts")
        self.inputs.validate_against_cash_bills(self.bills)
        return self


class SaveFinanceRequest(_FinanceModel):
    """Full canonical finance snapshot written with the cash CAS revision."""

    expected_revision: Annotated[StrictInt, Field(ge=0, le=MAX_EXPECTED_REVISION)]
    as_of: WorkspaceDate
    accounts: list[CashAccount] = Field(max_length=50)
    bills: list[CashBill] = Field(max_length=200)
    inputs: FinanceInputs

    @model_validator(mode="after")
    def validates_complete_finance_draft(self) -> "SaveFinanceRequest":
        _require_unique_ids(self.accounts, "Cash accounts")
        self.inputs.validate_against_cash_bills(self.bills)
        return self
