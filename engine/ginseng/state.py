"""Financial state model (spec section 7).

`FinancialState` is the single source of truth for every number Ginseng
displays. The three funding classes (spec section 8) and the funding-class
histories are always *derived* from the underlying `Transaction` ledger and
`Holding` records rather than stored as independent constants: nothing in
this module hardcodes a balance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


class TransactionType(str, Enum):
    """Explicit transaction semantics (spec section 11).

    A transaction today, a future liability, a credit liability, and an
    investment settlement are never conflated: each has distinct timing and
    a distinct effect on settled cash.
    """

    INCOME_FIXED = "income_fixed"
    INCOME_VARIABLE = "income_variable"
    EXPENSE_FIXED = "expense_fixed"
    EXPENSE_ESSENTIAL_VARIABLE = "expense_essential_variable"
    EXPENSE_DISCRETIONARY_VARIABLE = "expense_discretionary_variable"
    EXPENSE_IRREGULAR = "expense_irregular"
    FUTURE_OBLIGATION = "future_obligation"
    TRANSFER = "transfer"
    CREDIT_PURCHASE = "credit_purchase"
    CREDIT_PAYMENT = "credit_payment"
    INVESTMENT_BUY = "investment_buy"
    INVESTMENT_SELL = "investment_sell"


# Transaction types that settle immediately as a cash inflow or outflow.
# CREDIT_PURCHASE moves a credit-card balance, not settled cash. FUTURE_OBLIGATION
# is, by definition, not yet due. TRANSFER carries its own signed amount (it is
# used for the ledger's opening-balance entry and for account-to-account moves).
_CASH_INFLOW_TYPES = frozenset(
    {
        TransactionType.INCOME_FIXED,
        TransactionType.INCOME_VARIABLE,
        TransactionType.INVESTMENT_SELL,
    }
)
_CASH_OUTFLOW_TYPES = frozenset(
    {
        TransactionType.EXPENSE_FIXED,
        TransactionType.EXPENSE_ESSENTIAL_VARIABLE,
        TransactionType.EXPENSE_DISCRETIONARY_VARIABLE,
        TransactionType.EXPENSE_IRREGULAR,
        TransactionType.CREDIT_PAYMENT,
        TransactionType.INVESTMENT_BUY,
    }
)


@dataclass(frozen=True)
class Transaction:
    """One dated ledger entry.

    `amount` is a non-negative magnitude for every type except `TRANSFER`,
    whose `amount` is signed (positive credits settled cash, negative debits
    it) so it can represent both the ledger's opening-balance entry and
    ordinary account transfers.
    """

    txn_date: date
    transaction_type: TransactionType
    amount: float
    label: str

    @property
    def cash_effect(self) -> float:
        """Signed effect of this transaction on settled (immediate) cash."""
        if self.transaction_type is TransactionType.TRANSFER:
            return self.amount
        if self.transaction_type in _CASH_INFLOW_TYPES:
            return self.amount
        if self.transaction_type in _CASH_OUTFLOW_TYPES:
            return -self.amount
        return 0.0


@dataclass(frozen=True)
class Obligation:
    """A scheduled or future dollar event.

    Covers three roles from spec sections 10-11 and 37 with one shape: a
    recurring fixed-income line (`transaction_type=INCOME_FIXED`), a
    recurring fixed obligation (`EXPENSE_FIXED`/`CREDIT_PAYMENT`), or a
    one-off future obligation such as the canonical emergency repair
    (`FUTURE_OBLIGATION`, the default). `due_in_days` counts forward from
    the state's `as_of` date; `recurrence_days` is `None` for a one-off
    event or the repeat interval in days for a recurring line.
    """

    id: str
    label: str
    amount: float
    due_in_days: int
    transaction_type: TransactionType = TransactionType.FUTURE_OBLIGATION
    recurrence_days: int | None = None


@dataclass(frozen=True)
class TaxLot:
    """One tax lot: a dated purchase of shares at a specific cost basis."""

    lot_id: str
    symbol: str
    quantity: float
    cost_basis_per_share: float
    purchase_date: date

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.cost_basis_per_share

    def market_value(self, current_price: float) -> float:
        return self.quantity * current_price

    def unrealized_gain(self, current_price: float) -> float:
        return self.market_value(current_price) - self.cost_basis


@dataclass(frozen=True)
class Holding:
    """An aggregated position in one symbol, held in one account.

    `account` identifies the tax wrapper: "taxable", "traditional", or
    "roth". The legacy "retirement" value remains restricted until its
    tax wrapper is known; it is never silently treated as a traditional IRA.
    """

    symbol: str
    account: str
    current_price: float
    tax_lots: tuple[TaxLot, ...]

    @property
    def quantity(self) -> float:
        return sum(lot.quantity for lot in self.tax_lots)

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return sum(lot.cost_basis for lot in self.tax_lots)

    @property
    def unrealized_gain(self) -> float:
        return self.market_value - self.cost_basis


@dataclass(frozen=True)
class CreditAccount:
    """A synthetic credit card (spec section 39)."""

    account_id: str
    credit_limit: float
    current_balance: float
    purchase_apr: float
    statement_close_day: int
    payment_due_day: int
    grace_period_eligible: bool
    minimum_payment: float
    cash_advance_limit: float | None = None
    cash_advance_fee_pct: float = 0.0

    @property
    def available_credit(self) -> float:
        available = max(0.0, self.credit_limit - self.current_balance)
        return min(available, self.cash_advance_limit) if self.cash_advance_limit is not None else available


@dataclass(frozen=True)
class AssetReturnHistory:
    symbol: str
    daily_returns: tuple[tuple[date, float], ...]


@dataclass(frozen=True)
class FinancialState:
    """The financial digital twin (spec section 7).

    Every headline number a screen shows must be reachable from this
    object's fields or its derived properties; no critical metric is a
    separately authored constant.
    """

    as_of: date
    transactions: tuple[Transaction, ...]
    fixed_income_schedule: tuple[Obligation, ...]
    fixed_obligations: tuple[Obligation, ...]
    planned_discretionary_events: tuple[Obligation, ...]
    credit_accounts: tuple[CreditAccount, ...]
    holdings: tuple[Holding, ...]
    operating_buffer: float
    coverage_target: float
    forecast_horizon: int
    # Spec 8.3 (wrong-way risk): optional daily simple returns of the
    # marketable portfolio over the recorded history window, keyed by date
    # so they align with the joint bootstrap's day indices. Empty when the
    # twin carries no market history, which disables all portfolio-path
    # features rather than fabricating a flat market.
    portfolio_daily_returns: tuple[tuple[date, float], ...] = ()
    # Optional first day of complete classified history.  A zero-flow day is
    # still an observed day; keeping this boundary separately avoids inventing
    # a transaction merely to anchor a sparse real ledger.
    history_start: date | None = None
    # Last day of complete classified history.  Forecast ``as_of`` remains
    # opening-of-day and can be later, so bootstrap training never leaks
    # unobserved forecast-period zeroes into the sampled process.
    history_end: date | None = None
    asset_daily_returns: tuple[AssetReturnHistory, ...] = ()
    # Remaining regular Roth IRA contributions across the owner's Roth IRAs,
    # after prior distributions. This is NOT investment-lot cost basis.
    # Unknown basis defaults to zero accessible contributions.
    roth_contribution_basis: float = 0.0

    # ---- Funding classes (spec section 8): always derived ----

    @property
    def immediate_funding(self) -> float:
        """Spec 8.1: settled cash, derived from the transaction ledger."""
        return sum(t.cash_effect for t in self.transactions)

    @property
    def marketable_backup_capital(self) -> float:
        """Spec 8.2: taxable holdings at current market value."""
        return sum(h.market_value for h in self.holdings if h.account == "taxable")

    @property
    def restricted_capital(self) -> float:
        """Spec 8.3: retirement/restricted holdings at current market value."""
        return sum(h.market_value for h in self.holdings if h.account in ("retirement", "traditional", "roth"))

    @property
    def traditional_capital(self) -> float:
        return sum(h.market_value for h in self.holdings if h.account == "traditional")

    @property
    def roth_capital(self) -> float:
        return sum(h.market_value for h in self.holdings if h.account == "roth")

    # ---- Derived views over the ledger (spec section 7, 10) ----

    @property
    def taxable_portfolio(self) -> tuple[Holding, ...]:
        return tuple(h for h in self.holdings if h.account == "taxable")

    @property
    def tax_lots(self) -> tuple[TaxLot, ...]:
        return tuple(lot for h in self.taxable_portfolio for lot in h.tax_lots)

    @property
    def variable_income_history(self) -> tuple[Transaction, ...]:
        return self._by_type(TransactionType.INCOME_VARIABLE)

    @property
    def essential_variable_spending_history(self) -> tuple[Transaction, ...]:
        return self._by_type(TransactionType.EXPENSE_ESSENTIAL_VARIABLE)

    @property
    def discretionary_spending_history(self) -> tuple[Transaction, ...]:
        return self._by_type(TransactionType.EXPENSE_DISCRETIONARY_VARIABLE)

    @property
    def irregular_obligations(self) -> tuple[Transaction, ...]:
        """Spec 10: stored, but never mixed into routine spending distributions."""
        return self._by_type(TransactionType.EXPENSE_IRREGULAR)

    def _by_type(self, transaction_type: TransactionType) -> tuple[Transaction, ...]:
        return tuple(t for t in self.transactions if t.transaction_type is transaction_type)

    def daily_series(self, transaction_type: TransactionType, start: date, end: date) -> "pd.Series":
        """A dense daily series of transaction magnitudes for `transaction_type`
        over `[start, end]`, zero-filled on days with no matching activity."""
        import pandas as pd

        idx = pd.date_range(start, end, freq="D")
        matches = [
            (pd.Timestamp(t.txn_date), t.amount)
            for t in self.transactions
            if t.transaction_type is transaction_type and start <= t.txn_date <= end
        ]
        if not matches:
            return pd.Series(0.0, index=idx)
        dates_, amounts_ = zip(*matches)
        raw = pd.Series(amounts_, index=pd.DatetimeIndex(dates_), dtype=float)
        return raw.groupby(level=0).sum().reindex(idx, fill_value=0.0)
