"""Account-wrapper liquidity, using versioned constants with no network lookup.

Scope: taxable brokerage, fully pretax traditional IRA, and documented regular
Roth IRA contributions. No conversion/earnings eligibility, retirement-plan
access rules, loss rebates, or exact tax return is inferred from a holding.
"""

from dataclasses import asdict, dataclass
from datetime import date
from math import isfinite

from ginseng.state import FinancialState

ASSUMPTIONS_VERSION = "ira-withdrawals-2026-v1"
ASSUMED_AGE_YEARS = 35
ORDINARY_INCOME_RATE = 0.24
LONG_TERM_CAPITAL_GAINS_RATE = 0.15
EARLY_WITHDRAWAL_PENALTY = 0.10
SALE_SETTLEMENT_DAYS = 1
EXTERNAL_TRANSFER_DAYS = 2
AVAILABILITY_DELAY_DAYS = SALE_SETTLEMENT_DAYS + EXTERNAL_TRANSFER_DAYS


@dataclass(frozen=True)
class WithdrawalAssumptions:
    long_term_rate: float = LONG_TERM_CAPITAL_GAINS_RATE
    ordinary_rate: float = ORDINARY_INCOME_RATE
    early_penalty: float = EARLY_WITHDRAWAL_PENALTY

    def __post_init__(self):
        if any(not isfinite(v) or not 0 <= v <= 1 for v in (self.long_term_rate, self.ordinary_rate, self.early_penalty)):
            raise ValueError("Withdrawal rates must be finite fractions between zero and one.")
        if self.ordinary_rate + self.early_penalty >= 1:
            raise ValueError("Traditional withdrawal charges must leave positive spendable cash.")


DEFAULT_ASSUMPTIONS = WithdrawalAssumptions()


def is_long_term(purchase_date: date, as_of: date) -> bool:
    try:
        anniversary = purchase_date.replace(year=purchase_date.year + 1)
    except ValueError:
        anniversary = date(purchase_date.year + 1, 2, 28)
    return as_of > anniversary


@dataclass(frozen=True)
class WithdrawalUnit:
    key: str
    account_type: str
    capacity: float
    tax_per_dollar: float
    penalty_per_dollar: float = 0.0
    lot_id: str | None = None
    gain_fraction: float = 0.0
    holding_period: str | None = None

    @property
    def net_per_dollar(self):
        return 1 - self.tax_per_dollar - self.penalty_per_dollar


@dataclass(frozen=True)
class WithdrawalAllocation:
    key: str
    gross: float


@dataclass(frozen=True)
class AccountWithdrawal:
    account_type: str
    gross: float
    tax_reserve: float
    penalty_reserve: float
    net_cash: float


@dataclass(frozen=True)
class WithdrawalQuote:
    gross: float
    tax_reserve: float
    penalty_reserve: float
    net_cash: float
    realized_taxable_gain: float
    accounts: tuple[AccountWithdrawal, ...]
    allocations: tuple[WithdrawalAllocation, ...]


def withdrawal_units(state: FinancialState, assumptions=DEFAULT_ASSUMPTIONS) -> tuple[WithdrawalUnit, ...]:
    units = []
    if not isfinite(state.roth_contribution_basis) or state.roth_contribution_basis < 0:
        raise ValueError("Remaining Roth regular-contribution basis must be finite and nonnegative.")
    for holding in state.holdings:
        if not isfinite(holding.current_price) or holding.current_price <= 0:
            raise ValueError("Withdrawal holdings need a positive, finite current price.")
        for lot in holding.tax_lots:
            if (not isfinite(lot.quantity) or lot.quantity < 0 or
                    not isfinite(lot.cost_basis_per_share) or lot.cost_basis_per_share < 0):
                raise ValueError("Withdrawal lots need finite, nonnegative quantities and basis.")
            if holding.account != "taxable" or lot.quantity == 0:
                continue
            long_term = is_long_term(lot.purchase_date, state.as_of)
            gain = 1 - lot.cost_basis_per_share / holding.current_price
            rate = assumptions.long_term_rate if long_term else assumptions.ordinary_rate
            units.append(WithdrawalUnit(f"taxable:{lot.lot_id}", "taxable", lot.market_value(holding.current_price),
                         max(0, gain) * rate, lot_id=lot.lot_id, gain_fraction=gain,
                         holding_period="long_term" if long_term else "short_term"))
    if state.traditional_capital > 0:
        units.append(WithdrawalUnit("traditional", "traditional", state.traditional_capital,
                                    assumptions.ordinary_rate, assumptions.early_penalty))
    roth_available = min(state.roth_capital, state.roth_contribution_basis)
    if roth_available > 0:
        units.append(WithdrawalUnit("roth", "roth", roth_available, 0.0))
    if len({u.key for u in units}) != len(units):
        raise ValueError("Taxable lot identifiers must be unique across accounts.")
    return tuple(units)


def quote_withdrawals(state, allocations, assumptions=DEFAULT_ASSUMPTIONS) -> WithdrawalQuote:
    units = {u.key: u for u in withdrawal_units(state, assumptions)}
    amounts = {}
    for item in allocations:
        if item.key not in units or not isfinite(item.gross) or item.gross < 0:
            raise ValueError("Withdrawal allocation is invalid or its account is unavailable.")
        amounts[item.key] = amounts.get(item.key, 0.0) + item.gross
    for key, value in amounts.items():
        if value > units[key].capacity + 1e-6:
            raise ValueError("Withdrawal exceeds its holding value or remaining Roth contribution basis.")
    rows = []
    for kind in ("taxable", "traditional", "roth"):
        matching = [(units[key], value) for key, value in amounts.items() if units[key].account_type == kind]
        gross = sum(value for _, value in matching)
        tax = sum(u.tax_per_dollar * value for u, value in matching)
        penalty = sum(u.penalty_per_dollar * value for u, value in matching)
        rows.append(AccountWithdrawal(kind, gross, tax, penalty, gross - tax - penalty))
    return WithdrawalQuote(sum(r.gross for r in rows), sum(r.tax_reserve for r in rows),
        sum(r.penalty_reserve for r in rows), sum(r.net_cash for r in rows),
        sum(units[key].gain_fraction * value for key, value in amounts.items()), tuple(rows),
        tuple(WithdrawalAllocation(key, value) for key, value in amounts.items() if value > 0))


def allocate_net_cash(units, target: float) -> tuple[WithdrawalAllocation, ...]:
    """Exact fractional allocation for a fixed net target and shared timing.

    All units settle on the same day. Cost per net dollar determines the
    allocation; equal costs prefer taxable before Roth before traditional,
    preserving retirement capital without inventing an objective penalty.
    """
    if not isfinite(target) or target < 0:
        raise ValueError("Net withdrawal target must be finite and nonnegative.")
    rank = {"taxable": 0, "roth": 1, "traditional": 2}
    available = sorted((u for u in units if u.net_per_dollar > 0),
        key=lambda u: ((1 - u.net_per_dollar) / u.net_per_dollar, rank[u.account_type], u.key))
    allocations = []
    remaining = target
    for unit in available:
        gross = min(unit.capacity, remaining / unit.net_per_dollar)
        if gross > 0:
            allocations.append(WithdrawalAllocation(unit.key, gross))
        remaining = max(0.0, remaining - gross * unit.net_per_dollar)
    if remaining > 1e-5:
        raise ValueError("Net withdrawal target exceeds accessible account capacity.")
    return tuple(allocations)


def account_liquidity(state, assumptions=DEFAULT_ASSUMPTIONS) -> dict:
    units = withdrawal_units(state, assumptions)
    maximum = quote_withdrawals(state, [WithdrawalAllocation(u.key, u.capacity) for u in units], assumptions)
    balances = {"taxable": state.marketable_backup_capital, "traditional": state.traditional_capital, "roth": state.roth_capital}
    return {"accounts": [{**asdict(row), "balance": balances[row.account_type],
                          "excluded_balance": (max(0.0, balances["roth"] - row.gross)
                                               if row.account_type == "roth" else 0.0)} for row in maximum.accounts],
            "total_net_accessible": maximum.net_cash,
            "roth_contribution_basis": state.roth_contribution_basis,
            "unclassified_retirement_balance": sum(h.market_value for h in state.holdings if h.account == "retirement"),
            "assumptions_version": ASSUMPTIONS_VERSION,
            "availability_delay_days": AVAILABILITY_DELAY_DAYS,
            "assumptions": [
                {"label": "Ordinary income / short-term gain rate", "value": f"{assumptions.ordinary_rate * 100:g}%",
                 "source": "Selected model marginal federal rate; not inferred from income or a tax return.", "url": None},
                {"label": "Long-term capital gains rate", "value": f"{assumptions.long_term_rate * 100:g}%",
                 "source": "Selected marginal rate, applied only to positive lot gains held more than one year.", "url": "https://www.irs.gov/taxtopics/tc409"},
                {"label": "Traditional IRA early-withdrawal penalty", "value": f"{assumptions.early_penalty * 100:g}%",
                 "source": f"Assumed age {ASSUMED_AGE_YEARS}, no exception, fully pretax traditional IRA. Ordinary income tax also applies.", "url": "https://www.irs.gov/taxtopics/tc557"},
                {"label": "Roth IRA regular contributions", "value": "0% tax / 0% penalty",
                 "source": "Regular contributions come out first; capped at documented remaining contributions and account value. Earnings and conversions are excluded.", "url": "https://www.irs.gov/publications/p590b"},
                {"label": "Availability and tax reserve", "value": f"{AVAILABILITY_DELAY_DAYS} calendar days",
                 "source": "Model assumption for sale plus transfer. Tax and penalty estimates are set aside immediately from proceeds; this is not actual withholding or a tax payment date.", "url": None},
                {"label": "Other tax effects", "value": "Excluded",
                 "source": "State tax, NIIT, netting, loss deductions, bracket crossings, IRA nondeductible basis and exceptions require more household information. Losses create no cash rebate.", "url": None},
            ],
            "scope": "Fixed planning assumptions for taxable brokerage and IRAs, not employer retirement plans. No tax rules are fetched at runtime.",
            "tie_break": "At equal modeled cost, use taxable funds before Roth contributions before traditional withdrawals."}
