"""Deterministic synthetic persona generator (spec sections 2, 9, 10, 17, 36).

Produces 24 months of daily financial history for one variable-income
persona from a single `numpy.random.default_rng(seed)` stream (no global
random state), then explicitly checks the persona against the section 36
acceptance conditions via `acceptance_report`.

The variable series carry genuine short-range temporal dependence: a sticky
busy/dry work-climate regime clusters client payments into multi-week busy
runs separated by multi-week dry spells, and both spending series tighten
during dry spells while keeping mild momentum of their own. The composite
net-flow series Z (spec 17) therefore has a Politis-White mean block length
strictly inside the [7, 28] clip window - the variable-income persistence
that spec section 9 says the 24-month window exists to model.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from ginseng.metrics import compute_scenario_metrics
from ginseng.simulate import draw_bundle
from ginseng.state import (
    CreditAccount,
    AssetReturnHistory,
    FinancialState,
    Holding,
    Obligation,
    TaxLot,
    Transaction,
    TransactionType,
)

DEFAULT_SEED = 20260911
AS_OF = date(2026, 9, 11)
HISTORY_DAYS = 730  # 24 months, matching Plaid's max Transactions window (spec 9)
FORECAST_HORIZON = 30
OPERATING_BUFFER = 1000.0
COVERAGE_TARGET = 0.95

CANONICAL_REPAIR_TOTAL = 4500.0
CANONICAL_REPAIR_DEPOSIT_ID = "repair-deposit"
CANONICAL_REPAIR_DEPOSIT_AMOUNT = 1500.0
CANONICAL_REPAIR_DEPOSIT_DUE_IN_DAYS = 3
CANONICAL_REPAIR_BALANCE_ID = "repair-balance"
CANONICAL_REPAIR_BALANCE_AMOUNT = 3000.0
CANONICAL_REPAIR_BALANCE_DUE_IN_DAYS = 17

_IRREGULAR_LABELS = ("Travel", "Appliance repair", "One-off purchase", "Vet bill", "Home repair")

# --- Work-climate regime (spec sections 2, 9, 10, 15, 17) ---
# A sticky two-state busy/dry Markov chain drives the stochastic series, so
# work arrives in clusters and dries up in spells the way variable income
# actually behaves: busy runs average 1/(1-0.965) ~= 29 days and dry runs
# 1/(1-0.935) ~= 15 days, with a stationary mix near 65% busy. This puts
# genuine short-range temporal dependence in the composite net-flow series
# Z (spec 17), so its Politis-White estimate lands mid-window instead of on
# the 7-day floor - the persistence spec 9 invokes 24 months of history to
# model (spec section 9's stated reason for the long window).
_REGIME_STAY_BUSY = 0.965
_REGIME_STAY_DRY = 0.935

# Client-payment behavior by regime (spec 10): during busy runs small
# client/commission payments land on most days; during dry runs they
# trickle in occasionally at reduced size.
_PAYMENT_RATE_BUSY = 0.85
_PAYMENT_RATE_DRY = 0.15
_PAYMENT_MEAN_BUSY = 310.0
_PAYMENT_MEAN_DRY = 280.0
_PAYMENT_SHAPE = 6.0  # gamma shape; sd = mean / sqrt(6)

# Spending behavior by regime (spec sections 10, 15): essential spending
# tightens modestly and discretionary spending sharply during dry spells,
# so the three stochastic series stay cross-correlated with the income
# regime the way spec 15's joint resampling assumes.
_ESSENTIAL_RATE_BUSY = 0.65
_ESSENTIAL_RATE_DRY = 0.47
_ESSENTIAL_MEAN_BUSY = 36.0
_ESSENTIAL_MEAN_DRY = 26.4
_ESSENTIAL_SHAPE = 2.5

_DISCRETIONARY_RATE_BUSY = 0.52
_DISCRETIONARY_RATE_DRY = 0.22
_DISCRETIONARY_MEAN_BUSY = 32.0
_DISCRETIONARY_MEAN_DRY = 23.0
_DISCRETIONARY_SHAPE = 2.0

# Mild spending momentum of its own, independent of the income regime
# (spec 15): an AR(1) log-intensity multiplies each series' occurrence
# probability, so spending has household-level persistence beyond the
# regime it shares with income.
_SPEND_MOMENTUM = 0.75
_SPEND_MOMENTUM_SD = 0.18

# Market returns (spec 8.3): geometric Brownian motion whose drift follows
# the same busy/dry regime as income, so income-market correlation emerges
# naturally when the joint bootstrap resamples shared day indices. Returns
# are stored as simple returns derived from the GBM log returns,
# `exp(mu dt + sigma sqrt(dt) z) - 1`, so every daily growth factor is
# strictly positive and portfolio paths can never go below zero.
_MARKET_MU_BUSY = 0.10 / 365       # ~10% annualised drift in busy regimes
_MARKET_MU_DRY = -0.05 / 365       # mild drag through dry spells
_MARKET_SIGMA = 0.16 / np.sqrt(365)  # ~16% annualised volatility


def _next_occurrence(as_of: date, day_of_month: int) -> int:
    """Days from `as_of` (inclusive of `as_of` itself, offset 0) until the
    next date whose day-of-month is `day_of_month`."""
    for offset in range(0, 32):
        if (as_of + timedelta(days=offset)).day == day_of_month:
            return offset
    raise ValueError(f"no day-of-month {day_of_month} found within a month of {as_of}")


def _regime_path(rng: np.random.Generator, stay_busy: float, stay_dry: float, n_days: int) -> np.ndarray:
    """A sticky two-state busy/dry chain: each day continues the current
    state with its state's stay probability and otherwise flips."""
    draws = rng.random(n_days)
    busy = np.empty(n_days, dtype=bool)
    busy[0] = True
    for i in range(1, n_days):
        stay = draws[i] < (stay_busy if busy[i - 1] else stay_dry)
        busy[i] = busy[i - 1] if stay else not busy[i - 1]
    return busy


def _spend_momentum_path(rng: np.random.Generator, phi: float, innovation_sd: float, n_days: int) -> np.ndarray:
    """An AR(1) log-intensity path with a stationary start: mild spending
    momentum of its own, independent of the income regime (spec 15)."""
    shocks = rng.standard_normal(n_days) * innovation_sd
    path = np.empty(n_days)
    level = path[0] = shocks[0] / np.sqrt(1.0 - phi * phi)
    for i in range(1, n_days):
        level = phi * level + shocks[i]
        path[i] = level
    return path


def generate_persona(seed: int = DEFAULT_SEED) -> FinancialState:
    """Generate the canonical demo persona (spec sections 9, 10, 36):
    24 months of daily history for one variable-income investor with a
    taxable portfolio, a restricted retirement account, and one credit
    card, tuned so the section 36 acceptance conditions hold.

    Income and spending are driven by a persistent busy/dry work-climate
    regime (spec sections 2, 10): client payments cluster inside busy runs
    and dry spells stretch across multiple weeks, and spending tightens in
    step with income. The resulting composite net-flow persistence keeps
    the spec 17 block-length estimate strictly inside its clip window.
    """
    rng = np.random.default_rng(seed)
    start = AS_OF - timedelta(days=HISTORY_DAYS - 1)
    dates = [start + timedelta(days=i) for i in range(HISTORY_DAYS)]

    transactions: list[Transaction] = []

    # --- Fixed income: a modest stable retainer (spec 10) ---
    retainer_amount = float(rng.uniform(300.0, 450.0))
    retainer_day = int(rng.integers(1, 6))
    for d in dates:
        if d.day == retainer_day:
            transactions.append(Transaction(d, TransactionType.INCOME_FIXED, retainer_amount, "Monthly retainer"))

    # --- Fixed expenses: housing, insurance, subscriptions (spec 10) ---
    rent = float(rng.uniform(1100.0, 1350.0))
    insurance = float(rng.uniform(110.0, 170.0))
    subscriptions = float(rng.uniform(25.0, 55.0))
    rent_day = int(rng.integers(1, 4))
    insurance_day = int(rng.integers(5, 10))
    subs_day = int(rng.integers(10, 15))
    for d in dates:
        if d.day == rent_day:
            transactions.append(Transaction(d, TransactionType.EXPENSE_FIXED, rent, "Rent"))
        if d.day == insurance_day:
            transactions.append(Transaction(d, TransactionType.EXPENSE_FIXED, insurance, "Insurance"))
        if d.day == subs_day:
            transactions.append(Transaction(d, TransactionType.EXPENSE_FIXED, subscriptions, "Subscriptions"))

    # --- Work-climate regime: busy runs and multi-week dry spells
    # (spec sections 2, 9, 10) ---
    busy = _regime_path(rng, _REGIME_STAY_BUSY, _REGIME_STAY_DRY, HISTORY_DAYS)

    # --- Variable income: client payments clustered inside busy runs and
    # trickling through dry spells (spec sections 10, 17) ---
    payment_rate = np.where(busy, _PAYMENT_RATE_BUSY, _PAYMENT_RATE_DRY)
    paid = rng.random(HISTORY_DAYS) < payment_rate
    payment_mean = np.where(busy, _PAYMENT_MEAN_BUSY, _PAYMENT_MEAN_DRY)
    payment_amounts = rng.gamma(_PAYMENT_SHAPE, 1.0, HISTORY_DAYS) * (payment_mean / _PAYMENT_SHAPE)
    for d, occurs, amount in zip(dates, paid, payment_amounts):
        if occurs:
            transactions.append(
                Transaction(d, TransactionType.INCOME_VARIABLE, float(amount), "Client payment")
            )

    # --- Essential variable spending: groceries/transportation, tightened
    # modestly during dry spells, with mild momentum of its own
    # (spec sections 10, 15) ---
    essential_intensity = _spend_momentum_path(rng, _SPEND_MOMENTUM, _SPEND_MOMENTUM_SD, HISTORY_DAYS)
    essential_rate = (
        np.where(busy, _ESSENTIAL_RATE_BUSY, _ESSENTIAL_RATE_DRY) * np.exp(essential_intensity)
    )
    essential_occurs = rng.random(HISTORY_DAYS) < np.minimum(essential_rate, 0.98)
    essential_mean = np.where(busy, _ESSENTIAL_MEAN_BUSY, _ESSENTIAL_MEAN_DRY)
    essential_amounts = rng.gamma(_ESSENTIAL_SHAPE, 1.0, HISTORY_DAYS) * (
        essential_mean / _ESSENTIAL_SHAPE
    )
    for d, occurs, amount in zip(dates, essential_occurs, essential_amounts):
        if occurs:
            transactions.append(
                Transaction(d, TransactionType.EXPENSE_ESSENTIAL_VARIABLE, float(amount), "Groceries/transportation")
            )

    # --- Discretionary spending: dining/entertainment, cut sharply during
    # dry spells, with mild momentum of its own (spec sections 10, 15) ---
    discretionary_intensity = _spend_momentum_path(rng, _SPEND_MOMENTUM, _SPEND_MOMENTUM_SD, HISTORY_DAYS)
    discretionary_rate = (
        np.where(busy, _DISCRETIONARY_RATE_BUSY, _DISCRETIONARY_RATE_DRY)
        * np.exp(discretionary_intensity)
    )
    discretionary_occurs = rng.random(HISTORY_DAYS) < np.minimum(discretionary_rate, 0.98)
    discretionary_mean = np.where(busy, _DISCRETIONARY_MEAN_BUSY, _DISCRETIONARY_MEAN_DRY)
    discretionary_amounts = rng.gamma(_DISCRETIONARY_SHAPE, 1.0, HISTORY_DAYS) * (
        discretionary_mean / _DISCRETIONARY_SHAPE
    )
    for d, occurs, amount in zip(dates, discretionary_occurs, discretionary_amounts):
        if occurs:
            transactions.append(
                Transaction(d, TransactionType.EXPENSE_DISCRETIONARY_VARIABLE, float(amount), "Dining/entertainment")
            )

    # --- Irregular historical events: stored, kept out of routine distributions (spec 10) ---
    n_irregular = int(rng.integers(3, 6))
    irregular_day_idx = rng.choice(HISTORY_DAYS, size=n_irregular, replace=False)
    irregular_amounts = rng.uniform(200.0, 1200.0, size=n_irregular)
    for i, (idx, amount) in enumerate(zip(irregular_day_idx, irregular_amounts)):
        transactions.append(
            Transaction(
                dates[int(idx)],
                TransactionType.EXPENSE_IRREGULAR,
                float(amount),
                _IRREGULAR_LABELS[i % len(_IRREGULAR_LABELS)],
            )
        )

    # --- Credit card (spec section 39) ---
    credit_limit = float(rng.uniform(5000.0, 9000.0))
    credit_balance = float(rng.uniform(400.0, 1600.0))
    minimum_payment = round(max(35.0, credit_balance * 0.03), 2)
    credit_payment_day = int(rng.integers(15, 21))
    credit_account = CreditAccount(
        account_id="card-1",
        credit_limit=credit_limit,
        current_balance=credit_balance,
        purchase_apr=0.2399,
        statement_close_day=22,
        payment_due_day=credit_payment_day,
        grace_period_eligible=True,
        minimum_payment=minimum_payment,
    )
    for d in dates:
        if d.day == credit_payment_day:
            transactions.append(Transaction(d, TransactionType.CREDIT_PAYMENT, minimum_payment, "Credit card payment"))

    # A handful of ordinary credit-card purchases: no settled-cash effect,
    # only realism for the taxonomy and the card's outstanding balance.
    n_card_purchases = int(rng.integers(8, 16))
    card_purchase_idx = rng.integers(0, HISTORY_DAYS, size=n_card_purchases)
    card_purchase_amounts = rng.uniform(30.0, 180.0, size=n_card_purchases)
    for idx, amount in zip(card_purchase_idx, card_purchase_amounts):
        transactions.append(
            Transaction(dates[int(idx)], TransactionType.CREDIT_PURCHASE, float(amount), "Card purchase")
        )

    # --- Taxable portfolio: tax lots with cost basis (spec 9, 10) ---
    taxable_target = float(rng.uniform(15000.0, 25000.0))
    symbols = ("VTI", "VXUS")
    weights = (0.7, 0.3)
    holdings: list[Holding] = []
    for symbol, weight in zip(symbols, weights):
        target_value = taxable_target * weight
        current_price = float(rng.uniform(80.0, 250.0))
        n_lots = int(rng.integers(2, 4))
        lot_fractions = rng.dirichlet(np.ones(n_lots))
        lots: list[TaxLot] = []
        for i, fraction in enumerate(lot_fractions):
            lot_value = target_value * fraction
            gain_factor = float(rng.uniform(0.75, 1.15))
            cost_basis_per_share = current_price * gain_factor
            quantity = lot_value / current_price
            purchase_offset = int(rng.integers(30, HISTORY_DAYS))
            purchase_date = AS_OF - timedelta(days=purchase_offset)
            lots.append(TaxLot(f"{symbol}-lot-{i + 1}", symbol, quantity, cost_basis_per_share, purchase_date))
            if purchase_date >= start:
                transactions.append(
                    Transaction(
                        purchase_date,
                        TransactionType.INVESTMENT_BUY,
                        quantity * cost_basis_per_share,
                        f"Buy {symbol}",
                    )
                )
        holdings.append(Holding(symbol, "taxable", current_price, tuple(lots)))

    # --- Restricted retirement account (spec 8.3) ---
    restricted_target = float(rng.uniform(20000.0, 42000.0))
    retirement_price = float(rng.uniform(40.0, 120.0))
    retirement_quantity = restricted_target / retirement_price
    retirement_lot = TaxLot(
        "retirement-lot-1",
        "TARGET2055",
        retirement_quantity,
        retirement_price * float(rng.uniform(0.7, 1.0)),
        AS_OF - timedelta(days=int(rng.integers(400, HISTORY_DAYS))),
    )
    # Split the existing retirement balance by tax wrapper, not instrument.
    # No extra random draw changes the cash persona or its historical paths.
    for wrapper, fraction in (("traditional", 0.7), ("roth", 0.3)):
        lot = TaxLot(f"{wrapper}-lot-1", retirement_lot.symbol,
                     retirement_lot.quantity * fraction, retirement_lot.cost_basis_per_share,
                     retirement_lot.purchase_date)
        holdings.append(Holding("TARGET2055", wrapper, retirement_price, (lot,)))

    # --- Opening balance: calibrates immediate funding to the section 36
    # generation target while remaining a transaction-ledger entry, not a
    # separately authored constant (spec 7). The draw is centered inside
    # spec 36's $2,500-$3,500 design range so the shocked persona lands in
    # the target funding-gap band alongside the regime-driven persistence. ---
    target_immediate_funding = float(rng.uniform(2950.0, 3350.0))
    net_cash_flow_so_far = sum(t.cash_effect for t in transactions)
    opening_balance = target_immediate_funding - net_cash_flow_so_far
    transactions.append(Transaction(start, TransactionType.TRANSFER, opening_balance, "Opening balance"))

    transactions.sort(key=lambda t: t.txn_date)

    fixed_income_schedule = (
        Obligation(
            id="retainer",
            label="Monthly retainer",
            amount=retainer_amount,
            due_in_days=_next_occurrence(AS_OF, retainer_day),
            transaction_type=TransactionType.INCOME_FIXED,
            recurrence_days=30,
        ),
    )
    fixed_obligations = (
        Obligation(
            id="rent",
            label="Rent",
            amount=rent,
            due_in_days=_next_occurrence(AS_OF, rent_day),
            transaction_type=TransactionType.EXPENSE_FIXED,
            recurrence_days=30,
        ),
        Obligation(
            id="insurance",
            label="Insurance",
            amount=insurance,
            due_in_days=_next_occurrence(AS_OF, insurance_day),
            transaction_type=TransactionType.EXPENSE_FIXED,
            recurrence_days=30,
        ),
        Obligation(
            id="subscriptions",
            label="Subscriptions",
            amount=subscriptions,
            due_in_days=_next_occurrence(AS_OF, subs_day),
            transaction_type=TransactionType.EXPENSE_FIXED,
            recurrence_days=30,
        ),
        Obligation(
            id="credit_minimum_payment",
            label="Credit card payment",
            amount=minimum_payment,
            due_in_days=_next_occurrence(AS_OF, credit_payment_day),
            transaction_type=TransactionType.CREDIT_PAYMENT,
            recurrence_days=30,
        ),
    )

    # Market returns (spec 8.3): drawn after every other use of `rng` so the
    # existing random stream - and therefore every transaction amount, the
    # opening-balance calibration, and the spec 17 block-length estimate -
    # is bit-for-bit unchanged by this addition. The drift follows the same
    # busy/dry regime path the income series used, so a resampled dry-day
    # block brings its weak market day along with it (joint bootstrap).
    market_log_returns = np.where(busy, _MARKET_MU_BUSY, _MARKET_MU_DRY) + (
        rng.standard_normal(HISTORY_DAYS) * _MARKET_SIGMA
    )
    market_returns = np.exp(market_log_returns) - 1.0
    # Additional per-asset demonstration data use a separate stream so the
    # existing ledger and aggregate market series do not change. Center the
    # idiosyncratic component by current portfolio weights each day: the
    # same weighted combination reproduces the existing aggregate return.
    taxable = [holding for holding in holdings if holding.account == "taxable"]
    asset_weights = np.array([h.market_value for h in taxable])
    asset_weights /= asset_weights.sum()
    asset_rng = np.random.default_rng(np.random.SeedSequence([seed, 8129]))
    noise = asset_rng.normal(0, 0.008, (HISTORY_DAYS, len(taxable)))
    noise -= (noise @ asset_weights)[:, None]
    asset_returns = market_returns[:, None] + noise

    return FinancialState(
        as_of=AS_OF,
        transactions=tuple(transactions),
        fixed_income_schedule=fixed_income_schedule,
        fixed_obligations=fixed_obligations,
        planned_discretionary_events=(),
        credit_accounts=(credit_account,),
        holdings=tuple(holdings),
        operating_buffer=OPERATING_BUFFER,
        coverage_target=COVERAGE_TARGET,
        forecast_horizon=FORECAST_HORIZON,
        portfolio_daily_returns=tuple(
            (d, float(r)) for d, r in zip(dates, market_returns)
        ),
        asset_daily_returns=tuple(
            AssetReturnHistory(holding.symbol, tuple((d, float(r)) for d, r in zip(dates, asset_returns[:, i])))
            for i, holding in enumerate(taxable)
        ),
        roth_contribution_basis=1200.0,  # synthetic remaining regular contributions, not lot basis
    )


def canonical_shocks() -> tuple[Obligation, ...]:
    """The canonical repair schedule: a $1,500 deposit due in three days
    and a $3,000 balance due on day 17.

    The total remains $4,500. Its reserve effect depends on the timing of
    each path's trough; a nearly dollar-for-dollar shift is valid, too.
    """
    return (
        Obligation(
            id=CANONICAL_REPAIR_DEPOSIT_ID,
            label="Emergency vehicle repair deposit",
            amount=CANONICAL_REPAIR_DEPOSIT_AMOUNT,
            due_in_days=CANONICAL_REPAIR_DEPOSIT_DUE_IN_DAYS,
        ),
        Obligation(
            id=CANONICAL_REPAIR_BALANCE_ID,
            label="Emergency vehicle repair balance",
            amount=CANONICAL_REPAIR_BALANCE_AMOUNT,
            due_in_days=CANONICAL_REPAIR_BALANCE_DUE_IN_DAYS,
        ),
    )


def acceptance_report(state: FinancialState, seed: int = DEFAULT_SEED, n_paths: int = 2000) -> dict:
    """Check `state` against both section 36 acceptance blocks and return
    the underlying before/after numbers so the condition is checkable.

    Also reports the Politis-White mean block length estimated from the
    state's composite net-flow history (spec 17): the bundle is drawn with
    `mean_block_length=None`, so `bundle.mean_block_length` is exactly
    `estimate_mean_block_length(Z)`, and the persistence target (strictly
    inside the [7, 28] clip window, distinct from every fixed sensitivity
    row in spec 18) can be checked from this dict without a separate script.

    Before and after share one `DrawBundle` (spec 20: common random
    numbers), so the only difference between the two evaluations is the
    inserted shock obligation.
    """
    bundle = draw_bundle(state, horizon_days=FORECAST_HORIZON, n_paths=n_paths, seed=seed)
    before = compute_scenario_metrics(state, bundle, (), state.coverage_target, state.operating_buffer)
    after = compute_scenario_metrics(
        state, bundle, canonical_shocks(), state.coverage_target, state.operating_buffer
    )

    immediate_funding = state.immediate_funding
    marketable_backup_capital = state.marketable_backup_capital
    available_credit = sum(a.available_credit for a in state.credit_accounts)

    before_conditions = {
        "rlr_within_immediate_funding": before.required_liquidity_reserve <= immediate_funding,
        "cash_shortfall_probability_under_2pct": before.severity["cash_shortfall_probability"] < 0.02,
        "no_funding_gap": before.funding_gap == 0.0,
        "meaningful_taxable_investments": marketable_backup_capital >= 10000.0,
    }
    after_conditions = {
        "funding_gap_in_target_range": 1000.0 <= after.funding_gap <= 3000.0,
        "cash_shortfall_probability_is_probabilistic": 0.10
        <= after.severity["cash_shortfall_probability"]
        <= 0.90,
        "taxable_investments_cover_gap": marketable_backup_capital >= after.funding_gap,
        "credit_can_cover_some_of_gap": available_credit > 0.0,
    }

    return {
        "immediate_funding": immediate_funding,
        "marketable_backup_capital": marketable_backup_capital,
        "restricted_capital": state.restricted_capital,
        "available_credit": available_credit,
        "estimated_mean_block_length": bundle.mean_block_length,
        "before": {
            "required_liquidity_reserve": before.required_liquidity_reserve,
            "funding_gap": before.funding_gap,
            "cash_shortfall_probability": before.severity["cash_shortfall_probability"],
            "conditions": before_conditions,
        },
        "after": {
            "required_liquidity_reserve": after.required_liquidity_reserve,
            "funding_gap": after.funding_gap,
            "cash_shortfall_probability": after.severity["cash_shortfall_probability"],
            "conditions": after_conditions,
        },
        "meets_acceptance": all(before_conditions.values()) and all(after_conditions.values()),
    }
