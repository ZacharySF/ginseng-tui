"""Behavioral contracts for canonical personal forecast evaluation."""

from __future__ import annotations

from datetime import date, timedelta
from math import isclose
from uuid import UUID

import numpy as np

from ginseng.finance_models import (
    EventRule,
    EventSettlement,
    FinanceInputs,
    FinanceWorkspace,
    HistoricalTransaction,
    ModelAssumptions,
    PersonalCreditAccount,
    PersonalHolding,
    PersonalTaxLot,
)
from ginseng.metrics import compute_scenario_metrics
from ginseng.personal_forecast import (
    DAYS_PER_MONTH,
    MAX_PERSONAL_EVALUATION_HORIZON,
    _assumption_bundle,
    _schedule_for_workspace,
    _state_for_workspace,
    _with_horizon,
    backtest_personal_history,
    evaluate_personal_forecast,
)
from ginseng.simulate import cash_paths, draw_bundle
from ginseng.uncertainty import SensitivityRow, stability_verdict
from ginseng.workspace import CashAccount, CashBill

_ACCOUNT_ID = UUID("00000000-0000-0000-0000-000000000101")
_RENT_ID = UUID("00000000-0000-0000-0000-000000000102")
_INCOME_ID = UUID("00000000-0000-0000-0000-000000000103")


def scheduled_workspace(*, inputs: FinanceInputs | None = None) -> FinanceWorkspace:
    as_of = date(2024, 2, 28)
    return FinanceWorkspace(
        revision=7,
        as_of=as_of,
        currency="USD",
        accounts=[
            CashAccount(
                id=_ACCOUNT_ID,
                name="Checking",
                kind="checking",
                balance_cents=500_000,
            )
        ],
        bills=[CashBill(id=_RENT_ID, label="Rent", amount_cents=10_000, due_date=as_of)],
        inputs=inputs
        or FinanceInputs(
            mode="scheduled",
            income_events=[
                CashBill(
                    id=_INCOME_ID,
                    label="Month-end pay",
                    amount_cents=200_000,
                    due_date=date(2024, 1, 31),
                )
            ],
            event_rules=[EventRule(event_id=_INCOME_ID, recurrence="monthly", end_date=None)],
        ),
    )


def assumptions_workspace(
    assumptions: ModelAssumptions,
    *,
    balance_cents: int = 500_000,
    holdings: list[PersonalHolding] | None = None,
) -> FinanceWorkspace:
    as_of = date(2024, 2, 28)
    return FinanceWorkspace(
        revision=7,
        as_of=as_of,
        currency="USD",
        accounts=[
            CashAccount(
                id=_ACCOUNT_ID,
                name="Checking",
                kind="checking",
                balance_cents=balance_cents,
            )
        ],
        bills=[],
        inputs=FinanceInputs(
            mode="assumptions",
            assumptions=assumptions,
            holdings=holdings or [],
        ),
    )


def assumption_state(workspace: FinanceWorkspace, horizon_days: int = 14):
    schedule = _schedule_for_workspace(workspace, MAX_PERSONAL_EVALUATION_HORIZON)
    return _with_horizon(_state_for_workspace(workspace, schedule, history=False), horizon_days)


def test_scheduled_mode_uses_opening_day_and_month_end_occurrences() -> None:
    run = evaluate_personal_forecast(scheduled_workspace(), 14, paths=1)

    assert run.status == "ready"
    assert run.result is not None
    assert run.result.cash_paths.p10 == []
    assert run.result.cash_paths.p90 == []
    # Opening cash is measured before Feb 28 activity. Rent lands on day one,
    # and the Jan-31 monthly anchor clips to leap-day Feb 29.
    assert run.result.cash_paths.p50[:2] == [4_900.0, 6_900.0]
    assert run.result.cash_paths.known_income[1] == 2_000.0


def test_cent_sized_recurring_flows_cross_the_latest_opening_date_boundary() -> None:
    workspace = FinanceWorkspace(
        revision=9_007_199_254_740_991,
        as_of=date(2100, 12, 31),
        currency="USD",
        accounts=[CashAccount(id=_ACCOUNT_ID, name="Checking", kind="checking", balance_cents=-1)],
        bills=[CashBill(id=_RENT_ID, label="Monthly fee", amount_cents=1, due_date=date(2100, 12, 31))],
        inputs=FinanceInputs(event_rules=[EventRule(event_id=_RENT_ID, recurrence="monthly", end_date=None)]),
    )
    run = evaluate_personal_forecast(workspace, 60, paths=1)

    assert run.status == "ready"
    assert run.result is not None
    np.testing.assert_allclose(
        run.result.cash_paths.p50,
        [-0.02] * 31 + [-0.03] * 28 + [-0.04],
        rtol=0,
        atol=1e-12,
    )


def test_preopening_settlement_lands_once_on_its_saved_settlement_date() -> None:
    as_of = date(2024, 2, 28)
    settled_bill = CashBill(
        id=_RENT_ID,
        label="Settled utility bill",
        amount_cents=30_000,
        due_date=date(2024, 2, 20),
    )
    workspace = FinanceWorkspace(
        revision=7,
        as_of=as_of,
        currency="USD",
        accounts=[CashAccount(id=_ACCOUNT_ID, name="Checking", kind="checking", balance_cents=500_000)],
        bills=[settled_bill],
        inputs=FinanceInputs(
            event_rules=[
                EventRule(
                    event_id=_RENT_ID,
                    recurrence="none",
                    end_date=None,
                    settlements=[
                        EventSettlement(
                            due_date=date(2024, 2, 20),
                            status="settled",
                            settled_on=as_of,
                        )
                    ],
                )
            ]
        ),
    )

    run = evaluate_personal_forecast(workspace, 14, paths=1)
    assert run.status == "ready"
    assert run.result is not None
    assert run.result.cash_paths.p50[0] == 4_700.0
    assert run.result.cash_paths.p50[1] == 4_700.0


def test_current_credit_balance_adds_successive_saved_minimum_payments() -> None:
    inputs = FinanceInputs(
        mode="scheduled",
        credit_accounts=[
            PersonalCreditAccount(
                id=UUID("00000000-0000-0000-0000-000000000104"),
                name="Card",
                credit_limit_cents=100_000,
                current_balance_cents=30_000,
                purchase_apr=0.2,
                statement_close_day=20,
                payment_due_day=1,
                grace_period_eligible=False,
                minimum_payment_cents=10_000,
            )
        ],
    )
    run = evaluate_personal_forecast(scheduled_workspace(inputs=inputs), 60, paths=1)

    assert run.status == "ready"
    assert run.result is not None
    # Feb 28 opening, then Mar 1 and Apr 1 minimum payments. The known
    # current card balance is not silently dropped after its first due date.
    assert run.result.cash_paths.p50[2] == 4_800.0
    assert run.result.cash_paths.p50[33] == 4_700.0


def test_unresolved_preopening_one_time_bill_requires_reconciliation() -> None:
    workspace = scheduled_workspace()
    overdue = CashBill(
        id=_RENT_ID,
        label="Overdue rent",
        amount_cents=10_000,
        due_date=date(2024, 2, 27),
    )
    workspace = workspace.model_copy(update={"bills": [overdue]})

    run = evaluate_personal_forecast(workspace, 14, paths=1)

    assert run.status == "needs-input"
    assert run.result is None
    assert {requirement.code for requirement in run.requirements} == {"resolve-preopening-bills"}


def test_assumption_mode_generates_reproducible_direct_uncertainty_without_history() -> None:
    inputs = FinanceInputs(
        mode="assumptions",
        assumptions=ModelAssumptions(
            monthly_variable_income_cents=350_000,
            monthly_essential_spending_cents=125_000,
            monthly_discretionary_spending_cents=50_000,
            income_variability_pct=0.25,
            spending_variability_pct=0.15,
            persistence_days=10,
            income_spending_correlation=-0.2,
        ),
    )
    workspace = scheduled_workspace(inputs=inputs)

    first = evaluate_personal_forecast(workspace, 14, seed=91, paths=8)
    second = evaluate_personal_forecast(workspace, 14, seed=91, paths=8)

    assert first.status == second.status == "ready"
    assert first.result is not None and second.result is not None
    assert first.result.bootstrap_draw_id == second.result.bootstrap_draw_id
    assert first.result.cash_paths.p10 != []
    assert first.result.cash_paths.p10 == second.result.cash_paths.p10


def historical_workspace(days: int = 104) -> FinanceWorkspace:
    as_of = date(2025, 6, 1)
    history_start = as_of - timedelta(days=days)
    history_end = as_of - timedelta(days=1)
    records = []
    for offset in range(days):
        current = history_start + timedelta(days=offset)
        income = offset % 7 == 0
        records.append(
            HistoricalTransaction(
                id=UUID(int=10_000 + offset),
                date=current,
                description=f"Recorded activity {offset}",
                amount_cents=50_000 if income else -8_000,
                category="income_variable" if income else "expense_essential_variable",
                source_key=None,
            )
        )
    return FinanceWorkspace(
        revision=8,
        as_of=as_of,
        currency="USD",
        accounts=[CashAccount(id=_ACCOUNT_ID, name="Checking", kind="checking", balance_cents=500_000)],
        bills=[],
        inputs=FinanceInputs(
            mode="history",
            transactions=records,
            history_start=history_start,
            history_end=history_end,
            history_complete=True,
        ),
    )


def test_backtest_uses_only_training_history_and_bounded_origins() -> None:
    workspace = historical_workspace()

    result = backtest_personal_history(workspace, 14, paths=20)

    assert 1 <= result.periods <= 8
    assert all(window.end_date <= workspace.inputs.history_end for window in result.windows)
    assert result.warning is not None


def test_assumption_persistence_sensitivity_reuses_direct_paths_for_actual_metrics() -> None:
    assumptions = ModelAssumptions(
        monthly_variable_income_cents=300_000,
        monthly_essential_spending_cents=400_000,
        monthly_discretionary_spending_cents=100_000,
        income_variability_pct=0.75,
        spending_variability_pct=0.4,
        persistence_days=10,
        income_spending_correlation=-0.35,
    )
    workspace = assumptions_workspace(assumptions, balance_cents=100)
    state = assumption_state(workspace)

    run = evaluate_personal_forecast(workspace, 14, seed=113, paths=64)

    assert run.result is not None
    rows = {int(row["persistence_days"]): row for row in run.result.sensitivity}
    assert [row["block_label"] for row in run.result.sensitivity] == ["7d", "14d", "28d", "Current 10d"]
    assert set(rows) == {7, 10, 14, 28}
    expected_verdict_rows: list[SensitivityRow] = []
    for persistence_days, row in rows.items():
        stressed_assumptions = assumptions.model_copy(update={"persistence_days": persistence_days})
        stressed_inputs = workspace.inputs.model_copy(update={"assumptions": stressed_assumptions})
        expected = compute_scenario_metrics(
            state,
            _assumption_bundle(state, stressed_inputs, 113, 64),
            (),
            state.coverage_target,
            state.operating_buffer,
        )
        assert row["mean_block_length"] == persistence_days
        assert row["is_estimated"] is False
        assert row["was_clipped"] is False
        assert isclose(
            float(row["required_liquidity_reserve"]),
            expected.required_liquidity_reserve,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        assert isclose(float(row["funding_gap"]), expected.funding_gap, rel_tol=0.0, abs_tol=1e-12)
        assert isclose(
            float(row["cash_shortfall_probability"]),
            expected.severity["cash_shortfall_probability"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        expected_verdict_rows.append(
            SensitivityRow(
                block_label=str(row["block_label"]),
                mean_block_length=persistence_days,
                required_liquidity_reserve=expected.required_liquidity_reserve,
                is_estimated=False,
                was_clipped=False,
            )
        )


    assert run.result.sensitivity_verdict == stability_verdict(expected_verdict_rows)


def test_assumption_cash_paths_preserve_monthly_income_and_spending_means() -> None:
    income_assumptions = ModelAssumptions(
        monthly_variable_income_cents=425_000,
        income_variability_pct=2.0,
        persistence_days=30,
    )
    income_workspace = assumptions_workspace(income_assumptions)
    income_state = assumption_state(income_workspace)
    income_bundle = _assumption_bundle(income_state, income_workspace.inputs, seed=23, paths=4_000)

    spending_assumptions = ModelAssumptions(
        monthly_essential_spending_cents=250_000,
        monthly_discretionary_spending_cents=175_000,
        spending_variability_pct=2.0,
        persistence_days=30,
    )
    spending_workspace = assumptions_workspace(spending_assumptions)
    spending_state = assumption_state(spending_workspace)
    spending_bundle = _assumption_bundle(
        spending_state,
        spending_workspace.inputs,
        seed=23,
        paths=4_000,
    )

    sampled_income_monthly = float(np.mean(income_bundle.daily_cash_flows)) * DAYS_PER_MONTH
    sampled_essential_monthly = float(
        np.mean(-spending_bundle.daily_cash_flows - spending_bundle.discretionary_daily)
    ) * DAYS_PER_MONTH
    sampled_discretionary_monthly = float(np.mean(spending_bundle.discretionary_daily)) * DAYS_PER_MONTH
    assert isclose(sampled_income_monthly, 4_250.0, rel_tol=0.06)
    assert isclose(sampled_essential_monthly, 2_500.0, rel_tol=0.06)
    assert isclose(sampled_discretionary_monthly, 1_750.0, rel_tol=0.06)
    assert float(np.min(income_bundle.daily_cash_flows)) >= 0.0
    assert float(np.max(spending_bundle.daily_cash_flows)) <= 0.0


def test_assumption_market_paths_keep_annual_moments_across_persistence() -> None:
    holding = PersonalHolding(
        id=UUID("00000000-0000-0000-0000-000000000105"),
        symbol="VTI",
        account="taxable",
        current_price_cents=10_000,
        tax_lots=[
            PersonalTaxLot(
                id=UUID("00000000-0000-0000-0000-000000000106"),
                quantity=10.0,
                cost_basis_per_share_cents=8_000,
                purchase_date=date(2024, 1, 2),
            )
        ],
    )

    for persistence_days in (1, 28):
        assumptions = ModelAssumptions(
            persistence_days=persistence_days,
            market_assumptions_enabled=True,
            expected_annual_return_pct=0.08,
            annual_return_volatility_pct=0.20,
        )
        workspace = assumptions_workspace(assumptions, holdings=[holding])
        state = assumption_state(workspace)
        bundle = _assumption_bundle(state, workspace.inputs, seed=29, paths=4_000)

        assert bundle.portfolio_values is not None
        annual_log_returns = np.log(bundle.portfolio_values[:, 364] / state.marketable_backup_capital)
        annual_simple_returns = np.exp(annual_log_returns) - 1.0
        assert isclose(float(np.mean(annual_simple_returns)), 0.08, abs_tol=0.012)
        assert isclose(float(np.std(annual_log_returns)), 0.20, rel_tol=0.05)


def test_history_forecast_samples_only_declared_training_window_before_opening_gap() -> None:
    as_of = date(2025, 6, 1)
    history_end = as_of - timedelta(days=60)
    history_start = history_end - timedelta(days=89)
    records = [
        HistoricalTransaction(
            id=UUID(int=20_000 + offset),
            date=history_start + timedelta(days=offset),
            description=f"Observed income {offset}",
            amount_cents=10_000,
            category="income_variable",
            source_key=None,
        )
        for offset in range(90)
    ]
    workspace = FinanceWorkspace(
        revision=8,
        as_of=as_of,
        currency="USD",
        accounts=[CashAccount(id=_ACCOUNT_ID, name="Checking", kind="checking", balance_cents=500_000)],
        bills=[],
        inputs=FinanceInputs(
            mode="history",
            transactions=records,
            history_start=history_start,
            history_end=history_end,
            history_complete=True,
        ),
    )
    schedule = _schedule_for_workspace(workspace, 14)
    state = _with_horizon(_state_for_workspace(workspace, schedule, history=True), 14)

    bundle = draw_bundle(state, horizon_days=14, n_paths=20, seed=47, mean_block_length=7)
    sampled_cash = cash_paths(state, bundle)

    np.testing.assert_allclose(
        sampled_cash,
        np.broadcast_to(np.arange(1, 15, dtype=float) * 100.0, sampled_cash.shape),
    )
