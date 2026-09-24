"""Independent finite-scenario funding execution checks (not formal verification).

The reference path/action arithmetic does not call an execution backend, solver,
production funding evaluator, withdrawal quote, or production risk reducer.
Calendar interpretation is a shared validated primitive. Inputs are frozen
financial controls plus the existing immutable PreparedScenario.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from typing import Literal, TypedDict

import numpy as np

from ginseng.execution import PreparedScenario, array_id, snapshot
from ginseng.provenance import digest

VERSION = "decision-verification-v1"


class RiskMetrics(TypedDict, total=False):
    cash_failure_probability: float
    buffer_breach_probability: float
    expected_buffer_dollar_days: float
    buffer_deficit_cvar: float
    required_liquidity_reserve: float
    expected_max_cash_deficit: float
    signed_margin_cvar: float | None
    legacy_cash_failure_probability: float
    legacy_buffer_breach_probability: float
    expected_cost: float
    cvar_cost: float
    var_cost: float
    objective: float


class SolverEvidence(TypedDict, total=False):
    lower_bound: float | None
    global_lower_bound: float | None
    absolute_gap: float | None
    relative_gap: float | None
    executed_objective: float
    reported_objective: float
    iterations: int
    termination_reason: str
    scope: str


@dataclass(frozen=True)
class RiskContract:
    operating_buffer: float
    coverage_target: float
    mean_buffer_allowance: float | None = None
    tail_deficit_limit: float | None = None
    signed_margin_coverage: float | None = None
    max_cash_failure_probability: float = 0.05
    max_buffer_breach_probability: float | None = None
    max_credit_utilization: float = 1.0
    overdraft_apr: float = 0.2999
    objective: Literal["cvar", "expected"] = "cvar"
    money_tolerance: float = 1e-6
    probability_tolerance: float = 1e-12
    objective_tolerance: float = 0.001
    policy_boundary_tolerance: float = 1e-6

    def __post_init__(self):
        for k, v in asdict(self).items():
            if k == "objective" or v is None:
                continue
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or v < 0
            ):
                raise ValueError(f"Invalid risk contract field {k}")
        for k in (
            "coverage_target",
            "signed_margin_coverage",
            "max_cash_failure_probability",
            "max_buffer_breach_probability",
            "max_credit_utilization",
        ):
            if getattr(self, k) is not None and getattr(self, k) > 1:
                raise ValueError(f"{k} must be in [0,1]")
        if self.objective not in ("cvar", "expected"):
            raise ValueError("Unknown objective")

    def definitions(self):
        return dict(
            cash_failure="P(min end-of-day cash < 0); opening cash added exactly once",
            buffer_breach="P(min end-of-day cash < operating_buffer); equality is not breach",
            dollar_days="E[sum_day max(buffer-cash,0)], dollars × days",
            tail_deficit="CVaR_q(max_day max(buffer-cash,0)), dollars",
            reserve="inverse empirical CDF of max(0,buffer-min cumulative funded flows excluding opening cash)",
            objective="CVaR or mean of interest + fees + earmarked taxes/penalties + spending forgone + overdraft dollar-days × APR/365",
            policy_boundary=f"Legacy app policy uses cash < -{self.policy_boundary_tolerance:g} and cash < buffer-{self.policy_boundary_tolerance:g}; strict metrics are reported separately",
            quantile="inverse CDF, float64-rounded long-double cumulative mass; q=1 maximum; tail ties share boundary mass proportionally",
            scope="Finite supplied scenarios, fixed prices/rates, continuous withdrawal units, selected credit account. Not a population optimum or real-world safety claim.",
        )


@dataclass(frozen=True)
class ExecutablePlan:
    credit_account_id: str | None = None
    credit_draw: float = 0.0
    withdrawals: tuple[tuple[str, float], ...] = ()
    spending_fraction: float = 0.0
    spending_days: int = 30
    settlement_days: int = 1
    external_transfer_days: int = 2
    use_business_days: bool = False
    pay_in_full: bool = True
    long_term_rate: float = 0.15
    ordinary_rate: float = 0.24
    early_penalty: float = 0.10
    unfunded_cash: float = 0.0


@dataclass(frozen=True)
class ConstraintCheck:
    name: str
    units: str
    measured: float | None
    bound: float | None
    direction: str
    slack: float | None
    violation: float | None
    tolerance: float
    status: str


def check(name, units, measured, bound, direction="<=", tolerance=1e-6):
    if measured is None or bound is None or not np.isfinite([measured, bound]).all():
        return ConstraintCheck(
            name,
            units,
            None if measured is None else float(measured),
            bound,
            direction,
            None,
            None,
            tolerance,
            "unavailable",
        )
    slack = (
        bound - measured
        if direction == "<="
        else measured - bound
        if direction == ">="
        else -abs(measured - bound)
    )
    violation = max(0.0, -slack)
    return ConstraintCheck(
        name,
        units,
        float(measured),
        float(bound),
        direction,
        float(slack),
        float(violation),
        tolerance,
        "pass" if violation <= tolerance else "fail",
    )


@dataclass(frozen=True)
class EvaluationIdentity:
    prepared_inputs: dict
    financial_controls: str
    plan: str
    contract: str
    weights: str
    discretionary: str
    calculation: str = VERSION


@dataclass(frozen=True)
class PlanVerification:
    status: str
    reason: str | None
    constraints: tuple[ConstraintCheck, ...]
    policy_checks: tuple[ConstraintCheck, ...]
    policy_status: str
    metrics: RiskMetrics
    identity: EvaluationIdentity
    execution: dict
    definitions: dict


def controls_from_state(state):
    """Capture financial inputs, not computed withdrawal quotes/solver bounds."""
    return json.loads(
        json.dumps(
            dict(
                as_of=state.as_of.isoformat(),
                opening_cash=state.immediate_funding,
                credits=[asdict(c) for c in state.credit_accounts],
                holdings=[asdict(h) for h in state.holdings],
                roth_remaining_basis=state.roth_contribution_basis,
            ),
            default=str,
            allow_nan=False,
        )
    )


def reference_paths(prepared):
    """A day loop over frozen primitive data, independent of either backend."""
    n, h = prepared.n_paths, prepared.visible_horizon
    output = np.empty((n, h))
    total = np.zeros(n)
    for day in range(h):
        if prepared.source == "history":
            rows = prepared.history[prepared.indices[:, day]]
            daily = rows[:, 0] - rows[:, 1] - rows[:, 2] + prepared.schedule[day]
        else:
            daily = prepared.direct[:, day] - prepared.schedule[day]
        total = total + daily
        output[:, day] = total
    return output


def _weights(n, weights):
    w = (
        np.full(n, 1 / n)
        if weights is None
        else np.array(weights, dtype=float, copy=True)
    )
    if (
        w.shape != (n,)
        or not np.isfinite(w).all()
        or (w < 0).any()
        or not np.isfinite(w.sum())
        or w.sum() <= 0
    ):
        raise ValueError("Invalid verification weights")
    return w / w.sum()


def reference_quantile(x, q, w):
    order = sorted((float(v), float(p)) for v, p in zip(x, w) if p > 0)
    cumulative = np.cumsum(np.array([p for _, p in order], dtype=np.longdouble))
    cumulative = (cumulative / cumulative[-1]).astype(float)
    return (
        order[-1][0]
        if q == 1
        else order[min(int(np.searchsorted(cumulative, q)), len(order) - 1)][0]
    )


def reference_cvar(x, q, w):
    cutoff = reference_quantile(x, q, w)
    if q == 1:
        return cutoff
    above = np.asarray(x) > cutoff
    # Integrate the upper tail with exactly the remaining probability at its atom.
    return float(
        (
            np.dot(w[above], np.asarray(x)[above])
            + max(0.0, 1 - q - float(w[above].sum())) * cutoff
        )
        / (1 - q)
    )


def measure_risk(
    cumulative, opening, contract, weights=None, costs=None
) -> RiskMetrics:
    x = np.asarray(cumulative, dtype=float)
    if (
        x.ndim != 2
        or min(x.shape) < 1
        or not np.isfinite(x).all()
        or not math.isfinite(opening)
    ):
        raise ValueError("Finite nonempty cumulative paths required")
    w = _weights(len(x), weights)
    b = opening + x
    low = b.min(axis=1)
    deficit = np.maximum(0, contract.operating_buffer - low)
    required = np.maximum(0, contract.operating_buffer - x.min(axis=1))
    tol = contract.policy_boundary_tolerance
    result = dict(
        cash_failure_probability=float(w[low < 0].sum()),
        buffer_breach_probability=float(w[low < contract.operating_buffer].sum()),
        expected_buffer_dollar_days=float(
            w @ np.maximum(0, contract.operating_buffer - b).sum(axis=1)
        ),
        expected_max_cash_deficit=float(w @ np.maximum(0, -low)),
        buffer_deficit_cvar=reference_cvar(deficit, contract.coverage_target, w),
        signed_margin_cvar=reference_cvar(
            contract.operating_buffer - low, contract.signed_margin_coverage, w
        )
        if contract.signed_margin_coverage is not None
        else None,
        required_liquidity_reserve=reference_quantile(
            required, contract.coverage_target, w
        ),
        legacy_cash_failure_probability=float(w[low < -tol].sum()),
        legacy_buffer_breach_probability=float(
            w[low < contract.operating_buffer - tol].sum()
        ),
    )
    if costs is not None:
        result.update(
            expected_cost=float(w @ costs),
            cvar_cost=reference_cvar(costs, contract.coverage_target, w),
            var_cost=reference_quantile(costs, contract.coverage_target, w),
        )
        result["objective"] = result[
            "cvar_cost" if contract.objective == "cvar" else "expected_cost"
        ]
    return result


def _catalog(controls, plan):
    """Independent current-price capacities and earmarked charge arithmetic."""
    units = {}
    traditional = roth = 0.0
    as_of = date.fromisoformat(controls["as_of"])
    for h in controls["holdings"]:
        price = h["current_price"]
        if not np.isfinite(price) or price <= 0:
            raise ValueError("Invalid asset price")
        for lot in h["tax_lots"]:
            quantity, basis = lot["quantity"], lot["cost_basis_per_share"]
            if min(quantity, basis) < 0 or not np.isfinite([quantity, basis]).all():
                raise ValueError("Invalid lot")
            capacity = quantity * price
            if h["account"] == "traditional":
                traditional += capacity
            if h["account"] == "roth":
                roth += capacity
            if h["account"] != "taxable" or capacity == 0:
                continue
            purchase = date.fromisoformat(lot["purchase_date"])
            try:
                anniversary = purchase.replace(year=purchase.year + 1)
            except ValueError:
                anniversary = date(purchase.year + 1, 2, 28)
            rate = plan.long_term_rate if as_of > anniversary else plan.ordinary_rate
            key = "taxable:" + lot["lot_id"]
            if key in units:
                raise ValueError("Duplicate withdrawal lot")
            units[key] = (capacity, max(0, 1 - basis / price) * rate, 0.0)
    if traditional > 0:
        units["traditional"] = (traditional, plan.ordinary_rate, plan.early_penalty)
    basis = controls["roth_remaining_basis"]
    if not np.isfinite(basis) or basis < 0:
        raise ValueError("Invalid remaining Roth basis")
    if min(roth, basis) > 0:
        units["roth"] = (min(roth, basis), 0.0, 0.0)
    return units


def verify_plan(
    prepared: PreparedScenario,
    controls: dict,
    plan: ExecutablePlan,
    contract: RiskContract,
    weights=None,
    discretionary=None,
    reported=None,
):
    from ginseng.funding import _next_charge_payment_offset, settlement_forecast_day
    from ginseng.state import CreditAccount

    n, h = prepared.n_paths, prepared.visible_horizon
    w = _weights(n, weights)
    checks = []
    policy = []

    def add(name, units, value, bound, direction="<=", tol=None):
        checks.append(
            check(
                name,
                units,
                value,
                bound,
                direction,
                contract.money_tolerance if tol is None else tol,
            )
        )

    values = [
        plan.credit_draw,
        plan.spending_fraction,
        plan.long_term_rate,
        plan.ordinary_rate,
        plan.early_penalty,
        plan.unfunded_cash,
    ]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite executable plan")
    add("credit_nonnegative", "dollars", plan.credit_draw, 0, ">=")
    add("spending_lower", "fraction", plan.spending_fraction, 0, ">=")
    add("spending_upper", "fraction", plan.spending_fraction, 1)
    add("no_double_counted_cash", "dollars", plan.unfunded_cash, 0, "==")
    for label, rate in [
        ("long_term_rate", plan.long_term_rate),
        ("ordinary_rate", plan.ordinary_rate),
        ("early_penalty", plan.early_penalty),
    ]:
        if not 0 <= rate <= 1:
            raise ValueError("Invalid withdrawal charge rate " + label)
    if plan.ordinary_rate + plan.early_penalty >= 1:
        raise ValueError("Withdrawal charges consume gross capital")
    for value in (
        plan.spending_days,
        plan.settlement_days,
        plan.external_transfer_days,
    ):
        if type(value) is not int or value < 0:
            raise ValueError("Invalid execution day")
    if discretionary is None:
        if prepared.source != "history" and plan.spending_fraction > 0:
            raise ValueError("Missing discretionary paths for spending execution")
        discretionary = (
            prepared.history[prepared.indices[:, :h], 2]
            if prepared.source == "history"
            else np.zeros((n, h))
        )
    discretionary = np.asarray(discretionary, dtype=float)
    if (
        discretionary.shape != (n, h)
        or not np.isfinite(discretionary).all()
        or (discretionary < 0).any()
    ):
        raise ValueError("Invalid discretionary paths")
    baseline = reference_paths(prepared)
    actions = np.zeros((n, h))
    spent = plan.spending_fraction * discretionary[:, : min(h, plan.spending_days)]
    actions[:, : spent.shape[1]] += spent
    amounts = {}
    for key, gross in plan.withdrawals:
        if not np.isfinite(gross):
            raise ValueError("Nonfinite withdrawal")
        add("withdrawal_entry_nonnegative:" + key, "dollars", gross, 0, ">=")
        amounts[key] = amounts.get(key, 0.0) + gross
    units = _catalog(controls, plan)
    gross = tax = penalty = 0.0
    for key, value in amounts.items():
        add("withdrawal_nonnegative:" + key, "dollars", value, 0, ">=")
        add(
            "withdrawal_capacity:" + key,
            "dollars",
            value,
            units[key][0] if key in units else 0,
        )
        if key not in units:
            add("eligible_withdrawal:" + key, "boolean", 0, 1, "==", 0)
            continue
        gross += value
        tax += value * units[key][1]
        penalty += value * units[key][2]
    net = gross - tax - penalty
    sale_day = settlement_forecast_day(
        date.fromisoformat(controls["as_of"]),
        plan.settlement_days,
        plan.external_transfer_days,
        use_business_days=plan.use_business_days,
    )
    if gross > 0:
        add("sale_available_within_material_evaluation", "day", sale_day, h, tol=0)
        if sale_day <= h:
            actions[:, sale_day - 1] += net
    account = next(
        (c for c in controls["credits"] if c["account_id"] == plan.credit_account_id),
        None,
    )
    interest = fee = repayment = 0.0
    due = None
    utilization = 0.0
    if plan.credit_draw > 0:
        add("eligible_credit_account", "boolean", int(account is not None), 1, "==", 0)
        if account:
            card = CreditAccount(**account)
            add("credit_capacity", "dollars", plan.credit_draw, card.available_credit)
            policy_capacity = max(
                0,
                card.credit_limit * contract.max_credit_utilization
                - card.current_balance,
            )
            add(
                "credit_optimization_utilization_bound",
                "dollars",
                plan.credit_draw,
                policy_capacity,
            )
            due = _next_charge_payment_offset(
                date.fromisoformat(controls["as_of"]),
                card,
                one_indexed=plan.use_business_days,
            )
            add("repayment_within_material_evaluation", "day", due, h, tol=0)
            grace = (
                card.grace_period_eligible
                and card.current_balance <= 0
                and plan.pay_in_full
            )
            interest = (
                0
                if grace
                else plan.credit_draw
                * card.purchase_apr
                / 365
                * (due - 1 if plan.use_business_days else due)
            )
            fee = plan.credit_draw * card.cash_advance_fee_pct
            repayment = (
                plan.credit_draw + interest
                if plan.pay_in_full
                else min(card.minimum_payment, plan.credit_draw + interest)
            )
            actions[:, 0] += plan.credit_draw - fee
            if 1 <= due <= h:
                actions[:, due - 1] -= repayment
            utilization = (
                (card.current_balance + plan.credit_draw) / card.credit_limit
                if card.credit_limit > 0
                else 0
            )
    # Sum actions separately to preserve the engine's cumulative-flow convention.
    cumulative = baseline + np.cumsum(actions, axis=1)
    balances = controls["opening_cash"] + cumulative
    costs = (
        interest
        + fee
        + tax
        + penalty
        + spent.sum(axis=1)
        + contract.overdraft_apr / 365 * np.maximum(0, -balances).sum(axis=1)
    )
    metrics = measure_risk(cumulative, controls["opening_cash"], contract, w, costs)
    for name, value, bound, unit in [
        (
            "mean_buffer_dollar_days",
            metrics["expected_buffer_dollar_days"],
            contract.mean_buffer_allowance,
            "dollar-days",
        ),
        (
            "tail_buffer_deficit",
            metrics["buffer_deficit_cvar"],
            contract.tail_deficit_limit,
            "dollars",
        ),
        (
            "signed_margin_cvar",
            metrics["signed_margin_cvar"],
            0 if contract.signed_margin_coverage is not None else None,
            "dollars",
        ),
    ]:
        if bound is not None:
            add(
                name,
                unit,
                value,
                bound,
                tol=contract.money_tolerance * max(1, abs(bound)),
            )
    policy.append(
        check(
            "cash_failure_probability_strict",
            "probability",
            metrics["cash_failure_probability"],
            contract.max_cash_failure_probability,
            tolerance=contract.probability_tolerance,
        )
    )
    if contract.max_buffer_breach_probability is not None:
        policy.append(
            check(
                "buffer_breach_probability_strict",
                "probability",
                metrics["buffer_breach_probability"],
                contract.max_buffer_breach_probability,
                tolerance=contract.probability_tolerance,
            )
        )
    policy.append(
        check(
            "cash_failure_probability_legacy_boundary",
            "probability",
            metrics["legacy_cash_failure_probability"],
            contract.max_cash_failure_probability,
            tolerance=contract.probability_tolerance,
        )
    )
    if contract.max_buffer_breach_probability is not None:
        policy.append(
            check(
                "buffer_breach_probability_legacy_boundary",
                "probability",
                metrics["legacy_buffer_breach_probability"],
                contract.max_buffer_breach_probability,
                tolerance=contract.probability_tolerance,
            )
        )
    policy.append(
        check(
            "credit_utilization",
            "fraction",
            utilization,
            contract.max_credit_utilization,
            tolerance=contract.probability_tolerance,
        )
    )
    execution = dict(
        credit_draw=plan.credit_draw,
        repayment_day=due,
        repayment=repayment,
        credit_interest=interest,
        credit_fee=fee,
        withdrawal_gross=gross,
        withdrawal_net=net,
        withdrawal_tax=tax,
        withdrawal_penalty=penalty,
        availability_day=sale_day if gross > 0 else None,
        spending_fraction=plan.spending_fraction,
        spending_days=plan.spending_days,
        pay_in_full=plan.pay_in_full,
    )
    if reported:
        for name, (measured, unit, tolerance) in {
            "objective": (
                metrics["objective"],
                "dollars",
                contract.objective_tolerance,
            ),
            "cash_failure_probability": (
                metrics["legacy_cash_failure_probability"],
                "probability",
                contract.probability_tolerance,
            ),
            "buffer_breach_probability": (
                metrics["legacy_buffer_breach_probability"],
                "probability",
                contract.probability_tolerance,
            ),
            "cvar_cost": (
                metrics["cvar_cost"],
                "dollars",
                contract.objective_tolerance,
            ),
            "var_cost": (metrics["var_cost"], "dollars", contract.objective_tolerance),
            "expected_cost": (
                metrics["expected_cost"],
                "dollars",
                contract.objective_tolerance,
            ),
            "mean_buffer_dollar_days": (
                metrics["expected_buffer_dollar_days"],
                "dollar-days",
                contract.money_tolerance * max(1, contract.mean_buffer_allowance or 0),
            ),
            "tail_deficit": (
                metrics["buffer_deficit_cvar"],
                "dollars",
                contract.money_tolerance * max(1, metrics["buffer_deficit_cvar"]),
            ),
            "withdrawal_net": (net, "dollars", contract.money_tolerance),
            "withdrawal_gross": (gross, "dollars", contract.money_tolerance),
            "withdrawal_tax": (tax, "dollars", contract.money_tolerance),
            "withdrawal_penalty": (penalty, "dollars", contract.money_tolerance),
        }.items():
            if name in reported:
                add("reported_" + name, unit, measured, reported[name], "==", tolerance)
        if "lower_bound" in reported:
            add(
                "lower_bound_below_executed_objective",
                "dollars",
                metrics["objective"],
                reported["lower_bound"],
                ">=",
                contract.objective_tolerance,
            )
        if "solver_objective_upper" in reported:
            add(
                "no_postprocessing_cost_increase",
                "dollars",
                metrics["objective"],
                reported["solver_objective_upper"],
                tol=max(
                    contract.objective_tolerance,
                    abs(reported["solver_objective_upper"]) * 1e-6,
                ),
            )
        if "plan_identity" in reported:
            add(
                "frozen_plan_identity",
                "boolean",
                int(digest(asdict(plan)) == reported["plan_identity"]),
                1,
                "==",
                0,
            )
        if "cash_matrix" in reported:
            a = np.asarray(reported["cash_matrix"])
            add(
                "production_cash_execution",
                "dollars",
                float(np.max(np.abs(a - cumulative)))
                if a.shape == cumulative.shape
                else None,
                0,
                "==",
                contract.money_tolerance,
            )
    failed = [c.name for c in checks if c.status != "pass"]
    identity = EvaluationIdentity(
        prepared.identities,
        digest(controls),
        digest(asdict(plan)),
        digest(asdict(contract)),
        array_id(snapshot(w)),
        array_id(snapshot(discretionary)),
    )
    return PlanVerification(
        "failed" if failed else "verified",
        ", ".join(failed) or None,
        tuple(checks),
        tuple(policy),
        "pass" if all(c.status == "pass" for c in policy) else "fail",
        metrics,
        identity,
        execution,
        contract.definitions(),
    )


def executable_from_optimal(plan, tax_rate=0.15):
    return ExecutablePlan(
        plan.credit_account_id,
        plan.credit_draw,
        tuple((x.key, x.gross) for x in plan.withdrawal_allocations),
        plan.deferral_fraction,
        plan.decision_horizon_days or plan.evaluation_horizon_days,
        plan.settlement_days,
        plan.external_transfer_days,
        plan.use_business_days,
        long_term_rate=tax_rate,
    )


def verify_optimal(
    state,
    bundle,
    obligations,
    plan,
    contract,
    weights=None,
    tax_rate=0.15,
    *,
    check_report=True,
):
    from ginseng.execution import prepare_scenario
    from ginseng.funding import extend_draw_bundle
    from ginseng.simulate import discretionary_resampled_paths

    extended = extend_draw_bundle(bundle, plan.evaluation_horizon_days)
    prepared = prepare_scenario(state, extended, obligations)
    reported = None
    if check_report:
        reported = dict(
            objective=plan.cvar_cost
            if plan.objective_kind == "cvar"
            else plan.expected_cost,
            cash_failure_probability=plan.cash_shortfall_probability,
            buffer_breach_probability=plan.buffer_breach_probability,
            cvar_cost=plan.cvar_cost,
            var_cost=plan.var_cost,
            expected_cost=plan.expected_cost,
            mean_buffer_dollar_days=plan.dollar_days_below_buffer,
            tail_deficit=plan.tail_deficit,
            withdrawal_gross=plan.liquidation_amount,
            withdrawal_net=plan.withdrawal_net_cash,
            withdrawal_tax=plan.withdrawal_tax_reserve,
            withdrawal_penalty=plan.withdrawal_penalty_reserve,
        )
        evidence = getattr(plan, "solver_evidence", None) or {}
        if evidence.get("plan_identity"):
            reported["plan_identity"] = evidence["plan_identity"]
        if evidence.get("lower_bound") is not None:
            reported["lower_bound"] = evidence["lower_bound"]
        if evidence.get("reported_objective") is not None:
            reported["solver_objective_upper"] = evidence["reported_objective"]
    return verify_plan(
        prepared,
        controls_from_state(state),
        executable_from_optimal(plan, tax_rate),
        contract,
        weights,
        discretionary_resampled_paths(state, extended),
        reported,
    )


def verify_named(
    state,
    bundle,
    obligations,
    spec,
    result,
    contract,
    weights=None,
    decision_horizon=None,
):
    from ginseng.execution import prepare_scenario
    from ginseng.funding import _liquidate
    from ginseng.simulate import discretionary_resampled_paths

    allocations = spec.withdrawal_allocations
    if allocations is None:
        allocations = (
            _liquidate(
                state.taxable_portfolio,
                spec.liquidation_target,
                spec.lot_selection,
                spec.specific_lot_ids,
            ).allocations
            if spec.liquidation_target > 0
            else ()
        )
    plan = ExecutablePlan(
        spec.credit_account_id,
        spec.credit_draw,
        tuple((a.key, a.gross) for a in allocations),
        spec.discretionary_reduction_fraction,
        spec.discretionary_reduction_days or decision_horizon or bundle.horizon_days,
        spec.settlement_days,
        spec.external_transfer_days,
        spec.use_business_days,
        spec.pay_in_full,
        spec.withdrawal_assumptions.long_term_rate,
        spec.withdrawal_assumptions.ordinary_rate,
        spec.withdrawal_assumptions.early_penalty,
        spec.unfunded_cash_amount,
    )
    return verify_plan(
        prepare_scenario(state, bundle, obligations),
        controls_from_state(state),
        plan,
        contract,
        weights,
        discretionary_resampled_paths(state, bundle),
        dict(
            mean_buffer_dollar_days=result.dollar_days_below_buffer,
            tail_deficit=result.tail_deficit,
            withdrawal_gross=result.investment_sold,
            withdrawal_net=result.withdrawal_net_cash,
            withdrawal_tax=result.withdrawal_tax_reserve,
            withdrawal_penalty=result.withdrawal_penalty_reserve,
        ),
    )
