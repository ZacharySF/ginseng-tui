"""On-demand decision analysis with a shared budget and frozen-plan holdout."""

from dataclasses import asdict, replace
from time import monotonic
from types import SimpleNamespace

import numpy as np

from ginseng.funding import PlanKind, PlanSpec, evaluate_plan_paths
from ginseng.optimizer import OptimalPlan, OptimizationFailure, optimize_funding
from ginseng.risk import cvar, quantile, weight_hash, balance_risk
from ginseng.policy import policy_assessment, FundingPolicy
from ginseng.sampling import prepare_history, sample_bundle
from ginseng.decision_lab import contract_for_plan, evaluate_frozen, fixed_interval, stream_identity, VALIDATION_DOMAIN
from ginseng.verification import executable_from_optimal
from ginseng.stress import scenario_weights
from ginseng.withdrawals import WithdrawalAssumptions

ANALYSIS_BUDGET_SECONDS = 40.0


def _increase_credit_capacity(account, utilization_limit, bump):
    """Increase spendable borrowing capacity, not just the nominal card limit."""
    if utilization_limit <= 0:
        return None
    current_capacity = max(0.0, min(account.available_credit,
        account.credit_limit * utilization_limit - account.current_balance))
    limit = max(account.credit_limit + bump,
                (account.current_balance + current_capacity + bump) / utilization_limit)
    return replace(account, credit_limit=limit,
        cash_advance_limit=account.cash_advance_limit + bump if account.cash_advance_limit is not None else None)


def loss_metrics(evaluation, state, q, weights, tax_rate, overdraft_apr) -> dict:
    balances = state.immediate_funding + evaluation.cash_matrix
    result = evaluation.result
    cost = (result.interest_exposure + result.withdrawal_tax_reserve + result.withdrawal_penalty_reserve
            + evaluation.spending_reduction + (overdraft_apr / 365) * np.maximum(0.0, -balances).sum(axis=1))
    worst_deficit = np.maximum(0.0, state.operating_buffer - balances.min(axis=1))
    return {
        "expected_cost": float(weights @ cost), "cvar_cost": cvar(cost, q, weights),
        "var_cost": quantile(cost, q, weights), "tail_deficit": cvar(worst_deficit, q, weights),
        **balance_risk(balances, state.operating_buffer, q, weights),
    }


def frozen_plan_evaluation(state, bundle, obligations, plan, weights, q, tax_rate, overdraft_apr, decision_horizon_days=None):
    spec = PlanSpec("optimized", "Optimized", PlanKind.HYBRID,
                    credit_account_id=plan.credit_account_id,
                    credit_draw=plan.credit_draw, liquidation_target=plan.liquidation_amount,
                    settlement_days=plan.settlement_days, external_transfer_days=plan.external_transfer_days,
                    use_business_days=plan.use_business_days,
                    withdrawal_allocations=plan.withdrawal_allocations,
                    withdrawal_assumptions=WithdrawalAssumptions(long_term_rate=tax_rate),
                    discretionary_reduction_fraction=plan.deferral_fraction)
    evaluated = evaluate_plan_paths(state, bundle, obligations, spec, weights,
                                    decision_horizon_days=decision_horizon_days or plan.decision_horizon_days)
    return loss_metrics(evaluated, state, q, weights, tax_rate, overdraft_apr)


def funding_analysis(state, bundle, obligations, specs, weights, view, parameters) -> dict:
    deadline = monotonic() + ANALYSIS_BUDGET_SECONDS

    def solve(current_state=state, **overrides):
        remaining = deadline - monotonic()
        if remaining <= 0.1:
            return OptimizationFailure("solver_timeout")
        return optimize_funding(current_state, bundle, obligations, weights=weights,
                                **{**parameters, **overrides}, time_limit_seconds=min(10.0, remaining))

    q = parameters["coverage_target"]
    base = solve()
    policy = parameters.get("funding_policy", FundingPolicy(max_credit_utilization=1.0))
    if isinstance(base, OptimalPlan):
        policy = replace(policy, buffer_tolerance_dollar_days=base.buffer_tolerance_dollar_days,
                         tail_deficit_limit=base.tail_deficit_limit)
    anchors = []
    for spec in specs:
        evaluation = evaluate_plan_paths(state, bundle, obligations, spec, weights,
                                         decision_horizon_days=parameters.get("decision_horizon_days"))
        anchors.append({"id": spec.id, "label": spec.label, "feasible": evaluation.result.feasible,
                        **policy_assessment(evaluation.result, policy),
                        **loss_metrics(evaluation, state, q, weights, parameters["capital_gains_rate"], parameters["overdraft_apr"])})
    result = {
        "status": "ready", "evaluation_horizon_days": bundle.horizon_days,
        "evaluation_draw_id": bundle.bootstrap_draw_id, "evaluation_weight_hash": weight_hash(weights),
        "paths": bundle.n_paths, "anchors": anchors, "frontier": [], "shadow_checks": [], "holdout": None,
        "loss_definition": "Interest + account-specific tax and early-withdrawal penalty reserves + discretionary spending forgone + overdraft dollar-days priced at the assumed APR. Withdrawals add only net spendable cash; principal is not a cost.",
        "risk_definition": "Tail buffer deficit averages each path's largest dollar deficit in the worst (1-q) fraction of deficits. This need not select the same futures as the cost tail. At q=1 it is the largest modeled deficit.",
        "budget_seconds": ANALYSIS_BUDGET_SECONDS,
    }
    if not isinstance(base, OptimalPlan):
        return {**result, "status": "unavailable", "reason": base.reason}
    result["base"] = asdict(base)
    # Fresh random draws from the same historical model. Controls are frozen,
    # and this sample is never used to tune the solution or pick a frontier point.
    history = prepare_history(state, bundle.mean_block_length)
    holdout = sample_bundle(history, bundle.horizon_days, bundle.n_paths, bundle.seed,
                            'mc', bundle.horizon_days, domain=VALIDATION_DOMAIN)
    if holdout.seed == bundle.seed:
        return {**result, 'status': 'unavailable', 'reason': 'stream_identity_collision'}
    holdout_weights, holdout_stress = scenario_weights(state, holdout, view)
    if holdout_stress["status"] == "unsupported":
        result["holdout"] = {"status": "unavailable", "message": "Fresh scenarios do not support the active stress assumption."}
    else:
        observed = frozen_plan_evaluation(state, holdout, obligations, base, holdout_weights, q,
                                         parameters["capital_gains_rate"], parameters["overdraft_apr"], parameters.get("decision_horizon_days"))
        result["holdout"] = {"status": "ready", "evaluation_draw_id": holdout.bootstrap_draw_id,
            "evaluation_weight_hash": weight_hash(holdout_weights), "paths": holdout.n_paths,
            "label": "Fresh simulation check, not new historical evidence. The selected actions were not re-optimized.",
            **observed,
            "within_mean_buffer_limit": observed["dollar_days_below_buffer"] <= base.buffer_tolerance_dollar_days + 1e-6,
            "within_tail_deficit_limit": observed["tail_deficit"] <= base.tail_deficit_limit + 1e-6 if base.tail_deficit_limit is not None else None}
        verified = evaluate_frozen(state, holdout, obligations,
            executable_from_optimal(base, parameters['capital_gains_rate']),
            contract_for_plan(state, base, parameters), weights=holdout_weights)
        uniform = np.array_equal(holdout_weights, np.full(holdout.n_paths, 1 / holdout.n_paths))
        result['holdout'].update(verification=asdict(verified),
            interval=fixed_interval(verified.metrics['cash_failure_probability'],holdout.n_paths) if uniform and holdout_stress['status'] != 'active' else None,
            interval_status='fixed_sample_iid_mc' if uniform and holdout_stress['status'] != 'active' else 'unavailable_for_weighted_stress',
            stream=stream_identity(holdout), training_stream=dict(seed=bundle.seed,sampler=bundle.sampler,metadata=dict(bundle.sampling_metadata)),
            independence='Distinct PCG64 initialization via SeedSequence domain 711; holdout was not used in selection.',
            frozen_plan_identity=verified.identity.plan)
        checked_policy = replace(policy,
                                 buffer_tolerance_dollar_days=base.buffer_tolerance_dollar_days,
                                 tail_deficit_limit=base.tail_deficit_limit)
        result["holdout"].update(policy_assessment(SimpleNamespace(**observed,
            credit_draw=base.credit_draw, credit_utilization=base.credit_utilization), checked_policy))

    # Verify marginal values near the current solution, never extrapolate
    # them over the entire credit limit or the entire deficit allowance.
    account = next((account for account in state.credit_accounts
                    if account.account_id == base.credit_account_id), None)
    for resource in ("credit_capacity", "mean_buffer_allowance"):
        dual = base.implied_credit_price if resource == "credit_capacity" else base.implied_liquidity_price
        checks = []
        if resource == "credit_capacity" and account is None:
            continue
        for bump in (1.0, 10.0, 100.0):
            if resource == "credit_capacity":
                increased = _increase_credit_capacity(account, policy.max_credit_utilization, bump)
                if increased is None:
                    checks.append({"bump": bump, "value_per_unit": None, "status": "credit_disabled_by_policy"})
                    continue
                nudged = replace(state, credit_accounts=tuple(increased
                    if a.account_id == account.account_id else a for a in state.credit_accounts))
                solution = solve(nudged)
            else:
                solution = solve(buffer_tolerance_dollar_days=base.buffer_tolerance_dollar_days + bump)
            checks.append({"bump": bump, "value_per_unit": (base.cvar_cost - solution.cvar_cost) / bump if isinstance(solution, OptimalPlan) else None,
                           "status": solution.solver_status if isinstance(solution, OptimalPlan) else solution.reason})
        values = [row["value_per_unit"] for row in checks if row["value_per_unit"] is not None]
        result["shadow_checks"].append({"resource": resource, "solver_dual": dual, "checks": checks,
            "range": {"low": min(values), "high": max(values)} if values else None,
            "stable": len(values) == 3 and max(values) - min(values) <= max(0.00001, 0.1 * max(abs(v) for v in values))})

    scale = max(100.0, state.operating_buffer)
    limits = sorted({0.0, scale / 4, scale / 2, scale, scale * 2, base.tail_deficit})
    for limit in limits:
        point = solve(tail_deficit_limit=limit, objective_kind="expected")
        result["frontier"].append({"limit": limit, "status": point.solver_status if isinstance(point, OptimalPlan) else point.reason,
                                   "plan": asdict(point) if isinstance(point, OptimalPlan) else None})
    result["budget_exhausted"] = monotonic() >= deadline
    return result
