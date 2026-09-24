"""Fixed-model, fixed-size independent validation of a frozen executable plan.

This is simulator validation, not historical calibration. No holdout observation
is exposed to selection. Ordinary MC only; no sequential precision controller.
"""

from dataclasses import asdict, dataclass
from time import perf_counter

from scipy.stats import binomtest

from ginseng.execution import EvaluationContext, ExecutionConfig, prepare_scenario
from ginseng.optimizer import OptimalPlan, optimize_funding
from ginseng.provenance import digest
from ginseng.sampling import derive_seed, prepare_history, sample_bundle
from ginseng.simulate import discretionary_resampled_paths
from ginseng.verification import (
    ExecutablePlan,
    RiskContract,
    controls_from_state,
    executable_from_optimal,
    verify_plan,
)

TRAIN_DOMAIN = 710
VALIDATION_DOMAIN = 711


def stream_identity(bundle):
    return dict(
        dict(bundle.sampling_metadata),
        sampler=bundle.sampler,
        paths=bundle.n_paths,
        draw_id=bundle.bootstrap_draw_id,
        mean_block_length=bundle.mean_block_length,
        history_length=bundle.history_length,
    )


def independent_streams(training, validation):
    """Validate generator namespace and seed derivation, not display labels.

    SeedSequence domain separation gives independently initialized MC streams;
    it does not promise that sampled histories or resulting cash values differ.
    """
    for s in (training, validation):
        if s.get("sampler") != "mc" or s.get("sampler_version") != 1:
            raise ValueError("Independent validation requires ordinary MC version 1")
        if s["derived_seed"] != derive_seed(
            s["root_seed"], "mc", s["replicate"], s["domain"]
        ):
            raise ValueError("Invalid stream derivation")
    keys = lambda s: (s["root_seed"], s["domain"], s["replicate"])
    if (
        keys(training) == keys(validation)
        or training["derived_seed"] == validation["derived_seed"]
    ):
        raise ValueError("Training and validation streams overlap")
    return dict(
        method="SeedSequence namespace separation / PCG64",
        training_key=keys(training),
        validation_key=keys(validation),
        status="disjoint_initializations",
        limitation="PRNG independence assumption; repeated sampled histories are legitimate.",
    )


def fixed_interval(probability, n):
    successes = round(probability * n)
    if abs(successes / n - probability) > 1e-10:
        raise ValueError("Only uniform binary path indicators support this interval")
    ci = binomtest(successes, n).proportion_ci(confidence_level=0.95, method="exact")
    return dict(
        method="Clopper-Pearson fixed-sample exact binomial",
        confidence=0.95,
        low=float(ci.low),
        high=float(ci.high),
        failures=successes,
        paths=n,
        scope="Marginal per predeclared plan; not simultaneous across comparisons; no early stopping",
    )


def contract_for_plan(state, plan, parameters):
    policy = parameters.get("funding_policy")
    return RiskContract(
        parameters.get("operating_buffer", state.operating_buffer),
        parameters.get("coverage_target", state.coverage_target),
        mean_buffer_allowance=plan.buffer_tolerance_dollar_days,
        tail_deficit_limit=plan.tail_deficit_limit,
        signed_margin_coverage=plan.buffer_coverage_target,
        max_cash_failure_probability=policy.max_cash_shortfall_probability
        if policy
        else 0.05,
        max_buffer_breach_probability=policy.max_buffer_breach_probability
        if policy
        else None,
        max_credit_utilization=policy.max_credit_utilization if policy else 1.0,
        overdraft_apr=parameters.get("overdraft_apr", 0.2999),
        objective=plan.objective_kind,
    )


def evaluate_frozen(state, bundle, obligations, executable, contract, *, weights=None):
    prepared = prepare_scenario(state, bundle, obligations)
    discretionary = discretionary_resampled_paths(state, bundle)
    return verify_plan(
        prepared,
        controls_from_state(state),
        executable,
        contract,
        weights,
        discretionary,
    )


@dataclass
class LabRun:
    report: dict
    prepared: dict
    discretionary: dict
    controls: dict
    plan: ExecutablePlan
    contract: RiskContract
    config: ExecutionConfig


def run_lab(
    state,
    obligations=(),
    *,
    paths=2000,
    validation_paths=4000,
    horizon=30,
    root_seed=20260920,
    replications=3,
    config=ExecutionConfig(),
    parameters=None,
):
    if type(replications) is not int or not 1 <= replications <= 10:
        raise ValueError("Use 1 to 10 predeclared training replications")
    from ginseng.funding import (
        FundingConfig,
        _next_charge_payment_offset,
        settlement_forecast_day,
    )

    params = dict(
        coverage_target=state.coverage_target, operating_buffer=state.operating_buffer
    )
    params.update(parameters or {})
    fc = params.get("funding_config") or FundingConfig()
    material = max(
        [
            horizon,
            settlement_forecast_day(
                state.as_of,
                fc.settlement_days,
                fc.external_transfer_days,
                use_business_days=fc.use_business_days,
            ),
        ]
        + [
            _next_charge_payment_offset(
                state.as_of, c, one_indexed=fc.use_business_days
            )
            for c in state.credit_accounts
        ]
    )
    if material > horizon:
        material += fc.trailing_days
    params.update(evaluation_horizon_days=material, decision_horizon_days=horizon)
    timings = {}
    start = perf_counter()
    t = start
    history = prepare_history(state, 14)
    timings["historical_preparation"] = perf_counter() - t
    t = perf_counter()
    train = sample_bundle(
        history, material, paths, root_seed, "mc", material, domain=TRAIN_DOMAIN
    )
    timings["training_draw_generation"] = perf_counter() - t
    t = perf_counter()
    prepared = prepare_scenario(state, train, obligations, history.joint)
    discretionary = discretionary_resampled_paths(state, train)
    timings["scenario_preparation"] = perf_counter() - t
    with EvaluationContext(config) as context:
        t = perf_counter()
        context.evaluate(
            prepared, state.immediate_funding, state.operating_buffer, full=False
        )
        timings["path_summaries_cold"] = perf_counter() - t
        t = perf_counter()
        context.evaluate(
            prepared, state.immediate_funding, state.operating_buffer, full=False
        )
        timings["path_summaries_reused"] = perf_counter() - t
        t = perf_counter()
        selected = optimize_funding(state, train, obligations, **params)
        timings["optimization_including_gate"] = perf_counter() - t
        if not isinstance(selected, OptimalPlan):
            raise ValueError(
                f"Training selection unavailable: {selected.reason}; {selected.verification}"
            )
        plan = executable_from_optimal(selected, params.get("capital_gains_rate", 0.15))
        contract = contract_for_plan(state, selected, params)
        frozen = digest(asdict(plan))
        t = perf_counter()
        training = verify_plan(
            prepared,
            controls_from_state(state),
            plan,
            contract,
            discretionary=discretionary,
        )
        timings["independent_verification"] = perf_counter() - t
        if training.status != "verified":
            raise ValueError(
                "Selected plan failed verification: " + str(training.reason)
            )
        # Predeclared replications diagnose action instability; none replace selected replicate 0.
        t = perf_counter()
        stability = []
        replicate_prepared = {}
        replicate_discretionary = {}
        for r in range(replications):
            b = (
                train
                if r == 0
                else sample_bundle(
                    history,
                    material,
                    paths,
                    root_seed,
                    "mc",
                    material,
                    replicate=r,
                    domain=TRAIN_DOMAIN,
                )
            )
            p = (
                selected
                if r == 0
                else optimize_funding(state, b, obligations, **params)
            )
            stability.append(
                dict(
                    replicate=r,
                    stream=stream_identity(b),
                    status="selected" if r == 0 else "diagnostic_only",
                    objective=p.cvar_cost
                    if isinstance(p, OptimalPlan) and p.objective_kind == "cvar"
                    else p.expected_cost
                    if isinstance(p, OptimalPlan)
                    else None,
                    actions=asdict(
                        executable_from_optimal(
                            p, params.get("capital_gains_rate", 0.15)
                        )
                    )
                    if isinstance(p, OptimalPlan)
                    else None,
                    failure=None if isinstance(p, OptimalPlan) else p.reason,
                )
            )
            if r > 0 and isinstance(p, OptimalPlan):
                name = f"training_{r}"
                rp = prepare_scenario(state, b, obligations, history.joint)
                rd = discretionary_resampled_paths(state, b)
                replicate_prepared[name] = rp
                replicate_discretionary[name] = rd
                verification = verify_plan(
                    rp,
                    controls_from_state(state),
                    executable_from_optimal(p, params.get("capital_gains_rate", 0.15)),
                    contract,
                    discretionary=rd,
                )
                stability[-1]["prepared_reference"] = name
                stability[-1]["verification"] = asdict(verification)
        timings["training_replications"] = perf_counter() - t
        # Validation is first generated after all solves finish.
        t = perf_counter()
        validation = sample_bundle(
            history,
            material,
            validation_paths,
            root_seed,
            "mc",
            material,
            domain=VALIDATION_DOMAIN,
        )
        separation = independent_streams(
            stream_identity(train), stream_identity(validation)
        )
        timings["validation_draw_generation"] = perf_counter() - t
        t = perf_counter()
        vp = prepare_scenario(state, validation, obligations, history.joint)
        vd = discretionary_resampled_paths(state, validation)
        validation_result = verify_plan(
            vp, controls_from_state(state), plan, contract, discretionary=vd
        )
        baseline = verify_plan(
            vp,
            controls_from_state(state),
            ExecutablePlan(spending_days=horizon),
            contract,
            discretionary=vd,
        )
        timings["holdout_preparation_and_evaluation"] = perf_counter() - t
        assert digest(asdict(plan)) == frozen
        counters = dict(context.counters)
    successful = [r for r in stability if r["actions"] is not None]

    def span(values):
        return max(values) - min(values) if values else None

    cost_span = span([r["objective"] for r in successful])
    withdrawal_span = span(
        [sum(a[1] for a in r["actions"]["withdrawals"]) for r in successful]
    )
    stability_summary = dict(
        successful_replications=len(successful),
        objective_range_dollars=cost_span,
        credit_range_dollars=span([r["actions"]["credit_draw"] for r in successful]),
        withdrawal_range_dollars=withdrawal_span,
        spending_fraction_range=span(
            [r["actions"]["spending_fraction"] for r in successful]
        ),
        interpretation="Different actions with essentially equal costs are consistent with a flat/nonunique empirical optimum; no plan was selected using validation."
        if cost_span is not None and cost_span <= 0.001 and withdrawal_span > 0.01
        else "Training-sample sensitivity only; neither an error diagnosis nor a population-optimality certificate.",
    )
    report = dict(
        schema=1,
        fixture="caller_supplied",
        synthetic=False,
        scope="Validation against the frozen simulator; not real-world forecast calibration, formal verification, or population optimality.",
        selection="Training replicate 0 only. Validation never selects a winner; no refitting or pathwise reoptimization.",
        comparison="Selected plan and predeclared no-action plan share all validation futures (paired common random numbers).",
        plan=asdict(plan),
        contract=asdict(contract),
        definitions=contract.definitions(),
        training=asdict(training),
        validation=asdict(validation_result),
        validation_interval=fixed_interval(
            validation_result.metrics["cash_failure_probability"], validation_paths
        ),
        no_action_validation=asdict(baseline),
        no_action_interval=fixed_interval(
            baseline.metrics["cash_failure_probability"], validation_paths
        ),
        streams=dict(
            training=stream_identity(train),
            validation=stream_identity(validation),
            separation=separation,
        ),
        solver=selected.solver_evidence,
        stability=stability,
        stability_summary=stability_summary,
        equivalence="Replay compares frozen plan execution, objective and feasibility with tolerances. Nonunique optimal action coefficients need not match a re-solve.",
        dimensions=dict(
            training_paths=paths,
            validation_paths=validation_paths,
            visible_horizon=horizon,
            material_horizon=material,
            optimizer_path_day_cap=200000,
        ),
        timings_seconds=timings,
        context_counters=counters,
    )
    timings["end_to_end_without_artifact_io"] = perf_counter() - start
    return LabRun(
        report,
        dict(training=prepared, validation=vp, **replicate_prepared),
        dict(training=discretionary, validation=vd, **replicate_discretionary),
        controls_from_state(state),
        plan,
        contract,
        config,
    )
