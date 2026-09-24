"""Offline run orchestration and provenance, independent of application services."""

from dataclasses import asdict
from importlib.metadata import version
import os
import platform
import subprocess
from pathlib import Path
import numpy as np
from ginseng.provenance import digest, source_fingerprint
from ginseng.sampling import sample_bundle
from ginseng.simulate import cash_paths
from ginseng.metrics import cash_risk_summary
from ginseng.state import Obligation

TARGETS = (
    "required_liquidity_reserve",
    "cash_shortfall_probability",
    "expected_max_cash_deficit",
)


def environment():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    cpu = platform.processor()
    if not cpu and Path("/proc/cpuinfo").exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            platform.machine(),
        )
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        cpu=cpu,
        dependencies={
            k: version(k) for k in ("numpy", "pandas", "scipy", "arch", "ginseng-tui")
        },
        threads={
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        dtype="float64",
        git_commit=commit,
        source_fingerprint=source_fingerprint(),
    )


from ginseng.execution import execution_scope


@execution_scope
def run_core(
    case,
    prepared,
    method,
    paths,
    seed,
    horizon=None,
    material_horizon=None,
    replicate=0,
    domain=100,
    *,
    estimator="path",
    conditional_tables=None,
):
    if estimator not in ("path", "initial-block-cmc"):
        raise ValueError("Unknown estimator.")
    horizon = case.state.forecast_horizon if horizon is None else horizon
    bundle = sample_bundle(
        prepared, horizon, paths, seed, method, material_horizon, replicate, domain,
        trace_initial_block=estimator == "initial-block-cmc",
    )
    matrix = cash_paths(
        case.state, bundle, case.obligations, prepared_history=prepared.joint
    )
    summary = cash_risk_summary(
        matrix,
        case.state.immediate_funding,
        case.state.operating_buffer,
        case.state.coverage_target,
    )
    if estimator == "initial-block-cmc":
        from ginseng.conditional import failure_probability
        summary["cash_shortfall_probability"] = failure_probability(
            prepared, case.state, bundle, case.obligations, tables=conditional_tables
        )
    return bundle, matrix, summary


def manifest(case, prepared, bundle, summary, estimator="path"):
    inputs = digest(
        dict(
            state=asdict(case.state), obligations=[asdict(x) for x in case.obligations]
        )
    )
    model = dict(
        block_requested=prepared.requested_length,
        block_resolved=prepared.resolved_length,
        block_clipped=prepared.clipped,
        block_resolution=prepared.resolution,
        end_of_day=True,
        estimator=estimator,
        estimator_version=1,
    )
    if estimator == "initial-block-cmc":
        from ginseng.conditional import _inputs
        *_, identity = _inputs(prepared, case.state, case.obligations, bundle.horizon_days)
        model["conditional_table_identity"] = identity
        model["conditional_table_shape"] = [bundle.horizon_days, bundle.history_length]
        model["initial_block_trace_hash"] = digest(bundle.initial_block_lengths.tolist())
    env = environment()
    return dict(
        metric_estimators={key: estimator if key == "cash_shortfall_probability" else "path" for key in summary},
        fixture=case.name,
        input_hash=inputs,
        synthetic_data_seed=case.data_seed,
        opening_cash=case.state.immediate_funding,
        buffer=case.state.operating_buffer,
        coverage_target=case.state.coverage_target,
        history_length=len(prepared.joint),
        **model,
        sampler=bundle.sampler,
        **dict(bundle.sampling_metadata),
        visible_horizon=bundle.horizon_days,
        requested_n=bundle.n_paths,
        actual_n=bundle.n_paths,
        index_hash=bundle.bootstrap_draw_id,
        model_hash=digest(dict(model=model, source=env["source_fingerprint"])),
        result_hash=digest(
            dict(
                input=inputs,
                index=bundle.bootstrap_draw_id,
                model=model,
                source=env["source_fingerprint"],
                summary=summary,
            )
        ),
        environment=env,
    )


def diagnostics(case, prepared, bundle, matrix):
    minimum = matrix.min(axis=1)
    requirements = np.maximum(0, case.state.operating_buffer - minimum)
    deficit = np.maximum(0, -(case.state.immediate_funding + minimum))
    day = min(17, bundle.horizon_days)
    added = Obligation("diagnostic", "Diagnostic added payment", 100, day)
    shifted = cash_paths(
        case.state, bundle, (*case.obligations, added), prepared_history=prepared.joint
    )
    reduce = lambda x: cash_risk_summary(
        x,
        case.state.immediate_funding,
        case.state.operating_buffer,
        case.state.coverage_target,
    )
    result = dict(
        minimum_flow_quantiles=np.quantile(minimum, [0, 0.05, 0.5, 0.95, 1]).tolist(),
        minimum_flow_std=float(minimum.std()),
        distinct_index_paths=len(np.unique(bundle.index_matrix, axis=0)),
        distinct_requirements=len(np.unique(requirements)),
        zero_requirement_fraction=float(np.mean(requirements == 0)),
        zero_deficit_fraction=float(np.mean(deficit == 0)),
        minimum_day_counts=np.bincount(
            matrix.argmin(axis=1) + 1, minlength=bundle.horizon_days + 1
        )[1:].tolist(),
        minimum_tie_convention="earliest forecast day",
        baseline=reduce(matrix),
        added_obligation=reduce(shifted),
        added_obligation_day=day,
        added_obligation_amount=100,
    )

    if case.name == "canonical":
        from ginseng.generate import canonical_shocks

        repair = cash_paths(
            case.state, bundle, canonical_shocks(), prepared_history=prepared.joint
        )
        result["canonical_staged_repair"] = reduce(repair)
        result["canonical_staged_repair_events"] = [
            asdict(item) for item in canonical_shocks()
        ]
    return result
