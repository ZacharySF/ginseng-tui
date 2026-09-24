"""Fresh-process benchmark worker, also executable against archived old package."""

import argparse
import hashlib
import json
import os
import resource
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


def run(backend, n, h, case, full, workers):
    from ginseng.inputs import fixture
    from ginseng.metrics import (
        cash_risk_summary,
        compute_scenario_metrics,
        percentile_cash_paths,
        severity_metrics,
    )
    from ginseng.numerical import environment
    from ginseng.provenance import digest
    from ginseng.sampling import prepare_history, sample_bundle
    from ginseng.simulate import cash_paths
    from ginseng.state import Obligation

    state = fixture("canonical").state
    obligations = (
        (Obligation("stress", "Synthetic late bill", 5000.0, h - 2),)
        if case == "stressed"
        else ()
    )
    w = np.linspace(0.1, 2, n) if case == "weighted" else None
    if w is not None:
        w[::11] = 0
    timings = {}

    def timed(name, fn):
        start = time.perf_counter()
        result = fn()
        timings[name] = time.perf_counter() - start
        return result

    history = timed("history_preparation", lambda: prepare_history(state, 14))
    bundle = timed(
        "draw_generation", lambda: sample_bundle(history, h, n, 20260911, "mc", h)
    )
    opening, buffer, q = float(state.immediate_funding), 1000.0, 0.95
    metadata = {}

    def old_summary(x, b, target):
        return {
            **cash_risk_summary(x, opening, b, target, w),
            "dollar_days_below_buffer": severity_metrics(x, opening, b, w)[
                "dollar_days_below_buffer"
            ],
            "overdraft_dollar_days": severity_metrics(x, opening, 0, w)[
                "dollar_days_below_buffer"
            ],
        }

    if backend == "old":
        matrix = timed(
            "cash_kernel",
            lambda: cash_paths(
                state, bundle, obligations, prepared_history=history.joint
            ),
        )
        timed(
            "summaries",
            lambda: dict(
                minima=matrix.min(axis=1),
                terminal=matrix[:, -1],
                minimum_balance=(opening + matrix).min(axis=1),
                maximum_deficit=np.maximum(0, -(opening + matrix).min(axis=1)),
                buffer_dollar_days=np.maximum(0, buffer - (opening + matrix)).sum(
                    axis=1
                ),
                overdraft_dollar_days=np.maximum(0, -(opening + matrix)).sum(axis=1),
            ),
        )
        if full:
            timed(
                "chart_order_statistics",
                lambda: percentile_cash_paths(matrix, opening, w),
            )

        def complete():
            if full:
                return asdict(
                    compute_scenario_metrics(state, bundle, obligations, q, buffer, w)
                )
            x = cash_paths(state, bundle, obligations)
            return old_summary(x, buffer, q)

        output = timed("complete_evaluation", complete)

        def whatif():
            return [
                asdict(
                    compute_scenario_metrics(state, bundle, obligations, target, b, w)
                )
                if full
                else old_summary(cash_paths(state, bundle, obligations), b, target)
                for target, b in ((0.90, 1000), (0.95, 1000), (0.95, 1200))
            ]

        whatif_outputs = timed("three_whatifs", whatif)
        metadata = dict(
            input_bytes=bundle.index_matrix.nbytes + history.joint.nbytes,
            output_bytes=matrix.nbytes + n * 6 * 8,
            scratch_bound_bytes=None,
            note="Old summary request necessarily materializes full cash matrix; old transient arrays measured by RSS.",
        )
    else:
        from ginseng.engine_artifact import calculate, capture, replay
        from ginseng.execution import (
            EvaluationContext,
            ExecutionConfig,
            prepare_scenario,
        )

        config = ExecutionConfig(backend, workers, 1024)
        prepared = timed(
            "prepared_adapter",
            lambda: prepare_scenario(state, bundle, obligations, history.joint),
        )
        with EvaluationContext(config) as ctx:
            result = timed(
                "cash_kernel_fused_summaries",
                lambda: ctx.evaluate(prepared, opening, buffer, full=full),
            )
            metadata = result.metadata
            if full:
                timed(
                    "chart_order_statistics",
                    lambda: ctx.charts(result.paths, opening, w),
                )
        scalars = dict(opening_cash=opening, buffer=buffer, coverage_target=q)

        def complete():
            if full:
                with EvaluationContext(config):
                    return asdict(
                        compute_scenario_metrics(
                            state, bundle, obligations, q, buffer, w
                        )
                    )
            # Include preparation snapshots in complete calculation costs.
            p = prepare_scenario(state, bundle, obligations)
            return calculate(p, config, scalars, w, False)[1]

        output = timed("complete_evaluation", complete)

        def whatif():
            with EvaluationContext(config) as ctx:
                if full:
                    results = [
                        asdict(
                            compute_scenario_metrics(
                                state, bundle, obligations, target, b, w
                            )
                        )
                        for target, b in ((0.90, 1000), (0.95, 1000), (0.95, 1200))
                    ]
                else:
                    p = prepare_scenario(state, bundle, obligations)
                    from ginseng.risk import quantile

                    weights = ctx.weights(n, w)
                    results = []
                    for target, b in ((0.90, 1000), (0.95, 1000), (0.95, 1200)):
                        result = ctx.evaluate(p, opening, b, full=False)
                        stats = result.statistics
                        reserve = quantile(
                            np.maximum(0, b - stats.minima), target, weights
                        )
                        probability = float(
                            np.clip(weights[stats.maximum_deficit > 0].sum(), 0, 1)
                        )
                        mean = float(weights @ stats.maximum_deficit)
                        results.append(
                            dict(
                                required_liquidity_reserve=reserve,
                                cash_shortfall_probability=probability,
                                expected_max_cash_deficit=mean,
                                avg_cash_deficit_when_short=mean / probability
                                if probability
                                else 0.0,
                                funding_gap=max(0.0, reserve - opening),
                                dollar_days_below_buffer=float(
                                    weights @ stats.buffer_dollar_days
                                ),
                                overdraft_dollar_days=float(
                                    weights @ stats.overdraft_dollar_days
                                ),
                            )
                        )
                return results, dict(ctx.counters)

        whatif_outputs, metadata["whatif_counters"] = timed("three_whatifs", whatif)
        with tempfile.TemporaryDirectory(prefix="ginseng-bench-") as directory:
            path = Path(directory) / "run"
            timed(
                "capture_including_evaluation",
                lambda: capture(path, prepared, scalars, w, config=config, full=full),
            )
            result = timed("replay_including_evaluation", lambda: replay(path, config))
            if not result["match"]:
                raise AssertionError(result)
            metadata["artifact_bytes"] = sum(p.stat().st_size for p in path.iterdir())

    def end_to_end():
        hp = prepare_history(state, 14)
        b = sample_bundle(hp, h, n, 20260911, "mc", h)
        if backend == "old":
            if full:
                return compute_scenario_metrics(state, b, obligations, q, buffer, w)
            return old_summary(cash_paths(state, b, obligations), buffer, q)
        if full:
            with EvaluationContext(config):
                return compute_scenario_metrics(state, b, obligations, q, buffer, w)
        return calculate(
            prepare_scenario(state, b, obligations, hp.joint), config, scalars, w, False
        )

    timed("including_preparation_and_draws", end_to_end)
    # Original supported application pipeline for context, including candidates
    # and provenance, but solver and outer uncertainty explicitly disabled.
    if n <= 4000 and full and case == "normal":
        from ginseng.scenario_service import evaluate_scenario

        def application():
            if backend == "old":
                return evaluate_scenario(
                    state,
                    bundle,
                    obligations,
                    coverage_target=q,
                    operating_buffer=buffer,
                    include_optimizer=False,
                )
            with EvaluationContext(config):
                return evaluate_scenario(
                    state,
                    bundle,
                    obligations,
                    coverage_target=q,
                    operating_buffer=buffer,
                    include_optimizer=False,
                )

        timed("application_pipeline_no_solver_no_uncertainty", application)
    return dict(
        timings=timings,
        metadata=metadata,
        whatif_outputs=whatif_outputs,
        output=output,
        output_hash=digest(output),
        input_identity=digest(
            dict(
                history=hashlib.sha256(history.joint.tobytes()).hexdigest(),
                draws=hashlib.sha256(bundle.index_matrix.tobytes()).hexdigest(),
                obligations=[asdict(o) for o in obligations],
                weights=None if w is None else w.tolist(),
                opening=opening,
                buffer=buffer,
                q=q,
            )
        ),
        environment=environment(),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if os.uname().sysname == "Darwin" else 1024),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = vars(parser.parse_args())
    config = json.loads(args["config"])
    run(**config)  # one full warm-up per fresh process; never included in timings
    print(json.dumps(run(**config), allow_nan=False))
