"""Offline prepared-engine commands; previous numerical CLI remains unchanged."""

import argparse
import json
import sys
from pathlib import Path

from ginseng.engine_artifact import capture, diff, replay
from ginseng.execution import (
    ExecutionConfig,
    native_info,
    prepare_scenario,
)


def fixture_inputs(paths=2000, horizon=30, case_name="canonical"):
    from ginseng.inputs import fixture
    from ginseng.sampling import prepare_history, sample_bundle
    from ginseng.simulate import (
        discretionary_resampled_paths,
        known_flows,
        portfolio_value_paths,
    )

    case = fixture(case_name)
    history = prepare_history(case.state, 14)
    bundle = sample_bundle(history, horizon, paths, 20260911, "mc", horizon)
    prepared = prepare_scenario(case.state, bundle, case.obligations, history.joint)
    scalars = dict(
        opening_cash=float(case.state.immediate_funding),
        buffer=1000.0,
        coverage_target=0.95,
    )
    known = known_flows(case.state, case.obligations, horizon)
    extra = dict(
        known_income=known[0],
        known_obligations=known[1],
        discretionary=discretionary_resampled_paths(case.state, bundle),
    )
    portfolio = portfolio_value_paths(case.state, bundle)
    if portfolio is not None:
        extra["portfolio"] = portfolio
    return prepared, scalars, extra


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ginseng engine")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect")
    cap = sub.add_parser("capture")
    cap.add_argument("--fixture", choices=["canonical"], default="canonical")
    cap.add_argument("--output", type=Path, required=True)
    cap.add_argument("--paths", type=int, default=2000)
    cap.add_argument("--horizon", type=int, choices=[14, 30, 60], default=30)
    cap.add_argument("--summary-only", action="store_true")
    rep = sub.add_parser("replay")
    rep.add_argument("run", type=Path)
    rep.add_argument("--output", type=Path)
    dif = sub.add_parser("diff")
    dif.add_argument("first", type=Path)
    dif.add_argument("second", type=Path)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--suite", choices=["quant-engineering"], required=True)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--smoke", action="store_true")
    for p in (cap, rep):
        p.add_argument(
            "--backend", choices=["numpy", "native", "auto"], default="numpy"
        )
        p.add_argument("--workers", type=int, default=1)
        p.add_argument("--block-size", type=int, default=1024)
        p.add_argument("--memory-budget", type=int, default=512 * 1024**2)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = dict(
                api_version=1,
                native=native_info(),
                auto=dict(
                    backend="numpy", reason="Conservative default; see measured report."
                ),
                boundary="Prepared cash paths, summaries, risk metrics and optional charts; no solver/data-fetch replay.",
                exit_codes={
                    "success": 0,
                    "invalid_input_or_artifact": 2,
                    "unexpected_mismatch": 3,
                    "scenario_difference": 4,
                },
            )
        elif args.command == "capture":
            p, s, extra = fixture_inputs(args.paths, args.horizon)
            result = capture(
                args.output,
                p,
                s,
                config=ExecutionConfig(
                    args.backend, args.workers, args.block_size, args.memory_budget
                ),
                full=not args.summary_only,
                extra_inputs=extra,
                model=dict(
                    estimator="path",
                    sampler="mc",
                    seed=20260911,
                    mean_block_length=14,
                    mapping_version=1,
                ),
            )
        elif args.command == "replay":
            result = replay(
                args.run,
                ExecutionConfig(
                    args.backend, args.workers, args.block_size, args.memory_budget
                ),
                args.output,
            )
        elif args.command == "diff":
            result = diff(args.first, args.second)
        else:
            from ginseng.engine_benchmark import benchmark

            result = benchmark(args.output, smoke=args.smoke)
        print(json.dumps(result, indent=2, allow_nan=False))
        if result.get("match") is False:
            return 4 if result.get("comparison", "").startswith("intentional") else 3
        return 0
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        print(f"ginseng engine: {exc}", file=sys.stderr)
        return 2
