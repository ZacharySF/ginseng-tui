"""Installed offline numerical command. JSON stdout, diagnostics stderr."""

import argparse
import json
from pathlib import Path
import sys
from ginseng.inputs import fixture, load_input
from ginseng.sampling import prepare_history
from ginseng.numerical import run_core, manifest, diagnostics
from ginseng.exact import enumerate_exact


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "two-decision":
        from ginseng.two_decision_cli import main as experiment_main
        return experiment_main(argv[1:])
    if argv and argv[0] == 'precision-replay':
        from ginseng.precision_replay_cli import main as replay_main
        return replay_main(argv[1:])
    if argv and argv[0] == "decision":
        from ginseng.decision_cli import main as decision_main
        return decision_main(argv[1:])
    if argv and argv[0] == "engine":
        from ginseng.engine_cli import main as engine_main
        return engine_main(argv[1:])
    if argv and argv[0] == "tui":
        try:
            from ginseng.tui import main as tui_main
        except ImportError:
            print("ginseng: the tui command needs the 'tui' extra: uv sync --extra tui", file=sys.stderr)
            return 2
        return tui_main(argv[1:])
    parser = argparse.ArgumentParser(prog="ginseng")
    sub = parser.add_subparsers(dest="command", required=True)
    sim = sub.add_parser("simulate")
    source = sim.add_mutually_exclusive_group()
    source.add_argument(
        "--fixture",
        default=None,
        choices=["canonical", "tiny", "zero-heavy", "drought-heavy"],
    )
    source.add_argument("--input", type=Path)
    sim.add_argument("--sampler", choices=["mc", "sobol", "legacy_mc"], default="mc")
    sim.add_argument("--estimator", choices=["path", "initial-block-cmc"], default="path")
    sim.add_argument("--paths", type=int, default=2048)
    sim.add_argument("--seed", type=int, default=42)
    sim.add_argument("--replicate", type=int, default=0)
    sim.add_argument("--horizon", type=int)
    sim.add_argument("--material-horizon", type=int)
    sim.add_argument("--block-length", type=int)
    precision = sub.add_parser("precision", help="Estimate failure probability to a requested numerical precision")
    precision_source = precision.add_mutually_exclusive_group()
    precision_source.add_argument("--fixture", choices=["canonical", "tiny", "zero-heavy", "drought-heavy"])
    precision_source.add_argument("--input", type=Path)
    precision.add_argument("--sampler", choices=["mc"], default="mc")
    precision.add_argument("--estimator", choices=["path", "initial-block-cmc"], default="path")
    precision.add_argument("--absolute-error", type=float, default=0.005)
    precision.add_argument("--confidence", type=float, default=0.95)
    precision.add_argument("--max-paths", type=int, default=262144)
    precision.add_argument("--batch-size", type=int, default=1024)
    precision.add_argument("--seed", type=int, default=42)
    precision.add_argument("--replicate", type=int, default=0)
    precision.add_argument("--horizon", type=int)
    precision.add_argument("--material-horizon", type=int)
    precision.add_argument("--block-length", type=int)
    precision.add_argument('--chunk-size', type=int)
    precision.add_argument('--time-limit', type=float, default=30.)
    precision.add_argument('--memory-budget', type=int, default=128*1024**2)
    precision.add_argument('--fixed-budget', action='store_true')
    precision.add_argument('--capture', type=Path)
    precision.add_argument('--allow-personal-capture', action='store_true')
    precision.add_argument('--backend', choices=['numpy','native'], default='numpy')
    precision.add_argument('--workers', type=int, default=1)
    exact = sub.add_parser("exact")
    exact.add_argument("--fixture", choices=["tiny"], default="tiny")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--config", type=Path, required=True)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--input", type=Path, required=True)
    for p in (sim, precision, exact, bench, report_parser):
        p.add_argument("--out", type=Path, required=p in (bench, report_parser))
    args = parser.parse_args(argv)
    try:
        if args.command == "simulate":
            case = (
                load_input(args.input)
                if args.input
                else fixture(args.fixture or "canonical")
            )
            requested = (
                args.block_length
                if args.block_length is not None
                else (7 if case.name == "tiny" else None)
            )
            prepared = prepare_history(case.state, requested)
            bundle, x, summary = run_core(
                case,
                prepared,
                args.sampler,
                args.paths,
                args.seed,
                args.horizon,
                args.material_horizon,
                args.replicate,
                estimator=args.estimator,
            )
            result = dict(
                summary=summary,
                manifest=manifest(case, prepared, bundle, summary, args.estimator),
                diagnostics=diagnostics(case, prepared, bundle, x),
            )
        elif args.command == "precision":
            from ginseng.precision import PrecisionConfig, run_precision

            case = load_input(args.input) if args.input else fixture(args.fixture or "canonical")
            config = PrecisionConfig(args.absolute_error, args.confidence, args.max_paths, args.batch_size,
                                     chunk_size=args.chunk_size,time_limit_seconds=args.time_limit,
                                     memory_budget_bytes=args.memory_budget,stop_when_precise=not args.fixed_budget)
            from ginseng.execution import ExecutionConfig
            result = run_precision(case, config, estimator=args.estimator, sampler=args.sampler,
                                   execution=ExecutionConfig(args.backend,args.workers),capture_path=args.capture,
                                   synthetic=args.input is None,allow_personal_capture=args.allow_personal_capture,
                                   seed=args.seed, replicate=args.replicate, horizon=args.horizon,
                                   material_horizon=args.material_horizon,
                                   block_length=args.block_length if args.block_length is not None else (7 if case.name == "tiny" else None))
        elif args.command == "exact":
            from ginseng.provenance import digest
            from ginseng.numerical import environment
            from dataclasses import asdict

            result = enumerate_exact()
            result["manifest"] = dict(
                environment=environment(),
                input_hash=digest(asdict(fixture("tiny"))),
                method="independent rational enumeration",
                block_length=7,
                horizon=4,
                result_hash=digest(result),
            )
        elif args.command == "benchmark":
            from ginseng.benchmark import benchmark

            result = benchmark(json.loads(args.config.read_text()), args.out)
        else:
            from ginseng.benchmark import report

            result = report(args.input, args.out)
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.out and args.command in ("simulate", "precision", "exact"):
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(payload)
        else:
            sys.stdout.write(payload)
        return 0
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        print(f"ginseng: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
