"""Actual command-runner entry point for the synthetic Decision Verification Lab."""

import argparse
import json
import resource
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from ginseng.decision_artifact import capture_decision, replay_decision
from ginseng.decision_lab import run_lab
from ginseng.execution import ExecutionConfig
from ginseng.inputs import fixture


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ginseng decision")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--fixture", choices=["canonical"], default="canonical")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--paths", type=int, default=2000)
    run.add_argument("--validation-paths", type=int, default=4000)
    run.add_argument("--horizon", type=int, choices=[14, 30, 60], default=30)
    run.add_argument("--replications", type=int, default=3)
    run.add_argument("--seed", type=int, default=20260920)
    rep = sub.add_parser("replay")
    rep.add_argument("artifact", type=Path)
    for p in (run, rep):
        p.add_argument("--backend", choices=["numpy", "native"], default="numpy")
        p.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        config = ExecutionConfig(args.backend, args.workers)
        if args.command == "replay":
            result = replay_decision(args.artifact, config)
        else:
            from ginseng.generate import canonical_shocks

            case = replace(fixture(args.fixture), obligations=canonical_shocks())
            start = perf_counter()
            lab = run_lab(
                case.state,
                case.obligations,
                paths=args.paths,
                validation_paths=args.validation_paths,
                horizon=args.horizon,
                root_seed=args.seed,
                replications=args.replications,
                config=config,
            )
            lab.report.update(fixture="canonical synthetic repair", synthetic=True)
            t = perf_counter()
            capture = capture_decision(args.output, lab)
            capture_seconds = perf_counter() - t
            t = perf_counter()
            replay = replay_decision(args.output, config)
            replay_seconds = perf_counter() - t
            result = dict(
                report=lab.report,
                capture=capture,
                replay=replay,
                match=replay["match"],
                capture_seconds=capture_seconds,
                replay_seconds=replay_seconds,
                end_to_end_seconds=perf_counter() - start,
                memory=dict(
                    peak_process_rss_kib=resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss,
                    boundary="Linux process high-water RSS including interpreter, imported solver, native and NumPy allocations; not incremental lab allocation",
                ),
            )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["match"] else 3
    except (ValueError, OSError) as e:
        print(json.dumps(dict(error=str(e))))
        return 2
