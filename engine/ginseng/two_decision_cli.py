"""Offline runner for the explicitly experimental two-decision model."""

import argparse
import json
import resource
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from ginseng.two_decision import Model, run_experiment
from ginseng.two_decision_artifact import capture, replay


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ginseng two-decision")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--training-paths", type=int, default=128)
    run.add_argument("--validation-paths", type=int, default=4096)
    run.add_argument("--replications", type=int, default=5)
    run.add_argument("--seed", type=int, default=20260920)
    run.add_argument("--settlement-days", type=int, default=2)
    run.add_argument("--sale-fee", type=float, default=3.0)
    run.add_argument("--liquidity-charge", type=float, default=1.0)
    rep = sub.add_parser("replay")
    rep.add_argument("artifact", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "replay":
            result = replay(args.artifact)
        else:
            start = perf_counter()
            model = replace(
                Model(),
                settlement_days=args.settlement_days,
                sale_fee=args.sale_fee,
                liquidity_charge=args.liquidity_charge,
            )
            report, payload = run_experiment(
                model=model,
                training_paths=args.training_paths,
                validation_paths=args.validation_paths,
                root_seed=args.seed,
                replications=args.replications,
            )
            t = perf_counter()
            artifact = capture(args.output, payload)
            capture_seconds = perf_counter() - t
            t = perf_counter()
            replayed = replay(args.output)
            replay_seconds = perf_counter() - t
            result = dict(
                report=report,
                capture=artifact,
                replay=replayed,
                match=replayed["match"],
                capture_seconds=capture_seconds,
                replay_seconds=replay_seconds,
                end_to_end_seconds=perf_counter() - start,
                process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024,
                memory_boundary="Linux process high-water RSS including interpreter/libraries; not incremental experiment allocation",
            )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["match"] else 3
    except (ValueError, OSError) as error:
        print(json.dumps(dict(error=str(error))))
        return 2
