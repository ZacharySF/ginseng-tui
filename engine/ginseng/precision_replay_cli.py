"""Replay exact captured precision inputs without network calls."""

import argparse
import json
from pathlib import Path

from ginseng.execution import ExecutionConfig
from ginseng.precision_artifact import replay_precision


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ginseng precision-replay")
    parser.add_argument("capture", type=Path)
    parser.add_argument("--backend", choices=["numpy", "native"], default="numpy")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        result = replay_precision(
            args.capture, ExecutionConfig(args.backend, args.workers)
        )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["match"] else 3
    except (ValueError, OSError) as error:
        print(json.dumps(dict(error=str(error))))
        return 2
