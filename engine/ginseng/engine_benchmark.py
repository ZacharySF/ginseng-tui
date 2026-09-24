"""Same-machine interleaved benchmark with fresh-process RSS and old-source oracle."""

import hashlib
import json
import os
import statistics
import subprocess
import sys
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def equivalent(left, right, path=""):
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError("Output fields differ: " + path)
        for key in left:
            equivalent(left[key], right[key], path + "/" + key)
    elif isinstance(left, list):
        if len(left) != len(right):
            raise AssertionError("Output lengths differ: " + path)
        for i, (a, b) in enumerate(zip(left, right)):
            equivalent(a, b, path + "/" + str(i))
    elif isinstance(left, (float, int)) and not isinstance(left, bool):
        exact = any(
            key in path
            for key in (
                "p10",
                "p50",
                "p90",
                "counts",
                "probability",
                "required_liquidity_reserve",
            )
        )
        if (
            left != right
            if exact
            else not np.isclose(left, right, rtol=1e-12, atol=1e-9)
        ):
            raise AssertionError(f"{path}: {left} != {right}")
    elif left != right:
        raise AssertionError("Output differs: " + path)


def benchmark(output, smoke=False):
    output = Path(output)
    if output.exists():
        raise ValueError("Benchmark output directory already exists.")
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    archive = root / "benchmarks/quant-engineering/baseline-source.tar.gz"
    identity = json.loads(archive.with_name("baseline-source.json").read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != identity["archive_sha256"]:
        raise ValueError("Old source archive hash mismatch.")
    workloads = (
        [(128, 14, "normal")]
        if smoke
        else [
            (2000, 30, "normal"),
            (4000, 30, "stressed"),
            (4000, 60, "weighted"),
            (32768, 30, "normal"),
            (32768, 60, "stressed"),
            (32768, 60, "weighted"),
        ]
    )
    configs = [("old", 1), ("numpy", 1), ("native", 1), ("native", 2), ("native", 4)]
    from ginseng.execution import native_info

    native_info(True)
    records = []
    oracle = {}
    repetitions = 1 if smoke else 3
    worker = Path(__file__).with_name("engine_benchmark_worker.py")
    with tempfile.TemporaryDirectory(prefix="ginseng-old-") as directory:
        with tarfile.open(archive) as tar:
            tar.extractall(directory, filter="data")
        for n, h, case in workloads:
            for full in (False, True):
                for repetition in range(repetitions):
                    # Rotate configuration order deterministically each round.
                    order = configs[repetition:] + configs[:repetition]
                    for backend, workers in order:
                        config = dict(
                            backend=backend,
                            n=n,
                            h=h,
                            case=case,
                            full=full,
                            workers=workers,
                        )
                        env = dict(
                            os.environ,
                            PYTHONPATH=str(Path(directory) / "engine")
                            if backend == "old"
                            else str(root / "engine"),
                            OPENBLAS_NUM_THREADS="1",
                            OMP_NUM_THREADS="1",
                            MKL_NUM_THREADS="1",
                        )
                        completed = subprocess.run(
                            [sys.executable, str(worker), json.dumps(config)],
                            env=env,
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        record = json.loads(completed.stdout)
                        record.update(config=config, repetition=repetition)
                        key = (n, h, case, full)
                        if backend == "old":
                            oracle[key] = (
                                record["input_identity"],
                                record["output"],
                                record["whatif_outputs"],
                            )
                        if key in oracle:
                            if oracle[key][0] != record["input_identity"]:
                                raise AssertionError("Benchmark inputs differ.")
                            equivalent(oracle[key][1], record["output"])
                            equivalent(oracle[key][2], record["whatif_outputs"])
                        record["verified_against_old"] = key in oracle
                        records.append(record)
                        with (output / "raw.jsonl").open("a") as stream:
                            stream.write(json.dumps(record, allow_nan=False) + "\n")
    groups = defaultdict(list)
    for row in records:
        c = row["config"]
        groups[
            (c["n"], c["h"], c["case"], c["full"], c["backend"], c["workers"])
        ].append(row)
    summary = []
    for key, rows in groups.items():
        stages = {
            name: dict(
                median=statistics.median(r["timings"][name] for r in rows),
                minimum=min(r["timings"][name] for r in rows),
                maximum=max(r["timings"][name] for r in rows),
            )
            for name in rows[0]["timings"]
        }
        summary.append(
            dict(
                paths=key[0],
                horizon=key[1],
                case=key[2],
                full=key[3],
                backend=key[4],
                workers=key[5],
                stages=stages,
                peak_rss_bytes=max(r["peak_rss_bytes"] for r in rows),
            )
        )
    result = dict(
        baseline=identity,
        repetitions=repetitions,
        warmup="One complete untimed pass in every fresh process",
        policy="Fixed round-robin interleaving; one BLAS thread; each backend prepares its own data; no profiler",
        summary=summary,
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
