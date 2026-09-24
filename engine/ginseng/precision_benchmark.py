"""Reproducible stopping experiment; references are frozen from the CMC release."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ginseng.benchmark import save_json
from ginseng.generate import canonical_shocks
from ginseng.inputs import fixture
from ginseng.numerical import environment
from ginseng.precision import PrecisionConfig, run_precision
from ginseng.provenance import digest
from ginseng.sampling import derive_seed


def benchmark(config, out, references_dir=Path("artifacts/conditional")):
    out = Path(out)
    references = json.loads((references_dir / "references.json").read_text())
    frozen = json.loads((references_dir / "fixtures.json").read_text())
    precision = PrecisionConfig(
        **{
            key: config[key]
            for key in ("absolute_error", "confidence", "max_paths", "batch_size")
        }
    )
    if type(config["replicates"]) is not int or config["replicates"] < 1:
        raise ValueError("replicates must be a positive integer.")
    reference_manifest = json.loads((references_dir / "manifest.json").read_text())
    rows = []
    cases = {}
    for name in config["cases"]:
        case = fixture("canonical" if name == "repair" else name)
        if name == "repair":
            case = replace(case, name=name, obligations=tuple(canonical_shocks()))
        serial = json.loads(json.dumps(asdict(case), default=lambda x: x.isoformat()))
        if digest(serial) != digest(frozen[name]):
            raise ValueError(f"{name}: input differs from frozen numerical reference.")
        block = 7 if name == "tiny" else config["block_length"]
        if block != reference_manifest["preparation"][name]["block_length"]:
            raise ValueError(f"{name}: block length differs from frozen reference.")
        cases[name] = case
    # Warm library imports before serial timings; each call still prepares inputs.
    run_precision(
        fixture("tiny"), PrecisionConfig(0.1, 0.95, 1024, 256), block_length=7
    )
    jobs = [
        (name, rep, est)
        for name in config["cases"]
        for rep in range(config["replicates"])
        for est in ("path", "initial-block-cmc")
    ]
    np.random.default_rng(derive_seed(config["root_seed"], "mc", 0, 300)).shuffle(jobs)
    for order, (name, rep, est) in enumerate(jobs):
        result = run_precision(
            cases[name],
            precision,
            estimator=est,
            seed=config["root_seed"],
            replicate=rep,
            block_length=7 if name == "tiny" else config["block_length"],
        )
        summary = result["summary"]
        lo, hi = summary["numerical_probability_interval"]
        ref = references[name]
        rows.append(
            dict(
                case=name,
                replicate=rep,
                estimator=est,
                order=order,
                **summary,
                core_seconds=result["core_seconds"],
                checkpoints=result["checkpoints"],
                reference_probability=ref["probability"],
                reference_interval=ref["interval"],
                contains_reference_point=lo <= ref["probability"] <= hi,
                contains_reference_interval=lo <= ref["interval"][0]
                and ref["interval"][1] <= hi,
                input_hash=result["manifest"]["input_hash"],
                index_hash=result["manifest"]["index_hash"],
                derived_seed=result["manifest"]["derived_seed"],
                result_hash=result["manifest"]["result_hash"],
            )
        )
        if (order + 1) % 32 == 0:
            print(f"Completed {order + 1}/{len(jobs)} precision runs", flush=True)
    limited = run_precision(
        fixture("tiny"), PrecisionConfig(1e-6, 0.95, 2048, 512), block_length=7
    )
    save_json(out / "observations.json", rows)
    save_json(out / "budget-exhausted.json", limited)
    save_json(
        out / "manifest.json",
        dict(
            config=config,
            environment=environment(),
            fixture_hashes={name: digest(frozen[name]) for name in config["cases"]},
            reference_hash=digest(references),
            reference_file=str(references_dir / "references.json"),
            interval="Maurer-Pontil two-sided empirical Bernstein + alpha/(k*(k+1)) spending",
            timing="includes history and conditional table preparation, sampling and interval checks; excludes provenance environment collection and JSON",
        ),
    )
    lines = [
        "# Precision-driven MC stopping experiment",
        "",
        f"Predeclared absolute error {precision.absolute_error}, confidence {precision.confidence}, cap {precision.max_paths:,}, batch size {precision.batch_size:,}. {config['replicates']} independent replicates per case/estimator. Both estimators use the same advancing MC streams up to their stopping times.",
        "",
        "| Case | Estimator | Precision reached | Median N | Median core ms | Interval contains reference point |",
        "|---|---|---|---|---|---|",
    ]
    for name in config["cases"]:
        for est in ("path", "initial-block-cmc"):
            group = [r for r in rows if r["case"] == name and r["estimator"] == est]
            lines.append(
                f"| {name} | {est} | {sum(r['precision_met'] for r in group)}/{len(group)} | {np.median([r['actual_n'] for r in group]):.0f} | {1000 * np.median([r['core_seconds'] for r in group]):.3f} | {sum(r['contains_reference_point'] for r in group)}/{len(group)} |"
            )
    lines += [
        "",
        "Only tiny has exact truth (10846/27783); other reference points are independent million-path MC estimates with stored binomial intervals. Inclusion counts for those points are diagnostics, not measured coverage of the unknown true probability. Even exact-reference inclusion across 32 runs does not prove the coverage theorem. The mathematical guarantee comes from the bound and union argument documented in docs/precision-stopping.md.",
        "",
        "A separate tiny run requests error 0.000001 with a 2,048-path cap and reports `max_paths_reached`, `precision_met=false`; see budget-exhausted.json.",
        "",
        "This experiment measures the opt-in failure-only command. Its error target is a high-probability absolute numerical error, whereas the earlier CMC benchmark used replicate RMSE for full metric-returning calls. Timings and sample requirements from those two reports are not directly comparable.",
        "",
        "Recommendation: retain precision stopping as an optional offline MC capability. The conservative bound can require substantial work; no claim of optimal stopping or universal speedup is made. Sobol, reserve precision, model uncertainty and application integration remain outside this release.",
    ]
    (out / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("benchmarks/precision.json")
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/precision"))
    args = parser.parse_args()
    benchmark(json.loads(args.config.read_text()), args.out)
