"""Four-way paired experiment. Run with python -m ginseng.conditional_benchmark."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from ginseng.benchmark import reference, save_json
from ginseng.conditional import prepare_conditional
from ginseng.exact import enumerate_exact
from ginseng.generate import canonical_shocks
from ginseng.inputs import fixture
from ginseng.numerical import environment, run_core
from ginseng.provenance import digest
from ginseng.sampling import derive_seed, prepare_history


def benchmark(config, out):
    out = Path(out)
    observations, references, fixtures, preparation = [], {}, {}, {}
    # Warm imports/library initialization equally; no retained input tables.
    tiny = fixture("tiny")
    p = prepare_history(tiny.state, 7)
    for method in ("mc", "sobol"):
        for estimator in ("path", "initial-block-cmc"):
            run_core(tiny, p, method, 256, 0, estimator=estimator)
    for name in config["cases"]:
        case = fixture("canonical" if name == "repair" else name)
        if name == "repair":
            case = replace(case, name=name, obligations=tuple(canonical_shocks()))
        block = 7 if name == "tiny" else config["block_length"]
        t = perf_counter()
        p = prepare_history(case.state, block)
        history_seconds = perf_counter() - t
        t = perf_counter()
        tables = prepare_conditional(p, case.state, case.obligations)
        preparation[name] = dict(
            history_seconds=history_seconds,
            tables_seconds=perf_counter() - t,
            table_identity=tables.identity,
            block_length=p.resolved_length,
        )
        fixtures[name] = json.loads(
            json.dumps(asdict(case), default=lambda x: x.isoformat())
        )
        if name == "tiny":
            truth = enumerate_exact()["summary"]["cash_shortfall_probability"]
            references[name] = dict(
                kind="exact rational", probability=truth, interval=[truth, truth]
            )
        else:
            ref = reference(case, p, config)
            references[name] = dict(
                kind=ref["kind"],
                probability=ref["summary"]["cash_shortfall_probability"],
                interval=ref["intervals_95"]["cash_shortfall_probability"],
                details=ref,
            )
        jobs = [
            (n, r, m, e, c)
            for n in config["paths"]
            for r in range(config["replicates"])
            for m in ("mc", "sobol")
            for e in ("path", "initial-block-cmc")
            for c in ("fresh", "reuse")
        ]
        rng = np.random.default_rng(derive_seed(config["root_seed"], "mc", 0, 300))
        rng.shuffle(jobs)
        for order, (n, r, m, e, c) in enumerate(jobs):
            t = perf_counter()
            current = prepare_history(case.state, block) if c == "fresh" else p
            b, _, summary = run_core(
                case,
                current,
                m,
                n,
                config["root_seed"],
                replicate=r,
                estimator=e,
                conditional_tables=tables if c == "reuse" and e != "path" else None,
            )
            elapsed = perf_counter() - t
            observations.append(
                dict(
                    case=name,
                    n=n,
                    replicate=r,
                    sampler=m,
                    estimator=e,
                    cache=c,
                    seconds=elapsed,
                    probability=summary["cash_shortfall_probability"],
                    derived_seed=dict(b.sampling_metadata)["derived_seed"],
                    index_hash=b.bootstrap_draw_id,
                    order=order,
                )
            )
        print(f"Completed {name}", flush=True)
    save_json(out / "observations.json", observations)
    save_json(out / "references.json", references)
    save_json(out / "fixtures.json", fixtures)
    save_json(
        out / "manifest.json",
        dict(
            config=config,
            environment=environment(),
            preparation=preparation,
            fixture_hash=digest(fixtures),
            timing="full run_core plus fresh history preparation; excludes JSON, diagnostics and process startup",
        ),
    )
    report(config, observations, references, out)


def report(config, rows, refs, out):
    aggregates = []
    for name in config["cases"]:
        ref = refs[name]
        for cache in ("fresh", "reuse"):
            for method in ("mc", "sobol"):
                for estimator in ("path", "initial-block-cmc"):
                    for n in config["paths"]:
                        group = [
                            r
                            for r in rows
                            if (
                                r["case"],
                                r["cache"],
                                r["sampler"],
                                r["estimator"],
                                r["n"],
                            )
                            == (name, cache, method, estimator, n)
                        ]
                        a = np.array([r["probability"] for r in group])
                        lo, hi = ref["interval"]
                        rmse = lambda truth: float(np.sqrt(np.mean((a - truth) ** 2)))
                        aggregates.append(
                            dict(
                                case=name,
                                cache=cache,
                                sampler=method,
                                estimator=estimator,
                                n=n,
                                rmse=rmse(ref["probability"]),
                                rmse_low=rmse(np.clip(a.mean(), lo, hi)),
                                rmse_high=max(rmse(lo), rmse(hi)),
                                variance=float(a.var(ddof=1)),
                                median_seconds=float(
                                    np.median([r["seconds"] for r in group])
                                ),
                            )
                        )
    save_json(out / "aggregates.json", aggregates)
    lines = [
        "# Initial-block CMC measured experiment",
        "",
        "Full metric-returning calls, 32 independent replicates; probability RMSE target 0.005. "
        "Fresh includes history preparation and CMC tables; reuse uses explicitly prepared immutable inputs. "
        "Both include point generation, mapping, real cash paths, all baseline metrics and CMC work when selected. "
        "JSON, diagnostic charts, process startup and first-import costs are excluded equally.",
        "",
        "## Fastest observed configuration meeting the target",
        "",
        "Only points whose RMSE meets the target across the reference probability interval qualify. "
        "Intervals are marginal 95% binomial numerical-reference intervals, not simultaneous guarantees or confidence intervals on replicate RMSE.",
        "",
        "| Case | Cache | MC/path N; ms | MC/CMC N; ms | Sobol/path N; ms | Sobol/CMC N; ms |",
        "|---|---|---|---|---|---|",
    ]
    for name in config["cases"]:
        for cache in ("fresh", "reuse"):
            cells = []
            for method in ("mc", "sobol"):
                for estimator in ("path", "initial-block-cmc"):
                    candidates = [
                        a
                        for a in aggregates
                        if (a["case"], a["cache"], a["sampler"], a["estimator"])
                        == (name, cache, method, estimator)
                        and a["rmse_high"] <= config["probability_rmse_tolerance"]
                    ]
                    best = (
                        min(candidates, key=lambda a: a["median_seconds"])
                        if candidates
                        else None
                    )
                    cells.append(
                        f"{best['n']}; {1000 * best['median_seconds']:.3f}"
                        if best
                        else "not attained"
                    )
            lines.append("| " + " | ".join([name, cache, *cells]) + " |")
    lines += [
        "",
        "## Same-N variance ratios at N=2048 (path / CMC)",
        "",
        "| Case | MC | Sobol |",
        "|---|---|---|",
    ]
    for name in config["cases"]:
        cells = []
        for method in ("mc", "sobol"):
            pair = [
                next(
                    a
                    for a in aggregates
                    if (a["case"], a["cache"], a["sampler"], a["estimator"], a["n"])
                    == (name, "reuse", method, e, 2048)
                )
                for e in ("path", "initial-block-cmc")
            ]
            cells.append(
                f"{pair[0]['variance'] / pair[1]['variance']:.3f}"
                if pair[1]["variance"]
                else "undefined"
            )
        lines.append("| " + " | ".join([name, *cells]) + " |")
    lines += [
        "",
        "## Interpretation",
        "",
        "Keep CMC experimental and the path estimator as default. Consult the per-case cost table: "
        "lower observation variance does not ensure lower cost, and finite-replicate Sobol improvements have no universal variance guarantee. "
        "Quiet zero observations do not prove zero model risk. Results describe this machine, frozen synthetic inputs and tested grid; "
        "they do not establish real-world calibration. Reuse setup costs are recorded separately in manifest.json and must be amortized over actual calls.",
        "",
        "The original numerical release is preserved in artifacts/standard. This experiment uses its explicit block length 14 (tiny: 7); "
        "it does not reproduce the supplied prototype's fitted length 15 or 52-bit Sobol. Production mapping version 1 and 30-bit scrambling are preserved.",
    ]
    (out / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("benchmarks/conditional.json")
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/conditional"))
    args = parser.parse_args()
    benchmark(json.loads(args.config.read_text()), args.out)
