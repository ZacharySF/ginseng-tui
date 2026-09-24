"""Serial interleaved numerical experiments; bounded independent MC references."""

import json
import math
from pathlib import Path
from time import perf_counter
import sys
import numpy as np
from scipy.stats import beta, binom
from ginseng.inputs import fixture
from ginseng.exact import enumerate_exact
from ginseng.sampling import (
    prepare_history,
    derive_seed,
    unit_points,
    map_indices,
    validate_size,
)
from ginseng.numerical import run_core, manifest, environment, diagnostics, TARGETS
from ginseng.metrics import cash_risk_summary
from ginseng.simulate import DrawBundle, cash_paths, _compute_draw_id
from ginseng.provenance import digest


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def reference(case, prepared, config):
    n, batch = config["reference_paths"], config["reference_batch"]
    h = case.state.forecast_horizon
    validate_size("mc", batch, h)
    seed = derive_seed(config["root_seed"], "mc", 0, 200)
    rng = np.random.Generator(np.random.PCG64(seed))
    minima = np.empty(n)
    for start in range(0, n, batch):
        count = min(batch, n - start)
        idx = map_indices(
            rng.random((count, 2 * h - 1)),
            len(prepared.joint),
            prepared.resolved_length,
        )
        bundle = DrawBundle(
            seed,
            h,
            count,
            prepared.resolved_length,
            prepared.clipped,
            len(prepared.joint),
            idx,
            _compute_draw_id(idx),
        )
        x = cash_paths(
            case.state, bundle, case.obligations, prepared_history=prepared.joint
        )
        minima[start : start + count] = x.min(axis=1)
    c, b, q = (
        case.state.immediate_funding,
        case.state.operating_buffer,
        case.state.coverage_target,
    )
    summary = cash_risk_summary(minima[:, None], c, b, q)
    deficits = np.maximum(0, -(c + minima))
    k = int(np.count_nonzero(deficits))
    failure_interval = [
        float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0,
        float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0,
    ]
    requirements = np.sort(np.maximum(0, b - minima))
    # Binomial rank interval for the population inverse CDF; ties retained.
    lo = max(0, int(binom.ppf(0.025, n, q)) - 1)
    hi = min(n - 1, int(binom.ppf(0.975, n, q)))
    mean_se = float(deficits.std(ddof=1) / np.sqrt(n))
    # No observed failures cannot establish a zero mean. A finite model bound
    # times the exact binomial upper bound remains valid in that case.
    worst_index = int(
        np.argmin(prepared.joint[:, 0] - prepared.joint[:, 1] - prepared.joint[:, 2])
    )
    worst_bundle = DrawBundle(
        seed,
        h,
        1,
        prepared.resolved_length,
        prepared.clipped,
        len(prepared.joint),
        np.full((1, h), worst_index),
        "bound",
    )
    worst_path = cash_paths(
        case.state, worst_bundle, case.obligations, prepared_history=prepared.joint
    )
    max_possible_deficit = max(0.0, -(c + float(worst_path.min())))
    mean_interval = [
        max(0, summary["expected_max_cash_deficit"] - 1.96 * mean_se),
        summary["expected_max_cash_deficit"] + 1.96 * mean_se,
    ]
    if k == 0:
        mean_interval = [0.0, max_possible_deficit * failure_interval[1]]
    intervals = dict(
        required_liquidity_reserve=[float(requirements[lo]), float(requirements[hi])],
        cash_shortfall_probability=failure_interval,
        expected_max_cash_deficit=mean_interval,
    )
    return dict(
        kind="independent MC numerical reference",
        summary=summary,
        n=n,
        batch=batch,
        seed=seed,
        domain=200,
        intervals_95=intervals,
        mean_deficit_se=mean_se,
        max_possible_deficit=max_possible_deficit,
        interval_methods="Clopper-Pearson binomial; mean normal SE (zero events: model-bound times binomial upper bound); binomial order-statistic reserve interval (ties retained)",
    )


def aggregate(observations, references):
    groups = {}
    for row in observations:
        for target, value in row["estimates"].items():
            if target not in (*TARGETS, "integral"):
                continue
            groups.setdefault(
                (row["case"], row["method"], row["n"], target), []
            ).append((value, row["seconds"]))
    output = []
    for (case, method, n, target), rows in sorted(groups.items()):
        values, times = np.asarray(rows).T
        truth = references[case]["summary"][target]
        errors = values - truth
        interval = references[case].get("intervals_95", {}).get(target, [truth, truth])
        uncertainty = max(abs(truth - interval[0]), abs(interval[1] - truth))
        closest_truth = float(np.clip(values.mean(), *interval))
        rmse_low = float(np.sqrt(np.mean((values - closest_truth) ** 2)))
        rmse_high = float(
            max(np.sqrt(np.mean((values - endpoint) ** 2)) for endpoint in interval)
        )
        output.append(
            dict(
                case=case,
                method=method,
                n=n,
                target=target,
                replicates=len(rows),
                rmse_reference_interval=[rmse_low, rmse_high],
                bias=float(errors.mean()),
                rmse=float(np.sqrt(np.mean(errors**2))),
                sd=float(values.std(ddof=1)) if len(rows) > 1 else 0.0,
                seconds_median=float(np.median(times)),
                seconds_p10=float(np.quantile(times, 0.1)),
                seconds_p90=float(np.quantile(times, 0.9)),
                reference_kind=references[case]["kind"],
                reference_uncertainty_95=uncertainty,
            )
        )
    return output


def cost_table(aggregates, config):
    groups = {}
    for row in aggregates:
        groups.setdefault((row["case"], row["target"]), []).append(row)
    results = []
    for (case, target), rows in sorted(groups.items()):
        tol = config["tolerances"].get(case, config["tolerances"]["default"])[target]
        chosen = {}
        for method in sorted({r["method"] for r in rows}):
            eligible = [r for r in rows if r["method"] == method and r["rmse"] <= tol]
            chosen[method] = (
                min(eligible, key=lambda r: r["seconds_median"]) if eligible else None
            )
        ordinary = [r for m, r in chosen.items() if m != "sobol" and r is not None]
        baseline = (
            min(ordinary, key=lambda r: r["seconds_median"]) if ordinary else None
        )
        sobol = chosen.get("sobol")
        # Reference uncertainty near tolerance makes the attainment unresolved.
        resolved = max(r["reference_uncertainty_95"] for r in rows) < tol / 2
        if baseline and sobol:
            resolved = (
                resolved
                and baseline["rmse_reference_interval"][1] <= tol
                and sobol["rmse_reference_interval"][1] <= tol
            )
        speedup = (
            baseline["seconds_median"] / sobol["seconds_median"]
            if baseline and sobol and resolved
            else None
        )
        results.append(
            dict(
                case=case,
                target=target,
                tolerance=tol,
                methods=chosen,
                reference_resolved_at_tolerance=bool(resolved),
                best_ordinary=baseline["method"] if baseline else None,
                cost_speedup=speedup,
            )
        )
    return results


def benchmark(config, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for key in (
        "replicates",
        "reference_paths",
        "reference_batch",
        "root_seed",
        "block_length",
    ):
        if type(config[key]) is not int or config[key] < (
            0 if key == "root_seed" else 1
        ):
            raise ValueError(f"{key} must be a valid integer.")
    if (
        not 2 <= config["replicates"] <= 1024
        or not 2 <= config["reference_paths"] <= 2**24
    ):
        raise ValueError("Require 2..1024 replicates and 2..2^24 reference paths.")
    if (
        not config["paths"]
        or len(config["paths"]) > 32
        or len(set(config["paths"])) != len(config["paths"])
    ):
        raise ValueError("Provide 1..32 distinct path counts.")
    if (
        not config["cases"]
        or len(set(config["cases"])) != len(config["cases"])
        or not config["methods"]
        or len(set(config["methods"])) != len(config["methods"])
    ):
        raise ValueError("Provide nonempty, distinct cases and methods.")
    for group in config["tolerances"].values():
        if any(
            not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0
            for v in group.values()
        ):
            raise ValueError("Error tolerances must be finite and positive.")
    for method in config["methods"]:
        for n in config["paths"]:
            validate_size(method, n, 4)
    for name in config["cases"]:
        targets = ["integral"] if name == "smooth" else TARGETS
        for target in targets:
            if target not in config["tolerances"].get(
                name, config["tolerances"]["default"]
            ):
                raise ValueError(f"Missing tolerance for {name}/{target}.")
    cases = {name: fixture(name) for name in config["cases"] if name != "smooth"}
    prepared = {
        name: prepare_history(
            case.state, 7 if name == "tiny" else config["block_length"]
        )
        for name, case in cases.items()
    }
    for name, case in cases.items():
        for method in config["methods"]:
            for n in config["paths"]:
                validate_size(method, n, case.state.forecast_horizon)
    tasks = [
        dict(
            case=name,
            method=method,
            n=n,
            replicate=r,
            derived_seed=derive_seed(config["root_seed"], method, r),
        )
        for name in config["cases"]
        for n in config["paths"]
        for r in range(config["replicates"])
        for method in config["methods"]
        if not (name == "smooth" and method == "legacy_mc")
    ]
    order_seed = derive_seed(config["root_seed"], "mc", 0, 300)
    np.random.Generator(np.random.PCG64(order_seed)).shuffle(tasks)
    save_json(out / "schedule.json", tasks)
    save_json(out / "config.json", config)
    env = environment()
    run_manifest = dict(
        environment=env,
        configuration_hash=digest(config),
        execution_order_seed=order_seed,
        timing="perf_counter: sampler construction, randomization, point generation, mapping, draw ownership/hash, existing cash_paths, shared minimal reduction; input preparation and IO excluded",
        memory="512 MiB conservative estimated simultaneous array budget per run; not measured peak RSS. References keep global minima (8*N bytes), in bounded MC batches.",
        status="running",
        cases={},
    )
    run_manifest["sampler_settings"] = {
        "mc": {
            "generator": "PCG64",
            "layout": "float64 C-order fixed dimension",
            "mapping_version": 1,
        },
        "sobol": {
            "scramble": "LMS + digital shift",
            "bits": 30,
            "optimization": None,
            "api": "seed=Generator; random_base2",
            "mapping_version": 1,
        },
        "legacy_mc": {
            "generator": "PCG64",
            "mapping_version": "legacy",
            "prefix_guarantee": False,
        },
    }
    run_manifest["seed_scheme"] = (
        "SeedSequence([root, domain, method_id, replicate]).uint64; domains 100 experiment, 200 reference, 300 order; method IDs mc=1 sobol=2 legacy_mc=3"
    )
    run_manifest["reference_config"] = {
        k: config[k] for k in ("reference_paths", "reference_batch", "root_seed")
    }
    run_manifest["configuration_source_hash"] = digest(config)
    save_json(out / "manifest.json", run_manifest)
    refs = {}
    if "smooth" in config["cases"]:
        refs["smooth"] = dict(
            kind="analytic exact", summary={"integral": (4 * math.expm1(0.25)) ** 4}
        )
    for name, case in cases.items():
        print(f"Reference: {name}", file=sys.stderr, flush=True)
        refs[name] = (
            dict(kind="rational exact", **enumerate_exact())
            if name == "tiny"
            else reference(case, prepared[name], config)
        )
        bundle, x, summary = run_core(
            case, prepared[name], "mc", config["paths"][-1], config["root_seed"]
        )
        run_manifest["cases"][name] = manifest(case, prepared[name], bundle, summary)
        save_json(
            out / f"diagnostics-{name}.json",
            diagnostics(case, prepared[name], bundle, x),
        )
    save_json(out / "references.json", refs)
    save_json(out / "manifest.json", run_manifest)
    # Warm every generator/import without charging one method for import startup.
    for method in config["methods"]:
        if method != "legacy_mc":
            unit_points(method, 16, 4, derive_seed(0, method))
    observations = []
    tiny_support = [
        r["value"] for r in refs.get("tiny", {}).get("reserve_distribution", [])
    ]
    if tiny_support:
        i = tiny_support.index(refs["tiny"]["summary"]["required_liquidity_reserve"])
        cdf_values = tiny_support[max(0, i - 1) : i + 2]
    for count, task in enumerate(tasks):
        name, method, n, r = task["case"], task["method"], task["n"], task["replicate"]
        started = perf_counter()
        if name == "smooth":
            u = unit_points(method, n, 4, task["derived_seed"])
            summary = {"integral": float(np.exp(u.mean(axis=1)).mean())}
        else:
            bundle, x, summary = run_core(
                cases[name], prepared[name], method, n, config["root_seed"], replicate=r
            )
        elapsed = perf_counter() - started
        row = dict(**task, seconds=elapsed, estimates=summary)
        if name != "smooth":
            row["index_hash"] = bundle.bootstrap_draw_id
            row["input_hash"] = run_manifest["cases"][name]["input_hash"]
            row["model_hash"] = run_manifest["cases"][name]["model_hash"]
            row["result_hash"] = digest(
                {k: v for k, v in row.items() if k != "seconds"}
            )
        if name == "tiny":
            req = np.maximum(0, cases[name].state.operating_buffer - x.min(axis=1))
            row["reserve_cdf"] = {str(v): float(np.mean(req <= v)) for v in cdf_values}
        observations.append(row)
        if count % 100 == 0:
            print(f"Replicates: {count + 1}/{len(tasks)}", file=sys.stderr, flush=True)
    save_json(out / "observations.json", observations)
    aggregates = aggregate(observations, refs)
    save_json(out / "aggregates.json", aggregates)
    save_json(out / "costs.json", cost_table(aggregates, config))
    run_manifest["status"] = "complete"
    run_manifest["observation_count"] = len(observations)
    save_json(out / "manifest.json", run_manifest)
    return dict(status="complete", observations=len(observations), output=str(out))


def report(input_dir, out):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError(
            "Report requires research dependencies: uv sync --extra research"
        ) from exc
    src, out = Path(input_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    read = lambda name: json.loads((src / name).read_text())
    config, refs, obs = (
        read("config.json"),
        read("references.json"),
        read("observations.json"),
    )
    if read("manifest.json")["status"] != "complete":
        raise ValueError("Benchmark is incomplete.")
    agg = aggregate(obs, refs)
    costs = cost_table(agg, config)
    save_json(out / "aggregates.json", agg)
    save_json(out / "costs.json", costs)
    lines = [
        f"# Measured results ({config['profile']})",
        "",
        f"{config['replicates']} independent replicates; N={config['paths']}. Financial timings include the complete shared numerical core.",
        "",
        "Errors for larger cases are relative to independent MC, not exact truth. No forecast-calibration claim follows.",
        "",
        "| Case | Target | Method | N | RMSE | Median ms | Reference 95% half-width* |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in agg:
        if r["n"] == max(config["paths"]):
            lines.append(
                f"| {r['case']} | {r['target']} | {r['method']} | {r['n']} | {r['rmse']:.6g} | {1000 * r['seconds_median']:.3f} | {r['reference_uncertainty_95']:.5g} |"
            )
    lines += [
        "",
        "*Maximum distance from reference estimate to either interval endpoint.",
        "",
        "## Cost at predefined error tolerances",
        "",
        "Fastest observed eligible grid point; best of new and legacy MC is the ordinary baseline. Missing attainment means not reached on this grid. Reference uncertainty >= half the tolerance makes a comparison unresolved.",
        "",
        "| Case | Target | Tolerance | Best ordinary | MC/Sobol cost ratio |",
        "|---|---|---:|---|---:|",
    ]
    for r in costs:
        ratio = (
            f"{r['cost_speedup']:.3f}"
            if r["cost_speedup"] is not None
            else (
                "unresolved reference"
                if not r["reference_resolved_at_tolerance"]
                else "not reached on this grid"
            )
        )
        lines.append(
            f"| {r['case']} | {r['target']} | {r['tolerance']} | {r['best_ordinary'] or 'not reached'} | {ratio} |"
        )
    lines += [
        "",
        "Ratios above one favor Sobol; below one favor ordinary MC. Zero RMSE on an atom is retained, without dividing by RMSE. These timings describe this machine and grid only.",
        "",
        "The smooth integral is a four-dimensional positive control, not evidence for the financial model. Financial mapping contains discontinuities and atoms; no universal rate or speedup is asserted.",
        "",
    ]
    lines += [
        "## Reference estimates and 95% intervals",
        "",
        "| Case | Target | Reference | Interval |",
        "|---|---|---:|---|",
    ]
    for name, ref in refs.items():
        for target in ["integral"] if name == "smooth" else TARGETS:
            value = ref["summary"][target]
            interval = ref.get("intervals_95", {}).get(target)
            lines.append(
                f"| {name} | {target} | {value:.9g} | {interval if interval is not None else 'exact'} |"
            )
    lines += [
        "",
        "The raw aggregates also give the possible RMSE range when the reference varies within its interval; those ranges are reference sensitivity, not confidence intervals for replicate RMSE.",
        "",
        "## Path diversity and deterministic events",
        "",
        "Diagnostic MC sample uses the largest configured N, root seed, and replicate 0. Earliest minimum day breaks ties; full counts, path-minimum quantiles and settings are saved in diagnostics JSON.",
        "",
        "| Case | Distinct index paths | Distinct reserves | Zero reserve mass | Zero deficit mass | Baseline reserve | With $100 later bill |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in config["cases"]:
        if name == "smooth":
            continue
        d = read(f"diagnostics-{name}.json")
        lines.append(
            f"| {name} | {d['distinct_index_paths']} | {d['distinct_requirements']} | {d['zero_requirement_fraction']:.6g} | {d['zero_deficit_fraction']:.6g} | {d['baseline']['required_liquidity_reserve']:.6g} | {d['added_obligation']['required_liquidity_reserve']:.6g} |"
        )
    if "tiny" in refs:
        lines += [
            "",
            "## Exact-case CDF around the reserve",
            "",
            "A stable reserve can coexist with CDF error. The exact inverse CDF selects $90 once cumulative mass crosses 0.95; nearby CDF discrepancies are reported separately.",
            "",
            "| Method | N | Reserve support | Exact CDF | CDF RMSE |",
            "|---|---:|---:|---:|---:|",
        ]
        truth = {
            str(r["value"]): r["cdf"] for r in refs["tiny"]["reserve_distribution"]
        }
        cdf_aggregates = []
        for method in config["methods"]:
            for n in config["paths"]:
                rows = [
                    r
                    for r in obs
                    if r["case"] == "tiny" and r["method"] == method and r["n"] == n
                ]
                for v in rows[0]["reserve_cdf"] if rows else []:
                    error = float(
                        np.sqrt(
                            np.mean(
                                [(r["reserve_cdf"][v] - truth[v]) ** 2 for r in rows]
                            )
                        )
                    )
                    cdf_aggregates.append(
                        dict(
                            method=method,
                            n=n,
                            support=float(v),
                            truth=truth[v],
                            rmse=error,
                        )
                    )
                    if n == max(config["paths"]):
                        lines.append(
                            f"| {method} | {n} | {v} | {truth[v]:.9g} | {error:.6g} |"
                        )
        save_json(out / "cdf-aggregates.json", cdf_aggregates)
    lines += [
        "",
        "## Selected costs",
        "",
        "| Case | Target | Method | Selected N | Median seconds | Status |",
        "|---|---|---|---:|---:|---|",
    ]
    for r in costs:
        for method, selected in r["methods"].items():
            if selected:
                lines.append(
                    f"| {r['case']} | {r['target']} | {method} | {selected['n']} | {selected['seconds_median']:.6g} | observed attainment |"
                )
            else:
                lines.append(
                    f"| {r['case']} | {r['target']} | {method} | — | — | not reached on this grid |"
                )
    lines += ["", "## Error plots", ""]
    for case in config["cases"]:
        targets = ["integral"] if case == "smooth" else list(TARGETS)
        for axis in ("n", "seconds_median"):
            fig, axes = plt.subplots(
                1, len(targets), figsize=(5 * len(targets), 3.5), squeeze=False
            )
            for ax, target in zip(axes[0], targets):
                for method in config["methods"]:
                    rows = sorted(
                        [
                            r
                            for r in agg
                            if r["case"] == case
                            and r["target"] == target
                            and r["method"] == method
                        ],
                        key=lambda r: r[axis],
                    )
                    if rows:
                        ax.plot(
                            [r[axis] for r in rows],
                            [r["rmse"] for r in rows],
                            "o-",
                            label=method,
                        )
                ax.set_xscale("log")
                errors = [
                    r["rmse"]
                    for r in agg
                    if r["case"] == case and r["target"] == target
                ]
                if max(errors) == 0:
                    ax.set_ylim(0, 1)
                    ax.text(
                        0.5,
                        0.5,
                        "All observed RMSE = 0",
                        transform=ax.transAxes,
                        ha="center",
                        fontsize=9,
                    )
                elif min(errors) > 0:
                    ax.set_yscale("log")
                else:
                    ax.set_yscale("symlog", linthresh=max(errors) / 1000)
                    ax.set_ylim(bottom=0)
                ax.set_title(target.replace("_", " "), fontsize=9)
                ax.set_xlabel("Paths" if axis == "n" else "Core seconds")
                ax.set_ylabel("RMSE")
                ax.legend()
            fig.suptitle(case)
            fig.tight_layout()
            fig.savefig(out / f"{case}-{axis}.svg")
            plt.close(fig)
        lines += [
            f"![{case} error versus N]({case}-n.svg)",
            f"![{case} error versus cost]({case}-seconds_median.svg)",
            "",
        ]
    (out / "results.md").write_text("\n".join(lines))
    return dict(report=str(out / "results.md"))
