"""Versioned, explicit local capture of prepared cash calculations (no RNG replay)."""

import json
import os
import shutil
import tempfile
import time
from dataclasses import fields
from hashlib import sha256
from pathlib import Path

import numpy as np

from ginseng.execution import (
    EvaluationContext,
    ExecutionConfig,
    PathStatistics,
    PreparedScenario,
    array_id,
    snapshot,
)
from ginseng.provenance import digest, source_fingerprint
from ginseng.risk import probabilities, quantile

SCHEMA = 1
MAX_BYTES = 512 * 1024**2
# Exact discrete classifications and selected quantiles; accumulation/reductions
# allow sub-micro-dollar rounding differences, never a changed failure mask.
ATOL = 1e-9
RTOL = 1e-12
INPUT_NAMES = ("history", "indices", "direct", "schedule")


class ArtifactError(ValueError):
    pass


def calculate(prepared, config, scalars, weights, full=True):
    with EvaluationContext(config) as context:
        result = context.evaluate(
            prepared, scalars["opening_cash"], scalars["buffer"], full=full
        )
        stats = result.statistics
        w = context.weights(prepared.n_paths, weights)
        required = np.maximum(0, scalars["buffer"] - stats.minima)
        reserve = quantile(required, scalars["coverage_target"], w)
        failure = float(np.clip(w[stats.maximum_deficit > 0].sum(), 0, 1))
        mean = float(w @ stats.maximum_deficit)
        metrics = dict(
            required_liquidity_reserve=reserve,
            cash_shortfall_probability=failure,
            expected_max_cash_deficit=mean,
            avg_cash_deficit_when_short=mean / failure if failure else 0.0,
            funding_gap=max(0.0, reserve - scalars["opening_cash"]),
            dollar_days_below_buffer=float(w @ stats.buffer_dollar_days),
            overdraft_dollar_days=float(w @ stats.overdraft_dollar_days),
        )
        arrays = {
            f"stat_{field.name}": getattr(stats, field.name)
            for field in fields(PathStatistics)
        }
        arrays["cash_failure"] = snapshot(stats.maximum_deficit > 0, "|u1")
        arrays["buffer_breach"] = snapshot(
            stats.minimum_balance < scalars["buffer"], "|u1"
        )
        if full:
            arrays["paths"] = result.paths
            charts = context.charts(result.paths, scalars["opening_cash"], w)
            arrays.update({key: snapshot(value) for key, value in charts.items()})
        return arrays, metrics, dict(result.metadata, counters=dict(context.counters))


def capture(
    destination,
    prepared,
    scalars,
    weights=None,
    *,
    config=ExecutionConfig(),
    full=True,
    extra_inputs=None,
    model=None,
):
    destination = Path(destination)
    if destination.exists():
        raise ArtifactError("Capture destination already exists.")
    validate_scalars(scalars)
    scalars = {key: float(value) for key, value in scalars.items()}
    weights = snapshot(probabilities(prepared.n_paths, weights))
    start = time.perf_counter()
    outputs, metrics, metadata = calculate(prepared, config, scalars, weights, full)
    metadata["evaluation_seconds"] = time.perf_counter() - start
    inputs = {key: getattr(prepared, key) for key in INPUT_NAMES}
    inputs["weights"] = weights
    for key, value in (extra_inputs or {}).items():
        if key not in (
            "known_income",
            "known_obligations",
            "discretionary",
            "portfolio",
        ):
            raise ArtifactError("Unknown additional input field.")
        inputs[key] = snapshot(value)
    from ginseng.numerical import environment

    manifest = dict(
        schema=SCHEMA,
        complete=True,
        source=prepared.source,
        visible_horizon=prepared.visible_horizon,
        material_horizon=prepared.material_horizon,
        scalars=scalars,
        full=full,
        model=model
        or dict(
            estimator="path",
            cash_failure="strict < 0",
            quantile="float64-rounded long-double inverse CDF",
        ),
        execution=metadata,
        environment=environment(),
        metrics=metrics,
        members={},
    )
    manifest["input_identity"] = digest(
        dict(
            arrays={key: array_id(a) for key, a in inputs.items()},
            scalars=scalars,
            source=prepared.source,
            visible=prepared.visible_horizon,
            material=prepared.material_horizon,
            full=full,
        )
    )
    manifest["calculation_identity"] = digest(
        dict(
            model=manifest["model"],
            python_source=source_fingerprint(),
            native=metadata.get("native"),
        )
    )
    manifest["output_digest"] = digest(
        dict(arrays={key: array_id(a) for key, a in outputs.items()}, metrics=metrics)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix="." + destination.name + "-", dir=destination.parent)
    )
    try:
        for kind, arrays in [("input", inputs), ("output", outputs)]:
            for key, a in arrays.items():
                filename = f"{kind}-{key}.npy"
                path = temporary / filename
                with path.open("wb") as stream:
                    np.save(stream, a, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                manifest["members"][f"{kind}:{key}"] = dict(
                    path=filename,
                    dtype=a.dtype.str,
                    shape=list(a.shape),
                    nbytes=a.nbytes,
                    file_bytes=path.stat().st_size,
                    sha256=sha256(path.read_bytes()).hexdigest(),
                )
        manifest["run_identity"] = digest(
            {
                key: manifest[key]
                for key in ("input_identity", "calculation_identity", "output_digest")
            }
        )
        with (temporary / "manifest.json").open("w") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return manifest


def validate_scalars(scalars):
    if (
        set(scalars) != {"opening_cash", "buffer", "coverage_target"}
        or not all(
            isinstance(v, (int, float, np.floating))
            and not isinstance(v, bool)
            and np.isfinite(v)
            for v in scalars.values()
        )
        or not 0 <= scalars["coverage_target"] <= 1
    ):
        raise ArtifactError("Invalid scalar inputs.")


def load(directory):
    directory = Path(directory)
    try:
        manifest_path = directory / "manifest.json"
        if manifest_path.stat().st_size > 1024**2:
            raise ArtifactError("Manifest is too large.")
        m = json.loads(manifest_path.read_text())
        if (
            m["schema"] != SCHEMA
            or m["complete"] is not True
            or type(m["full"]) is not bool
        ):
            raise ArtifactError("Unsupported or incomplete artifact.")
        validate_scalars(m["scalars"])
        members = m["members"]
        if not isinstance(members, dict) or len(members) > 32:
            raise ArtifactError("Invalid member table.")
        required_inputs = {"input:" + key for key in (*INPUT_NAMES, "weights")}
        required_outputs = {"output:stat_" + f.name for f in fields(PathStatistics)} | {
            "output:cash_failure",
            "output:buffer_breach",
        }
        if m["full"]:
            required_outputs |= {
                "output:" + key for key in ("paths", "p10", "p50", "p90")
            }
        allowed = (
            required_inputs
            | required_outputs
            | {
                "input:" + key
                for key in (
                    "known_income",
                    "known_obligations",
                    "discretionary",
                    "portfolio",
                )
            }
        )
        if (
            not required_inputs | required_outputs <= members.keys()
            or not members.keys() <= allowed
        ):
            raise ArtifactError("Missing or unexpected calculation members.")
        arrays = {}
        total = 0
        seen = set()
        for name, info in members.items():
            relative = Path(info["path"])
            if (
                relative.is_absolute()
                or len(relative.parts) != 1
                or relative.suffix != ".npy"
                or relative.name in seen
            ):
                raise ArtifactError("Unsafe or duplicate member path.")
            seen.add(relative.name)
            path = directory / relative
            if path.is_symlink() or path.resolve().parent != directory.resolve():
                raise ArtifactError("Unsafe member link.")
            shape = info["shape"]
            dtype = np.dtype(info["dtype"])
            if (
                dtype.str not in ("<f8", "<i8", "|u1")
                or not isinstance(shape, list)
                or len(shape) > 2
                or any(type(v) is not int or v < 0 for v in shape)
            ):
                raise ArtifactError("Invalid shape or dtype.")
            expected = int(np.prod(shape, dtype=object)) * dtype.itemsize
            total += expected
            if (
                expected != info["nbytes"]
                or total > MAX_BYTES
                or info["file_bytes"] > expected + 65536
                or path.stat().st_size != info["file_bytes"]
            ):
                raise ArtifactError("Artifact size mismatch or resource limit.")
            if sha256(path.read_bytes()).hexdigest() != info["sha256"]:
                raise ArtifactError("Corrupted member: " + name)
            # mmap validates the NPY header before any array-sized allocation.
            a = np.load(path, allow_pickle=False, mmap_mode="r")
            if (
                a.dtype.str != dtype.str
                or list(a.shape) != shape
                or a.nbytes != expected
                or not np.all(np.isfinite(a))
            ):
                raise ArtifactError("Member header/content mismatch: " + name)
            arrays[name] = snapshot(a, dtype.str)
        inputs = {
            name.split(":")[1]: a
            for name, a in arrays.items()
            if name.startswith("input:")
        }
        outputs = {
            name.split(":")[1]: a
            for name, a in arrays.items()
            if name.startswith("output:")
        }
        p = PreparedScenario(
            *(inputs[key] for key in INPUT_NAMES),
            m["visible_horizon"],
            m["material_horizon"],
            m["source"],
        )
        probabilities(p.n_paths, inputs["weights"])
        for key, a in outputs.items():
            expected_shape = (
                (p.n_paths, p.visible_horizon)
                if key == "paths"
                else (
                    (p.visible_horizon,)
                    if key in ("p10", "p50", "p90")
                    else (p.n_paths,)
                )
            )
            if a.shape != expected_shape:
                raise ArtifactError("Output shape mismatch: " + key)
        input_identity = digest(
            dict(
                arrays={key: array_id(a) for key, a in inputs.items()},
                scalars=m["scalars"],
                source=p.source,
                visible=p.visible_horizon,
                material=p.material_horizon,
                full=m["full"],
            )
        )
        output_digest = digest(
            dict(
                arrays={key: array_id(a) for key, a in outputs.items()},
                metrics=m["metrics"],
            )
        )
        if (
            input_identity != m["input_identity"]
            or output_digest != m["output_digest"]
            or m["run_identity"]
            != digest(
                {
                    key: m[key]
                    for key in (
                        "input_identity",
                        "calculation_identity",
                        "output_digest",
                    )
                }
            )
        ):
            raise ArtifactError("Semantic identity/integrity mismatch.")
        return m, p, inputs, outputs
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ArtifactError):
            raise
        raise ArtifactError("Invalid artifact: " + str(exc)) from exc


def replay(directory, config=ExecutionConfig(), output=None):
    m, p, inputs, recorded = load(directory)
    arrays, metrics, metadata = calculate(
        p, config, m["scalars"], inputs["weights"], m["full"]
    )
    difference = compare_outputs(recorded, m["metrics"], arrays, metrics)
    if output:
        capture(
            output,
            p,
            m["scalars"],
            inputs["weights"],
            config=config,
            full=m["full"],
            extra_inputs={
                key: a
                for key, a in inputs.items()
                if key not in (*INPUT_NAMES, "weights")
            },
            model=m["model"],
        )
    return dict(
        input_identity=m["input_identity"],
        recorded_run=m["run_identity"],
        execution=metadata,
        match=difference is None,
        first_divergence=difference,
    )


def mismatch(stage, name, a, b, exact=False):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return dict(
            stage=stage, field=name, left_shape=list(a.shape), right_shape=list(b.shape)
        )
    same = (a == b) if exact else np.isclose(a, b, atol=ATOL, rtol=RTOL)
    if np.all(same):
        return None
    index = tuple(int(i) for i in np.argwhere(~same)[0])
    av, bv = float(a[index]), float(b[index])
    absolute = abs(av - bv)
    return dict(
        stage=stage,
        field=name,
        index=list(index),
        left=av,
        right=bv,
        absolute_error=absolute,
        relative_error=absolute / max(abs(av), abs(bv), np.finfo(float).tiny),
    )


def compare_outputs(left, lmetrics, right, rmetrics):
    for stage, keys in [
        ("cumulative paths", ["paths"]),
        (
            "per-path summaries",
            [f"stat_{f.name}" for f in fields(PathStatistics)]
            + ["cash_failure", "buffer_breach"],
        ),
        ("aggregated metrics", ["p10", "p50", "p90"]),
    ]:
        for key in keys:
            if (key in left) != (key in right):
                return dict(stage=stage, field=key, reason="Output options differ")
            if key in left:
                difference = mismatch(
                    stage,
                    key,
                    left[key],
                    right[key],
                    key in ("cash_failure", "buffer_breach", "p10", "p50", "p90"),
                )
                if difference:
                    return difference
    for key in sorted(lmetrics.keys() | rmetrics.keys()):
        if key not in lmetrics or key not in rmetrics:
            return dict(stage="aggregated metrics", field=key, reason="Missing metric")
        difference = mismatch(
            "aggregated metrics",
            key,
            lmetrics[key],
            rmetrics[key],
            key in ("required_liquidity_reserve", "cash_shortfall_probability"),
        )
        if difference:
            return difference
    return None


def diff(first, second):
    lm, lp, li, lo = load(first)
    rm, rp, ri, ro = load(second)
    intentional = (
        lm["input_identity"] != rm["input_identity"] or lm["model"] != rm["model"]
    )
    difference = None
    for stage, keys in [
        (
            "preparation",
            [
                "history",
                "weights",
                "discretionary",
                "portfolio",
                "known_income",
                "known_obligations",
            ],
        ),
        ("draws", ["indices", "direct"]),
        ("deterministic schedule", ["schedule"]),
    ]:
        for key in keys:
            if (key in li) != (key in ri):
                difference = dict(stage=stage, field=key, reason="Input field differs")
                break
            if key in li:
                difference = mismatch(stage, key, li[key], ri[key], True)
                if difference:
                    break
        if difference:
            break
    if difference is None:
        for key in (
            "scalars",
            "visible_horizon",
            "material_horizon",
            "full",
            "source",
            "model",
        ):
            if lm[key] != rm[key]:
                difference = dict(
                    stage="preparation", field=key, left=lm[key], right=rm[key]
                )
                break
    if difference is None:
        difference = compare_outputs(lo, lm["metrics"], ro, rm["metrics"])
    if (
        difference
        and not lm["full"]
        and difference.get("stage") == "per-path summaries"
        and difference.get("index")
    ):
        from ginseng.execution import numpy_block

        row = difference["index"][0]
        left_row, _ = numpy_block(
            lp,
            row,
            row + 1,
            lm["scalars"]["opening_cash"],
            lm["scalars"]["buffer"],
            True,
        )
        right_row, _ = numpy_block(
            rp,
            row,
            row + 1,
            rm["scalars"]["opening_cash"],
            rm["scalars"]["buffer"],
            True,
        )
        difference["reconstructed_reference_row"] = dict(
            path=row, left=left_row[0].tolist(), right=right_row[0].tolist()
        )
    return dict(
        left_run=lm["run_identity"],
        right_run=rm["run_identity"],
        input_identities=[lm["input_identity"], rm["input_identity"]],
        comparison="intentional scenario/configuration difference"
        if intentional
        else "backend correctness comparison",
        match=difference is None,
        first_divergence=difference,
    )
