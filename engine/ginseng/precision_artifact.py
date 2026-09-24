"""Bounded exact-input precision evidence, wrapping the prepared engine artifact."""

import json
import os
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path

import numpy as np

from ginseng import engine_artifact
from ginseng.engine_artifact import ArtifactError
from ginseng.execution import EvaluationContext, ExecutionConfig
from ginseng.provenance import digest


def capture_precision(destination, prepared, result, opening, trace, execution):
    destination = Path(destination)
    if destination.exists():
        raise ArtifactError("Capture destination exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".precision-", dir=destination.parent))
    try:
        engine_artifact.capture(
            tmp / "engine",
            prepared,
            dict(opening_cash=opening, buffer=0.0, coverage_target=0.95),
            config=execution,
            full=False,
            model=dict(model_identity=result["manifest"]["model_identity"]),
        )
        m = dict(
            schema=1,
            complete=True,
            result=result,
            engine_sha256=sha256(
                (tmp / "engine" / "manifest.json").read_bytes()
            ).hexdigest(),
            trace=None,
        )
        if trace is not None:
            np.save(
                tmp / "trace.npy", np.asarray(trace, dtype="<i8"), allow_pickle=False
            )
            m["trace"] = dict(
                bytes=(tmp / "trace.npy").stat().st_size,
                sha256=sha256((tmp / "trace.npy").read_bytes()).hexdigest(),
                n=len(trace),
            )
        m["integrity"] = digest(m)
        data = json.dumps(m, indent=2, allow_nan=False).encode()
        if len(data) > 2**20:
            raise ArtifactError("Precision manifest exceeds 1 MiB")
        with (tmp / "precision.json").open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, destination)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def load_precision(directory):
    directory = Path(directory)
    try:
        file = directory / "precision.json"
        if directory.is_symlink() or file.is_symlink() or file.stat().st_size > 2**20:
            raise ArtifactError("Unsafe precision manifest")
        m = json.loads(file.read_text())
        integrity = m.pop("integrity")
        if digest(m) != integrity or m["schema"] != 1 or m["complete"] is not True:
            raise ArtifactError("Invalid precision integrity/schema")
        result = m["result"]
        model = result["manifest"]
        from ginseng.precision import PrecisionConfig
        from ginseng.precision_stream import PrecisionStream

        config = PrecisionConfig(**model["config"])
        if (
            model["checkpoint_schedule"] != list(config.checkpoints())
            or type(model["effective_chunk_size"]) is not int
            or model["effective_chunk_size"] < 1
        ):
            raise ArtifactError("Invalid precision checkpoint/chunk contract")
        stream = PrecisionStream(
            model["root_seed"], model["replicate"], model["material_horizon"]
        )
        if (
            model["stream_layout"] != stream.identity()
            or model["derived_seed"] != stream.seed
        ):
            raise ArtifactError("Incompatible random-input stream")
        if (
            model["version"] != 2
            or model["interval_method"]
            != "empirical-bernstein-alpha-spending-intersection-v2"
            or model["sampler"] != "mc"
        ):
            raise ArtifactError("Incompatible precision model/method")
        if model["estimator"] not in ("path", "initial-block-cmc"):
            raise ArtifactError("Unsupported captured estimator")
        folder = directory / "engine"
        file = folder / "manifest.json"
        if folder.is_symlink() or file.is_symlink() or file.stat().st_size > 2**20:
            raise ArtifactError("Unsafe prepared artifact")
        if sha256(file.read_bytes()).hexdigest() != m["engine_sha256"]:
            raise ArtifactError("Prepared artifact digest mismatch")
        meta, p, inputs, _ = engine_artifact.load(folder)
        if (
            p.n_paths != result["summary"]["actual_n"]
            or p.visible_horizon != model["visible_horizon"]
            or p.material_horizon != model["material_horizon"]
            or p.source != "history"
        ):
            raise ArtifactError("Incompatible precision dimensions")
        if not 1 <= p.n_paths <= config.max_paths:
            raise ArtifactError("Invalid precision observation count")
        identity = digest(
            dict(
                history=p.history.tolist(),
                daily=p.schedule[: p.visible_horizon].tolist(),
                opening_cash=meta["scalars"]["opening_cash"],
                block_length=model["block_resolved"],
                horizon=p.visible_horizon,
                law="stationary-bootstrap uniform starts v1",
                failure_boundary="strict < 0",
            )
        )
        if (
            identity != model["model_identity"]
            or identity != result["summary"]["model_identity"]
            or identity != meta["model"]["model_identity"]
        ):
            raise ArtifactError("Prepared model identity mismatch")
        if (
            sha256(p.indices[:, : p.visible_horizon].tobytes()).hexdigest()
            != model["index_hash"]
        ):
            raise ArtifactError("Precision index identity mismatch")
        if p.n_paths * p.material_horizon * 80 > 512 * 1024**2:
            raise ArtifactError("Replay working arrays exceed budget")
        if not np.array_equal(inputs["weights"], np.full(p.n_paths, 1 / p.n_paths)):
            raise ArtifactError("Weighted precision artifact unsupported")
        trace = None
        if m["trace"] is not None:
            info = m["trace"]
            file = directory / "trace.npy"
            if (
                file.is_symlink()
                or info["n"] != p.n_paths
                or not 8 * p.n_paths <= info["bytes"] <= 8 * p.n_paths + 65536
                or file.stat().st_size != info["bytes"]
            ):
                raise ArtifactError("Invalid trace size/path")
            if sha256(file.read_bytes()).hexdigest() != info["sha256"]:
                raise ArtifactError("Trace corruption")
            trace = np.load(file, mmap_mode="r", allow_pickle=False)
            if (
                trace.dtype.str != "<i8"
                or trace.shape != (p.n_paths,)
                or np.any((trace < 1) | (trace > p.visible_horizon))
            ):
                raise ArtifactError("Invalid trace array")
        if (
            trace is not None
            and sha256(trace.tobytes()).hexdigest() != model["initial_block_trace_hash"]
        ):
            raise ArtifactError("Precision trace identity mismatch")
        if (model["estimator"] == "initial-block-cmc") != (trace is not None):
            raise ArtifactError("Missing or unexpected initial-block trace")
        expected = digest(
            dict(
                manifest={k: v for k, v in model.items() if k != "result_hash"},
                summary=result["summary"],
                checkpoints=result["checkpoints"],
            )
        )
        if expected != model["result_hash"]:
            raise ArtifactError("Result digest mismatch")
        return result, p, meta, trace
    except (KeyError, TypeError, OSError, ValueError, OverflowError) as e:
        if isinstance(e, ArtifactError):
            raise
        raise ArtifactError("Invalid precision capture: " + str(e)) from e


def replay_precision(directory, execution=ExecutionConfig()):
    from ginseng.conditional import (
        conditional_contributions,
        prepare_conditional_arrays,
    )
    from ginseng.precision import (
        BoundedMoments,
        PrecisionConfig,
        checkpoint_interval,
        intersect_checkpoint,
    )
    from ginseng.simulate import DrawBundle, _compute_draw_id

    result, p, meta, trace = load_precision(directory)
    model = result["manifest"]
    config = PrecisionConfig(**model["config"])
    opening = meta["scalars"]["opening_cash"]
    h = p.visible_horizon
    if trace is None:
        with EvaluationContext(execution) as context:
            evaluated = context.evaluate(p, opening, 0, full=False)
            values = (evaluated.statistics.minimum_balance < 0).astype(float)
            actual_execution = evaluated.metadata
    else:
        tables = prepare_conditional_arrays(
            p.history[:, 0] - p.history[:, 1] - p.history[:, 2], p.schedule[:h], opening
        )
        bundle = DrawBundle(
            model["derived_seed"],
            h,
            p.n_paths,
            model["block_resolved"],
            model["block_clipped"],
            len(p.history),
            p.indices[:, :h],
            _compute_draw_id(p.indices[:, :h]),
            "mc",
            p.indices,
            (),
            model["block_requested"],
            trace,
        )
        values = conditional_contributions(tables, bundle)
        actual_execution = dict(backend="numpy", estimator="initial-block-cmc")
    moments = BoundedMoments()
    observed = []
    intersection = [0.0, 1.0]
    last_n = 0
    mismatches = []
    for look, n in enumerate(config.checkpoints(), 1):
        if n > len(values):
            break
        while moments.n < n:
            stop = min(n, moments.n + model["effective_chunk_size"])
            moments.update(values[moments.n : stop])
        try:
            row = intersect_checkpoint(
                checkpoint_interval(moments, look, config.confidence), intersection
            )
        except ArithmeticError:
            if result["summary"]["status"] != "numerical_failure":
                mismatches.append("intersection")
            break
        observed.append(row)
        intersection = row["interval"]
        last_n = n
        if (
            config.stop_when_precise
            and row["absolute_error_bound"] <= config.absolute_error
            and n != len(values)
        ):
            mismatches.append("missed_first_qualifying_stop")
    if moments.n < len(values):
        moments.update(values[moments.n :])
    mismatches += _differences(observed, result["checkpoints"], "checkpoints")
    checked = dict(
        cash_shortfall_probability=moments.mean,
        numerical_probability_interval=intersection,
        actual_n=moments.n,
        interval_observations=last_n,
        absolute_error_bound=max(
            abs(moments.mean - intersection[0]), abs(intersection[1] - moments.mean)
        ),
    )
    mismatches += _differences(
        checked, {k: result["summary"][k] for k in checked}, "summary"
    )
    if result["summary"]["status"] not in (
        "precision_met",
        "budget_exhausted",
        "cancelled",
        "numerical_failure",
    ):
        mismatches.append("unsupported_status")
    if result["summary"]["precision_met"] != (
        result["summary"]["status"] == "precision_met"
    ):
        mismatches.append("inconsistent_precision_status")
    if result["summary"]["precision_met"] and (
        last_n != moments.n or checked["absolute_error_bound"] > config.absolute_error
    ):
        mismatches.append("unsupported_precision_claim")
    return dict(
        match=not mismatches,
        mismatches=mismatches,
        summary=result["summary"],
        execution=actual_execution,
        scope="Exact captured futures and checkpoint arithmetic; wall-clock cancellation/deadline timing is recorded, not reproduced.",
    )


def _differences(actual, expected, path):
    """Probability arithmetic tolerance, distinct from exact input integrity."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            return [path + ".keys"]
        return [
            p
            for k in actual
            for p in _differences(actual[k], expected[k], path + "." + k)
        ]
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return [path + ".length"]
        return [
            p
            for i, (a, b) in enumerate(zip(actual, expected))
            for p in _differences(a, b, f"{path}[{i}]")
        ]
    if type(actual) is float and isinstance(expected, (int, float)):
        return [] if np.isclose(actual, expected, atol=2e-14, rtol=2e-13) else [path]
    return [] if type(actual) is type(expected) and actual == expected else [path]
