"""Checkpoint-valid empirical Bernstein intervals for IID bounded MC outputs.

Maurer & Pontil (2009), Theorem 4, applied to X and 1-X, then a
union bound over predetermined looks with delta_k = alpha/(k*(k+1)).
This interval describes numerical uncertainty under a fixed historical model.
"""

import math
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from time import perf_counter

import numpy as np

from ginseng.conditional import conditional_contributions, prepare_conditional
from ginseng.numerical import environment
from ginseng.provenance import digest
from ginseng.sampling import map_indices, prepare_history
from ginseng.simulate import DrawBundle, _compute_draw_id, cash_paths


@dataclass(frozen=True)
class PrecisionConfig:
    absolute_error: float = 0.005
    confidence: float = 0.95
    max_paths: int = 262144
    batch_size: int = 1024
    chunk_size: int | None = None
    time_limit_seconds: float | None = 30.0
    memory_budget_bytes: int = 128 * 1024**2
    stop_when_precise: bool = True

    def __post_init__(self):
        for name in ("absolute_error", "confidence"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value < 1
            ):
                raise ValueError(
                    f"{name} must be a finite number strictly between 0 and 1."
                )
        for name in ("max_paths", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 2:
                raise ValueError(f"{name} must be an integer at least 2.")
        if self.max_paths > 2**30:
            raise ValueError("max_paths exceeds the supported limit 2^30.")

        if self.chunk_size is not None and (
            type(self.chunk_size) is not int or self.chunk_size < 1
        ):
            raise ValueError("chunk_size must be a positive integer")
        if self.time_limit_seconds is not None and (
            isinstance(self.time_limit_seconds, bool)
            or not isinstance(self.time_limit_seconds, (int, float))
            or not math.isfinite(self.time_limit_seconds)
            or self.time_limit_seconds <= 0
        ):
            raise ValueError("time_limit_seconds must be finite and positive, or None")
        if (
            type(self.memory_budget_bytes) is not int
            or not 1024 <= self.memory_budget_bytes <= 512 * 1024**2
        ):
            raise ValueError("memory_budget_bytes must be between 1 KiB and 512 MiB")
        if type(self.stop_when_precise) is not bool:
            raise ValueError("stop_when_precise must be boolean")

    def checkpoints(self):
        n = min(self.batch_size, self.max_paths)
        while n < self.max_paths:
            yield n
            n = min(2 * n, self.max_paths)
        yield self.max_paths


@dataclass
class BoundedMoments:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values):
        x = np.asarray(values, dtype=float)
        if (
            x.ndim != 1
            or len(x) == 0
            or not np.all(np.isfinite(x))
            or np.any((x < 0) | (x > 1))
        ):
            raise ValueError(
                "Precision observations must be a nonempty finite vector in [0,1]."
            )
        count = len(x)
        mean = float(x.mean())
        delta = mean - self.mean
        total = self.n + count
        self.m2 += (
            float(np.sum((x - mean) ** 2)) + delta * delta * self.n * count / total
        )
        self.mean += delta * count / total
        self.n = total

    @property
    def variance(self):
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0


def checkpoint_interval(moments, look, confidence):
    if (
        type(moments.n) is not int
        or moments.n < 2
        or type(look) is not int
        or look < 1
        or isinstance(confidence, bool)
        or not 0 < confidence < 1
        or not np.isfinite([moments.mean, moments.m2]).all()
        or not 0 <= moments.mean <= 1
        or moments.m2 < 0
    ):
        raise ValueError(
            "Interval requires n>=2, a positive look and confidence in (0,1)."
        )
    # log(4/delta_k) avoids underflow when alpha or delta is small.
    log_term = (
        math.log(4) - math.log1p(-confidence) + math.log(look) + math.log(look + 1)
    )
    radius = math.sqrt(
        2 * max(0.0, moments.variance) * log_term / moments.n
    ) + 7 * log_term / (3 * (moments.n - 1))
    low = max(0.0, moments.mean - radius)
    high = min(1.0, moments.mean + radius)
    return dict(
        n=moments.n,
        look=look,
        estimate=moments.mean,
        sample_variance=moments.variance,
        interval=[low, high],
        absolute_error_bound=max(moments.mean - low, high - moments.mean),
        error_allowance=math.exp(
            math.log1p(-confidence) - math.log(look) - math.log(look + 1)
        ),
        log_error_allowance=math.log1p(-confidence)
        - math.log(look)
        - math.log(look + 1),
    )


def _bundle_from_points(points, prepared, horizon, seed, trace):
    mapped = map_indices(
        points,
        len(prepared.joint),
        prepared.resolved_length,
        return_initial_lengths=trace,
    )
    indices, lengths = mapped if trace else (mapped, None)
    visible = indices[:, :horizon]
    return DrawBundle(
        seed,
        horizon,
        len(points),
        prepared.resolved_length,
        prepared.clipped,
        len(prepared.joint),
        visible,
        _compute_draw_id(visible),
        "mc",
        indices,
        (),
        prepared.requested_length,
        np.minimum(lengths, horizon) if trace else None,
    )


def intersect_checkpoint(current, previous):
    """Intersection preserves simultaneous coverage; an empty set is diagnostic."""
    if not np.isfinite([*previous, *current["interval"], current["estimate"]]).all():
        raise ArithmeticError("Nonfinite checkpoint intersection")
    lo = max(previous[0], current["interval"][0])
    hi = min(previous[1], current["interval"][1])
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ArithmeticError("Empty or zero-width checkpoint intersection")
    return {
        **current,
        "raw_interval": current["interval"],
        "interval": [lo, hi],
        "absolute_error_bound": max(
            abs(current["estimate"] - lo), abs(hi - current["estimate"])
        ),
    }


def run_precision(
    case,
    config=None,
    *,
    estimator="path",
    sampler="mc",
    seed=42,
    replicate=0,
    horizon=None,
    material_horizon=None,
    block_length=None,
    weights=None,
    execution=None,
    cancelled=None,
    capture_path=None,
    allow_personal_capture=False,
    synthetic=False,
):
    """Fixed-model failure only. Precision stopping occurs at declared looks.

    Time/cancellation are checked cooperatively between bounded batches. On an
    interruption the last checkpoint interval (or [0,1]) remains the only
    certified interval; no fresh, unbudgeted look is invented.
    """
    from ginseng.execution import EvaluationContext, ExecutionConfig, PreparedScenario
    from ginseng.precision_stream import PrecisionStream
    from ginseng.simulate import _deterministic_daily_flow

    config = PrecisionConfig() if config is None else config
    if not isinstance(config, PrecisionConfig):
        raise ValueError("config must be a PrecisionConfig")
    if sampler != "mc":
        raise ValueError("Precision stopping supports independent mc only")
    if estimator not in ("path", "initial-block-cmc"):
        raise ValueError("Unknown precision estimator")
    if weights is not None:
        raise ValueError("Precision stopping does not support supplied stress weights")
    h = case.state.forecast_horizon if horizon is None else horizon
    material = h if material_horizon is None else material_horizon
    stream = PrecisionStream(seed, replicate, material)
    if type(h) is not int or not 1 <= h <= material:
        raise ValueError("Visible horizon must lie within material horizon")
    execution = execution or ExecutionConfig()
    execution = replace(
        execution,
        memory_budget=min(execution.memory_budget, config.memory_budget_bytes),
    )
    start = perf_counter()
    deadline = (
        start + config.time_limit_seconds
        if config.time_limit_seconds is not None
        else math.inf
    )
    prepared = prepare_history(case.state, block_length)
    daily = _deterministic_daily_flow(case.state, case.obligations, material)
    trace = estimator == "initial-block-cmc"
    fixed_bytes = (
        prepared.joint.nbytes
        + daily.nbytes
        + (48 * len(prepared.joint) * h if trace else 0)
    )
    per_path_bytes = 8 * (16 * material * execution.workers + 16)
    capacity = max(0, (execution.memory_budget - fixed_bytes) // per_path_bytes)
    chunk = min(config.chunk_size or config.batch_size, capacity)
    if capture_path is not None:
        if not synthetic and not allow_personal_capture:
            raise ValueError("Personal precision capture requires explicit consent")
        if config.max_paths * (material + 1) * 8 > execution.memory_budget // 3:
            raise ValueError(
                "Requested precision capture exceeds its retained-input budget"
            )
        # Reserve capture storage as part of the caller budget.
        capacity = max(
            0,
            (
                execution.memory_budget
                - fixed_bytes
                - config.max_paths * (material + 1) * 8 * 3
            )
            // per_path_bytes,
        )
        chunk = min(chunk, capacity)
    moments = BoundedMoments()
    history = []
    intersection = [0.0, 1.0]
    last_n = 0
    index_hash, trace_hash = sha256(), sha256()
    indices = []
    traces = []
    status = "budget_exhausted"
    reason = "max_paths_reached"
    tables = None
    actual_execution = None
    if chunk < 1:
        reason = "memory_budget_exhausted"
    elif cancelled is not None and cancelled():
        status = "cancelled"
        reason = "cancelled"
    elif perf_counter() >= deadline:
        reason = "time_budget_exhausted"
    else:
        tables = (
            prepare_conditional(prepared, case.state, case.obligations, h)
            if trace
            else None
        )
        for look, checkpoint in enumerate(config.checkpoints(), 1):
            interrupted = False
            while moments.n < checkpoint:
                if cancelled is not None and cancelled():
                    status = "cancelled"
                    reason = "cancelled"
                    interrupted = True
                    break
                if perf_counter() >= deadline:
                    reason = "time_budget_exhausted"
                    interrupted = True
                    break
                count = min(chunk, checkpoint - moments.n)
                points = stream.points(moments.n, count)
                bundle = _bundle_from_points(points, prepared, h, stream.seed, trace)
                index_hash.update(bundle.index_matrix.tobytes())
                if trace:
                    actual_execution = dict(
                        backend="numpy",
                        workers=1,
                        native=None,
                        estimator="initial-block-cmc",
                    )
                    values = conditional_contributions(tables, bundle)
                    trace_hash.update(bundle.initial_block_lengths.tobytes())
                else:
                    # Request-local contexts prevent accumulating old chunks in a cache.
                    with EvaluationContext(execution) as context:
                        x = cash_paths(
                            case.state,
                            bundle,
                            case.obligations,
                            prepared_history=prepared.joint,
                        )
                    actual_execution = dict(
                        backend=context.backend,
                        workers=execution.workers,
                        native=context.native,
                    )
                    if not np.isfinite(x).all():
                        raise ValueError("Precision cash paths overflowed float64")
                    values = (case.state.immediate_funding + x.min(axis=1) < 0).astype(
                        float
                    )
                if capture_path is not None:
                    indices.append(bundle.material_indices)
                    if trace:
                        traces.append(bundle.initial_block_lengths)
                moments.update(values)
            if interrupted:
                break
            try:
                current = intersect_checkpoint(
                    checkpoint_interval(moments, look, config.confidence), intersection
                )
            except ArithmeticError:
                status = "numerical_failure"
                reason = "invalid_interval_intersection"
                break
            intersection = current["interval"]
            last_n = moments.n
            history.append(current)
            if (
                current["absolute_error_bound"] <= config.absolute_error
                and config.stop_when_precise
            ):
                status = "precision_met"
                reason = "precision_reached"
                break
        if (
            not config.stop_when_precise
            and moments.n == config.max_paths
            and status != "numerical_failure"
        ):
            reason = "fixed_budget_completed"
            if history[-1]["absolute_error_bound"] <= config.absolute_error:
                status = "precision_met"
    mean = moments.mean if moments.n else None
    error = (
        max(abs(mean - intersection[0]), abs(intersection[1] - mean))
        if mean is not None
        else None
    )
    summary = dict(
        cash_shortfall_probability=mean,
        numerical_probability_interval=intersection,
        absolute_error_bound=error,
        requested_absolute_error=config.absolute_error,
        confidence=config.confidence,
        precision_met=status == "precision_met",
        stop_reason=reason,
        status=status,
        actual_n=moments.n,
        interval_observations=last_n,
        method="empirical-bernstein-alpha-spending-intersection-v2",
    )
    model_identity = digest(
        dict(
            history=prepared.joint.tolist(),
            daily=daily[:h].tolist(),
            opening_cash=case.state.immediate_funding,
            block_length=prepared.resolved_length,
            horizon=h,
            law="stationary-bootstrap uniform starts v1",
            failure_boundary="strict < 0",
        )
    )
    metadata = dict(
        version=2,
        interval_method=summary["method"],
        checkpoint_schedule=list(config.checkpoints()),
        error_spending="alpha/(k*(k+1)), k=1,2,...",
        scope="numerical uncertainty of fixed historical model; not historical-data uncertainty or forecast calibration",
        config=asdict(config),
        sampler="mc",
        estimator=estimator,
        metric_estimators={"cash_shortfall_probability": estimator},
        root_seed=seed,
        replicate=replicate,
        domain=400,
        derived_seed=stream.seed,
        seed_scheme="SeedSequence([root, 400, 1, replicate]).uint64",
        bit_generator="PCG64",
        mapping_version=1,
        dimension=stream.dimension,
        material_horizon=material,
        visible_horizon=h,
        stream_layout=stream.identity(),
        block_requested=prepared.requested_length,
        block_resolved=prepared.resolved_length,
        block_clipped=prepared.clipped,
        block_resolution=prepared.resolution,
        input_hash=digest(asdict(case)),
        model_identity=model_identity,
        synthetic_data_seed=case.data_seed,
        index_hash=index_hash.hexdigest(),
        initial_block_trace_hash=trace_hash.hexdigest() if trace else None,
        conditional_table_identity=tables.identity if tables else None,
        execution=asdict(execution),
        actual_execution=actual_execution,
        effective_chunk_size=chunk,
        unsupported_metrics=[
            "reserve_quantile",
            "CVaR",
            "modeled_cost",
            "importance_weighted_probability",
        ],
        resource_boundary="Cooperative deadline between batches; preparation and a running batch are not preempted.",
        environment=environment(),
    )
    summary["model_identity"] = model_identity
    metadata["result_hash"] = digest(
        dict(manifest=metadata, summary=summary, checkpoints=history)
    )
    result = dict(
        summary=summary,
        checkpoints=history,
        manifest=metadata,
        core_seconds=perf_counter() - start,
    )
    if capture_path is not None:
        if not indices:
            result["capture"] = dict(status="unavailable", reason="no_observations")
            return result
        from ginseng.precision_artifact import capture_precision

        p = PreparedScenario(
            prepared.joint,
            np.concatenate(indices),
            np.empty((0, 0)),
            daily,
            h,
            material,
        )
        capture_precision(
            capture_path,
            p,
            result,
            case.state.immediate_funding,
            np.concatenate(traces) if traces else None,
            execution,
        )
    return result


def estimate_failure(
    case, config=None, *, metric="cash_shortfall_probability", **kwargs
):
    """Opt-in public API: unsupported statistical modes are explicit results."""
    config = config or PrecisionConfig()
    unsupported = (
        metric != "cash_shortfall_probability"
        or kwargs.get("sampler", "mc") != "mc"
        or kwargs.get("estimator", "path") not in ("path", "initial-block-cmc")
        or kwargs.get("weights") is not None
    )
    if unsupported:
        identity = digest(asdict(case))
        return dict(
            summary=dict(
                status="unsupported_estimator",
                stop_reason="unsupported_metric_or_sampling_law",
                precision_met=False,
                actual_n=0,
                interval_observations=0,
                cash_shortfall_probability=None,
                numerical_probability_interval=[0.0, 1.0],
                absolute_error_bound=None,
                requested_absolute_error=config.absolute_error,
                confidence=config.confidence,
                model_identity=identity,
                method=None,
            ),
            checkpoints=[],
            manifest=dict(
                input_hash=identity,
                model_identity=identity,
                interval_method=None,
                supported="unweighted ordinary MC path indicators or verified independent initial-block CMC only",
            ),
        )
    return run_precision(case, config, **kwargs)
