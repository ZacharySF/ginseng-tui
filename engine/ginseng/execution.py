"""Owned, request-scoped cash calculation boundary. No sampling or services.

Arrays are native little-endian f64/i64, C contiguous and aligned. Preparation
makes explicit immutable bytes-backed snapshots; strict native calls never cast.
"""

from __future__ import annotations

import importlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cached_property, wraps
from hashlib import sha256
from pathlib import Path

import numpy as np

from ginseng.provenance import digest

API_VERSION = 1
_CURRENT = ContextVar("ginseng_execution", default=None)


class ResourceLimitError(ValueError):
    """Requested resident arrays and scratch exceed the execution budget."""


@dataclass(frozen=True)
class ExecutionConfig:
    backend: str = "auto"
    workers: int = 1
    block_size: int = 1024
    memory_budget: int = 512 * 1024**2

    def __post_init__(self):
        if self.backend not in ("auto", "numpy", "native"):
            raise ValueError("Backend must be auto, numpy or native.")
        if (
            any(
                type(x) is not int or x < 1
                for x in (self.workers, self.block_size, self.memory_budget)
            )
            or self.workers > 4
        ):
            raise ValueError(
                "Positive integer budget/block size and 1–4 workers required."
            )


def snapshot(value, dtype="<f8"):
    a = np.asarray(value, dtype=dtype)
    owner = a
    while isinstance(owner, np.ndarray):
        owner = owner.base
    if isinstance(owner, bytes) and a.flags.c_contiguous and a.flags.aligned:
        return a
    return np.frombuffer(a.tobytes(order="C"), dtype=dtype).reshape(a.shape)


def array_id(a):
    return digest(
        (
            a.dtype.str,
            a.shape,
            sha256(
                memoryview(np.ascontiguousarray(a)).cast("B") if a.size else b""
            ).hexdigest(),
        )
    )


def native_info(required=False):
    try:
        module = importlib.import_module("ginseng_native._core")
    except ModuleNotFoundError as exc:
        if exc.name not in ("ginseng_native", "ginseng_native._core"):
            raise
        if required:
            raise ValueError(
                "Native backend unavailable; install the optional native wheel."
            ) from exc
        return {"available": False}
    info = dict(module.build_info())
    if info["api_version"] != API_VERSION:
        raise ValueError("Incompatible native API version.")
    source = Path(__file__).resolve().parents[2] / "native/src/core.cpp"
    if (
        source.exists()
        and sha256(source.read_bytes()).hexdigest() != info["source_hash"]
    ):
        raise ValueError("Stale native extension: rebuild the optional wheel.")
    kernel = source.with_name("kernels.hpp")
    if (
        kernel.exists()
        and sha256(kernel.read_bytes()).hexdigest() != info["kernel_hash"]
    ):
        raise ValueError("Stale native kernels: rebuild the optional wheel.")
    for filename, key in (
        ("CMakeLists.txt", "cmake_hash"),
        ("pyproject.toml", "package_hash"),
    ):
        build_file = source.parents[1] / filename
        if (
            build_file.exists()
            and sha256(build_file.read_bytes()).hexdigest() != info[key]
        ):
            raise ValueError(
                "Stale native build configuration: rebuild the optional wheel."
            )
    path = Path(module.__file__).resolve()
    return dict(
        info,
        available=True,
        path=str(path),
        binary_hash=sha256(path.read_bytes()).hexdigest(),
    )


@dataclass(frozen=True)
class PreparedScenario:
    history: np.ndarray
    indices: np.ndarray
    direct: np.ndarray
    schedule: np.ndarray
    visible_horizon: int
    material_horizon: int
    source: str = "history"

    def __post_init__(self):
        if any(
            type(value) is not int or value < 1
            for value in (self.visible_horizon, self.material_horizon)
        ):
            raise ValueError("Visible and material horizons must be positive integers.")
        # Reject excessive supplied storage before making conversion snapshots.
        if (
            sum(
                np.asarray(getattr(self, name)).size * 8
                for name in ("history", "indices", "direct", "schedule")
            )
            > 512 * 1024**2
        ):
            raise ResourceLimitError(
                "Prepared inputs exceed the supported 512 MiB boundary."
            )
        for name in ("history", "indices", "direct", "schedule"):
            raw = np.asarray(getattr(self, name))
            if name == "indices" and raw.dtype != np.dtype("int64"):
                raise ValueError(
                    "Indices must be native int64; no implicit index conversion."
                )
            a = snapshot(raw, "<i8" if name == "indices" else "<f8")
            if not np.all(np.isfinite(a)):
                raise ValueError("Prepared arrays must be finite.")
            object.__setattr__(self, name, a)
        if (
            self.history.ndim != 2
            or self.history.shape[1] != 3
            or self.indices.ndim != 2
            or self.direct.ndim != 2
        ):
            raise ValueError("Prepared arrays have invalid dimensions.")
        if self.source not in ("history", "direct"):
            raise ValueError("Unknown prepared source.")
        a = self.indices if self.source == "history" else self.direct
        if (
            a.ndim != 2
            or min(a.shape) < 1
            or self.material_horizon != a.shape[1]
            or not 1 <= self.visible_horizon <= self.material_horizon
        ):
            raise ValueError("Invalid visible/material horizon or path shape.")
        if self.schedule.shape != (self.material_horizon,):
            raise ValueError("Schedule must span the material horizon.")
        if self.source == "history" and (
            self.history.ndim != 2
            or self.history.shape[1] != 3
            or len(self.history) < 1
            or np.any(self.indices < 0)
            or np.any(self.indices >= len(self.history))
        ):
            raise ValueError("Invalid joint history or out-of-range index.")

    @property
    def n_paths(self):
        return len(self.indices if self.source == "history" else self.direct)

    @property
    def nbytes(self):
        return sum(
            getattr(self, k).nbytes
            for k in ("history", "indices", "direct", "schedule")
        )

    @cached_property
    def _identities(self):
        return tuple(
            dict(
                preparation=array_id(self.history),
                draws=array_id(
                    self.indices if self.source == "history" else self.direct
                ),
                schedule=array_id(self.schedule),
                horizons=digest(
                    (self.visible_horizon, self.material_horizon, self.source)
                ),
            ).items()
        )

    @property
    def identities(self):
        return dict(self._identities)


@dataclass(frozen=True)
class PathStatistics:
    minima: np.ndarray
    terminal: np.ndarray
    minimum_balance: np.ndarray
    maximum_deficit: np.ndarray
    buffer_dollar_days: np.ndarray
    overdraft_dollar_days: np.ndarray


@dataclass(frozen=True)
class ExecutionResult:
    paths: np.ndarray | None
    statistics: PathStatistics
    metadata: dict


def numpy_block(p, start, stop, opening, buffer, full):
    h = p.visible_horizon
    if p.source == "history":
        idx = p.indices[start:stop, :h]
        a, b, c = p.history.T
        daily = a[idx] - b[idx] - c[idx] + p.schedule[:h]
    else:
        daily = p.direct[start:stop, :h] - p.schedule[:h]
    x = np.cumsum(daily, axis=1)
    balances = opening + x
    minimum = x.min(axis=1)
    low = balances.min(axis=1)
    stats = np.column_stack(
        (
            minimum,
            x[:, -1],
            low,
            np.maximum(0, -low),
            np.maximum(0, buffer - balances).sum(axis=1),
            np.maximum(0, -balances).sum(axis=1),
        )
    )
    return x if full else None, stats


class EvaluationContext:
    def __init__(self, config=ExecutionConfig()):
        self.config = config
        self.cache = {}
        self.resident_bytes = 0
        self.counters = Counter()
        self.pool = None
        self.closed = False
        self.entered = False
        self.backend = "numpy" if config.backend == "auto" else config.backend
        self.reason = (
            "Conservative default; explicit native available for measured workloads."
            if config.backend == "auto"
            else "Explicit selection."
        )
        self.native = native_info(True) if self.backend == "native" else None

    def __enter__(self):
        if self.closed or self.entered:
            raise ValueError("Context is closed or already active.")
        self.entered = True
        self.token = _CURRENT.set(self)
        return self

    def __exit__(self, *exc):
        try:
            if self.pool:
                self.pool.shutdown(wait=True, cancel_futures=True)
            self.cache.clear()
            self.resident_bytes = 0
            self.closed = True
            self.entered = False
        finally:
            _CURRENT.reset(self.token)

    def check(self, extra):
        if self.closed:
            raise ValueError("Context is closed.")
        if self.resident_bytes + extra > self.config.memory_budget:
            raise ResourceLimitError(
                "Execution arrays and scratch exceed configured memory budget."
            )

    def remember(self, key, create, size=lambda x: x.nbytes):
        if key in self.cache:
            self.counters["cache_hits"] += 1
            return self.cache[key]
        value = create()
        amount = size(value)
        self.check(amount)
        self.cache[key] = value
        self.resident_bytes += amount
        return value

    def evaluate(self, p, opening=0.0, buffer=0.0, *, full=True):
        if not np.all(np.isfinite([opening, buffer])):
            raise ValueError("Opening cash and buffer must be finite.")
        n, h = p.n_paths, p.visible_horizon
        # Includes owned inputs, full output, O(N) summaries, bounded concurrent
        # NumPy scratch (native is smaller) and immutable publication copies.
        scratch = self.config.workers * min(n, self.config.block_size) * h * 8 * 10
        output = n * (6 + (h if full else 0)) * 8
        key = ("evaluation", digest(p.identities), opening, buffer, full, self.backend)
        if key in self.cache:
            self.counters["cache_hits"] += 1
            return self.cache[key]
        self.check(p.nbytes + 2 * output + scratch)
        path_key = ("paths", digest(p.identities), self.backend)
        if full and path_key in self.cache:
            matrix = self.cache[path_key]
            base = self.cache[("path_statistics", digest(p.identities), self.backend)]
            low = opening + base.minima
            below, over = np.empty(n), np.empty(n)
            for a in range(0, n, self.config.block_size):
                b = min(n, a + self.config.block_size)
                balances = opening + matrix[a:b]
                below[a:b] = np.maximum(0, buffer - balances).sum(axis=1)
                over[a:b] = np.maximum(0, -balances).sum(axis=1)
            stats = PathStatistics(
                base.minima,
                base.terminal,
                snapshot(low),
                snapshot(np.maximum(0, -low)),
                snapshot(below),
                snapshot(over),
            )
            result = ExecutionResult(
                matrix,
                stats,
                dict(
                    backend=self.backend,
                    reason=self.reason,
                    workers=self.config.workers,
                    block_size=self.config.block_size,
                    input_bytes=p.nbytes,
                    output_bytes=n * 6 * 8,
                    scratch_bound_bytes=scratch,
                    native=self.native,
                ),
            )
            self.cache[key] = result
            self.resident_bytes += n * 4 * 8
            self.counters["cache_hits"] += 1
            return result
        matrix = np.empty((n, h)) if full else None
        summaries = np.empty((n, 6))
        module = (
            importlib.import_module("ginseng_native._core")
            if self.backend == "native"
            else None
        )

        def block(bounds):
            a, b = bounds
            if module:
                return module.evaluate(
                    p.history,
                    p.indices,
                    p.direct,
                    p.schedule,
                    h,
                    a,
                    b,
                    opening,
                    buffer,
                    full,
                    p.source == "history",
                )
            return numpy_block(p, a, b, opening, buffer, full)

        ranges = [
            (a, min(n, a + self.config.block_size))
            for a in range(0, n, self.config.block_size)
        ]
        # At most workers futures and block results exist at once; executor.map
        # would eagerly queue every block on Python 3.12.
        if self.config.workers > 1 and self.pool is None:
            self.pool = ThreadPoolExecutor(
                max_workers=self.config.workers, thread_name_prefix="ginseng"
            )
        for offset in range(0, len(ranges), self.config.workers):
            batch = ranges[offset : offset + self.config.workers]
            futures = [self.pool.submit(block, r) for r in batch] if self.pool else None
            try:
                for i, (a, b) in enumerate(batch):
                    x, s = futures[i].result() if futures else block((a, b))
                    if full:
                        matrix[a:b] = x
                    summaries[a:b] = s
            except BaseException:
                if futures:
                    for f in futures:
                        f.cancel()
                    for f in futures:
                        if not f.cancelled():
                            try:
                                f.result()
                            except BaseException:
                                pass
                raise
        if not np.all(np.isfinite(summaries)) or (
            full and not np.all(np.isfinite(matrix))
        ):
            raise ValueError("Non-finite accumulated result.")
        self.counters["path_materializations"] += int(full)
        self.counters["row_reductions"] += 1
        result = ExecutionResult(
            snapshot(matrix) if full else None,
            PathStatistics(*(snapshot(summaries[:, i]) for i in range(6))),
            dict(
                backend=self.backend,
                reason=self.reason,
                workers=self.config.workers,
                block_size=self.config.block_size,
                input_bytes=p.nbytes,
                output_bytes=output,
                scratch_bound_bytes=scratch,
                native=self.native,
            ),
        )
        self.cache[key] = result
        if full:
            self.cache[("path_statistics", digest(p.identities), self.backend)] = (
                result.statistics
            )
            self.publish_paths(p, result.paths)
        self.resident_bytes += output
        return result

    def weights(self, n, weights=None):
        from ginseng.risk import _NormalizedWeights, probabilities

        if (
            isinstance(weights, _NormalizedWeights)
            and probabilities(n, weights) is weights
        ):
            return weights
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("At least one scenario is required.")
        if weights is not None:
            raw = np.asarray(weights)
            if raw.shape != (n,):
                raise ValueError("Weights must match the path count.")
            self.check(n * 8)  # A strided vector may need a contiguous hash copy.
        key = ("weights", n, None if weights is None else array_id(raw))
        if key not in self.cache:
            self.check(n * 8 * 3)

        def create():
            self.counters["weight_normalizations"] += 1
            value = snapshot(probabilities(n, weights)).view(_NormalizedWeights)
            value._normalization_length = n
            return value

        return self.remember(key, create)

    def publish_paths(self, p, matrix):
        # Alias only: allocation is already charged to the evaluation entry.
        self.cache[("paths", digest(p.identities), self.backend)] = matrix

    def cash_paths(self, p):
        key = ("paths", digest(p.identities), self.backend)
        if key in self.cache:
            self.counters["cache_hits"] += 1
            return self.cache[key]
        result = self.evaluate(p)
        self.publish_paths(p, result.paths)
        return result.paths

    def quantiles(self, values, qs, weights=None):
        from ginseng.risk import quantiles

        raw_x, raw_q = np.asarray(values), np.asarray(qs)
        if raw_x.ndim != 1 or raw_q.ndim != 1 or not len(raw_x):
            raise ValueError(
                "Quantiles require a nonempty value vector and query vector."
            )
        self.check(raw_x.size * 128 + raw_q.size * 32)
        x, q = snapshot(raw_x), snapshot(raw_q)
        w = self.weights(len(x), weights)
        self.counters["orderings"] += 1
        if self.backend == "native":
            return importlib.import_module("ginseng_native._core").quantiles(x, w, q)
        return quantiles(x, q, w)

    def charts(self, matrix, opening=0.0, weights=None):
        from ginseng.risk import quantiles

        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or min(matrix.shape) < 1 or not np.isfinite(opening):
            raise ValueError(
                "Charts require nonempty paths-by-days values and finite opening cash."
            )
        if not matrix.flags.c_contiguous:
            self.check(matrix.nbytes)
        w = self.weights(len(matrix), weights)
        key = ("charts", array_id(matrix), float(opening), array_id(w), self.backend)
        if key in self.cache:
            self.counters["cache_hits"] += 1
            return {name: list(values) for name, values in self.cache[key].items()}
        self.check(matrix.size * 8 + self.config.workers * len(matrix) * 128)
        self.counters["orderings"] += matrix.shape[1]
        if self.backend == "native":
            module = importlib.import_module("ginseng_native._core")
            values = module.charts(
                snapshot(matrix), snapshot(w), snapshot([0.1, 0.5, 0.9]), opening
            )
        else:
            values = np.array(
                [quantiles(column + opening, [0.1, 0.5, 0.9], w) for column in matrix.T]
            ).T
        result = {
            f"p{int(q * 100)}": values[i].tolist()
            for i, q in enumerate((0.1, 0.5, 0.9))
        }
        self.cache[key] = {name: tuple(items) for name, items in result.items()}
        self.resident_bytes += values.nbytes
        return result


def current_context():
    return _CURRENT.get()


def execution_scope(function):
    """Optional explicit context, otherwise one context for this synchronous call."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        context = kwargs.pop("context", None)
        if current_context() is not None and (
            context is None or context is current_context()
        ):
            return function(*args, **kwargs)
        with context or EvaluationContext():
            return function(*args, **kwargs)

    return wrapped


def prepare_scenario(state, bundle, obligations=(), prepared_history=None):
    from ginseng.simulate import (
        PathBundle,
        _deterministic_daily_flow,
        _future_obligation_daily,
        _joint_history,
    )

    if isinstance(bundle, PathBundle):
        schedule = _future_obligation_daily(obligations, bundle.available_horizon_days)
        key = (
            "prepared",
            "direct",
            array_id(bundle.daily_cash_flows),
            array_id(schedule),
            bundle.horizon_days,
        )
        create = lambda: PreparedScenario(
            np.empty((0, 3)),
            np.empty((0, 0), dtype=np.int64),
            bundle.daily_cash_flows,
            _future_obligation_daily(obligations, bundle.available_horizon_days),
            bundle.horizon_days,
            bundle.available_horizon_days,
            "direct",
        )
        ctx = current_context()
        if ctx:
            if key not in ctx.cache:
                ctx.check(bundle.daily_cash_flows.nbytes + schedule.nbytes)
            return ctx.remember(key, create)
        return create()
    indices = (
        bundle.material_indices
        if bundle.material_indices is not None
        else bundle.index_matrix
    )
    history = (
        _joint_history(state).to_numpy()
        if prepared_history is None
        else prepared_history
    )
    schedule = _deterministic_daily_flow(state, obligations, indices.shape[1])
    key = (
        "prepared",
        "history",
        array_id(history),
        array_id(indices),
        array_id(schedule),
        bundle.horizon_days,
    )
    create = lambda: PreparedScenario(
        history,
        indices,
        np.empty((0, 0)),
        schedule,
        bundle.horizon_days,
        indices.shape[1],
    )
    ctx = current_context()
    if ctx:
        if key not in ctx.cache:
            ctx.check(history.nbytes + indices.nbytes + schedule.nbytes)
        return ctx.remember(key, create)
    return create()
