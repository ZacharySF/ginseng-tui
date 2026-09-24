"""Experimental initial-block conditioning for unweighted historical failure.

Preparation is an explicit immutable value, never a process-global cache.
For each k, sorted totals include only starts whose initial minimum survives.
The denominator is always the entire history, including ineligible starts.
"""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from ginseng.sampling import MAX_ARRAY_BYTES
from ginseng.simulate import DrawBundle, _deterministic_daily_flow

VERSION = 1
ESTIMATOR = "initial-block-cmc"


def _immutable(a):
    a = np.asarray(a, dtype=float)
    return np.frombuffer(a.tobytes(), dtype=float).reshape(a.shape)


def _inputs(prepared, state, obligations, horizon):
    net = prepared.joint[:, 0] - prepared.joint[:, 1] - prepared.joint[:, 2]
    daily = _deterministic_daily_flow(state, obligations, horizon)
    cash = state.immediate_funding
    if (
        not np.all(np.isfinite(net))
        or not np.all(np.isfinite(daily))
        or not np.isfinite(cash)
    ):
        raise ValueError("Conditional inputs must be finite.")
    key = sha256()
    for a in (net, daily, np.array([cash])):
        key.update(np.asarray(a.shape, dtype="<i8").tobytes())
        key.update(np.asarray(a, dtype="<f8").tobytes())
    return net, daily, cash, key.hexdigest()


@dataclass(frozen=True)
class ConditionalTables:
    identity: str
    net: np.ndarray
    daily: np.ndarray
    cash: float
    totals: np.ndarray
    minima: np.ndarray
    eligible_totals: tuple


def prepare_conditional(prepared, state, obligations=(), horizon=None):
    h = state.forecast_horizon if horizon is None else horizon
    if not isinstance(h, int) or h < 1:
        raise ValueError("Horizon must be a positive integer.")
    net, daily, cash, _ = _inputs(prepared, state, obligations, h)
    return prepare_conditional_arrays(net, daily, cash)


def prepare_conditional_arrays(net, daily, cash):
    """Same table construction from exact captured primitive inputs."""
    net, daily = np.asarray(net, dtype=float), np.asarray(daily, dtype=float)
    if (
        net.ndim != 1
        or daily.ndim != 1
        or min(len(net), len(daily)) < 1
        or not np.isfinite(net).all()
        or not np.isfinite(daily).all()
        or not np.isfinite(cash)
    ):
        raise ValueError("Finite nonempty conditional inputs required")
    key = sha256()
    for a in (net, daily, np.array([cash])):
        key.update(np.asarray(a.shape, dtype="<i8").tobytes())
        key.update(np.asarray(a, dtype="<f8").tobytes())
    identity = key.hexdigest()
    h, n = len(daily), len(net)
    if 48 * n * h > MAX_ARRAY_BYTES:
        raise ValueError(
            "Conditional preparation exceeds 512 MiB estimated array budget."
        )
    totals = np.empty((h, n))
    minima = np.empty_like(totals)
    current = np.zeros(n)
    low = np.full(n, np.inf)
    starts = np.arange(n)
    eligible = []
    with np.errstate(over="raise", invalid="raise"):
        try:
            for t in range(h):
                current = current + (net[(starts + t) % n] + daily[t])
                low = np.minimum(low, current)
                totals[t], minima[t] = current, low
                eligible.append(_immutable(np.sort(current[cash + low >= 0])))
        except FloatingPointError as exc:
            raise ValueError("Conditional cumulative flows overflow float64.") from exc
    return ConditionalTables(
        identity,
        _immutable(net),
        _immutable(daily),
        cash,
        _immutable(totals),
        _immutable(minima),
        tuple(eligible),
    )


def _validate(tables, bundle, weights):
    if weights is not None:
        raise ValueError(
            "Initial-block CMC does not support stress weights, including uniform supplied weights."
        )
    if not isinstance(bundle, DrawBundle) or bundle.sampler not in ("mc", "sobol"):
        raise ValueError(
            "Initial-block CMC requires an unweighted historical mc/sobol DrawBundle."
        )
    k = bundle.initial_block_lengths
    h, n = len(tables.daily), len(tables.net)
    if (
        bundle.horizon_days != h
        or bundle.history_length != n
        or k is None
        or k.shape != (bundle.n_paths,)
        or np.any((k < 1) | (k > h))
        or bundle.index_matrix.shape != (bundle.n_paths, h)
        or np.any((bundle.index_matrix < 0) | (bundle.index_matrix >= n))
    ):
        raise ValueError(
            "Conditional tables require matching history/horizon and explicit initial-block traces."
        )
    return k


def slow_contributions(tables, bundle, *, weights=None):
    """Independent all-start path reduction; deliberately bounded to small checks."""
    lengths = _validate(tables, bundle, weights)
    if bundle.n_paths * len(tables.net) * bundle.horizon_days > 10_000_000:
        raise ValueError("Slow reference is limited to ten million path-days.")
    out = []
    for row, k in zip(bundle.index_matrix, lengths):
        out.append(_direct_row(tables, row, k))
    return np.array(out)


def _direct_row(tables, row, k):
    failed = 0
    for start in range(len(tables.net)):
        idx = row.copy()
        idx[:k] = (start + np.arange(k)) % len(tables.net)
        x = np.cumsum(tables.net[idx] + tables.daily)
        failed += tables.cash + x.min() < 0
    return failed / len(tables.net)


def conditional_contributions(tables, bundle, *, weights=None):
    lengths = _validate(tables, bundle, weights)
    h, n = len(tables.daily), len(tables.net)
    out = np.empty(bundle.n_paths)
    # Reassociation at the block boundary can change a last-bit comparison.
    # Close boundaries use the ordinary full-path arithmetic as a fallback.
    scale = abs(tables.cash) + h * (
        np.max(np.abs(tables.net)) + np.max(np.abs(tables.daily))
    )
    guard = 32 * np.finfo(float).eps * max(1.0, scale) * h
    for k in np.unique(lengths):
        rows = np.flatnonzero(lengths == k)
        eligible = tables.eligible_totals[k - 1]
        if k == h:
            out[rows] = 1 - len(eligible) / n
            continue
        daily = tables.net[bundle.index_matrix[rows, k:]] + tables.daily[k:]
        remainder = np.cumsum(daily, axis=1).min(axis=1)
        if not np.all(np.isfinite(remainder)):
            raise ValueError("Conditional remainder overflowed float64.")
        threshold = -tables.cash - remainder
        counts = len(eligible) - np.searchsorted(eligible, threshold, side="left")
        out[rows] = 1 - counts / n
        near = np.searchsorted(
            eligible, threshold + guard, side="right"
        ) > np.searchsorted(eligible, threshold - guard, side="left")
        for row in rows[near]:
            out[row] = _direct_row(tables, bundle.index_matrix[row], k)
    return out


def failure_probability(
    prepared, state, bundle, obligations=(), *, tables=None, weights=None
):
    """Validated public boundary; rejects stale explicitly reused preparation."""
    if not isinstance(bundle, DrawBundle):
        raise ValueError(
            "Initial-block CMC does not support prospective/scheduled paths."
        )
    if tables is None:
        tables = prepare_conditional(prepared, state, obligations, bundle.horizon_days)
    else:
        *_, identity = _inputs(prepared, state, obligations, bundle.horizon_days)
        if identity != tables.identity:
            raise ValueError(
                "Stale conditional tables: history, bills, horizon or opening cash changed."
            )
    return float(conditional_contributions(tables, bundle, weights=weights).mean())
