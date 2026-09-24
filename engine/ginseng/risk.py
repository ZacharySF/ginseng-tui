"""Empirical probabilities and exact fractional tails, including tied losses."""

from __future__ import annotations

import hashlib
import numpy as np

# A micro-dollar absorbs solver roundoff without concealing a cent-sized gap.
MONEY_TOLERANCE = 1e-6
PROBABILITY_TOLERANCE = 1e-12


def balance_risk(balances, buffer, q, weights=None) -> dict:
    w = probabilities(len(balances), weights)
    minima = np.min(balances, axis=1)
    return {
        "cash_shortfall_probability": float(w[minima < -MONEY_TOLERANCE].sum() / w.sum()),
        "buffer_breach_probability": float(w[minima < buffer - MONEY_TOLERANCE].sum() / w.sum()),
        "dollar_days_below_buffer": float(w @ np.maximum(0, buffer - balances).sum(axis=1)),
        "tail_deficit": cvar(np.maximum(0, buffer - minima), q, w),
    }


class _NormalizedWeights(np.ndarray):
    """Internal immutable weights validated and owned by an execution context."""

    def __array_finalize__(self, original):
        self._normalization_length = getattr(original, '_normalization_length', None)


def probabilities(n: int, weights: np.ndarray | None = None) -> np.ndarray:
    if isinstance(weights, _NormalizedWeights) and weights.shape == (n,) and weights._normalization_length == n:
        owner = weights
        while isinstance(owner, np.ndarray):
            owner = owner.base
        if isinstance(owner, bytes):
            return weights
    if n < 1:
        raise ValueError("At least one scenario is required.")
    w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w < 0) or not np.isfinite(w.sum()) or w.sum() <= 0:
        raise ValueError("Scenario weights must be finite, nonnegative, and have positive mass.")
    total = float(w.sum())
    # Re-normalizing an already normalized vector can alternate its last
    # floating-point bit, giving identical comparisons different hashes.
    return w.copy() if abs(total - 1.0) <= 1e-14 else w / total


def weight_hash(weights: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(weights, dtype="<f8").tobytes()).hexdigest()


def quantiles(values: np.ndarray, qs, weights: np.ndarray | None = None) -> np.ndarray:
    """Inverse empirical CDF: one stable ordering for every requested query.

    Preserve float64-rounded long-double CDF boundaries, including tiny tails.
    No interpolation or uniform-weight shortcut is used.
    """
    x = np.asarray(values, dtype=float)
    queries = np.asarray(qs, dtype=float)
    if (x.ndim != 1 or not np.all(np.isfinite(x)) or queries.ndim != 1
            or not np.all(np.isfinite(queries)) or np.any((queries < 0) | (queries > 1))):
        raise ValueError("Quantiles require finite vectors and qs in [0, 1].")
    w = probabilities(len(x), weights)
    positive = w > 0
    order = np.argsort(x[positive], kind="stable")
    positive_x, positive_w = x[positive][order], w[positive][order]
    cumulative = np.cumsum(positive_w.astype(np.longdouble))
    cumulative = (cumulative / cumulative[-1]).astype(float)
    indices = np.minimum(np.searchsorted(cumulative, queries), len(order) - 1)
    indices[queries == 1] = len(order) - 1
    return positive_x[indices]


def quantile(values: np.ndarray, q: float, weights: np.ndarray | None = None) -> float:
    """Smallest observed value whose cumulative probability reaches q."""
    return float(quantiles(values, [q], weights)[0])


def tail_probabilities(values: np.ndarray, q: float, weights: np.ndarray | None = None) -> np.ndarray:
    """Normalize exactly 1-q tail mass; split boundary ties proportionally.

    At q=1 this is the empirical maximum, with tied maxima sharing mass.
    Ties may give a tail ENS above (1-q)*n; that is intentional, and avoids
    selecting arbitrary scenario indices to break an atom.
    """
    x = np.asarray(values, dtype=float)
    w = probabilities(len(x), weights)
    cutoff = quantile(x, q, w)
    if q == 1:
        tail = w * (x == cutoff)
    else:
        above = x > cutoff
        tail = w * above
        boundary = x == cutoff
        remaining = max(0.0, 1.0 - q - float(tail.sum()))
        tail += w * boundary * (remaining / float(w[boundary].sum()))
    return probabilities(len(x), tail)


def cvar(values: np.ndarray, q: float, weights: np.ndarray | None = None) -> float:
    return float(np.dot(np.asarray(values), tail_probabilities(values, q, weights)))


def effective_scenarios(weights: np.ndarray) -> float:
    positive = weights[weights > 0]
    return float(np.exp(-np.dot(positive, np.log(positive))))


def concentration(values: np.ndarray, q: float, weights: np.ndarray) -> dict:
    w = probabilities(len(values), weights)
    tail = tail_probabilities(values, q, w)
    return {
        "ens_overall": effective_scenarios(w),
        "ens_tail": effective_scenarios(tail),
        "max_weight": float(w.max()),
        "tail_mass": 1.0 - q,
        "tail_definition": "Exact upper-tail mass with proportional allocation at tied losses; q=1 uses maxima.",
    }
