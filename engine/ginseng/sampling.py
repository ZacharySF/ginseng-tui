"""Fixed-coordinate stationary bootstrap, mapping version 1.

SeedSequence([root, domain, method, replicate]); domains 100=experiment,
200=reference, 300=execution order. Method IDs 1=MC, 2=Sobol, 3=legacy.
MC uses PCG64 float64 uniforms in a fixed-width C-order matrix. SciPy's
public seed=Generator API is supported since 1.14 (including newer versions).
"""

from dataclasses import dataclass
import numpy as np
from scipy.stats import qmc
from ginseng.simulate import (
    DrawBundle,
    _joint_history,
    _estimate_mean_block_length,
    _stationary_bootstrap_indices,
    _compute_draw_id,
)

METHODS = {"mc": 1, "sobol": 2, "legacy_mc": 3}
BITS = 30
MAX_ARRAY_BYTES = 512 * 1024**2


def derive_seed(root, method, replicate=0, domain=100):
    if any(
        not isinstance(v, (int, np.integer)) or v < 0 for v in (root, replicate, domain)
    ):
        raise ValueError(
            "Seeds, replicate IDs and domains must be nonnegative integers."
        )
    return int(
        np.random.SeedSequence(
            [root, domain, METHODS[method], replicate]
        ).generate_state(1, dtype=np.uint64)[0]
    )


def validate_size(method, n_paths, material_horizon):
    if method not in METHODS:
        raise ValueError("Sampler must be mc, sobol, or legacy_mc.")
    if (
        not isinstance(n_paths, int)
        or not isinstance(material_horizon, int)
        or min(n_paths, material_horizon) < 1
    ):
        raise ValueError("Paths and material horizon must be positive integers.")
    dimension = 2 * material_horizon - 1
    if dimension > 21201:
        raise ValueError(
            "Material horizon exceeds the supported dimension limit 21201."
        )
    if method == "sobol" and (n_paths & (n_paths - 1) or n_paths > 2**BITS):
        raise ValueError(
            "Sobol requires N=2^m (for example 1024 or 2048), at most 2^30."
        )
    # Conservative simultaneous uniform/index/cash/temporary array estimate.
    if n_paths * (dimension + 8 * material_horizon) * 8 > MAX_ARRAY_BYTES:
        raise ValueError(
            "Request exceeds 512 MiB estimated array budget; reduce paths/material horizon or batch MC references."
        )
    return dimension


def unit_points(method, n_paths, dimension, seed):
    if method not in ("mc", "sobol"):
        raise ValueError("Unit points require mc or sobol.")
    if not isinstance(dimension, int) or not 1 <= dimension <= 21201:
        raise ValueError("Dimension must be between 1 and 21201.")
    validate_size(method, n_paths, (dimension + 2) // 2)
    rng = np.random.Generator(np.random.PCG64(seed))
    if method == "mc":
        return rng.random((n_paths, dimension))
    return qmc.Sobol(
        d=dimension, scramble=True, bits=BITS, seed=rng, optimization=None
    ).random_base2(n_paths.bit_length() - 1)


def map_indices(points, history_length, mean_block_length, *, return_initial_lengths=False):
    u = np.asarray(points, dtype=float)
    if (
        u.ndim != 2
        or min(u.shape) < 1
        or u.shape[1] % 2 != 1
        or not np.all(np.isfinite(u))
        or np.any((u < 0) | (u >= 1))
    ):
        raise ValueError("Mapping requires finite [0,1) points of odd dimension.")
    if (
        not isinstance(history_length, int)
        or history_length < 1
        or not np.isfinite(mean_block_length)
        or mean_block_length < 1
    ):
        raise ValueError("History length and mean block length must be positive.")
    indices = np.empty((len(u), (u.shape[1] + 1) // 2), dtype=np.int64)
    indices[:, 0] = np.floor(history_length * u[:, 0]).astype(np.int64)
    lengths = np.full(len(u), indices.shape[1], dtype=np.int64) if return_initial_lengths else None
    for t in range(1, indices.shape[1]):
        if return_initial_lengths:
            restart = u[:, 2 * t - 1] >= 1 - 1 / mean_block_length
            lengths[(lengths == indices.shape[1]) & restart] = t
        indices[:, t] = np.where(
            u[:, 2 * t - 1] < 1 - 1 / mean_block_length,
            (indices[:, t - 1] + 1) % history_length,
            np.floor(history_length * u[:, 2 * t]).astype(np.int64),
        )
    return (indices, lengths) if return_initial_lengths else indices


@dataclass(frozen=True)
class PreparedHistory:
    joint: np.ndarray
    requested_length: int | None
    resolved_length: int
    clipped: bool
    resolution: str

    def __post_init__(self):
        a = np.asarray(self.joint, dtype=float)
        if a.ndim != 2 or a.shape[1] != 3 or len(a) < 1 or not np.all(np.isfinite(a)):
            raise ValueError("History must contain finite joint daily records.")
        object.__setattr__(
            self, "joint", np.frombuffer(a.tobytes(), dtype=float).reshape(a.shape)
        )


def prepare_history(state, mean_block_length=None):
    joint = _joint_history(state).to_numpy(dtype=float)
    if mean_block_length is None:
        net = joint[:, 0] - joint[:, 1] - joint[:, 2]
        length, clipped = _estimate_mean_block_length(net)
        # Record the actual estimator branch without changing its resolution policy.
        resolution = "constant/short-history default 14"
        if len(net) >= 2 and np.ptp(net) != 0:
            try:
                from arch.bootstrap import optimal_block_length

                estimate = optimal_block_length(net)
                col = "stationary" if "stationary" in estimate.columns else "b_sb"
                val = float(estimate[col].iloc[0])
                resolution = "arch Politis-White stationary"
            except (ImportError, ValueError, FloatingPointError):
                from ginseng.simulate import _fallback_block_length

                val = _fallback_block_length(net)
                resolution = "lag-1 autocorrelation fallback"
            if not np.isfinite(val) or val <= 0:
                resolution += "; invalid estimate default 14"
    else:
        if not isinstance(mean_block_length, int) or mean_block_length < 1:
            raise ValueError("Requested block length must be a positive integer.")
        length = int(np.clip(mean_block_length, 7, 28))
        clipped, resolution = length != mean_block_length, "explicit"
    return PreparedHistory(joint, mean_block_length, length, clipped, resolution)


def sample_bundle(
    prepared,
    horizon,
    paths,
    root_seed,
    method="mc",
    material_horizon=None,
    replicate=0,
    domain=100,
    *,
    trace_initial_block=False,
):
    material = horizon if material_horizon is None else material_horizon
    dimension = validate_size(method, paths, material)
    if not isinstance(horizon, int) or not 1 <= horizon <= material:
        raise ValueError("Visible horizon must lie within declared material horizon.")
    seed = derive_seed(root_seed, method, replicate, domain)
    lengths = None
    if trace_initial_block and method == "legacy_mc":
        raise ValueError("Initial-block traces require mc or sobol.")
    if method == "legacy_mc":
        if material != horizon:
            raise ValueError(
                "legacy_mc has no predeclared material plan; use mc or sobol, or omit --material-horizon."
            )
        full = _stationary_bootstrap_indices(
            np.random.Generator(np.random.PCG64(seed)),
            len(prepared.joint),
            paths,
            horizon,
            prepared.resolved_length,
        )
    else:
        full = map_indices(
            unit_points(method, paths, dimension, seed),
            len(prepared.joint),
            prepared.resolved_length,
            return_initial_lengths=trace_initial_block,
        )
        if trace_initial_block:
            full, lengths = full
            lengths = np.minimum(lengths, horizon)
    visible = full[:, :horizon]
    metadata = dict(
        sampler_version=1,
        mapping_version="legacy" if method == "legacy_mc" else 1,
        material_horizon=material,
        dimension=None if method == "legacy_mc" else dimension,
        root_seed=root_seed,
        replicate=replicate,
        domain=domain,
        derived_seed=seed,
        seed_scheme="SeedSequence([root, domain, method_id, replicate]).uint64",
        bit_generator="PCG64",
        bits=BITS if method == "sobol" else None,
        scramble="LMS + digital shift" if method == "sobol" else None,
        optimization=None,
        stream_layout="C-order fixed-width float64" if method == "mc" else method,
    )
    return DrawBundle(
        seed,
        horizon,
        paths,
        prepared.resolved_length,
        prepared.clipped,
        len(prepared.joint),
        visible,
        _compute_draw_id(visible),
        method,
        full if method != "legacy_mc" else None,
        tuple(metadata.items()),
        prepared.requested_length,
        lengths,
    )
