"""A bounded portfolio experiment with a liquidity-conditioned third objective.

The bank-cash event is frozen independently of hypothetical portfolio weights.
Assets are bought and held; a zero-yield CASH sleeve belongs to the research
portfolio and does not alter that bank account. Discovery and evaluation use
independent MC paths from the same history, not different historical periods.
"""

from __future__ import annotations

import hashlib
import time
import warnings

import numpy as np

from ginseng.portfolio import aligned_asset_returns
from ginseng.risk import cvar
from ginseng.sampling import prepare_history, sample_bundle
from ginseng.simulate import cash_paths

PATHS_PER_SPLIT = 2048
MIN_PRESSURE_PATHS = 128
TAIL_LEVEL = 0.95
MAX_HORIZON = 60
MAX_HISTORY_DAYS = 3660
RANDOM_CANDIDATES = 128
TOTAL_BUDGET_SECONDS = 8.0
SOLVE_BUDGET_SECONDS = 0.5
PARETO_TOLERANCE = 1e-10


def _first_pressure_returns(growth, balances, buffer):
    """Return terminal and first-breach asset returns, with an inert cash sleeve.

    ``growth`` contains cumulative gross asset returns with shape N,H,A.
    Zero is an ordinary end-of-day observation; equality with the buffer is
    survival. Paths without a breach never enter the conditional tail.
    """
    growth, balances = np.asarray(growth), np.asarray(balances)
    if (
        growth.ndim != 3
        or growth.shape[:2] != balances.shape
        or min(growth.shape) < 1
        or not np.all(np.isfinite(growth))
        or not np.all(np.isfinite(balances))
        or not np.isfinite(buffer)
    ):
        raise ValueError("Finite, aligned asset and cash paths are required.")
    breach = balances < buffer
    pressured = np.any(breach, axis=1)
    indices = np.flatnonzero(pressured)
    days = np.argmax(breach[pressured], axis=1)
    terminal = np.column_stack((growth[:, -1, :] - 1, np.zeros(len(growth))))
    pressure = np.column_stack((growth[indices, days, :] - 1, np.zeros(len(indices))))
    return terminal, pressure, days + 1


def _metrics(terminal, pressure, weights):
    returns = terminal @ weights
    return {
        "mean_return": float(np.mean(returns)),
        "volatility": float(np.std(returns, ddof=1)),
        "pressure_cvar": cvar(-(pressure @ weights), TAIL_LEVEL),
    }


def _pareto_mask(metrics):
    """Nondominance among these finite candidates; lower, higher, lower.

    Equal objective vectors are both retained. A tiny absolute tolerance in
    return units prevents solver roundoff deciding visually identical points.
    """
    objectives = np.array([
        [row["volatility"], -row["mean_return"], row["pressure_cvar"]]
        for row in metrics
    ])
    if objectives.ndim != 2 or objectives.shape[1] != 3 or not np.all(np.isfinite(objectives)):
        raise ValueError("Pareto comparisons require finite objective triples.")
    flags = []
    for candidate in objectives:
        no_worse = np.all(objectives <= candidate + PARETO_TOLERANCE, axis=1)
        better = np.any(objectives < candidate - PARETO_TOLERANCE, axis=1)
        flags.append(not np.any(no_worse & better))
    return flags


def _feasible_weights(value, count):
    if value is None:
        return None
    value = np.asarray(value, dtype=float)
    if (
        value.shape != (count,)
        or not np.all(np.isfinite(value))
        or value.min() < -1e-7
        or value.max() > 1 + 1e-7
        or abs(float(value.sum()) - 1) > 1e-6
    ):
        return None
    # Only repair negligible floating-point residuals of a feasible solve.
    value = np.clip(value, 0, 1)
    return value / value.sum()


def _optimized_candidates(terminal, pressure, deadline):
    """Solve 15 convex scalarizations; never use evaluation data for selection."""
    statistics = {
        "solver": "CLARABEL",
        "attempted": 0,
        "succeeded": 0,
        "failed": 0,
        "budget_exhausted": False,
        "solve_limit_seconds": SOLVE_BUDGET_SECONDS,
        "runs": [],
    }
    try:
        import cvxpy as cp
    except ImportError:
        statistics["message"] = "Install the optimization extra to add convex solver candidates."
        return [], statistics
    if "CLARABEL" not in cp.installed_solvers():
        statistics["message"] = "CLARABEL is unavailable; the explored candidate cloud is still evaluated."
        return [], statistics

    count = terminal.shape[1]
    mean = np.mean(terminal, axis=0)
    centered = terminal - mean
    covariance = centered.T @ centered / (len(terminal) - 1)
    variance_scale = max(float(np.diag(covariance).max()), 1e-12)
    mean_scale = max(float(np.abs(mean).max()), 1e-6)
    tail_scale = max(max(abs(cvar(-pressure[:, i], TAIL_LEVEL)) for i in range(count)), 1e-6)
    statistics["normalization"] = {
        "variance": variance_scale,
        "mean_return": mean_scale,
        "pressure_cvar": tail_scale,
    }
    allocation = cp.Variable(count)
    cutoff = cp.Variable()
    excess = cp.Variable(len(pressure), nonneg=True)
    preference = cp.Parameter(3, nonneg=True)
    conditional_tail = cutoff + cp.sum(excess) / ((1 - TAIL_LEVEL) * len(pressure))
    objective = (
        preference[0] * cp.quad_form(allocation, cp.psd_wrap(covariance / variance_scale))
        - preference[1] * (mean @ allocation) / mean_scale
        + preference[2] * conditional_tail / tail_scale
    )
    problem = cp.Problem(cp.Minimize(objective), [
        allocation >= 0,
        cp.sum(allocation) == 1,
        excess >= -pressure @ allocation - cutoff,
    ])
    # Put the three interpretable anchors first if the wall-clock budget runs out.
    choices = [(4, 0, 0), (0, 4, 0), (0, 0, 4)]
    choices += [
        (i, j, 4 - i - j)
        for i in range(5) for j in range(5 - i)
        if (i, j, 4 - i - j) not in choices
    ]
    names = {
        (4, 0, 0): ("min-volatility", "Minimum volatility"),
        (0, 4, 0): ("max-return", "Maximum modeled return"),
        (0, 0, 4): ("min-pressure-tail", "Minimum pressure tail"),
    }
    candidates = []
    for index, choice in enumerate(choices):
        remaining = deadline - time.monotonic()
        if remaining < 0.1:
            statistics["budget_exhausted"] = True
            break
        preference.value = np.asarray(choice) / 4
        statistics["attempted"] += 1
        started = time.monotonic()
        run = {"preference": preference.value.tolist()}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                problem.solve(
                    solver="CLARABEL", warm_start=False,
                    time_limit=min(SOLVE_BUDGET_SECONDS, remaining),
                    max_iter=100, tol_gap_abs=1e-8, tol_gap_rel=1e-8, tol_feas=1e-8,
                )
            run["status"] = problem.status
            valid = _feasible_weights(allocation.value, count)
            if problem.status != cp.OPTIMAL or valid is None:
                statistics["failed"] += 1
            else:
                statistics["succeeded"] += 1
                # These two simplex extrema are known exactly. Solving the
                # common program validates the path, then avoid displaying a
                # small numerical residue as risk in the all-cash portfolio.
                if choice == (4, 0, 0):
                    valid = np.eye(count)[-1]
                elif choice == (0, 4, 0):
                    valid = np.eye(count)[int(np.argmax(mean))]
                candidate_id, label = names.get(choice, (f"optimized-{index}", f"Trade-off {index - 2}"))
                candidates.append({"id": candidate_id, "label": label,
                                   "weights": valid, "source": "optimized"})
        except (cp.error.SolverError, ValueError, ArithmeticError) as exc:
            statistics["failed"] += 1
            run["status"] = "solver_error"
            run["message"] = type(exc).__name__
        run["seconds"] = time.monotonic() - started
        statistics["runs"].append(run)
    return candidates, statistics


def _input_identity(state, history, prepared, obligations, values):
    digest = hashlib.sha256()
    for array in (history, prepared.joint, values):
        digest.update(np.asarray(array, dtype="<f8").tobytes())
    digest.update(repr((
        state.as_of,
        state.history_start or min(item.txn_date for item in state.transactions),
        state.history_end or state.as_of,
        state.forecast_horizon, state.immediate_funding,
        state.operating_buffer, tuple(state.fixed_income_schedule),
        tuple(state.fixed_obligations), tuple(obligations), prepared.resolved_length,
        tuple(holding.symbol for holding in state.taxable_portfolio),
    )).encode())
    return digest.hexdigest()


def portfolio_frontier(state, obligations=(), seed=42, block_length=None):
    """Explore long-only allocations using 2,048 discovery and evaluation paths.

    Invalid or unsupported inputs return ``status='unavailable'`` with a reason.
    A ready result can contain a partial solver set; candidates are still valid
    and metadata records every skipped or failed optimization. Returns are
    horizon fractions, never annualized. No actual trades are generated.
    """
    started = time.monotonic()
    deadline = started + TOTAL_BUDGET_SECONDS

    def unavailable(reason, **details):
        return {"status": "unavailable", "message": reason, **details}

    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        return unavailable("The sampling seed must be a nonnegative integer.")
    if (
        not isinstance(state.forecast_horizon, int)
        or isinstance(state.forecast_horizon, bool)
        or not 1 <= state.forecast_horizon <= MAX_HORIZON
    ):
        return unavailable("The portfolio experiment supports horizons from 1 to 60 days.")
    if block_length is not None and (
        not isinstance(block_length, int) or isinstance(block_length, bool) or block_length < 1
    ):
        return unavailable("The block length must be a positive integer.")
    holdings = state.taxable_portfolio
    symbols = [holding.symbol for holding in holdings]
    if not 2 <= len(symbols) <= 8 or len(set(symbols)) != len(symbols) or "CASH" in symbols:
        return unavailable("Two to eight distinct taxable assets are required; CASH is reserved for the research sleeve.")
    if not state.transactions and state.history_start is None:
        return unavailable("Complete classified historical cash flows are required.")
    first = state.history_start or min(item.txn_date for item in state.transactions)
    last = state.history_end or state.as_of
    if last > state.as_of:
        return unavailable("Recorded historical observations cannot extend beyond the forecast as-of date.")
    if not 2 <= (last - first).days + 1 <= MAX_HISTORY_DAYS:
        return unavailable("The portfolio experiment requires 2 to 3,660 aligned history days.")
    values = np.array([holding.market_value for holding in holdings], dtype=float)
    obligations = tuple(obligations)
    amounts = [item.amount for item in (
        *state.transactions, *state.fixed_income_schedule, *state.fixed_obligations, *obligations,
    )]
    if (
        not np.all(np.isfinite(values)) or np.any(values < 0) or not np.isfinite(values.sum())
        or values.sum() <= 0 or not np.all(np.isfinite(amounts))
        or not np.isfinite(state.immediate_funding) or not np.isfinite(state.operating_buffer)
    ):
        return unavailable("Finite cash flows, a finite buffer, and positive taxable portfolio value are required.")
    try:
        aligned = aligned_asset_returns(state)
        if aligned is None:
            return unavailable("Aligned daily returns for every taxable asset are required; missing observations cannot be filled with zero.")
        _, history = aligned
        prepared = prepare_history(state, block_length)
        datasets, bundles = [], []
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for domain in (600, 601):
                bundle = sample_bundle(prepared, state.forecast_horizon, PATHS_PER_SPLIT, int(seed), domain=domain)
                balances = state.immediate_funding + cash_paths(
                    state, bundle, obligations, prepared_history=prepared.joint,
                )
                growth = np.cumprod(1 + history[bundle.index_matrix], axis=1)
                datasets.append(_first_pressure_returns(growth, balances, state.operating_buffer))
                bundles.append(bundle)
    except (ValueError, FloatingPointError, OverflowError) as exc:
        return unavailable(f"The supplied history cannot support this experiment: {exc}")
    pressure = {
        "discovery_count": len(datasets[0][1]),
        "evaluation_count": len(datasets[1][1]),
        "discovery_probability": len(datasets[0][1]) / PATHS_PER_SPLIT,
        "evaluation_probability": len(datasets[1][1]) / PATHS_PER_SPLIT,
        "minimum_count": MIN_PRESSURE_PATHS,
        "discovery_tail_observations": len(datasets[0][1]) * (1 - TAIL_LEVEL),
        "evaluation_tail_observations": len(datasets[1][1]) * (1 - TAIL_LEVEL),
        "discovery_tail_mass_count": len(datasets[0][1]) * (1 - TAIL_LEVEL),
        "evaluation_tail_mass_count": len(datasets[1][1]) * (1 - TAIL_LEVEL),
    }
    if min(pressure["discovery_count"], pressure["evaluation_count"]) < MIN_PRESSURE_PATHS:
        return unavailable(
            "Too few simulated cash-pressure paths to compare conditional tails: "
            f"{pressure['discovery_count']} discovery and {pressure['evaluation_count']} evaluation; "
            f"at least {MIN_PRESSURE_PATHS} in each are required. Increase the forecast horizon or add a cash event.",
            pressure=pressure,
        )
    symbols += ["CASH"]
    count = len(symbols)
    candidates = [
        {"id": "current", "label": "Current allocation", "weights": np.append(values / values.sum(), 0), "source": "anchor"},
        {"id": "equal-weight", "label": "Equal weight", "weights": np.full(count, 1 / count), "source": "anchor"},
    ]
    candidates.extend(
        {"id": f"pure-{i}", "label": f"100% {symbol}", "weights": np.eye(count)[i], "source": "anchor"}
        for i, symbol in enumerate(symbols)
    )
    allocation_seed = np.random.SeedSequence([int(seed), 602, 1, 0])
    rng = np.random.default_rng(allocation_seed)
    candidates.extend(
        {"id": f"sampled-{i}", "label": f"Allocation {i + 1}", "weights": weights, "source": "sampled"}
        for i, weights in enumerate(rng.dirichlet(np.ones(count), RANDOM_CANDIDATES))
    )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            optimized, solver = _optimized_candidates(*datasets[0][:2], deadline)
            candidates.extend(optimized)
            for candidate in candidates:
                candidate["discovery"] = _metrics(*datasets[0][:2], candidate["weights"])
            discovery_flags = _pareto_mask([row["discovery"] for row in candidates])
            # Freeze discovery selection before touching evaluation returns.
            for candidate, flag in zip(candidates, discovery_flags):
                candidate["pareto"] = bool(flag)
            for candidate in candidates:
                candidate["evaluation"] = _metrics(*datasets[1][:2], candidate["weights"])
                candidate["weights"] = candidate["weights"].tolist()
    except (ValueError, FloatingPointError, OverflowError) as exc:
        return unavailable(f"Portfolio metrics exceed the numerical range supported by this experiment: {exc}")
    evaluation_flags = _pareto_mask([row["evaluation"] for row in candidates])
    for candidate, evaluation_flag in zip(candidates, evaluation_flags):
        candidate["evaluation_pareto"] = bool(evaluation_flag)
    input_id = _input_identity(state, history, prepared, obligations, values)
    return {
        "status": "ready", "symbols": symbols, "points": candidates, "pressure": pressure,
        "metadata": {
            "version": 1, "horizon_days": state.forecast_horizon,
            "history_days": len(history), "paths_per_split": PATHS_PER_SPLIT,
            "paths_per_sample": PATHS_PER_SPLIT,
            "tail_level": TAIL_LEVEL, "seed": int(seed), "sampler": "mc",
            "block_length": prepared.resolved_length,
            "block_length_requested": block_length,
            "block_length_was_clipped": prepared.clipped,
            "discovery_domain": 600, "evaluation_domain": 601, "allocation_domain": 602,
            "discovery_draw_id": bundles[0].bootstrap_draw_id,
            "evaluation_draw_id": bundles[1].bootstrap_draw_id,
            "input_id": input_id, "input_hash": input_id,
            "solver": solver,
            "partial_optimization": solver["succeeded"] != 15,
            "elapsed_seconds": time.monotonic() - started,
            "computation_budget_seconds": TOTAL_BUDGET_SECONDS,
            "portfolio_value": float(values.sum()),
            "data_source": "Supplied aligned asset returns; demo asset histories are synthetic.",
            "source": "Supplied aligned asset returns; demo asset histories are synthetic.",
            "return_convention": "Buy and hold, horizon simple returns; volatility uses sample standard deviation (ddof=1); no annualization.",
            "pressure_definition": "Original bank cash first falls strictly below its operating buffer at a modeled end of day; the event is fixed across allocations.",
            "tail_definition": "95% CVaR of signed portfolio loss at the first cash-buffer breach, conditional on a breach; exact fractional upper 5% mass, including ties. Negative loss means a gain.",
            "cash_sleeve": "Zero-yield hypothetical portfolio cash; it does not top up bank cash or change the pressure event.",
            "evaluation_scope": "Independent simulation paths from the same fitted history. Frozen discovery allocations and frontier flags; not an out-of-time backtest or a forecast-confidence interval.",
            "frontier_scope": "Nondominated among explored candidates, not the complete continuous frontier. Lower volatility and pressure tail, higher mean return are preferred.",
            "execution_scope": "Hypothetical long-only allocations without fees, taxes, settlement, or trade execution.",
            "pareto_tolerance": PARETO_TOLERANCE,
        },
    }
