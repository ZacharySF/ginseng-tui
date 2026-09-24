"""Constraint generation for the same scenario CVaR linear program.

Evaluate every path at each candidate, then add supporting planes for cost
and violated risk limits. The small master LP gives a lower objective bound.
Return a plan only when full-path feasibility and the objective gap pass.
No scenario is removed, sampled again, or assigned a different probability.
"""

from dataclasses import dataclass
from time import monotonic

import numpy as np
from scipy.optimize import linprog

from ginseng.risk import tail_probabilities


class CutSolveFailure(Exception):
    pass


@dataclass(frozen=True)
class CutSolution:
    credit: float
    withdrawals: np.ndarray
    deferral: float
    cost: float
    stochastic_var: float
    buffer_dual: float
    credit_dual: float
    iterations: int
    evidence: dict


def solve_funding_cuts(base, savings, credit_effect, availability, capacities, net_rates,
                       charge_rates, credit_capacity, interest_rate, overdraft_rate,
                       buffer, allowance, tail_limit, q, weights, time_limit, objective_kind="cvar",
                       buffer_coverage_target=None):
    """All monetary inputs/outputs share the caller's numerical dollar scale."""
    from ginseng.risk import quantile

    deadline = monotonic() + time_limit
    dimensions = len(capacities) + 2  # credit, withdrawal units, spending fraction
    upper = np.concatenate(([credit_capacity], capacities, [1.0]))
    path_indices = np.arange(len(base))

    def evaluate(x):
        balance = base + x[0] * credit_effect + (net_rates @ x[1:-1]) * availability + x[-1] * savings
        negative = balance < 0
        stochastic = x[-1] * savings[:, -1] + overdraft_rate * np.maximum(0, -balance).sum(axis=1)
        common = interest_rate * x[0] + charge_rates @ x[1:-1]
        tail_weights = weights if objective_kind == "expected" else tail_probabilities(stochastic, q, weights)
        cost = common + tail_weights @ stochastic
        cost_gradient = np.concatenate((
            [interest_rate - overdraft_rate * (tail_weights @ (negative @ credit_effect))],
            charge_rates - overdraft_rate * net_rates * (tail_weights @ (negative @ availability)),
            [tail_weights @ savings[:, -1] - overdraft_rate * (tail_weights @ (negative * savings).sum(axis=1))],
        ))
        below = balance < buffer
        mean = weights @ np.maximum(0, buffer - balance).sum(axis=1)
        mean_gradient = np.concatenate((
            [-weights @ (below @ credit_effect)],
            -net_rates * (weights @ (below @ availability)),
            [-weights @ (below * savings).sum(axis=1)],
        ))
        tail, tail_gradient = None, None
        if tail_limit is not None:
            index = np.argmin(balance, axis=1)
            deficit = np.maximum(0, buffer - balance[path_indices, index])
            tail_weights = tail_probabilities(deficit, q, weights)
            active_weights = tail_weights * (deficit > 0)
            tail = tail_weights @ deficit
            tail_gradient = np.concatenate((
                [-active_weights @ credit_effect[index]],
                -net_rates * (active_weights @ availability[index]),
                [-active_weights @ savings[path_indices, index]],
            ))
        coverage, coverage_gradient = None, None
        if buffer_coverage_target is not None:
            index = np.argmin(balance, axis=1)
            margin = buffer - balance[path_indices, index]
            tw = tail_probabilities(margin, buffer_coverage_target, weights)
            coverage = tw @ margin
            coverage_gradient = np.concatenate(([-tw @ credit_effect[index]],
                -net_rates * (tw @ availability[index]), [-tw @ savings[path_indices, index]]))
        return cost, cost_gradient, mean, mean_gradient, tail, tail_gradient, quantile(stochastic, q, weights), coverage, coverage_gradient

    rows, rhs, mean_rows = [], [], []

    def add_planes(x, observation):
        cost, cg, mean, mg, tail, tg, _, coverage, coverage_gradient = observation
        rows.append(np.append(cg, -1.0))
        rhs.append(float(cg @ x - cost))
        mean_rows.append(len(rows))
        rows.append(np.append(mg, 0.0))
        rhs.append(float(allowance + mg @ x - mean))
        if tail_limit is not None:
            rows.append(np.append(tg, 0.0))
            rhs.append(float(tail_limit + tg @ x - tail))
        if coverage is not None:
            rows.append(np.append(coverage_gradient, 0.0))
            rhs.append(float(coverage_gradient @ x - coverage))

    zero = np.zeros(dimensions)
    add_planes(zero, evaluate(zero))
    funded = np.concatenate(([0.0], capacities, [0.0]))
    add_planes(funded, evaluate(funded))
    objective = np.zeros(dimensions + 1); objective[-1] = 1.0
    bounds = [(0, float(limit)) for limit in upper] + [(0, None)]

    for iteration in range(1, 121):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise CutSolveFailure("solver_timeout")
        result = linprog(objective, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bounds,
                         method="highs", options={"time_limit": remaining,
                             "primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
        if not result.success:
            raise CutSolveFailure({1: "solver_timeout", 2: "infeasible", 3: "unbounded"}.get(result.status, "solver_error"))
        x = np.clip(result.x[:-1], 0, upper)
        observation = evaluate(x)
        cost, _, mean, _, tail, _, stochastic_var, coverage, _ = observation
        # At a $1,000 scale these floors are a micro-dollar of feasibility
        # and a hundredth of a cent of objective accuracy; larger values retain
        # a relative 1e-7 objective gap. Final caller checks are independent.
        feasibility_tolerance = 1e-9
        gap_tolerance = max(1e-7, 1e-7 * abs(cost))
        if (mean <= allowance + feasibility_tolerance
                and (tail is None or tail <= tail_limit + feasibility_tolerance)
                and (coverage is None or coverage <= feasibility_tolerance)
                and cost <= result.fun + gap_tolerance):
            # A conservative Lagrangian bound, including reduced-cost residuals
            # over the finite control box. Never promote the primal master value
            # itself to a certified lower bound.
            dual = np.minimum(np.asarray(result.ineqlin.marginals), 0.)
            cost_rows = np.array(rows)[:, -1] == -1
            mass = -dual[cost_rows].sum()
            if mass > 1: dual[cost_rows] /= mass * (1 + 1e-14)
            reduced = objective - np.array(rows).T @ dual
            lower = float(np.dot(rhs, dual) + np.dot(np.minimum(reduced[:-1], 0), upper))
            if reduced[-1] < 0:
                lower = None
            evidence = dict(lower_bound=lower, master_primal_objective=float(result.fun),
                candidate_objective=float(cost), absolute_gap=max(0.,cost-lower) if lower is not None else None,
                relative_gap=max(0.,cost-lower)/max(1.,abs(cost)) if lower is not None else None,
                master_primal_residual=float(max(0., -min(result.ineqlin.residual))),
                master_complementarity_residual=float(np.max(np.abs(result.ineqlin.marginals * result.ineqlin.residual))),
                iterations=iteration, master_iterations=int(result.nit), termination_reason='full_path_feasible_and_gap',
                bound_method='Lagrangian supporting-plane bound with finite-box residual correction',
                scope='finite supplied scenarios; fixed selected credit account; continuous fixed-price withdrawal units; no unexamined regimes',
                global_lower_bound=None, global_bound_status='unavailable outside this declared convex subproblem',
                feasibility_tolerance=feasibility_tolerance, objective_gap_tolerance=gap_tolerance)
            return CutSolution(float(x[0]), x[1:-1], float(x[-1]), float(cost), float(stochastic_var),
                float(-np.sum(result.ineqlin.marginals[mean_rows])),
                float(-result.upper.marginals[0]), iteration, evidence)
        add_planes(x, observation)
    raise CutSolveFailure("solver_limit")
