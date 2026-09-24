"""CVaR-optimal funding plan (Telser safety-first + Rockafellar-Uryasev).

The model evaluates every supplied bootstrap path under common random numbers.
Buffer-shortfall and overdraft auxiliaries are created only for path-days that
can incur those costs under the control bounds. Provably zero terms are omitted,
never paths or their probability weights. Requests whose evaluation bundle
exceeds ``MAX_SCENARIO_DAYS`` return an explicit resource-limit failure.
Above 20,000 scenario-days, a HiGHS master LP adds supporting planes for the
same costs and constraints. Every candidate is evaluated on every path;
acceptance requires full-path feasibility and a bounded objective gap.

The default Telser tolerance is a *mean* dollar-day limit:
``operating_buffer * evaluation_horizon_days * policy shortfall probability``.
It allows the same aggregate buffer erosion as fully missing the operating
buffer on the policy's permitted fraction of forecast days, and is independent
of the number of bootstrap paths.

Withdrawals execute today at recorded prices, then become available after
settlement and transfer, net of the assumed tax and penalty reserve. Taxable
lots use their holding periods; traditional IRA withdrawals price ordinary
income plus the early penalty; Roth access is capped at remaining regular
contributions. Future returns never reprice an executed sale.
"""

from __future__ import annotations

from ginseng.execution import execution_scope

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from ginseng.funding import (
    FundingConfig,
    _next_charge_payment_offset,
    _select_primary_credit_account,
    extend_draw_bundle,
    settlement_forecast_day,
)
from ginseng.policy import FundingPolicy, policy_assessment
from ginseng.risk import probabilities, quantile, cvar, weight_hash, balance_risk, MONEY_TOLERANCE
from ginseng.simulate import (
    DrawBundle,
    PathBundle,
    cash_paths,
    discretionary_resampled_paths,
)
from ginseng.state import FinancialState, Obligation
from ginseng.withdrawals import (
    AccountWithdrawal, WithdrawalAllocation, WithdrawalAssumptions,
    LONG_TERM_CAPITAL_GAINS_RATE, withdrawal_units, quote_withdrawals, allocate_net_cash,
)

# Keep the worst-case LP within an interactive request's memory/time envelope.
# This covers the UI's 14/30/60-day horizons at up to 3,000 paths; larger
# evaluations omit the optional optimizer, never substitute fewer paths.
MAX_SCENARIO_DAYS = 200_000
SOLVER_TIME_LIMIT_SECONDS = 10.0
# Larger LPs use full-path supporting planes to avoid a path-day-sized KKT solve.
CONSTRAINT_GENERATION_THRESHOLD = 20_000
_NUMERICAL_TOLERANCE = 1e-6
# Express the LP in thousands of dollars to keep retirement-sized withdrawals
# and small probability-weighted losses on a better-conditioned numeric scale.
# Both objective and dollar constraints use this same scale, so dual prices
# retain their original dollars-per-dollar units.
_DOLLAR_SCALE = 1000.0
_OPTIMAL_STATUSES = frozenset(("optimal", "optimal_inaccurate"))
_INFEASIBLE_STATUSES = frozenset(("infeasible",))

OptimizationFailureReason = Literal[
    "invalid_input", "no_funding_levers", "resource_limit", "cvxpy_unavailable",
    "solver_unavailable", "solver_timeout", "solver_limit", "solver_error",
    "infeasible", "unbounded", "invalid_solution",
]


@dataclass(frozen=True)
class OptimizationFailure:
    reason: OptimizationFailureReason
    verification: dict | None = None


@dataclass(frozen=True)
class OptimalPlan:
    credit_draw: float
    liquidation_amount: float
    deferral_fraction: float
    cvar_cost: float
    var_cost: float
    expected_cost: float
    cash_shortfall_probability: float
    implied_liquidity_price: float | None
    cost_is_path_dependent: bool
    solver_status: str
    evaluation_horizon_days: int
    evaluation_draw_id: str
    evaluation_paths: int
    cost_coverage_target: float
    buffer_breach_probability: float
    dollar_days_below_buffer: float
    buffer_tolerance_dollar_days: float
    buffer_constraint_binding: bool
    evaluation_weight_hash: str = ""
    tail_deficit: float = 0.0
    tail_deficit_limit: float | None = None
    implied_credit_price: float | None = None
    credit_constraint_binding: bool = False
    objective_kind: str = "cvar"
    withdrawal_accounts: tuple[AccountWithdrawal, ...] = ()
    withdrawal_allocations: tuple[WithdrawalAllocation, ...] = ()
    withdrawal_net_cash: float = 0.0
    withdrawal_tax_reserve: float = 0.0
    withdrawal_penalty_reserve: float = 0.0
    solver_method: str = "clarabel"
    credit_utilization: float = 0.0
    meets_policy: bool = False
    policy_reason: str | None = None
    buffer_coverage_target: float | None = None
    lot_selection: str = "minimum_withdrawal_charge"
    credit_account_id: str | None = None
    decision_horizon_days: int | None = None
    settlement_days: int = 1
    external_transfer_days: int = 2
    use_business_days: bool = False
    verification: dict | None = None
    solver_evidence: dict | None = None


def _finite_nonnegative(value: float) -> bool:
    return isfinite(value) and value >= 0.0


def _scalar_value(value: object) -> float | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size != 1:
        return None
    result = float(array.item())
    return result if isfinite(result) else None


def _bound_tolerance(lower: float, upper: float) -> float:
    return _NUMERICAL_TOLERANCE * max(1.0, abs(lower), abs(upper))


def _bounded_solution(value: object, lower: float, upper: float) -> float | None:
    result = _scalar_value(value)
    if result is None:
        return None
    tolerance = _bound_tolerance(lower, upper)
    if result < lower - tolerance or result > upper + tolerance:
        return None
    return float(np.clip(result, lower, upper))


def _trim_redundant_liquidation(
    balances: np.ndarray,
    liquidation: float,
    settlement_column: int,
    operating_buffer: float,
    buffer_tolerance: float,
    weights: np.ndarray | None = None,
    tail_deficit_limit: float | None = None,
    q: float = 0.95,
    buffer_coverage_target: float | None = None,
) -> float:
    """Remove surplus from a sale with zero tax cost, holding other levers fixed.

    With no assumed tax benefit, multiple sale amounts can have the same
    objective. Reduce proceeds only where every post-settlement balance
    remains nonnegative and the mean buffer dollar-day constraint holds.
    This preserves every path's cost without an invented penalty weight or
    a second solver run. It does not claim a global minimum-sale solution.
    """
    if liquidation <= 0.0:
        return liquidation
    settled = balances[:, settlement_column:]
    removable = min(
        liquidation, max(0.0, float(np.min(settled)) - _NUMERICAL_TOLERANCE)
    )
    if removable <= 0.0:
        return liquidation

    w = probabilities(len(balances), weights)
    fixed_shortfall = np.maximum(0.0, operating_buffer - balances[:, :settlement_column]).sum(axis=1)

    def fits_buffer(reduction: float) -> bool:
        shortfall = fixed_shortfall + np.maximum(
            0.0, operating_buffer - settled + reduction
        ).sum(axis=1)
        if np.dot(w, shortfall) > buffer_tolerance:
            return False
        if buffer_coverage_target is not None:
            adjusted = balances.copy()
            adjusted[:, settlement_column:] -= reduction
            if cvar(operating_buffer - adjusted.min(axis=1), buffer_coverage_target, w) > 0:
                return False
        if tail_deficit_limit is not None:
            adjusted = balances.copy()
            adjusted[:, settlement_column:] -= reduction
            deficits = np.maximum(0.0, operating_buffer - adjusted.min(axis=1))
            if cvar(deficits, q, w) > tail_deficit_limit:
                return False
        return True

    if fits_buffer(removable):
        return liquidation - removable

    lower, upper = 0.0, removable
    for _ in range(40):
        middle = (lower + upper) / 2.0
        if fits_buffer(middle):
            lower = middle
        else:
            upper = middle
    return liquidation - lower


@execution_scope
def optimize_funding(
    state: FinancialState,
    bundle: DrawBundle | PathBundle,
    obligations: Sequence[Obligation],
    coverage_target: float = 0.95,
    operating_buffer: float = 1000.0,
    overdraft_apr: float = 0.2999,
    buffer_tolerance_dollar_days: float | None = None,
    capital_gains_rate: float = LONG_TERM_CAPITAL_GAINS_RATE,
    funding_config: FundingConfig | None = None,
    funding_policy: FundingPolicy | None = None,
    *,
    weights: np.ndarray | None = None,
    tail_deficit_limit: float | None = None,
    objective_kind: Literal["cvar", "expected"] = "cvar",
    time_limit_seconds: float = SOLVER_TIME_LIMIT_SECONDS,
    evaluation_horizon_days: int | None = None,
    decision_horizon_days: int | None = None,
) -> OptimalPlan | OptimizationFailure:
    """Return the bounded scenario CVaR-optimal funding mix, if it solves.

    The credit leg uses only the funding module's primary card and respects
    both that account's available credit and the supplied policy's utilization
    ceiling. The optimizer selects account and lot amounts by modeled withdrawal
    cost; FIFO/HIFO is the named taxable-plan rule, not an optimizer restriction.

    ``coverage_target == 1`` is the empirical worst-case objective. Other
    targets in ``[0, 1)`` use the Rockafellar-Uryasev CVaR formulation.
    Returns an ``OptimizationFailure`` with a stable reason when unavailable,
    infeasible, resource-limited, or rejected by the numerical checks.
    """
    q = float(coverage_target)
    if not isfinite(q) or q < 0.0 or q > 1.0:
        return OptimizationFailure("invalid_input")
    if not _finite_nonnegative(float(operating_buffer)):
        return OptimizationFailure("invalid_input")
    if not _finite_nonnegative(float(overdraft_apr)):
        return OptimizationFailure("invalid_input")
    if not _finite_nonnegative(float(capital_gains_rate)):
        return OptimizationFailure("invalid_input")
    if buffer_tolerance_dollar_days is not None and not _finite_nonnegative(
        float(buffer_tolerance_dollar_days)
    ):
        return OptimizationFailure("invalid_input")
    if bundle.n_paths <= 0 or bundle.horizon_days <= 0:
        return OptimizationFailure("invalid_input")
    if (tail_deficit_limit is not None and not _finite_nonnegative(tail_deficit_limit)) or objective_kind not in ("cvar", "expected"):
        return OptimizationFailure("invalid_input")
    if not isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        return OptimizationFailure("invalid_input")
    try:
        w = probabilities(bundle.n_paths, weights)
    except ValueError:
        return OptimizationFailure("invalid_input")
    resolved_funding_config = funding_config or FundingConfig()
    resolved_funding_policy = funding_policy or FundingPolicy()
    buffer_coverage_target = (1 - resolved_funding_policy.max_buffer_breach_probability
        if resolved_funding_policy.max_buffer_breach_probability is not None else None)
    if buffer_coverage_target is not None and not 0 <= buffer_coverage_target <= 1:
        return OptimizationFailure("invalid_input")
    max_cash_shortfall_probability = float(
        resolved_funding_policy.max_cash_shortfall_probability
    )
    max_credit_utilization = (
        float(resolved_funding_policy.max_credit_utilization)
        if funding_policy is not None
        else 1.0
    )
    if (
        not isfinite(max_cash_shortfall_probability)
        or not 0.0 <= max_cash_shortfall_probability <= 1.0
        or not isfinite(max_credit_utilization)
        or not 0.0 <= max_credit_utilization <= 1.0
    ):
        return OptimizationFailure("invalid_input")
    primary_account = _select_primary_credit_account(state, max_credit_utilization)
    available_credit = (
        min(
            float(primary_account.available_credit),
            max(
                0.0,
                float(primary_account.credit_limit) * max_credit_utilization
                - float(primary_account.current_balance),
            ),
        )
        if primary_account is not None
        else 0.0
    )
    try:
        tax_assumptions = WithdrawalAssumptions(long_term_rate=capital_gains_rate)
        units = withdrawal_units(state, tax_assumptions)
    except ValueError:
        return OptimizationFailure("invalid_input")
    # A zero-capacity dummy keeps the LP shape valid when no account is available.
    capacities = np.array([u.capacity for u in units] or [0.0])
    net_rates = np.array([u.net_per_dollar for u in units] or [0.0])
    charge_rates = np.array([u.tax_per_dollar + u.penalty_per_dollar for u in units] or [0.0])
    initial_market_value = float(capacities.sum())
    if not _finite_nonnegative(available_credit) or not _finite_nonnegative(initial_market_value):
        return OptimizationFailure("invalid_input")

    credit_due_day = 0
    interest_per_credit_dollar = 0.0
    if primary_account is not None and available_credit > 0.0:
        credit_due_day = _next_charge_payment_offset(
            state.as_of,
            primary_account,
            one_indexed=resolved_funding_config.use_business_days,
        )
        grace_applies = (
            primary_account.grace_period_eligible and primary_account.current_balance <= 0.0
        )
        if not grace_applies:
            elapsed_days = (
                credit_due_day - 1
                if resolved_funding_config.use_business_days
                else credit_due_day
            )
            interest_per_credit_dollar = primary_account.purchase_apr / 365.0 * elapsed_days
            if not _finite_nonnegative(interest_per_credit_dollar):
                return OptimizationFailure("invalid_input")

    settlement_day = settlement_forecast_day(
        state.as_of,
        resolved_funding_config.settlement_days,
        resolved_funding_config.external_transfer_days,
        use_business_days=resolved_funding_config.use_business_days,
    )
    material_days = [bundle.horizon_days]
    if available_credit > 0.0:
        material_days.append(credit_due_day)
    if initial_market_value > 0.0:
        material_days.append(settlement_day)
    latest_material_day = max(material_days)
    evaluation_horizon = (
        latest_material_day
        if latest_material_day <= bundle.horizon_days
        else latest_material_day + resolved_funding_config.trailing_days
    )
    if evaluation_horizon_days is not None:
        if evaluation_horizon_days < evaluation_horizon:
            return OptimizationFailure("invalid_input")
        evaluation_horizon = evaluation_horizon_days
    if bundle.n_paths * evaluation_horizon > MAX_SCENARIO_DAYS:
        return OptimizationFailure("resource_limit")
    evaluation_bundle = extend_draw_bundle(bundle, evaluation_horizon)

    try:
        import cvxpy as cp
    except ImportError:
        return OptimizationFailure("cvxpy_unavailable")
    if cp.CLARABEL not in cp.installed_solvers():
        return OptimizationFailure("solver_unavailable")

    baseline_cash = cash_paths(state, evaluation_bundle, obligations)
    discretionary_daily = discretionary_resampled_paths(state, evaluation_bundle).copy()
    if decision_horizon_days is not None:
        discretionary_daily[:, decision_horizon_days:] = 0
    discretionary_savings = np.cumsum(discretionary_daily, axis=1)
    n_paths, horizon_days = baseline_cash.shape
    # A funding decision may be evaluated through a later repayment date, but
    # it must not pay for that debt by assuming discretionary cuts beyond the
    # user-visible decision horizon.
    if bundle.horizon_days < horizon_days:
        discretionary_savings[:, bundle.horizon_days:] = discretionary_savings[
            :, [bundle.horizon_days - 1]
        ]
    if (
        baseline_cash.shape != (evaluation_bundle.n_paths, evaluation_bundle.horizon_days)
        or discretionary_savings.shape != baseline_cash.shape
        or n_paths != bundle.n_paths
        or horizon_days != evaluation_horizon
    ):
        return OptimizationFailure("invalid_input")
    if not np.all(np.isfinite(baseline_cash)) or not np.all(np.isfinite(discretionary_savings)):
        return OptimizationFailure("invalid_input")
    if available_credit == 0.0 and initial_market_value == 0.0 and not np.any(discretionary_savings > 0.0):
        return OptimizationFailure("no_funding_levers")
    credit_effect_daily = np.zeros(horizon_days, dtype=float)
    advance_fee = primary_account.cash_advance_fee_pct if primary_account else 0.0
    if available_credit > 0.0:
        credit_effect_daily[0] = 1.0 - advance_fee
        credit_effect_daily[credit_due_day - 1] -= 1.0 + interest_per_credit_dollar
    credit_effect = np.cumsum(credit_effect_daily)
    credit_effect_matrix = np.broadcast_to(credit_effect, (n_paths, horizon_days))

    settlement_column = settlement_day - 1
    liquidation_effect = np.zeros((n_paths, horizon_days), dtype=float)
    liquidation_effect[:, settlement_column:] = 1.0

    deferred_cost_per_fraction = float(np.dot(w, discretionary_savings[:, -1]))
    if not isfinite(deferred_cost_per_fraction):
        return OptimizationFailure("invalid_input")
    if buffer_tolerance_dollar_days is None:
        buffer_tolerance = (
            operating_buffer * horizon_days * max_cash_shortfall_probability
        )
    else:
        buffer_tolerance = float(buffer_tolerance_dollar_days)

    credit = cp.Variable(nonneg=True)
    withdrawals = cp.Variable(len(capacities), nonneg=True)
    withdrawal_cash = net_rates @ withdrawals
    deferral_fraction = cp.Variable(nonneg=True)
    eta = cp.Variable()

    # Drop only hinge terms that are provably zero throughout the control
    # bounds. All paths remain in the objective and its probability weights.
    # This avoids solving for two auxiliaries on every already-funded day.
    minimum_balance = (
        float(state.immediate_funding)
        + baseline_cash
        + np.minimum(0.0, available_credit * credit_effect_matrix)
        + np.minimum(0.0, discretionary_savings)
    )

    def selected_balance(rows: np.ndarray, columns: np.ndarray):
        return (
            (float(state.immediate_funding) + baseline_cash[rows, columns]) / _DOLLAR_SCALE
            + credit * credit_effect[columns]
            + withdrawal_cash * liquidation_effect[rows, columns]
            + deferral_fraction * discretionary_savings[rows, columns] / _DOLLAR_SCALE
        )

    credit_constraint = credit <= available_credit / _DOLLAR_SCALE
    constraints = [
        credit_constraint,
        withdrawals <= capacities / _DOLLAR_SCALE,
        deferral_fraction <= 1.0,
    ]
    buffer_rows, buffer_columns = np.nonzero(minimum_balance < operating_buffer)
    buffer_total = cp.Constant(0.0)
    if buffer_rows.size:
        buffer_shortfall = cp.Variable(buffer_rows.size, nonneg=True)
        constraints.append(
            buffer_shortfall >= operating_buffer / _DOLLAR_SCALE - selected_balance(buffer_rows, buffer_columns)
        )
        buffer_total = w[buffer_rows] @ buffer_shortfall
    buffer_constraint = buffer_total <= buffer_tolerance / _DOLLAR_SCALE
    constraints.append(buffer_constraint)

    if buffer_coverage_target is not None:
        # Signed worst margin retains the slack of safe futures. CVaR <= 0
        # conservatively enforces the requested empirical no-breach frequency.
        margin = cp.Variable(n_paths)
        balances = ((float(state.immediate_funding) + baseline_cash) / _DOLLAR_SCALE
                    + credit * credit_effect_matrix + withdrawal_cash * liquidation_effect
                    + deferral_fraction * discretionary_savings / _DOLLAR_SCALE)
        constraints.append(cp.reshape(margin, (n_paths, 1), order="C") >= operating_buffer / _DOLLAR_SCALE - balances)
        if buffer_coverage_target == 1:
            constraints.append(margin[w > 0] <= 0)
        else:
            margin_eta = cp.Variable()
            margin_excess = cp.Variable(n_paths, nonneg=True)
            constraints.extend([margin_excess >= margin - margin_eta,
                                margin_eta + w @ margin_excess / (1 - buffer_coverage_target) <= 0])

    if tail_deficit_limit is not None:
        worst_buffer_deficit = cp.Variable(n_paths, nonneg=True)
        if buffer_rows.size:
            constraints.append(worst_buffer_deficit[buffer_rows] >= operating_buffer / _DOLLAR_SCALE - selected_balance(buffer_rows, buffer_columns))
        if q == 1:
            constraints.append(worst_buffer_deficit[w > 0] <= tail_deficit_limit / _DOLLAR_SCALE)
        else:
            deficit_eta = cp.Variable()
            deficit_excess = cp.Variable(n_paths, nonneg=True)
            constraints.append(deficit_excess >= worst_buffer_deficit - deficit_eta)
            constraints.append(deficit_eta + w @ deficit_excess / (1 - q) <= tail_deficit_limit / _DOLLAR_SCALE)

    overdraft_rows, overdraft_columns = np.nonzero(minimum_balance < 0.0)
    overdraft_cost = cp.Constant(np.zeros(n_paths))
    if overdraft_rows.size and overdraft_apr > 0.0:
        overdraft = cp.Variable(overdraft_rows.size, nonneg=True)
        constraints.append(overdraft >= -selected_balance(overdraft_rows, overdraft_columns))
        per_path_sum = csr_matrix(
            (np.ones(overdraft_rows.size), (overdraft_rows, np.arange(overdraft_rows.size))),
            shape=(n_paths, overdraft_rows.size),
        )
        overdraft_cost = (overdraft_apr / 365.0) * (per_path_sum @ overdraft)
    # CVaR(X + a) = CVaR(X) + a when a is the same in every future.
    # Keep deterministic withdrawal charges and interest outside the path
    # epigraph. This is the same objective with much sparser constraints.
    common_cost = (interest_per_credit_dollar + advance_fee) * credit + charge_rates @ withdrawals
    path_cost = discretionary_savings[:, -1] * deferral_fraction / _DOLLAR_SCALE + overdraft_cost

    if q == 1.0:
        constraints.append(eta >= path_cost[w > 0])
        objective = common_cost + eta
    else:
        cvar_excess = cp.Variable(n_paths, nonneg=True)
        constraints.append(cvar_excess >= path_cost - eta)
        objective = common_cost + eta + w @ cvar_excess / (1.0 - q)

    if objective_kind == "expected":
        objective = common_cost + w @ path_cost

    problem = cp.Problem(cp.Minimize(objective), constraints)
    solver_method = "clarabel"
    solver_evidence = {}
    if n_paths * horizon_days > CONSTRAINT_GENERATION_THRESHOLD:
        from ginseng.funding_cuts import solve_funding_cuts, CutSolveFailure
        try:
            solution = solve_funding_cuts(
                (float(state.immediate_funding) + baseline_cash) / _DOLLAR_SCALE,
                discretionary_savings / _DOLLAR_SCALE, credit_effect, liquidation_effect[0],
                capacities / _DOLLAR_SCALE, net_rates, charge_rates,
                available_credit / _DOLLAR_SCALE, interest_per_credit_dollar + advance_fee, overdraft_apr / 365.0,
                operating_buffer / _DOLLAR_SCALE, buffer_tolerance / _DOLLAR_SCALE,
                None if tail_deficit_limit is None else tail_deficit_limit / _DOLLAR_SCALE,
                q, w, time_limit_seconds, objective_kind,
                buffer_coverage_target=buffer_coverage_target,
            )
        except CutSolveFailure as failure:
            return OptimizationFailure(str(failure))
        credit.value = solution.credit
        withdrawals.value = solution.withdrawals
        deferral_fraction.value = solution.deferral
        eta.value = solution.stochastic_var
        objective_value = solution.cost
        buffer_dual_value = solution.buffer_dual
        credit_dual_value = solution.credit_dual
        status = "optimal"
        solver_method = "highs_constraint_generation"
        solver_evidence = dict(solution.evidence)
        for name in ('lower_bound','master_primal_objective','candidate_objective','absolute_gap','master_primal_residual','master_complementarity_residual','feasibility_tolerance','objective_gap_tolerance'):
            if solver_evidence.get(name) is not None: solver_evidence[name] *= _DOLLAR_SCALE
        lower = solver_evidence.get('lower_bound')
        solver_evidence['relative_gap'] = (max(0.,solver_evidence['candidate_objective']-lower)/max(1.,abs(solver_evidence['candidate_objective'])) if lower is not None else None)
    else:
        try:
            problem.solve(
                solver=cp.CLARABEL,
                verbose=False,
                time_limit=time_limit_seconds,
                tol_gap_abs=1e-10,
                tol_feas=1e-10,
            )
        except cp.error.SolverError:
            return OptimizationFailure("solver_error")

        status = str(problem.status or "")
        if status == cp.USER_LIMIT:
            solve_time = getattr(problem.solver_stats, "solve_time", None)
            reason = (
                "solver_timeout"
                if solve_time is not None and solve_time >= time_limit_seconds
                else "solver_limit"
            )
            return OptimizationFailure(reason)
        if status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
            return OptimizationFailure("infeasible")
        if status in (cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE):
            return OptimizationFailure("unbounded")
        if status not in _OPTIMAL_STATUSES:
            return OptimizationFailure("solver_error")

        objective_value = problem.value
        buffer_dual_value = buffer_constraint.dual_value
        credit_dual_value = credit_constraint.dual_value
        solver_evidence = dict(lower_bound=None, global_lower_bound=None,
            global_bound_status='unavailable: no independently validated Clarabel dual bound exposed by this adapter',
            absolute_gap=None, relative_gap=None, iterations=getattr(problem.solver_stats,'num_iters',None),
            termination_reason=status, master_primal_residual=None, master_complementarity_residual=None,
            scope='finite supplied scenarios; fixed selected credit account and withdrawal charge regime',
            feasibility_tolerance=1e-10, objective_gap_tolerance=1e-10)

    credit_value = _bounded_solution(None if credit.value is None else credit.value * _DOLLAR_SCALE, 0.0, available_credit)
    withdrawal_values = np.asarray(withdrawals.value, dtype=float) * _DOLLAR_SCALE if withdrawals.value is not None else np.array([])
    deferral_value = _bounded_solution(deferral_fraction.value, 0.0, 1.0)
    eta_value = _scalar_value(None if eta.value is None else eta.value * _DOLLAR_SCALE)
    cvar_value = _scalar_value(None if objective_value is None else objective_value * _DOLLAR_SCALE)
    if (
        credit_value is None
        or withdrawal_values.shape != capacities.shape
        or not np.all(np.isfinite(withdrawal_values))
        or deferral_value is None
        or eta_value is None
        or cvar_value is None
    ):
        return OptimizationFailure("invalid_solution")
    if np.any(withdrawal_values < -_NUMERICAL_TOLERANCE) or np.any(withdrawal_values > capacities + _NUMERICAL_TOLERANCE * np.maximum(1, capacities)):
        return OptimizationFailure("invalid_solution")
    withdrawal_values = np.clip(withdrawal_values, 0, capacities)
    quote = quote_withdrawals(state, tuple(WithdrawalAllocation(u.key, float(x)) for u, x in zip(units, withdrawal_values)), tax_assumptions)
    if deferred_cost_per_fraction == 0.0:
        # A free auxiliary choice cannot imply reducing nonexistent spending.
        deferral_value = 0.0

    adjusted_balance_value = (
        float(state.immediate_funding)
        + baseline_cash
        + credit_value * credit_effect_matrix
        + quote.net_cash * liquidation_effect
        + deferral_value * discretionary_savings
    )
    if not np.all(np.isfinite(adjusted_balance_value)):
        return OptimizationFailure("invalid_solution")

    free_withdrawal = float(sum(x for u, x in zip(units, withdrawal_values) if u.tax_per_dollar + u.penalty_per_dollar == 0))
    net_target = quote.net_cash
    if free_withdrawal > 0:
        trimmed = _trim_redundant_liquidation(
            adjusted_balance_value, free_withdrawal, settlement_column,
            operating_buffer, buffer_tolerance,
            w, tail_deficit_limit, q,
            buffer_coverage_target,
        )
        net_target -= free_withdrawal - trimmed
    # Resolve equal-cost allocations deterministically, preserving retirement
    # funds when an equally cheap taxable sale provides the same net cash.
    try:
        quote = quote_withdrawals(state, allocate_net_cash(units, max(0.0, net_target)), tax_assumptions)
    except ValueError:
        return OptimizationFailure("invalid_solution")
    adjusted_balance_value = (float(state.immediate_funding) + baseline_cash
        + credit_value * credit_effect_matrix + quote.net_cash * liquidation_effect
        + deferral_value * discretionary_savings)

    actual_buffer_shortfall = float(
        np.dot(w, np.maximum(0.0, operating_buffer - adjusted_balance_value).sum(axis=1))
    )
    buffer_feasibility_tolerance = _NUMERICAL_TOLERANCE * max(1.0, abs(buffer_tolerance))
    if actual_buffer_shortfall > buffer_tolerance + buffer_feasibility_tolerance:
        return OptimizationFailure("invalid_solution")
    actual_tail_deficit = cvar(np.maximum(0.0, operating_buffer - adjusted_balance_value.min(axis=1)), q, w)
    if buffer_coverage_target is not None and cvar(operating_buffer - adjusted_balance_value.min(axis=1), buffer_coverage_target, w) > MONEY_TOLERANCE:
        return OptimizationFailure("invalid_solution")
    if tail_deficit_limit is not None and actual_tail_deficit > tail_deficit_limit + _NUMERICAL_TOLERANCE * max(1.0, tail_deficit_limit):
        return OptimizationFailure("invalid_solution")

    actual_overdraft_cost = (overdraft_apr / 365.0) * np.maximum(
        0.0, -adjusted_balance_value
    ).sum(axis=1)
    withdrawal_cost_value = quote.tax_reserve + quote.penalty_reserve
    path_cost_value = (
        (interest_per_credit_dollar + advance_fee) * credit_value
        + discretionary_savings[:, -1] * deferral_value
        + withdrawal_cost_value
        + actual_overdraft_cost
    )
    if not np.all(np.isfinite(path_cost_value)):
        return OptimizationFailure("invalid_solution")
    if q == 1.0:
        maximum_path_cost = float(np.max(path_cost_value[w > 0]))
        eta_value += (interest_per_credit_dollar + advance_fee) * credit_value + withdrawal_cost_value
        eta_feasibility_tolerance = _NUMERICAL_TOLERANCE * max(
            1.0, abs(eta_value), abs(maximum_path_cost)
        )
        if eta_value < maximum_path_cost - eta_feasibility_tolerance:
            return OptimizationFailure("invalid_solution")

    expected_cost = float(np.dot(w, path_cost_value))
    # Report observed variation in this plan's costs, including cash-flow
    # risk without market history. Market data alone cannot make this true.
    cost_is_path_dependent = bool(
        np.ptp(path_cost_value)
        > _NUMERICAL_TOLERANCE * max(1.0, float(np.max(np.abs(path_cost_value))))
    )
    risk = balance_risk(adjusted_balance_value, operating_buffer, q, w)
    cash_shortfall_probability = risk["cash_shortfall_probability"]
    if not isfinite(expected_cost) or not isfinite(cash_shortfall_probability):
        return OptimizationFailure("invalid_solution")

    dual_value = _scalar_value(buffer_dual_value)
    if dual_value is None:
        implied_liquidity_price = None
    elif dual_value < -_NUMERICAL_TOLERANCE:
        return OptimizationFailure("invalid_solution")
    else:
        implied_liquidity_price = max(0.0, dual_value)
    credit_dual = _scalar_value(credit_dual_value)
    if credit_dual is not None and credit_dual < -_NUMERICAL_TOLERANCE:
        return OptimizationFailure("invalid_solution")
    implied_credit_price = max(0.0, credit_dual) if credit_dual is not None else None

    result = OptimalPlan(
        credit_draw=credit_value,
        liquidation_amount=quote.gross,
        deferral_fraction=deferral_value,
        cvar_cost=cvar(path_cost_value, q, w),
        var_cost=quantile(path_cost_value, q, w),
        expected_cost=expected_cost,
        cash_shortfall_probability=cash_shortfall_probability,
        implied_liquidity_price=implied_liquidity_price,
        cost_is_path_dependent=cost_is_path_dependent,
        solver_status=status,
        evaluation_horizon_days=evaluation_bundle.horizon_days,
        evaluation_draw_id=evaluation_bundle.bootstrap_draw_id,
        evaluation_paths=n_paths,
        cost_coverage_target=q,
        buffer_breach_probability=risk["buffer_breach_probability"],
        dollar_days_below_buffer=actual_buffer_shortfall,
        buffer_tolerance_dollar_days=buffer_tolerance,
        buffer_constraint_binding=abs(buffer_tolerance - actual_buffer_shortfall) <= buffer_feasibility_tolerance,
        evaluation_weight_hash=weight_hash(w),
        tail_deficit=actual_tail_deficit,
        tail_deficit_limit=tail_deficit_limit,
        implied_credit_price=implied_credit_price,
        credit_constraint_binding=abs(available_credit - credit_value) <= _bound_tolerance(0, available_credit),
        objective_kind=objective_kind,
        withdrawal_accounts=quote.accounts,
        withdrawal_allocations=quote.allocations,
        withdrawal_net_cash=quote.net_cash,
        withdrawal_tax_reserve=quote.tax_reserve,
        withdrawal_penalty_reserve=quote.penalty_reserve,
        solver_method=solver_method,
        credit_utilization=((primary_account.current_balance + credit_value) / primary_account.credit_limit
                            if primary_account is not None and primary_account.credit_limit > 0 else 0),
        buffer_coverage_target=buffer_coverage_target,
        credit_account_id=primary_account.account_id if primary_account else None,
        decision_horizon_days=decision_horizon_days or bundle.horizon_days,
        settlement_days=resolved_funding_config.settlement_days,
        external_transfer_days=resolved_funding_config.external_transfer_days,
        use_business_days=resolved_funding_config.use_business_days,
    )
    from dataclasses import replace, asdict
    from importlib.metadata import version
    from ginseng.verification import RiskContract, executable_from_optimal, verify_optimal
    from ginseng.provenance import digest
    solver_version = version('clarabel')
    if solver_method.startswith('highs'):
        try:
            from scipy.optimize._highspy._core import _Highs
            solver_version = _Highs().version()
        except (ImportError, AttributeError):
            solver_version = 'unavailable (SciPy adapter ' + version('scipy') + ')'
    solver_evidence.update(solver=solver_method, solver_version=solver_version,
        scipy_adapter_version=version('scipy') if solver_method.startswith('highs') else None,
        evaluation_draw_id=evaluation_bundle.bootstrap_draw_id,evaluation_weight_hash=weight_hash(w),
        objective_kind=objective_kind,mean_buffer_allowance=buffer_tolerance,
        tail_deficit_limit=tail_deficit_limit,signed_margin_coverage=buffer_coverage_target,
        time_limit_seconds=time_limit_seconds, scenario_day_limit=MAX_SCENARIO_DAYS,
        reported_objective=cvar_value, executed_objective=result.cvar_cost if objective_kind=='cvar' else result.expected_cost,
        objective_units='dollars', plan_identity=digest(asdict(executable_from_optimal(result,capital_gains_rate))),
        postprocessing='clip to bounds; reconstruct net quotes; trim free withdrawals; equal charges prefer taxable, Roth, traditional, then unit key',
        sensitivities=dict(buffer=dict(value=implied_liquidity_price,units='dollars / dollar-day',sign='objective change approximately -value * allowance increase'),
                           credit=dict(value=implied_credit_price,units='dollars / dollar',sign='objective change approximately -value * available-credit increase')))
    solver_evidence['postprocessing_objective_difference']=solver_evidence['executed_objective']-cvar_value
    if solver_evidence.get('lower_bound') is not None:
        solver_evidence['absolute_gap']=max(0.,solver_evidence['executed_objective']-solver_evidence['lower_bound'])
        solver_evidence['relative_gap']=solver_evidence['absolute_gap']/max(1.,abs(solver_evidence['executed_objective']))
    result=replace(result,solver_evidence=solver_evidence)
    contract=RiskContract(operating_buffer,q,buffer_tolerance,tail_deficit_limit,buffer_coverage_target,
        max_cash_shortfall_probability,resolved_funding_policy.max_buffer_breach_probability,max_credit_utilization,
        overdraft_apr,objective_kind)
    verification=verify_optimal(state,evaluation_bundle,obligations,result,contract,w,capital_gains_rate)
    if verification.status!='verified':
        return OptimizationFailure('invalid_solution',asdict(verification))
    solver_evidence['independent_primal_checks']=[asdict(c) for c in verification.constraints]
    return replace(result, solver_evidence=solver_evidence, verification=asdict(verification), **policy_assessment(result, resolved_funding_policy))
