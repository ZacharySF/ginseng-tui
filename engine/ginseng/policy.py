"""Funding-policy decision layer (spec sections 42-45, 60).

Feasibility, Pareto filtering, and the funding policy are applied in that
order (spec 45) to pick one recommended plan out of the `PlanResult`s
`funding.evaluate_plan` produces, and to explain the choice in plain
language that names the binding constraint and the deciding priority —
never a weighted numeric score (spec 42, 45).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Callable, Sequence

from ginseng.funding import PlanResult
from ginseng.risk import MONEY_TOLERANCE, PROBABILITY_TOLERANCE

# --------------------------------------------------------------------------
# Pareto filtering (spec 43)
# --------------------------------------------------------------------------

# Every spec-42 dimension, explicitly declared as "smaller is better" (a
# plan is never scored, but every objective still needs a stated
# direction for dominance to be well defined). `realized_gain_loss`'s sign
# is not itself "worse" (a realized loss can be tax-advantaged; Ginseng
# never computes final tax liability), so its *magnitude* — the size of
# the potential taxable event — is what Pareto treats as a cost.
_OBJECTIVES: tuple[tuple[str, Callable[[PlanResult], float]], ...] = (
    ("cash_shortfall_probability", lambda r: r.cash_shortfall_probability),
    ("buffer_breach_probability", lambda r: r.buffer_breach_probability),
    ("buffer_duration", lambda r: r.dollar_days_below_buffer),
    ("tail_deficit", lambda r: r.tail_deficit),
    ("avg_cash_deficit_when_short", lambda r: r.avg_cash_deficit_when_short),
    ("new_debt", lambda r: r.new_debt),
    ("interest_exposure", lambda r: r.interest_exposure),
    ("investment_sold", lambda r: r.investment_sold),
    ("withdrawal_charges", lambda r: r.withdrawal_tax_reserve + r.withdrawal_penalty_reserve),
    ("taxable_event_size", lambda r: abs(r.realized_gain_loss)),
    ("deferred_spending", lambda r: r.deferred_spending),
)

_TOLERANCE = 1e-9


def _dominates(a: PlanResult, b: PlanResult) -> bool:
    """`a` dominates `b` (spec 43): no worse than `b` on every declared
    objective, and strictly better on at least one."""
    no_worse = all(metric(a) <= metric(b) + _TOLERANCE for _, metric in _OBJECTIVES)
    strictly_better = any(metric(a) < metric(b) - _TOLERANCE for _, metric in _OBJECTIVES)
    return no_worse and strictly_better


def _require_common_evaluation(results: Sequence[PlanResult]) -> None:
    if len({(r.evaluation_horizon_days, r.evaluation_draw_id, r.evaluation_weight_hash) for r in results}) > 1:
        raise ValueError("Funding plans must share the same evaluation horizon and draw bundle, and identical probability weights.")


def pareto_filter(results: Sequence[PlanResult]) -> list[PlanResult]:
    """Mark each plan dominated when another feasible plan is no worse on
    every objective and strictly better on at least one (spec 43). Only
    feasible plans can dominate; every plan's own feasibility is left
    untouched here (spec 45 removes infeasible plans separately)."""
    _require_common_evaluation(results)
    feasible_pool = [r for r in results if r.feasible]
    filtered: list[PlanResult] = []
    for candidate in results:
        dominator = next(
            (
                other
                for other in feasible_pool
                if other.id != candidate.id and _dominates(other, candidate)
            ),
            None,
        )
        filtered.append(
            replace(candidate, dominated=dominator is not None, dominated_by=dominator.id if dominator else None)
        )
    return filtered


# --------------------------------------------------------------------------
# Funding policy (spec 44)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FundingPolicy:
    """Hard numeric limits plus ordered tradeoff priorities.

    Structural funding-operation validity is not a policy toggle: candidate
    evaluation rejects an unavailable credit draw, an underfunded sale, or a
    missing account before this policy ranks alternatives.
    """

    max_cash_shortfall_probability: float = 0.05
    max_credit_utilization: float = 0.30
    priorities: tuple[str, ...] = (
        "avoid_interest_bearing_debt",
        "minimize_taxable_sales",
        "minimize_deferred_spending",
    )
    capital_gains_rate: float = 0.15
    overdraft_apr: float = 0.0
    max_buffer_breach_probability: float | None = None
    buffer_tolerance_dollar_days: float | None = None
    tail_deficit_limit: float | None = None


@dataclass(frozen=True)
class Recommendation:
    plan_id: str
    explanation: str


_PRIORITY_INFO: dict[str, dict[str, str]] = {
    "avoid_interest_bearing_debt": {
        "metric": "interest_exposure",
        "comparative": "lowest-interest-exposure",
        "avoided": "creating an interest-bearing revolving balance",
    },
    "minimize_taxable_sales": {
        "metric": "investment_sold",
        "comparative": "lowest-investment-sale",
        "avoided": "a taxable investment sale",
    },
    "minimize_deferred_spending": {
        "metric": "deferred_spending",
        "comparative": "lowest-deferred-spending",
        "avoided": "deferred discretionary spending",
    },
}


def _meets_hard_requirements(result: PlanResult, policy: FundingPolicy) -> tuple[bool, str | None]:
    """Check the active frequency, deficit, and credit limits.

    Structural funding-operation feasibility is already enforced by
    `evaluate_plan`; `recommend` removes infeasible candidates before this
    function is called.
    """
    if policy.max_buffer_breach_probability is not None and result.buffer_breach_probability > policy.max_buffer_breach_probability + PROBABILITY_TOLERANCE:
        return False, (
            f"its buffer-breach probability of {result.buffer_breach_probability:.2%} exceeds "
            f"your {policy.max_buffer_breach_probability:.2%} limit"
        )
    for field, limit, label in (
        ("dollar_days_below_buffer", policy.buffer_tolerance_dollar_days, "average buffer deficit in dollar-days"),
        ("tail_deficit", policy.tail_deficit_limit, "tail buffer deficit"),
    ):
        if limit is not None and getattr(result, field) > limit + MONEY_TOLERANCE:
            return False, f"its {label} of {getattr(result, field):,.2f} exceeds your {limit:,.2f} limit"
    if result.cash_shortfall_probability > policy.max_cash_shortfall_probability + PROBABILITY_TOLERANCE:
        return False, (
            f"its modeled cash-shortfall probability of {result.cash_shortfall_probability:.0%} exceeds "
            f"your {policy.max_cash_shortfall_probability:.0%} limit"
        )
    new_credit = getattr(result, "new_debt", getattr(result, "credit_draw", 0.0))
    if new_credit > MONEY_TOLERANCE and result.credit_utilization > policy.max_credit_utilization + PROBABILITY_TOLERANCE:
        return False, (
            f"its credit utilization of {result.credit_utilization:.0%} exceeds your "
            f"{policy.max_credit_utilization:.0%} limit"
        )
    return True, None


def policy_assessment(result, policy: FundingPolicy) -> dict:
    """One acceptance rule for named plans, optimizer results and holdouts."""
    if not getattr(result, "feasible", True):
        return {"meets_policy": False, "policy_reason": result.infeasibility_reason or "Funding operation unavailable."}
    meets, reason = _meets_hard_requirements(result, policy)
    return {"meets_policy": meets, "policy_reason": reason}


def _binding_constraint_text(
    chosen: PlanResult, rejected: Sequence[tuple[PlanResult, str | None]], policy: FundingPolicy
) -> str:
    """Name the hard requirement that actually excluded the most rivals
    (the constraint the recommendation is binding against). If no rival
    was excluded by a hard requirement — the winner was decided purely by
    priority ordering — fall back to whichever of the chosen plan's own
    two constraints sits closest to its limit."""
    if policy.max_buffer_breach_probability is not None:
        return (f"keeps the buffer in {1 - chosen.buffer_breach_probability:.2%} of evaluated futures "
                f"and meets the other funding limits")
    shortfall_rejections = sum(
        1 for r, _ in rejected if r.cash_shortfall_probability > policy.max_cash_shortfall_probability
    )
    utilization_rejections = sum(
        1 for r, _ in rejected if r.credit_utilization > policy.max_credit_utilization
    )
    if shortfall_rejections == 0 and utilization_rejections == 0:
        shortfall_ratio = (
            chosen.cash_shortfall_probability / policy.max_cash_shortfall_probability
            if policy.max_cash_shortfall_probability > 0
            else float("inf")
        )
        utilization_ratio = (
            chosen.credit_utilization / policy.max_credit_utilization
            if policy.max_credit_utilization > 0
            else float("inf")
        )
        prefer_utilization = utilization_ratio >= shortfall_ratio
    else:
        prefer_utilization = utilization_rejections > shortfall_rejections
    if prefer_utilization:
        return (
            f"keeps credit utilization at {chosen.credit_utilization:.0%}, within your "
            f"{policy.max_credit_utilization:.0%} limit"
        )
    return (
        f"keeps modeled cash-shortfall probability at {chosen.cash_shortfall_probability:.2%}, within your "
        f"{policy.max_cash_shortfall_probability:.0%} limit"
    )



def _priority_value(result: PlanResult, priority: str, policy: FundingPolicy) -> float:
    """Return the policy-specific tradeoff value for one feasible plan."""
    if priority == "minimize_taxable_sales":
        # Sale principal and positive gains are distinct economic costs. The
        # user's tax rate makes the latter comparable without inventing a tax
        # payment date in the cash path.
        return result.investment_sold + result.withdrawal_tax_reserve + result.withdrawal_penalty_reserve
    if priority == "avoid_interest_bearing_debt":
        return result.interest_exposure + result.overdraft_interest_exposure
    info = _PRIORITY_INFO[priority]
    return float(getattr(result, info["metric"]))


def recommend(results: Sequence[PlanResult], policy: FundingPolicy) -> Recommendation:
    """Apply spec 45's process — remove infeasible plans, Pareto filter,
    apply the funding policy — and explain the winner in one sentence that
    names the binding constraint and the deciding priority. Never a
    weighted score."""
    if not results:
        return Recommendation(plan_id="", explanation="No candidate plans were generated.")

    _require_common_evaluation(results)
    feasible = [r for r in results if r.feasible]
    if not feasible:
        reason = results[0].infeasibility_reason or "no candidate plan can fund the gap in full"
        return Recommendation(plan_id="", explanation=f"No candidate plan is feasible: {reason}.")

    frontier = pareto_filter(feasible)
    non_dominated = [r for r in frontier if not r.dominated]

    rejected = [(r, _meets_hard_requirements(r, policy)[1]) for r in non_dominated]
    qualifying = [r for r in non_dominated if _meets_hard_requirements(r, policy)[0]]
    if not qualifying:
        # No non-dominated plan clears every hard requirement: relax to
        # the full feasible set so a plan and a binding constraint can
        # still be named, rather than reporting nothing.
        rejected = [(r, _meets_hard_requirements(r, policy)[1]) for r in feasible]
        qualifying = [r for r in feasible if _meets_hard_requirements(r, policy)[0]]

    if not qualifying:
        closest = min(feasible, key=lambda r: r.cash_shortfall_probability)
        _, reason = _meets_hard_requirements(closest, policy)
        return Recommendation(
            plan_id="",
            explanation=(
                f"{closest.label} is the closest available option, but no candidate plan satisfies every hard "
                f"requirement in your funding policy: {reason}."
            ),
        )

    ranked = qualifying
    deciding_priority: str | None = None
    satisfied_without: list[str] = []
    for priority in policy.priorities:
        info = _PRIORITY_INFO.get(priority)
        if info is None or len(ranked) <= 1:
            break
        best = min(_priority_value(result, priority, policy) for result in ranked)
        tied = [
            result
            for result in ranked
            if abs(_priority_value(result, priority, policy) - best) < _TOLERANCE
        ]
        if len(tied) < len(ranked):
            # This priority narrows the field; it becomes the deciding
            # priority unless a later priority narrows further. Keep
            # going so ties within `tied` still get broken by the
            # remaining, lower-ranked priorities.
            deciding_priority = priority
        elif best <= _TOLERANCE:
            satisfied_without.append(info["avoided"])
        ranked = tied

    chosen = ranked[0]
    rejected = [(r, reason) for r, reason in rejected if r.id not in {q.id for q in qualifying}]
    constraint_text = _binding_constraint_text(chosen, rejected, policy)
    if deciding_priority is not None:
        lead = f"it is the {_PRIORITY_INFO[deciding_priority]['comparative']} plan that {constraint_text}"
    else:
        lead = f"it {constraint_text}"
    tail = f" without {' or '.join(satisfied_without)}" if satisfied_without else ""
    explanation = f"{chosen.label} is recommended because {lead}{tail}."
    return Recommendation(plan_id=chosen.id, explanation=explanation)


# --------------------------------------------------------------------------
# Contract serialization (spec 60)
# --------------------------------------------------------------------------


def to_contract(
    results: Sequence[PlanResult], recommendation: Recommendation, policy: FundingPolicy | None = None
) -> tuple[list[dict], dict]:
    """Serialize evaluated plans and the recommendation to the frozen
    Plans Screen JSON shape (spec 60)."""
    _require_common_evaluation(results)
    frontier = pareto_filter([r for r in results if r.feasible])
    by_id = {r.id: r for r in frontier}

    plans: list[dict] = []
    for result in results:
        marked = by_id.get(result.id)
        if not result.feasible:
            explanation = result.infeasibility_reason or "This plan is not feasible."
            dominated = False
            dominated_by = None
        elif result.id == recommendation.plan_id:
            explanation = recommendation.explanation
            dominated = marked.dominated if marked else False
            dominated_by = marked.dominated_by if marked else None
        elif marked is not None and marked.dominated:
            dominated = True
            dominated_by = marked.dominated_by
            explanation = f"Another feasible plan ({dominated_by}) is no worse on every compared measure."
        else:
            explanation = "Tradeoff remains."
            dominated = False
            dominated_by = None

        plans.append(
            {
                "id": result.id,
                "label": result.label,
                "evaluation_horizon_days": result.evaluation_horizon_days,
                "evaluation_draw_id": result.evaluation_draw_id,
                "evaluation_weight_hash": result.evaluation_weight_hash,
                "cash_shortfall_probability": result.cash_shortfall_probability,
                "buffer_breach_probability": result.buffer_breach_probability,
                "dollar_days_below_buffer": result.dollar_days_below_buffer,
                "tail_deficit": result.tail_deficit,
                **(policy_assessment(result, policy) if policy is not None else {}),
                "infeasible_reason": result.infeasibility_reason,
                "avg_cash_deficit_when_short": result.avg_cash_deficit_when_short,
                "new_debt": result.new_debt,
                "interest_exposure": result.interest_exposure + result.overdraft_interest_exposure,
                "investment_sold": result.investment_sold,
                "withdrawal_tax_reserve": result.withdrawal_tax_reserve,
                "withdrawal_penalty_reserve": result.withdrawal_penalty_reserve,
                "withdrawal_net_cash": result.withdrawal_net_cash,
                "withdrawal_accounts": [asdict(row) for row in result.withdrawal_accounts],
                "realized_gain_loss": result.realized_gain_loss,
                "deferred_spending": result.deferred_spending,
                "feasible": result.feasible,
                "verification": result.verification,
                "dominated": dominated,
                "dominated_by": dominated_by,
                "recommended": result.id == recommendation.plan_id,
                "explanation": explanation,
            }
        )

    recommendation_dict = {"plan_id": recommendation.plan_id, "explanation": recommendation.explanation}
    return plans, recommendation_dict
