"""Versioned, synthetic, bounded two-decision experiment. No production optimizer changes.

Time zero and start-of-review-day controls share one asset and one facility.
The objective is an expectation, allowing conditional finite-grid enumeration.
"""

from dataclasses import asdict, dataclass
from itertools import product
from time import perf_counter

import numpy as np

from ginseng.execution import snapshot
from ginseng.provenance import digest
from ginseng.risk import cvar, probabilities
from ginseng.sampling import derive_seed

VERSION = "two-decision-grid-v1"
TRAIN_DOMAIN, VALIDATION_DOMAIN = 720, 721
NODES = ("tight/low", "tight/high", "steady/low", "steady/high")
TOL = 1e-9


@dataclass(frozen=True)
class Model:
    opening_cash: float = 60.0
    holdings: int = 2
    today_price: float = 100.0
    cost_basis: float = 100.0
    credit_limit: float = 100.0
    credit_apr: float = 0.365
    credit_fee: float = 2.0
    sale_fee: float = 3.0
    sale_fee_rate: float = 0.02
    capital_gains_rate: float = 0.2
    liquidity_charge: float = 1.0
    visible_horizon: int = 4
    review_day: int = 2
    settlement_days: int = 2
    repayment_day: int = 6
    version: str = VERSION

    def __post_init__(self):
        if self.version != VERSION:
            raise ValueError("Unsupported experimental model")
        for name in (
            "holdings",
            "visible_horizon",
            "review_day",
            "settlement_days",
            "repayment_day",
        ):
            v = getattr(self, name)
            if type(v) is not int or v < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.holdings > 3
            or self.review_day > self.visible_horizon
            or self.repayment_day <= self.review_day
            or self.material_horizon > 30
        ):
            raise ValueError(
                "Model exceeds bounded grid/horizon or repayment is before review"
            )
        for name in (
            "opening_cash",
            "today_price",
            "cost_basis",
            "credit_limit",
            "credit_apr",
            "credit_fee",
            "sale_fee",
            "sale_fee_rate",
            "capital_gains_rate",
            "liquidity_charge",
        ):
            v = getattr(self, name)
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not np.isfinite(v)
                or not 0 <= v <= 1e6
            ):
                raise ValueError(f"Invalid nonnegative {name}")
        if (
            self.today_price <= 0
            or self.capital_gains_rate > 1
            or self.sale_fee_rate > 1
        ):
            raise ValueError("Invalid price/rate")

    @property
    def material_horizon(self):
        return max(
            self.visible_horizon,
            self.repayment_day,
            self.review_day + self.settlement_days,
        )

    @property
    def identity(self):
        return digest(asdict(self))


@dataclass(frozen=True, order=True)
class Action:
    sell_units: int = 0
    credit_draw: float = 0.0

    def __post_init__(self):
        if (
            type(self.sell_units) is not int
            or self.sell_units < 0
            or isinstance(self.credit_draw, bool)
            or not np.isfinite(self.credit_draw)
            or self.credit_draw < 0
        ):
            raise ValueError("Invalid action amount")


ZERO = Action()


@dataclass(frozen=True)
class Scenarios:
    flows: np.ndarray
    prices: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        f, p = np.asarray(self.flows), np.asarray(self.prices)
        if (
            f.ndim != 2
            or min(f.shape) < 1
            or f.shape[0] > 128
            or f.shape[1] > 30
            or p.shape != (f.shape[0], f.shape[1] + 1)
        ):
            raise ValueError(
                "Scenarios require 1–128 rows, 1–30 days and day-zero prices"
            )
        if (
            not np.isfinite(f).all()
            or not np.isfinite(p).all()
            or (p <= 0).any()
            or max(np.max(abs(f)), np.max(p)) > 1e6
        ):
            raise ValueError("Invalid finite scenario cash/price values")
        for name, value in [
            ("flows", f),
            ("prices", p),
            ("weights", probabilities(len(f), self.weights)),
        ]:
            object.__setattr__(self, name, snapshot(value))

    @property
    def identity(self):
        return digest(
            dict(
                flows=self.flows.tolist(),
                prices=self.prices.tolist(),
                weights=self.weights.tolist(),
            )
        )


def validate_scenarios(model, scenarios):
    if scenarios.flows.shape[1] != model.material_horizon or not np.all(
        scenarios.prices[:, 0] == model.today_price
    ):
        raise ValueError(
            "Scenarios must share today price and span all settlements/repayments"
        )


def canonical(model=Model()):
    """Four deliberately synthetic leaves. No estimated market relationship."""
    h = model.material_horizon
    flows = np.zeros((4, h))
    prices = np.full((4, h + 1), model.today_price)
    flows[:, 0] = [20, 20, -20, -20]
    flows[:, model.review_day] = [
        -60,
        -180,
        -60,
        -180,
    ]  # Unknown at the review decision.
    flows[:, -1] += 100
    prices[0:2, model.review_day : h] = model.today_price * 1.05
    prices[2:4, model.review_day : h] = model.today_price * 0.95
    return Scenarios(flows, prices, np.array([0.4, 0.1, 0.1, 0.4]))


@dataclass(frozen=True)
class Observation:
    # Only realized exogenous cash through yesterday and today's opening quote.
    realized_flows: tuple[float, ...]
    current_price: float
    available_cash: float
    remaining_holdings: int
    outstanding_debt: float
    unsettled_gross: float
    reserved_charges: float


def node_for(observation, model):
    if len(observation.realized_flows) != model.review_day - 1:
        raise ValueError(
            "Review information cannot contain same-day or later cash flows"
        )
    net = float(sum(observation.realized_flows))
    price = observation.current_price
    if not np.isfinite([net, price]).all():
        raise ValueError("Invalid observed state")
    if (
        not -100 <= net <= 100
        or not 0.5 * model.today_price <= price <= 1.5 * model.today_price
    ):
        return "outside-support"
    return (
        ("tight" if net < 0 else "steady")
        + "/"
        + ("low" if price < model.today_price else "high")
    )


@dataclass(frozen=True)
class Policy:
    kind: str
    model_identity: str
    root: Action
    rules: tuple[tuple[str, Action], ...] = ()
    # Hindsight keys inspect the ENTIRE future and are forbidden for live execution.
    future_rules: tuple[tuple[str, Action], ...] = ()
    fallback: Action = ZERO

    def __post_init__(self):
        # Copy caller-owned lists; freezing a dataclass alone does not freeze them.
        object.__setattr__(self, "rules", tuple((k, a) for k, a in self.rules))
        object.__setattr__(
            self, "future_rules", tuple((k, a) for k, a in self.future_rules)
        )
        controls = (
            self.root,
            self.fallback,
            *(a for _, a in (*self.rules, *self.future_rules)),
        )
        if not all(isinstance(a, Action) for a in controls):
            raise ValueError("Policy controls must be immutable Actions")
        if (
            self.kind not in ("static", "nonanticipative", "hindsight")
            or self.fallback != ZERO
        ):
            raise ValueError("Unsupported policy class/fallback")
        if self.kind == "static" and (self.rules or self.future_rules):
            raise ValueError("Static controls are time-zero only")
        if self.kind != "hindsight" and self.future_rules:
            raise ValueError("Executable policies cannot inspect future keys")
        if self.kind == "hindsight" and self.rules:
            raise ValueError("Invalid hindsight mapping")
        if len({k for k, _ in self.rules}) != len(self.rules) or any(
            k not in NODES for k, _ in self.rules
        ):
            raise ValueError("Invalid or duplicate observation node")
        if len({k for k, _ in self.future_rules}) != len(self.future_rules):
            raise ValueError("Duplicate future key")

    @property
    def identity(self):
        return digest(asdict(self))


def review_action(policy, observation, model):
    if policy.model_identity != model.identity:
        raise ValueError("Policy belongs to another model")
    if policy.kind == "hindsight":
        raise ValueError("Hindsight is not an executable information-consistent policy")
    if policy.kind == "static":
        return ZERO, False
    node = node_for(observation, model)
    rules = dict(policy.rules)
    return rules.get(node, policy.fallback), node not in rules


def future_key(flows, prices):
    return digest(dict(flows=list(flows), prices=list(prices)))


def actions(model, root=ZERO):
    credits = (
        (0.0,)
        if root.credit_draw or not model.credit_limit
        else (0.0, model.credit_limit)
    )
    return tuple(
        Action(units, credit)
        for units, credit in product(
            range(model.holdings - root.sell_units + 1), credits
        )
    )


def validate_policy(model, policy):
    if policy.model_identity != model.identity or policy.root not in actions(model):
        raise ValueError("Invalid root control/model identity")
    allowed = actions(model, policy.root)
    if (
        any(a not in allowed for _, a in (*policy.rules, *policy.future_rules))
        or policy.fallback not in allowed
    ):
        raise ValueError(
            "Policy exceeds remaining asset/credit capacity or discrete controls"
        )


def execute(model, scenarios, policy, *, review_override=None):
    """Day-loop ledger. No solver auxiliaries; override is private enumeration input."""
    validate_scenarios(model, scenarios)
    validate_policy(model, policy)
    if review_override is not None and review_override not in actions(
        model, policy.root
    ):
        raise ValueError("Invalid review enumeration action")
    n, h = scenarios.flows.shape
    arrays = {
        k: np.zeros((n, h))
        for k in (
            "available_cash",
            "bank_cash",
            "debt",
            "holdings",
            "unsettled_gross",
            "unsettled_charges",
            "reserved_charges",
            "interest_due",
            "repayments",
            "sale_settlements",
            "charge_payments",
        )
    }
    losses = []
    wealths = []
    costs = []
    reviews = []
    fallbacks = []
    observations = []
    trades = []
    for i, (flow, price) in enumerate(zip(scenarios.flows, scenarios.prices)):
        bank = model.opening_cash
        units = model.holdings
        debt = interest = reserved = cost = 0.0
        pending = []
        events = []

        def trade(action, day):
            nonlocal bank, units, debt, cost, reserved
            if (
                action.sell_units > units
                or debt + action.credit_draw > model.credit_limit + TOL
            ):
                raise ValueError("Executable action exceeds inventory/credit")
            if action.sell_units:
                gross = action.sell_units * price[day]
                charge = (
                    model.sale_fee
                    + model.sale_fee_rate * gross
                    + model.capital_gains_rate
                    * action.sell_units
                    * max(0, price[day] - model.cost_basis)
                )
                units -= action.sell_units
                cost += charge
                pending.append((day + model.settlement_days, gross, charge))
                events.append(
                    dict(
                        day=day,
                        units=action.sell_units,
                        execution_price=float(price[day]),
                        gross=float(gross),
                        charges=float(charge),
                        available_day=day + model.settlement_days,
                    )
                )
            if action.credit_draw:
                bank += action.credit_draw - model.credit_fee
                debt += action.credit_draw
                cost += model.credit_fee

        trade(policy.root, 0)
        chosen = ZERO
        fallback = False
        obs = None
        for day in range(1, h + 1):
            settled = 0.0
            for due, gross, charge in pending:
                if due == day:
                    bank += gross
                    reserved += charge
                    settled += gross
            pending = [x for x in pending if x[0] > day]
            if day == model.review_day:
                obs = Observation(
                    tuple(float(x) for x in flow[: day - 1]),
                    float(price[day]),
                    float(bank - reserved),
                    units,
                    float(debt),
                    float(sum(x[1] for x in pending)),
                    float(reserved),
                )
                if review_override is not None:
                    chosen = review_override
                elif policy.kind == "hindsight":
                    key = future_key(flow, price)
                    chosen = dict(policy.future_rules).get(key, ZERO)
                    fallback = key not in dict(policy.future_rules)
                else:
                    chosen, fallback = review_action(policy, obs, model)
                trade(chosen, day)
            bank += flow[day - 1]
            interest += debt * model.credit_apr / 365
            repayment = 0.0
            if day == model.repayment_day:
                repayment = debt + interest
                bank -= repayment
                cost += interest
                debt = interest = 0.0
            paid = 0.0
            if day == h:
                paid = reserved
                bank -= reserved
                reserved = 0.0
            state = dict(
                available_cash=bank - reserved,
                bank_cash=bank,
                debt=debt,
                holdings=units,
                unsettled_gross=sum(x[1] for x in pending),
                unsettled_charges=sum(x[2] for x in pending),
                reserved_charges=reserved,
                interest_due=interest,
                repayments=repayment,
                sale_settlements=settled,
                charge_payments=paid,
            )
            for k, value in state.items():
                arrays[k][i, day - 1] = value
        terminal = (
            bank
            - reserved
            + sum(g - c for _, g, c in pending)
            + units * price[-1]
            - debt
            - interest
        )
        passive = model.opening_cash + float(sum(flow)) + model.holdings * price[-1]
        dollar_days = float(np.maximum(0, -arrays["available_cash"][i]).sum())
        losses.append(passive - terminal + model.liquidity_charge * dollar_days)
        wealths.append(terminal)
        costs.append(cost)
        reviews.append(chosen)
        fallbacks.append(fallback)
        observations.append(obs)
        trades.append(events)
    arrays = {k: snapshot(v) for k, v in arrays.items()}
    return dict(
        arrays=arrays,
        loss=snapshot(losses),
        wealth=snapshot(wealths),
        cost=snapshot(costs),
        review_actions=tuple(reviews),
        observations=tuple(observations),
        fallback=snapshot(fallbacks, "|u1"),
        trades=trades,
    )


def summarize(model, scenarios, policy, executed=None):
    e = executed or execute(model, scenarios, policy)
    w = scenarios.weights
    a = e["arrays"]
    cash = a["available_cash"]
    deficits = np.maximum(0, -cash.min(axis=1))
    dd = np.maximum(0, -cash).sum(axis=1)
    passive = (
        model.opening_cash
        + scenarios.flows.sum(axis=1)
        + model.holdings * scenarios.prices[:, -1]
    )
    # This conservation residual detects forgotten debt, fees or sale proceeds.
    sold_opportunity = np.array(
        [
            sum(
                t["units"] * (scenarios.prices[i, -1] - t["execution_price"])
                for t in events
            )
            for i, events in enumerate(e["trades"])
        ]
    )
    residual = passive - e["wealth"] - e["cost"] - sold_opportunity
    return dict(
        objective_dollars=float(w @ e["loss"]),
        expected_terminal_wealth_dollars=float(w @ e["wealth"]),
        expected_wealth_loss_dollars=float(w @ (passive - e["wealth"])),
        expected_explicit_cost_dollars=float(w @ e["cost"]),
        cash_failure_probability=float(w[cash.min(axis=1) < 0].sum()),
        visible_cash_failure_probability=float(
            w[cash[:, : model.visible_horizon].min(axis=1) < 0].sum()
        ),
        expected_negative_cash_dollar_days=float(w @ dd),
        expected_maximum_deficit_dollars=float(w @ deficits),
        deficit_cvar_95_dollars=cvar(deficits, 0.95, w),
        root_sale_units=policy.root.sell_units,
        root_credit_dollars=policy.root.credit_draw,
        expected_review_sale_units=float(
            w @ np.array([x.sell_units for x in e["review_actions"]])
        ),
        expected_review_credit_dollars=float(
            w @ np.array([x.credit_draw for x in e["review_actions"]])
        ),
        fallback_probability=float(w @ e["fallback"]),
        max_conservation_residual_dollars=float(np.max(abs(residual))),
        terminal_debt_max_dollars=float(a["debt"][:, -1].max()),
        terminal_unsettled_max_dollars=float(a["unsettled_gross"][:, -1].max()),
        min_holdings=float(a["holdings"].min()),
        max_credit_dollars=float(a["debt"].max()),
        policy_identity=policy.identity,
        executable=policy.kind != "hindsight",
    )


def select_policies(model, training):
    """Enumerate every root and conditional action in the declared finite grid.

    Expected loss separates across disjoint node groups. No tail-risk objective or
    aggregate chance constraint is silently assumed separable here.
    """
    validate_scenarios(model, training)
    candidates = {kind: [] for kind in ("static", "nonanticipative", "hindsight")}
    evaluated_pairs = 0
    for root in actions(model):
        base = Policy("static", model.identity, root)
        options = actions(model, root)
        evaluations = [
            execute(model, training, base, review_override=a) for a in options
        ]
        evaluated_pairs += len(options)
        loss = np.array([e["loss"] for e in evaluations])
        nodes = np.array([node_for(o, model) for o in evaluations[0]["observations"]])
        keys = [future_key(f, p) for f, p in zip(training.flows, training.prices)]
        for kind in candidates:
            chosen = np.zeros(len(training.flows), dtype=int)
            rules = []
            if kind != "static":
                labels = nodes if kind == "nonanticipative" else np.array(keys)
                for label in sorted(set(labels)):
                    mask = (labels == label) & (training.weights > 0)
                    if not mask.any() or (
                        kind == "nonanticipative" and label == "outside-support"
                    ):
                        continue
                    scores = loss[:, mask] @ training.weights[mask]
                    # Exact ties use the first lexicographic control; no missing branches.
                    best = int(np.argmin(scores))
                    chosen[mask] = best
                    rules.append((str(label), options[best]))
            p = Policy(
                kind,
                model.identity,
                root,
                tuple(rules) if kind == "nonanticipative" else (),
                tuple(rules) if kind == "hindsight" else (),
            )
            value = float(training.weights @ loss[chosen, np.arange(len(chosen))])
            candidates[kind].append((value, p))
    selected = {}
    evidence = {}
    for kind, rows in candidates.items():
        minimum = min(v for v, _ in rows)
        value, p = next((v, p) for v, p in rows if v <= minimum + TOL)
        selected[kind] = p
        evidence[kind] = dict(
            objective_dollars=value,
            enumerated_minimum_dollars=minimum,
            selection_tolerance_dollars=TOL,
            objective_gap_dollars=value - minimum,
            root_candidates=len(rows),
            evaluated_root_review_pairs=evaluated_pairs,
            scope="Finite supplied scenarios and discrete action grid only; expectation objective; no population or continuous-control optimum claim",
        )
    values = [
        evidence[k]["objective_dollars"]
        for k in ("hindsight", "nonanticipative", "static")
    ]
    if not values[0] <= values[1] + 5 * TOL or not values[1] <= values[2] + 5 * TOL:
        raise ArithmeticError("Finite-grid feasible-set inclusion failed")
    return selected, evidence


def stream_identity(root, domain, replicate):
    return dict(
        root_seed=root,
        domain=domain,
        replicate=replicate,
        derived_seed=derive_seed(root, "mc", replicate, domain),
        sampler="mc",
        sampler_version=1,
        mapping_version=1,
        purpose="two-decision training"
        if domain == TRAIN_DOMAIN
        else "two-decision validation",
        law="iid categorical inverse CDF over frozen synthetic leaves",
    )


def draw_support(support, n, root, domain, replicate=0):
    if type(n) is not int or not 1 <= n <= 65536:
        raise ValueError("Use 1–65,536 independent draws")
    identity = stream_identity(root, domain, replicate)
    u = np.random.Generator(np.random.PCG64(identity["derived_seed"])).random(n)
    cdf = np.cumsum(support.weights)
    cdf[-1] = 1.0
    indices = np.searchsorted(cdf, u, side="right").astype("int64")
    return subset(support, indices), snapshot(indices, "<i8"), identity


def subset(support, indices):
    idx = np.asarray(indices)
    if (
        idx.ndim != 1
        or not len(idx)
        or idx.dtype.kind not in "iu"
        or np.any((idx < 0) | (idx >= len(support.flows)))
    ):
        raise ValueError("Invalid captured leaf indices")
    rows, counts = np.unique(idx, return_counts=True)
    return Scenarios(support.flows[rows], support.prices[rows], counts / len(idx))


def information_contract(model):
    return dict(
        decision_times=[0, model.review_day],
        review_timing="Start of day, after earlier settlements, before current-day exogenous cash flow",
        observed="Realized cash through yesterday, current execution quote, cash/debt/remaining holdings and pending settlements from own past actions",
        grouping="Predeclared cumulative-realized-flow sign × current-price-below-today; four possible nodes. Same group shares one action.",
        fallback="No review action for absent training nodes or net observed flow outside [-100,100] / price outside [0.5,1.5] × today price. No holdout fitting.",
        hindsight="Common root action, future-keyed review actions: impossible as a real policy. Training-only lower bound by feasible-set inclusion.",
        objective="Expected (passive terminal wealth − policy terminal wealth) + liquidity_charge × expected negative-cash dollar-days, dollars",
        constraints="Integer sell units within remaining holdings; draws in {0, credit_limit} within shared facility; positive calendar-day settlement; full repayment and charge settlement in material horizon",
        tail_metric="95% empirical CVaR of maximum cash deficit is descriptive only; no dynamically time-consistent or precommitment CVaR objective is implemented",
        ordering="Settle earlier sales → observe/review/execute → exogenous cash flow → daily debt interest → scheduled principal/interest repayment → terminal reserved-charge payment → strict-negative end-of-day check",
    )


def training_trace(model, training, policies):
    training_execution = {k: execute(model, training, p) for k, p in policies.items()}
    ledger = []
    nodes = []
    trades = []
    for kind, e in training_execution.items():
        for i, obs in enumerate(e["observations"]):
            nodes.append(
                dict(
                    policy=kind,
                    leaf=i,
                    node=node_for(obs, model),
                    observed_price=obs.current_price,
                    observed_available_cash=obs.available_cash,
                    review_sale_units=e["review_actions"][i].sell_units,
                    review_credit_dollars=e["review_actions"][i].credit_draw,
                    observed=dict(asdict(obs), realized_flows=list(obs.realized_flows)),
                    review_action=asdict(e["review_actions"][i]),
                    fallback=bool(e["fallback"][i]),
                )
            )
            trades.extend(
                dict(policy=kind, leaf=i, **trade) for trade in e["trades"][i]
            )
            for day in range(model.material_horizon):
                ledger.append(
                    dict(
                        policy=kind,
                        leaf=i,
                        day=day + 1,
                        **{k: float(v[i, day]) for k, v in e["arrays"].items()},
                    )
                )
    return dict(
        training_review_nodes=nodes, training_ledger=ledger, training_trades=trades
    )


def run_experiment(
    *,
    model=Model(),
    training_paths=128,
    validation_paths=4096,
    root_seed=20260920,
    replications=5,
    cancelled=None,
):
    if type(replications) is not int or not 1 <= replications <= 10:
        raise ValueError("Use 1–10 predeclared training replications")
    from ginseng.decision_lab import independent_streams
    from ginseng.numerical import environment

    t = perf_counter()
    support = canonical(model)
    policies_by_rep = []
    training_sets = []
    draws = []
    rep_rows = []
    evidences = []
    for r in range(replications):
        if cancelled and cancelled():
            raise ValueError("Two-decision experiment cancelled before holdout")
        training, indices, identity = draw_support(
            support, training_paths, root_seed, TRAIN_DOMAIN, r
        )
        policies, evidence = select_policies(model, training)
        training_sets.append(training)
        draws.append(dict(indices=indices.tolist(), stream=identity))
        policies_by_rep.append(policies)
        evidences.append(evidence)
        for kind, policy in policies.items():
            rep_rows.append(
                dict(
                    replication=r,
                    policy=kind,
                    **summarize(model, training, policy),
                    rules=[dict(node=k, **asdict(a)) for k, a in policy.rules],
                )
            )
    frozen = [{k: p.identity for k, p in ps.items()} for ps in policies_by_rep]
    selected_time = perf_counter() - t
    t = perf_counter()
    # No validation randomness is constructed or consumed before ALL policies freeze.
    validation, validation_indices, val_identity = draw_support(
        support, validation_paths, root_seed, VALIDATION_DOMAIN
    )
    disjoint = [independent_streams(d["stream"], val_identity) for d in draws]
    comparison = []
    for kind, policy in policies_by_rep[0].items():
        tr = summarize(model, training_sets[0], policy)
        val = summarize(model, validation, policy)
        comparison.extend(
            [
                dict(collection="training", policy=kind, **tr),
                dict(collection="independent_validation", policy=kind, **val),
            ]
        )
    if frozen != [{k: p.identity for k, p in ps.items()} for ps in policies_by_rep]:
        raise ArithmeticError("Holdout mutated a selected policy")
    validation_time = perf_counter() - t
    policies = policies_by_rep[0]
    trace = training_trace(model, training_sets[0], policies)
    report = dict(
        version=VERSION,
        synthetic=True,
        model=asdict(model),
        model_identity=model.identity,
        summary=dict(
            scope="Offline synthetic two-decision experiment",
            training_paths=training_paths,
            validation_paths=validation_paths,
            material_horizon=model.material_horizon,
            visible_horizon=model.visible_horizon,
            review_day=model.review_day,
            training_static_objective=evidences[0]["static"]["objective_dollars"],
            training_nonanticipative_objective=evidences[0]["nonanticipative"][
                "objective_dollars"
            ],
            training_hindsight_bound=evidences[0]["hindsight"][
                "enumerated_minimum_dollars"
            ],
            selection="Replication zero fixed in advance; remaining training fits diagnose instability, never select on validation",
        ),
        information=information_contract(model),
        comparison=comparison,
        training_replications=rep_rows,
        policies={k: asdict(v) for k, v in policies.items()},
        solver_evidence=evidences[0],
        **trace,
        stability=[
            dict(
                policy=kind,
                distinct_root_actions=len({p[kind].root for p in policies_by_rep}),
                distinct_policies=len({p[kind].identity for p in policies_by_rep}),
                training_objective_min=min(
                    row["objective_dollars"]
                    for row in rep_rows
                    if row["policy"] == kind
                ),
                training_objective_max=max(
                    row["objective_dollars"]
                    for row in rep_rows
                    if row["policy"] == kind
                ),
            )
            for kind in ("static", "nonanticipative", "hindsight")
        ],
        validation_stream=val_identity,
        training_streams=[d["stream"] for d in draws],
        stream_disjointness=disjoint,
        timings=dict(
            training_and_selection_seconds=selected_time,
            holdout_seconds=validation_time,
        ),
        environment=environment(),
        limitations=[
            "Synthetic cash/price law, not household evidence or observed market dependence.",
            "Discrete restricted controls and four coarse observation bins; no real trading or production recommendations.",
            "Expectations and empirical tails only; no confidence intervals or population-optimality guarantee.",
            "Hindsight is an infeasible diagnostic. Its finite-training lower bound need not order frozen holdout policies.",
            "No holdout-based selection; no reoptimization per revealed validation path.",
        ],
    )
    capture = dict(
        model=asdict(model),
        support=dict(
            flows=support.flows.tolist(),
            prices=support.prices.tolist(),
            weights=support.weights.tolist(),
        ),
        training_draws=draws,
        validation_draws=dict(indices=validation_indices.tolist(), stream=val_identity),
        policies=[{k: asdict(p) for k, p in ps.items()} for ps in policies_by_rep],
        report=report,
    )
    return report, capture
