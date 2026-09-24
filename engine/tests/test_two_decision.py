"""Independent arithmetic and information-set tests for the bounded experiment."""

import asyncio
import json
from dataclasses import asdict, replace
from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest
from ginseng.two_decision import (
    VALIDATION_DOMAIN,
    ZERO,
    Action,
    Model,
    Observation,
    Policy,
    Scenarios,
    actions,
    canonical,
    execute,
    node_for,
    review_action,
    run_experiment,
    select_policies,
    summarize,
)


def direct_available(model, flow, price, root, review):
    """Independent event-sum reference: no production ledger or reductions."""
    m = asdict(model)
    x = lambda k: F(str(m[k]))
    transactions = [(0, root), (model.review_day, review)]
    balances = []
    for t in range(1, model.material_horizon + 1):
        balance = x("opening_cash") + sum(F(str(a)) for a in flow[:t])
        for day, a in transactions:
            if day > t:
                continue
            draw = F(str(a.credit_draw))
            px = F(str(price[day]))
            balance += draw - (x("credit_fee") if draw else 0)
            if a.sell_units and day + model.settlement_days <= t:
                gross = a.sell_units * px
                charges = (
                    x("sale_fee")
                    + x("sale_fee_rate") * gross
                    + x("capital_gains_rate")
                    * a.sell_units
                    * max(0, px - x("cost_basis"))
                )
                balance += gross - charges
            if t >= model.repayment_day:
                days = model.repayment_day - max(day, 1) + 1
                balance -= draw + draw * x("credit_apr") * days / 365
        balances.append(balance)
    return balances


def independent_loss(model, flow, price, root, review):
    balances = direct_available(model, flow, price, root, review)
    wealth = balances[-1] + (model.holdings - root.sell_units - review.sell_units) * F(
        str(price[-1])
    )
    passive = (
        F(str(model.opening_cash))
        + sum(F(str(v)) for v in flow)
        + model.holdings * F(str(price[-1]))
    )
    return (
        passive
        - wealth
        + F(str(model.liquidity_charge)) * sum(max(0, -v) for v in balances)
    )


def independent_grid(model, scenarios, kind):
    # Full joint product; no conditional minimization or production action generator.
    roots = [
        Action(u, d)
        for u, d in product(range(model.holdings + 1), (0.0, model.credit_limit))
    ]
    best = None
    labels = (
        ["steady", "steady", "tight", "tight"]
        if kind == "nonanticipative"
        else list(range(4))
    )
    groups = list(dict.fromkeys(labels))
    for root in roots:
        options = [
            Action(u, d)
            for u, d in product(
                range(model.holdings - root.sell_units + 1),
                (0.0,) if root.credit_draw else (0.0, model.credit_limit),
            )
        ]
        mappings = (
            [(ZERO,) * len(groups)]
            if kind == "static"
            else product(options, repeat=len(groups))
        )
        for values in mappings:
            total = sum(
                F(str(w))
                * independent_loss(model, f, p, root, values[groups.index(label)])
                for w, f, p, label in zip(
                    scenarios.weights, scenarios.flows, scenarios.prices, labels
                )
            )
            if best is None or total < best:
                best = total
    return float(best)


@pytest.mark.parametrize("kind", ["static", "nonanticipative", "hindsight"])
def test_independent_full_product_matches_conditional_grid(kind):
    m = replace(Model(), holdings=1)
    s = canonical(m)
    policies, evidence = select_policies(m, s)
    actual = summarize(m, s, policies[kind])
    assert actual["objective_dollars"] == pytest.approx(
        independent_grid(m, s, kind), abs=1e-9
    )
    assert evidence[kind]["objective_gap_dollars"] <= 1e-9


def test_every_tiny_action_matches_independent_day_arithmetic():
    m = Model()
    s = canonical(m)
    for root in actions(m):
        for later in actions(m, root):
            result = execute(
                m, s, Policy("static", m.identity, root), review_override=later
            )
            for i, (flow, price) in enumerate(zip(s.flows, s.prices)):
                expected = [
                    float(v) for v in direct_available(m, flow, price, root, later)
                ]
                np.testing.assert_allclose(
                    result["arrays"]["available_cash"][i], expected, atol=1e-10
                )
                assert result["loss"][i] == pytest.approx(
                    float(independent_loss(m, flow, price, root, later)), abs=1e-9
                )


def test_shared_observations_share_actions_and_future_cannot_change_frozen_rule():
    m = replace(Model(), sale_fee=0)
    s = canonical(m)
    policy = select_policies(m, s)[0]["nonanticipative"]
    before = policy.identity
    result = execute(m, s, policy)
    assert result["review_actions"][0] == result["review_actions"][1]
    assert result["review_actions"][2] == result["review_actions"][3]
    assert result["observations"][0] == result["observations"][1]
    flows = s.flows.copy()
    prices = s.prices.copy()
    flows[:, m.review_day - 1 :] -= (
        700  # Includes current-day flow: not observed at review.
    )
    prices[:, m.review_day + 1 :] *= 0.5
    altered = execute(m, Scenarios(flows, prices, s.weights), policy)
    assert (
        altered["review_actions"] == result["review_actions"]
        and policy.identity == before
    )
    assert altered["observations"] == result["observations"]
    with pytest.raises(ValueError, match="Hindsight"):
        review_action(
            select_policies(m, s)[0]["hindsight"], result["observations"][0], m
        )


def test_unknown_review_state_has_frozen_fallback():
    m = Model()
    p = Policy("nonanticipative", m.identity, ZERO, (("steady/high", Action(1, 0)),))
    known = Observation((20,), 105.0, 80.0, 2, 0.0, 0.0, 0.0)
    assert review_action(p, known, m) == (Action(1, 0), False)
    assert review_action(p, replace(known, current_price=95), m) == (
        ZERO,
        True,
    )  # Absent node.
    assert review_action(p, replace(known, realized_flows=(101,)), m) == (
        ZERO,
        True,
    )  # Outside declared support.
    with pytest.raises(ValueError, match="information"):
        node_for(replace(known, realized_flows=(20, 200)), m)


def test_settlement_reserved_charges_and_current_execution_price():
    m = replace(Model(), opening_cash=0, holdings=1, credit_limit=0, cost_basis=80)
    flow = np.zeros((1, m.material_horizon))
    price = np.full((1, m.material_horizon + 1), 100.0)
    s = Scenarios(flow, price, [1.0])
    p = Policy("static", m.identity, Action(1, 0))
    e = execute(m, s, p)
    a = e["arrays"]
    assert a["available_cash"][0, 0] == 0 and a["unsettled_gross"][0, 0] == 100
    assert a["bank_cash"][0, 1] == 100 and a["reserved_charges"][0, 1] == 9
    assert a["available_cash"][0, 1] == 91 and a["charge_payments"][0, -1] == 9
    assert a["available_cash"][0, -1] == 91 and a["reserved_charges"][0, -1] == 0
    assert summarize(m, s, p)["max_conservation_residual_dollars"] < 1e-10
    price[0, m.review_day] = 105
    s = Scenarios(flow, price, [1.0])
    e = execute(
        m,
        s,
        Policy("nonanticipative", m.identity, ZERO, (("steady/high", Action(1, 0)),)),
    )
    assert e["trades"][0][0]["execution_price"] == 105
    assert e["trades"][0][0]["available_day"] == 4
    assert (
        e["arrays"]["available_cash"][0, 2] == 0
    )  # Sale is still unavailable day three.


def test_late_repayment_no_free_terminal_debt_and_strict_cash_boundary():
    m = replace(Model(), opening_cash=0)
    s = Scenarios(
        np.zeros((1, m.material_horizon)),
        np.full((1, m.material_horizon + 1), 100.0),
        [1.0],
    )
    p = Policy("static", m.identity, Action(0, 100))
    e = execute(m, s, p)
    summary = summarize(m, s, p, e)
    assert summary["visible_cash_failure_probability"] == 0
    assert summary["cash_failure_probability"] == 1
    assert e["arrays"]["repayments"][0, -1] == pytest.approx(100.6)
    assert e["wealth"][0] == pytest.approx(197.4)
    assert e["arrays"]["debt"][0, -1] == 0 and summary[
        "expected_explicit_cost_dollars"
    ] == pytest.approx(2.6)
    zero = summarize(m, s, Policy("static", m.identity, ZERO))
    assert zero["cash_failure_probability"] == 0  # Cash == 0 is not failure.
    flow = s.flows.copy()
    flow[0, 0] = -1
    flow[0, 1] = 2
    recovery = summarize(
        m, Scenarios(flow, s.prices, [1.0]), Policy("static", m.identity, ZERO)
    )
    assert recovery["cash_failure_probability"] == 1


def test_capacity_and_ownership_guards_and_no_asset_giveaway_reward():
    m = replace(Model(), sale_fee=0, sale_fee_rate=0, capital_gains_rate=0)
    s = Scenarios(np.zeros((1, 6)), np.full((1, 7), 100.0), [1.0])
    for units in range(3):
        row = summarize(m, s, Policy("static", m.identity, Action(units, 0)))
        assert (
            row["expected_terminal_wealth_dollars"] == 260
            and row["objective_dollars"] == 0
        )
    with pytest.raises(ValueError, match="root"):
        execute(m, s, Policy("static", m.identity, Action(3, 0)))
    with pytest.raises(ValueError, match="capacity"):
        execute(
            m,
            s,
            Policy(
                "nonanticipative",
                m.identity,
                Action(1, 100),
                (("steady/high", Action(2, 100)),),
            ),
        )
    with pytest.raises(ValueError):
        s.flows.setflags(write=True)
    bad = s.prices.copy()
    bad[0, 0] = 99
    with pytest.raises(ValueError, match="today"):
        execute(m, Scenarios(s.flows, bad, [1.0]), Policy("static", m.identity, ZERO))


def test_holdout_is_drawn_only_after_freezing_all_training_policies(monkeypatch):
    import ginseng.two_decision as mod

    original_draw, original_select = mod.draw_support, mod.select_policies
    fitted = []

    def select(*a, **kw):
        result = original_select(*a, **kw)
        fitted.append(tuple(p.identity for p in result[0].values()))
        return result

    def draw(s, n, root, domain, replicate=0):
        if domain == VALIDATION_DOMAIN:
            assert len(fitted) == 3
        return original_draw(s, n, root, domain, replicate)

    monkeypatch.setattr(mod, "select_policies", select)
    monkeypatch.setattr(mod, "draw_support", draw)
    report, capture = run_experiment(
        training_paths=16, validation_paths=64, replications=3
    )
    assert len(fitted) == 3
    assert (
        report["validation_stream"]["domain"] != report["training_streams"][0]["domain"]
    )
    assert len(report["training_replications"]) == 9 and len(report["comparison"]) == 6
    assert report["summary"]["selection"].startswith("Replication zero")
    for d in report["stream_disjointness"]:
        assert d["status"] == "disjoint_initializations"


def test_nonunique_flat_grid_compares_objective_and_feasibility():
    m = replace(
        Model(),
        sale_fee=0,
        sale_fee_rate=0,
        capital_gains_rate=0,
        credit_fee=0,
        credit_apr=0,
        liquidity_charge=0,
    )
    s = Scenarios(np.zeros((1, 6)), np.full((1, 7), 100.0), [1.0])
    chosen, evidence = select_policies(m, s)
    for p in chosen.values():
        assert summarize(m, s, p)["objective_dollars"] == 0
    # A different optimal plan is also correct; coefficients need not agree.
    alternative = Policy("static", m.identity, Action(2, 100))
    assert summarize(m, s, alternative)["objective_dollars"] == 0
    assert evidence["static"]["enumerated_minimum_dollars"] == 0


def test_capture_replay_cli_and_corruption(tmp_path, capsys, monkeypatch):
    import socket

    from ginseng.cli import main
    from ginseng.two_decision_artifact import load, replay

    path = tmp_path / "run"
    assert (
        main(
            [
                "two-decision",
                "run",
                "--output",
                str(path),
                "--training-paths",
                "32",
                "--validation-paths",
                "64",
                "--replications",
                "2",
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["match"]
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **kw: pytest.fail("Network used")
    )
    assert replay(path)["match"]
    assert main(["two-decision", "replay", str(path)]) == 0
    capsys.readouterr()
    file = path / "experiment.json"
    data = json.loads(file.read_text())
    data["payload"]["validation_draws"]["indices"][0] = 999
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Corrupt"):
        load(path)


def test_tui_real_two_decision_run_and_replay(tmp_path, monkeypatch):
    from ginseng.studio_ui import ResearchStudio
    from ginseng.tui import GinsengApp
    from textual.widgets import Static, TextArea

    monkeypatch.setenv("GINSENG_MOTION", "0")

    async def finish(studio, pilot):
        for _ in range(100):
            if not studio.busy:
                break
            await pilot.pause(0.05)
        assert not studio.busy

    async def run():
        app = GinsengApp()
        async with app.run_test(size=(100, 38)) as pilot:
            await pilot.press("space")
            app.open_studio("two-decision")
            await pilot.pause()
            studio = app.query_one(ResearchStudio)
            studio.query_one("#studio-params", TextArea).load_text(
                json.dumps(
                    dict(
                        training_paths=32,
                        validation_paths=64,
                        replications=2,
                        capture_path=str(tmp_path / "run"),
                    )
                )
            )
            studio.launch()
            await finish(studio, pilot)
            assert len(studio.result["result"]["comparison"]) == 6
            assert "comparison" in studio.tables and "training_ledger" in studio.tables
            app.open_studio("two-decision-replay")
            await pilot.pause()
            studio.query_one("#studio-params", TextArea).load_text(
                json.dumps(dict(input=str(tmp_path / "run")))
            )
            studio.launch()
            await finish(studio, pilot)
            assert studio.result["result"]["match"]
            assert "Replay: MATCH" in str(
                studio.query_one("#studio-summary", Static).content
            )

    asyncio.run(run())


def test_replay_rejects_resigned_stale_ledger_and_invalid_stream(tmp_path):
    from ginseng.provenance import digest
    from ginseng.two_decision_artifact import capture, load, replay

    _, payload = run_experiment(training_paths=32, validation_paths=64, replications=1)
    capture(tmp_path / "run", payload)
    file = tmp_path / "run" / "experiment.json"
    data = json.loads(file.read_text())
    data["payload"]["report"]["training_ledger"][0]["available_cash"] += 1
    data["output_digest"] = digest(data["payload"]["report"])
    data.pop("integrity")
    data["integrity"] = digest(data)
    file.write_text(json.dumps(data))
    result = replay(tmp_path / "run")
    assert not result["match"] and any(
        "training_ledger" in x for x in result["mismatches"]
    )
    data["payload"]["validation_draws"]["stream"] = data["payload"]["training_draws"][
        0
    ]["stream"]
    data["input_identity"] = digest(
        {k: v for k, v in data["payload"].items() if k != "report"}
    )
    data.pop("integrity")
    data["integrity"] = digest(data)
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="stream"):
        load(tmp_path / "run")


def test_zero_weight_unseen_node_is_not_fitted_and_holdout_can_fail_policy():
    m = Model()
    base = canonical(m)
    training = Scenarios(base.flows, base.prices, [1, 0, 0, 0])
    policies, _ = select_policies(m, training)
    assert all(node == "steady/high" for node, _ in policies["nonanticipative"].rules)
    evaluation = summarize(m, base, policies["nonanticipative"])
    assert evaluation["fallback_probability"] == pytest.approx(0.5)
    assert evaluation["cash_failure_probability"] > 0


def test_model_caps_and_incompatible_or_mutated_policy_are_rejected():
    for kw in [
        dict(holdings=100),
        dict(settlement_days=0),
        dict(review_day=6),
        dict(credit_apr=float("nan")),
        dict(sale_fee_rate=2),
    ]:
        with pytest.raises(ValueError):
            Model(**kw)
    with pytest.raises(ValueError, match="model identity"):
        execute(Model(), canonical(), Policy("static", "wrong", ZERO))
    with pytest.raises(ValueError, match="root"):
        execute(Model(), canonical(), Policy("static", Model().identity, Action(1, 50)))


def test_policy_owns_immutable_rule_mapping():
    source = [["steady/high", Action(1, 0)]]
    m = Model()
    p = Policy("nonanticipative", m.identity, ZERO, source)
    identity = p.identity
    source[0][1] = Action(2, 100)
    source.append(["tight/low", ZERO])
    assert p.identity == identity and p.rules == (("steady/high", Action(1, 0)),)
