import json
from dataclasses import replace
from fractions import Fraction
from itertools import product

import numpy as np
import pytest
from ginseng.cli import main
from ginseng.conditional import (
    conditional_contributions,
    failure_probability,
    prepare_conditional,
    slow_contributions,
)
from ginseng.inputs import fixture
from ginseng.numerical import run_core
from ginseng.sampling import (
    PreparedHistory,
    map_indices,
    prepare_history,
    sample_bundle,
)
from ginseng.state import Obligation


def test_exact_trace_law():
    case = fixture("tiny")
    prepared = prepare_history(case.state, 7)
    tables = prepare_conditional(prepared, case.state, case.obligations)
    base = sample_bundle(prepared, 4, 1, 42, trace_initial_block=True)
    mean = second = ordinary_second = Fraction(0)
    for trace in product(range(4), repeat=3):
        points = np.zeros((1, 7))
        weight = Fraction(1)
        for t, branch in enumerate(trace, 1):
            points[0, 2 * t - 1] = 0 if branch == 0 else 0.99
            points[0, 2 * t] = max(0, branch - 1) / 3 + 1 / 6
            weight *= Fraction(6, 7) if branch == 0 else Fraction(1, 21)
        indices, k = map_indices(points, 3, 7, return_initial_lengths=True)
        bundle = replace(base, index_matrix=indices, initial_block_lengths=k)
        slow = slow_contributions(tables, bundle)[0]
        fast = conditional_contributions(tables, bundle)[0]
        assert fast == pytest.approx(slow, abs=1e-15)
        g = Fraction(round(3 * slow), 3)
        mean += weight * g
        second += weight * g * g
        ordinary_second += weight * g
    assert mean == Fraction(10846, 27783)
    assert float(
        (ordinary_second - mean * mean) / (second - mean * mean)
    ) == pytest.approx(9.53764365, rel=1e-5)


@pytest.mark.parametrize("method", ["mc", "sobol"])
def test_trace_preserves_draws_and_prefix(method):
    p = prepare_history(fixture("tiny").state, 7)
    a = sample_bundle(p, 4, 32, 42, method, 8)
    b = sample_bundle(p, 4, 32, 42, method, 8, trace_initial_block=True)
    c = sample_bundle(p, 8, 32, 42, method, 8, trace_initial_block=True)
    np.testing.assert_array_equal(a.material_indices, b.material_indices)
    np.testing.assert_array_equal(
        b.initial_block_lengths, np.minimum(c.initial_block_lengths, 4)
    )
    assert a.bootstrap_draw_id == b.bootstrap_draw_id
    with pytest.raises(ValueError):
        b.initial_block_lengths.setflags(write=True)
    # Restart on day 2 chooses exactly the continuation index.
    idx, k = map_indices([[0, 0.99, 0.5, 0, 0]], 3, 7, return_initial_lengths=True)
    assert idx.tolist() == [[0, 1, 2]]
    assert k.tolist() == [1]


@pytest.mark.parametrize("n,h", [(1, 1), (1, 8), (3, 1), (3, 9), (13, 5)])
@pytest.mark.parametrize("cash", [-30.0, 0.0, 30.0, 0.3])
def test_fast_slow_boundaries(n, h, cash):
    case = fixture("tiny")
    rng = np.random.default_rng(n + h)
    joint = rng.integers(0, 10, (n, 3)).astype(float) / 10
    prepared = PreparedHistory(joint, 7, 7, False, "test")
    # Explicit test construction isolates cash without mutating FinancialState.
    tables = prepare_conditional(
        prepared,
        case.state,
        (Obligation("a", "first", 0.1, 1), Obligation("z", "last", 0.2, h)),
        h,
    )
    from ginseng.conditional import _immutable

    eligible = tuple(
        _immutable(np.sort(s[cash + a >= 0]))
        for s, a in zip(tables.totals, tables.minima)
    )
    tables = replace(tables, cash=cash, eligible_totals=eligible)
    bundle = sample_bundle(prepared, h, 128, 91, trace_initial_block=True)
    np.testing.assert_allclose(
        conditional_contributions(tables, bundle),
        slow_contributions(tables, bundle),
        atol=1e-15,
    )


def test_cache_and_unsupported_modes():
    case = fixture("tiny")
    p = prepare_history(case.state, 7)
    b = sample_bundle(p, 4, 16, 42, trace_initial_block=True)
    tables = prepare_conditional(p, case.state, case.obligations)
    assert failure_probability(p, case.state, b, case.obligations, tables=tables) >= 0
    for changed, bills in [(p, ()), (replace(p, joint=p.joint + 1), case.obligations)]:
        with pytest.raises(ValueError, match="Stale"):
            failure_probability(changed, case.state, b, bills, tables=tables)
    with pytest.raises(ValueError, match="Stale"):
        failure_probability(
            p,
            case.state,
            b,
            case.obligations,
            tables=replace(tables, identity="other cash"),
        )
    with pytest.raises(ValueError, match="weights"):
        failure_probability(p, case.state, b, case.obligations, weights=np.ones(16))
    with pytest.raises(ValueError, match="prospective"):
        failure_probability(p, case.state, object())
    with pytest.raises(ValueError, match="traces"):
        conditional_contributions(tables, replace(b, initial_block_lengths=None))
    with pytest.raises(ValueError):
        sample_bundle(p, 4, 16, 42, "legacy_mc", trace_initial_block=True)


def test_cli_and_unchanged_other_metrics(tmp_path):
    case = fixture("tiny")
    p = prepare_history(case.state, 7)
    a, x, ordinary = run_core(case, p, "mc", 128, 42)
    b, y, conditional = run_core(case, p, "mc", 128, 42, estimator="initial-block-cmc")
    np.testing.assert_array_equal(x, y)
    for key in ordinary:
        if key != "cash_shortfall_probability":
            assert ordinary[key] == conditional[key]
    path = tmp_path / "result.json"
    assert (
        main(
            [
                "simulate",
                "--fixture",
                "tiny",
                "--estimator",
                "initial-block-cmc",
                "--out",
                str(path),
            ]
        )
        == 0
    )
    result = json.loads(path.read_text())
    assert (
        result["manifest"]["metric_estimators"]["cash_shortfall_probability"]
        == "initial-block-cmc"
    )
    assert (
        result["manifest"]["metric_estimators"]["avg_cash_deficit_when_short"] == "path"
    )
    assert result["manifest"]["initial_block_trace_hash"]
    assert (
        main(["simulate", "--estimator", "initial-block-cmc", "--sampler", "legacy_mc"])
        == 2
    )


def test_actual_cash_change_and_exact_zero():
    from ginseng.state import Transaction, TransactionType

    case = fixture("tiny")
    p = prepare_history(case.state, 7)
    b = sample_bundle(p, 4, 4, 42, trace_initial_block=True)
    tables = prepare_conditional(p, case.state, case.obligations)
    changed = replace(
        case.state,
        transactions=(
            *case.state.transactions,
            Transaction(case.state.as_of, TransactionType.TRANSFER, 1, "cash change"),
        ),
    )
    with pytest.raises(ValueError, match="Stale"):
        failure_probability(p, changed, b, case.obligations, tables=tables)
    p = PreparedHistory(np.array([[0.0, 30.0, 0.0]]), 7, 7, False, "test")
    b = sample_bundle(p, 1, 4, 42, trace_initial_block=True)
    assert failure_probability(p, case.state, b) == 0  # Opening cash 30, day flow -30.
    p = replace(p, joint=np.array([[0.0, 30.00000001, 0.0]]))
    assert failure_probability(p, case.state, b) == 1


def test_benchmark_artifacts(tmp_path):
    from ginseng.conditional_benchmark import benchmark

    config = dict(
        paths=[2048],
        replicates=2,
        root_seed=42,
        cases=["tiny", "zero-heavy"],
        reference_paths=128,
        reference_batch=128,
        block_length=14,
        probability_rmse_tolerance=0.005,
    )
    benchmark(config, tmp_path)
    rows = json.loads((tmp_path / "observations.json").read_text())
    assert len(rows) == 32
    for case in config["cases"]:
        for method in ("mc", "sobol"):
            group = [
                r
                for r in rows
                if r["case"] == case and r["sampler"] == method and r["replicate"] == 0
            ]
            assert len({r["index_hash"] for r in group}) == 1
    assert (tmp_path / "results.md").exists()
    assert json.loads((tmp_path / "manifest.json").read_text())["fixture_hash"]
