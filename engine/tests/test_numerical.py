from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from ginseng.exact import enumerate_exact
from ginseng.inputs import fixture, load_input
from ginseng.sampling import (
    prepare_history,
    sample_bundle,
    map_indices,
    unit_points,
    derive_seed,
)
from ginseng.simulate import cash_paths, DrawBundle, draw_bundle
from ginseng.metrics import cash_risk_summary
from ginseng.risk import quantile, cvar, tail_probabilities
from ginseng.funding import extend_draw_bundle
from ginseng.state import Obligation


def test_exact_independent_financial_integration():
    exact = enumerate_exact()
    assert exact["count"] == 81
    assert exact["rational"] == dict(
        failure="10846/27783", mean_deficit="574790/27783", total_probability="1"
    )
    assert exact["summary"]["required_liquidity_reserve"] == 90
    case = fixture("tiny")
    prepared = prepare_history(case.state, 7)
    idx = np.array([r["indices"] for r in exact["sequences"]])
    w = np.array([r["probability"] for r in exact["sequences"]])
    bundle = DrawBundle(0, 4, 81, 7, False, 3, idx, "exact")
    x = cash_paths(
        case.state, bundle, case.obligations, prepared_history=prepared.joint
    )
    np.testing.assert_array_equal(x, [r["cumulative"] for r in exact["sequences"]])
    actual = cash_risk_summary(x, 30, 10, 0.95, w)
    assert actual == pytest.approx(exact["summary"])
    assert actual["expected_max_cash_deficit"] == pytest.approx(
        actual["cash_shortfall_probability"] * actual["avg_cash_deficit_when_short"]
    )
    successor = next(r for r in exact["sequences"] if r["indices"] == [0, 1, 2, 0])
    assert (
        Fraction(successor["rational_probability"])
        == Fraction(1, 3) * Fraction(19, 21) ** 3
    )
    with pytest.raises(ValueError, match="enumeration"):
        enumerate_exact(net=tuple(range(20)), deterministic=(0,) * 10)


@pytest.mark.parametrize(
    "x,c,b,expected",
    [
        ([[1200]], 0, 1000, (0, 0, 0)),
        ([[100]], -50, 0, (0, 0, 0)),
        ([[0]], 0, 10, (10, 0, 0)),
        ([[-20, 100]], 0, 0, (20, 1, 20)),
        ([[10]], -50, 0, (0, 1, 40)),
    ],
)
def test_end_of_day_semantics(x, c, b, expected):
    r = cash_risk_summary(x, c, b, 0.95)
    assert (
        tuple(
            r[k]
            for k in (
                "required_liquidity_reserve",
                "cash_shortfall_probability",
                "expected_max_cash_deficit",
            )
        )
        == expected
    )


@pytest.mark.parametrize(
    "x,c,b,q,w",
    [
        ([], 0, 0, 0.95, None),
        ([[]], 0, 0, 0.95, None),
        ([[float("nan")]], 0, 0, 0.95, None),
        ([[0]], float("inf"), 0, 0.95, None),
        ([[0]], 0, 0, 1.1, None),
        ([[0]], 0, 0, 0.95, [-1]),
        ([[0]], 0, 0, 0.95, [0]),
        ([[0]], 0, 0, 0.95, [1, 2]),
    ],
)
def test_summary_invalid(x, c, b, q, w):
    with pytest.raises(ValueError):
        cash_risk_summary(x, c, b, q, w)


def test_weighted_boundaries_and_fractional_ties():
    assert quantile([0, 100], 1, [1 - 1e-15, 1e-15]) == 100
    assert cvar([0, 100], 1, [1 - 1e-15, 1e-15]) == 100
    assert quantile([-100, 0, 100], 0, [0, 1, 0]) == 0
    assert quantile([-100, 0, 100], 1, [0, 1, 0]) == 0
    assert quantile([0, 100], 0.9, [0.9, 0.1]) == 0
    assert quantile([0, 100], np.nextafter(0.9, 1), [0.9, 0.1]) == 100
    assert cvar([0, 10, 10, 20], 0.5, [0.5, 0.1, 0.2, 0.2]) == pytest.approx(14)
    np.testing.assert_allclose(
        tail_probabilities([0, 10, 10], 1, [0.7, 0.1, 0.2]), [0, 1 / 3, 2 / 3]
    )
    assert quantile([0, 1], 0.95, [0.99, 0.01]) == 0
    assert quantile([0, 1], 1, [1, 1e-300]) == 1
    assert cvar([0, 1], 1, [1, 1e-300]) == 1


def test_injected_mapping_threshold_wrap_restart():
    p = 1 - 1 / 7
    u = np.array([[0.99, 0, 0.99, p, 0.01], [0, p, 0.99, 0, 0]])
    np.testing.assert_array_equal(map_indices(u, 3, 7), [[2, 0, 0], [0, 2, 0]])
    np.testing.assert_array_equal(map_indices(u, 1, 7), np.zeros((2, 3)))
    np.testing.assert_array_equal(map_indices([[0], [0.999]], 3, 7), [[0], [2]])
    for bad in ([[1]], [[-0.1]], [[np.nan]], [[0.1, 0.2]]):
        with pytest.raises(ValueError):
            map_indices(bad, 3, 7)


@pytest.mark.parametrize("method", ["mc", "sobol"])
def test_prefix_ownership_extension_and_paired_obligations(method):
    case = fixture("tiny")
    p = prepare_history(case.state, 7)
    a = sample_bundle(p, 4, 64, 42, method, 60)
    b = sample_bundle(p, 30, 128, 42, method, 60)
    np.testing.assert_array_equal(a.index_matrix, b.index_matrix[:64, :4])
    np.testing.assert_array_equal(
        a.index_matrix, sample_bundle(p, 4, 64, 42, method, 60).index_matrix
    )
    assert (
        a.bootstrap_draw_id == sample_bundle(p, 4, 64, 42, method, 60).bootstrap_draw_id
    )
    assert (
        a.bootstrap_draw_id
        != sample_bundle(p, 4, 64, 42, method, 60, replicate=1).bootstrap_draw_id
    )
    direct = extend_draw_bundle(a, 60)
    staged = extend_draw_bundle(extend_draw_bundle(a, 30), 60)
    np.testing.assert_array_equal(direct.index_matrix, staged.index_matrix)
    assert direct.sampling_metadata == a.sampling_metadata
    with pytest.raises(ValueError, match="material horizon"):
        extend_draw_bundle(a, 61)
    for arr in [a.index_matrix, a.material_indices, p.joint]:
        with pytest.raises(ValueError):
            arr.flat[0] = 999
        with pytest.raises(ValueError):
            arr.setflags(write=True)
    x = cash_paths(case.state, a, case.obligations, prepared_history=p.joint)
    bill = Obligation("new", "new", 50, 2)
    y = cash_paths(case.state, a, (*case.obligations, bill), prepared_history=p.joint)
    np.testing.assert_array_equal(y - x, np.tile([0, -50, -50, -50], (64, 1)))
    r = cash_risk_summary(x, 30, 10, 0.95)
    s = cash_risk_summary(y, 30, 10, 0.95)
    assert 0 <= s["required_liquidity_reserve"] - r["required_liquidity_reserve"] <= 50
    wealth = cash_risk_summary(x, 80, 10, 0.95)
    assert wealth["required_liquidity_reserve"] == r["required_liquidity_reserve"]
    assert wealth["expected_max_cash_deficit"] <= r["expected_max_cash_deficit"]
    assert wealth["cash_shortfall_probability"] <= r["cash_shortfall_probability"]
    first = cash_paths(
        case.state,
        a,
        (*case.obligations, replace(bill, due_in_days=1)),
        prepared_history=p.joint,
    )
    np.testing.assert_array_equal(first, x - 50)
    bigger = cash_risk_summary(x, 30, 60, 0.95)
    assert (
        0
        <= bigger["required_liquidity_reserve"] - r["required_liquidity_reserve"]
        <= 50
    )


def test_sobol_validation_and_points_prefix():
    for method in ["mc", "sobol"]:
        a = unit_points(method, 64, 59, 123)
        b = unit_points(method, 128, 59, 123)
        np.testing.assert_array_equal(a, b[:64])
    for n, d in [(2000, 59), (2**31, 1), (64, 21202), (2**24, 59)]:
        with pytest.raises(ValueError):
            unit_points("sobol", n, d, 123)
    assert derive_seed(42, "mc", 0) != derive_seed(42, "sobol", 0)
    assert derive_seed(42, "mc", 0) != derive_seed(42, "mc", 0, 200)


def test_requested_block_diagnostics():
    prepared = prepare_history(fixture("tiny").state, 7)
    with pytest.raises(ValueError, match="no predeclared material plan"):
        sample_bundle(prepared, 4, 64, 42, "legacy_mc", 60)
    assert (
        dict(sample_bundle(prepared, 4, 64, 42, "legacy_mc").sampling_metadata)[
            "dimension"
        ]
        is None
    )
    case = fixture("tiny")
    p = prepare_history(case.state, 5)
    assert (p.requested_length, p.resolved_length, p.clipped) == (5, 7, True)
    b = draw_bundle(case.state, 4, 8, 42, 5)
    assert b.mean_block_length_was_clipped and b.requested_mean_block_length == 5


def test_local_contract_and_installed_command(tmp_path):
    example = Path(__file__).parents[2] / "examples/tiny-history.json"
    data = json.loads(example.read_text())
    case = load_input(data)
    assert case.state.immediate_funding == 30
    assert len(prepare_history(case.state, 7).joint) == 3
    out = tmp_path / "sim.json"
    result = subprocess.run(
        [
            str(Path(sys.executable).parent / "ginseng"),
            "simulate",
            "--input",
            str(example),
            "--block-length",
            "7",
            "--sampler",
            "sobol",
            "--out",
            str(out),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())["summary"]["required_liquidity_reserve"] == 90
    bad = subprocess.run(
        [
            sys.executable,
            "-m",
            "ginseng",
            "simulate",
            "--sampler",
            "sobol",
            "--paths",
            "2000",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0 and "1024 or 2048" in bad.stderr
    for mutate in [
        lambda d: d["history"][0].update(essential_spending=-1),
        lambda d: d["history"][0].update(date="2026-01-02"),
        lambda d: d.update(history_end="2026-01-04"),
        lambda d: d.update(opening_cash=float("nan")),
    ]:
        d = json.loads(example.read_text())
        mutate(d)
        with pytest.raises(ValueError):
            load_input(d)


def test_severity_serialization():
    from ginseng.scenario_service import SeverityMetrics
    from ginseng.metrics import severity_metrics

    r = severity_metrics(np.array([[-10, 20], [20, 30]]), 0, 0)
    assert SeverityMetrics(**r).model_dump()["expected_max_cash_deficit"] == 5


def test_aggregate_independent_known_values():
    from ginseng.benchmark import aggregate, cost_table

    rows = [
        dict(case="smooth", method="mc", n=2, seconds=t, estimates={"integral": v})
        for v, t in [(2, 1), (4, 3)]
    ]
    result = aggregate(rows, {"smooth": dict(kind="exact", summary={"integral": 2})})[0]
    assert result["bias"] == 1 and result["rmse"] == pytest.approx(2**0.5)
    assert result["sd"] == pytest.approx(2**0.5) and result["seconds_median"] == 2


def test_reference_batches_preserve_global_quantile():
    from ginseng.benchmark import reference

    case = fixture("tiny")
    p = prepare_history(case.state, 7)
    config = dict(reference_paths=257, reference_batch=31, root_seed=10)
    actual = reference(case, p, config)
    _, _, expected = __import__("ginseng.numerical", fromlist=["run_core"]).run_core(
        case, p, "mc", 257, 10, domain=200
    )
    assert actual["summary"] == pytest.approx(expected)
    assert (
        reference(case, p, {**config, "reference_batch": 257})["summary"]
        == actual["summary"]
    )


def test_benchmark_smoke_and_report_regeneration(tmp_path):
    from ginseng.benchmark import benchmark, report

    root = Path(__file__).parents[2]
    config = json.loads((root / "benchmarks/smoke.json").read_text())
    config.update(paths=[16, 32], reference_paths=128, reference_batch=32, replicates=2)
    result = benchmark(config, tmp_path / "raw")
    assert result["observations"] == 56
    manifest = json.loads((tmp_path / "raw/manifest.json").read_text())
    assert (
        manifest["status"] == "complete"
        and manifest["sampler_settings"]["sobol"]["bits"] == 30
    )
    report(tmp_path / "raw", tmp_path / "report")
    assert (tmp_path / "report/tiny-n.svg").is_file()
    assert json.loads((tmp_path / "raw/aggregates.json").read_text()) == json.loads(
        (tmp_path / "report/aggregates.json").read_text()
    )
    assert (tmp_path / "report/cdf-aggregates.json").is_file()


def test_new_bundle_preserves_aligned_market_indices():
    from datetime import date
    from ginseng.simulate import portfolio_value_paths
    from ginseng.state import Holding, TaxLot

    case = fixture("tiny")
    lot = TaxLot("lot", "TEST", 1, 100, date(2025, 1, 1))
    state = replace(
        case.state,
        holdings=(Holding("TEST", "taxable", 100, (lot,)),),
        portfolio_daily_returns=tuple(
            (date(2026, 1, i + 1), v) for i, v in enumerate([0.1, -0.2, 0.3])
        ),
    )
    p = prepare_history(state, 7)
    bundle = sample_bundle(p, 4, 16, 42, "sobol")
    expected = 100 * np.cumprod(
        1 + np.array([0.1, -0.2, 0.3])[bundle.index_matrix], axis=1
    )
    np.testing.assert_allclose(portfolio_value_paths(state, bundle), expected)


def test_input_draw_and_result_identities_are_distinct():
    from ginseng.numerical import run_core, manifest

    a = fixture("tiny")
    p = prepare_history(a.state, 7)
    bundle, x, summary = run_core(a, p, "sobol", 64, 42)
    b = replace(a, obligations=(*a.obligations, Obligation("bill", "bill", 100, 1)))
    other, _, other_summary = run_core(b, p, "sobol", 64, 42)
    first = manifest(a, p, bundle, summary)
    second = manifest(b, p, other, other_summary)
    assert first["index_hash"] == second["index_hash"]
    assert first["input_hash"] != second["input_hash"]
    assert first["result_hash"] != second["result_hash"]
    assert first["model_hash"] == second["model_hash"]


def test_cli_invalid_benchmark_resource_config(tmp_path):
    from ginseng.cli import main

    p = tmp_path / "bad.json"
    p.write_text(
        json.dumps(
            {
                "replicates": 10000000,
                "reference_paths": 1000,
                "reference_batch": 32,
                "root_seed": 42,
                "block_length": 14,
            }
        )
    )
    assert main(["benchmark", "--config", str(p), "--out", str(tmp_path / "bad")]) == 2
