import json
from hashlib import sha256

import numpy as np
import pytest
from ginseng.cli import main
from ginseng.exact import enumerate_exact
from ginseng.inputs import fixture
from ginseng.precision import (
    BoundedMoments,
    PrecisionConfig,
    checkpoint_interval,
    run_precision,
)
from ginseng.sampling import derive_seed, map_indices, prepare_history
from ginseng.simulate import _deterministic_daily_flow
from scipy.stats import binom


def test_stream_and_partial_cap_match_one_shot():
    case = fixture("tiny")
    p = prepare_history(case.state, 7)
    cfg = PrecisionConfig(1e-8, 0.95, 77, 16)
    result = run_precision(case, cfg, block_length=7, material_horizon=6)
    rng = np.random.default_rng(derive_seed(42, "mc", 0, 400))
    idx = map_indices(rng.random((77, 11)), 3, 7)[:, :4]
    net = p.joint[:, 0] - p.joint[:, 1] - p.joint[:, 2]
    cash = case.state.immediate_funding + np.cumsum(
        net[idx] + _deterministic_daily_flow(case.state, case.obligations, 4), axis=1
    )
    assert result["summary"]["cash_shortfall_probability"] == pytest.approx(
        np.mean(cash.min(axis=1) < 0)
    )
    assert (
        result["manifest"]["index_hash"]
        == sha256(idx.astype("<i8").tobytes()).hexdigest()
    )
    assert [x["n"] for x in result["checkpoints"]] == [16, 32, 64, 77]
    assert result["summary"]["stop_reason"] == "max_paths_reached"
    assert not result["summary"]["precision_met"]
    # Chunk size changes checks, but not the underlying stream at the cap.
    other = run_precision(
        case, PrecisionConfig(1e-8, 0.95, 77, 23), block_length=7, material_horizon=6
    )
    assert other["manifest"]["index_hash"] == result["manifest"]["index_hash"]
    assert other["summary"]["cash_shortfall_probability"] == pytest.approx(
        result["summary"]["cash_shortfall_probability"]
    )


def test_moments_are_stable_and_zero_does_not_prove_zero():
    rng = np.random.default_rng(4)
    values = rng.beta(1, 5, 1031)
    m = BoundedMoments()
    for x in np.array_split(values, 17):
        m.update(x)
    assert m.mean == pytest.approx(values.mean(), abs=1e-15)
    assert m.variance == pytest.approx(values.var(ddof=1), abs=1e-15)
    m = BoundedMoments()
    m.update(np.zeros(1024))
    low, high = checkpoint_interval(m, 1, 0.95)["interval"]
    assert low == 0 and 0 < high < 1
    one = BoundedMoments()
    one.update(np.ones(1024))
    assert checkpoint_interval(one, 1, 0.95)["interval"] == pytest.approx([1 - high, 1])
    # Spending telescopes; the unspent tail remains alpha/(K+1).
    spending = sum(
        checkpoint_interval(m, k, 0.95)["error_allowance"] for k in range(1, 101)
    )
    assert spending == pytest.approx(0.05 * (1 - 1 / 101))


@pytest.mark.parametrize("p", [0.001, 0.05, 0.2, 0.5, 0.95])
def test_exact_bernoulli_probability_of_any_missed_checkpoint(p):
    # Propagate count probabilities and remove paths at their FIRST interval miss.
    # This checks simultaneous coverage, not only coverage after stopping.
    live = np.array([1.0])
    previous = 0
    missed = 0.0
    for look, n in enumerate([16, 32, 64, 128, 256, 512], 1):
        live = np.convolve(
            live, binom.pmf(np.arange(n - previous + 1), n - previous, p)
        )
        for successes in range(n + 1):
            mean = successes / n
            m = BoundedMoments(n, mean, n * mean * (1 - mean))
            lo, hi = checkpoint_interval(m, look, 0.95)["interval"]
            if not lo <= p <= hi:
                missed += live[successes]
                live[successes] = 0
        previous = n
    assert missed <= 0.05 + 1e-12
    assert missed + live.sum() == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize("estimator", ["path", "initial-block-cmc"])
def test_stopping_exact_fixture_and_reproducibility(estimator):
    config = PrecisionConfig(0.025, 0.95, 65536, 512)
    a = run_precision(fixture("tiny"), config, estimator=estimator, block_length=7)
    b = run_precision(fixture("tiny"), config, estimator=estimator, block_length=7)
    assert a["summary"] == b["summary"]
    assert a["checkpoints"] == b["checkpoints"]
    assert a["manifest"]["result_hash"] == b["manifest"]["result_hash"]
    summary = a["summary"]
    assert summary["precision_met"]
    assert summary["absolute_error_bound"] <= 0.025
    assert all(x["absolute_error_bound"] > 0.025 for x in a["checkpoints"][:-1])
    lo, hi = summary["numerical_probability_interval"]
    assert lo <= enumerate_exact()["summary"]["cash_shortfall_probability"] <= hi
    assert "required_liquidity_reserve" not in summary


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_paths": 1},
        {"batch_size": True},
        {"absolute_error": float("nan")},
        {"absolute_error": 0},
        {"confidence": 1},
        {"confidence": False},
        {"max_paths": 2**31},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        PrecisionConfig(**kwargs)


def test_rejections_and_cli(tmp_path, capsys):
    for kwargs in (
        {"sampler": "sobol"},
        {"sampler": "legacy_mc"},
        {"weights": [1]},
        {"estimator": "other"},
        {"horizon": 0},
    ):
        with pytest.raises(ValueError):
            run_precision(fixture("tiny"), **kwargs)
    for invalid in ([float("nan")], [-0.1], [1.1], [], [[0.5]]):
        with pytest.raises(ValueError):
            BoundedMoments().update(invalid)
    target = tmp_path / "out.json"
    assert (
        main(
            [
                "precision",
                "--fixture",
                "tiny",
                "--max-paths",
                "17",
                "--batch-size",
                "8",
                "--out",
                str(target),
            ]
        )
        == 0
    )
    output = json.loads(target.read_text())
    assert output["summary"]["actual_n"] == 17
    assert output["summary"]["stop_reason"] == "max_paths_reached"
    assert output["manifest"]["domain"] == 400
    assert main(["precision", "--absolute-error", "nan"]) == 2
    assert "absolute_error" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["precision", "--sampler", "sobol"])


def test_cmc_stream_agrees_with_all_start_reference():
    from ginseng.conditional import prepare_conditional, slow_contributions
    from ginseng.precision import _bundle_from_points

    case = fixture("tiny")
    prepared = prepare_history(case.state, 7)
    seed = derive_seed(9, "mc", 0, 400)
    points = np.random.default_rng(seed).random((123, 7))
    bundle = _bundle_from_points(points, prepared, 4, seed, True)
    expected = slow_contributions(
        prepare_conditional(prepared, case.state, case.obligations), bundle
    ).mean()
    result = run_precision(
        case,
        PrecisionConfig(1e-8, 0.95, 123, 16),
        estimator="initial-block-cmc",
        seed=9,
        block_length=7,
    )
    assert result["summary"]["cash_shortfall_probability"] == pytest.approx(
        expected, abs=1e-15
    )


def test_precision_benchmark_artifacts(tmp_path):
    from ginseng.precision_benchmark import benchmark

    config = dict(
        cases=["tiny"],
        replicates=2,
        root_seed=42,
        block_length=14,
        absolute_error=0.1,
        confidence=0.95,
        max_paths=2048,
        batch_size=256,
    )
    from pathlib import Path

    references = Path(__file__).parent / "fixtures" / "precision-reference"
    benchmark(config, tmp_path, references_dir=references)
    rows = json.loads((tmp_path / "observations.json").read_text())
    assert len(rows) == 4
    assert (tmp_path / "results.md").exists()
    assert not json.loads((tmp_path / "budget-exhausted.json").read_text())["summary"][
        "precision_met"
    ]
