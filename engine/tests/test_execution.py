"""Independent hand fixtures and boundary/differential tests for execution."""

import time
from dataclasses import fields, replace

import numpy as np
import pytest
from ginseng.execution import (
    EvaluationContext,
    ExecutionConfig,
    PathStatistics,
    PreparedScenario,
    ResourceLimitError,
    native_info,
    snapshot,
)
from ginseng.metrics import cash_risk_summary, compute_scenario_metrics
from ginseng.risk import quantile, quantiles


def direct(a, h=None):
    a = np.array(a, dtype=float)
    return PreparedScenario(
        np.empty((0, 3)),
        np.empty((0, 0), dtype=np.int64),
        a,
        np.zeros(a.shape[1]),
        h or a.shape[1],
        a.shape[1],
        "direct",
    )


def native_available():
    try:
        return native_info()["available"]
    except ValueError:
        return False


BACKENDS = ["numpy"] + (["native"] if native_available() else [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_hand_paths(backend):
    p = direct([[-2, 3, 0], [0, 0, 0], [-2, 1, 1], [5, 0, -8]])
    with EvaluationContext(ExecutionConfig(backend, block_size=3)) as c:
        r = c.evaluate(p, 0, 1)
        s = c.evaluate(p, 0, 1, full=False)
        assert s.paths is None
        assert c.counters["path_materializations"] == 1
        np.testing.assert_array_equal(
            r.paths, [[-2, 1, 1], [0, 0, 0], [-2, -1, 0], [5, 5, -3]]
        )
        expected = dict(
            minima=[-2, 0, -2, -3],
            terminal=[1, 0, 0, -3],
            minimum_balance=[-2, 0, -2, -3],
            maximum_deficit=[2, 0, 2, 3],
            buffer_dollar_days=[3, 3, 6, 4],
            overdraft_dollar_days=[2, 0, 3, 3],
        )
        for name, value in expected.items():
            np.testing.assert_array_equal(getattr(r.statistics, name), value)
            np.testing.assert_array_equal(getattr(s.statistics, name), value)
        metrics = cash_risk_summary(r.paths, 0, 1, 0.5, [0.2, 0, 0.3, 0.5])
        assert metrics["cash_shortfall_probability"] == 1
        assert metrics["required_liquidity_reserve"] == 3


def test_quantile_oracle_boundaries():
    x = np.array([9, 1, 4, 4, 100.0])
    w = np.array([0.1, 0.2, 0.3, 0.4, 0])
    qs = [1, 0, 0.2, np.nextafter(0.2, 1), 0.9, 0.5, 0.2]
    np.testing.assert_array_equal(quantiles(x, qs, w), [9, 1, 1, 4, 4, 4, 1])
    np.testing.assert_array_equal(quantiles(x, qs, w), [quantile(x, q, w) for q in qs])
    # Positive upper tail may never disappear into a blanket epsilon.
    np.testing.assert_array_equal(
        quantiles([0, 1], [1 - 1e-15, 1], [1 - 2e-15, 2e-15]), [1, 1]
    )
    for values, queries, weights in [
        ([], [0.5], None),
        ([1], [np.nan], None),
        ([1], [-0.1], None),
        ([np.inf], [0.5], None),
        ([1], [0.5], [0]),
        ([1], [0.5], [-1]),
        ([1], [0.5], [1, 2]),
        ([1], [[0.5]], None),
    ]:
        with pytest.raises(ValueError):
            quantiles(values, queries, weights)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.parametrize("block", [1, 7, 64])
def test_differential(backend, workers, block):
    rng = np.random.default_rng(42)
    history = rng.uniform(0, 100, (13, 3))
    indices = rng.integers(0, 13, (37, 17), dtype=np.int64)
    schedule = rng.normal(size=17)
    for p in [
        PreparedScenario(history, indices, np.empty((0, 0)), schedule, 14, 17),
        direct(rng.normal(size=(37, 17)), 14),
    ]:
        with EvaluationContext(ExecutionConfig("numpy", block_size=37)) as c:
            oracle = c.evaluate(p, 13.4, 77.1)
        with EvaluationContext(ExecutionConfig(backend, workers, block)) as c:
            actual = c.evaluate(p, 13.4, 77.1)
            summary = c.evaluate(p, 13.4, 77.1, full=False)
            np.testing.assert_array_equal(actual.paths, oracle.paths)
            for f in fields(PathStatistics):
                np.testing.assert_allclose(
                    getattr(actual.statistics, f.name),
                    getattr(oracle.statistics, f.name),
                    atol=1e-9,
                    rtol=1e-12,
                )
                np.testing.assert_allclose(
                    getattr(summary.statistics, f.name),
                    getattr(oracle.statistics, f.name),
                    atol=1e-9,
                    rtol=1e-12,
                )
            weights = rng.uniform(size=37)
            weights[::5] = 0
            chart = c.charts(actual.paths, 13.4, weights)
            for q in (0.1, 0.5, 0.9):
                assert chart[f"p{int(q * 100)}"] == [
                    quantile(col + 13.4, q, weights) for col in oracle.paths.T
                ]


def test_owned_snapshot_and_invalid_inputs():
    raw = np.arange(12.0).reshape(3, 4)
    p = direct(raw)
    raw[:] = 100
    assert p.direct[0, 0] == 0
    with pytest.raises(ValueError):
        p.direct.setflags(write=True)
    for a in [np.empty((0, 3)), [[np.nan]], [[np.inf]]]:
        with pytest.raises(ValueError):
            direct(a)
    for index in [
        np.array([[-1]], dtype=np.int64),
        np.array([[1]], dtype=np.int64),
        np.array([[0]], dtype=np.int32),
    ]:
        with pytest.raises(ValueError):
            PreparedScenario(
                np.ones((1, 3)), index, np.empty((0, 0)), np.zeros(1), 1, 1
            )
    for kwargs in [
        dict(workers=0),
        dict(workers=5),
        dict(block_size=0),
        dict(memory_budget=0),
        dict(backend="gpu"),
    ]:
        with pytest.raises(ValueError):
            ExecutionConfig(**kwargs)
    with EvaluationContext(ExecutionConfig(memory_budget=100)) as c:
        with pytest.raises(ResourceLimitError):
            c.evaluate(p)
    with pytest.raises(ValueError):
        c.evaluate(p)


@pytest.mark.skipif(not native_available(), reason="Optional native wheel absent")
def test_native_strict_boundary_and_fuzz():
    from ginseng_native import _core

    rng = np.random.default_rng(12)
    for _ in range(80):
        n = int(rng.integers(1, 70))
        x = snapshot(rng.integers(-4, 5, n).astype(float))
        w = rng.uniform(size=n)
        w[::3] = 0
        if not w.any():
            w[:] = 1
        w = snapshot(w)
        q = snapshot([0, 0.1, 0.5, 0.9, 1])
        # Adapter normalization is the shared contract.
        from ginseng.risk import probabilities

        w = snapshot(probabilities(n, w))
        np.testing.assert_array_equal(_core.quantiles(x, w, q), quantiles(x, q, w))
    valid = snapshot([1.0, 2.0, 3.0])
    w = snapshot([1.0, 1.0, 1.0])
    q = snapshot([0.5])
    unaligned = np.ndarray((3,), dtype=float, buffer=bytes(25), offset=1)
    for x in [
        valid[::-1],
        np.ones(3, dtype=np.float32),
        np.ones((1, 3)),
        unaligned,
        np.array([1.0, 2.0, 3.0]),
        snapshot([np.nan, 2, 3]),
        snapshot([np.inf, 2, 3]),
        snapshot([], "<f8"),
    ]:
        with pytest.raises((ValueError, TypeError)):
            _core.quantiles(x, w, q)
    alias = np.ones(3)
    readonly = alias.view()
    readonly.flags.writeable = False
    with pytest.raises(ValueError):
        _core.quantiles(readonly, w, q)


def test_summary_never_materializes_full_and_failure_cleanup(monkeypatch):
    import ginseng.execution as e

    p = direct(np.ones((33, 9)))
    original = e.numpy_block
    observed = []

    def watch(p, a, b, *args):
        observed.append((b - a, args[-1]))
        return original(p, a, b, *args)

    monkeypatch.setattr(e, "numpy_block", watch)
    with EvaluationContext(ExecutionConfig(block_size=7, workers=4)) as c:
        r = c.evaluate(p, full=False)
        assert r.paths is None and c.counters["path_materializations"] == 0
        assert max(n for n, _ in observed) == 7 and not any(
            full for _, full in observed
        )

    def fail(p, a, b, *args):
        if a == 7:
            raise RuntimeError("deliberate failure")
        time.sleep(0.002)
        return original(p, a, b, *args)

    monkeypatch.setattr(e, "numpy_block", fail)
    with (
        pytest.raises(RuntimeError),
        EvaluationContext(ExecutionConfig(block_size=7, workers=4)) as c,
    ):
        c.evaluate(p)
    assert c.closed and not c.cache
    assert all(not t.is_alive() for t in c.pool._threads)


def test_actual_consumers_reuse_and_invalidate():
    from ginseng.execution import prepare_scenario
    from ginseng.funding import PlanSpec, evaluate_plan
    from ginseng.inputs import fixture
    from ginseng.simulate import cash_paths, discretionary_resampled_paths, draw_bundle
    from ginseng.state import Obligation, Transaction, TransactionType

    case = fixture("canonical")
    state = case.state
    with EvaluationContext() as c:
        bundle = draw_bundle(state, 30, 64, 42, 14)
        compute_scenario_metrics(state, bundle, (), 0.95, 1000)
        before = c.counters.copy()
        matrix = cash_paths(state, bundle)
        frozen = matrix.copy()
        compute_scenario_metrics(state, bundle, (), 0.90, 1000)
        compute_scenario_metrics(state, bundle, (), 0.90, 1200)
        compute_scenario_metrics(state, bundle, (), 0.95, 1000, np.linspace(0, 2, 64))
        opened = c.evaluate(
            prepare_scenario(state, bundle), state.immediate_funding + 1, 1000
        )
        assert opened.paths is matrix
        assert c.counters["path_materializations"] == before["path_materializations"]
        assert c.counters["row_reductions"] == before["row_reductions"]
        assert c.counters["history_preparations"] == 1
        for _ in range(2):
            discretionary_resampled_paths(state, bundle)
        for fraction in (0.0, 0.1, 0.2):
            evaluate_plan(
                state,
                bundle,
                (),
                PlanSpec(
                    "candidate",
                    "Candidate",
                    "cash",
                    discretionary_reduction_fraction=fraction,
                    discretionary_reduction_days=14,
                    trailing_days=0,
                ),
            )
        assert c.counters["path_materializations"] == before["path_materializations"]
        compute_scenario_metrics(
            state, bundle, (Obligation("new", "late", 500, 30),), 0.95, 1000
        )
        assert (
            c.counters["path_materializations"] == before["path_materializations"] + 1
        )
        day = state.history_start or state.transactions[0].txn_date
        changed_history = replace(
            state,
            transactions=state.transactions
            + (
                Transaction(
                    day, TransactionType.INCOME_VARIABLE, 1000.0, "Changed history"
                ),
            ),
        )
        compute_scenario_metrics(changed_history, bundle, (), 0.95, 1000)
        assert c.counters["history_preparations"] == 2
        assert (
            c.counters["path_materializations"] == before["path_materializations"] + 2
        )
        np.testing.assert_array_equal(matrix, frozen)
        assert c.counters["cache_hits"] > 0




@pytest.mark.parametrize("backend", BACKENDS)
def test_exact_threshold_classifications_and_adapter(backend):
    tiny = np.nextafter(0.0, 1.0)
    p = direct([[-tiny, tiny], [0.0, 0.0], [-1.0, 1.0], [1.0, -1.0]])
    with EvaluationContext(ExecutionConfig(backend, 2, 3)) as c:
        r = c.evaluate(p, 0, 0)
        np.testing.assert_array_equal(
            r.statistics.maximum_deficit > 0, [True, False, True, False]
        )
        np.testing.assert_array_equal(
            r.statistics.minimum_balance < 0, [True, False, True, False]
        )
    data = np.arange(24.0).reshape(4, 6)
    for a in [
        data[:, ::2],
        data[::-1],
        data.astype(">f8"),
        np.ndarray((4, 6), dtype=float, buffer=bytearray(193), offset=1),
    ]:
        p = PreparedScenario(
            np.empty((0, 3)),
            np.empty((0, 0), dtype=np.int64),
            a,
            np.zeros(a.shape[1]),
            a.shape[1],
            a.shape[1],
            "direct",
        )
        assert p.direct.flags.c_contiguous and p.direct.flags.aligned
        with EvaluationContext(ExecutionConfig(backend)) as c:
            np.testing.assert_array_equal(c.evaluate(p).paths, np.cumsum(a, axis=1))


@pytest.mark.skipif(not native_available(), reason="Optional native wheel absent")
def test_raw_native_index_and_shape_validation():
    from ginseng_native import _core

    history = snapshot(np.ones((2, 3)))
    direct = snapshot(np.empty((0, 0)))
    schedule = snapshot(np.zeros(3))
    indices = snapshot([[0, 1, 0]], "<i8")

    def run(idx=indices, **kwargs):
        config = dict(
            horizon=3,
            start=0,
            stop=1,
            opening=0.0,
            buffer=0.0,
            full=True,
            historical=True,
        )
        config.update(kwargs)
        return _core.evaluate(history, idx, direct, schedule, **config)

    for idx in (
        snapshot([[-1, 0, 0]], "<i8"),
        snapshot([[0, 2, 0]], "<i8"),
        snapshot([[0, 0, 0]], "<i4"),
        indices[:, ::-1],
    ):
        with pytest.raises((ValueError, TypeError)):
            run(idx)
    for kwargs in (
        dict(horizon=0),
        dict(horizon=4),
        dict(start=-1),
        dict(stop=2),
        dict(opening=np.nan),
        dict(buffer=np.inf),
    ):
        with pytest.raises(ValueError):
            run(**kwargs)


@pytest.mark.skipif(not native_available(), reason="Optional native wheel absent")
def test_native_releases_gil_and_worker_repeatability():
    import threading

    from ginseng_native import _core

    rng = np.random.default_rng(5)
    matrix = snapshot(rng.normal(size=(32768, 30)))
    weights = snapshot(np.full(32768, 1 / 32768))
    qs = snapshot([0.1, 0.5, 0.9])
    started = threading.Event()
    completed = threading.Event()

    def work():
        started.set()
        _core.charts(matrix, weights, qs, 0.0)
        completed.set()

    thread = threading.Thread(target=work)
    thread.start()
    started.wait()
    try:
        assert not completed.is_set(), (
            "The numerical call must let Python execute before it finishes."
        )
    finally:
        thread.join()
    p = direct(rng.normal(size=(129, 60)))
    reference = None
    for workers, block in ((1, 129), (2, 17), (4, 31)):
        with EvaluationContext(ExecutionConfig("native", workers, block)) as c:
            r = c.evaluate(p, 123.4, 250.2)
            if reference is not None:
                np.testing.assert_array_equal(r.paths, reference.paths)
                for f in fields(PathStatistics):
                    np.testing.assert_array_equal(
                        getattr(r.statistics, f.name),
                        getattr(reference.statistics, f.name),
                    )
            reference = r


def test_assumption_schedule_edits_reuse_components():
    from ginseng.finance_models import ModelAssumptions
    from ginseng.personal_forecast import (
        _assumption_bundle,
        _schedule_for_workspace,
        _state_for_workspace,
        _with_horizon,
    )
    from ginseng.state import Obligation

    from tests.test_personal_forecast import assumptions_workspace

    workspace = assumptions_workspace(ModelAssumptions())
    state = _with_horizon(
        _state_for_workspace(
            workspace, _schedule_for_workspace(workspace, 365), history=False
        ),
        30,
    )
    with EvaluationContext() as c:
        a = _assumption_bundle(state, workspace.inputs, 42, 16)
        changed = replace(
            state,
            fixed_obligations=state.fixed_obligations
            + (Obligation("late", "Late", 20, 29),),
        )
        b = _assumption_bundle(changed, workspace.inputs, 42, 16)
        assert c.counters["assumption_preparations"] == 1
        np.testing.assert_array_equal(a.discretionary_daily, b.discretionary_daily)
        assert not a.daily_cash_flows.flags.writeable
        altered = workspace.inputs.model_copy(
            update={
                "assumptions": workspace.inputs.assumptions.model_copy(
                    update={"income_variability_pct": 0.8}
                )
            }
        )
        _assumption_bundle(state, altered, 42, 16)
        assert c.counters["assumption_preparations"] == 2


def test_missing_and_incompatible_backend_fail_clearly(monkeypatch):
    import ginseng.execution as execution

    real_import = execution.importlib.import_module

    def absent(name):
        if name == "ginseng_native._core":
            raise ModuleNotFoundError(name="ginseng_native")
        return real_import(name)

    monkeypatch.setattr(execution.importlib, "import_module", absent)
    with pytest.raises(ValueError, match="unavailable"):
        EvaluationContext(ExecutionConfig("native"))
    with EvaluationContext() as c:
        assert c.backend == "numpy" and c.reason

    class Incompatible:
        @staticmethod
        def build_info():
            return {"api_version": 999}

    monkeypatch.setattr(
        execution.importlib, "import_module", lambda name: Incompatible()
    )
    with pytest.raises(ValueError, match="Incompatible"):
        EvaluationContext(ExecutionConfig("native"))


def test_normalized_weight_copies_are_revalidated_and_explicit_scope_is_reusable():
    from ginseng.inputs import fixture
    from ginseng.risk import probabilities
    from ginseng.simulate import draw_bundle

    with EvaluationContext() as c:
        weights = c.weights(2, [0.2, 0.8])
        altered = weights.copy()
        altered[0] = -1
        with pytest.raises(ValueError):
            probabilities(2, altered)
        np.testing.assert_array_equal(probabilities(1, weights[:1]), [1.0])
        case = fixture("canonical")
        bundle = draw_bundle(case.state, 14, 8, 42, 14)
        compute_scenario_metrics(case.state, bundle, (), 0.95, 1000, context=c)
        assert not c.closed
    assert c.closed


def test_prepared_allocation_guard_precedes_copy():
    huge = np.lib.stride_tricks.as_strided(
        np.zeros(1), shape=(2**30, 2), strides=(0, 0)
    )
    with pytest.raises(ResourceLimitError):
        PreparedScenario(
            np.empty((0, 3)),
            np.empty((0, 0), dtype=np.int64),
            huge,
            np.zeros(2),
            2,
            2,
            "direct",
        )
    for horizon in (1.5, True, 0):
        with pytest.raises(ValueError):
            PreparedScenario(
                np.empty((0, 3)),
                np.empty((0, 0), dtype=np.int64),
                np.zeros((1, 2)),
                np.zeros(2),
                horizon,
                2,
                "direct",
            )


def test_query_budget_is_checked_before_snapshot(monkeypatch):
    import ginseng.execution as execution

    with EvaluationContext(ExecutionConfig(memory_budget=100)) as c:

        def forbidden(*args, **kwargs):
            raise AssertionError("Snapshot must not precede the budget guard")

        monkeypatch.setattr(execution, "snapshot", forbidden)
        with pytest.raises(ResourceLimitError):
            c.quantiles(np.arange(100.0), [0.5])
        with pytest.raises(ResourceLimitError):
            c.weights(100)


def test_draw_bundle_never_silently_converts_indices():
    from ginseng.inputs import fixture
    from ginseng.simulate import draw_bundle

    bundle = draw_bundle(fixture("canonical").state, 14, 4, 42, 14)
    for dtype in (np.float64, np.int32):
        with pytest.raises(ValueError):
            replace(bundle, index_matrix=bundle.index_matrix.astype(dtype))
