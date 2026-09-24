"""Numerical and interaction contracts for the terminal research studio."""

import asyncio
import json

import numpy as np
import pytest

from ginseng.studio import BY_KEY, run_experiment, tail_metrics


def test_tail_metrics_include_opening_peak_and_mid_horizon_failures():
    result = tail_metrics(np.array([[-20, 5], [5, 10]]), 10, [0.5, 0.95])
    assert result["no_shortfall_probability"] == 0.5
    assert result["first_passage"][0]["probability"] == 0.5
    assert result["first_passage"][1]["probability"] == 0
    assert result["expected_max_cash_deficit"] == 5
    assert result["expected_max_drawdown"] == 10
    assert result["tail_risk"][1]["deficit_cvar"] == 10


def test_surface_is_monotone_in_cash_and_time():
    result = run_experiment(
        "surface", dict(fixture="tiny", horizon=4, paths=64, cash_offsets=[0, 50, 500])
    )["result"]
    odds = np.array(
        [row["shortfall_probability"] for row in result["surface"]]
    ).reshape(3, 4)
    assert np.all(np.diff(odds, axis=0) <= 0)
    assert np.all(np.diff(odds, axis=1) >= 0)
    assert np.all(odds[-1] == 0)
    assert result["experiment_identity"]["input_source"] == "synthetic fixture"


def test_sampler_comparison_has_exact_reference_only_for_matching_case():
    params = dict(path_counts=[16], replicates=2)
    result = run_experiment("compare", params)["result"]
    assert result["exact_reference"] is not None
    assert len(result["comparison"]) == 6
    assert all(row["rmse"] is not None for row in result["comparison"])
    other = run_experiment("compare", {**params, "horizon": 3})["result"]
    assert other["exact_reference"] is None
    assert all(row["rmse"] is None for row in other["comparison"])


def test_precision_matches_public_engine():
    from ginseng.inputs import fixture
    from ginseng.precision import PrecisionConfig, run_precision

    params = dict(fixture="tiny", horizon=4, max_paths=64, batch_size=32, seed=42)
    actual = run_experiment("precision", params)["result"]
    expected = run_precision(
        fixture("tiny"),
        PrecisionConfig(0.005, 0.95, 64, 32),
        estimator="initial-block-cmc",
        seed=42,
        block_length=7,
    )
    assert actual["summary"] == expected["summary"]


def test_funding_predeclares_extended_horizon():
    from ginseng.studio import _sample
    from ginseng.funding import (
        FundingConfig,
        build_candidates,
        optimizer_comparison_bundle,
    )

    case, _, bundle, _ = _sample(
        {**BY_KEY["funding"].defaults, "paths": 16}, funding=True
    )
    specs = build_candidates(case.state, case.obligations, 1000, FundingConfig())
    comparison = optimizer_comparison_bundle(case.state, bundle, specs)
    assert comparison.horizon_days >= bundle.horizon_days
    assert np.array_equal(
        comparison.index_matrix[:, : bundle.horizon_days], bundle.index_matrix
    )


@pytest.mark.parametrize(
    "key, params",
    [
        ("tails", {"paths": -1}),
        ("tails", {"paths": True}),
        ("tails", {"horizon": 100000}),
        ("tails", {"paths": 1000000}),
        ("tails", {"quantiles": [1.1]}),
        ("tails", {"typo": 1}),
        ("precision", {"sampler": "sobol"}),
        ("compare", {"path_counts": [17]}),
        ("tails", {"seed": float("nan")}),
    ],
)
def test_invalid_recipes_are_rejected(key, params):
    with pytest.raises(ValueError):
        run_experiment(key, params)


def test_personal_forecast_uses_validated_overrides(tmp_path):
    from tests.test_personal_forecast import scheduled_workspace

    workspace = scheduled_workspace()
    path = tmp_path / "workspace.json"
    path.write_text(workspace.model_dump_json())
    result = run_experiment(
        "forecast",
        dict(
            workspace=str(path), horizon=30, paths=16, overrides={"mode": "scheduled"}
        ),
    )
    assert result["result"]["model_mode"] == "scheduled"
    assert json.loads(path.read_text()) == workspace.model_dump(mode="json")


def test_service_does_not_send_credentials_to_redirects(monkeypatch):
    import httpx

    seen = []
    original = httpx.Client

    def handle(request):
        seen.append(request)
        return httpx.Response(307, headers={"location": "https://elsewhere.invalid/"})

    monkeypatch.setenv("GINSENG_API_URL", "https://example.invalid")
    monkeypatch.setenv("GINSENG_API_TOKEN", "test-token")
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: original(transport=httpx.MockTransport(handle), **kw),
    )
    with pytest.raises(httpx.HTTPStatusError):
        run_experiment("service", {})
    assert len(seen) == 1
    assert seen[0].headers["authorization"] == "Bearer test-token"


def test_service_refuses_remote_plaintext(monkeypatch):
    monkeypatch.setenv("GINSENG_API_URL", "http://example.invalid")
    with pytest.raises(ValueError, match="HTTPS"):
        run_experiment("service", {})


def test_research_navigation_execution_export_and_import(monkeypatch, tmp_path):
    from ginseng.tui import GinsengApp
    from ginseng.studio_ui import ResearchStudio
    from textual.widgets import TextArea, Input

    monkeypatch.setenv("GINSENG_MOTION", "0")
    monkeypatch.setattr("ginseng.studio_ui.NOTEBOOK_DIR", tmp_path)

    async def run():
        app = GinsengApp()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.press("space")
            app.open_studio("surface")
            await pilot.pause()
            studio = app.query_one(ResearchStudio)
            studio.query_one("#studio-params", TextArea).load_text(
                json.dumps(dict(fixture="tiny", horizon=4, paths=32))
            )
            studio.launch()
            for _ in range(100):
                if not studio.busy:
                    break
                await pilot.pause(0.1)
            assert studio.result is not None
            assert studio.result["operation"] == "surface"
            assert studio.query_one("#studio-atlas").display
            studio.query_one("#studio-export").scroll_visible(
                animate=False, immediate=True
            )
            await pilot.pause()
            await pilot.click("#studio-export", offset=(3, 0))
            await pilot.pause()
            saved = list(tmp_path.glob("*.json"))
            assert len(saved) == 1
            payload = json.loads(saved[0].read_text())
            assert payload["parameters"]["paths"] == 32
            studio.query_one("#studio-import-path", Input).value = str(saved[0])
            studio.import_record()
            assert studio.result == payload
            app.action_toggle_sidebar()
            assert not app.query_one("#sidebar").display

    asyncio.run(run())


def test_bad_simulation_file_does_not_crash_app(monkeypatch):
    from ginseng.tui import GinsengApp, QUICK_SIMULATE

    monkeypatch.setenv("GINSENG_MOTION", "0")

    async def run():
        app = GinsengApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("space")
            app.launch(
                "simulate",
                {**QUICK_SIMULATE, "source": "file", "path": "/missing/history.json"},
            )
            await pilot.pause()
            assert app.query_one("#body").current == "panel-error"
            assert app._status == "FAULT"
            app._tick_pulse()
            assert app._status == "FAULT"
            app.open_studio("tails")
            await pilot.pause()
            assert app.query_one("#body").current == "panel-studio"

    asyncio.run(run())


def test_cancel_stops_the_computation_process(monkeypatch):
    import sys
    from ginseng.tui import GinsengApp
    from ginseng.studio_ui import ResearchStudio

    monkeypatch.setenv("GINSENG_MOTION", "0")
    original = asyncio.create_subprocess_exec
    processes = []

    async def slow_process(*args, **kwargs):
        process = await original(
            sys.executable, "-c", "import time; time.sleep(60)", **kwargs
        )
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", slow_process)

    async def run():
        app = GinsengApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("space")
            app.open_studio("tails")
            studio = app.query_one(ResearchStudio)
            studio.launch()
            for _ in range(20):
                if processes:
                    break
                await pilot.pause(0.05)
            assert processes
            studio.cancel()
            for _ in range(50):
                if not studio.busy:
                    break
                await pilot.pause(0.05)
            assert not studio.busy
            assert processes[0].returncode is not None
            assert studio.result is None

    asyncio.run(run())


def test_opening_cash_and_bill_overrides_preserve_fixture():
    from ginseng.studio import _case
    from ginseng.inputs import fixture

    original = fixture("tiny")
    case = _case(
        dict(
            fixture="tiny",
            opening_cash=500,
            obligations=[dict(id="rent", amount=200, due_in_days=3)],
        )
    )
    assert case.state.immediate_funding == 500
    assert case.obligations[0].amount == 200
    assert fixture("tiny") == original
    assert len(case.state.transactions) == len(original.state.transactions) + 1




def test_small_terminal_keeps_run_button_visible(monkeypatch):
    from ginseng.tui import GinsengApp

    monkeypatch.setenv("GINSENG_MOTION", "0")

    async def run():
        app = GinsengApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("space")
            app.open_studio("precision")
            await pilot.pause()
            button = app.query_one("#studio-run")
            assert button.region.bottom <= 23
            assert app.query_one("#studio-params").size.height >= 5

    asyncio.run(run())




def test_precision_controls_capture_and_replay_adapter(tmp_path):
    from ginseng.precision import run_precision, PrecisionConfig
    from ginseng.inputs import fixture
    from ginseng.execution import ExecutionConfig

    params = dict(
        fixture="tiny",
        horizon=4,
        estimator="path",
        max_paths=128,
        batch_size=32,
        chunk_size=7,
        time_limit_seconds=5.0,
        memory_budget_bytes=2**20,
        stop_when_precise=False,
        backend="numpy",
        workers=2,
        capture_path=str(tmp_path / "capture"),
    )
    r = run_experiment("precision", params)["result"]
    expected = run_precision(
        fixture("tiny"),
        PrecisionConfig(
            0.005,
            0.95,
            128,
            32,
            chunk_size=7,
            time_limit_seconds=5.0,
            memory_budget_bytes=2**20,
            stop_when_precise=False,
        ),
        block_length=7,
        execution=ExecutionConfig("numpy", 2, memory_budget=2**20),
    )
    assert r["summary"] == expected["summary"]
    assert r["manifest"]["config"] == expected["manifest"]["config"]
    assert r["manifest"]["actual_execution"]["workers"] == 2
    replay = run_experiment("precision-replay", dict(input=str(tmp_path / "capture")))[
        "result"
    ]
    assert replay["match"] and replay["mismatches"] == []
    assert replay["summary"] == r["summary"]


def test_studio_cancel_retains_partial_result_and_capture(tmp_path):
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls == 7

    params = dict(
        fixture="tiny",
        horizon=4,
        estimator="path",
        max_paths=200,
        batch_size=16,
        chunk_size=7,
        capture_path=str(tmp_path / "partial"),
    )
    record = run_experiment("precision", params, cancelled=cancelled)
    s = record["result"]["summary"]
    assert s["status"] == "cancelled"
    assert s["actual_n"] > s["interval_observations"] == 16
    assert run_experiment("precision-replay", dict(input=str(tmp_path / "partial")))[
        "result"
    ]["match"]
    params["capture_path"] = str(tmp_path / "empty")
    empty = run_experiment("precision", params, cancelled=lambda: True)["result"]
    assert (
        empty["summary"]["status"] == "cancelled" and empty["summary"]["actual_n"] == 0
    )
    assert empty["capture"] == dict(status="unavailable", reason="no_observations")
    assert not (tmp_path / "empty").exists()


def test_tui_precision_capture_replay_and_cooperative_cancel(tmp_path, monkeypatch):
    from ginseng.tui import GinsengApp
    from ginseng.studio_ui import ResearchStudio
    from textual.widgets import TextArea, Static

    monkeypatch.setenv("GINSENG_MOTION", "0")

    async def finished(studio, pilot):
        for _ in range(150):
            if not studio.busy:
                break
            await pilot.pause(0.05)
        assert not studio.busy

    async def run():
        app = GinsengApp()
        async with app.run_test(size=(100, 38)) as pilot:
            await pilot.press("space")
            app.open_studio("precision")
            studio = app.query_one(ResearchStudio)
            await pilot.pause()
            defaults = json.loads(studio.query_one("#studio-params", TextArea).text)
            assert {
                "chunk_size",
                "time_limit_seconds",
                "memory_budget_bytes",
                "capture_path",
                "stop_when_precise",
            } <= defaults.keys()
            studio.query_one("#studio-params", TextArea).load_text(
                json.dumps(
                    dict(
                        fixture="tiny",
                        horizon=4,
                        max_paths=128,
                        batch_size=32,
                        chunk_size=7,
                        capture_path=str(tmp_path / "evidence"),
                    )
                )
            )
            studio.launch()
            await finished(studio, pilot)
            assert studio.result["result"]["summary"]["actual_n"] == 128
            assert "numerical interval" in str(
                studio.query_one("#studio-summary", Static).content
            )
            app.open_studio("precision-replay")
            await pilot.pause()
            studio.query_one("#studio-params", TextArea).load_text(
                json.dumps(dict(input=str(tmp_path / "evidence")))
            )
            studio.launch()
            await finished(studio, pilot)
            assert studio.result["result"]["match"]
            assert "Replay: MATCH" in str(
                studio.query_one("#studio-summary", Static).content
            )
            app.open_studio("precision")
            await pilot.pause()
            studio.query_one("#studio-params", TextArea).load_text(
                json.dumps(
                    dict(
                        fixture="tiny",
                        horizon=4,
                        max_paths=2**25,
                        batch_size=16,
                        chunk_size=7,
                        absolute_error=1e-10,
                    )
                )
            )
            studio.launch()
            # Cancel immediately, including before the Python subprocess has started.
            studio.cancel()
            await finished(studio, pilot)
            assert studio.result["result"]["summary"]["status"] == "cancelled"
            assert len(studio.notebook) == 3
            assert studio.cancel_path is None

    asyncio.run(run())


def test_studio_personal_capture_requires_explicit_boolean_consent(
    tmp_path, monkeypatch
):
    import ginseng.studio as studio
    from ginseng.inputs import fixture

    monkeypatch.setattr(studio, "_case", lambda params: fixture("tiny"))
    params = dict(
        input="private-history.json",
        horizon=4,
        max_paths=32,
        batch_size=16,
        block_length=7,
        capture_path=str(tmp_path / "personal"),
    )
    with pytest.raises(ValueError, match="consent"):
        run_experiment("precision", params)
    with pytest.raises(ValueError, match="boolean"):
        run_experiment("precision", {**params, "allow_personal_capture": "false"})
    run_experiment("precision", {**params, "allow_personal_capture": True})
    assert (tmp_path / "personal" / "precision.json").is_file()


def test_personal_precision_controls_and_cancellation(tmp_path):
    from tests.test_personal_forecast import historical_workspace

    path = tmp_path / "workspace.json"
    original = historical_workspace().model_dump_json()
    path.write_text(original)
    r = run_experiment(
        "personal-numerics",
        dict(
            workspace=str(path),
            action="precision",
            horizon=14,
            confidence=0.99,
            estimator="path",
            time_limit_seconds=5.0,
            memory_budget_bytes=2**20,
            max_paths=1024,
        ),
        cancelled=lambda: True,
    )["result"]
    assert r["summary"]["status"] == "cancelled"
    assert r["summary"]["confidence"] == 0.99 and r["summary"]["actual_n"] == 0
    assert r["resources"] == dict(time_limit_seconds=5.0, memory_budget_bytes=2**20)
    assert path.read_text() == original
