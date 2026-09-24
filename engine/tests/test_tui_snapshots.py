"""Visual regression for the integrated app (sidebar + coord + real Dashboard).

Two real, reproducible engine runs (not the fixed 88x29 the standalone
ginseng_rice tests use -- this app carries extra chrome around the
Dashboard, so 120x45 is more representative), plus one NO_COLOR check.

First run on a new machine or Textual/Rich version: pytest --snapshot-update
then open the SVGs in __snapshots__ and check them by eye before committing.
"""
from __future__ import annotations

import asyncio
from datetime import datetime

import pytest
from ginseng import tui as tui_module
from ginseng.tui import GinsengApp

SIZE = (120, 45)


class _FrozenDateTime(datetime):
    """The event log and telemetry strip render wall-clock timestamps and an
    elapsed duration; freezing `datetime.now()` makes both reproducible
    (elapsed always reads 0.00s, since start and end read the same instant)."""

    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 1, 12, 0, 0, tzinfo=tz)


@pytest.fixture(autouse=True)
def _color_default(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)


@pytest.fixture(autouse=True)
def _frozen_clock(monkeypatch):
    monkeypatch.setattr(tui_module, "datetime", _FrozenDateTime)


@pytest.fixture(autouse=True)
def _no_motion(monkeypatch):
    # The voice-line typewriter and henshin coord transitions are real
    # animations; snapshot tests capture one instant, so skip straight to it.
    monkeypatch.setenv("GINSENG_MOTION", "0")

# canonical fixture, seed 42, mc/2048 paths: reproducible, lands calm (0% shortfall).
CANONICAL = dict(source="fixture", fixture_name="canonical", path="", sampler="mc",
                  estimator="path", paths=2048, seed=42, replicate=0,
                  horizon=None, material_horizon=None, block_length=None)


async def _wait_for_result(pilot) -> None:
    app = pilot.app
    for _ in range(200):
        if app.query_one("#body").current == "panel-result":
            return
        await pilot.pause(0.05)
    raise AssertionError("run did not reach panel-result in time")


async def _dismiss_boot(pilot) -> None:
    await pilot.pause()
    await pilot.press("space")
    await pilot.pause()


async def _run_simulate(pilot) -> None:
    await _dismiss_boot(pilot)
    pilot.app.launch("simulate", dict(CANONICAL))
    await _wait_for_result(pilot)


async def _run_exact(pilot) -> None:
    # The exact oracle's own default model: reproducible, lands short (~39% shortfall).
    await _dismiss_boot(pilot)
    pilot.app.launch("exact", None)
    await _wait_for_result(pilot)


def test_boot_screen(snap_compare):
    async def stable_provenance(pilot):
        # Real source digests change after any engine edit. Keep the visual
        # fixture deterministic; provenance integrity has separate engine tests.
        from textual.widgets import RichLog
        await pilot.pause()
        log = pilot.app.screen.query_one("#boot-log", RichLog)
        log.clear()
        log.write("[dim]boot[/]  synthetic snapshot environment")
        log.write("[dim]sha256[/] 0123456789abcdef  verification.py")
        log.write("[bold]source_fingerprint[/] " + "0123456789abcdef" * 4)
        await pilot.pause()
    assert snap_compare(GinsengApp(), terminal_size=SIZE, run_before=stable_provenance)


def test_boot_screen_dismisses_on_key():
    async def run():
        app = GinsengApp()
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            assert len(app.screen_stack) == 2
            await pilot.press("space")
            await pilot.pause()
            assert len(app.screen_stack) == 1

    asyncio.run(run())


def test_calm_simulate_run(snap_compare):
    assert snap_compare(GinsengApp(), terminal_size=SIZE, run_before=_run_simulate)


def test_short_exact_run(snap_compare):
    assert snap_compare(GinsengApp(), terminal_size=SIZE, run_before=_run_exact)


def test_short_exact_run_without_color(snap_compare, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert snap_compare(GinsengApp(), terminal_size=SIZE, run_before=_run_exact)


def test_softclub_home(snap_compare, monkeypatch):
    monkeypatch.delenv('NO_COLOR', raising=False)
    assert snap_compare(GinsengApp(), terminal_size=(140,48), run_before=_dismiss_boot)


async def _show_atlas(pilot):
    from ginseng.studio import run_experiment
    from ginseng.studio_ui import ResearchStudio
    await _dismiss_boot(pilot)
    pilot.app.open_studio('surface')
    record = run_experiment('surface', dict(fixture='drought-heavy', paths=256, horizon=30))
    record['elapsed_seconds'] = 0
    pilot.app.query_one(ResearchStudio).present(record)
    await pilot.pause()


def test_softclub_atlas(snap_compare, monkeypatch):
    monkeypatch.delenv('NO_COLOR', raising=False)
    assert snap_compare(GinsengApp(), terminal_size=(140,48), run_before=_show_atlas)


async def _open_collection(pilot):
    await _dismiss_boot(pilot)
    pilot.app.action_art_collection()
    await pilot.pause()


def test_art_collection(snap_compare):
    assert snap_compare(GinsengApp(), terminal_size=(140, 48), run_before=_open_collection)


async def _show_cash_signals(pilot):
    await _dismiss_boot(pilot)
    pilot.app.launch('simulate', dict(CANONICAL, fixture_name='drought-heavy', paths=256))
    await _wait_for_result(pilot)
    pilot.app.query_one('#cash-signals').scroll_visible(top=True, animate=False)
    await pilot.pause()


def test_cash_signals(snap_compare):
    assert snap_compare(GinsengApp(), terminal_size=(140, 48), run_before=_show_cash_signals)


async def _show_tail_plot(pilot):
    from ginseng.studio import run_experiment
    from ginseng.studio_ui import ResearchStudio
    from textual.widgets import Select
    await _dismiss_boot(pilot)
    pilot.app.open_studio('tails')
    record = run_experiment('tails', dict(fixture='drought-heavy', paths=256, horizon=30))
    record['elapsed_seconds'] = 0
    pilot.app.query_one(ResearchStudio).present(record)
    pilot.app.query_one('#studio-table-choice', Select).value = 'first_passage'
    await pilot.pause()
    pilot.app.query_one('#plot-x', Select).value = 'day'
    pilot.app.query_one('#plot-y', Select).value = 'probability'
    pilot.app.query_one('#studio-visual').scroll_visible(top=True, animate=False)
    await pilot.pause()


def test_research_scatter(snap_compare):
    assert snap_compare(GinsengApp(), terminal_size=(140, 48), run_before=_show_tail_plot)


def test_compact_collection_without_color(snap_compare, monkeypatch):
    monkeypatch.setenv('NO_COLOR', '1')
    assert snap_compare(GinsengApp(), terminal_size=(80, 24), run_before=_open_collection)


@pytest.mark.parametrize("size", [(80, 24), (60, 20)])
def test_compact_home(snap_compare, size):
    assert snap_compare(GinsengApp(), terminal_size=size, run_before=_dismiss_boot)


async def _open_recipe(pilot):
    await _dismiss_boot(pilot)
    pilot.app.open_studio()
    await pilot.pause()


def test_softclub_recipe(snap_compare):
    assert snap_compare(GinsengApp(), terminal_size=(140, 48), run_before=_open_recipe)


async def _show_simulation_form(pilot):
    await _dismiss_boot(pilot)
    pilot.app.navigate("simulate")
    await pilot.pause()


def test_instrument_simulation_form(snap_compare):
    assert snap_compare(GinsengApp(), terminal_size=(140, 48), run_before=_show_simulation_form)
