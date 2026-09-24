"""Exercise the fixed theme and art controls and the meaning of the new visualizations."""

import asyncio
import math

import pytest
from ginseng.ginseng_rice.lint import check
from ginseng.rice_charts import (
    ResearchPlot,
    TableVisual,
    dot_plot,
    numeric_columns,
    plot_points,
    spark,
)
from ginseng.rice_ui import active_scene, scene_content
from ginseng.scene_art import SCENES, fit_scene, scene_rows
from ginseng.tui import SOFT_CLUB, GinsengApp, WorkspaceCommands
from textual.widgets import ContentSwitcher, Footer, Select


@pytest.fixture(autouse=True)
def quiet_motion(monkeypatch):
    monkeypatch.setenv("GINSENG_MOTION", "0")


def test_plots_preserve_numeric_axes_and_omit_missing_values():
    rows = [
        {"day": 3, "cash": -20},
        {"day": 10, "cash": 0},
        {"day": 100, "cash": 40},
        {"day": 101, "cash": None},
        {"day": 102, "cash": math.inf},
        {"day": False, "cash": 100},
        {"day": 104, "cash": True},
    ]
    assert numeric_columns(["day", "cash", "missing"], rows) == ["day", "cash"]
    assert plot_points(rows, "day", "cash") == [
        (3.0, -20.0),
        (10.0, 0.0),
        (100.0, 40.0),
    ]
    assert plot_points(rows[:3], None, "cash") == [
        (1.0, -20.0),
        (2.0, 0.0),
        (3.0, 40.0),
    ]
    rows = dot_plot([(0, 0), (100, 100)], 10, 4)
    assert len(rows) == 4 and all(len(row) == 10 for row in rows)
    assert rows[0][-1] != "\u2800" and rows[-1][0] != "\u2800"
    assert dot_plot([], 10, 4) == []
    assert len(dot_plot([(0, 0)], 1, 1)) == 1
    assert spark([-10, 0, 10], 3) == "▁▅█"
    assert spark([0, 0, 0], 3) == "▄▄▄"


def test_art_fits_the_terminal_without_patched_fonts():
    for name, scene in SCENES.items():
        assert all(ch == "\n" or "\u2800" <= ch <= "\u28ff" for ch in scene.art)
        rows = scene_rows(name)
        assert rows and rows[0].strip("⠀") and rows[-1].strip("⠀")
        assert fit_scene(name, len(rows[0])) == rows
        assert scene_content(name).plain == "\n".join(rows)
        for width in (8, 22, 28, 60):
            fitted = fit_scene(name, width, 14)
            assert 1 <= len(fitted) <= 14
            assert any(row.strip("⠀") for row in fitted)
    assert set(SCENES) == {"cyberpunk"}
    assert fit_scene("cyberpunk", 0) == ()


def test_soft_club_palette_is_readable():
    assert not [problem for problem in check(SOFT_CLUB) if not problem.startswith("note")]


@pytest.mark.parametrize("size", [(140, 48), (80, 24), (60, 20)])
def test_fixed_theme_and_original_art_collection(size, monkeypatch):
    monkeypatch.setenv("GINSENG_COORD", "miku")

    async def run():
        app = GinsengApp()
        async with app.run_test(size=size) as pilot:
            await pilot.press("space", "ctrl+w")
            await pilot.pause()
            assert app.theme == "ginseng-softclub"
            assert app.query_one("#body", ContentSwitcher).current == "panel-art"
            assert active_scene(app) == "cyberpunk"
            assert not app.query("#rice-coord")
            assert "Theme" not in [c.title for c in app.get_system_commands(app.screen)]
            commands = WorkspaceCommands(app.screen)._commands()
            assert not any(name.startswith("wear ") for name, _, _ in commands)
            assert not any(name.startswith("art:") for name, _, _ in commands)
            assert not app.query("#rice-art")
            assert "a" not in [binding.key for binding in app.BINDINGS]
            await app.wear("miku")
            app.action_change_theme()
            app.action_toggle_dark()
            app.query_one("#rice-home").focus()
            await pilot.press("a", "t")
            await pilot.pause()
            assert app.theme == "ginseng-softclub"
            assert active_scene(app) == "cyberpunk"
            app.action_go_welcome()
            app.action_art_collection()
            assert active_scene(app) == "cyberpunk"
            assert not app.query_one("#panel-art").max_scroll_x
            assert not app.query_one(Footer).max_scroll_x

    asyncio.run(run())


def test_result_visual_axes_table_switch_and_empty_state():
    async def run():
        app = GinsengApp()
        async with app.run_test(size=(100, 35)) as pilot:
            await pilot.press("space")
            app.open_studio("tails")
            visual = app.query_one(TableVisual)
            visual.show(
                ["day", "cash"], [{"day": 2, "cash": -10}, {"day": 9, "cash": 30}]
            )
            await pilot.pause()
            app.query_one("#plot-x", Select).value = "day"
            app.query_one("#plot-y", Select).value = "cash"
            await pilot.pause()
            plot = app.query_one(ResearchPlot)
            assert plot.points == [(2.0, -10.0), (9.0, 30.0)]
            assert plot.x_label == "day" and plot.y_label == "cash"
            visual.show(["label"], [{"label": "unavailable"}])
            await pilot.pause()
            assert not visual.display
            assert plot.points == []
            visual.show(["x"], [{"x": i} for i in range(5001)])
            await pilot.pause()
            assert len(plot.points) == 5000 and plot.total_rows == 5001
            assert "first 5,000 of 5,001" in plot.render().plain

    asyncio.run(run())


def test_layout_adapts_after_terminal_resize():
    async def run():
        app = GinsengApp()
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.press("space")
            await pilot.resize_terminal(60, 20)
            await pilot.pause()
            assert app.screen.has_class("-tiny")
            assert app.screen.has_class("-short")
            assert not app.query_one("#sidebar").display
            await pilot.resize_terminal(140, 48)
            await pilot.pause()
            assert not app.screen.has_class("-compact")
            assert app.query_one("#sidebar").display
            assert app.query_one("#portrait").display
            assert app.theme == "ginseng-softclub"

    asyncio.run(run())


def test_workflow_table_and_navigation_rail_are_operable():
    from textual.widgets import DataTable

    async def run():
        app = GinsengApp()
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.press("space")
            await pilot.pause()
            assert "HOME" in app.query_one("#nav").render_line(0).text
            assert app.query_one("#inspector").display
            table = app.query_one("#home-workflows", DataTable)
            assert table.row_count == 6
            table.focus()
            table.move_cursor(row=2)
            await pilot.press("enter")
            await pilot.pause()
            assert app.query_one("#body", ContentSwitcher).current == "panel-studio"
            app.action_go_welcome()
            app.action_toggle_sidebar()
            assert not app.query_one("#inspector").display
            assert not app.query_one("#sidebar").display
            app.action_toggle_sidebar()
            await pilot.resize_terminal(80, 24)
            assert not app.query_one("#inspector").display
            table.focus()
            table.move_cursor(row=0)
            await pilot.press("enter")
            assert app.query_one("#body", ContentSwitcher).current == "panel-form"

    asyncio.run(run())


def test_portrait_keeps_every_original_cell_in_each_view():
    async def run():
        app = GinsengApp()
        expected = "\n".join(scene_rows("cyberpunk"))
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            assert app.screen.query_one("#boot-art").render().plain == expected
            await pilot.press("space")
            await pilot.pause()
            portrait = app.query_one("#portrait")
            assert portrait.render().plain == expected
            assert portrait.size.height == 31
            assert portrait.content_size.width >= 27
            for index, row in enumerate(scene_rows("cyberpunk")):
                assert row in portrait.render_line(index).text
            assert portrait.region.bottom <= app.query_one("#inspector").region.bottom
            # Native glyphs stay intact in the scrollable gallery on small screens.
            await pilot.resize_terminal(60, 20)
            app.action_art_collection()
            await pilot.pause()
            assert app.query_one("#gallery-art").render().plain == expected

    asyncio.run(run())
