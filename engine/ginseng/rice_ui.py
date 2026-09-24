"""The sole original Braille portrait within the fixed soft club workspace."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.widgets import Button, Static

from ginseng.scene_art import DEFAULT_SCENE, scene_rows


def active_scene(app) -> str:
    return DEFAULT_SCENE


def scene_content(name: str, tone: str = "focus") -> Content:
    """Render original cells; layout handles centering and scrolling, never resampling."""
    return Content.assemble(("\n".join(scene_rows(name)), f"$g-{tone}"))


class AnimeScene(Static):
    DEFAULT_CLASSES = "g-glyphs"
    DEFAULT_CSS = """
    AnimeScene { width: auto; min-width: 100%; height: auto;
                 text-wrap: nowrap; content-align: center top; }
    """

    def render(self):
        return scene_content(active_scene(self.app))


class ArtPanel(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield Static("PORTRAIT / ORIGINAL GLYPHS", classes="panel-heading")
        with Horizontal(id="art-workbench"):
            with VerticalScroll(classes="rice-gallery instrument-panel"):
                yield AnimeScene(id="gallery-art")
            with Vertical(id="art-controls", classes="instrument-panel"):
                yield Static("CYBERPUNK PORTRAIT", classes="panel-heading")
                yield Static(
                    "Original glyphs preserved.\nFull-size, scrollable artwork.",
                    classes="welcome-copy",
                )
                yield Static("SIGNAL / LEGEND", classes="panel-heading")
                yield Static(Content.assemble(
                    ("LOW       ━━──\n", "$g-safe"),
                    ("MODERATE  ━━━─\n", "$g-thin"),
                    ("HIGH      ━━━━", "$g-short"),
                ), classes="welcome-copy")
                yield Button("Return home", id="rice-home")

    @on(Button.Pressed, "#rice-home")
    def go_home(self) -> None:
        self.app.action_go_welcome()
