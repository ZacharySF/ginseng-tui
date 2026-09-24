"""The whole dashboard as one widget. Put it in any Dressable app and call show(data)."""
from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.content import Content

from ..tokens import Coord
from ..widgets import Trim
from .charts import FanChart, TroughHistogram
from .model import DashboardData
from .panels import CollarHeader, LedgerStrip, OptionsTable, ReserveCard, VoiceLine

TITLES = {
    "g-reserve": "Reserve",
    "g-paths": "Cash paths, p5 to p95",
    "g-troughs": "Worst balance per path",
    "g-options": "Closing the gap",
    "g-ledger": "Ledger",
}
NARROW_BELOW = 80


class Dashboard(Vertical):
    DEFAULT_CLASSES = "g-dressable"

    def compose(self) -> ComposeResult:
        yield CollarHeader(id="g-collar", classes="g-glyphs")
        yield Trim("─", id="g-collar-trim", classes="g-ornament")
        with Horizontal(id="g-row-top"):
            yield ReserveCard(id="g-reserve", classes="g-panel")
            yield FanChart(id="g-paths", classes="g-panel g-glyphs")
        with Horizontal(id="g-row-mid"):
            yield TroughHistogram(id="g-troughs", classes="g-panel")
            yield OptionsTable(id="g-options", classes="g-panel g-glyphs")
        yield LedgerStrip(id="g-ledger", classes="g-panel")
        yield VoiceLine(id="g-voice", classes="g-glyphs")

    def on_mount(self) -> None:
        self.dress(getattr(self.app, "coord", None))

    def on_resize(self, event: events.Resize) -> None:
        self.set_class(event.size.width < NARROW_BELOW, "-narrow")

    def dress(self, coord: Coord | None) -> None:
        dressed = coord is not None and getattr(self.app, "ornament_level", 3) > 0
        left = coord.borders.get("title_left", "") if dressed else ""
        right = coord.borders.get("title_right", "") if dressed else ""
        for widget_id, title in TITLES.items():
            # Content, not str: a title like '[ Reserve ]' would otherwise be parsed as a markup tag.
            self.query_one(f"#{widget_id}").border_title = Content(f"{left}{title}{right}")
        self.query_one("#g-collar-trim", Trim).motif = coord.borders.get("trim", "─") if coord else "─"

    def show(self, data: DashboardData) -> None:
        """Push one engine run into every panel. Safe to call after every run."""
        report = getattr(self.app, "report_shortfall", None)
        if report is not None:
            report(data.shortfall_p)
        self.query_one(CollarHeader).data = data
        self.query_one(ReserveCard).show(data)
        self.query_one(FanChart).bands = data.bands
        self.query_one(TroughHistogram).data = data
        self.query_one(OptionsTable).data = data
        self.query_one(LedgerStrip).data = data
        self.query_one(VoiceLine).data = data
        self.dress(getattr(self.app, "coord", None))
