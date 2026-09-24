"""Panels around the charts: header, reserve card, funding options, ledger, voice line."""
from __future__ import annotations

import os

from rich.cells import cell_len, set_cell_size
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Digits, Label

from .fmt import money, pct
from .kit import Line, assemble, glyphs
from .model import DashboardData

OPTION_ROLE = {"credit": "$g-opt-credit", "liquidate": "$g-opt-liquidate", "hybrid": "$g-opt-hybrid",
               "protective": "$g-opt-protective"}
STAMP = {
    "settled": ("済", "bold reverse $g-stamp"),
    "pending": ("[未]", "$g-muted"),
    "estimate": ("見込", "italic $g-thin"),
}
UNDRESSED_VOICE = {
    "calm": "Reserve holds at p95 for all {h} days.",
    "thin": "Add {rlr} to hold p95. Cheapest option: {c0_name}, about {c0}.",
    "short": "Shortfall likely: {sp} of paths go below $0. At p95, cash lasts {sol} days. Add {rlr}.",
}
_NOWRAP = "text-wrap: nowrap; text-overflow: clip;"


def voice_values(data: DashboardData) -> dict[str, object]:
    best = data.cheapest
    name = best.name if best else "no option"
    return {"rlr": money(data.reserve_to_add), "c0": money(best.cost) if best else "n/a",
            "c0_name": name, "C0_NAME": name.upper(), "sp": pct(data.shortfall_p),
            "sol": data.solvent_days, "sol1": data.solvent_days + 1, "h": data.horizon}


class CollarHeader(Widget):
    DEFAULT_CSS = f"CollarHeader {{ height: 1; {_NOWRAP} }}"
    data: reactive[DashboardData | None] = reactive(None)

    def render(self) -> Content:
        coord = getattr(self.app, "coord", None)
        brand, label = " ginseng ", (f"  {coord.label} {coord.name}" if coord else "")
        line = Line().add(brand, "bold reverse $g-focus").add(label, "$g-ornament")
        if self.data is not None:
            d = self.data
            meta = "   ".join([d.plan, f"{d.horizon} days", f"{d.paths:,} paths", d.sampler, f"draw {d.draw_id}"])
            room = self.size.width - cell_len(brand) - cell_len(label) - 2
            if room > 0:
                meta = set_cell_size(meta, min(room, cell_len(meta)))
                gap = self.size.width - cell_len(brand) - cell_len(label) - cell_len(meta) - 1
                line.add(" " * max(1, gap)).add(meta, "$g-muted")
        return assemble([line])


class Metric(Widget):
    DEFAULT_CSS = f"Metric {{ height: 1; {_NOWRAP} }}"
    value: reactive[str] = reactive("")
    tone: reactive[str] = reactive("ink")

    def __init__(self, caption: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.caption = caption

    def render(self) -> Content:
        gap = max(1, self.size.width - cell_len(self.caption) - cell_len(self.value))
        return assemble([Line().add(self.caption, "$g-ink").add(" " * gap).add(self.value, f"bold $g-{self.tone}")])


class Gauge(Widget):
    """Solvent days as beads or bars; glyphs come from the coord."""

    DEFAULT_CSS = f"Gauge {{ height: 1; {_NOWRAP} }}"
    days: reactive[int] = reactive(0)
    horizon: reactive[int] = reactive(30)
    tone: reactive[str] = reactive("safe")

    def render(self) -> Content:
        kit = glyphs(self.app)
        label = f" {self.days}/{self.horizon}"
        slots = max(1, self.size.width - len(label))
        filled = round(self.days / max(1, self.horizon) * slots)
        return assemble([Line().add(kit["gauge_on"] * filled, f"$g-{self.tone}")
                         .add(kit["gauge_off"] * (slots - filled), "$g-rule").add(label, "$g-muted")])


class ReserveCard(Vertical):
    DEFAULT_CSS = "ReserveCard Digits { height: 3; }"

    def compose(self) -> ComposeResult:
        yield Label("Reserve to add, p95", classes="g-muted")
        yield Digits("$0", id="g-rlr")
        yield Metric("Shortfall odds", id="g-odds")
        yield Metric("CVaR95 trough", id="g-cvar")
        yield Metric("Cash deficit", id="g-deficit")
        yield Metric("Solvent at p95", id="g-solvent")
        yield Gauge(id="g-gauge", classes="g-glyphs")

    def show(self, data: DashboardData) -> None:
        digits = self.query_one("#g-rlr", Digits)
        digits.update(f"${round(data.reserve_to_add)}")
        for tone in ("safe", "thin", "short"):
            digits.set_class(tone == data.tone, f"g-{tone}")
        self._metric("#g-odds", pct(data.shortfall_p), data.tone)
        self._metric("#g-cvar", money(data.cvar95_trough), "short" if data.cvar95_trough < 0 else "safe")
        self._metric("#g-deficit", f"{round(data.deficit_dollar_days):,} $-days", "muted")
        self._metric("#g-solvent", f"{data.solvent_days} days", data.tone)
        gauge = self.query_one(Gauge)
        gauge.days, gauge.horizon, gauge.tone = data.solvent_days, data.horizon, data.tone

    def _metric(self, selector: str, value: str, tone: str) -> None:
        metric = self.query_one(selector, Metric)
        metric.value, metric.tone = value, tone


class OptionsTable(Widget):
    DEFAULT_CSS = f"OptionsTable {{ {_NOWRAP} }}"
    data: reactive[DashboardData | None] = reactive(None)

    def render(self) -> Content:
        data, width = self.data, self.size.width
        if data is None:
            return assemble([Line()])
        if not data.options:
            text = "Nothing to fund at p95." if data.reserve_to_add < 0.5 else "The engine returned no funding options."
            return assemble([Line().add(text, "$g-muted")])
        kit = glyphs(self.app)
        name_w = max(8, width - 23)
        lines = [Line().add("  " + "option".ljust(name_w) + "cost".rjust(7) + "ready".rjust(7) + "tail".rjust(7),
                            "underline $g-muted")]
        best = data.cheapest
        for option in data.options:
            chosen = option is best
            lines.append(Line()
                         .add((kit["pick"] + " ") if chosen else "  ", "bold $g-focus")
                         .add(set_cell_size(option.name, name_w), ("bold " if chosen else "") + OPTION_ROLE[option.kind])
                         .add(money(option.cost).rjust(7), "$g-ink")
                         .add(("now" if option.ready_days == 0 else f"+{option.ready_days}d").rjust(7), "$g-muted")
                         .add(pct(option.tail).rjust(7), "$g-ink"))
        lines.append(Line())
        top = max(option.cost for option in data.options) or 1.0
        room = max(1, width - 16)
        for option in data.options:
            bar = max(1, round(option.cost / top * room))
            lines.append(Line().add(set_cell_size(option.name.split()[0], 8), "$g-muted")
                         .add("█" * bar, OPTION_ROLE[option.kind]).add(" " + money(option.cost), "$g-muted"))
        return assemble(lines)


class LedgerStrip(Widget):
    DEFAULT_CSS = f"LedgerStrip {{ {_NOWRAP} }}"
    data: reactive[DashboardData | None] = reactive(None)

    def render(self) -> Content:
        data, width = self.data, self.size.width
        if data is None:
            return assemble([Line()])
        entries, used = Line(), 0
        for entry in data.ledger:
            stamp, stamp_style = STAMP[entry.status]
            pieces = [(f"d{entry.day} ", "$g-muted"), (f"{entry.label} ", "$g-ink"),
                      (money(entry.amount, plus=True), "bold $g-inflow" if entry.amount > 0 else "bold $g-outflow"),
                      (" ", ""), (stamp, stamp_style)]
            need = sum(cell_len(text) for text, _ in pieces)
            if used + need > width:
                break
            for text, style in pieces:
                entries.add(text, style)
            entries.add("   ")
            used += need + 3
        v, bits = data.validation, []
        if v.oracle_ok is not None:
            bits.append(("oracle ✓", "bold $g-safe") if v.oracle_ok else ("oracle ✗", "bold $g-short"))
        if v.coverage is not None:
            text = f"p95 coverage {v.coverage:.1%}"
            if v.windows:
                text += f", {v.windows} windows"
            if v.kupiec_p is not None:
                text += f", Kupiec p={v.kupiec_p:.2f}"
            bits.append((text, "$g-muted"))
        bits.append((f"draw {data.draw_id}", "$g-muted"))
        checks, used = Line(), 0
        for text, style in bits:
            if used + cell_len(text) > width:
                break
            checks.add(text, style).add("   ")
            used += cell_len(text) + 3
        return assemble([entries, checks])


class VoiceLine(Widget):
    """One sentence in the coord's voice. At ornament level 0 every coord says the same plain thing.

    Typed on, one cell at a time, like a message arriving -- `GINSENG_MOTION=0`
    reveals it whole immediately, same escape hatch `henshin.py` uses."""

    DEFAULT_CSS = f"VoiceLine {{ height: 1; padding: 0 1; {_NOWRAP} }}"
    data: reactive[DashboardData | None] = reactive(None)
    _shown: reactive[int] = reactive(0)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._timer = None

    def _full_text(self) -> str | None:
        data = self.data
        if data is None:
            return None
        coord = getattr(self.app, "coord", None)
        template = UNDRESSED_VOICE[data.state]
        if coord is not None and getattr(self.app, "ornament_level", 3) > 0:
            template = coord.voice.get(data.state, template)
        return template.format(**voice_values(data))

    def watch_data(self, data: DashboardData | None) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        text = self._full_text()
        if text is None or os.environ.get("GINSENG_MOTION") == "0":
            self._shown = cell_len(text) if text else 0
            return
        self._shown = 0
        self._timer = self.set_interval(1 / 60, self._type_on)

    def _type_on(self) -> None:
        text = self._full_text()
        total = cell_len(text) if text else 0
        self._shown = min(total, self._shown + max(1, total // 40))
        if self._shown >= total and self._timer is not None:
            self._timer.stop()
            self._timer = None

    def render(self) -> Content:
        data, text = self.data, self._full_text()
        if data is None or text is None:
            return assemble([Line()])
        shown = set_cell_size(text, self._shown)
        style = "bold $g-short" if data.state == "short" else "$g-ink"
        line = Line().add(glyphs(self.app)["voice_mark"] + " ", f"bold $g-{data.tone}").add(shown, style)
        if self._shown < cell_len(text):
            line.add("▌", f"bold $g-{data.tone}")
        return assemble([line])
