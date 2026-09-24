"""Small, responsive terminal plots of actual dashboard and research data."""

from __future__ import annotations

import math

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.reactive import reactive
from textual.widgets import Select, Static

from ginseng.braille import _cells_from_canvas
from ginseng.ginseng_rice.dashboard.fmt import compact
from ginseng.ginseng_rice.dashboard.model import DashboardData

BLOCKS = "▁▂▃▄▅▆▇█"


def spark(values, width: int) -> str:
    """Endpoint-preserving sampling, with a documented per-series scale."""
    if not values or width < 1:
        return ""
    lo, hi = min(values), max(values)
    return "".join(
        BLOCKS[
            round(
                7
                * (values[round(i * (len(values) - 1) / max(1, width - 1))] - lo)
                / (hi - lo)
            )
        ]
        if hi != lo
        else "▄"
        for i in range(width)
    )


class PathSignals(Static):
    data: reactive[DashboardData | None] = reactive(None)

    def render(self):
        data = self.data
        if data is None:
            return Content("Cash signals appear after a run.")
        width = max(6, self.content_size.width - 2)
        bands = data.bands
        series = [
            ("P5 CASH / $", bands.p5, "safe"),
            ("MEDIAN CASH / $", bands.p50, "focus"),
            (
                "P95 − P5 SPREAD / $",
                tuple(hi - lo for hi, lo in zip(bands.p95, bands.p5)),
                "accent",
            ),
        ]
        parts = [("CASH SIGNALS\n", "bold $g-focus")]
        for label, values, role in series:
            parts.append(
                (
                    f"{label}   {compact(min(values))} .. {compact(max(values))}\n",
                    "$g-muted",
                )
            )
            # Negative balances remain visible even in a palette with a soft accent.
            for i, ch in enumerate(spark(values, width)):
                value = values[round(i * (len(values) - 1) / max(1, width - 1))]
                parts.append((ch, "$g-short" if value < 0 else f"$g-{role}"))
            parts.append(("\n", ""))
        parts.append((f"d1 → d{data.horizon} · each row has its own scale", "$g-muted"))
        return Content.assemble(*parts)


class FundingBars(Static):
    data: reactive[DashboardData | None] = reactive(None)

    def render(self):
        data = self.data
        parts = [("FUNDING / COST & RESIDUAL RISK\n", "bold $g-focus")]
        if data is None or not data.options:
            return Content.assemble(
                *parts, ("No funding candidates in this run.", "$g-muted")
            )
        ceiling = max(option.cost for option in data.options) or 1
        width = max(4, self.content_size.width - 12)
        for option in data.options:
            parts.append((f"{option.name} · ready d{option.ready_days}\n", "$g-ink"))
            size = round(width * option.cost / ceiling)
            parts.extend(
                [
                    ("━" * size + "─" * (width - size), f"$g-opt-{option.kind}"),
                    (f" ${option.cost:,.2f}\n", "$g-ink"),
                ]
            )
            role = (
                "safe"
                if option.tail < 0.05
                else "thin"
                if option.tail < 0.25
                else "short"
            )
            parts.append((f"  remaining shortfall {option.tail:.1%}\n", f"$g-{role}"))
        parts.append(("Bar length = expected cost / $ · shared zero", "$g-muted"))
        return Content.assemble(*parts)


def is_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def numeric_columns(columns, rows) -> list[str]:
    return [key for key in columns if any(is_number(row.get(key)) for row in rows)]


def plot_points(rows, x_key: str | None, y_key: str) -> list[tuple[float, float]]:
    points = []
    for i, row in enumerate(rows):
        x, y = i + 1 if x_key is None else row.get(x_key), row.get(y_key)
        if is_number(x) and is_number(y):
            points.append((float(x), float(y)))
    return points


def dot_plot(points, width: int, height: int) -> list[str]:
    """Scatter marks, without invented interpolation across missing observations."""
    if not points or width < 1 or height < 1:
        return []
    xs, ys = zip(*points)
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    canvas = [[False] * (width * 2) for _ in range(height * 4)]
    for x, y in points:
        col = (
            round((x - xmin) / (xmax - xmin) * (width * 2 - 1))
            if xmax != xmin
            else width - 1
        )
        row = (
            round((ymax - y) / (ymax - ymin) * (height * 4 - 1))
            if ymax != ymin
            else height * 2 - 1
        )
        canvas[row][col] = True
        # A two-dot mark is readable on both light and dark palettes.
        canvas[min(height * 4 - 1, row + 1)][col] = True
    return _cells_from_canvas(canvas, width * 2, height * 4)


class ResearchPlot(Static):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.points = []
        self.x_label = "row"
        self.y_label = ""
        self.row_count = 0
        self.total_rows = 0

    def render(self):
        if not self.points:
            return Content("No finite numeric pairs for these axes.")
        width = max(6, self.content_size.width - 12)
        xs, ys = zip(*self.points)
        rows = dot_plot(self.points, width, 8)
        parts = [(f"{self.y_label} ↑\n", "bold $g-focus")]
        for i, row in enumerate(rows):
            tick = (
                f"{max(ys):.4g}"
                if i == 0
                else f"{min(ys):.4g}"
                if i == len(rows) - 1
                else ""
            )
            parts.extend([(f"{tick:>9} │", "$g-muted"), (row + "\n", "$g-accent")])
        parts.extend(
            [
                ("          └" + "─" * width + "\n", "$g-rule"),
                (
                    f"           {min(xs):.5g} → {max(xs):.5g}  / {self.x_label}\n",
                    "$g-muted",
                ),
                (
                    f"{len(self.points):,} finite pairs / {self.row_count:,} rows",
                    "$g-muted",
                ),
            ]
        )
        if self.total_rows > self.row_count:
            parts.append(
                (f" · first {self.row_count:,} of {self.total_rows:,} rows", "$g-muted")
            )
        return Content.assemble(*parts)


class TableVisual(Vertical):
    """Select any numeric axes from the current result table; raw table stays below."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rows = []
        self.total_rows = 0

    def compose(self) -> ComposeResult:
        yield Static("✦  DATA SCOPE / inspect a result column", classes="rule")
        with Horizontal(classes="plot-controls"):
            yield Select([], id="plot-x", prompt="Horizontal axis")
            yield Select([], id="plot-y", prompt="Vertical axis")
        yield ResearchPlot(id="research-plot")

    def show(self, columns, rows) -> None:
        self.rows, self.total_rows = rows[:5000], len(rows)
        keys = numeric_columns(columns, self.rows)
        self.display = bool(keys)
        x = self.query_one("#plot-x", Select)
        y = self.query_one("#plot-y", Select)
        x.set_options([("x: row number", "__row__")] + [(f"x: {k}", k) for k in keys])
        y.set_options([(f"y: {k}", k) for k in keys])
        x.value = "__row__"
        if keys:
            y.value = (
                keys[1]
                if len(keys) > 1 and keys[0] in ("day", "step", "index")
                else keys[0]
            )
        self.draw()

    @on(Select.Changed, "#plot-x")
    @on(Select.Changed, "#plot-y")
    def axes_changed(self) -> None:
        self.draw()

    def draw(self) -> None:
        x, y = (
            self.query_one("#plot-x", Select).value,
            self.query_one("#plot-y", Select).value,
        )
        plot = self.query_one(ResearchPlot)
        if not isinstance(x, str) or not isinstance(y, str):
            plot.points = []
        else:
            plot.points = plot_points(self.rows, None if x == "__row__" else x, y)
            plot.x_label, plot.y_label = ("row number" if x == "__row__" else x), y
        plot.row_count, plot.total_rows = len(self.rows), self.total_rows
        plot.refresh()
