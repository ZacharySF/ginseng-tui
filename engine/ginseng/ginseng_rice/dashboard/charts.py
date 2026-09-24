"""Charts drawn straight into terminal cells: the cash-path fan and the trough histogram."""
from __future__ import annotations

from textual.content import Content
from textual.reactive import reactive
from textual.widget import Widget

from .fmt import compact, money
from .kit import EIGHTHS, Line, assemble, glyphs, place
from .model import DashboardData, PathBands


class FanChart(Widget):
    """Solid p25..p75 band, textured p5..p95 band, median on top. Anything under $0 turns the short color."""

    DEFAULT_CSS = "FanChart { text-wrap: nowrap; text-overflow: clip; }"
    bands: reactive[PathBands | None] = reactive(None)

    LABEL_W = 7
    RIGHT_W = 4

    def render(self) -> Content:
        bands, width, height = self.bands, self.size.width, self.size.height
        if bands is None:
            return assemble([Line().add("waiting for a run", "$g-muted")])
        plot_w, rows = width - self.LABEL_W - self.RIGHT_W, height - 1
        if plot_w < 8 or rows < 2:
            return self._sparkline(bands, width)
        kit = glyphs(self.app)
        top, bottom = max(bands.p95), min(min(bands.p5), 0.0)
        pad = (top - bottom) * 0.04 or 1.0
        top, bottom = top + pad, bottom - pad
        step = (top - bottom) / rows
        zero_row = min(rows - 1, max(0, int(top // step)))
        days = bands.days
        columns = []
        for col in range(plot_w):
            t = col * (days - 1) / (plot_w - 1)
            i = min(int(t), days - 2)
            f = t - i
            columns.append(tuple(q[i] + (q[i + 1] - q[i]) * f
                                 for q in (bands.p5, bands.p25, bands.p50, bands.p75, bands.p95)))
        right: dict[int, str] = {}
        for series, name in ((bands.p95, "p95"), (bands.p50, "p50"), (bands.p5, "p5")):
            right.setdefault(min(rows - 1, max(0, int((top - series[-1]) // step))), name)
        lines = []
        for r in range(rows):
            hi = top - r * step
            lo = hi - step
            if r == zero_row:
                label = "$0"
            elif r == 0:
                label = compact(hi - step / 2)
            elif r == rows - 1:
                label = compact(lo + step / 2)
            else:
                label = ""
            line = Line().add(label.rjust(self.LABEL_W - 1), "$g-muted").add("┤" if label else "│", "$g-rule")
            below = hi <= 0
            for p5, p25, p50, p75, p95 in columns:
                # A cell counts as short if it sits wholly under $0, or it is the $0 row on a day the p5 path is negative.
                short = below or (r == zero_row and p5 < 0)
                if lo <= p50 < hi:
                    line.add(kit["median"], ("bold $g-short" if short else "bold $g-median") + " on $g-band-inner")
                elif lo < p75 and hi > p25:
                    line.add(kit["band_inner"] if short else " ", "$g-short on $g-band-inner" if short else "on $g-band-inner")
                elif lo < p95 and hi > p5:
                    line.add(kit["band_outer"], ("$g-short" if short else "$g-band-inner") + " on $g-band-outer")
                elif r == zero_row:
                    line.add(kit["zero"], "$g-short")
                else:
                    line.add(" ")
            lines.append(line.add(" " + right.get(r, "").ljust(self.RIGHT_W - 1), "$g-muted"))
        axis = [" "] * width
        for day in sorted({1, max(2, days // 3), max(3, 2 * days // 3), days}):
            text = f"d{day}"
            col = self.LABEL_W + round((day - 1) * (plot_w - 1) / (days - 1))
            place(axis, col - (len(text) - 1 if day == days else 0), text)
        lines.append(Line().add("".join(axis), "$g-muted"))
        return assemble(lines)

    def _sparkline(self, bands: PathBands, width: int) -> Content:
        lo, hi = min(bands.p50), max(bands.p50)
        span = (hi - lo) or 1.0
        line = Line()
        for col in range(max(1, width)):
            value = bands.p50[round(col * (bands.days - 1) / max(1, width - 1))]
            line.add(EIGHTHS[1 + round((value - lo) / span * 7)], "$g-short" if value < 0 else "$g-median")
        return assemble([line])


class TroughHistogram(Widget):
    """How low each path's balance gets, binned: short below $0, thin under the cushion."""

    DEFAULT_CSS = "TroughHistogram { text-wrap: nowrap; text-overflow: clip; }"
    data: reactive[DashboardData | None] = reactive(None)

    def render(self) -> Content:
        data, width, height = self.data, self.size.width, self.size.height
        rows = height - 2
        if data is None or rows < 1 or width < 12:
            return assemble([Line()])
        edges, counts = data.trough_edges, data.trough_counts
        group = -(-len(counts) // width)  # merge bins when there are more bins than columns
        bins = [(edges[k], edges[min(k + group, len(counts))], sum(counts[k:k + group]))
                for k in range(0, len(counts), group)]
        bin_w = max(1, width // len(bins))
        tallest = max(n for _, _, n in bins) or 1
        lines = []
        for r in range(rows):
            line = Line()
            for _, hi, n in bins:
                level = round(n / tallest * rows * 8) - (rows - 1 - r) * 8
                role = "short" if hi <= 0 else ("thin" if hi <= data.cushion else "safe")
                line.add(EIGHTHS[max(0, min(8, level))] * bin_w, f"$g-{role}")
            lines.append(line)
        ticks = [" "] * width
        place(ticks, 0, compact(bins[0][0]))
        zero = next((i for i, (lo, _, _) in enumerate(bins) if lo >= 0), None)
        if zero:
            place(ticks, zero * bin_w - 1, "$0")
        last = compact(bins[-1][1])
        place(ticks, bin_w * len(bins) - len(last), last)
        lines.append(Line().add("".join(ticks), "$g-muted"))
        legend, used = Line(), 0
        for text, style in (("▇", "$g-short"), (" below $0  ", "$g-muted"), ("▇", "$g-thin"),
                            (f" under {money(data.cushion)}  ", "$g-muted"), ("▇", "$g-safe"), (" clear", "$g-muted")):
            if used + len(text) > width:
                break
            legend.add(text, style)
            used += len(text)
        lines.append(legend)
        return assemble(lines)
