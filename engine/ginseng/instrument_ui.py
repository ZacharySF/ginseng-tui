"""Compact instrument panels driven by the current session."""

from __future__ import annotations

import math

from textual.content import Content
from textual.widgets import Static

from ginseng.braille import _cells_from_canvas


class WorkspaceCanvas(Static):
    """An explicitly decorative wireframe water surface, never a data plot."""

    def render(self):
        width = max(1, self.content_size.width)
        height = max(1, self.content_size.height - 3)
        sw, sh = width * 2, height * 4
        canvas = [[False] * sw for _ in range(sh)]

        def project(u, v):
            radius = math.hypot(u, v)
            z = math.cos(radius * 2.5) * math.exp(-radius * 0.35)
            return (sw * (0.5 + (u - v) * 0.063),
                    sh * (0.48 + (u + v) * 0.06 - z * 0.17))

        def line(a, b):
            x1, y1 = a
            x2, y2 = b
            steps = max(1, math.ceil(max(abs(x2 - x1), abs(y2 - y1))))
            for i in range(steps + 1):
                x = round(x1 + (x2 - x1) * i / steps)
                y = round(y1 + (y2 - y1) * i / steps)
                if 0 <= x < sw and 0 <= y < sh:
                    canvas[y][x] = True

        for axis in range(2):
            for n in range(19):
                fixed = -3 + n / 3
                points = [project(fixed, -3 + j / 12) if axis else
                          project(-3 + j / 12, fixed) for j in range(73)]
                for a, b in zip(points, points[1:]):
                    line(a, b)
        rows = _cells_from_canvas(canvas, sw, sh)
        return Content.assemble(
            ("+  LIQUID TRANQUILITY / STILL FIELD\n", "$g-muted"),
            *((row + "\n", "$g-accent" if i == height // 2 else "$g-band-inner")
              for i, row in enumerate(rows)),
            ("+  AMBIENT GEOMETRY / NOT MODEL OUTPUT", "$g-muted"),
        )


class SessionReadout(Static):
    def render(self):
        app = self.app
        return Content.assemble(
            ("OBSERVATIONS\n", "$g-muted"),
            (f"{app.last_paths!s} paths\n\n", "bold $g-ink"),
            (f"RUNS       {app.run_counter:04d}\n", "$g-muted"),
            (f"SEED       {app.last_seed}\n", "$g-muted"),
            (f"ELAPSED    {app.last_elapsed:.2f}s\n", "$g-muted"),
            (f"RATE       {app.last_throughput:,.0f}/s", "$g-focus"),
        )


class SessionRisk(Static):
    def render(self):
        values = self.app.risk_history[-20:]
        if not values:
            return Content.assemble(
                ("NO COMPLETED RUN\n\n", "$g-muted"),
                ("Shortfall history appears\nafter a simulation or exact run.", "$g-muted"),
            )
        cells = "▁▂▃▄▅▆▇█"
        parts = [("P(SHORTFALL) / SESSION\n\n", "$g-muted")]
        for value in values:
            role = "safe" if value < 0.05 else "thin" if value < 0.25 else "short"
            parts.append((cells[min(7, round(value * 7))], f"$g-{role}"))
        parts.append((f"\n\nLAST {values[-1]:.2%}   /   N={len(values)}", "$g-ink"))
        return Content.assemble(*parts)


class SessionEvents(Static):
    def render(self):
        events = self.app.event_log[-4:]
        return Content("\n".join(events) if events else "Session ready.\nNo engine events recorded.")
