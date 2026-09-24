"""Shared drawing helpers: the glyph kit for the active coord and styled line assembly."""
from __future__ import annotations

from typing import Any

from textual.content import Content

DEFAULTS = {
    "band_outer": "░", "band_inner": "▒", "median": "━", "zero": "┄",
    "gauge_on": "▰", "gauge_off": "▱", "pick": "▸", "voice_mark": "▸",
}
# At ornament level 0 every coord draws with this same plain kit.
UNDRESSED = {**DEFAULTS, "gauge_on": "█", "gauge_off": "░", "pick": ">", "voice_mark": "!"}
EIGHTHS = " ▁▂▃▄▅▆▇█"


def glyphs(app: Any) -> dict[str, str]:
    if getattr(app, "ornament_level", 3) == 0:
        return dict(UNDRESSED)
    kit = dict(DEFAULTS)
    coord = getattr(app, "coord", None)
    if coord is not None:
        kit.update({key: value for key, value in coord.glyphs.items() if key in DEFAULTS})
    return kit


class Line:
    """One row of styled text. Adjacent runs with the same style merge into one span."""

    __slots__ = ("parts",)

    def __init__(self) -> None:
        self.parts: list[tuple[str, str]] = []

    def add(self, text: str, style: str = "") -> "Line":
        if text:
            if self.parts and self.parts[-1][1] == style:
                self.parts[-1] = (self.parts[-1][0] + text, style)
            else:
                self.parts.append((text, style))
        return self


def assemble(lines: list[Line]) -> Content:
    parts: list[Any] = []
    for i, line in enumerate(lines):
        if i:
            parts.append("\n")
        parts.extend(text if not style else (text, style) for text, style in line.parts)
    return Content.assemble(*parts)


def place(row: list[str], col: int, text: str) -> None:
    """Write a label into a character row only if it fits and touches no other label."""
    if col < 0 or col + len(text) > len(row):
        return
    if any(ch != " " for ch in row[max(0, col - 1): col + len(text) + 1]):
        return
    row[col: col + len(text)] = list(text)
