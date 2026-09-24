"""The single supplied Braille portrait, kept as a lossless UTF-8 text asset."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

from ginseng.braille import _BITS, _cells_from_canvas

ART_DIR = Path(__file__).with_name("art")
DEFAULT_SCENE = "cyberpunk"


@dataclass(frozen=True)
class Scene:
    label: str
    caption: str
    art: str


def _scene(name: str, label: str, caption: str) -> Scene:
    return Scene(label, caption, (ART_DIR / f"{name}.txt").read_text(encoding="utf-8"))


SCENES = {
    "cyberpunk": _scene("cyberpunk", "Cyberpunk portrait", "after-hours transmission"),
}


@cache
def scene_rows(name: str) -> tuple[str, ...]:
    """Trim only empty outer margins; keep every internal cell and blank row."""
    rows = SCENES[name].art.splitlines()
    occupied = [i for i, row in enumerate(rows) if row.strip("⠀ ")]
    if not occupied:
        return ()
    rows = rows[occupied[0] : occupied[-1] + 1]
    left = min(len(row) - len(row.lstrip("⠀ ")) for row in rows if row.strip("⠀ "))
    right = max(len(row.rstrip("⠀ ")) for row in rows)
    return tuple(row[left:right].ljust(right - left, "⠀") for row in rows)


@lru_cache(maxsize=256)
def fit_scene(name: str, width: int, max_height: int | None = None) -> tuple[str, ...]:
    """Fit the full picture, pooling dots so thin strokes survive downsampling.

    A glyph is a 2×4 dot tile. Scaling both pixel axes by the same ratio
    preserves the terminal artwork's aspect ratio; no characters are cropped.
    """
    rows = scene_rows(name)
    if not rows or width <= 0 or (max_height is not None and max_height <= 0):
        return ()
    native_w, native_h = len(rows[0]), len(rows)
    ratio = min(1.0, width / native_w)
    if max_height is not None:
        ratio = min(ratio, max_height / native_h)
    out_w, out_h = max(1, int(native_w * ratio)), max(1, int(native_h * ratio))
    if (out_w, out_h) == (native_w, native_h):
        return rows
    canvas = [[False] * (out_w * 2) for _ in range(out_h * 4)]
    for y, row in enumerate(rows):
        for x, char in enumerate(row):
            bits = ord(char) - 0x2800
            for dy in range(4):
                for dx in range(2):
                    if bits & (1 << _BITS[dy][dx]):
                        dest_y = (y * 4 + dy) * out_h // native_h
                        dest_x = (x * 2 + dx) * out_w // native_w
                        canvas[dest_y][dest_x] = True
    return tuple(_cells_from_canvas(canvas, out_w * 2, out_h * 4))
