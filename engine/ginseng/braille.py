"""Braille-dot rendering: resizing hand-drawn art, and plotting real engine
data at Braille's native 2x4 sub-cell resolution.

`normalize`/`scale` resize the fixed portrait/splash art (see `portrait.py`,
`splash.py`). `ridge_surface` and `rain_frame` are the tui's own plots --
`ridge_surface` a hidden-line depth-layered plot of real simulated cash
paths, `rain_frame` one frame of a cascade over the real per-path,
per-day historical indices `DrawBundle.index_matrix` resampled. Both take
plain sequences (no numpy import here), so the tui can pass a numpy array
straight through -- it merely needs to support `len()` and `row[i]`.
"""
from __future__ import annotations

_BITS = ((0, 3), (1, 4), (2, 5), (6, 7))


def _cells_from_canvas(canvas: list[list[bool]], sub_w: int, sub_h: int) -> list[str]:
    out = []
    for y in range(0, sub_h, 4):
        chars = []
        for x in range(0, sub_w, 2):
            value = 0
            for dy in range(4):
                for dx in range(2):
                    if canvas[y + dy][x + dx]:
                        value |= 1 << _BITS[dy][dx]
            chars.append(chr(0x2800 + value))
        out.append("".join(chars))
    return out


def ridge_surface(series: list, width: int, height: int, depth: int | None = None) -> list[str]:
    """A hidden-line ridge plot: each row of `series` is one real trajectory,
    front to back. A farther-back ridge never draws over a sub-cell a
    nearer one already claimed, so overlapping paths read as a genuine
    layered surface (each one's peaks occluding whatever sits behind it)
    instead of a tangle of overlapping scribbles.

    `depth` is the per-ridge vertical offset in sub-rows; left at its
    default it reserves about a third of the plot for stacking so the
    remaining two-thirds still show real curve shape.
    """
    if len(series) == 0 or width <= 0 or height <= 0:
        return [""] * max(0, height)
    sub_w, sub_h = width * 2, height * 4
    flat = [v for row in series for v in row]
    if not flat:
        return [""] * height
    lo, hi = min(flat), max(flat)
    span = (hi - lo) or 1.0
    if depth is None:
        depth = max(1, (sub_h // 3) // max(1, len(series)))

    thickness = max(1, depth - 1)
    canvas = [[False] * sub_w for _ in range(sub_h)]
    for i, row in enumerate(series):
        n = len(row)
        if n == 0:
            continue
        offset = min(i * depth, sub_h - 1)
        headroom = sub_h - 1 - offset
        for col in range(sub_w):
            value = row[min(n - 1, col * n // sub_w)]
            norm = (value - lo) / span
            line = offset + int(round((1 - norm) * headroom))
            for r in range(max(0, line), min(sub_h, line + thickness)):
                if not canvas[r][col]:
                    canvas[r][col] = True
    return _cells_from_canvas(canvas, sub_w, sub_h)


def rain_frame(index_matrix, width: int, height: int, phase: int, history_length: int) -> list[str]:
    """One frame of the resampled-day cascade. Cell (x, y) lights up
    proportionally to `index_matrix[path][day]` -- the real historical-day
    index the stationary bootstrap actually drew for that path and day
    (see `simulate.DrawBundle`) -- so a deeper, brighter column is a path
    that is genuinely leaning on older history, not a random flourish.
    `phase` shifts which day feeds each row, which is what makes repeated
    frames read as falling rather than static."""
    n_paths = len(index_matrix)
    n_days = len(index_matrix[0]) if n_paths else 0
    if not n_paths or not n_days or width <= 0 or height <= 0:
        return [""] * max(0, height)
    sub_w, sub_h = width * 2, height * 4
    canvas = [[False] * sub_w for _ in range(sub_h)]
    for y in range(sub_h):
        day = (y + phase) % n_days
        for x in range(sub_w):
            path = x % n_paths
            idx = index_matrix[path][day]
            level = (idx * 8) // max(1, history_length)
            canvas[y][x] = level > ((y % 4) * 2 + (x % 2))
    return _cells_from_canvas(canvas, sub_w, sub_h)


def normalize(art: str) -> list[str]:
    rows = [row.rstrip() for row in art.splitlines()]
    left = min(len(row) - len(row.lstrip("⠀")) for row in rows)
    return [row[left:] for row in rows]


def scale(rows: list[str], width: int) -> list[str]:
    """Downsample a Braille dot matrix to `width` cells, preserving aspect ratio."""
    native_width = max(map(len, rows))
    native_height = len(rows)
    width = max(1, min(width, native_width))
    if width == native_width:
        return rows
    height = max(1, round(native_height * width / native_width))
    out = []
    for y in range(height):
        line = []
        for x in range(width):
            value = 0
            for dy in range(4):
                for dx in range(2):
                    sx = min(native_width * 2 - 1, int((x * 2 + dx) * native_width / width))
                    sy = min(native_height * 4 - 1, int((y * 4 + dy) * native_height / height))
                    char = rows[sy // 4].ljust(native_width, "⠀")[sx // 2]
                    if (ord(char) - 0x2800) & (1 << _BITS[sy % 4][sx % 2]):
                        value |= 1 << _BITS[dy][dx]
            line.append(chr(0x2800 + value))
        out.append("".join(line))
    return out
