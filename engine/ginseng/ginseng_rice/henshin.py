"""Henshin: change coords through short-lived themes interpolated in OKLab."""
from __future__ import annotations

import asyncio
import os

from textual.app import App

from .oklab import mix
from .tokens import ROLES, Coord


def ease_in_out(t: float) -> float:
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def frame_coords(src: Coord, dst: Coord, frames: int) -> list[Coord]:
    """The in-between palettes, excluding src and dst themselves."""
    out = []
    for i in range(1, frames):
        t = ease_in_out(i / frames)
        palette = {role: mix(src.palette[role], dst.palette[role], t) for role in ROLES}
        out.append(Coord(f"henshin-{i}", dst.label, dst.dark if t >= 0.5 else src.dark,
                         palette, dst.borders, dst.glyphs, dst.motion, dst.voice))
    return out


async def henshin(app: App, src: Coord, dst: Coord, frames: int = 4, duration: float = 0.24) -> None:
    """Walk the app through in-between themes, then land on dst.

    Set GINSENG_MOTION=0 to switch instantly. Each frame costs a full style refresh
    (about 59 ms headless on the machine this was tested on), so keep frames low.
    """
    if frames <= 1 or os.environ.get("GINSENG_MOTION") == "0":
        app.theme = dst.theme_name
        return
    steps = frame_coords(src, dst, frames)
    for step in steps:
        app.register_theme(step.to_theme(step.name))
    try:
        for step in steps:
            app.theme = step.name
            await asyncio.sleep(duration / frames)
    finally:
        app.theme = dst.theme_name
        for step in steps:
            app.unregister_theme(step.name)   # keep them out of the command palette
