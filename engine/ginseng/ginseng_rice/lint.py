"""Lint a coord before anyone wears it: contrast, color-vision separation, glyph widths."""
from __future__ import annotations
import itertools
import sys
import unicodedata
from pathlib import Path
from .oklab import contrast, hex_to_oklab
from string import Formatter

from .tokens import VOICE_FIELDS, Coord, load_all

TEXT_ROLES = ("ink", "muted", "focus", "safe", "thin", "short", "inflow", "outflow",
              "opt_credit", "opt_liquidate", "opt_hybrid", "opt_protective")
MIN_TEXT = 4.5          # WCAG AA for normal text
MIN_STATE_DE = 0.09     # house rule: safe/thin/short stay apart in OKLab under simulated CVD


def _delta(a: str, b: str) -> float:
    return sum((x - y) ** 2 for x, y in zip(hex_to_oklab(a), hex_to_oklab(b))) ** 0.5


def _structural(ch: str) -> bool:
    return 0x2500 <= ord(ch) <= 0x259F   # box drawing + blocks: every TUI already depends on these


def check(coord: Coord) -> list[str]:
    p, problems = coord.palette, []
    for role in TEXT_ROLES:
        for ground in ("paper", "panel"):
            ratio = contrast(p[role], p[ground])
            if ratio < MIN_TEXT:
                problems.append(f"contrast  {role} on {ground} is {ratio:.2f}:1, needs {MIN_TEXT}")
    try:
        from colorspacious import cspace_convert
        import numpy as np
        for kind in ("deuteranomaly", "protanomaly", "tritanomaly"):
            space = {"name": "sRGB1+CVD", "cvd_type": kind, "severity": 100}
            sim = {}
            for state in ("safe", "thin", "short"):
                rgb = np.array([int(p[state][i:i + 2], 16) / 255 for i in (1, 3, 5)])
                out = np.clip(cspace_convert(rgb, space, "sRGB1"), 0, 1)
                sim[state] = "#" + "".join("%02X" % round(v * 255) for v in out)
            for a, b in itertools.combinations(sim, 2):
                d = _delta(sim[a], sim[b])
                if d < MIN_STATE_DE:
                    problems.append(f"cvd       {a}/{b} under {kind}: dE_ok {d:.3f} < {MIN_STATE_DE}")
    except ImportError:
        problems.append("note      colorspacious not installed; color-vision check skipped")
    for section in ("borders", "glyphs"):
        for key, text in getattr(coord, section).items():
            if key in ("panel", "focus"):
                continue          # these name Textual border types, not glyphs
            for ch in dict.fromkeys(text):
                eaw = unicodedata.east_asian_width(ch)
                if eaw in ("W", "F"):
                    problems.append(f"width     {section}.{key}: {ch!r} U+{ord(ch):04X} is double-width")
                elif eaw == "A" and not _structural(ch):
                    problems.append(f"width     {section}.{key}: {ch!r} U+{ord(ch):04X} is ambiguous-width")
    for state in ("calm", "thin", "short"):
        template = coord.voice.get(state)
        if template is None:
            problems.append(f"voice     [voice] is missing {state}")
            continue
        for _, field, _, _ in Formatter().parse(template):
            if field is not None and field not in VOICE_FIELDS:
                problems.append(f"voice     {state} uses unknown placeholder {{{field}}}")
    return problems


def main(argv: list[str]) -> int:
    coords = [Coord.load(Path(a)) for a in argv] if argv else list(load_all().values())
    failed = 0
    for coord in coords:
        problems = [x for x in check(coord) if not x.startswith("note")]
        print(f"{'ok  ' if not problems else 'FAIL'}  {coord.name}")
        for line in problems:
            print("      " + line)
        failed += bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
