"""Number formatting. One place, so every coord shows identical numbers."""
from __future__ import annotations

MINUS = "\u2212"  # neutral-width minus sign; the ASCII hyphen reads as a dash in tables


def money(value: float, plus: bool = False) -> str:
    text = f"${abs(value):,.0f}"
    if value <= -0.5:
        return MINUS + text
    return f"+{text}" if plus and value >= 0.5 else text


def compact(value: float) -> str:
    if abs(value) < 50:
        return "$0"
    text = f"${abs(value) / 1000:.1f}k"
    return MINUS + text if value < 0 else text


def pct(p: float) -> str:
    return f"{p * 100:.1f}%"
