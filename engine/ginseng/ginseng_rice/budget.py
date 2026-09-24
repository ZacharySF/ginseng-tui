"""The ornament budget: decoration shrinks as shortfall odds grow, with hysteresis so it never flickers."""
from __future__ import annotations

# (upper bound on shortfall probability, ornament level)
BANDS = ((0.05, 3), (0.25, 2), (0.40, 1))
MARGIN = 0.02


def level_for(short_p: float, margin: float = 0.0) -> int:
    for bound, level in BANDS:
        if short_p < bound - margin:
            return level
    return 0


class OrnamentBudget:
    def __init__(self, level: int = 3) -> None:
        self.level = level

    def update(self, short_p: float) -> int:
        worse = level_for(short_p)
        if worse < self.level:
            self.level = worse                      # bad news strips ornaments at once
        else:
            self.level = max(self.level, level_for(short_p, MARGIN))  # good news must clear a margin
        return self.level
