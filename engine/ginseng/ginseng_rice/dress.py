"""Drop-in mixin. `class GinsengTUI(Dressable, App)` gets coords, henshin and the ornament budget."""
from __future__ import annotations

from .budget import OrnamentBudget
from .henshin import henshin
from .tokens import Coord, load_all

# Module level on purpose: Textual calls get_theme_variable_defaults() inside
# App.__init__, before any attribute you set in your own __init__ exists.
COORDS: dict[str, Coord] = load_all()


class Dressable:
    """Mix in before App.

    Colors follow the theme on their own. Glyphs don't, so whenever the coord or the
    ornament level changes, widgets with class `g-dressable` get `dress(coord)` called
    and widgets with class `g-glyphs` are refreshed.
    """

    DEFAULT_COORD = "seifuku"

    def get_theme_variable_defaults(self) -> dict[str, str]:
        base = super().get_theme_variable_defaults()  # type: ignore[misc]
        return {**base, **COORDS[self.DEFAULT_COORD].css_variables()}

    @property
    def coord(self) -> Coord:
        return COORDS[getattr(self, "_coord_name", self.DEFAULT_COORD)]

    @property
    def ornament_level(self) -> int:
        budget = getattr(self, "_ornament_budget", None)
        return 3 if budget is None else budget.level

    def dress_up(self, name: str | None = None) -> None:
        """Call from on_mount. Registers every coord and wears one without animation."""
        for coord in COORDS.values():
            self.register_theme(coord.to_theme())  # type: ignore[attr-defined]
        self._coord_name = name if name in COORDS else self.DEFAULT_COORD
        self.theme = self.coord.theme_name  # type: ignore[attr-defined]
        self.on_coord_changed(self.coord)
        self._redress()

    async def wear(self, name: str) -> None:
        src = self.coord
        self._coord_name = name
        dst = self.coord
        await henshin(self, src, dst,  # type: ignore[arg-type]
                      frames=int(dst.motion.get("henshin_frames", 4)),
                      duration=float(dst.motion.get("henshin_ms", 240)) / 1000)
        self.on_coord_changed(dst)
        self._redress()

    async def action_next_coord(self) -> None:
        names = list(COORDS)
        await self.wear(names[(names.index(self.coord.name) + 1) % len(names)])

    def on_coord_changed(self, coord: Coord) -> None:
        """Override for app-specific glyph swaps. Themes only carry color."""

    def report_shortfall(self, short_p: float) -> int:
        """Feed it the engine's shortfall probability after every run. Returns the ornament level."""
        budget = getattr(self, "_ornament_budget", None)
        if budget is None:
            budget = self._ornament_budget = OrnamentBudget()
        previous = budget.level
        level = budget.update(short_p)
        for screen in self.screen_stack:  # type: ignore[attr-defined]
            for n in range(4):
                screen.set_class(level == n, f"-ornate-{n}")
        if level != previous:
            self._redress()
        return level

    def _redress(self) -> None:
        coord = self.coord
        for widget in self.query(".g-dressable"):  # type: ignore[attr-defined]
            widget.dress(coord)
        for widget in self.query(".g-glyphs"):  # type: ignore[attr-defined]
            widget.refresh()
