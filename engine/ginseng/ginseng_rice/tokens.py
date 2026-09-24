"""A coord is a theme file: semantic roles in, a Textual Theme out."""
from __future__ import annotations
import tomllib
from dataclasses import dataclass
from pathlib import Path
from textual.theme import Theme

ROLES = (
    "paper", "panel", "ink", "muted", "rule", "focus", "accent",
    "band_outer", "band_inner", "median", "floor",
    "safe", "thin", "short", "inflow", "outflow",
    "opt_credit", "opt_liquidate", "opt_hybrid", "opt_protective", "stamp", "ornament",
)
VOICE_FIELDS = ("rlr", "c0", "c0_name", "C0_NAME", "sp", "sol", "sol1", "h")
COORD_DIR = Path(__file__).parent / "coords"


@dataclass(frozen=True)
class Coord:
    name: str
    label: str
    dark: bool
    palette: dict[str, str]
    borders: dict[str, str]
    glyphs: dict[str, str]
    motion: dict[str, float]
    voice: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> "Coord":
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        palette = data["palette"]
        missing = [role for role in ROLES if role not in palette]
        if missing:
            raise ValueError(f"{path.name}: palette is missing {', '.join(missing)}")
        meta = data["coord"]
        return cls(meta["name"], meta["label"], meta["dark"], palette,
                   data.get("borders", {}), data.get("glyphs", {}),
                   data.get("motion", {}), data.get("voice", {}))

    @property
    def theme_name(self) -> str:
        return f"ginseng-{self.name}"

    def css_variables(self) -> dict[str, str]:
        """Every role becomes $g-<role> in TCSS, e.g. band_outer -> $g-band-outer."""
        return {f"g-{role.replace('_', '-')}": self.palette[role] for role in ROLES}

    def to_theme(self, name: str | None = None) -> Theme:
        p = self.palette
        return Theme(
            name=name or self.theme_name,
            primary=p["focus"], secondary=p["accent"], accent=p["accent"],
            foreground=p["ink"], background=p["paper"], surface=p["paper"], panel=p["panel"],
            success=p["safe"], warning=p["thin"], error=p["short"],
            dark=self.dark,
            variables=self.css_variables() | {
                "border": p["focus"],
                "border-blurred": p["rule"],
                "footer-key-foreground": p["focus"],
                "block-cursor-background": p["focus"],
                "block-cursor-foreground": p["paper"],
            },
        )


def load_all(directory: Path = COORD_DIR) -> dict[str, Coord]:
    coords = (Coord.load(path) for path in sorted(directory.glob("*.toml")))
    return {coord.name: coord for coord in coords}
