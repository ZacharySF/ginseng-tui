"""Accessories: Trim (repeating motif rows), Soroban (bead gauge), Hanko (settlement stamps)."""
from __future__ import annotations
from rich.cells import cell_len
from textual.content import Content
from textual.reactive import reactive
from textual.widget import Widget


class Trim(Widget):
    """A one-row ribbon of a repeating motif: '─o' chain, '◡' lace, '◠◡' waves."""

    DEFAULT_CSS = "Trim { height: 1; color: $g-ornament; }"
    motif: reactive[str] = reactive("─")

    def __init__(self, motif: str = "─", **kwargs) -> None:
        super().__init__(**kwargs)
        self.motif = motif

    def render(self) -> Content:
        width = self.size.width
        unit = max(1, cell_len(self.motif))
        text = (self.motif * (width // unit + 1))
        while cell_len(text) > width:
            text = text[:-1]
        return Content(text)


class Soroban(Widget):
    """Days of solvency as abacus beads. Filled beads take the state color."""

    DEFAULT_CSS = "Soroban { height: 1; }"
    days: reactive[int] = reactive(0)
    horizon: reactive[int] = reactive(30)
    tone: reactive[str] = reactive("safe")

    def __init__(self, on: str = "◉", off: str = "◌", **kwargs) -> None:
        super().__init__(**kwargs)
        self.on, self.off = on, off

    def render(self) -> Content:
        label = f" {self.days}/{self.horizon}"
        slots = max(1, self.size.width - len(label))
        filled = round(self.days / self.horizon * slots)
        return Content.assemble(
            (self.on * filled, f"$g-{self.tone}"),
            (self.off * (slots - filled), "$g-rule"),
            (label, "$g-muted"),
        )


class Hanko(Widget):
    """済 settled, [未] scheduled, 見込 estimate. Width is measured, never assumed."""

    DEFAULT_CSS = """
    Hanko { width: auto; height: 1; }
    Hanko.-settled { background: $g-stamp; color: $g-paper; text-style: bold; }
    Hanko.-pending { color: $g-muted; }
    Hanko.-estimate { color: $g-thin; text-style: italic; }
    """
    GLYPHS = {"settled": "済", "pending": "[未]", "estimate": "見込"}

    def __init__(self, kind: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.kind = kind
        self.add_class(f"-{kind}")

    def render(self) -> Content:
        return Content(self.GLYPHS[self.kind])

    def get_content_width(self, container, viewport) -> int:
        return cell_len(self.GLYPHS[self.kind])
