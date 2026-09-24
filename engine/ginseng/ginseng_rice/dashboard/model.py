"""What the dashboard draws. Build one per engine run; widgets never touch the engine.

Every field is checked on construction, so a bad adapter fails loudly instead of
drawing a plausible-looking wrong chart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

KINDS = ("credit", "liquidate", "hybrid", "protective")
STATUSES = ("settled", "pending", "estimate")
CALM_BELOW, THIN_BELOW = 0.05, 0.25


def _floats(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


@dataclass(frozen=True)
class PathBands:
    """Daily balance percentiles across simulated paths, day 1 first."""

    p5: tuple[float, ...]
    p25: tuple[float, ...]
    p50: tuple[float, ...]
    p75: tuple[float, ...]
    p95: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("p5", "p25", "p50", "p75", "p95"):
            object.__setattr__(self, name, _floats(getattr(self, name)))
        days = len(self.p50)
        if days < 2:
            raise ValueError("PathBands needs at least 2 days")
        for name in ("p5", "p25", "p75", "p95"):
            if len(getattr(self, name)) != days:
                raise ValueError(f"PathBands.{name} has {len(getattr(self, name))} days, p50 has {days}")
        for d in range(days):
            q = (self.p5[d], self.p25[d], self.p50[d], self.p75[d], self.p95[d])
            if any(b < a - 1e-6 for a, b in zip(q, q[1:])):
                raise ValueError(f"PathBands day {d + 1}: percentiles out of order {q}")

    @property
    def days(self) -> int:
        return len(self.p50)


@dataclass(frozen=True)
class FundingOption:
    name: str          # shown as-is, e.g. "credit line"
    kind: str          # one of KINDS; picks the color role
    cost: float        # expected dollar cost of closing the gap this way
    ready_days: int    # 0 means usable today
    tail: float        # shortfall probability left after funding, 0..1

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"FundingOption.kind {self.kind!r} not in {KINDS}")
        if not 0.0 <= self.tail <= 1.0:
            raise ValueError(f"FundingOption.tail {self.tail} outside 0..1")
        if self.cost < 0 or self.ready_days < 0:
            raise ValueError("FundingOption cost and ready_days must be >= 0")


@dataclass(frozen=True)
class LedgerEntry:
    day: int           # relative to today; negative is the past
    label: str
    amount: float      # signed dollars, inflow positive
    status: str        # one of STATUSES

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"LedgerEntry.status {self.status!r} not in {STATUSES}")


@dataclass(frozen=True)
class Validation:
    """Leave a field None when the engine didn't produce it; the ledger line hides it."""

    oracle_ok: bool | None = None
    coverage: float | None = None    # observed coverage of the p95 band, 0..1
    windows: int | None = None
    kupiec_p: float | None = None


@dataclass(frozen=True)
class DashboardData:
    plan: str
    paths: int
    sampler: str
    draw_id: str
    reserve_to_add: float        # dollars to add so the p5 path never goes below $0
    shortfall_p: float           # share of paths whose lowest balance is below $0
    cvar95_trough: float         # mean of the worst 5% of lowest balances, dollars
    deficit_dollar_days: float   # mean over paths of the summed daily dollars below $0
    solvent_days: int            # days before the p5 path first goes below $0; horizon if never
    bands: PathBands
    trough_edges: tuple[float, ...]
    trough_counts: tuple[int, ...]
    cushion: float = 400.0       # lowest balances in [0, cushion) draw as thin
    options: tuple[FundingOption, ...] = ()
    ledger: tuple[LedgerEntry, ...] = ()
    validation: Validation = field(default_factory=Validation)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trough_edges", _floats(self.trough_edges))
        object.__setattr__(self, "trough_counts", tuple(int(c) for c in self.trough_counts))
        object.__setattr__(self, "options", tuple(self.options))
        object.__setattr__(self, "ledger", tuple(self.ledger))
        if not 0.0 <= self.shortfall_p <= 1.0:
            raise ValueError(f"shortfall_p {self.shortfall_p} outside 0..1")
        if self.reserve_to_add < 0:
            raise ValueError("reserve_to_add must be >= 0")
        if not 0 <= self.solvent_days <= self.bands.days:
            raise ValueError(f"solvent_days {self.solvent_days} outside 0..{self.bands.days}")
        if len(self.trough_counts) != len(self.trough_edges) - 1:
            raise ValueError("trough_counts needs exactly one fewer entry than trough_edges")
        if any(b <= a for a, b in zip(self.trough_edges, self.trough_edges[1:])):
            raise ValueError("trough_edges must strictly increase")

    @property
    def horizon(self) -> int:
        return self.bands.days

    @property
    def state(self) -> str:
        if self.shortfall_p < CALM_BELOW:
            return "calm"
        return "thin" if self.shortfall_p < THIN_BELOW else "short"

    @property
    def tone(self) -> str:
        return {"calm": "safe", "thin": "thin", "short": "short"}[self.state]

    @property
    def cheapest(self) -> FundingOption | None:
        return min(self.options, key=lambda option: option.cost) if self.options else None
