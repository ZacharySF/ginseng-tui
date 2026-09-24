"""Ginseng's Gen-X soft club terminal workspace.

Simulation and exact dashboards sit alongside the complete research studio,
with one visual identity, searchable commands and the original ASCII art collection.
Launch with ``ginseng tui``; see docs/tui-studio.md for optional engine extras.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button,
    ContentSwitcher,
    DataTable,
    Footer,
    Input,
    Label,
    OptionList,
    RichLog,
    Select,
    Static,
)
from textual.widgets.option_list import Option
from textual.worker import Worker, WorkerState

from ginseng.braille import rain_frame, ridge_surface
from ginseng.exact import enumerate_exact
from ginseng.ginseng_rice.dashboard.adapter import EngineRun, from_engine
from ginseng.ginseng_rice.dashboard.board import Dashboard
from ginseng.ginseng_rice.dashboard.model import (
    CALM_BELOW,
    THIN_BELOW,
    DashboardData,
    FundingOption,
    LedgerEntry,
    PathBands,
    Validation,
)
from ginseng.ginseng_rice.dress import Dressable
from ginseng.ginseng_rice.tokens import Coord
from ginseng.inputs import fixture, load_input
from ginseng.instrument_ui import (
    SessionEvents,
    SessionReadout,
    SessionRisk,
    WorkspaceCanvas,
)
from ginseng.numerical import diagnostics, environment, manifest, run_core
from ginseng.provenance import source_fingerprint
from ginseng.rice_charts import FundingBars, PathSignals
from ginseng.rice_ui import ArtPanel, active_scene, scene_content
from ginseng.sampling import prepare_history
from ginseng.studio import EXPERIMENTS
from ginseng.studio_ui import ResearchStudio

SOFT_CLUB = Coord.load(Path(__file__).parent / "ginseng_rice" / "softclub.toml")

FIXTURES = ["canonical", "tiny", "zero-heavy", "drought-heavy"]
SAMPLERS = ["mc", "sobol", "legacy_mc"]
ESTIMATORS = ["path", "initial-block-cmc"]
ARTIFACT_DIR = Path("artifacts/tui")

# run_exact() below always calls enumerate_exact() with no arguments, so
# these are read off its own defaults rather than duplicated by hand.
_EXACT_DEFAULTS = inspect.signature(enumerate_exact).parameters
EXACT_OPENING_CASH = _EXACT_DEFAULTS["opening_cash"].default
EXACT_SEQUENCE_COUNT = len(_EXACT_DEFAULTS["net"].default) ** len(_EXACT_DEFAULTS["deterministic"].default)

# Lifecycle and risk-tier text use the same semantic roles as the charts.
LIFECYCLE_ROLE = {"READY": "primary", "COMPUTE": "accent", "COMPLETE": "accent", "FAULT": "error"}
TIER_ROLE = {"idle": "primary", "safe": "success", "thin": "warning", "short": "error"}
RISK_LABEL = {"idle": "--", "safe": "LOW", "thin": "MODERATE", "short": "HIGH"}

SPARK_BLOCKS = " ▁▂▃▄▅▆▇█"
SPARK_WIDTH = 12


def risk_sparkline(history: list[float]) -> str:
    """A real seismograph of this session's own runs -- every shortfall
    probability actually computed so far, most recent on the right, each
    bar tinted by the same calm/thin/short tiers the dashboard body uses.
    Empty slots (no run yet) stay blank rather than reading as zero risk."""
    recent = history[-SPARK_WIDTH:]
    pad = [None] * (SPARK_WIDTH - len(recent))
    cells = []
    for p in pad + recent:
        if p is None:
            cells.append(" ")
            continue
        level = min(8, max(1, round(p * 8 / 0.5)))
        role = "success" if p < CALM_BELOW else ("warning" if p < THIN_BELOW else "error")
        cells.append(f"[${role}]{SPARK_BLOCKS[level]}[/]")
    return "".join(cells)


def money(x: float) -> str:
    return f"${x:,.2f}"


# ---------------------------------------------------------------------------
# Pure simulation wrappers -- unchanged business logic. run_core/enumerate_exact
# are the actual engine; everything below only *reads* their output.
# ---------------------------------------------------------------------------


def load_case(kind: str, params: dict[str, Any] | None):
    """The case a run will use, resolved without sampling anything -- cheap
    enough to call from the UI thread so `RunningPanel` can show real
    obligations before the worker's `run_core`/`enumerate_exact` call even
    starts. Mirrors `run_simulation`'s own case resolution exactly."""
    if kind != "simulate":
        from ginseng.inputs import fixture as fx
        return fx("tiny")
    if params["source"] == "fixture":
        return fixture(params["fixture_name"])
    return load_input(Path(params["path"]))


def run_simulation(params: dict[str, Any]) -> tuple[dict, np.ndarray, Any, Any]:
    case = load_case("simulate", params)
    block_length = params["block_length"]
    requested = block_length if block_length is not None else (7 if case.name == "tiny" else None)
    prepared = prepare_history(case.state, requested)
    bundle, matrix, summary = run_core(
        case,
        prepared,
        params["sampler"],
        params["paths"],
        params["seed"],
        params["horizon"],
        params["material_horizon"],
        params["replicate"],
        estimator=params["estimator"],
    )
    result = dict(
        summary=summary,
        manifest=manifest(case, prepared, bundle, summary, params["estimator"]),
        diagnostics=diagnostics(case, prepared, bundle, matrix),
    )
    return result, matrix, case, bundle


def run_exact() -> dict:
    from ginseng.inputs import fixture as fx
    from ginseng.provenance import digest

    result = enumerate_exact()
    result["manifest"] = dict(
        environment=environment(),
        input_hash=digest(asdict(fx("tiny"))),
        method="independent rational enumeration",
        block_length=7,
        horizon=4,
        result_hash=digest(result),
    )
    return result


def _engine_run_from_simulation(result: dict, matrix: np.ndarray, case, bundle) -> EngineRun:
    return EngineRun(
        kind="simulate",
        summary=result["summary"],
        manifest=result["manifest"],
        matrix=matrix,
        opening_cash=case.state.immediate_funding,
        state=case.state,
        obligations=case.obligations,
        bundle=bundle,
    )


def _engine_run_from_exact(result: dict) -> EngineRun:
    sequences = result["sequences"]
    matrix = np.array([row["cumulative"] for row in sequences], dtype=float)
    weights = np.array([row["probability"] for row in sequences], dtype=float)
    return EngineRun(
        kind="exact",
        summary=result["summary"],
        manifest=result["manifest"],
        matrix=matrix,
        opening_cash=EXACT_OPENING_CASH,
        weights=weights,
    )


def _dashboard_to_dict(data: DashboardData) -> dict:
    return asdict(data)


def _dashboard_from_dict(payload: dict) -> DashboardData:
    payload = dict(payload)
    bands = PathBands(**payload.pop("bands"))
    options = tuple(FundingOption(**o) for o in payload.pop("options", ()))
    ledger = tuple(LedgerEntry(**e) for e in payload.pop("ledger", ()))
    validation = Validation(**payload.pop("validation", {}))
    return DashboardData(bands=bands, options=options, ledger=ledger, validation=validation, **payload)


# ---------------------------------------------------------------------------
# Sidebar: compact operator chip (portrait + status block), instrument-panel
# styled subsystem nav.
# ---------------------------------------------------------------------------


class Portrait(Static):
    """The original artwork, tinted to match the last result’s risk tier."""

    DEFAULT_CSS = "Portrait { width: auto; min-width: 100%; height: auto; text-wrap: nowrap; }"
    DEFAULT_CLASSES = "g-glyphs"
    tone: reactive[str] = reactive("focus")

    def render(self):
        return scene_content(active_scene(self.app), self.tone)

    def set_tone(self, tone: str) -> None:
        self.tone = tone


class OperatorPanel(Static):
    """STATE tracks the run lifecycle (READY/COMPUTE/COMPLETE/FAULT); RISK/
    P(x) track the last result's shortfall tier -- two distinct axes that a
    single reused label previously conflated. Tier names (idle/safe/thin/
    short) match DashboardData.tone, so the sidebar and the dashboard body
    always agree on how risky a run was."""

    lifecycle: reactive[str] = reactive("READY")
    tier: reactive[str] = reactive("idle")
    engine: reactive[str] = reactive("STANDBY")
    value_text: reactive[str] = reactive("--")
    flavor: reactive[str] = reactive("awaiting model parameters")

    def render(self):
        lifecycle_text = "READY" if self.lifecycle == "READY" else self.lifecycle
        lrole, rrole = LIFECYCLE_ROLE[self.lifecycle], TIER_ROLE[self.tier]
        return (
            f"[bold $primary]SIMULATION / EXACT[/]\n"
            f"[dim $g-muted]STATE[/]   [bold ${lrole}]{lifecycle_text:<13}[/]\n"
            f"[dim $g-muted]ENGINE[/]  [$foreground]{self.engine[:10]:<10}[/]\n"
            f"[dim $g-muted]RISK[/]    [bold ${rrole}]{RISK_LABEL[self.tier]:<10}[/]\n"
            f"[dim $g-muted]P(x)[/]    [bold ${rrole}]{self.value_text:<10}[/]\n"
            f"[dim $primary]> {self.flavor}[/]"
        )

    def set_idle(self) -> None:
        self.lifecycle, self.tier, self.engine, self.value_text = "READY", "idle", "STANDBY", "--"
        self.flavor = "awaiting model parameters"

    def set_busy(self, engine: str) -> None:
        self.lifecycle, self.tier, self.engine, self.value_text = "COMPUTE", "idle", engine.upper(), "--"
        self.flavor = {
            "sobol": "initializing low-discrepancy sequence",
            "mc": "initializing pseudorandom sequence",
            "legacy_mc": "initializing legacy generator",
        }.get(engine, "initializing sampler")

    def set_result(self, data: DashboardData) -> None:
        self.lifecycle, self.tier = "COMPLETE", data.tone
        self.engine, self.value_text = data.sampler.upper(), f"{data.shortfall_p:.4f}"
        self.flavor = "convergence nominal" if data.tone != "short" else "shortfall risk elevated"

    def set_error(self) -> None:
        self.lifecycle, self.tier, self.engine, self.value_text = "FAULT", "idle", "FAULT", "--"
        self.flavor = "run aborted -- see fault panel"


# ---------------------------------------------------------------------------
# Panels -- thin single-line borders with a border_title label instead of a
# boxed heading; real telemetry instead of decorative filler.
# ---------------------------------------------------------------------------


class Splash(Static):
    """Show the selected original portrait on the boot screen."""

    DEFAULT_CLASSES = "g-glyphs"

    def render(self):
        return scene_content(active_scene(self.app))


GINSENG_LOGO = r"""
  ___ ___ _  _ ___ ___ _  _  ___
 / __|_ _| \| / __| __| \| |/ __|
| (_ || || .` \__ \ _|| .` | (_ |
 \___|___|_|\_|___/___|_|\_|\___|""".strip("\n")


def _logo_content(diag_a: str, diag_b: str, ink: str) -> Content:
    """Color the wordmark by stroke direction, not position: every `\\`
    takes one role, every `/` the other, verticals and underscores stay the
    coord's own ink -- the same two-tone diagonal look as the reference
    image, built from real per-character spans instead of a flat string."""
    spans = []
    for ch in GINSENG_LOGO:
        role = diag_a if ch == "\\" else diag_b if ch == "/" else ink
        spans.append((ch, role))
    return Content.assemble(*spans)


class BootScreen(Screen):
    """First thing you see, alone, before the instrument boots -- dismissed
    by any key, click, or after the sequence finishes and idles a while.
    The sidebar portrait only appears once this is gone, so the two never
    show at the same time.

    The wordmark and the character both appear together, right after a
    brief flicker -- they used to wait behind the whole provenance stream
    below, which read as a bug (the art simply took a long time to show
    up). The stream is still real -- `environment()`'s actual dependency
    versions and thread-pool env vars, then a genuine SHA-256 of every
    source file under this package, ending on the exact digest
    `source_fingerprint()` feeds into every run's manifest -- it's just
    detail underneath now, not a gate in front of the art."""

    DEFAULT_CSS = """
    BootScreen { align: center middle; background: $g-paper; }
    #boot-frame { layout: horizontal; width: 1fr; height: 1fr; max-width: 140; padding: 1; }
    #boot-identity { width: 1fr; height: 1fr; padding: 1; margin-right: 1; border: solid $g-rule; background: $g-panel; }
    #boot-identity, #boot-log { border-title-color: $g-muted; }
    #boot-crt { color: $g-muted; text-align: center; height: 2; }
    #boot-logo { width: 1fr; height: auto; text-align: center; margin: 1 0; }
    #boot-art-view { width: 1fr; height: 1fr; min-height: 8; }
    #boot-art { width: auto; min-width: 100%; height: auto; text-wrap: nowrap; content-align: center top; }
    #boot-hint { color: $g-muted; text-align: center; width: 1fr; height: 2; }
    #boot-log { width: 1fr; height: 1fr; border: solid $g-rule; background: $g-panel; padding: 1; }
    BootScreen.-compact #boot-frame { layout: vertical; }
    BootScreen.-compact #boot-identity { width: 1fr; height: auto; min-height: 18; margin: 0 0 1 0; }
    BootScreen.-compact #boot-art-view { height: 14; }
    BootScreen.-compact #boot-log { width: 1fr; height: 10; }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="boot-frame"):
            with Vertical(id="boot-identity"):
                yield Static("", id="boot-crt")
                yield Static("", id="boot-logo")
                with VerticalScroll(id="boot-art-view"):
                    yield Splash(id="boot-art")
                yield Static("", id="boot-hint")
            yield RichLog(id="boot-log", markup=True, auto_scroll=True, max_lines=80, wrap=False)

    def on_mount(self) -> None:
        self.set_class(self.app.size.width < 110, "-compact")
        self.query_one("#boot-identity").border_title = "GINSENG / INITIALIZE"
        self.query_one("#boot-log").border_title = "PROVENANCE / SOURCE VERIFICATION"
        self._skipped = False
        self.query_one(Splash).display = False
        self.query_one("#boot-hint", Static).display = False
        self.run_worker(self._sequence(), exclusive=True)

    def on_key(self, event) -> None:
        self._leave()

    def on_click(self) -> None:
        self._leave()

    def _leave(self) -> None:
        self._skipped = True
        if self.is_current:
            self.dismiss()

    def _role(self, name: str) -> str:
        """RichLog parses Rich markup, not Textual's `$role` Content markup,
        so boot-log lines resolve the coord's actual hex directly -- still
        the active coord's own color, just fetched a layer earlier."""
        return self.app.coord.palette[name]

    async def _sleep(self, seconds: float) -> bool:
        """Sleep unless the boot's already been skipped. Returns False when
        the caller should stop -- checked between every write so a keypress
        lands within one step, never after the whole sequence finishes.

        `GINSENG_MOTION=0` (same switch `henshin` honors) skips the real
        delay, not the content: every real line still gets written, just
        with no pacing, so the sequence settles on its final, reveal state
        practically at once -- which is also what makes it deterministic
        for snapshot tests instead of racing wall-clock time."""
        if self._skipped:
            return False
        if os.environ.get("GINSENG_MOTION") == "0":
            await asyncio.sleep(0)
            return not self._skipped
        await asyncio.sleep(seconds)
        return not self._skipped

    async def _sequence(self) -> None:
        crt = self.query_one("#boot-crt", Static)
        logo = self.query_one("#boot-logo", Static)
        log = self.query_one("#boot-log", RichLog)

        # Stage 1: a brief thin-line flicker -- dashes, not a solid CRT bar.
        width = 44
        for n in range(4, width + 1, 6):
            crt.update(("╌" * n).center(width))
            if not await self._sleep(0.012):
                return
        crt.update("[dim $g-muted]GINSENG / SOFT CLUB     —     CASH-FLOW RESEARCH[/]")
        if not await self._sleep(0.12):
            return

        # Stage 2: the wordmark and the character reveal together, right
        # away -- the whole point of moving them ahead of the log below.
        logo.update(_logo_content("$g-focus", "$g-accent", "$g-ink"))
        self.query_one(Splash).display = True
        hint = self.query_one("#boot-hint", Static)
        hint.update("[$g-muted]press any key to enter / research workspace[/]")
        hint.display = True
        if not await self._sleep(0.2):
            return

        # Stage 3: the real provenance stream, now just detail underneath --
        # environment()'s actual dependency versions and thread-pool env
        # vars, then a genuine SHA-256 of every source file, ending on the
        # exact digest every manifest carries as `source_fingerprint`.
        muted, safe, thin = (self._role(r) for r in ("muted", "safe", "thin"))
        env = environment()
        log.write(f"[{muted}]boot[/]  python {env.get('python', '--')}  {env.get('platform', '--')}")
        log.write(f"[{muted}]boot[/]  cpu    {env.get('cpu', '--')}")
        if not await self._sleep(0.06):
            return
        for name, ver in env.get("dependencies", {}).items():
            log.write(f"[{safe}]alloc[/]   {name:<12} {ver or '?'}")
            if not await self._sleep(0.03):
                return
        for key, val in env.get("threads", {}).items():
            log.write(f"[{muted}]env[/]    {key}={val or 'unset'}")
        if not await self._sleep(0.1):
            return

        root = Path(__file__).parent
        files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
        log.write(f"[{thin}]hash[/]   fingerprinting {len(files)} source files")
        if not await self._sleep(0.05):
            return
        for path in files:
            digest = sha256(path.read_bytes()).hexdigest()
            log.write(f"[{muted}]sha256[/] {digest[:16]}  {path.relative_to(root).as_posix()}")
            if not await self._sleep(0.008):
                return
        log.write(f"[bold {thin}]source_fingerprint[/] {source_fingerprint()}")


class WelcomePanel(Vertical):
    def compose(self) -> ComposeResult:
        yield Static("WORKSPACE / OVERVIEW", classes="panel-heading")
        yield WorkspaceCanvas(id="home-canvas", classes="instrument-panel")
        with Vertical(id="workflow-panel", classes="instrument-panel"):
            yield Static("WORKFLOW TABLE  /  SELECT TO OPEN", classes="panel-heading")
            yield DataTable(id="home-workflows", cursor_type="row")
            with Horizontal(classes="desk-actions"):
                yield Button("Simulation", id="home-simulate")
                yield Button("Research", id="home-studio")
                yield Button("Artwork", id="home-art")

    def on_mount(self) -> None:
        self.query_one("#home-canvas").border_title = "VISUAL FIELD / LIQUID STUDY"
        table = self.query_one("#home-workflows", DataTable)
        table.add_columns("WORKSPACE", "SCOPE", "SOURCE")
        for key, title, scope, source in (
            ("simulate", "01 / Simulation", "MC · Sobol · conditional", "fixture / file"),
            ("exact", "02 / Exact oracle", "Exhaustive enumeration", "tiny fixture"),
            ("studio", "03 / Research", f"{len(EXPERIMENTS)} experiments", "editable recipe"),
            ("surface", "04 / Risk atlas", "Cash × time", "cash paths"),
            ("portfolio", "05 / Portfolio", "Tax lots · allocation", "portfolio model"),
            ("history", "06 / Archive", "Saved engine runs", "local files"),
        ):
            table.add_row(title, scope, source, key=key)

    @on(DataTable.RowSelected, "#home-workflows")
    def open_workflow(self, event: DataTable.RowSelected) -> None:
        self.app.navigate(str(event.row_key.value))



class SimulateForm(VerticalScroll):
    def on_mount(self) -> None:
        self.border_title = "SIMULATE // MC-SOBOL ENGINE"
        self.update_visibility()

    def compose(self) -> ComposeResult:
        yield Static("SIMULATION / CONFIGURE RUN", classes="panel-heading")
        with Horizontal(id="simulation-setup"):
            with Vertical(classes="setup-panel instrument-panel"):
                yield Static("01 / INPUT SOURCE", classes="rule")
                yield Select([("fixture", "fixture"), ("input file", "file")], id="f-source",
                             value="fixture", allow_blank=False)
                yield Label("fixture", id="label-fixture", classes="field-label")
                yield Select([(name, name) for name in FIXTURES], id="f-fixture",
                             value="canonical", allow_blank=False)
                yield Label("input file path", id="label-path", classes="field-label")
                yield Input(value="examples/tiny-history.json", id="f-path")

            with Vertical(classes="setup-panel instrument-panel"):
                yield Static("02 / SAMPLING METHOD", classes="rule")
                yield Select([(name, name) for name in SAMPLERS], id="f-sampler",
                             value="mc", allow_blank=False)
                yield Select([(name, name) for name in ESTIMATORS], id="f-estimator",
                             value="path", allow_blank=False)

        yield Static("03 / PARAMETERS", classes="rule")
        with Horizontal(classes="field-row"):
            with Vertical(classes="field-col"):
                yield Label("paths", classes="field-label")
                yield Input(value="2048", type="integer", id="f-paths")
            with Vertical(classes="field-col"):
                yield Label("seed", classes="field-label")
                yield Input(value="42", type="integer", id="f-seed")
            with Vertical(classes="field-col"):
                yield Label("replicate", classes="field-label")
                yield Input(value="0", type="integer", id="f-replicate")
        with Horizontal(classes="field-row"):
            with Vertical(classes="field-col"):
                yield Label("horizon", classes="field-label")
                yield Input(placeholder="default", type="integer", id="f-horizon", valid_empty=True)
            with Vertical(classes="field-col"):
                yield Label("material horizon", classes="field-label")
                yield Input(placeholder="none", type="integer", id="f-material-horizon", valid_empty=True)
            with Vertical(classes="field-col"):
                yield Label("block length", classes="field-label")
                yield Input(placeholder="auto", type="integer", id="f-block-length", valid_empty=True)

        yield Static("Fixture inputs are synthetic. Review assumptions before running.", classes="hint")
        yield Button(Text("[ Run simulation ]"), id="btn-run", variant="success")

    @on(Select.Changed, "#f-source")
    def _source_changed(self, _event: Select.Changed) -> None:
        self.update_visibility()

    def update_visibility(self) -> None:
        is_fixture = self.query_one("#f-source", Select).value == "fixture"
        for sel in ("#label-fixture", "#f-fixture"):
            self.query_one(sel).display = is_fixture
        for sel in ("#label-path", "#f-path"):
            self.query_one(sel).display = not is_fixture

    def collect(self) -> dict[str, Any]:
        def int_of(widget_id, default=None):
            raw = self.query_one(widget_id, Input).value.strip()
            return default if raw == "" else int(raw)

        return dict(
            source=self.query_one("#f-source", Select).value,
            fixture_name=self.query_one("#f-fixture", Select).value,
            path=self.query_one("#f-path", Input).value.strip(),
            sampler=self.query_one("#f-sampler", Select).value,
            estimator=self.query_one("#f-estimator", Select).value,
            paths=int_of("#f-paths", 2048),
            seed=int_of("#f-seed", 42),
            replicate=int_of("#f-replicate", 0),
            horizon=int_of("#f-horizon"),
            material_horizon=int_of("#f-material-horizon"),
            block_length=int_of("#f-block-length"),
        )


class ObligationsView(Static):
    """The real input obligations for this run -- known the instant the case
    loads, well before `run_core` samples anything, so it's honest to show
    immediately instead of waiting for a result that doesn't exist yet."""

    DEFAULT_CSS = "ObligationsView { height: auto; color: $foreground; }"

    def show(self, case) -> None:
        if not case.obligations:
            self.update("[dim]no scheduled obligations[/]")
            return
        lines = [
            f"{o.due_in_days:>4}d  {o.label:<26} {o.amount:>+11,.2f}  {o.transaction_type.value}"
            for o in case.obligations
        ]
        self.update("\n".join(lines))


class ExecutingWave(Static):
    """Motion while the worker thread is inside `run_core`/`enumerate_exact`.

    Deliberately abstract, not a progress bar: `run_core` samples the whole
    draw in one call with no incremental hook to observe, so anything
    claiming to show live sampler iteration here would be invented. This is
    a synthetic waveform through the same hidden-line ridge renderer real
    results use in `VolSurface` -- the visual language carries over, the
    data doesn't, and the real per-path surface replaces it the instant the
    worker actually returns."""

    DEFAULT_CSS = "ExecutingWave { height: 10; color: $g-thin; }"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._t = 0
        self._timer = None

    def on_mount(self) -> None:
        if os.environ.get("GINSENG_MOTION") != "0":
            self._timer = self.set_interval(0.08, self._tick)

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()

    def _tick(self) -> None:
        if not self.display:
            return
        self._t += 1
        self.refresh()

    def render(self):
        width, height = max(10, self.size.width), max(4, self.size.height)
        series = []
        for i in range(10):
            phase = self._t * 0.18 + i * 0.6
            series.append([
                math.sin(x * 0.25 + phase) + 0.15 * math.sin(x * 0.7 - phase * 1.3)
                for x in range(48)
            ])
        return Content("\n".join(ridge_surface(series, width, height)))


class VolSurface(Static):
    """A hidden-line depth plot of a real sample of this run's simulated
    cash paths -- the actual `matrix` `cash_paths` returned, not the fan
    chart's summary bands, ordered worst-trough-first so the most stressed
    paths sit in front."""

    DEFAULT_CSS = "VolSurface { height: 12; color: $g-focus; }"
    # always_update: comparing an old numpy array against a new one (or
    # against None) for equality raises "ambiguous truth value" -- a new
    # run's matrix is always genuinely new, so the equality check buys
    # nothing here and always_update skips it.
    matrix: reactive[object] = reactive(None, always_update=True)

    def render(self):
        if self.matrix is None or len(self.matrix) == 0:
            return Content("No path matrix available for this run.")
        width, height = max(10, self.size.width), max(4, self.size.height)
        order = np.argsort(self.matrix.min(axis=1))
        pick = order[np.linspace(0, len(order) - 1, min(16, len(order))).astype(int)]
        return Content("\n".join(ridge_surface(self.matrix[pick], width, height)))


class IndexRain(Static):
    """Live cascade of the real per-path, per-day historical-day indices the
    stationary bootstrap actually resampled for this run
    (`DrawBundle.index_matrix`) -- see `braille.rain_frame`. Ticks its own
    timer only while mounted; skips redraw while hidden behind another
    panel."""

    DEFAULT_CSS = "IndexRain { height: 10; color: $g-thin; }"
    bundle: reactive[object] = reactive(None, always_update=True)  # see VolSurface.matrix

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._phase = 0
        self._timer = None

    def on_mount(self) -> None:
        if os.environ.get("GINSENG_MOTION") != "0":
            self._timer = self.set_interval(0.12, self._tick)

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()

    def _tick(self) -> None:
        if self.bundle is None or not self.display:
            return
        self._phase += 1
        self.refresh()

    def render(self):
        if self.bundle is None or self.bundle.index_matrix is None:
            return Content("No draw bundle for this run.")
        width, height = max(10, self.size.width), max(4, self.size.height)
        lines = rain_frame(self.bundle.index_matrix, width, height, self._phase, self.bundle.history_length)
        return Content("\n".join(lines))


class RunningPanel(VerticalScroll):
    """What's real before the worker even starts: the loaded case's own
    obligations and configuration. What's after them is honestly abstract
    -- see `ExecutingWave`."""

    def compose(self) -> ComposeResult:
        yield Static("ENGINE / RUN IN PROGRESS", classes="panel-heading")
        with Horizontal(id="run-inputs"):
            with Vertical(classes="setup-panel instrument-panel"):
                yield Static("CONFIGURATION", classes="panel-heading")
                yield Static("", id="run-config", classes="mono-block")
            with Vertical(classes="setup-panel instrument-panel"):
                yield Static("INPUT / OBLIGATIONS", classes="panel-heading")
                yield ObligationsView(id="run-obligations")
        with Vertical(classes="instrument-panel", id="run-activity"):
            yield Static("SAMPLING / ENGINE RUNNING", classes="panel-heading")
            yield ExecutingWave(id="run-wave")
            yield Static("Abstract activity display. Results appear when the run completes.", classes="mono-dim")

    def set_context(self, case, kind: str) -> None:
        state = case.state
        self.query_one("#run-config", Static).update(
            f"case      {case.name}\n"
            f"as_of     {state.as_of}   horizon {state.forecast_horizon}d   method {kind}\n"
            f"buffer    {money(state.operating_buffer)}   opening {money(state.immediate_funding)}   "
            f"coverage {state.coverage_target:.0%}"
        )
        self.query_one(ObligationsView).show(case)


class ResultPanel(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield Static("ANALYSIS / CASH-FLOW OBSERVATORY", classes="panel-heading")
        with Horizontal(id="surface-row"):
            with Vertical(classes="surface-col instrument-panel"):
                yield Static("CASH PATHS / OBSERVED DRAWS", classes="panel-heading")
                yield VolSurface(id="vol-surface")
            with Vertical(id="draw-panel", classes="surface-col instrument-panel"):
                yield Static("DRAW BUNDLE / INDEX FIELD", classes="panel-heading")
                yield IndexRain(id="index-rain")
        yield Dashboard(id="dashboard")
        with Horizontal(id="cash-signals"):
            yield PathSignals(id="path-signals")
            yield FundingBars(id="funding-bars")
        with Vertical(id="log-pane", classes="instrument-panel"):
            yield Static("ACTIVITY / ENGINE EVENTS", classes="panel-heading")
            yield Static("", id="event-log", classes="mono-dim")
        yield Button("Save run", id="btn-save", variant="primary")


class HistoryPanel(VerticalScroll):
    def on_mount(self) -> None:
        self.border_title = "RUN ARCHIVE // artifacts/tui"
        self.query_one(DataTable).add_columns("saved", "plan", "sampler", "reserve", "p(shortfall)")

    def compose(self) -> ComposeResult:
        yield Label("no archived runs -- save one from a result panel", id="history-empty", classes="hint")
        yield DataTable(id="history-table", cursor_type="row")


class ErrorPanel(VerticalScroll):
    def on_mount(self) -> None:
        self.border_title = "FAULT // RUN FAILED"

    def compose(self) -> ComposeResult:
        yield Static("", id="error-text", classes="mono-block tier-high")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

NAV_OPTIONS = [
    ("welcome", "01 / HOME", "OVERVIEW"),
    ("simulate", "02 / SIMULATE", "CASH PATHS"),
    ("exact", "03 / EXACT", "ORACLE"),
    ("studio", "04 / RESEARCH", f"{len(EXPERIMENTS)} EXPERIMENTS"),
    ("surface", "05 / ATLAS", "CASH × TIME"),
    ("portfolio", "06 / PORTFOLIO", "LOT LAB"),
    ("history", "07 / ARCHIVE", "SAVED RUNS"),
    ("art", "08 / ART", "ORIGINAL PORTRAIT"),
    ("quit", "QUIT", "END SESSION"),
]


def _nav_label(tag: str, hint: str) -> Content:
    # Content, not a markup string: an Option label built from plain str
    # brackets in `hint` could otherwise be misread as a markup tag.
    return Content.assemble((tag.replace(" / ", " "), "$g-muted"))


# The same defaults SimulateForm.collect() falls back to, for the command
# palette's one-keystroke "just run something" shortcuts.
QUICK_SIMULATE = dict(source="fixture", fixture_name="canonical", path="", sampler="mc",
                       estimator="path", paths=2048, seed=42, replicate=0,
                       horizon=None, material_horizon=None, block_length=None)


class WorkspaceCommands(Provider):
    """Search the workspaces, experiments and original artwork."""

    def _commands(self) -> list[tuple[str, str, object]]:
        app = self.app
        commands = []
        commands.append(("run: canonical simulate (mc, 2048 paths)", "quick simulate launch",
                          (app.launch, "simulate", dict(QUICK_SIMULATE))))
        commands.append(("run: exact oracle", "quick exact-oracle launch", (app.launch, "exact", None)))
        commands.extend((f"research: {entry.title}", entry.description,
                         (app.open_studio, entry.key)) for entry in EXPERIMENTS)
        commands.extend((f"workspace: {tag}", hint, (app.navigate, key))
                        for key, tag, hint in NAV_OPTIONS if key != "quit")
        return commands

    async def discover(self) -> Hits:
        for text, help_text, (func, *args) in self._commands():
            yield DiscoveryHit(text, partial(func, *args), help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for text, help_text, (func, *args) in self._commands():
            score = matcher.match(text)
            if score > 0:
                yield Hit(score, matcher.highlight(text), partial(func, *args), help=help_text)


class GinsengApp(Dressable, App):
    TITLE = "ginseng"
    SUB_TITLE = "soft club / research workspace"
    COMMANDS = App.COMMANDS | {WorkspaceCommands}
    CSS_PATH = ["ginseng_rice/rice.tcss", "ginseng_rice/dashboard/dashboard.tcss", "tui.tcss"]

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "go_welcome", "Menu"),
        Binding("ctrl+w", "art_collection", "Artwork"),
        Binding("ctrl+b", "toggle_sidebar", "Focus"),
        Binding("ctrl+r", "research", "Research"),
    ]

    def __init__(self):
        super().__init__()
        self.register_theme(SOFT_CLUB.to_theme())
        self.theme = SOFT_CLUB.theme_name
        self.last_result: dict | None = None
        self.last_data: DashboardData | None = None
        self.last_kind: str | None = None
        self.last_run: EngineRun | None = None
        self._history_payloads: dict[str, dict] = {}
        self.run_counter = 0
        self.last_engine = "--"
        self.last_paths: Any = "--"
        self.last_seed: Any = "--"
        self.last_elapsed = 0.0
        self.last_throughput = 0.0
        self.event_log: list[str] = []
        self.risk_history: list[float] = []
        self._pulse = 0
        self._marquee = 0
        self._env: dict | None = None
        self._sidebar_hidden = False
        self._status = "READY"
        self._simulation_busy = False

    @property
    def coord(self) -> Coord:
        return SOFT_CLUB

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return {**App.get_theme_variable_defaults(self), **SOFT_CLUB.css_variables()}

    def dress_up(self, name: str | None = None) -> None:
        self.theme = SOFT_CLUB.theme_name
        self._redress()

    async def wear(self, name: str) -> None:
        # Legacy dress actions cannot change the workspace’s fixed identity.
        self.dress_up()

    async def action_next_coord(self) -> None:
        pass

    def action_change_theme(self) -> None:
        pass

    def action_toggle_dark(self) -> None:
        pass

    def get_system_commands(self, screen):
        for command in super().get_system_commands(screen):
            if command.title != "Theme":
                yield command

    def compose(self) -> ComposeResult:
        yield Static("", id="telemetry")
        with Horizontal(id="workspace"):
            with VerticalScroll(id="sidebar"):
                yield Static("G / S\n────────────", id="sidebar-brand")
                yield OptionList(
                    *(Option(_nav_label(tag, hint), id=key) for key, tag, hint in NAV_OPTIONS),
                    id="nav",
                )
            with ContentSwitcher(initial="panel-welcome", id="body"):
                yield WelcomePanel(id="panel-welcome")
                yield SimulateForm(id="panel-form")
                yield RunningPanel(id="panel-running")
                yield ResultPanel(id="panel-result")
                yield HistoryPanel(id="panel-history")
                yield ErrorPanel(id="panel-error")
                yield ResearchStudio(id="panel-studio")
                yield ArtPanel(id="panel-art")
            with VerticalScroll(id="inspector"):
                with Vertical(id="inspector-art", classes="instrument-panel"):
                    yield Static("ORIGINAL / PORTRAIT", classes="panel-heading")
                    with VerticalScroll(id="portrait-viewport"):
                        yield Portrait(id="portrait")
                yield OperatorPanel(id="operator", classes="instrument-panel")
                yield SessionReadout(id="session-readout", classes="instrument-panel")
                yield SessionRisk(id="session-risk", classes="instrument-panel")
                yield SessionEvents(id="session-events", classes="instrument-panel")
        yield Static("", id="session-tape")
        yield Footer()

    def on_mount(self) -> None:
        self.dress_up()
        self.update_telemetry("READY")
        self._apply_responsive_layout()
        self.set_interval(0.8, self._tick_pulse)
        self.run_worker(self._load_env(), exclusive=False)
        self.push_screen(BootScreen())

    async def _load_env(self) -> None:
        """`environment()` shells out to git and hashes the source tree, so
        it runs off the UI thread once at startup -- the same real
        dependency/thread-env data the boot log streamed, kept around for
        the telemetry strip's THREADS readout instead of re-shelling out
        on every 0.8s tick."""
        self._env = await asyncio.to_thread(environment)
        self.update_telemetry(self._status)

    def _tick_pulse(self) -> None:
        """Small CRT-like heartbeat in the telemetry strip. Frozen under
        GINSENG_MOTION=0: a real 0.8s wall-clock tick racing however long a
        test's worker takes to finish is exactly the kind of nondeterminism
        that flag exists to remove."""
        if os.environ.get("GINSENG_MOTION") == "0":
            return
        self._pulse = (self._pulse + 1) % 4
        self._marquee += 2
        self.update_telemetry(self._status)

    def on_resize(self, event) -> None:
        # App processes the new terminal size after user resize handlers.
        self.call_after_refresh(self._apply_responsive_layout)

    def _apply_responsive_layout(self) -> None:
        """Keep navigation usable while reducing decoration in smaller terminals."""
        width = self.size.width
        self.query_one("#inspector-art").display = self.size.height >= 32
        self.query_one(Portrait).display = width >= 100 and self.size.height >= 32
        sidebar = self.query_one("#sidebar")
        sidebar.styles.width = 17 if width >= 110 else 16
        self.query_one("#inspector").display = width >= 120 and not self._sidebar_hidden
        sidebar.display = not self._sidebar_hidden and width >= 65
        for screen in self.screen_stack:
            screen.set_class(width < 110, "-compact")
            screen.set_class(width < 65, "-tiny")
            screen.set_class(self.size.height < 32, "-short")
        self.query_one(Footer).compact = width < 110

    def _marquee_text(self, width: int) -> str:
        """A real scroll through this session's own event log -- not
        fabricated hex, the exact lines `show_result`/`show_error` append."""
        body = "  ◆  ".join(self.event_log[-5:]) if self.event_log else "awaiting first run"
        loop = body + "     " + body
        offset = self._marquee % (len(body) + 5)
        return loop[offset:offset + width]

    def update_telemetry(self, status: str) -> None:
        self._status = status
        self.query_one("#telemetry", Static).update(
            f"[bold $g-ink]ginseng[/]  [$g-muted]/ RESEARCH TERMINAL[/]"
            f"    [${LIFECYCLE_ROLE[status]}]{status}[/]"
            f"    [$g-muted]RUN {self.run_counter:04d}  /  {self.last_engine}[/]"
        )
        self.query_one("#session-tape", Static).update(
            Content("SESSION / " + (self.event_log[-1] if self.event_log else "Ready. Select a workflow to begin."))
        )
        for panel_type in (SessionReadout, SessionRisk, SessionEvents):
            self.query_one(panel_type).refresh()

    def action_go_welcome(self) -> None:
        self.query_one("#body", ContentSwitcher).current = "panel-welcome"
        self.query_one("#nav", OptionList).highlighted = 0

    def action_toggle_sidebar(self) -> None:
        self._sidebar_hidden = not self._sidebar_hidden
        self._apply_responsive_layout()

    def action_research(self) -> None:
        self.open_studio()

    def action_art_collection(self) -> None:
        self.navigate("art")

    def on_coord_changed(self, coord) -> None:
        self.sub_title = "soft club / research workspace"

    def open_studio(self, key: str | None = None) -> None:
        self.query_one("#body", ContentSwitcher).current = "panel-studio"
        self.query_one("#nav", OptionList).highlighted = next(i for i, option in enumerate(NAV_OPTIONS) if option[0] == "studio")
        if key:
            self.query_one(ResearchStudio).open_experiment(key)

    def navigate(self, key: str) -> None:
        for index, (name, _, _) in enumerate(NAV_OPTIONS):
            if name == key:
                self.query_one("#nav", OptionList).highlighted = index
                break
        if key == "welcome":
            self.action_go_welcome()
        elif key == "simulate":
            self.query_one("#body", ContentSwitcher).current = "panel-form"
        elif key == "exact":
            self.launch("exact", None)
        elif key in ("studio", "surface", "portfolio"):
            self.open_studio(None if key == "studio" else key)
        elif key == "history":
            self.refresh_history()
            self.query_one("#body", ContentSwitcher).current = "panel-history"
        elif key == "art":
            self.query_one("#body", ContentSwitcher).current = "panel-art"
            self.query_one("#nav", OptionList).highlighted = next(
                i for i, option in enumerate(NAV_OPTIONS) if option[0] == "art"
            )
        elif key == "quit":
            self.exit()

    @on(OptionList.OptionSelected, "#nav")
    def _nav_selected(self, event: OptionList.OptionSelected) -> None:
        self.navigate(event.option.id)

    @on(Button.Pressed, "#home-simulate")
    def _home_simulate(self) -> None:
        self.navigate("simulate")

    @on(Button.Pressed, "#home-studio")
    def _home_studio(self) -> None:
        self.open_studio()

    @on(Button.Pressed, "#home-art")
    def _home_art(self) -> None:
        self.action_art_collection()

    @on(Button.Pressed, "#btn-run")
    def _run_pressed(self) -> None:
        form = self.query_one(SimulateForm)
        try:
            params = form.collect()
        except ValueError:
            self.notify("Numeric fields must be valid integers.", severity="error")
            return
        self.launch("simulate", params)

    @on(Button.Pressed, "#btn-save")
    def _save_pressed(self) -> None:
        if not self.last_data:
            return
        try:
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            out = ARTIFACT_DIR / f"{stamp}.json"
            payload = {"dashboard": _dashboard_to_dict(self.last_data), "engine_result": self.last_result}
            with out.open("x") as handle:
                handle.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            self.notify(f"saved {out}", title="♡ saved", severity="information")
        except (OSError, ValueError) as error:
            self.notify(str(error), title="Could not save run", severity="error")

    @on(DataTable.RowSelected, "#history-table")
    def _history_row_selected(self, event: DataTable.RowSelected) -> None:
        payload = self._history_payloads.get(str(event.row_key.value))
        if payload:
            self.last_result = payload.get("engine_result")
            self.last_data = _dashboard_from_dict(payload.get("dashboard", payload))
            self._present_result(self.last_data, "RESULT // ARCHIVED RUN")

    # ---- background work ------------------------------------------------
    def launch(self, kind: str, params: dict | None) -> None:
        if self._simulation_busy:
            self.notify("A simulation is already running.")
            return
        engine = params["sampler"] if kind == "simulate" else "oracle"
        try:
            case = load_case(kind, params)
        except (ValueError, OSError, KeyError, TypeError) as error:
            self.show_error(error)
            return
        self._simulation_busy = True
        self.query_one(OperatorPanel).set_busy(engine)
        self.query_one(Portrait).set_tone("thin")
        self.last_engine = engine.upper()
        self.last_paths = params["paths"] if kind == "simulate" else EXACT_SEQUENCE_COUNT
        self.last_seed = params["seed"] if kind == "simulate" else "--"
        self.update_telemetry("COMPUTE")
        ts = datetime.now().strftime("%H:%M:%S")
        self.event_log = [f"{ts}  engine.init sampler={engine} paths={self.last_paths}"]
        self.query_one(RunningPanel).set_context(case, kind)
        self.query_one("#body", ContentSwitcher).current = "panel-running"
        self._run_started = datetime.now()
        self.run_job(kind, params)

    @work(thread=True, exclusive=True, exit_on_error=False)
    def run_job(self, kind: str, params: dict | None):
        if kind == "simulate":
            result, matrix, case, bundle = run_simulation(params)
            run = _engine_run_from_simulation(result, matrix, case, bundle)
        else:
            result = run_exact()
            run = _engine_run_from_exact(result)
        return kind, result, from_engine(run), run

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "run_job":
            return
        if event.state in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            self._simulation_busy = False
        if event.state == WorkerState.SUCCESS:
            kind, result, data, run = event.worker.result
            self.show_result(kind, result, data, run)
        elif event.state == WorkerState.ERROR:
            self.show_error(event.worker.error)

    # ---- result rendering ------------------------------------------------
    def set_urgency(self, tone: str) -> None:
        """Shift the instrument's own chrome, not just the mascot, toward
        the alarm color -- same rule as everywhere else in this app: it only
        moves for `thin`/`short`, and idle/safe leave it alone."""
        self.screen.set_class(tone == "thin", "-urgent-thin")
        self.screen.set_class(tone == "short", "-urgent-short")

    def _present_result(self, data: DashboardData, title: str, run: EngineRun | None = None) -> None:
        panel = self.query_one(ResultPanel)
        panel.border_title = title
        panel.query_one(Dashboard).show(data)
        panel.query_one(PathSignals).data = data
        panel.query_one(FundingBars).data = data
        panel.query_one("#event-log", Static).update("\n".join(self.event_log[-6:]))
        panel.query_one(VolSurface).matrix = run.matrix if run is not None else None
        panel.query_one(IndexRain).bundle = run.bundle if run is not None else None
        self.query_one(OperatorPanel).set_result(data)
        self.query_one(Portrait).set_tone(data.tone)
        self.set_urgency(data.tone)
        self.last_engine = data.sampler.upper()
        self.last_paths = data.paths
        self.update_telemetry("COMPLETE")
        self.query_one("#body", ContentSwitcher).current = "panel-result"

    def show_result(self, kind: str, result: dict, data: DashboardData, run: EngineRun) -> None:
        self.last_result = result
        self.last_data = data
        self.last_kind = kind
        self.last_run = run
        self.run_counter += 1
        self.risk_history.append(data.shortfall_p)
        elapsed = (datetime.now() - self._run_started).total_seconds() if hasattr(self, "_run_started") else 0.0
        self.last_elapsed = elapsed
        self.last_throughput = data.paths / elapsed if elapsed > 0 else 0.0
        ts = datetime.now().strftime("%H:%M:%S")
        self.event_log.append(f"{ts}  engine.done elapsed={elapsed:.2f}s")
        self.event_log.append(f"{ts}  risk.compute p_shortfall={data.shortfall_p:.4f} state={data.state}")
        if run.bundle is not None:
            self.event_log.append(f"{ts}  draw.bundle id={run.bundle.bootstrap_draw_id} "
                                   f"paths={run.bundle.n_paths} history={run.bundle.history_length}")
        title = "RESULT // ENUMERATION ORACLE" if kind != "simulate" else "RESULT // RUN TELEMETRY"
        self._present_result(data, title, run)

    def show_error(self, error: BaseException) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.event_log.append(f"{ts}  fault {type(error).__name__}: {error}")
        self.query_one(OperatorPanel).set_error()
        self.query_one(Portrait).set_tone("short")
        self.set_urgency("short")
        self.query_one("#error-text", Static).update(Text(f"{type(error).__name__}: {error}"))
        self.update_telemetry("FAULT")
        self.query_one("#body", ContentSwitcher).current = "panel-error"
        self.notify(str(error), title="run failed", severity="error", timeout=8)

    def refresh_history(self) -> None:
        table = self.query_one("#history-table", DataTable)
        table.clear()
        self._history_payloads.clear()
        rows = []
        if ARTIFACT_DIR.exists():
            rows = sorted(ARTIFACT_DIR.glob("*.json"), reverse=True)
        for path in rows:
            try:
                payload = json.loads(path.read_text())
                dashboard = payload.get("dashboard", payload)
                _dashboard_from_dict(dashboard)
                reserve = money(dashboard["reserve_to_add"])
                prob = f"{dashboard['shortfall_p']:.4f}"
                plan, sampler = dashboard.get("plan", "?"), dashboard.get("sampler", "")
            except (ValueError, OSError, KeyError, TypeError):
                continue
            self._history_payloads[str(path)] = payload
            table.add_row(path.name, plan, sampler, reserve, prob, key=str(path))
        self.query_one("#history-empty", Label).display = not self._history_payloads


def main(argv=None) -> int:
    # A stray warnings.warn() writes straight to the terminal and corrupts
    # the alternate screen; the interactive UI has no channel to show it.
    warnings.simplefilter("ignore")
    # Fullscreen TUI: NO_COLOR is for piped text output — it would make
    # Textual strip every color via Monochrome filter.  Guard truecolor
    # detection too: Rich's _TERM_COLORS maps "kitty" → EIGHT_BIT.
    os.environ.pop("NO_COLOR", None)
    os.environ.setdefault("COLORTERM", "truecolor")
    import argparse
    parser = argparse.ArgumentParser(prog="ginseng-tui", description="Gen-X soft club research workspace")
    parser.parse_args(argv)
    GinsengApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
