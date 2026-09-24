"""Soft club research studio: editable recipes, inspectable results and notebook."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import signal
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Collapsible,
    DataTable,
    Input,
    Label,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from ginseng.rice_charts import TableVisual
from ginseng.studio import BY_KEY, EXPERIMENTS, SERVICE_ROUTES, result_tables

NOTEBOOK_DIR = Path("artifacts/tui/notebook")


class RiskAtlas(Static):
    """A compact terminal heatmap with numeric legend and table counterpart."""

    def show(self, result):
        rows = result.get("surface", []) if isinstance(result, dict) else []
        self.display = bool(rows)
        if not rows:
            return
        from textual.content import Content

        offsets = list(dict.fromkeys(row["extra_cash"] for row in rows))
        days = max(row["day"] for row in rows)
        stride = max(1, (days + 39) // 40)
        by_cell = {
            (row["extra_cash"], row["day"]): row["shortfall_probability"]
            for row in rows
        }
        parts = [("CASH-BUFFER ATLAS  ·  ever below $0\n", "bold $primary")]
        parts.append(
            (f"extra cash ↓   day 1 → {days}  (one cell / {stride} days)\n", "$g-muted")
        )
        for cash in offsets[:24]:
            parts.append((f"${cash:>8,.0f}  ", "$foreground"))
            for day in range(1, days + 1, stride):
                p = by_cell[(cash, day)]
                glyph = (
                    "·"
                    if p == 0
                    else "░"
                    if p < 0.05
                    else "▒"
                    if p < 0.25
                    else "▓"
                    if p < 0.5
                    else "█"
                )
                role = "$success" if p < 0.05 else "$warning" if p < 0.25 else "$error"
                parts.append((glyph, role))
            parts.append((f"  {by_cell[(cash, days)]:6.1%}\n", "$foreground"))
        parts.append(
            (
                "· 0%   ░ <5%   ▒ <25%   ▓ <50%   █ ≥50%  | final-day risk at right",
                "$g-muted",
            )
        )
        self.update(Content.assemble(*parts))


class ResearchStudio(Vertical):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active_key = "precision"
        self.result = None
        self.tables = {}
        self.notebook = []
        self.drafts = {}
        self.job = None
        self.busy = False
        self.baseline = None
        self.cancel_directory = None
        self.cancel_path = None
        self.cooperative = False

    def compose(self) -> ComposeResult:
        yield Static("RESEARCH / EXPERIMENT WORKBENCH", classes="panel-heading")
        with Horizontal(classes="studio-picker"):
            yield Input(
                placeholder="Search experiments… tail, portfolio, replay",
                id="studio-search",
            )
            yield Select(
                [(f"{x.group} · {x.title}", x.key) for x in EXPERIMENTS],
                value="precision",
                allow_blank=False,
                id="studio-operation",
            )
        yield Static("", id="studio-description", markup=False)
        with TabbedContent(id="studio-tabs"):
            with TabPane("Recipe", id="studio-recipe"):
                yield Select(
                    [
                        (label, str(i))
                        for i, (label, _, _, _) in enumerate(SERVICE_ROUTES)
                    ],
                    prompt="Choose an application route",
                    id="studio-route",
                )
                with Collapsible(
                    title="Quick setup · apply values to the recipe",
                    collapsed=True,
                    id="studio-quick",
                ):
                    with Horizontal(classes="quick-fields"):
                        with Vertical(classes="field-col"):
                            yield Label("Fixture")
                            yield Select(
                                [
                                    (x, x)
                                    for x in (
                                        "canonical",
                                        "tiny",
                                        "zero-heavy",
                                        "drought-heavy",
                                    )
                                ],
                                value="canonical",
                                allow_blank=False,
                                id="quick-fixture",
                            )
                        with Vertical(classes="field-col"):
                            yield Label("Paths")
                            yield Input("2048", type="integer", id="quick-paths")
                        with Vertical(classes="field-col"):
                            yield Label("Days")
                            yield Input("30", type="integer", id="quick-horizon")
                        with Vertical(classes="field-col"):
                            yield Label("Seed")
                            yield Input("42", type="integer", id="quick-seed")
                    yield Input(
                        placeholder="Local history JSON (optional; overrides fixture)",
                        id="quick-input",
                    )
                    yield Button("Apply quick setup", id="quick-apply")
                yield Static(
                    "Edit the recipe · input overrides fixture · dollars unless labeled otherwise",
                    classes="mono-dim",
                )
                yield TextArea(
                    json.dumps(BY_KEY["precision"].defaults, indent=2),
                    id="studio-params",
                    show_line_numbers=True,
                    soft_wrap=False,
                )
                with Horizontal(classes="studio-actions"):
                    yield Button("Run experiment", id="studio-run", variant="primary")
                    yield Button("Reset recipe", id="studio-reset")
                    yield Button("Cancel", id="studio-cancel", disabled=True)
            with TabPane("Results", id="studio-results"):
                with VerticalScroll(id="studio-result-scroll"):
                    with Horizontal(id="studio-analysis"):
                        with Vertical(id="studio-evidence", classes="instrument-panel"):
                            yield Static("EVIDENCE / TABLE + SIGNAL", classes="panel-heading")
                            yield Select(
                                [], id="studio-table-choice", prompt="Select a result table"
                            )
                            yield DataTable(
                                id="studio-table", cursor_type="row", zebra_stripes=True
                            )
                            yield TableVisual(id="studio-visual")
                            yield RiskAtlas(id="studio-atlas")
                        with VerticalScroll(id="studio-findings", classes="instrument-panel"):
                            yield Static("RUN / FINDINGS", classes="panel-heading")
                            yield Static(
                                "No result yet. Configure a recipe and run the experiment.",
                                id="studio-summary", markup=False,
                            )
                    yield Label(
                        "Complete result · parameters + provenance + engine output",
                        classes="field-label",
                    )
                    yield TextArea(
                        "{}", id="studio-json", read_only=True, soft_wrap=False
                    )
                with Horizontal(classes="studio-actions"):
                    yield Button("JSON", id="studio-export")
                    yield Button("CSV", id="studio-csv")
                    yield Button("Pin", id="studio-pin")
                    yield Button("Compare", id="studio-compare")
            with TabPane("Notebook", id="studio-notebook"):
                yield Static(
                    "Session discoveries · select a run to revisit it. Exports persist the complete recipe.",
                    classes="mono-dim",
                )
                yield DataTable(
                    id="studio-notebook-table", cursor_type="row", zebra_stripes=True
                )
                yield Input(
                    placeholder="Path to an exported experiment JSON",
                    id="studio-import-path",
                )
                yield Button("Open saved experiment", id="studio-import")
        yield Static(
            "Ready when you are.  Ctrl+P opens every workspace.",
            id="studio-status",
            markup=False,
        )

    def on_mount(self):
        self.query_one("#studio-description", Static).update(
            BY_KEY[self.active_key].description
        )
        self.query_one("#studio-atlas").display = False
        self.query_one(TableVisual).display = False
        self.query_one("#studio-route").display = False
        self.query_one("#studio-notebook-table", DataTable).add_columns(
            "#", "Experiment", "Source", "Elapsed"
        )

    def open_experiment(self, key):
        self.query_one("#studio-search", Input).value = ""
        selector = self.query_one("#studio-operation", Select)
        selector.set_options([(f"{x.group} · {x.title}", x.key) for x in EXPERIMENTS])
        selector.value = key
        self.select_experiment(key)
        self.query_one("#studio-tabs", TabbedContent).active = "studio-recipe"

    def select_experiment(self, key):
        if key == self.active_key:
            return
        editor = self.query_one("#studio-params", TextArea)
        self.drafts[self.active_key] = editor.text
        self.active_key = key
        editor.load_text(
            self.drafts.get(key, json.dumps(BY_KEY[key].defaults, indent=2))
        )
        self.query_one("#studio-description", Static).update(BY_KEY[key].description)
        self.query_one("#studio-route").display = key == "service"
        self.query_one("#studio-quick").display = "fixture" in BY_KEY[key].defaults
        recipe = json.loads(json.dumps(BY_KEY[key].defaults))
        try:
            recipe.update(json.loads(editor.text))
        except (ValueError, TypeError):
            pass
        if "fixture" in recipe:
            self.query_one("#quick-fixture", Select).value = recipe["fixture"]
            self.query_one("#quick-input", Input).value = recipe.get("input", "")
        for name in ("paths", "horizon", "seed"):
            control = self.query_one(f"#quick-{name}", Input)
            control.disabled = name not in recipe
            control.value = str(recipe.get(name, ""))

    @on(Select.Changed, "#studio-operation")
    def select_changed(self, event):
        if isinstance(event.value, str) and event.value in BY_KEY:
            self.select_experiment(event.value)

    @on(Input.Changed, "#studio-search")
    def search_changed(self, event):
        words = event.value.lower().split()
        matches = [
            x
            for x in EXPERIMENTS
            if all(
                word in f"{x.title} {x.group} {x.description}".lower() for word in words
            )
        ]
        selector = self.query_one("#studio-operation", Select)
        # Select requires a nonempty option list when allow_blank=False.
        if matches:
            selector.set_options([(f"{x.group} · {x.title}", x.key) for x in matches])
            selector.value = (
                self.active_key
                if any(x.key == self.active_key for x in matches)
                else matches[0].key
            )
        self.query_one("#studio-status", Static).update(
            f"{len(matches)} experiments match"
            if words
            else "Explore a recipe. Every result comes from the engine."
        )

    @on(Select.Changed, "#studio-route")
    def route_changed(self, event):
        if not isinstance(event.value, str):
            return
        _, method, route, body = SERVICE_ROUTES[int(event.value)]
        # A fetched snapshot can be edited and saved with its optimistic revision.
        if (
            method == "PUT"
            and self.result
            and self.result.get("operation") == "service"
        ):
            snapshot = self.result["result"]
            if isinstance(snapshot, dict) and "revision" in snapshot:
                fields = (
                    ("as_of", "accounts", "bills", "inputs")
                    if route == "/finance"
                    else ("as_of", "accounts", "bills")
                )
                body = {
                    "expected_revision": snapshot["revision"],
                    **{k: snapshot[k] for k in fields if k in snapshot},
                }
        self.query_one("#studio-params", TextArea).load_text(
            json.dumps(dict(method=method, route=route, body=body), indent=2)
        )
        if method == "PUT":
            self.query_one("#studio-status", Static).update(
                "PUT saves the complete snapshot to your account. Review the body and expected_revision before Run."
            )

    @on(Button.Pressed, "#quick-apply")
    def apply_quick(self):
        try:
            editor = self.query_one("#studio-params", TextArea)
            recipe = json.loads(editor.text)
            if not isinstance(recipe, dict):
                raise ValueError("Recipe must be a JSON object.")
            allowed = BY_KEY[self.active_key].defaults
            for name in ("paths", "horizon", "seed"):
                if name in allowed:
                    recipe[name] = int(self.query_one(f"#quick-{name}", Input).value)
            recipe["fixture"] = self.query_one("#quick-fixture", Select).value
            recipe["input"] = self.query_one("#quick-input", Input).value.strip()
            editor.load_text(json.dumps(recipe, indent=2))
            self.query_one("#studio-quick", Collapsible).collapsed = True
        except (ValueError, TypeError) as error:
            self.app.notify(str(error), severity="error")

    @on(Button.Pressed, "#studio-pin")
    def pin_baseline(self):
        if self.result:
            self.baseline = self.result
            self.query_one("#studio-status", Static).update(
                f"Pinned {BY_KEY[self.result['operation']].title}. Run or open another experiment to compare."
            )

    @on(Button.Pressed, "#studio-compare")
    def compare_baseline(self):
        if not self.result or not self.baseline:
            self.app.notify("Pin a baseline result first.")
            return
        if self.result["operation"] != self.baseline["operation"]:
            self.app.notify("Compare two results from the same experiment.")
            return

        def numbers(value, prefix=""):
            result = {}
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in ("experiment_identity", "manifest", "provenance"):
                        continue
                    path = f"{prefix}.{key}" if prefix else key
                    if isinstance(child, (int, float)) and not isinstance(child, bool):
                        result[path] = child
                    elif isinstance(child, dict):
                        result.update(numbers(child, path))
            return result

        before, after = numbers(self.baseline["result"]), numbers(self.result["result"])
        rows = [
            dict(
                metric=key,
                baseline=before[key],
                current=after[key],
                delta=after[key] - before[key],
            )
            for key in before.keys() & after.keys()
        ]
        if not rows:
            self.app.notify(
                "These results have no common scalar metrics; inspect their exported tables."
            )
            return
        self.tables["baseline comparison"] = (
            ["metric", "baseline", "current", "delta"],
            sorted(rows, key=lambda r: r["metric"]),
        )
        selector = self.query_one("#studio-table-choice", Select)
        selector.set_options([(key, key) for key in self.tables])
        selector.value = "baseline comparison"
        selector.display = True
        self.query_one("#studio-table").display = True
        self.show_table("baseline comparison")
        self.query_one("#studio-table").scroll_visible()
        self.query_one("#studio-status", Static).update(
            "Current minus baseline · descriptive deltas, not a significance test. Recipes are retained in the notebook."
        )

    @on(Button.Pressed, "#studio-reset")
    def reset_recipe(self):
        self.query_one("#studio-params", TextArea).load_text(
            json.dumps(BY_KEY[self.active_key].defaults, indent=2)
        )

    @on(Button.Pressed, "#studio-run")
    def launch(self):
        if self.busy:
            return
        try:
            params = json.loads(self.query_one("#studio-params", TextArea).text)
            if not isinstance(params, dict):
                raise ValueError("Recipe must be a JSON object.")
            json.dumps(params, allow_nan=False)
        except (ValueError, TypeError) as error:
            self.query_one("#studio-status", Static).update(
                f"Recipe needs attention: {error}"
            )
            self.app.notify(str(error), severity="error")
            return
        self.cancel_directory = tempfile.TemporaryDirectory(prefix="ginseng-studio-")
        self.cancel_path = Path(self.cancel_directory.name) / "cancel"
        self.cooperative = self.active_key == "precision" or (
            self.active_key == "personal-numerics"
            and params.get("action", "surface") == "precision"
        )
        self.busy = True
        self.query_one("#studio-run", Button).disabled = True
        self.query_one("#studio-cancel", Button).disabled = False
        self.query_one("#studio-status", Static).update(
            f"Computing {BY_KEY[self.active_key].title}… you can keep browsing or cancel."
        )
        self.job = self.studio_job(self.active_key, params)

    @work(exclusive=True, exit_on_error=False, group="studio")
    async def studio_job(self, key, params):
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "ginseng.studio",
                "--cancel-file",
                str(self.cancel_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
            stdout, stderr = await process.communicate(
                json.dumps({"key": key, "parameters": params}).encode()
            )
            if process.returncode:
                raise ValueError(
                    stderr.decode(errors="replace")[-3000:] or "Experiment failed"
                )
            record = json.loads(stdout)
            self.notebook.append(record)
            table = self.query_one("#studio-notebook-table", DataTable)
            table.add_row(
                str(len(self.notebook)),
                BY_KEY[key].title,
                str(
                    params.get("input")
                    or params.get("workspace")
                    or params.get("fixture", "—")
                ),
                f"{record['elapsed_seconds']:.2f}s",
                key=str(len(self.notebook) - 1),
            )
            self.present(record)
            payload = record["result"]
            summary = payload.get("summary") if isinstance(payload, dict) else None
            outcome = (
                summary.get("status", "complete")
                if isinstance(summary, dict)
                else "complete"
            )
            if key in ("precision-replay", "two-decision-replay"):
                outcome = "replay matched" if payload["match"] else "REPLAY MISMATCH"
            self.query_one("#studio-status", Static).update(
                f"{BY_KEY[key].title} · {outcome} · {record['elapsed_seconds']:.2f}s · added to notebook"
            )
        except asyncio.CancelledError:
            self.query_one("#studio-status", Static).update(
                "Experiment cancelled. Earlier notebook results are still available."
            )
            raise
        except Exception as error:
            self.query_one("#studio-status", Static).update(
                f"Experiment needs attention: {error}"
            )
            self.app.notify(str(error), severity="error", timeout=10)
        finally:
            if process is not None and process.returncode is None:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    await process.wait()
                except ProcessLookupError:
                    pass
            if self.cancel_directory is not None:
                self.cancel_directory.cleanup()
            self.cancel_directory = self.cancel_path = None
            self.busy = False
            if self.is_mounted:
                self.query_one("#studio-run", Button).disabled = False
                self.query_one("#studio-cancel", Button).disabled = True

    @on(Button.Pressed, "#studio-cancel")
    def cancel(self):
        if self.job and self.busy:
            if self.cooperative and self.cancel_path is not None:
                self.cancel_path.touch()
                self.query_one("#studio-cancel", Button).disabled = True
                self.query_one("#studio-status", Static).update(
                    "Cancellation requested · finishing the current batch; completed observations will stay in the notebook."
                )
            else:
                self.job.cancel()

    def present(self, record):
        self.result = record
        payload = record["result"]
        self.tables = result_tables(payload)
        self.query_one("#studio-json", TextArea).load_text(
            json.dumps(record, indent=2, allow_nan=False)
        )
        summary = payload.get("summary", payload) if isinstance(payload, dict) else {}
        scalars = [
            f"{key.replace('_', ' ')}: {value:.6g}"
            if isinstance(value, float)
            else f"{key.replace('_', ' ')}: {value}"
            for key, value in summary.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        ]
        if "numerical_probability_interval" in summary:
            value = summary["cash_shortfall_probability"]
            estimate = "Not estimated" if value is None else f"{value:.4%}"
            lo, hi = summary["numerical_probability_interval"]
            scalars = [
                f"Status: {summary.get('status', summary['stop_reason'])}",
                f"Cash failure: {estimate}",
                f"{summary['confidence']:.0%} numerical interval: [{lo:.4%}, {hi:.4%}]",
                f"Requested error: {summary['requested_absolute_error']:.4%}",
                f"Observations: {summary['actual_n']:,}; certified checkpoint: {summary.get('interval_observations', summary['actual_n']):,}",
                f"Stop reason: {summary['stop_reason']}",
                "Fixed simulator uncertainty; not forecast calibration.",
            ]
            if "capture" in payload:
                scalars.append(f"Capture: {payload['capture']}")
            elif record["parameters"].get("capture_path"):
                scalars.append(
                    f"Captured inputs: {record['parameters']['capture_path']}"
                )
        if record["operation"] in ("precision-replay", "two-decision-replay"):
            scalars.insert(
                0, "Replay: MATCH" if payload["match"] else "Replay: MISMATCH"
            )
            scalars.extend(f"Mismatch: {value}" for value in payload["mismatches"])
        self.query_one("#studio-summary", Static).update(
            f"{BY_KEY[record['operation']].title}\n" + "\n".join(scalars[:14])
        )
        self.query_one(RiskAtlas).show(payload)
        selector = self.query_one("#studio-table-choice", Select)
        selector.set_options(
            [
                (f"{key} · {len(rows)} rows", key)
                for key, (_, rows) in self.tables.items()
            ]
        )
        if self.tables:
            selector.value = next(iter(self.tables))
            self.show_table(str(selector.value))
        else:
            self.query_one("#studio-table", DataTable).clear(columns=True)
            self.query_one(TableVisual).display = False
        self.query_one("#studio-table").display = bool(self.tables)
        selector.display = bool(self.tables)
        self.query_one("#studio-tabs", TabbedContent).active = "studio-results"

    def show_table(self, key):
        if key not in self.tables:
            return
        columns, rows = self.tables[key]
        self.query_one(TableVisual).show(columns, rows)
        table = self.query_one("#studio-table", DataTable)
        table.clear(columns=True)
        table.add_columns(*(c.replace("_", " ") for c in columns))
        for row in rows[:5000]:
            table.add_row(
                *(
                    Text(
                        f"{row[c]:,.6g}"
                        if isinstance(row.get(c), float)
                        else str(row.get(c, "—"))
                    )
                    for c in columns
                )
            )
        if len(rows) > 5000:
            self.query_one("#studio-status", Static).update(
                "Table preview shows 5,000 rows; JSON and CSV exports include every row."
            )

    @on(Select.Changed, "#studio-table-choice")
    def table_changed(self, event):
        self.show_table(event.value)

    @on(DataTable.RowSelected, "#studio-notebook-table")
    def revisit(self, event):
        self.present(self.notebook[int(event.row_key.value)])

    @on(Button.Pressed, "#studio-export")
    @on(Button.Pressed, "#studio-csv")
    def export(self, event):
        if self.result is None:
            self.app.notify("Run an experiment first.")
            return
        suffix = "csv" if event.button.id == "studio-csv" else "json"
        if suffix == "csv":
            key = self.query_one("#studio-table-choice", Select).value
            if key not in self.tables:
                self.app.notify("Choose a result table first.")
                return
            columns, rows = self.tables[key]
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            data = stream.getvalue()
        else:
            data = json.dumps(self.result, indent=2, allow_nan=False) + "\n"
        try:
            NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            path = NOTEBOOK_DIR / f"{stamp}-{self.result['operation']}.{suffix}"
            with path.open("x") as handle:
                handle.write(data)
            self.query_one("#studio-status", Static).update(f"Saved {path}")
            self.app.notify(f"Saved {path}")
        except OSError as error:
            self.app.notify(str(error), severity="error")

    @on(Button.Pressed, "#studio-import")
    def import_record(self):
        try:
            record = json.loads(
                Path(self.query_one("#studio-import-path", Input).value).read_text()
            )
            if (
                record["operation"] not in BY_KEY
                or not isinstance(record["parameters"], dict)
                or not isinstance(record["result"], dict)
            ):
                raise ValueError("Not a studio experiment export.")
            self.open_experiment(record["operation"])
            self.query_one("#studio-params", TextArea).load_text(
                json.dumps(record["parameters"], indent=2)
            )
            self.present(record)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.app.notify(str(error), severity="error")
