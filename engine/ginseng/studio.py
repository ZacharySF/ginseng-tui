"""Offline research operations used by the terminal studio.

Adapters call the same public engines as the CLI and web application. Nothing
runs on import, and no credentials or service writes are implicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from urllib.parse import urlparse

import numpy as np


@dataclass(frozen=True)
class Experiment:
    key: str
    title: str
    group: str
    description: str
    defaults: dict


CASE = dict(
    fixture="canonical",
    input="",
    paths=2048,
    seed=42,
    horizon=30,
    block_length=None,
    sampler="mc",
    opening_cash=None,
    obligations=None,
)
EXPERIMENTS = (
    Experiment(
        "precision",
        "Precision stopping",
        "Numerics",
        "Stop IID Monte Carlo at a checkpoint-valid error target. The interval measures numerical error, not forecast accuracy.",
        {
            **{k: v for k, v in CASE.items() if k != "paths"},
            "estimator": "initial-block-cmc",
            "absolute_error": 0.005,
            "confidence": 0.95,
            "max_paths": 65536,
            "batch_size": 1024,
            "replicate": 0,
            "material_horizon": None,
            "chunk_size": 1024,
            "time_limit_seconds": 30.0,
            "memory_budget_bytes": 134217728,
            "stop_when_precise": True,
            "backend": "numpy",
            "workers": 1,
            "capture_path": "",
            "allow_personal_capture": False,
        },
    ),
    Experiment(
        "precision-replay",
        "Replay precision evidence",
        "Engineering",
        "Recompute captured failure observations and checkpoint intervals offline. Mismatches remain explicit.",
        dict(
            input="examples/precision-b/path",
            backend="numpy",
            workers=1,
            memory_budget_bytes=134217728,
        ),
    ),
    Experiment(
        "two-decision",
        "Two-decision funding experiment",
        "Planning",
        "Synthetic only: static vs observation-based review vs infeasible hindsight. Frozen independent validation; never a production recommendation.",
        dict(
            training_paths=128,
            validation_paths=4096,
            replications=5,
            seed=20260920,
            settlement_days=2,
            sale_fee=3.0,
            liquidity_charge=1.0,
            capture_path="",
        ),
    ),
    Experiment(
        "two-decision-replay",
        "Replay two-decision experiment",
        "Engineering",
        "Replay exact synthetic futures, frozen policies and finite-grid training objectives offline.",
        dict(input="examples/two-decision/canonical"),
    ),
    Experiment(
        "surface",
        "Cash-buffer atlas",
        "Numerics",
        "Sweep extra opening cash and time over the same paths. Compare ever-negative probability and expected worst deficit.",
        {**CASE, "cash_offsets": [0, 250, 500, 1000, 2000, 5000]},
    ),
    Experiment(
        "tails",
        "Tail-risk microscope",
        "Numerics",
        "Cash-deficit VaR / CVaR, terminal quantiles, first-passage counts and path drawdowns. These are dollar cash-flow risks, not investment returns.",
        {**CASE, "quantiles": [0.9, 0.95, 0.99]},
    ),
    Experiment(
        "compare",
        "Sampler tournament",
        "Numerics",
        "Independent replicate comparisons of MC and scrambled Sobol. Tiny cases include exact-oracle errors; larger cases report replicate dispersion, not accuracy.",
        {
            **{k: v for k, v in CASE.items() if k not in ("paths", "sampler")},
            "fixture": "tiny",
            "horizon": 4,
            "block_length": 7,
            "path_counts": [256, 1024, 4096],
            "replicates": 4,
            "estimator": "path",
        },
    ),
    Experiment(
        "scenario",
        "Funding & policy atelier",
        "Planning",
        "Evaluate funding candidates, CVaR optimization, account liquidity, policy ranking and wrong-way risk on one shared draw bundle.",
        {
            **CASE,
            "coverage_target": 0.95,
            "operating_buffer": 1000,
            "funding_config": {},
            "funding_policy": {},
            "overdraft_apr": 0.2999,
            "capital_gains_rate": 0.15,
            "buffer_tolerance_dollar_days": None,
        },
    ),
    Experiment(
        "stress",
        "Drought stress garden",
        "Planning",
        "Entropy-pool a drought assumption over unchanged paths; inspect effective tail support alongside baseline and stressed risk.",
        {**CASE, "drought_probability": 0.5, "window_days": 14, "income_fraction": 0.5},
    ),
    Experiment(
        "funding",
        "Funding decision lab",
        "Planning",
        "Cost / tail-risk frontier, shadow checks and independently simulated frozen-plan holdout.",
        {
            **CASE,
            "coverage_target": 0.95,
            "operating_buffer": 1000,
            "capital_gains_rate": 0.15,
            "overdraft_apr": 0.2999,
            "tail_deficit_limit": None,
            "buffer_tolerance_dollar_days": None,
        },
    ),
    Experiment(
        "portfolio",
        "Tax-lot laboratory",
        "Portfolio",
        "Sell-only liquidation, joint asset returns and shrunk covariance using the existing portfolio engine.",
        {**CASE, "target": 1000, "long_rate": 0.15, "short_rate": 0.24},
    ),
    Experiment(
        "frontier",
        "Liquidity-aware frontier",
        "Portfolio",
        "Allocation candidates, Pareto efficiency and independently simulated validation. Synthetic fixtures remain explicitly synthetic.",
        dict(
            fixture="canonical",
            input="",
            seed=42,
            block_length=None,
            horizon=30,
            operating_buffer=1000,
            opening_cash=None,
            obligations=[
                {
                    "id": "repair-deposit",
                    "label": "Repair deposit",
                    "amount": 1500,
                    "due_in_days": 3,
                },
                {
                    "id": "repair-balance",
                    "label": "Repair balance",
                    "amount": 3000,
                    "due_in_days": 17,
                },
            ],
        ),
    ),
    Experiment(
        "calibration",
        "Walk-forward observatory",
        "Validation",
        "Historical reserve calibration, coverage, PIT, CRPS, pinball loss and formal tests. Short histories may provide no usable windows.",
        {
            **{k: v for k, v in CASE.items() if k not in ("sampler", "block_length")},
            "training_days": 180,
            "max_windows": 12,
        },
    ),
    Experiment(
        "persistence",
        "Persistence sensitivity",
        "Validation",
        "Compare block-length assumptions with the engine's stability verdict.",
        {k: v for k, v in CASE.items() if k not in ("sampler", "block_length")},
    ),
    Experiment(
        "uncertainty",
        "Reserve uncertainty",
        "Validation",
        "Outer-bootstrap reserve estimate band, separate from Monte Carlo precision.",
        {**CASE, "outer_paths": 20},
    ),
    Experiment(
        "accounts",
        "Account liquidity ledger",
        "Planning",
        "Account-specific spendable capital, tax and penalty reserves, withdrawal restrictions and assumptions.",
        dict(fixture="canonical", input="", long_term_rate=0.15),
    ),
    Experiment(
        "forecast",
        "Personal forecast",
        "Personal",
        "Load a FinanceWorkspace JSON export. Its mode selects scheduled, historical or assumptions-based forecasts; no server is needed.",
        dict(
            workspace="local/finance-workspace.json",
            horizon=30,
            paths=2048,
            seed=42,
            overrides={},
        ),
    ),
    Experiment(
        "backtest",
        "Personal history backtest",
        "Personal",
        "Backtest a local FinanceWorkspace with the same history requirements as the application.",
        dict(workspace="local/finance-workspace.json", horizon=30, paths=2048, seed=42),
    ),
    Experiment(
        "personal-numerics",
        "Personal numerical explorer",
        "Personal",
        "Precision or cash/time surface for the historical personal workspace.",
        dict(
            workspace="local/finance-workspace.json",
            horizon=30,
            seed=42,
            action="surface",
            absolute_error=0.005,
            confidence=0.95,
            estimator="path",
            time_limit_seconds=15.0,
            memory_budget_bytes=134217728,
            max_paths=65536,
        ),
    ),
    Experiment(
        "benchmark",
        "Numerical benchmark",
        "Engineering",
        "Run a benchmark configuration and write its observations and manifest to a new output directory.",
        dict(config="benchmarks/standard.json", output="artifacts/tui/benchmark"),
    ),
    Experiment(
        "report",
        "Research report",
        "Engineering",
        "Render an existing benchmark into tables and plots without resampling.",
        dict(input="artifacts/standard", output="artifacts/tui/report"),
    ),
    Experiment(
        "inspect",
        "Engine inspection",
        "Engineering",
        "Inspect NumPy/native availability and execution boundaries.",
        {},
    ),
    Experiment(
        "capture",
        "Capture prepared engine",
        "Engineering",
        "Capture a reproducible canonical scenario for offline replay.",
        dict(
            output="artifacts/tui/capture",
            paths=2048,
            horizon=30,
            backend="numpy",
            workers=1,
            block_size=1024,
            memory_budget=536870912,
            summary_only=False,
        ),
    ),
    Experiment(
        "replay",
        "Replay artifact",
        "Engineering",
        "Replay captured arrays with explicit backend, worker and memory settings.",
        dict(
            input="examples/engine/canonical",
            output=None,
            backend="numpy",
            workers=1,
            block_size=1024,
            memory_budget=536870912,
        ),
    ),
    Experiment(
        "diff",
        "Diff engine artifacts",
        "Engineering",
        "Compare identities, arrays and numerical results from two captured runs.",
        dict(first="examples/engine/canonical", second="artifacts/tui/capture"),
    ),
    Experiment(
        "engine-benchmark",
        "Engine performance suite",
        "Engineering",
        "Measure the prepared execution engine with the existing quant-engineering benchmark.",
        dict(output="artifacts/tui/engine-benchmark", smoke=True),
    ),
    Experiment(
        "service",
        "Connected application",
        "Connected",
        "Access every application route: workspace/finance read and save, forecasts, backtests, scenario analyses, provider data and chat. Set GINSENG_API_URL and GINSENG_API_TOKEN; requests only run when launched. Use GET /openapi.json to inspect contracts. PUT writes to the connected account.",
        dict(method="GET", route="/health", body={}),
    ),
)
BY_KEY = {entry.key: entry for entry in EXPERIMENTS}

# Every public application route has an explicit entry; no account calls at startup.
SERVICE_ROUTES = (
    ("Health", "GET", "/health", {}),
    ("API contracts", "GET", "/openapi.json", {}),
    ("Cash workspace", "GET", "/workspace", {}),
    ("Finance workspace", "GET", "/finance", {}),
    ("Save cash workspace (PUT)", "PUT", "/workspace", {"expected_revision": 0}),
    ("Save finance workspace (PUT)", "PUT", "/finance", {"expected_revision": 0}),
    ("Scenario + optimizer", "POST", "/scenario", {}),
    ("Calibration", "POST", "/analysis/calibration", {}),
    ("Funding decisions", "POST", "/analysis/funding", {}),
    ("Portfolio laboratory", "POST", "/analysis/portfolio", {}),
    ("Portfolio frontier", "POST", "/analysis/frontier", {}),
    (
        "Demo numerical explorer",
        "POST",
        "/demo/numerics",
        {"options": {"action": "surface"}},
    ),
    (
        "Personal forecast",
        "POST",
        "/finance/forecast",
        {"expected_revision": 0, "horizon_days": 30},
    ),
    (
        "Personal backtest",
        "POST",
        "/finance/backtest",
        {"expected_revision": 0, "horizon_days": 30},
    ),
    (
        "Personal numerical explorer",
        "POST",
        "/finance/numerics",
        {"expected_revision": 0, "horizon_days": 30, "options": {"action": "surface"}},
    ),
    ("Provider status", "GET", "/providers/nessie/status", {}),
    ("Provider sample", "GET", "/providers/nessie/sample", {}),
    (
        "Personal assistant",
        "POST",
        "/chat",
        {
            "source": "personal",
            "horizon_days": 30,
            "message": "Explain my current cash forecast.",
            "history": [],
        },
    ),
)


def jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, Path)):
        return str(value)
    return value


def _case(p):
    from ginseng.inputs import fixture, load_input

    case = (
        load_input(Path(p["input"]))
        if p.get("input")
        else fixture(p.get("fixture", "canonical"))
    )
    updates = {
        dest: p[src]
        for src, dest in (
            ("horizon", "forecast_horizon"),
            ("coverage_target", "coverage_target"),
            ("operating_buffer", "operating_buffer"),
        )
        if p.get(src) is not None
    }
    state = replace(case.state, **updates)
    if p.get("opening_cash") is not None:
        from ginseng.state import Transaction, TransactionType

        amount = p["opening_cash"]
        if (
            isinstance(amount, bool)
            or not isinstance(amount, (int, float))
            or not np.isfinite(amount)
        ):
            raise ValueError("Opening cash must be a finite dollar amount.")
        adjustment = Transaction(
            state.as_of,
            TransactionType.TRANSFER,
            amount - state.immediate_funding,
            "Studio opening cash scenario",
        )
        state = replace(state, transactions=(*state.transactions, adjustment))
    obligations = case.obligations
    if p.get("obligations") is not None:
        from ginseng.state import Obligation

        if not isinstance(p["obligations"], list) or len(p["obligations"]) > 200:
            raise ValueError("Supply at most 200 obligations.")
        obligations = []
        seen = set()
        for row in p["obligations"]:
            amount, day, identity = row["amount"], row["due_in_days"], row["id"]
            if (
                not isinstance(amount, (int, float))
                or isinstance(amount, bool)
                or not np.isfinite(amount)
                or amount < 0
                or type(day) is not int
                or day < 1
                or identity in seen
            ):
                raise ValueError(
                    "Bills need unique IDs, nonnegative dollar amounts and positive integer due days."
                )
            seen.add(identity)
            obligations.append(
                Obligation(identity, row.get("label", "Scenario bill"), amount, day)
            )
        obligations = tuple(obligations)
    if not 0 < state.coverage_target <= 1 or state.operating_buffer < 0:
        raise ValueError(
            "Coverage must be in (0, 1] and the operating buffer nonnegative."
        )
    return replace(case, state=state, obligations=obligations)


def _sample(p, *, funding=False):
    from ginseng.sampling import prepare_history, sample_bundle
    from ginseng.simulate import cash_paths

    case = _case(p)
    history = prepare_history(
        case.state, p.get("block_length") or (7 if case.name == "tiny" else None)
    )
    material = case.state.forecast_horizon
    if funding:
        from types import SimpleNamespace
        from ginseng.funding import (
            FundingConfig,
            PlanSpec,
            PlanKind,
            plan_evaluation_horizon,
        )

        config = FundingConfig(**p.get("funding_config", {}))
        # Predeclare every possible credit settlement date before drawing paths.
        specs = [
            PlanSpec(
                "bounds",
                "Funding timing",
                PlanKind.HYBRID,
                credit_account_id=account.account_id if account else None,
                credit_draw=1,
                liquidation_target=1,
                settlement_days=config.settlement_days,
                external_transfer_days=config.external_transfer_days,
                trailing_days=config.trailing_days,
                use_business_days=config.use_business_days,
            )
            for account in (list(case.state.credit_accounts) or [None])
        ]
        material = max(
            plan_evaluation_horizon(
                case.state, SimpleNamespace(horizon_days=material), spec
            )
            for spec in specs
        )
        if (
            p.get("sampler", "mc") == "legacy_mc"
            and material > case.state.forecast_horizon
        ):
            raise ValueError("Funding extensions require mc or sobol in the studio.")
    if p.get("paths", 2048) * material > 10_000_000:
        raise ValueError("Funding material plan exceeds 10 million path-days.")
    bundle = sample_bundle(
        history,
        case.state.forecast_horizon,
        p.get("paths", 2048),
        p.get("seed", 42),
        p.get("sampler", "mc"),
        material_horizon=material,
    )
    matrix = cash_paths(
        case.state, bundle, case.obligations, prepared_history=history.joint
    )
    return case, history, bundle, matrix


def tail_metrics(matrix, opening, levels):
    from ginseng.risk import cvar, quantile

    balances = opening + np.asarray(matrix, dtype=float)
    losses = np.maximum(0, -balances.min(axis=1))
    # Opening cash is the initial high-water mark, not a sampled future day.
    peaks = np.maximum.accumulate(
        np.column_stack((np.full(len(balances), opening), balances)), axis=1
    )[:, 1:]
    drawdowns = np.max(peaks - balances, axis=1)
    crossed = balances < 0
    first = np.where(crossed.any(axis=1), crossed.argmax(axis=1) + 1, 0)
    return dict(
        tail_risk=[
            dict(
                confidence=q,
                deficit_var=quantile(losses, q),
                deficit_cvar=cvar(losses, q),
                drawdown_var=quantile(drawdowns, q),
                drawdown_cvar=cvar(drawdowns, q),
            )
            for q in levels
        ],
        first_passage=[
            dict(
                day=i,
                paths=int(np.sum(first == i)),
                probability=float(np.mean(first == i)),
            )
            for i in range(1, balances.shape[1] + 1)
        ],
        no_shortfall_probability=float(np.mean(first == 0)),
        terminal_cash={
            str(q): quantile(balances[:, -1], q) for q in (0.01, 0.05, 0.5, 0.95, 0.99)
        },
        expected_max_cash_deficit=float(losses.mean()),
        expected_max_drawdown=float(drawdowns.mean()),
        scope="Empirical dollar cash-flow loss. Drawdown includes opening cash; no price-return or Sharpe-ratio interpretation.",
    )


def _cli(args):
    # An isolated process keeps CLI stdout/warnings away from Textual's terminal.
    result = subprocess.run(
        [sys.executable, "-m", "ginseng", *args],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if result.returncode not in (0, 3, 4):
        raise ValueError(
            result.stderr.strip() or result.stdout.strip() or "Command failed"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "output": result.stdout,
            "diagnostics": result.stderr,
            "exit_code": result.returncode,
        }


def run_experiment(key: str, p: dict, *, cancelled=None) -> dict:
    """Execute one explicit request and attach a reproducible input record."""
    if key not in BY_KEY:
        raise ValueError(f"Unknown experiment: {key}")
    if not isinstance(p, dict):
        raise ValueError("Parameters must be a JSON object.")
    unknown = p.keys() - BY_KEY[key].defaults.keys()
    if unknown:
        raise ValueError(f"Unknown parameters: {', '.join(sorted(unknown))}")
    p = {**BY_KEY[key].defaults, **p}
    # Validate before allocating matrices, including JSON's otherwise accepted NaN.
    json.dumps(p, allow_nan=False)
    for name in (
        "paths",
        "horizon",
        "replicates",
        "training_days",
        "max_windows",
        "outer_paths",
    ):
        if name in p and (type(p[name]) is not int or p[name] < 1):
            raise ValueError(f"{name} must be a positive integer.")
    if "horizon" in p and p["horizon"] > 3650:
        raise ValueError("Studio horizons are limited to 3,650 days.")
    if p.get("paths", 0) * p.get("horizon", 30) > 10_000_000:
        raise ValueError(
            "Studio limit: 10 million path-days per run; reduce paths or horizon."
        )
    started = perf_counter()
    output = jsonable(_execute(key, p, cancelled=cancelled))
    # Service payloads may contain private account data: only explicit exports persist them.
    from ginseng.numerical import environment

    return dict(
        operation=key,
        parameters=p,
        environment=environment(),
        elapsed_seconds=perf_counter() - started,
        created_at=datetime.now(timezone.utc).isoformat(),
        result=output,
    )


def _execute(key, p, *, cancelled=None):
    if key == "two-decision":
        from ginseng.two_decision import Model, run_experiment as run_two_decision
        from ginseng.two_decision_artifact import capture

        report, payload = run_two_decision(
            model=replace(
                Model(),
                settlement_days=p["settlement_days"],
                sale_fee=p["sale_fee"],
                liquidity_charge=p["liquidity_charge"],
            ),
            training_paths=p["training_paths"],
            validation_paths=p["validation_paths"],
            root_seed=p["seed"],
            replications=p["replications"],
            cancelled=cancelled,
        )
        if p["capture_path"]:
            report["capture"] = capture(p["capture_path"], payload)
        return report
    if key == "two-decision-replay":
        from ginseng.two_decision_artifact import replay

        return replay(p["input"])
    if key == "service":
        import httpx

        base = os.environ.get("GINSENG_API_URL", "http://127.0.0.1:8000").rstrip("/")
        parsed = urlparse(base)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("GINSENG_API_URL must be an HTTP(S) base URL.")
        if parsed.scheme == "http" and parsed.hostname not in (
            "localhost",
            "127.0.0.1",
            "::1",
        ):
            raise ValueError("Use HTTPS for a remote service connection.")
        route = p["route"]
        if (
            not route.startswith("/")
            or route.startswith("//")
            or urlparse(route).netloc
        ):
            raise ValueError(
                "Route must be a relative API path beginning with one slash."
            )
        if p["method"] not in ("GET", "POST", "PUT"):
            raise ValueError("Choose GET, POST or PUT.")
        token = os.environ.get("GINSENG_API_TOKEN", "")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        with httpx.Client(timeout=180, follow_redirects=False) as client:
            response = client.request(
                p["method"],
                base + route,
                headers=headers,
                **({"json": p["body"]} if p["method"] != "GET" else {}),
            )
            response.raise_for_status()
            if "application/x-ndjson" in response.headers.get("content-type", ""):
                events = [
                    json.loads(line)
                    for line in response.text.splitlines()
                    if line.strip()
                ]
                return {
                    "reply": "".join(
                        event.get("text", "")
                        for event in events
                        if event.get("type") == "delta"
                    ),
                    "events": events,
                }
            if "text/event-stream" in response.headers.get("content-type", ""):
                return {"events": response.text}
            return response.json()
    if key in ("benchmark", "report"):
        if Path(p["output"]).exists():
            raise ValueError("Output already exists. Choose a new directory.")
        return _cli(
            [
                key,
                "--config" if key == "benchmark" else "--input",
                p.get("config", p.get("input")),
                "--out",
                p["output"],
            ]
        )
    if key == "engine-benchmark":
        raise ValueError("The native performance suite requires the full Ginseng repository: https://github.com/nunera/ginseng. Local capture, replay and diff are available here.")
    if key in ("inspect", "capture", "replay", "diff", "engine-benchmark"):
        command = "benchmark" if key == "engine-benchmark" else key
        args = ["engine", command]
        if key == "diff":
            args += [p["first"], p["second"]]
        elif key == "replay":
            args += [p["input"]]
        elif key == "engine-benchmark":
            args += ["--suite", "quant-engineering"]
        for name in (
            "output",
            "paths",
            "horizon",
            "backend",
            "workers",
            "block_size",
            "memory_budget",
        ):
            if name in p and p[name] is not None:
                args += ["--" + name.replace("_", "-"), str(p[name])]
        for name in ("summary_only", "smoke"):
            if p.get(name):
                args.append("--" + name.replace("_", "-"))
        if p.get("output") and Path(p["output"]).exists():
            raise ValueError(
                "Output already exists. Choose a new directory to preserve earlier research."
            )
        return _cli(args)
    if key in ("forecast", "backtest", "personal-numerics"):
        from ginseng.finance_models import FinanceWorkspace, ScenarioOverrides
        from ginseng.personal_forecast import (
            evaluate_personal_forecast,
            backtest_personal_history,
            historical_numerical_case,
            apply_scenario_overrides,
        )

        workspace = FinanceWorkspace.model_validate_json(
            Path(p["workspace"]).read_text()
        )
        if key == "forecast":
            if p["overrides"]:
                workspace = apply_scenario_overrides(
                    workspace, ScenarioOverrides.model_validate(p["overrides"])
                )
            return evaluate_personal_forecast(
                workspace, p["horizon"], p["seed"], p["paths"]
            )
        if key == "backtest":
            return backtest_personal_history(
                workspace, p["horizon"], seed=p["seed"], paths=p["paths"]
            )
        from ginseng.risk_explorer import explore, NumericalOptions

        return explore(
            historical_numerical_case(workspace, p["horizon"]),
            NumericalOptions(
                action=p["action"],
                absolute_error=p["absolute_error"],
                max_paths=p["max_paths"],
                confidence=p["confidence"],
                estimator=p["estimator"],
                time_limit_seconds=p["time_limit_seconds"],
                memory_budget_bytes=p["memory_budget_bytes"],
            ),
            cancelled=cancelled,
            seed=p["seed"],
        )
    if key == "precision-replay":
        from ginseng.precision_artifact import replay_precision
        from ginseng.execution import ExecutionConfig

        return replay_precision(
            p["input"],
            ExecutionConfig(
                p["backend"], p["workers"], memory_budget=p["memory_budget_bytes"]
            ),
        )
    if key == "precision":
        from ginseng.precision import PrecisionConfig, run_precision
        from ginseng.execution import ExecutionConfig

        if p["sampler"] != "mc":
            raise ValueError("Precision stopping requires independent MC.")
        if type(p["allow_personal_capture"]) is not bool:
            raise ValueError("allow_personal_capture must be boolean")
        case = _case(p)
        material = p["material_horizon"] or case.state.forecast_horizon
        if (
            min(p["chunk_size"] or p["batch_size"], p["max_paths"]) * material
            > 10_000_000
        ):
            raise ValueError("Precision batch exceeds ten million path-days.")
        return run_precision(
            case,
            PrecisionConfig(
                p["absolute_error"],
                p["confidence"],
                p["max_paths"],
                p["batch_size"],
                chunk_size=p["chunk_size"],
                time_limit_seconds=p["time_limit_seconds"],
                memory_budget_bytes=p["memory_budget_bytes"],
                stop_when_precise=p["stop_when_precise"],
            ),
            estimator=p["estimator"],
            cancelled=cancelled,
            execution=ExecutionConfig(
                p["backend"], p["workers"], memory_budget=p["memory_budget_bytes"]
            ),
            capture_path=p["capture_path"] or None,
            allow_personal_capture=p["allow_personal_capture"],
            synthetic=not bool(p["input"]),
            seed=p["seed"],
            replicate=p["replicate"],
            material_horizon=p["material_horizon"],
            block_length=p["block_length"] or (7 if case.name == "tiny" else None),
        )
    if key == "compare":
        from ginseng.numerical import run_core
        from ginseng.sampling import prepare_history
        from ginseng.exact import enumerate_exact

        case = _case(p)
        prepared = prepare_history(
            case.state, p["block_length"] or (7 if case.name == "tiny" else None)
        )
        if not p["path_counts"] or len(p["path_counts"]) > 12 or p["replicates"] > 32:
            raise ValueError("Choose 1–12 path counts and at most 32 replicates.")
        if any(
            type(n) is not int or n < 2 or n & (n - 1) or n * p["horizon"] > 10_000_000
            for n in p["path_counts"]
        ):
            raise ValueError(
                "Path counts must be powers of two within 10 million path-days."
            )
        oracle = (
            enumerate_exact()["summary"]
            if case.name == "tiny"
            and p["horizon"] == 4
            and prepared.resolved_length == 7
            and p.get("opening_cash") is None
            and p.get("obligations") is None
            else None
        )
        rows = []
        for n in p["path_counts"]:
            for sampler in ("mc", "sobol"):
                samples, times = [], []
                for rep in range(p["replicates"]):
                    start = perf_counter()
                    _, _, summary = run_core(
                        case,
                        prepared,
                        sampler,
                        n,
                        p["seed"],
                        p["horizon"],
                        None,
                        rep,
                        estimator=p["estimator"],
                    )
                    samples.append(summary)
                    times.append(perf_counter() - start)
                for metric in (
                    "cash_shortfall_probability",
                    "required_liquidity_reserve",
                    "expected_max_cash_deficit",
                ):
                    values = np.array([s[metric] for s in samples])
                    rows.append(
                        dict(
                            paths=n,
                            sampler=sampler,
                            metric=metric,
                            mean=float(values.mean()),
                            replicate_sd=float(values.std(ddof=1))
                            if len(values) > 1
                            else None,
                            rmse=float(np.sqrt(np.mean((values - oracle[metric]) ** 2)))
                            if oracle
                            else None,
                            mean_seconds=float(np.mean(times)),
                        )
                    )
        return dict(
            comparison=rows,
            exact_reference=oracle,
            scope="Replicate dispersion is not error against truth. RMSE is available only for the matching tiny oracle.",
        )
    case = _case(p)
    if key == "frontier":
        from ginseng.frontier import portfolio_frontier

        return portfolio_frontier(
            case.state, case.obligations, p["seed"], p["block_length"]
        )
    if key == "accounts":
        from ginseng.withdrawals import account_liquidity, WithdrawalAssumptions

        return account_liquidity(
            case.state, WithdrawalAssumptions(long_term_rate=p["long_term_rate"])
        )
    if key == "calibration":
        from ginseng.calibration import walk_forward

        return walk_forward(
            case.state,
            p["horizon"],
            p["paths"],
            p["seed"],
            case.state.coverage_target,
            case.state.operating_buffer,
            training_days=p["training_days"],
            max_windows=p["max_windows"],
            source=case.name,
        )
    if key == "persistence":
        from ginseng.uncertainty import persistence_sensitivity, stability_verdict

        rows = persistence_sensitivity(
            case.state,
            case.obligations,
            coverage_target=case.state.coverage_target,
            operating_buffer=case.state.operating_buffer,
            horizon_days=p["horizon"],
            n_paths=p["paths"],
            seed=p["seed"],
        )
        return dict(
            sensitivity=[vars(row) for row in rows], verdict=stability_verdict(rows)
        )
    case, history, bundle, matrix = _sample(p, funding=key in ("scenario", "funding"))
    from ginseng.provenance import digest

    identity = dict(
        input_hash=digest(asdict(case)),
        input=case.name,
        input_source="local" if p.get("input") else "synthetic fixture",
        seed=bundle.seed,
        paths=bundle.n_paths,
        draw_id=bundle.bootstrap_draw_id,
        block_length=history.resolved_length,
    )
    if key == "tails":
        if not p["quantiles"] or any(not 0 < q < 1 for q in p["quantiles"]):
            raise ValueError(
                "Tail confidence levels must be strictly between zero and one."
            )
        result = tail_metrics(matrix, case.state.immediate_funding, p["quantiles"])
    elif key == "surface":
        from ginseng.risk_explorer import surface_from_paths

        odds, deficits = surface_from_paths(
            matrix, case.state.immediate_funding, p["cash_offsets"]
        )
        result = dict(
            surface=[
                dict(
                    extra_cash=cash,
                    day=day + 1,
                    shortfall_probability=odds[i][day],
                    expected_max_deficit=deficits[i][day],
                )
                for i, cash in enumerate(p["cash_offsets"])
                for day in range(matrix.shape[1])
            ],
            opening_cash=case.state.immediate_funding,
            scope="Shared paths, extra cash from day one, no funding costs or simultaneous precision guarantee.",
        )
    elif key == "uncertainty":
        from ginseng.uncertainty import estimate_band
        from ginseng.metrics import compute_scenario_metrics

        metrics = compute_scenario_metrics(
            case.state,
            bundle,
            case.obligations,
            case.state.coverage_target,
            case.state.operating_buffer,
        )
        result = dict(
            point_estimate=metrics.required_liquidity_reserve,
            band=jsonable(
                estimate_band(
                    case.state,
                    case.obligations,
                    coverage_target=case.state.coverage_target,
                    operating_buffer=case.state.operating_buffer,
                    point_estimate=metrics.required_liquidity_reserve,
                    point_mean_block_length=bundle.mean_block_length,
                    horizon_days=bundle.horizon_days,
                    n_paths=bundle.n_paths,
                    n_outer=p["outer_paths"],
                    seed=p["seed"],
                )
            ),
        )
    elif key == "scenario":
        from ginseng.scenario_service import evaluate_scenario
        from ginseng.funding import FundingConfig
        from ginseng.policy import FundingPolicy

        result = evaluate_scenario(
            case.state,
            bundle,
            case.obligations,
            coverage_target=p["coverage_target"],
            operating_buffer=p["operating_buffer"],
            funding_config=FundingConfig(**p["funding_config"]),
            funding_policy=FundingPolicy(**p["funding_policy"]),
            overdraft_apr=p["overdraft_apr"],
            capital_gains_rate=p["capital_gains_rate"],
            buffer_tolerance_dollar_days=p["buffer_tolerance_dollar_days"],
        ).response.model_dump(mode="json")
    elif key == "portfolio":
        from ginseng.portfolio import portfolio_lab

        result = portfolio_lab(
            case.state,
            bundle,
            case.obligations,
            p["target"],
            long_rate=p["long_rate"],
            short_rate=p["short_rate"],
        )
    elif key == "stress":
        from ginseng.stress import DroughtView, scenario_weights
        from ginseng.metrics import (
            compute_scenario_metrics,
            required_liquidity_per_path,
        )
        from ginseng.risk import concentration

        if (
            type(p["window_days"]) is not int
            or p["window_days"] < 1
            or not 0 <= p["income_fraction"] <= 1
        ):
            raise ValueError(
                "Use a positive drought window and an income fraction in [0, 1]."
            )
        weights, metadata = scenario_weights(
            case.state,
            bundle,
            DroughtView(
                p["drought_probability"], p["window_days"], p["income_fraction"]
            ),
        )
        metadata.update(
            concentration(
                required_liquidity_per_path(matrix, case.state.operating_buffer),
                case.state.coverage_target,
                weights,
            )
        )
        base = compute_scenario_metrics(
            case.state,
            bundle,
            case.obligations,
            case.state.coverage_target,
            case.state.operating_buffer,
        )
        stressed = compute_scenario_metrics(
            case.state,
            bundle,
            case.obligations,
            case.state.coverage_target,
            case.state.operating_buffer,
            weights,
        )
        result = dict(
            stress=metadata,
            baseline=asdict(base),
            stressed=asdict(stressed) if metadata["status"] != "unsupported" else None,
        )
    elif key == "funding":
        from ginseng.decision import funding_analysis
        from ginseng.funding import (
            FundingConfig,
            build_candidates,
            optimizer_comparison_bundle,
        )
        from ginseng.metrics import compute_scenario_metrics
        from ginseng.policy import FundingPolicy
        from ginseng.withdrawals import WithdrawalAssumptions
        from ginseng.risk import probabilities

        policy = FundingPolicy(
            max_cash_shortfall_probability=1 - p["coverage_target"],
            max_buffer_breach_probability=1 - p["coverage_target"],
        )
        config = FundingConfig(max_credit_utilization=policy.max_credit_utilization)
        metrics = compute_scenario_metrics(
            case.state,
            bundle,
            case.obligations,
            p["coverage_target"],
            p["operating_buffer"],
        )
        specs = build_candidates(
            case.state,
            case.obligations,
            metrics.funding_gap,
            config,
            WithdrawalAssumptions(long_term_rate=p["capital_gains_rate"]),
        )
        comparison = optimizer_comparison_bundle(case.state, bundle, specs, config)
        parameters = {
            name: p[name]
            for name in (
                "coverage_target",
                "operating_buffer",
                "capital_gains_rate",
                "overdraft_apr",
                "tail_deficit_limit",
                "buffer_tolerance_dollar_days",
            )
        }
        result = funding_analysis(
            case.state,
            comparison,
            case.obligations,
            specs,
            probabilities(bundle.n_paths),
            None,
            {
                **parameters,
                "funding_config": config,
                "funding_policy": policy,
                "decision_horizon_days": bundle.horizon_days,
            },
        )
    else:
        raise ValueError(f"No adapter for {key}")
    return {**jsonable(result), "experiment_identity": identity}


def result_tables(value, prefix=""):
    """Discover genuine record arrays for table inspection; retain full raw JSON."""
    tables = {}
    if isinstance(value, dict):
        for key, child in value.items():
            tables.update(result_tables(child, f"{prefix}.{key}" if prefix else key))
    elif (
        isinstance(value, list)
        and value
        and all(isinstance(row, dict) for row in value)
    ):
        columns = list(
            dict.fromkeys(
                k
                for row in value
                for k, v in row.items()
                if v is None or isinstance(v, (str, int, float, bool))
            )
        )
        if columns:
            tables[prefix] = (columns, value)
    return tables


def main():
    import contextlib
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cancel-file", type=Path)
    args = parser.parse_args()

    try:
        request = json.load(sys.stdin)
        with contextlib.redirect_stdout(sys.stderr):
            result = run_experiment(
                request["key"],
                request["parameters"],
                cancelled=args.cancel_file.exists if args.cancel_file else None,
            )
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
