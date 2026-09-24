"""Maps one real engine run onto `DashboardData`. All engine-to-UI mapping lives here.

The engine and the dashboard disagree on one thing: `required_liquidity_reserve`
is a *total* reserve target (covering `state.operating_buffer`, at whatever
`coverage_target` the case configured), not the increment the dashboard's
`reserve_to_add` describes. `summary["funding_gap"]` is already that
increment (`max(0, RLR - opening_cash)`), computed by the engine itself, so
that is what `reserve_to_add` uses -- no new arithmetic, just the engine's
own conversion. Everything else `DashboardData` needs that the engine has no
field for (`cvar95_trough`, `deficit_dollar_days`, `solvent_days`, `bands`)
is a fresh reduction of the engine's own path matrix, built from
`ginseng.risk`'s own quantile/cvar primitives at the fixed $0 threshold and
q=0.95 the dashboard's docstrings specify, since no engine figure exists for
them today.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ginseng.funding import (
    PlanSpec,
    build_candidates,
    comparison_draw_bundle,
    evaluate_plan,
    settlement_forecast_day,
)
from ginseng.risk import cvar, probabilities, quantile
from ginseng.state import FinancialState, Obligation

from .model import DashboardData, FundingOption, LedgerEntry, PathBands

BAND_LEVELS = (5, 25, 50, 75, 95)
TROUGH_Q = 0.95
TROUGH_BINS = 20

# Same default used everywhere else in the engine that prices overdraft cost
# (scenario_service.py, optimizer.py, api.py all default to this). The TUI's
# simulate/exact flow never priced funding plans before, so it has no
# established default of its own.
OVERDRAFT_APR = 0.2999


@dataclass
class EngineRun:
    """Everything `from_engine` needs, assembled by whoever calls the engine.

    `matrix` is cash-flow only (opening cash excluded) -- the same convention
    `ginseng.metrics.cash_risk_summary` and `ginseng.exact.enumerate_exact`
    already use. `weights` is None for equally-likely simulated paths, or the
    exact oracle's own per-scenario probabilities.
    """

    kind: str                       # "simulate" or "exact"
    summary: dict
    manifest: dict
    matrix: np.ndarray
    opening_cash: float
    weights: np.ndarray | None = None
    state: FinancialState | None = None
    obligations: Sequence[Obligation] = ()
    bundle: object | None = None    # DrawBundle/PathBundle; needed only to price funding options


def _ready_days(state: FinancialState, spec: PlanSpec) -> int:
    if spec.liquidation_target <= 0:
        return 0
    return settlement_forecast_day(
        state.as_of, spec.settlement_days, spec.external_transfer_days,
        use_business_days=spec.use_business_days,
    )


def _funding_options(run: EngineRun, gap: float) -> tuple[FundingOption, ...]:
    """Price the same candidate plans `ginseng.funding` builds for the funding gap.

    Only available when the caller supplied a real `FinancialState` and draw
    bundle (the exact oracle has neither -- it is deliberately independent of
    the production `FinancialState`, so it gets no funding options).
    """
    if gap < 0.5 or run.state is None or run.bundle is None:
        return ()
    specs = build_candidates(run.state, run.obligations, gap)
    bundle = comparison_draw_bundle(run.state, run.bundle, specs)
    options = []
    for spec in specs:
        result = evaluate_plan(
            run.state, bundle, run.obligations, spec,
            operating_buffer=run.state.operating_buffer, overdraft_apr=OVERDRAFT_APR,
        )
        if not result.feasible:
            continue
        cost = (result.interest_exposure + result.withdrawal_tax_reserve
                + result.withdrawal_penalty_reserve + result.deferred_spending
                + result.overdraft_interest_exposure)
        options.append(FundingOption(
            name=spec.label, kind=spec.kind.value, cost=cost,
            ready_days=_ready_days(run.state, spec), tail=result.cash_shortfall_probability,
        ))
    return tuple(options)


def _ledger(run: EngineRun) -> tuple[LedgerEntry, ...]:
    """Every obligation is forward-looking here; the engine has no notion of
    'settled' or 'estimate' for a scheduled obligation, so all read 'pending'."""
    return tuple(
        LedgerEntry(day=o.due_in_days, label=o.label, amount=o.amount, status="pending")
        for o in run.obligations
    )


def from_engine(run: EngineRun) -> DashboardData:
    summary, manifest = run.summary, run.manifest
    balance = run.opening_cash + np.asarray(run.matrix, dtype=float)
    n_days = balance.shape[1]
    w = probabilities(balance.shape[0], run.weights)
    minima = balance.min(axis=1)

    band_values = {
        level: np.array([quantile(balance[:, day], level / 100, w) for day in range(n_days)])
        for level in BAND_LEVELS
    }
    below_zero = band_values[5] < 0
    solvent_days = int(np.argmax(below_zero)) if below_zero.any() else n_days

    counts, edges = np.histogram(minima, bins=TROUGH_BINS)

    draw_id = (manifest.get("index_hash") or manifest.get("result_hash") or "")[:12]
    gap = summary["funding_gap"]

    return DashboardData(
        plan=manifest.get("fixture", "exact-oracle"),
        paths=int(manifest.get("actual_n", balance.shape[0])),
        sampler=manifest.get("sampler") or "oracle",
        draw_id=draw_id,
        reserve_to_add=gap,
        shortfall_p=summary["cash_shortfall_probability"],
        cvar95_trough=-cvar(-minima, TROUGH_Q, w),
        deficit_dollar_days=float(w @ np.maximum(0.0, -balance).sum(axis=1)),
        solvent_days=solvent_days,
        bands=PathBands(band_values[5], band_values[25], band_values[50], band_values[75], band_values[95]),
        trough_edges=tuple(edges), trough_counts=tuple(counts),
        options=_funding_options(run, gap),
        ledger=_ledger(run),
    )
