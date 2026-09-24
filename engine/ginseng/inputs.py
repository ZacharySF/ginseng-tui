"""Version 1 offline dollar-input contract; complete daily windows required."""

from datetime import date, timedelta
import json
from pathlib import Path
from dataclasses import dataclass
import math
from ginseng.state import FinancialState, Transaction, TransactionType as T, Obligation


@dataclass(frozen=True)
class InputCase:
    name: str
    state: FinancialState
    obligations: tuple
    data_seed: int | None = None


def load_input(data, name="local"):
    if isinstance(data, (str, Path)):
        data = json.loads(Path(data).read_text())
    if data.get("version") != 1:
        raise ValueError("Input version must be 1.")
    start, end, as_of = (
        date.fromisoformat(data[k]) for k in ("history_start", "history_end", "as_of")
    )
    if not start <= end < as_of:
        raise ValueError(
            "Require history_start <= history_end < as_of; no forecast-gap observations are manufactured."
        )

    def number(value, label, nonnegative=False):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (nonnegative and value < 0)
        ):
            raise ValueError(
                f"{label} must be finite"
                + (" and nonnegative." if nonnegative else ".")
            )
        return float(value)

    opening = number(data["opening_cash"], "opening_cash")
    buffer = number(data["buffer"], "buffer", True)
    q = number(data["coverage_target"], "coverage_target")
    horizon = data["horizon"]
    if not 0 <= q <= 1 or type(horizon) is not int or horizon < 1:
        raise ValueError("Coverage must be in [0,1] and horizon a positive integer.")
    rows = data["history"]
    expected = (end - start).days + 1
    if len(rows) != expected:
        raise ValueError(
            "Complete window requires one explicit row per day, including zero days."
        )
    transactions, seen = [], set()
    columns = [
        ("variable_income", T.INCOME_VARIABLE),
        ("essential_spending", T.EXPENSE_ESSENTIAL_VARIABLE),
        ("discretionary_spending", T.EXPENSE_DISCRETIONARY_VARIABLE),
    ]
    for row in rows:
        day = date.fromisoformat(row["date"])
        if day in seen or not start <= day <= end:
            raise ValueError(
                "Duplicate daily row or date outside declared history window."
            )
        seen.add(day)
        for key, kind in columns:
            transactions.append(
                Transaction(day, kind, number(row[key], key, True), key)
            )
    # Reconcile separately supplied opening cash, without resampling the transfer.
    adjustment = opening - sum(t.cash_effect for t in transactions)
    transactions.append(
        Transaction(as_of, T.TRANSFER, adjustment, "Opening cash reconciliation")
    )
    obligations = []
    for i, item in enumerate(data.get("obligations", [])):
        day = item["day"]
        if type(day) is not int or day < 1:
            raise ValueError(
                "Obligation day is a positive, one-indexed forecast offset."
            )
        obligations.append(
            Obligation(
                str(i),
                item.get("label", "Known obligation"),
                number(item["amount"], "obligation amount", True),
                day,
            )
        )
    state = FinancialState(
        as_of,
        tuple(transactions),
        (),
        (),
        (),
        (),
        (),
        buffer,
        q,
        horizon,
        history_start=start,
        history_end=end,
    )
    return InputCase(name, state, tuple(obligations))


def fixture(name):
    if name == "canonical":
        from ginseng.generate import generate_persona, DEFAULT_SEED

        return InputCase(name, generate_persona(DEFAULT_SEED), (), DEFAULT_SEED)
    if name == "tiny":
        values, opening, buffer, horizon = (
            [(0, 30, 10), (50, 30, 10), (100, 20, 10)],
            30,
            10,
            4,
        )
        obligations = [{"day": 2, "amount": 50}, {"day": 4, "amount": 20}]
    elif name == "zero-heavy":
        # Fixed 60-day synthetic history: mostly positive net, isolated losses.
        values, opening, buffer, horizon = (
            [(100, 20, 10) if i % 15 else (0, 80, 20) for i in range(60)],
            100,
            10,
            30,
        )
        obligations = []
    elif name == "drought-heavy":
        # Fixed persistent 40-day dry / 20-day busy cycle, no fitted regimes.
        values, opening, buffer, horizon = (
            [(0, 60, 20) if i % 60 < 40 else (400, 60, 20) for i in range(180)],
            600,
            1000,
            30,
        )
        obligations = [{"day": 17, "amount": 500}]
    else:
        raise ValueError(
            "Unknown fixture; choose canonical, tiny, zero-heavy, drought-heavy."
        )
    start = date(2026, 1, 1)
    data = dict(
        version=1,
        history_start=str(start),
        history_end=str(start + timedelta(days=len(values) - 1)),
        as_of=str(start + timedelta(days=len(values))),
        opening_cash=opening,
        buffer=buffer,
        coverage_target=0.95,
        horizon=horizon,
        obligations=obligations,
        history=[
            dict(
                date=str(start + timedelta(days=i)),
                variable_income=a,
                essential_spending=b,
                discretionary_spending=c,
            )
            for i, (a, b, c) in enumerate(values)
        ],
    )
    return load_input(data, name)
