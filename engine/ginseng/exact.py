"""Bounded rational oracle. Deliberately independent of production reductions."""

from fractions import Fraction
from itertools import product


def enumerate_exact(
    net=(-40, 10, 70),
    deterministic=(0, -50, 0, -20),
    opening_cash=30,
    buffer=10,
    q=Fraction(19, 20),
    block_length=7,
    max_sequences=100_000,
):
    n, horizon = len(net), len(deterministic)
    if n < 1 or horizon < 1 or block_length < 1 or not 0 <= q <= 1:
        raise ValueError("Invalid exact model settings.")
    if any(
        not isinstance(x, int)
        for x in (*net, *deterministic, opening_cash, buffer, block_length)
    ):
        raise ValueError("Exact oracle requires integer dollar flows and block length.")
    if n**horizon > max_sequences:
        raise ValueError(
            f"Exact enumeration exceeds {max_sequences} sequences; use simulation."
        )
    rows, masses = [], {}
    failure = mean = Fraction(0)
    for indices in product(range(n), repeat=horizon):
        probability = Fraction(1, n)
        for a, z in zip(indices, indices[1:]):
            probability *= Fraction(1, block_length * n) + (
                Fraction(block_length - 1, block_length) if z == (a + 1) % n else 0
            )
        cumulative, trajectory = 0, []
        for index, fixed in zip(indices, deterministic):
            cumulative += net[index] + fixed
            trajectory.append(cumulative)
        minimum = min(trajectory)
        reserve = max(0, buffer - minimum)
        deficit = max(0, -(opening_cash + minimum))
        masses[reserve] = masses.get(reserve, Fraction(0)) + probability
        failure += probability * (deficit > 0)
        mean += probability * deficit
        rows.append(
            dict(
                indices=list(indices),
                cumulative=trajectory,
                reserve=reserve,
                deficit=deficit,
                probability=float(probability),
                rational_probability=str(probability),
            )
        )
    cumulative = Fraction(0)
    support = []
    answer = None
    for value, mass in sorted(masses.items()):
        cumulative += mass
        support.append(
            dict(
                value=value,
                probability=float(mass),
                cdf=float(cumulative),
                rational_probability=str(mass),
            )
        )
        if answer is None and cumulative >= q:
            answer = value
    summary = dict(
        required_liquidity_reserve=answer,
        cash_shortfall_probability=float(failure),
        expected_max_cash_deficit=float(mean),
        avg_cash_deficit_when_short=float(mean / failure) if failure else 0,
        funding_gap=max(0, answer - opening_cash),
    )
    return dict(
        summary=summary,
        rational=dict(
            failure=str(failure),
            mean_deficit=str(mean),
            total_probability=str(cumulative),
        ),
        sequences=rows,
        reserve_distribution=support,
        count=len(rows),
    )
