"""A one-view entropy projection. Weights change; simulated paths do not."""

from dataclasses import dataclass

import numpy as np

from ginseng.risk import probabilities
from ginseng.simulate import DrawBundle, _joint_history
from ginseng.state import FinancialState


@dataclass(frozen=True)
class DroughtView:
    probability: float
    window_days: int = 14
    income_fraction: float = 0.5


def entropy_project(event: np.ndarray, target: float, prior: np.ndarray | None = None) -> np.ndarray:
    """Exact minimum KL(w || prior) subject to sum(w[event]) == target.

    For one binary view, scaling each group's prior mass is the closed-form
    entropy-pooling solution; a general nonlinear solver adds no value.
    """
    event = np.asarray(event, dtype=bool)
    p = probabilities(len(event), prior)
    if not np.isfinite(target) or not 0 <= target <= 1:
        raise ValueError("Stress probability must be in [0, 1].")
    mass = float(p[event].sum())
    other_mass = float(p[~event].sum())
    if (mass == 0 and target > 0) or (other_mass == 0 and target < 1):
        raise ValueError("The requested view has no supporting scenarios.")
    w = np.zeros_like(p)
    if mass > 0:
        w[event] = p[event] * target / mass
    if other_mass > 0:
        w[~event] = p[~event] * (1 - target) / other_mass
    return probabilities(len(w), w)


def scenario_weights(state: FinancialState, bundle: DrawBundle, view: DroughtView | None) -> tuple[np.ndarray, dict]:
    prior = probabilities(bundle.n_paths)
    metadata = {"status": "inactive", "label": "Stress assumption — not an estimated probability"}
    if view is None:
        return prior, metadata
    # Always condition on the same initial days when the funding comparison
    # extends the chart. Otherwise extending a plan could change its weights.
    window = min(view.window_days, bundle.horizon_days)
    income = _joint_history(state)["variable_income"].to_numpy()
    threshold = float(income.mean() * window * view.income_fraction)
    event = income[bundle.index_matrix[:, :window]].sum(axis=1) <= threshold
    metadata.update(
        target_probability=view.probability,
        baseline_probability=float(prior[event].sum()),
        window_days=window, income_threshold=threshold,
        definition=f"Variable income of at most ${threshold:,.2f} over the first {window} days.",
    )
    try:
        weights = entropy_project(event, view.probability, prior)
    except ValueError as error:
        return prior, {**metadata, "status": "unsupported", "message": str(error)}
    achieved = float(weights[event].sum())
    return weights, {**metadata, "status": "active", "achieved_probability": achieved,
                     "constraint_residual": abs(achieved - view.probability)}
