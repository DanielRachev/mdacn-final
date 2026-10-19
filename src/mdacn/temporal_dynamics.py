
"""Temporal unsigned and signed NCO simulations."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from mdacn.dynamics import initialize_opinions, _prepare_matrix, _prepare_state, _step
from mdacn.temporal_data import TemporalSnapshots


TemporalModel = Literal["unsigned", "signed"]


@dataclass(frozen=True)
class TemporalNCOResult:
    """Opinion trajectories and per-window simulation statistics."""

    initial_state: np.ndarray
    final_state: np.ndarray
    timestamps: tuple[pd.Timestamp, ...]
    positive_fraction: np.ndarray
    negative_fraction: np.ndarray
    changed_count: np.ndarray
    states: np.ndarray | None

    @property
    def steps(self) -> int:
        return len(self.timestamps)


def simulate_temporal_nco(snapshots: TemporalSnapshots, *, model: TemporalModel = "signed",
                          initial_state: np.ndarray | None = None,
                          negative_fraction: float | None = None, seed: int | None = None, record_states: bool = False)\
        -> TemporalNCOResult:
    """Run one synchronous NCO update per snapshot, without early stopping."""

    if model not in ("unsigned", "signed"):
        raise ValueError("model must be 'unsigned' or 'signed'")

    if initial_state is not None:
        if negative_fraction is not None or seed is not None:
            raise ValueError("Provide either initial_state or negative_fraction and seed")
        state = _prepare_state(initial_state, len(snapshots.nodes))
    else:
        if negative_fraction is None or seed is None:
            raise ValueError("Provide initial_state or both negative_fraction and seed")
        state = initialize_opinions(len(snapshots.nodes), negative_fraction, seed=seed)

    n_steps = len(snapshots.timestamps)
    matrices = snapshots.signed if model == "signed" else snapshots.unsigned

    if len(matrices) != n_steps or len(snapshots.event_counts) != n_steps:
        raise ValueError("Snapshot matrices, timestamps, and event counts must be aligned")

    for matrix in matrices:
        if matrix.shape != (len(state), len(state)):
            raise ValueError("Snapshot matrix shape does not match the node count")

    original = state.copy()
    positive_fraction = np.empty(n_steps + 1, dtype=np.float64)
    negative_fraction_history = np.empty(n_steps + 1, dtype=np.float64)
    changed_count = np.zeros(n_steps, dtype=np.int64)
    states = np.empty((n_steps + 1, len(state)), dtype=np.int8) if record_states else None

    positive_fraction[0] = np.mean(state == 1) if len(state) else 0.0
    negative_fraction_history[0] = np.mean(state == -1) if len(state) else 0.0

    if states is not None:
        states[0] = state

    for t, matrix in enumerate(matrices):
        matrix = _prepare_matrix(matrix, model)
        next_state = _step(matrix, state)

        changed_count[t] = np.count_nonzero(next_state != state)
        state = next_state

        positive_fraction[t + 1] = np.mean(state == 1) if len(state) else 0.0
        negative_fraction_history[t + 1] = np.mean(state == -1) if len(state) else 0.0

        if states is not None:
            states[t + 1] = state

    return TemporalNCOResult(
        initial_state=original,
        final_state=state.copy(),
        timestamps=snapshots.timestamps,
        positive_fraction=positive_fraction,
        negative_fraction=negative_fraction_history,
        changed_count=changed_count,
        states=states,
    )
