"""Synchronous unsigned and signed NCO with state-inertia tie breaking."""

import numpy as np
from scipy.sparse import csr_matrix

from mdacn.contracts import Model, OpinionState, SimulationResult, StopReason


def initialize_opinions(n_nodes: int, negative_fraction: float, *, seed: int) -> OpinionState:
    """Owner: Daniel. Return a seeded +/-1 state with the requested negative fraction.

    negative_fraction lies in [0, 1]. Realized counts are subject to integer
    rounding (nearest integer, with halves rounded up); experiments record the
    realized fraction. Exactly that many nodes are sampled without replacement.
    """
    _validate_nonnegative_integer(n_nodes, "n_nodes")
    if not np.isfinite(negative_fraction) or not 0 <= negative_fraction <= 1:
        raise ValueError("negative_fraction must be finite and in [0, 1]")
    state = np.ones(n_nodes, dtype=np.int8)
    count = int(np.floor(n_nodes * negative_fraction + 0.5))
    state[np.random.default_rng(seed).choice(n_nodes, size=count, replace=False)] = -1
    return state


def _validate_nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _prepare_matrix(matrix: csr_matrix, model: Model) -> csr_matrix:
    """Validate the static undirected contract and prevent small-integer overflow."""
    if model not in ("unsigned", "signed"):
        raise ValueError("model must be 'unsigned' or 'signed'")
    matrix = csr_matrix(matrix, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("interaction_matrix must be square")
    allowed = (1,) if model == "unsigned" else (-1, 1)
    if not np.isin(matrix.data, allowed).all():
        raise ValueError(f"{model} matrix contains invalid interaction weights")
    if np.any(matrix.diagonal()):
        raise ValueError("interaction_matrix must have a zero diagonal; self votes are added once")
    if (matrix != matrix.T).nnz:
        raise ValueError("interaction_matrix must be symmetric")
    return matrix


def _prepare_state(state: OpinionState, n_nodes: int) -> OpinionState:
    state = np.asarray(state)
    if state.shape != (n_nodes,) or not np.isin(state, (-1, 1)).all():
        raise ValueError("state must be a one-dimensional +/-1 array matching the matrix")
    return state.astype(np.int8, copy=True)


def _step(matrix: csr_matrix, state: OpinionState) -> OpinionState:
    """Update from one unchanged prior state; a zero vote keeps that state."""
    votes = state + matrix @ state
    return np.where(votes > 0, 1, np.where(votes < 0, -1, state)).astype(np.int8)


def detect_attractor(history: list[OpinionState]) -> tuple[StopReason, int | None] | None:
    """Owner: Daniel. Classify the latest state against earlier states.

    Return ('steady_state', None), ('limit_cycle', period), or None when neither
    is detected. History includes the initial state and completed updates.
    """
    if len(history) < 2:
        return None
    for period in range(1, len(history)):
        if np.array_equal(history[-1], history[-1 - period]):
            return ("steady_state", None) if period == 1 else ("limit_cycle", period)
    return None


def simulate_nco(
    interaction_matrix: csr_matrix,
    initial_state: OpinionState,
    *,
    model: Model,
    max_steps: int,
) -> SimulationResult:
    """Owner: Daniel. Run unsigned or signed NCO until an attractor or the update limit.

    model is 'unsigned' or 'signed'. Return the stop reason, completed update
    count, terminal state, and cycle length when applicable.
    """
    _validate_nonnegative_integer(max_steps, "max_steps")
    matrix = _prepare_matrix(interaction_matrix, model)
    state = _prepare_state(initial_state, matrix.shape[0])
    # Exact packed keys prevent false cycle matches and use one bit per node.
    # Validation happens once per simulation, outside the sparse update loop.
    seen = {np.packbits(state > 0).tobytes(): 0}
    for step in range(1, max_steps + 1):
        state = _step(matrix, state)
        key = np.packbits(state > 0).tobytes()
        previous_step = seen.get(key)
        if previous_step is not None:
            period = step - previous_step
            if period == 1:
                return SimulationResult(state, step, "steady_state")
            return SimulationResult(state, step, "limit_cycle", period)
        seen[key] = step
    return SimulationResult(state, max_steps, "max_steps")
