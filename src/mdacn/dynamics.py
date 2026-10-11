"""Synchronous unsigned and signed NCO interfaces."""

from scipy.sparse import csr_matrix

from mdacn.contracts import Model, OpinionState, SimulationResult, StopReason


def initialize_opinions(n_nodes: int, negative_fraction: float, *, seed: int) -> OpinionState:
    """Owner: Daniel. Return a seeded +/-1 state with the requested negative fraction.

    negative_fraction lies in [0, 1]. Realized counts are subject to integer
    rounding; experiments record the realized fraction.
    """
    raise NotImplementedError("Daniel: initialize_opinions")


def unsigned_step(adjacency: csr_matrix, state: OpinionState) -> OpinionState:
    """Owner: Daniel. Return one synchronous unsigned NCO update without input mutation.

    Each opinion follows the majority of its neighbors and its own prior state.
    A tied vote preserves the prior state. Adjacency is binary and unsigned.
    """
    raise NotImplementedError("Daniel: unsigned_step")


def signed_step(weights: csr_matrix, state: OpinionState) -> OpinionState:
    """Owner: Daniel. Return one synchronous signed NCO update without input mutation.

    Positive ties favor agreement; negative ties favor disagreement. The node's
    prior state contributes to its vote. A tied vote preserves the prior state.
    """
    raise NotImplementedError("Daniel: signed_step")


def detect_attractor(history: list[OpinionState]) -> tuple[StopReason, int | None] | None:
    """Owner: Daniel. Classify the latest state against earlier states.

    Return ('steady_state', None), ('limit_cycle', period), or None when neither
    is detected. History includes the initial state and completed updates.
    """
    raise NotImplementedError("Daniel: detect_attractor")


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
    raise NotImplementedError("Daniel: simulate_nco")
