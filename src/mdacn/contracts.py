"""Shared data contracts."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

OpinionState = NDArray[np.int8]
Model = Literal["unsigned", "signed"]
StopReason = Literal["steady_state", "limit_cycle", "max_steps"]
Strategy = Literal["random", "degree", "leverage"]


@dataclass(frozen=True)
class NetworkMatrices:
    """Owner: Jacek. Node order and matching unsigned and signed matrices.

    Both matrices are square, symmetric, and have a zero diagonal.
    Node names are alphabetical and align with matrix rows and columns.
    Matrices have float64 values: unsigned entries are 0/1, signed entries 0/+1/-1
    """

    nodes: tuple[str, ...]
    unsigned: csr_matrix
    signed: csr_matrix


@dataclass(frozen=True)
class SimulationResult:
    """Owner: Daniel. Terminal state, stop reason, and elapsed update count.

    cycle_length is present only for a limit cycle. A terminal state is not
    necessarily a steady state. Its order matches NetworkMatrices.nodes.
    """

    final_state: OpinionState
    steps: int
    stop_reason: StopReason
    cycle_length: int | None = None
