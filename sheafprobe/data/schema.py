"""Frozen data contract for SheafProbe.

A `Sample` is one RNA molecule with its candidate base-pair graph and two
chemical-probing "views" (SHAPE + DMS). The two reagents are deliberately kept as
separate fields: they are distinct linear projections of one latent pairing state,
never to be averaged. `label_multistate` is the held-out validation label for the
killer experiment (it must NOT be used during structure fitting / E* computation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Sample:
    id: str
    seq: str                  # RNA over {A,C,G,U}
    n: int
    edges: np.ndarray         # [2, E] int64 candidate base-pair edges, i<j
    edge_weight: np.ndarray   # [E] float in (0,1], BPP-like prior weight
    backbone: np.ndarray      # [2, n-1] int64 backbone edges (i, i+1)
    react_shape: np.ndarray   # [n] float; np.nan where missing
    react_dms: np.ndarray     # [n] float; np.nan where missing (typically non-A/C positions)
    label_multistate: int     # 1 = multi-state / pseudoknot, 0 = single-state
    true_n_states: int = -1   # ground-truth #states (synthetic); -1 if unknown
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.edges = np.asarray(self.edges, dtype=np.int64).reshape(2, -1)
        self.backbone = np.asarray(self.backbone, dtype=np.int64).reshape(2, -1)
        self.edge_weight = np.asarray(self.edge_weight, dtype=np.float64).reshape(-1)
        self.react_shape = np.asarray(self.react_shape, dtype=np.float64).reshape(-1)
        self.react_dms = np.asarray(self.react_dms, dtype=np.float64).reshape(-1)
        assert self.edges.shape[1] == self.edge_weight.shape[0], "edges/edge_weight mismatch"
        assert self.react_shape.shape[0] == self.n, "react_shape length != n"
        assert self.react_dms.shape[0] == self.n, "react_dms length != n"


Dataset = List[Sample]


# Canonical base -> index map used across the package.
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 0}


def base_mask(seq: str, bases: str = "AC") -> np.ndarray:
    """Boolean per-position mask of which residues belong to `bases`.

    Used to apply the DMS probe head only on Watson-Crick A/C positions.
    """
    want = set(bases.upper())
    return np.array([b.upper() in want for b in seq], dtype=bool)


def one_hot(seq: str) -> np.ndarray:
    """[n, 4] one-hot encoding over {A,C,G,U}."""
    oh = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq):
        oh[i, BASE_TO_IDX.get(b.upper(), 0)] = 1.0
    return oh


class DataUnavailable(RuntimeError):
    """Raised by real-data loaders when the dataset cannot be obtained.

    Carries a human-readable hint (URL / Kaggle slug) so the CLI can skip gracefully.
    """
