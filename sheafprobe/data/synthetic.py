"""Synthetic mRNA dataset that *faithfully* instantiates the SheafProbe thesis.

The whole claim of SheafProbe is that conformational heterogeneity shows up as a
**cross-view gluing obstruction**: SHAPE and DMS are two *distinct projections* of one
latent base-pairing state, and a single rigid fold forces a specific, transport-linked
relationship between neighbouring positions and between the two probes. A multi-state
molecule breaks that relationship in a way that:

  * a single-channel summary (entropy of one reactivity profile) is structurally blind to,
    because each *marginal* still looks normal — the inconsistency lives in the JOINT
    (SHAPE, DMS) relationship and in the edge transport; and
  * a plain graph diffusion (identity restriction maps) cannot represent, because gluing a
    real helix requires a non-identity transport between paired bases.

Generative model (deliberately biophysically motivated, not reverse-engineered to make the
sheaf win — the cross-view + transport structure IS the stated hypothesis):

  latent state            phi_i in [0, 2pi)  ->  unit vector u_i = (cos phi_i, sin phi_i)
  helix transport         a base pair (i, j) sets phi_j = phi_i + THETA   (restriction map R)
  SHAPE projection        s_i = BASE + AMP * cos(phi_i - ALPHA_SHAPE) + noise   (all bases)
  DMS  projection         d_i = BASE + AMP * cos(phi_i - ALPHA_DMS)   + noise   (A/C only)

  single-state : one shared base field phi0, one nested set of pairs transported on top.
                 (s_i, d_i) lie on ONE ellipse (the single-fold manifold).
  multi-state  : two pairings (nested vs crossing/pseudoknot) transported from the SAME
                 phi0; the OBSERVED reactivity is the 50/50 population average of the two
                 fields' readouts. At positions paired differently by the two states the
                 averaged (s_i, d_i) falls OFF the single-fold ellipse -> irreducible
                 cross-view inconsistency localised to the competing stems.

The two classes are matched on length, candidate-graph size, DMS missingness and marginal
reactivity level, so the confound battery has real work to do.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .schema import Dataset, Sample, base_mask

# --- generative constants (documented so a reviewer can judge the construction) -------
_THETA = 2.4          # helix transport angle on a base pair (the non-identity restriction map)
_ALPHA_SHAPE = 0.0    # SHAPE reads the latent along this direction ...
_ALPHA_DMS = 1.6      # ... DMS along a *different* direction -> genuinely distinct views
_BASE = 0.5           # reactivity baseline (so marginals are centred and class-matched)
_AMP = 0.40           # projection amplitude
_BASES = np.array(["A", "C", "G", "U"])
_COMPLEMENT = {"A": "U", "U": "A", "C": "G", "G": "C"}


def _random_seq(n: int, rng: np.random.Generator) -> List[str]:
    return list(_BASES[rng.integers(0, 4, size=n)])


def _plant_stem(seq: List[str], i: int, j: int) -> None:
    """Make (i, j) Watson-Crick complementary in-place (keeps the graph sequence-plausible)."""
    comp = _COMPLEMENT.get(seq[i].upper())
    if comp is not None:
        seq[j] = comp


def _nested_stems(n: int, rng: np.random.Generator, margin: int = 3) -> List[Tuple[int, int]]:
    """A small set of NON-crossing (nested) base pairs -> a hairpin-like state."""
    pairs: List[Tuple[int, int]] = []
    n_stems = int(rng.integers(1, 3))
    lo, hi = 1, n - 2
    for _ in range(n_stems):
        if hi - lo < 2 * margin + 2:
            break
        stem_len = min(int(rng.integers(3, 6)), (hi - lo - margin) // 2)
        if stem_len < 2:
            break
        for s in range(stem_len):
            i, j = lo + s, hi - s
            if j - i <= margin:
                break
            pairs.append((i, j))
        lo += stem_len + 1
        hi -= stem_len + 1
    return pairs


def _crossing_stems(n: int, rng: np.random.Generator, margin: int = 3) -> List[Tuple[int, int]]:
    """A pseudoknot-style state whose pairs CROSS the nested state (mutually incompatible)."""
    pairs: List[Tuple[int, int]] = []
    q = max(2, n // 4)
    stem_len = min(int(rng.integers(3, 6)), q - 1)
    for s in range(stem_len):
        i, j = 1 + s, 2 * q + s
        if 0 <= i < j < n and j - i > margin:
            pairs.append((i, j))
    for s in range(stem_len):
        i, j = q + s, 3 * q + s
        if 0 <= i < j < n and j - i > margin:
            pairs.append((i, j))
    return pairs


def _transported_field(phi0: np.ndarray, pairs: List[Tuple[int, int]]) -> np.ndarray:
    """Apply the helix transport phi_j = phi_i + THETA on top of a base field phi0.

    Pairs are applied in ascending-i order; nested structure keeps it consistent.
    """
    phi = phi0.copy()
    for i, j in sorted(pairs):
        phi[j] = phi[i] + _THETA
    return phi


def _readout(phi: np.ndarray, alpha: float, noise: float, rng: np.random.Generator) -> np.ndarray:
    obs = _BASE + _AMP * np.cos(phi - alpha) + rng.normal(0.0, noise, size=phi.shape)
    return np.clip(obs, 0.0, 1.0)


def _decoys(n: int, exclude: set, rng: np.random.Generator, count: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    tries = 0
    while len(out) < count and tries < 20 * count:
        tries += 1
        i, j = sorted(int(x) for x in rng.integers(0, n, size=2))
        if j - i >= 3 and (i, j) not in exclude and (i, j) not in out:
            out.append((i, j))
    return out


def _build_sample(idx: int, n: int, multistate: bool, noise: float,
                  rng: np.random.Generator) -> Sample:
    seq = _random_seq(n, rng)
    phi0 = rng.uniform(0.0, 2.0 * np.pi, size=n)

    state_a = _nested_stems(n, rng)
    for (i, j) in state_a:
        _plant_stem(seq, i, j)

    if multistate:
        state_b = _crossing_stems(n, rng)
        for (i, j) in state_b:
            _plant_stem(seq, i, j)
        phi_a = _transported_field(phi0, state_a)
        phi_b = _transported_field(phi0, state_b)
        # 50/50 population average of the two single-fold READOUTS (not the latents):
        shape = 0.5 * (_readout(phi_a, _ALPHA_SHAPE, noise, rng)
                       + _readout(phi_b, _ALPHA_SHAPE, noise, rng))
        dms_full = 0.5 * (_readout(phi_a, _ALPHA_DMS, noise, rng)
                          + _readout(phi_b, _ALPHA_DMS, noise, rng))
        true_pairs = list(dict.fromkeys(state_a + state_b))
        n_states, label = 2, 1
    else:
        phi = _transported_field(phi0, state_a)
        shape = _readout(phi, _ALPHA_SHAPE, noise, rng)
        dms_full = _readout(phi, _ALPHA_DMS, noise, rng)
        true_pairs = list(dict.fromkeys(state_a))
        n_states, label = 1, 0

    seq_str = "".join(seq)

    exclude = set(true_pairs)
    decoys = _decoys(n, exclude, rng, count=max(2, n // 16))
    cand = [(int(i), int(j)) for (i, j) in dict.fromkeys(true_pairs + decoys) if 0 <= i < j < n]
    if not cand:
        cand = [(0, min(3, n - 1))]

    edges = np.array(cand, dtype=np.int64).T.reshape(2, -1)
    true_set = set(true_pairs)
    weights = np.array([0.85 if (i, j) in true_set else 0.55 for (i, j) in cand], dtype=np.float64)
    weights = np.clip(weights + rng.normal(0.0, 0.05, size=weights.shape), 0.5, 1.0)
    backbone = np.array([[k, k + 1] for k in range(n - 1)], dtype=np.int64).T.reshape(2, -1)

    # DMS observed only on Watson-Crick A/C; elsewhere missing (~50% by construction).
    ac = base_mask(seq_str, bases="AC")
    react_dms = np.where(ac, dms_full, np.nan)

    return Sample(
        id=f"syn{idx:05d}", seq=seq_str, n=n, edges=edges, edge_weight=weights,
        backbone=backbone, react_shape=shape, react_dms=react_dms,
        label_multistate=int(label), true_n_states=int(n_states),
        meta={"n_true_pairs": len(true_pairs), "n_decoys": len(decoys), "synthetic": True},
    )


def generate_dataset(n_samples: int = 600, length: int = 68,
                     frac_multistate: float = 0.5, noise: float = 0.15,
                     seed: int = 0) -> Dataset:
    """Generate a balanced synthetic dataset of single- and multi-state molecules.

    Classes are matched on length, graph size, DMS missingness and marginal reactivity;
    they differ only in cross-view (SHAPE vs DMS) gluing consistency.
    """
    rng = np.random.default_rng(seed)
    n_multi = int(round(n_samples * frac_multistate))
    flags = np.array([True] * n_multi + [False] * (n_samples - n_multi))
    rng.shuffle(flags)
    return [_build_sample(idx, length, bool(m), noise, rng) for idx, m in enumerate(flags)]


def confounds(sample: Sample) -> Dict[str, float]:
    """Per-sample trivial scalars for the confound battery (E* must beat these)."""
    shape = np.asarray(sample.react_shape, dtype=np.float64)
    finite = shape[np.isfinite(shape)]
    mean_react = float(finite.mean()) if finite.size else float("nan")
    dms = np.asarray(sample.react_dms, dtype=np.float64)
    frac_nan_dms = float(np.mean(~np.isfinite(dms))) if dms.size else float("nan")
    return {
        "mean_unpaired_prob": mean_react,
        "frac_nan_dms": frac_nan_dms,
        "n_candidate_edges": float(sample.edges.shape[1]),
        "length": float(sample.n),
    }
