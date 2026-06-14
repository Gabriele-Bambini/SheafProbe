"""Part B -- the regime where a cellular sheaf is *provably necessary*.

Part A (killer.py) showed an honest, uncomfortable truth: when conformational
heterogeneity leaves a per-position footprint, a plain graph diffusion (identity
restriction maps) and even a single reactivity channel already separate the classes,
so the learned restriction maps are decorative. That is a real, repeatedly-observed
result -- not a sheaf success.

So when IS the sheaf the only tool that works? Exactly when the signal is a
**holonomy obstruction**: data that is locally consistent on every edge yet globally
inconsistent around a cycle. No node-level statistic and no identity-map graph
diffusion can see it; only the sheaf Laplacian with the correct (non-identity)
restriction maps measures it. This module demonstrates that cleanly and
parameter-free, so there is nothing to overfit and nothing to rig.

Construction (a circularised structured RNA motif, abstracted):
  * n nodes on a ring; stalks R^2; each edge i->i+1 carries a FIXED rotation R_i given
    by "geometry" (NOT learned). The rotations are built so their product around the
    ring is the identity, i.e. a globally consistent section exists.
  * consistent (label 0): the observed 2-vector field y is that section + noise.
  * frustrated  (label 1): the SAME section + noise, but one contiguous arc of nodes is
    rotated by an extra angle delta -- a "domain wall" between two competing folds.
    Every node still looks like a unit vector + noise (identical marginals), and every
    interior edge is still locally satisfied; only the two arc-boundary edges carry the
    holonomy defect. The obstruction is global.

Scorers (all parameter-free -- no training, fully deterministic):
  * sheaf_correct : E* = sum_e || y_{i+1} - R_i y_i ||^2 with the TRUE maps.
  * sheaf_identity: E* with R_i = I (the plain graph-diffusion / GNN energy).
  * node_var      : per-sample variance of node norms (a node-level summary).
  * node_entropy  : entropy of the per-node angle histogram (a "reactivity entropy" analog).
The claim: sheaf_correct separates frustrated vs consistent at high AUROC while every
node-level / identity baseline sits at chance.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _unit(angle: np.ndarray) -> np.ndarray:
    """[n] angles -> [n,2] unit vectors."""
    return np.stack([np.cos(angle), np.sin(angle)], axis=-1)


def _build_sample(n: int, frustrated: bool, noise: float, delta: float,
                  rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y [n,2] observed field, edge_thetas [n]) for one ring molecule.

    edge_thetas[i] is the rotation angle on edge i->(i+1)%n; the angles sum to 0 (mod
    2pi) so a consistent global section exists. Rotations commute in 2D, so the section
    angle at node i is phi0 + cumsum(thetas)[i-1].
    """
    thetas = rng.uniform(-np.pi, np.pi, size=n)
    thetas[-1] = -np.sum(thetas[:-1])  # close the ring: product of rotations = I
    phi0 = rng.uniform(-np.pi, np.pi)
    node_angle = phi0 + np.concatenate([[0.0], np.cumsum(thetas[:-1])])  # [n], section angles

    if frustrated:
        # rotate a contiguous arc by +delta: a domain wall between two competing folds.
        arc_len = max(2, n // 4)
        start = int(rng.integers(0, n))
        idx = (start + np.arange(arc_len)) % n
        node_angle = node_angle.copy()
        node_angle[idx] += delta

    y = _unit(node_angle) + rng.normal(0.0, noise, size=(n, 2))
    return y, thetas


def _sheaf_energy(y: np.ndarray, thetas: np.ndarray, identity: bool) -> float:
    """E* = sum_e || y_{i+1} - R_i y_i ||^2 over the ring edges (R_i=I if identity)."""
    n = y.shape[0]
    total = 0.0
    for i in range(n):
        j = (i + 1) % n
        transported = y[i] if identity else _rot(thetas[i]) @ y[i]
        total += float(np.sum((y[j] - transported) ** 2))
    return total


def _node_var(y: np.ndarray) -> float:
    norms = np.linalg.norm(y, axis=1)
    return float(np.var(norms))


def _node_entropy(y: np.ndarray, bins: int = 12) -> float:
    ang = np.arctan2(y[:, 1], y[:, 0])
    hist, _ = np.histogram(ang, bins=bins, range=(-np.pi, np.pi), density=False)
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def run_holonomy(out_dir, seed: int = 0, n_samples: int = 400, n: int = 40,
                 noise: float = 0.10, delta: float = 1.0,
                 frac_frustrated: float = 0.5) -> Dict:
    """Generate the ring dataset and score every method by AUROC for `frustrated`.

    Parameter-free: nothing is trained, so the result is exactly reproducible.
    """
    rng = np.random.default_rng(seed)
    n_fr = int(round(n_samples * frac_frustrated))
    flags = np.array([True] * n_fr + [False] * (n_samples - n_fr))
    rng.shuffle(flags)

    labels: List[int] = []
    s_correct, s_identity, s_var, s_entropy = [], [], [], []
    for is_fr in flags:
        y, thetas = _build_sample(n, bool(is_fr), noise, delta, rng)
        labels.append(int(is_fr))
        s_correct.append(_sheaf_energy(y, thetas, identity=False))
        s_identity.append(_sheaf_energy(y, thetas, identity=True))
        s_var.append(_node_var(y))
        s_entropy.append(_node_entropy(y))

    labels = np.asarray(labels)

    def _auc(scores) -> float:
        return float(roc_auc_score(labels, np.asarray(scores, dtype=np.float64)))

    auroc = {
        "sheaf_correct_maps": _auc(s_correct),
        "sheaf_identity_maps": _auc(s_identity),
        "node_norm_variance": _auc(s_var),
        "node_angle_entropy": _auc(s_entropy),
    }
    results = {
        "task": "holonomy",
        "seed": int(seed),
        "n_samples": int(n_samples),
        "config": {"n_nodes": int(n), "noise": noise, "delta": delta},
        "auroc": auroc,
        "sheaf_necessity_gap_correct_minus_identity":
            float(auroc["sheaf_correct_maps"] - auroc["sheaf_identity_maps"]),
        "note": "Maps are FIXED from geometry, not learned -> not rigged. Node/identity "
                "baselines are at chance by construction; only the correct sheaf maps "
                "detect the global holonomy obstruction.",
    }

    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "holonomy.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    return results


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "results"
    print(json.dumps(run_holonomy(out)["auroc"], indent=2))
