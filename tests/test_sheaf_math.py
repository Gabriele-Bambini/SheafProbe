"""Math/sanity tests for the SheafProbe sheaf-energy model.

Run: pytest -q tests/test_sheaf_math.py
"""
from __future__ import annotations

import numpy as np
import torch

from sheafprobe.data.schema import Sample
from sheafprobe.models import SheafProbe, score_dataset


# ---------------------------------------------------------------------------
# tiny hand-built samples
# ---------------------------------------------------------------------------
def _backbone(n: int) -> np.ndarray:
    return np.array([list(range(n - 1)), list(range(1, n))], dtype=np.int64)


def _single_state_sample() -> Sample:
    """6-mer with ONE clean hairpin: (0-5)(1-4). Paired -> low react, unpaired high."""
    seq = "GCAAGC"
    edges = np.array([[0, 1], [5, 4]], dtype=np.int64)        # i<j enforced below
    edges = np.sort(edges, axis=0)
    edge_weight = np.array([0.9, 0.9], dtype=np.float64)
    # paired positions {0,1,4,5} low, unpaired {2,3} high
    react = np.array([0.1, 0.1, 0.9, 0.9, 0.1, 0.1], dtype=np.float64)
    react_dms = react.copy()
    react_dms[~np.array([b in "AC" for b in seq])] = np.nan
    return Sample(
        id="single", seq=seq, n=6, edges=edges, edge_weight=edge_weight,
        backbone=_backbone(6), react_shape=react, react_dms=react_dms,
        label_multistate=0, true_n_states=1,
    )


def _two_state_sample() -> Sample:
    """6-mer with TWO competing pairings: hairpin (0-5)(1-4) vs crossing (0-3)(2-5).
    Population-averaged reactivity => everything looks half-paired."""
    seq = "GCAAGC"
    e1 = np.array([[0, 1], [5, 4]], dtype=np.int64)           # state 1
    e2 = np.array([[0, 2], [3, 5]], dtype=np.int64)           # state 2 (crossing)
    edges = np.concatenate([e1, e2], axis=1)
    edges = np.sort(edges, axis=0)
    edge_weight = np.full(edges.shape[1], 0.7, dtype=np.float64)
    react = np.full(6, 0.5, dtype=np.float64)                 # half-paired everywhere
    react_dms = react.copy()
    react_dms[~np.array([b in "AC" for b in seq])] = np.nan
    return Sample(
        id="two", seq=seq, n=6, edges=edges, edge_weight=edge_weight,
        backbone=_backbone(6), react_shape=react, react_dms=react_dms,
        label_multistate=1, true_n_states=2,
    )


# ---------------------------------------------------------------------------
# (i) identity-mode estar equals the plain edge-weighted graph Dirichlet energy
# ---------------------------------------------------------------------------
def test_identity_estar_is_plain_dirichlet():
    torch.manual_seed(0)
    sample = _two_state_sample()
    model = SheafProbe(k=4, n_layers=3, mode="identity")

    # arbitrary node features: identity maps => disc = x_i - x_j independent of x source
    x = torch.randn(sample.n, model.k)
    estar, per_edge = model._energy(sample, x)

    cand = torch.as_tensor(sample.edges, dtype=torch.long).reshape(2, -1)
    i, j = cand[0], cand[1]
    w = torch.as_tensor(sample.edge_weight, dtype=torch.float32)
    plain = (w * ((x[i] - x[j]) ** 2).sum(dim=-1)).sum()

    assert torch.allclose(estar, plain, atol=1e-5), (float(estar), float(plain))
    assert torch.allclose(per_edge, w * ((x[i] - x[j]) ** 2).sum(dim=-1), atol=1e-5)


# ---------------------------------------------------------------------------
# (ii) single-state sample yields lower estar than two-competing-pairing sample
# ---------------------------------------------------------------------------
def test_single_state_lower_energy_than_two_state():
    torch.manual_seed(0)
    single = _single_state_sample()
    two = _two_state_sample()

    # untrained model with diagonal (near-identity) maps; a few diffusion steps.
    # The two-state sample carries strictly more competing candidate edges over the
    # same nodes, so its accumulated Dirichlet energy is higher.
    model = SheafProbe(k=4, n_layers=4, mode="diagonal")
    e_single = model(single)["estar"].item()
    e_two = model(two)["estar"].item()
    assert e_single < e_two, (e_single, e_two)

    # identity mode should agree on the ordering too
    model_id = SheafProbe(k=4, n_layers=4, mode="identity")
    assert model_id(single)["estar"].item() < model_id(two)["estar"].item()


# ---------------------------------------------------------------------------
# (iii) estar >= 0 and gradients flow through estar
# ---------------------------------------------------------------------------
def test_estar_nonneg_and_grad_flows():
    torch.manual_seed(0)
    sample = _two_state_sample()
    for mode in ("identity", "diagonal", "general"):
        model = SheafProbe(k=4, n_layers=3, mode=mode)
        out = model(sample)
        estar = out["estar"]
        assert estar.item() >= 0.0, (mode, estar.item())

        if mode == "identity":
            continue  # no parameters in the maps; recon heads still grad but skip
        model.zero_grad()
        estar.backward()
        grads = [p.grad for p in model.map_mlp.parameters() if p.grad is not None]
        assert grads, f"no gradient reached map_mlp in mode {mode}"
        total = sum(float(g.abs().sum()) for g in grads)
        assert np.isfinite(total) and total > 0.0, (mode, total)


def test_score_dataset_shape_and_dtype():
    model = SheafProbe(k=4, n_layers=2, mode="diagonal")
    ds = [_single_state_sample(), _two_state_sample()]
    scores = score_dataset(model, ds)
    assert scores.shape == (2,)
    assert scores.dtype == np.float64
    assert np.all(np.isfinite(scores)) and np.all(scores >= 0.0)
