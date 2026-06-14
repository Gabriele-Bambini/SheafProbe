"""Baselines for SheafProbe.

Two reference scorers the learned sheaf energy `E*` must beat on the
multistate-separation task:

1. `entropy_bpp_score` — a NON-LEARNED gold standard. It turns reactivity into a
   per-position pairing probability `p_i`, blends in the BPP-like `edge_weight`
   prior, and returns the summed binary entropy. A multi-state molecule has many
   half-paired positions (p near 0.5 => high entropy), so total entropy is a
   natural, parameter-free heterogeneity proxy.
2. `TransformerRecon` — a small per-position transformer trained to reconstruct
   BOTH reactivity channels (SHAPE + DMS). Its per-sample reconstruction residual
   is the learned-but-sheaf-free heterogeneity score: a single-state molecule is
   easy to reconstruct; a population-averaged multi-state one is not.

CPU-only, small, fast. No `torch_geometric`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from sheafprobe.data.schema import BASE_TO_IDX, Dataset, Sample, base_mask


# ----------------------------------------------------------------------------
# 1. Non-learned gold standard: entropy of pairing probability
# ----------------------------------------------------------------------------
def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """Elementwise binary entropy in bits, clipped to avoid log(0)."""
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def entropy_bpp_score(sample: Sample) -> float:
    """Gold-standard non-learned heterogeneity score.

    Per-position pairing probability `p_i` is derived from reactivity (low
    reactivity => high probability of being paired) and blended with the
    `edge_weight` BPP prior aggregated to nodes. The score is `Σ_i H2(p_i)`:
    half-paired positions (p≈0.5) dominate, so multi-state molecules score high.

    Args:
        sample: a `Sample`.

    Returns:
        Summed binary entropy of the per-position pairing probability.
    """
    n = sample.n

    # Reactivity -> pairing prob. Prefer SHAPE (all bases); fall back to DMS.
    react = np.array(sample.react_shape, dtype=np.float64)
    dms = np.array(sample.react_dms, dtype=np.float64)
    fill = np.isnan(react)
    react[fill] = dms[fill]  # use DMS where SHAPE missing

    # Normalize reactivity to [0,1] robustly; nan -> neutral 0.5 reactivity.
    finite = np.isfinite(react)
    if finite.sum() >= 2:
        lo, hi = np.nanpercentile(react[finite], [5, 95])
        if hi - lo < 1e-9:
            hi = lo + 1.0
        react_norm = (react - lo) / (hi - lo)
    else:
        react_norm = np.full(n, 0.5)
    react_norm = np.clip(react_norm, 0.0, 1.0)
    react_norm[~np.isfinite(react_norm)] = 0.5

    # low reactivity -> high p_paired
    p_react = 1.0 - react_norm

    # BPP prior: aggregate edge_weight onto incident nodes (max-pool), bounded.
    p_prior = np.zeros(n, dtype=np.float64)
    edges = sample.edges
    w = np.asarray(sample.edge_weight, dtype=np.float64)
    if edges.shape[1] > 0:
        for col in range(edges.shape[1]):
            i, j = int(edges[0, col]), int(edges[1, col])
            p_prior[i] = max(p_prior[i], w[col])
            p_prior[j] = max(p_prior[j], w[col])

    # Blend reactivity-derived and prior-derived pairing probabilities.
    p = 0.5 * p_react + 0.5 * p_prior
    return float(_binary_entropy(p).sum())


# ----------------------------------------------------------------------------
# 2. Learned, sheaf-free baseline: per-position transformer reconstruction
# ----------------------------------------------------------------------------
def _sample_to_features(sample: Sample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-position inputs/targets/observation-masks for the transformer.

    Returns:
        feats   [n, 6]  : base one-hot (4) + normalized position + node BPP prior.
        targets [n, 2]  : observed (react_shape, react_dms), nan -> 0.
        obs     [n, 2]  : 1.0 where the channel is observed (finite), else 0.
    """
    n = sample.n
    seq = sample.seq

    oh = np.zeros((n, 4), dtype=np.float32)
    for i, b in enumerate(seq):
        oh[i, BASE_TO_IDX.get(b.upper(), 0)] = 1.0
    pos = (np.arange(n, dtype=np.float32) / max(n - 1, 1)).reshape(n, 1)

    prior = np.zeros((n, 1), dtype=np.float32)
    edges = sample.edges
    w = np.asarray(sample.edge_weight, dtype=np.float32)
    for col in range(edges.shape[1]):
        i, j = int(edges[0, col]), int(edges[1, col])
        prior[i, 0] = max(prior[i, 0], w[col])
        prior[j, 0] = max(prior[j, 0], w[col])

    feats = np.concatenate([oh, pos, prior], axis=1)  # [n, 6]

    shape = np.asarray(sample.react_shape, dtype=np.float32)
    dms = np.asarray(sample.react_dms, dtype=np.float32)
    obs = np.stack([np.isfinite(shape), np.isfinite(dms)], axis=1).astype(np.float32)
    tgt = np.stack([np.nan_to_num(shape), np.nan_to_num(dms)], axis=1).astype(np.float32)

    return (
        torch.from_numpy(feats),
        torch.from_numpy(tgt),
        torch.from_numpy(obs),
    )


class TransformerRecon(nn.Module):
    """Small per-position transformer reconstructing both reactivity channels.

    Sheaf-free learned baseline: it sees per-residue features (base, position,
    BPP prior) and a learned positional encoding, then predicts the two
    reactivity channels. The per-sample reconstruction residual on *observed*
    positions is the heterogeneity score (see `transformer_scores`).
    """

    def __init__(self, in_dim: int = 6, d_model: int = 32, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 68) -> None:
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            batch_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 2)  # -> (pred_shape, pred_dms)
        self.max_len = max_len

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats [n, in_dim] -> preds [n, 2] (reactivity channels)."""
        n = feats.shape[0]
        idx = torch.arange(min(n, self.max_len))
        h = self.in_proj(feats)
        h = h + self.pos_emb(idx.clamp(max=self.max_len - 1))[:n]
        h = self.encoder(h.unsqueeze(0)).squeeze(0)  # add/remove batch dim
        return self.head(h)


def _masked_recon_mse(pred: torch.Tensor, tgt: torch.Tensor,
                      obs: torch.Tensor) -> torch.Tensor:
    """Mean squared reconstruction error over observed channel entries only."""
    diff2 = (pred - tgt) ** 2 * obs
    denom = obs.sum().clamp_min(1.0)
    return diff2.sum() / denom


def transformer_scores(dataset: Dataset, epochs: int = 150, seed: int = 0,
                       d_model: int = 32, lr: float = 1e-2) -> np.ndarray:
    """Train a shared `TransformerRecon` on the dataset, return per-sample residual.

    The model is trained to reconstruct observed reactivities across the whole
    dataset; the per-sample reconstruction residual then measures how poorly a
    single per-position model explains that molecule — high for population-
    averaged multi-state RNAs. Labels are never used.

    Args:
        dataset: list of `Sample`.
        epochs:  training epochs (<=200, CPU-fast).
        seed:    RNG seed.
        d_model: transformer width.
        lr:      Adam learning rate.

    Returns:
        np.ndarray[float] of length |dataset|: per-sample recon residual (score).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if len(dataset) == 0:
        return np.zeros(0, dtype=np.float64)

    max_len = max(s.n for s in dataset)
    feats = [_sample_to_features(s) for s in dataset]

    model = TransformerRecon(in_dim=6, d_model=d_model, max_len=max_len)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(int(epochs)):
        opt.zero_grad()
        loss = torch.zeros(())
        for f, t, o in feats:
            pred = model(f)
            loss = loss + _masked_recon_mse(pred, t, o)
        loss = loss / len(feats)
        loss.backward()
        opt.step()

    model.eval()
    scores = np.zeros(len(dataset), dtype=np.float64)
    with torch.no_grad():
        for k, (f, t, o) in enumerate(feats):
            pred = model(f)
            scores[k] = float(_masked_recon_mse(pred, t, o))
    return scores
