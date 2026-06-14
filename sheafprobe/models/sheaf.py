"""SheafProbe: sheaf-diffusion model whose Dirichlet (inconsistency) energy E*
separates multi-state / pseudoknotted mRNAs from single-state ones.

Two chemical-probing views (SHAPE on all bases, DMS on A/C only) are treated as
distinct linear projections of one latent pairing state. We learn a cellular sheaf
on the nucleotide graph (backbone + candidate base-pair edges): a single rigid
molecule glues into a globally consistent 0-cochain (section) with LOW Dirichlet
energy, while two competing pairings cannot be glued by any single section, so the
irreducible energy `E*` stays high. A plain GNN (identity restriction maps) cannot
represent the bond-frame transport, so its energy separates the classes less well.

Adapts the clamp-and-diffuse discipline of ../sheaf-epistasis (SheafDiffusion,
inconsistency_energy) but is self-contained: diagonal/general restriction maps
initialised near identity, CPU-only, manual message passing via index_add_.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..data.schema import Dataset, Sample, base_mask, one_hot


# ---------------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------------
def _candidate_edge_index(sample: Sample) -> torch.Tensor:
    """[2, E] long tensor of candidate base-pair edges (row0=dst i, row1=src j)."""
    if sample.edges.size == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.as_tensor(sample.edges, dtype=torch.long).reshape(2, -1)


def _diffusion_edge_index(sample: Sample) -> torch.Tensor:
    """Symmetrised backbone + candidate edges used for the diffusion sweep.

    Diffusion needs both directions; energy is measured separately on the
    directed candidate edges only.
    """
    parts = []
    if sample.backbone.size:
        bb = torch.as_tensor(sample.backbone, dtype=torch.long).reshape(2, -1)
        parts.append(bb)
    if sample.edges.size:
        ce = torch.as_tensor(sample.edges, dtype=torch.long).reshape(2, -1)
        parts.append(ce)
    if not parts:
        return torch.zeros(2, 0, dtype=torch.long)
    ei = torch.cat(parts, dim=1)
    # symmetrise: add reverse edges
    rev = ei.flip(0)
    return torch.cat([ei, rev], dim=1)


def _positional_encoding(n: int, k: int) -> torch.Tensor:
    """Sinusoidal positional encoding [n, k] (k even-padded)."""
    pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, k, 2, dtype=torch.float32) * (-np.log(10000.0) / max(k, 1)))
    pe = torch.zeros(n, k)
    pe[:, 0::2] = torch.sin(pos * div)
    if k > 1:
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class SheafProbe(nn.Module):
    """Sheaf-diffusion model with selectable restriction-map family.

    Parameters
    ----------
    k : int
        Stalk dimension (per-node feature size). Keep <= 8 for CPU speed.
    n_layers : int
        Number of sheaf-diffusion sweeps.
    eps : float
        Diffusion step size.
    mode : {"general", "diagonal", "identity"}
        Restriction-map family. "identity" reduces E* to the plain graph
        Dirichlet energy (the GNN baseline); "diagonal" learns a length-k scale
        per edge; "general" learns a full k x k map per edge.
    """

    def __init__(self, k: int = 8, n_layers: int = 4, eps: float = 0.3, mode: str = "general"):
        super().__init__()
        assert mode in ("general", "diagonal", "identity"), f"bad mode {mode!r}"
        self.k = int(k)
        self.n_layers = int(n_layers)
        self.eps = float(eps)
        self.mode = mode

        # node encoder: one-hot(4) + reactivity[shape, dms_filled, dms_mask](3)
        # + positional(k) -> x0 [n, k]. Ingesting BOTH probe views is what lets the
        # latent recover the per-position state and the sheaf detect cross-view conflict.
        self.encoder = nn.Linear(4 + 3 + self.k, self.k)

        # restriction-map generators on the directed candidate edges
        if mode == "diagonal":
            # MLP on concatenated base one-hots -> length-k diagonal (init ~1)
            self.map_mlp = nn.Sequential(
                nn.Linear(8, 2 * self.k), nn.ReLU(),
                nn.Linear(2 * self.k, 2 * self.k),
            )
            nn.init.zeros_(self.map_mlp[-1].weight)
            nn.init.zeros_(self.map_mlp[-1].bias)
        elif mode == "general":
            # MLP on concatenated base one-hots -> two k x k maps (init ~ identity)
            self.map_mlp = nn.Sequential(
                nn.Linear(8, 4 * self.k), nn.ReLU(),
                nn.Linear(4 * self.k, 2 * self.k * self.k),
            )
            nn.init.zeros_(self.map_mlp[-1].weight)
            nn.init.zeros_(self.map_mlp[-1].bias)
        # identity: no parameters

        # probe heads
        self.W_shape = nn.Linear(self.k, 1)
        self.W_dms = nn.Linear(self.k, 1)

    # -- encoding -----------------------------------------------------------
    def _encode(self, sample: Sample) -> torch.Tensor:
        oh = torch.as_tensor(one_hot(sample.seq), dtype=torch.float32)  # [n,4]
        shape = torch.as_tensor(sample.react_shape, dtype=torch.float32)
        dms = torch.as_tensor(sample.react_dms, dtype=torch.float32)
        s_mask = torch.isfinite(shape)
        d_mask = torch.isfinite(dms)
        react = torch.stack([
            torch.where(s_mask, shape, torch.zeros_like(shape)),
            torch.where(d_mask, dms, torch.zeros_like(dms)),
            d_mask.float(),                                            # DMS missingness flag
        ], dim=-1)                                                     # [n,3]
        pe = _positional_encoding(sample.n, self.k)                     # [n,k]
        return self.encoder(torch.cat([oh, react, pe], dim=-1))        # [n,k]

    # -- restriction maps on the *candidate* (directed) edges ---------------
    def _candidate_maps(self, sample: Sample, edge_index: torch.Tensor):
        """Return (F_dst, F_src) for the candidate edges.

        Shapes:
          diagonal -> [E, k] each (elementwise scale)
          general  -> [E, k, k] each (matrix transport)
          identity -> [E, k] of ones (elementwise; equals plain Dirichlet)
        """
        E = edge_index.shape[1]
        if self.mode == "identity" or E == 0:
            ones = torch.ones(E, self.k)
            return ones, ones

        oh = torch.as_tensor(one_hot(sample.seq), dtype=torch.float32)  # [n,4]
        i, j = edge_index[0], edge_index[1]
        feat = torch.cat([oh[i], oh[j]], dim=-1)                        # [E,8]
        out = self.map_mlp(feat)

        if self.mode == "diagonal":
            F_dst = 1.0 + out[:, : self.k]
            F_src = 1.0 + out[:, self.k:]
            return F_dst, F_src

        # general: reshape to matrices, add identity
        m = out.view(E, 2, self.k, self.k)
        eye = torch.eye(self.k).unsqueeze(0)
        F_dst = eye + m[:, 0]
        F_src = eye + m[:, 1]
        return F_dst, F_src

    @staticmethod
    def _apply_map(F: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply restriction map(s) F to node features x[E, k].

        F is [E,k] (diagonal/identity) -> elementwise, or [E,k,k] -> matmul.
        """
        if F.dim() == 2:
            return F * x
        return torch.bmm(F, x.unsqueeze(-1)).squeeze(-1)

    def _discrepancy(self, x: torch.Tensor, edge_index: torch.Tensor,
                     F_dst: torch.Tensor, F_src: torch.Tensor) -> torch.Tensor:
        """Per-edge sheaf discrepancy F_dst x_i - F_src x_j on candidate edges -> [E,k]."""
        i, j = edge_index[0], edge_index[1]
        return self._apply_map(F_dst, x[i]) - self._apply_map(F_src, x[j])

    # -- diffusion ----------------------------------------------------------
    def _diffuse(self, sample: Sample, x0: torch.Tensor) -> torch.Tensor:
        """Run n_layers sheaf-diffusion sweeps over backbone + candidate edges."""
        diff_ei = _diffusion_edge_index(sample)
        if diff_ei.shape[1] == 0:
            return x0
        F_dst, F_src = self._candidate_maps(sample, diff_ei)
        i = diff_ei[0]
        x = x0
        for _ in range(self.n_layers):
            disc = self._discrepancy(x, diff_ei, F_dst, F_src)        # [E,k]
            msg = self._apply_map(F_dst, disc)                        # F_dst^T-ish transport back
            agg = torch.zeros_like(x).index_add_(0, i, msg)          # scatter to dst
            x = x - self.eps * agg
            x = torch.tanh(x)
        return x

    # -- public API ---------------------------------------------------------
    def forward(self, sample: Sample) -> dict:
        """Encode, diffuse, probe; return features, predictions, E*, per-edge energy."""
        x0 = self._encode(sample)
        x = self._diffuse(sample, x0)

        pred_shape = self.W_shape(x).squeeze(-1)                      # [n]
        pred_dms = self.W_dms(x).squeeze(-1)                          # [n]

        estar, edge_energy = self._energy(sample, x)
        return {
            "x": x,
            "pred_shape": pred_shape,
            "pred_dms": pred_dms,
            "estar": estar,
            "edge_energy": edge_energy,
        }

    def _energy(self, sample: Sample, x: torch.Tensor):
        """Edge-weighted sheaf Dirichlet energy on candidate edges.

        E* = sum_e w_e * ||F_dst x_i - F_src x_j||^2 ; also return per-edge term.
        """
        cand_ei = _candidate_edge_index(sample)
        E = cand_ei.shape[1]
        if E == 0:
            return x.new_zeros(()), x.new_zeros(0)
        F_dst, F_src = self._candidate_maps(sample, cand_ei)
        disc = self._discrepancy(x, cand_ei, F_dst, F_src)           # [E,k]
        w = torch.as_tensor(sample.edge_weight, dtype=torch.float32)  # [E]
        per_edge = w * (disc ** 2).sum(dim=-1)                       # [E]
        return per_edge.sum(), per_edge


# ---------------------------------------------------------------------------
# training / scoring
# ---------------------------------------------------------------------------
def make_model(k: int = 8, n_layers: int = 4, eps: float = 0.3,
               mode: str = "general", seed: int = 0) -> "SheafProbe":
    """Seed THEN construct, so weight initialisation is reproducible.

    (Setting the seed only inside the training loop, after construction, made the
    random init depend on prior RNG draws and produced irreproducible AUROCs.)
    """
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    return SheafProbe(k=k, n_layers=n_layers, eps=eps, mode=mode)


def _recon_loss(pred_shape: torch.Tensor, pred_dms: torch.Tensor,
                sample: Sample) -> torch.Tensor:
    """MSE of predicted vs observed reactivity, ignoring nan; DMS on A/C only."""
    shape_obs = torch.as_tensor(sample.react_shape, dtype=torch.float32)
    dms_obs = torch.as_tensor(sample.react_dms, dtype=torch.float32)

    terms = []
    s_valid = ~torch.isnan(shape_obs)
    if s_valid.any():
        terms.append(((pred_shape[s_valid] - shape_obs[s_valid]) ** 2).mean())

    ac = torch.as_tensor(base_mask(sample.seq, "AC"), dtype=torch.bool)
    d_valid = (~torch.isnan(dms_obs)) & ac
    if d_valid.any():
        terms.append(((pred_dms[d_valid] - dms_obs[d_valid]) ** 2).mean())

    if not terms:
        return pred_shape.new_zeros(())
    return torch.stack(terms).mean()


def train_sheafprobe(model: SheafProbe, dataset: Dataset, epochs: int = 150,
                     lr: float = 1e-2, lam: float = 1.0, seed: int = 0) -> SheafProbe:
    """Fit reconstruction + lam * E* over the dataset with Adam (CPU).

    The validation label `label_multistate` is never read here.
    """
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(int(epochs)):
        opt.zero_grad()
        total = torch.zeros(())
        for sample in dataset:
            out = model(sample)
            recon = _recon_loss(out["pred_shape"], out["pred_dms"], sample)
            total = total + recon + lam * out["estar"]
        loss = total / max(len(dataset), 1)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def score_dataset(model: SheafProbe, dataset: Dataset) -> np.ndarray:
    """Return E* per sample as a float64 numpy array (len = |dataset|)."""
    model.eval()
    return np.array([float(model(s)["estar"]) for s in dataset], dtype=np.float64)


@torch.no_grad()
def per_edge_energy(model: SheafProbe, sample: Sample) -> np.ndarray:
    """Per-candidate-edge sheaf energy for the competing-stem localization figure."""
    model.eval()
    return model(sample)["edge_energy"].detach().cpu().numpy().astype(np.float64)
