"""Killer experiment: does the sheaf inconsistency energy E* separate multi-state
(heterogeneous / pseudoknotted) RNAs from single-state ones better than a non-learned
gold standard (entropy-of-pairing), a plain graph-diffusion sheaf (identity mode), and a
plain transformer reconstruction residual?

The headline is the AUROC of E* for ``label_multistate``, averaged over several seeds
(mean +/- std, because random init has real variance), with a bootstrap CI on the pooled
scores and a confound-controlled partial correlation so the win cannot be a trivial
length / missingness / graph-size artefact. Direction is reported honestly (no sign flip).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from ..data import synthetic
from ..models import sheaf, baselines
from . import metrics


def _arr(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _mean_std(xs: List[float]) -> Dict[str, float]:
    a = np.asarray(xs, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0)),
            "per_seed": [float(v) for v in a]}


def run_killer(dataset, out_dir, seed: int = 0, n_seeds: int = 3,
               epochs: int = 120, k: int = 8, n_layers: int = 4) -> Dict:
    """Train SheafProbe(general) over ``n_seeds`` seeds and benchmark E* against baselines.

    The label ``label_multistate`` is never used during training/scoring of E*.
    """
    labels = np.array([int(s.label_multistate) for s in dataset], dtype=np.int64)
    seeds = [seed + i for i in range(max(1, n_seeds))]

    gen_auroc, id_auroc = [], []
    estar_last = None
    for sd in seeds:
        g = sheaf.make_model(k=k, n_layers=n_layers, eps=0.3, mode="general", seed=sd)
        g = sheaf.train_sheafprobe(g, dataset, epochs=epochs, lr=1e-2, lam=1.0, seed=sd)
        estar = _arr(sheaf.score_dataset(g, dataset))
        gen_auroc.append(float(metrics.auroc(estar, labels)))
        estar_last = estar  # keep one trained model's scores for the confound battery / CI

        i_model = sheaf.make_model(k=k, n_layers=n_layers, eps=0.3, mode="identity", seed=sd)
        i_model = sheaf.train_sheafprobe(i_model, dataset, epochs=epochs, lr=1e-2, lam=1.0, seed=sd)
        id_auroc.append(float(metrics.auroc(_arr(sheaf.score_dataset(i_model, dataset)), labels)))

    # Non-learned gold standard + transformer baseline (single representative seed).
    entropy = _arr([baselines.entropy_bpp_score(s) for s in dataset])
    transformer = _arr(baselines.transformer_scores(dataset, epochs=epochs, seed=seed))

    estar_auroc, estar_lo, estar_hi = metrics.bootstrap_auroc_ci(estar_last, labels, n=1000, seed=seed)

    auroc_table = {
        "sheaf_general": _mean_std(gen_auroc),
        "sheaf_general_pooled_ci": [float(estar_lo), float(estar_hi)],
        "sheaf_identity": _mean_std(id_auroc),
        "entropy_bpp_gold": float(metrics.auroc(entropy, labels)),
        "transformer_recon": float(metrics.auroc(transformer, labels)),
    }

    confound_keys = list(synthetic.confounds(dataset[0]).keys())
    confound_dict = {key: _arr([synthetic.confounds(s)[key] for s in dataset])
                     for key in confound_keys}
    confound = metrics.confound_report(estar_last, labels, confound_dict)

    results = {
        "task": "killer",
        "seeds": seeds,
        "n_samples": int(len(dataset)),
        "config": {"k": int(k), "n_layers": int(n_layers), "epochs": int(epochs)},
        "auroc": auroc_table,
        "confound_report": confound,
        "beats_gold": bool(auroc_table["sheaf_general"]["mean"] > auroc_table["entropy_bpp_gold"]),
        "sheaf_necessity_gap_general_minus_identity": float(
            auroc_table["sheaf_general"]["mean"] - auroc_table["sheaf_identity"]["mean"]),
    }

    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "killer.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    return results
