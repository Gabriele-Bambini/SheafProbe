"""Ablation battery for SheafProbe.

Three knobs, each isolated so the contribution is attributable:
  1. restriction-map family   : mode in {general, diagonal, identity}
  2. stalk dimension          : k in {1, 4, 8}
  3. number of reagents       : dual (SHAPE + DMS) vs single (DMS masked to all-nan)

Every cell reports the AUROC of E* for `label_multistate`. The dual-vs-single contrast tests
the core thesis that *two distinct probing views* are what expose the gluing obstruction.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Dict, List

import numpy as np

from ..data.schema import Sample
from ..models import sheaf
from . import metrics


def _labels(dataset) -> np.ndarray:
    return np.array([int(s.label_multistate) for s in dataset], dtype=np.int64)


def _auroc_for(dataset, labels, *, mode: str, k: int, n_layers: int,
               epochs: int, seed: int) -> float:
    """Train a SheafProbe with the given config and return E* AUROC vs labels."""
    model = sheaf.make_model(k=k, n_layers=n_layers, eps=0.3, mode=mode, seed=seed)
    model = sheaf.train_sheafprobe(model, dataset, epochs=epochs, lr=1e-2, lam=1.0, seed=seed)
    scores = np.asarray(sheaf.score_dataset(model, dataset), dtype=np.float64).reshape(-1)
    return float(metrics.auroc(scores, labels))


def _mask_dms(dataset) -> List[Sample]:
    """Return a copy of `dataset` with the DMS channel removed (all np.nan).

    Tests the single-reagent regime: only SHAPE remains, so there is no second view to
    fail to glue against.
    """
    masked: List[Sample] = []
    for s in dataset:
        nan_dms = np.full(s.n, np.nan, dtype=np.float64)
        masked.append(replace(s, react_dms=nan_dms))
    return masked


def run_ablations(dataset, out_dir, seed: int = 0,
                  epochs: int = 150, n_layers: int = 4) -> Dict:
    """Run the three-knob ablation grid and write ``results/ablations.json``.

    Returns the JSON-able results dict.
    """
    labels = _labels(dataset)

    # 1. Restriction-map family (stalk dim fixed at 8). ------------------------------
    mode_auroc = {
        mode: _auroc_for(dataset, labels, mode=mode, k=8, n_layers=n_layers,
                         epochs=epochs, seed=seed)
        for mode in ("general", "diagonal", "identity")
    }

    # 2. Stalk dimension (general mode). ---------------------------------------------
    k_auroc = {
        str(k): _auroc_for(dataset, labels, mode="general", k=k, n_layers=n_layers,
                           epochs=epochs, seed=seed)
        for k in (1, 4, 8)
    }

    # 3. Dual vs single reagent (general mode, k=8). ---------------------------------
    dual = _auroc_for(dataset, labels, mode="general", k=8, n_layers=n_layers,
                      epochs=epochs, seed=seed)
    single = _auroc_for(_mask_dms(dataset), labels, mode="general", k=8,
                        n_layers=n_layers, epochs=epochs, seed=seed)
    reagent_auroc = {"dual_shape_dms": float(dual), "single_shape_only": float(single)}

    results = {
        "task": "ablations",
        "seed": int(seed),
        "n_samples": int(len(dataset)),
        "config": {"n_layers": int(n_layers), "epochs": int(epochs)},
        "mode_auroc": mode_auroc,
        "stalk_dim_auroc": k_auroc,
        "reagent_auroc": reagent_auroc,
    }

    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ablations.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    return results
