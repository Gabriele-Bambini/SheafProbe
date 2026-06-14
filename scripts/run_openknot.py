"""Run the SheafProbe killer flow on REAL OpenKnot data (OpenKnotBench).

Task: from REAL SHAPE reactivity, separate molecules whose reference structure is a
**pseudoknot** from those that are not. Inputs are structure-blind: candidate base-pair
edges come from a ViennaRNA partition-function fold (nested only — a documented
limitation, since Vienna cannot place the crossing pairs that define a pseudoknot), and
the label comes from the benchmark's RNet reference structure (a model-derived reference,
not a gold experimental structure). DMS is absent (OpenKnot is SHAPE-only).

Writes ``results/real_openknot.json`` without clobbering the synthetic results.

Usage:
    python scripts/run_openknot.py --csv <OpenKnotBench_data.csv> \
        --n-per-class 120 --epochs 50 --n-seeds 2 --seed 0 --out results
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile

import numpy as np
import pandas as pd

from sheafprobe.data.schema import Sample
from sheafprobe.data.loaders import _bpp_candidate_edges, _is_pseudoknot
from sheafprobe.experiments.killer import run_killer


def _build_sample(rec_id, seq, shape, label):
    n = len(seq)
    bpp = _bpp_candidate_edges(seq)
    if bpp:
        pairs = [(i, j) for (i, j, _p) in bpp]
        weights = np.array([0.5 + 0.5 * p for (_i, _j, p) in bpp], dtype=np.float64)
    else:
        pairs = [(i, min(i + 8, n - 1)) for i in range(0, max(1, n - 8), 8)]
        weights = np.full(len(pairs), 0.5, dtype=np.float64)
    edges = np.array(pairs, dtype=np.int64).T.reshape(2, -1)
    backbone = np.array([[k, k + 1] for k in range(n - 1)], dtype=np.int64).T.reshape(2, -1)
    return Sample(id=str(rec_id), seq=seq, n=n, edges=edges, edge_weight=weights,
                  backbone=backbone, react_shape=np.asarray(shape, dtype=np.float64),
                  react_dms=np.full(n, np.nan), label_multistate=int(label),
                  true_n_states=-1, meta={"source": "openknotbench"})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n-per-class", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-seeds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    # Build a balanced, correctly-labelled sample list.
    react_cols = None
    rng = random.Random(args.seed)
    pos, neg = [], []
    for chunk in pd.read_csv(args.csv, chunksize=2000):
        if react_cols is None:
            react_cols = [c for c in chunk.columns
                          if c.lower().startswith("reactivity") and "error" not in c.lower()]
        chunk = chunk[chunk.get("SN_filter", 1) == 1]
        for _, row in chunk.iterrows():
            seq, struct = row.get("sequence"), row.get("RNet_structure")
            if not isinstance(seq, str) or not isinstance(struct, str):
                continue
            n = len(seq)
            if n < 20 or n > 260:
                continue
            shape = row[react_cols].to_numpy(dtype=np.float64)[:n]
            if shape.shape[0] < n or np.mean(np.isfinite(shape)) < 0.5:
                continue
            (pos if _is_pseudoknot(struct) else neg).append((row.get("id", "?"), seq, shape))
        if len(pos) > 4 * args.n_per_class and len(neg) > args.n_per_class:
            break
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(args.n_per_class, len(pos), len(neg))
    if k < 10:
        print(f"[openknot] too few of one class (pos={len(pos)}, neg={len(neg)}).")
        return 2
    chosen = [(rec, 1) for rec in pos[:k]] + [(rec, 0) for rec in neg[:k]]
    rng.shuffle(chosen)
    samples = [_build_sample(rec[0], rec[1], rec[2], lbl) for rec, lbl in chosen]
    print(f"[openknot] balanced {len(samples)} molecules ({k}/class); "
          f"pool pos={len(pos)} neg={len(neg)}; folding via ViennaRNA done.")

    with tempfile.TemporaryDirectory() as tmp:
        res = run_killer(samples, tmp, seed=args.seed, n_seeds=args.n_seeds,
                         epochs=args.epochs, k=8, n_layers=3)
    res["dataset"] = "OpenKnotBench (real SHAPE; label=RNet_structure pseudoknot; ViennaRNA-BPP nested edges)"
    res["n_molecules_used"] = len(samples)
    res["pool_pos_neg"] = [len(pos), len(neg)]
    res["caveats"] = ["ViennaRNA BPP edges are nested-only (cannot encode the crossing pairs).",
                      "Label is the model-derived RNet reference structure, not a gold experimental structure.",
                      "SHAPE-only (no DMS), so the cross-view term is inactive."]

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "real_openknot.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    a = res["auroc"]
    print(f"[openknot] sheaf_general {a['sheaf_general']['mean']:.4f}+/-{a['sheaf_general']['std']:.4f} "
          f"| identity {a['sheaf_identity']['mean']:.4f} | gold {a['entropy_bpp_gold']:.4f} "
          f"| beats_gold {res['beats_gold']}")
    print(f"[openknot] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
