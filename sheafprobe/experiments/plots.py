"""Figures for SheafProbe.

Three PNGs, written to ``<results_dir>/figures/``:
  1. auroc_bar.png        : E* (sheaf general) vs the three baselines, from killer.json.
  2. ablation_curves.png  : restriction-map, stalk-dim and reagent ablations, from ablations.json.
  3. edge_energy_heatmap.png : per-candidate-edge sheaf energy on one multi-state molecule,
                               localising the competing-stem gluing obstruction.

The heatmap needs a live model + a sample, so a small SheafProbe(general) is quick-trained on a
tiny synthetic dataset inside this function (CPU-fast); the bar / curve figures only read JSON.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless / CPU-only; no display required
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load_json(results_dir: str, name: str) -> Optional[Dict]:
    path = os.path.join(results_dir, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _auroc_bar(killer: Dict, fig_dir: str) -> Optional[str]:
    """Bar chart: sheaf-general E* AUROC vs gold / identity / transformer."""
    auroc = killer.get("auroc", {})
    order = [
        ("SheafProbe\n(general)", "sheaf_general"),
        ("entropy-bpp\n(gold)", "entropy_bpp_gold"),
        ("SheafProbe\n(identity)", "sheaf_identity"),
        ("Transformer\nrecon", "transformer_recon"),
    ]
    def _as_float(v):
        return float(v["mean"]) if isinstance(v, dict) else float(v)

    names = [n for n, key in order if key in auroc]
    vals = [_as_float(auroc[key]) for _, key in order if key in auroc]
    if not vals:
        return None

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = ["#2a6f97"] + ["#9aa0a6"] * (len(vals) - 1)
    bars = ax.bar(names, vals, color=colors)
    # Bootstrap CI whisker on the sheaf-general bar if present.
    ci = auroc.get("sheaf_general_pooled_ci") or auroc.get("sheaf_general_ci")
    if ci and "sheaf_general" in auroc:
        mid = _as_float(auroc["sheaf_general"])
        ax.errorbar(0, mid, yerr=[[mid - ci[0]], [ci[1] - mid]], fmt="none",
                    ecolor="black", capsize=4, lw=1.2)
    ax.axhline(0.5, color="red", ls="--", lw=1, label="chance")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("AUROC (multi-state separation)")
    ax.set_title("E* separates multi-state RNAs vs baselines")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(fig_dir, "auroc_bar.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _ablation_curves(abl: Dict, fig_dir: str) -> Optional[str]:
    """Three-panel ablation figure: mode, stalk dim, reagent count."""
    mode_auroc = abl.get("mode_auroc", {})
    k_auroc = abl.get("stalk_dim_auroc", {})
    reagent = abl.get("reagent_auroc", {})
    if not (mode_auroc or k_auroc or reagent):
        return None

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))

    # Panel 1: restriction-map family.
    modes = [m for m in ("general", "diagonal", "identity") if m in mode_auroc]
    axes[0].bar(modes, [mode_auroc[m] for m in modes], color="#2a6f97")
    axes[0].set_title("Restriction map")
    axes[0].set_ylabel("AUROC")

    # Panel 2: stalk dimension (sorted numerically).
    ks = sorted(k_auroc.keys(), key=lambda s: int(s))
    axes[1].plot([int(k) for k in ks], [k_auroc[k] for k in ks],
                 marker="o", color="#2a6f97")
    axes[1].set_title("Stalk dimension k")
    axes[1].set_xlabel("k")

    # Panel 3: dual vs single reagent.
    rnames = [r for r in ("dual_shape_dms", "single_shape_only") if r in reagent]
    labels = {"dual_shape_dms": "dual\n(SHAPE+DMS)", "single_shape_only": "single\n(SHAPE)"}
    axes[2].bar([labels[r] for r in rnames], [reagent[r] for r in rnames],
                color=["#2a6f97", "#9aa0a6"])
    axes[2].set_title("Reagents")

    for ax in axes:
        ax.axhline(0.5, color="red", ls="--", lw=1)
        ax.set_ylim(0.0, 1.05)
    fig.suptitle("SheafProbe ablations (E* AUROC)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(fig_dir, "ablation_curves.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _edge_energy_heatmap(fig_dir: str, seed: int = 0) -> Optional[str]:
    """Per-edge sheaf energy on one multi-state example, drawn as an n x n contact map.

    Quick-trains a tiny SheafProbe(general) so the figure does not depend on a serialized
    model. Each candidate edge (i, j) is coloured by its energy; competing stems light up.
    """
    # Local imports keep matplotlib-only callers from pulling torch unnecessarily.
    from ..data import synthetic
    from ..models import sheaf

    dataset = synthetic.generate_dataset(n_samples=60, length=48, frac_multistate=0.5,
                                          noise=0.15, seed=seed)
    multistate = next((s for s in dataset if int(s.label_multistate) == 1), None)
    if multistate is None:
        return None

    model = sheaf.SheafProbe(k=4, n_layers=3, eps=0.3, mode="general")
    model = sheaf.train_sheafprobe(model, dataset, epochs=40, lr=1e-2, lam=1.0, seed=seed)
    energy = np.asarray(sheaf.per_edge_energy(model, multistate), dtype=np.float64).reshape(-1)

    n = int(multistate.n)
    edges = np.asarray(multistate.edges, dtype=np.int64).reshape(2, -1)
    mat = np.full((n, n), np.nan, dtype=np.float64)
    e = min(edges.shape[1], energy.shape[0])
    for idx in range(e):
        i, j = int(edges[0, idx]), int(edges[1, idx])
        mat[i, j] = energy[idx]
        mat[j, i] = energy[idx]  # symmetric for readability

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(mat, cmap="magma", origin="lower")
    ax.set_title(f"Per-edge E* on multi-state {multistate.id}\n(true_n_states="
                 f"{multistate.true_n_states})")
    ax.set_xlabel("nucleotide j")
    ax.set_ylabel("nucleotide i")
    fig.colorbar(im, ax=ax, label="edge energy", fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = os.path.join(fig_dir, "edge_energy_heatmap.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _holonomy_bar(holo: Dict, fig_dir: str) -> Optional[str]:
    """Part B: sheaf-with-correct-maps vs identity / node baselines (all at chance)."""
    auroc = holo.get("auroc", {})
    order = [
        ("Sheaf\n(correct maps)", "sheaf_correct_maps"),
        ("Sheaf\n(identity)", "sheaf_identity_maps"),
        ("Node\nnorm-var", "node_norm_variance"),
        ("Node\nangle-entropy", "node_angle_entropy"),
    ]
    names = [n for n, key in order if key in auroc]
    vals = [float(auroc[key]) for _, key in order if key in auroc]
    if not vals:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = ["#2a9d3f"] + ["#9aa0a6"] * (len(vals) - 1)
    bars = ax.bar(names, vals, color=colors)
    ax.axhline(0.5, color="red", ls="--", lw=1, label="chance")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("AUROC (holonomy-frustration detection)")
    ax.set_title("Part B: sheaf is necessary for a holonomy obstruction")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(fig_dir, "holonomy_bar.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def make_plots(results_dir) -> List[str]:
    """Render all available figures into ``<results_dir>/figures/``.

    Parameters
    ----------
    results_dir : str | os.PathLike
        Directory containing ``killer.json`` / ``ablations.json`` (i.e. the ``results`` dir).

    Returns
    -------
    list[str]
        Absolute paths of the PNGs successfully written.
    """
    results_dir = os.fspath(results_dir)
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    written: List[str] = []

    killer = _load_json(results_dir, "killer.json")
    if killer is not None:
        path = _auroc_bar(killer, fig_dir)
        if path:
            written.append(os.path.abspath(path))

    abl = _load_json(results_dir, "ablations.json")
    if abl is not None:
        path = _ablation_curves(abl, fig_dir)
        if path:
            written.append(os.path.abspath(path))

    holo = _load_json(results_dir, "holonomy.json")
    if holo is not None:
        path = _holonomy_bar(holo, fig_dir)
        if path:
            written.append(os.path.abspath(path))

    try:
        path = _edge_energy_heatmap(fig_dir)
        if path:
            written.append(os.path.abspath(path))
    except Exception as exc:  # heatmap is best-effort; never sink the whole figure run
        print(f"[plots] skipped edge-energy heatmap: {exc}")

    return written
