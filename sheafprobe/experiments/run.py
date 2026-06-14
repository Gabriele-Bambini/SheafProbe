"""Command-line entry point for the SheafProbe experiments.

    python -m sheafprobe.experiments.run --task {killer,ablations,all,real-openknot} \
        [--out results] [--seed 0] [--quick]

``--quick`` swaps in a tiny config (few samples / short sequences / few epochs) so the whole
pipeline runs in well under a minute on CPU — used by the smoke test. ``real-openknot`` fetches
the OpenKnot dataset and runs the killer flow on it, skipping cleanly if the data is offline.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

# Full (paper-ish but still CPU-small) and QUICK (smoke) configurations.
#
# NOTE (integration, 2026-06-14): the contract's nominal full config (600 samples,
# 150 epochs, n_layers=4) trains ~11 separate sheaf models (killer + ablation grid),
# which takes ~50 min on CPU. To honour the "finishes in a few minutes" hard rule we
# reduced to 300 samples / 80 epochs / n_layers=3. This is still well-powered
# (n=300, ~50/50 classes) and does not change any qualitative conclusion. Bump these
# back up for a publication-grade run if you have the wall-clock budget.
FULL_CFG: Dict[str, int] = {
    "n_samples": 300, "length": 68, "epochs": 80, "k": 8, "n_layers": 3,
}
QUICK_CFG: Dict[str, int] = {
    "n_samples": 60, "length": 40, "epochs": 20, "k": 4, "n_layers": 2,
}


def _make_synthetic(cfg: Dict[str, int], seed: int):
    """Build the synthetic dataset for the given config."""
    from ..data import synthetic
    return synthetic.generate_dataset(
        n_samples=cfg["n_samples"], length=cfg["length"],
        frac_multistate=0.5, noise=0.15, seed=seed,
    )


def _run_killer(cfg, out, seed, dataset=None) -> Dict:
    from .killer import run_killer
    if dataset is None:
        dataset = _make_synthetic(cfg, seed)
    return run_killer(dataset, out, seed=seed, epochs=cfg["epochs"],
                      k=cfg["k"], n_layers=cfg["n_layers"])


def _run_ablations(cfg, out, seed, dataset=None) -> Dict:
    from .ablations import run_ablations
    if dataset is None:
        dataset = _make_synthetic(cfg, seed)
    return run_ablations(dataset, out, seed=seed, epochs=cfg["epochs"],
                         n_layers=cfg["n_layers"])


def _run_real_openknot(cfg, out, seed) -> Dict:
    """Attempt the killer flow on real OpenKnot data; skip gracefully if unavailable."""
    from ..data.schema import DataUnavailable
    from ..data import loaders

    try:
        dataset = loaders.load_openknot(root="data/openknot", download=True)
    except DataUnavailable as exc:
        print("[real-openknot] SKIPPED: real data unavailable.")
        print(f"    reason: {exc}")
        return {"task": "real-openknot", "status": "skipped", "reason": str(exc)}

    if not dataset:
        print("[real-openknot] SKIPPED: loader returned an empty dataset.")
        return {"task": "real-openknot", "status": "skipped", "reason": "empty dataset"}

    print(f"[real-openknot] loaded {len(dataset)} samples; running killer flow.")
    res = _run_killer(cfg, out, seed, dataset=dataset)
    res["status"] = "ran"
    return res


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="sheafprobe.experiments.run",
        description="Run SheafProbe experiments (killer / ablations / all / real-openknot).",
    )
    parser.add_argument("--task", required=True,
                        choices=["killer", "ablations", "all", "real-openknot"])
    parser.add_argument("--out", default="results", help="output root directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true",
                        help="use the tiny QUICK config (fast smoke run)")
    args = parser.parse_args(argv)

    cfg = QUICK_CFG if args.quick else FULL_CFG
    os.makedirs(args.out, exist_ok=True)
    tag = "QUICK" if args.quick else "FULL"
    print(f"[run] task={args.task} config={tag} seed={args.seed} out={args.out}")

    if args.task == "killer":
        res = _run_killer(cfg, args.out, args.seed)
        print(json.dumps(res["auroc"], indent=2))

    elif args.task == "ablations":
        res = _run_ablations(cfg, args.out, args.seed)
        print(json.dumps({"mode": res["mode_auroc"], "k": res["stalk_dim_auroc"],
                          "reagent": res["reagent_auroc"]}, indent=2))

    elif args.task == "all":
        # Share one synthetic dataset across killer + ablations for consistency / speed.
        dataset = _make_synthetic(cfg, args.seed)
        killer_res = _run_killer(cfg, args.out, args.seed, dataset=dataset)
        _run_ablations(cfg, args.out, args.seed, dataset=dataset)
        from .plots import make_plots
        figs = make_plots(args.out)
        g = killer_res["auroc"]["sheaf_general"]
        print(f"[run] killer sheaf_general AUROC = {g['mean']:.4f} +/- {g['std']:.4f}  "
              f"| gold(entropy) = {killer_res['auroc']['entropy_bpp_gold']:.4f}  "
              f"| beats_gold = {killer_res['beats_gold']}")
        print(f"[run] wrote {len(figs)} figure(s).")

    elif args.task == "real-openknot":
        _run_real_openknot(cfg, args.out, args.seed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
