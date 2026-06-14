"""Smoke test: the whole experiment pipeline runs end-to-end on the QUICK config and
produces result JSONs whose AUROC values are finite numbers in [0, 1].

Kept CPU-tiny so it finishes in seconds. It exercises the public CLI entry point
(`run.main`) with ``--task all --quick``, then validates the artefacts on disk.
"""
from __future__ import annotations

import json
import math
import os

from sheafprobe.experiments.run import main


def _finite_unit(x) -> bool:
    # AUROC entries are either a float or a {"mean", "std", "per_seed"} dict (multi-seed).
    if isinstance(x, dict):
        x = x.get("mean")
    return isinstance(x, (int, float)) and math.isfinite(x) and 0.0 <= float(x) <= 1.0


def test_pipeline_all_quick(tmp_path):
    out = str(tmp_path)
    rc = main(["--task", "all", "--out", out, "--seed", "0", "--quick"])
    assert rc == 0

    results_dir = out
    killer_path = os.path.join(results_dir, "killer.json")
    abl_path = os.path.join(results_dir, "ablations.json")

    assert os.path.exists(killer_path), "killer.json was not written"
    assert os.path.exists(abl_path), "ablations.json was not written"

    with open(killer_path, encoding="utf-8") as fh:
        killer = json.load(fh)
    auroc = killer["auroc"]
    for key in ("sheaf_general", "sheaf_identity", "entropy_bpp_gold", "transformer_recon"):
        assert key in auroc, f"missing AUROC entry: {key}"
        assert _finite_unit(auroc[key]), f"AUROC {key}={auroc[key]} not a finite value in [0,1]"
    lo, hi = killer["auroc"]["sheaf_general_pooled_ci"]
    assert _finite_unit(lo) and _finite_unit(hi) and lo <= hi
    assert isinstance(killer["confound_report"], dict) and killer["confound_report"]

    with open(abl_path, encoding="utf-8") as fh:
        abl = json.load(fh)
    for cell in abl["mode_auroc"].values():
        assert _finite_unit(cell)
    for cell in abl["stalk_dim_auroc"].values():
        assert _finite_unit(cell)
    for cell in abl["reagent_auroc"].values():
        assert _finite_unit(cell)


def test_figures_written(tmp_path):
    """`make_plots` should emit at least one PNG once result JSONs exist."""
    out = str(tmp_path)
    main(["--task", "all", "--out", out, "--seed", "0", "--quick"])

    from sheafprobe.experiments.plots import make_plots
    figs = make_plots(out)
    assert figs, "no figures were produced"
    for path in figs:
        assert os.path.exists(path) and path.endswith(".png")
