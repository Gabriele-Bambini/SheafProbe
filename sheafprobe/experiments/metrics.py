"""Evaluation metrics for SheafProbe.

AUROC for multistate separation, a bootstrap CI on that AUROC, a partial
Spearman correlation that regresses out nuisance columns, and the full
confound battery report. The confound report is the load-bearing scientific
check: it asks whether `E*` still tracks the multistate label *after* removing
trivial explanations (mean unpaired probability, DMS missingness, edge count,
length).

Pure numpy / scipy / sklearn. No torch needed here.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


# ----------------------------------------------------------------------------
# AUROC + bootstrap CI
# ----------------------------------------------------------------------------
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via sklearn.

    Args:
        scores: higher = more likely to be the positive (multistate) class.
        labels: binary {0,1}.

    Returns:
        AUROC in [0,1]; returns nan if only one class is present.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def bootstrap_auroc_ci(scores: np.ndarray, labels: np.ndarray, n: int = 1000,
                       seed: int = 0) -> Tuple[float, float, float]:
    """Bootstrap percentile confidence interval for AUROC.

    Args:
        scores: prediction scores (higher = positive class).
        labels: binary {0,1}.
        n:      number of bootstrap resamples.
        seed:   RNG seed.

    Returns:
        (point_auroc, ci_lo_2.5%, ci_hi_97.5%).
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    point = auroc(scores, labels)

    rng = np.random.default_rng(seed)
    m = len(scores)
    boots = np.empty(n, dtype=np.float64)
    valid = 0
    for b in range(n):
        idx = rng.integers(0, m, size=m)
        sl, ll = scores[idx], labels[idx]
        if len(np.unique(ll)) < 2:
            continue
        boots[valid] = roc_auc_score(ll, sl)
        valid += 1

    if valid == 0:
        return (point, float("nan"), float("nan"))
    boots = boots[:valid]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (point, float(lo), float(hi))


# ----------------------------------------------------------------------------
# Partial correlation (Spearman on residuals after regressing out controls)
# ----------------------------------------------------------------------------
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank transform (Spearman uses ranks)."""
    return stats.rankdata(a)


def _regress_out(target: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Residual of `target` after least-squares regression on `controls` (+intercept)."""
    if controls.size == 0:
        return target - target.mean()
    X = np.column_stack([np.ones(len(target)), controls])
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    return target - X @ beta


def partial_corr(x: np.ndarray, y: np.ndarray,
                 controls: np.ndarray) -> Tuple[float, float]:
    """Partial Spearman correlation of x, y controlling for `controls`.

    Spearman is implemented as Pearson on rank-transformed variables; control
    columns (also rank-transformed for a rank-based partialling) are regressed
    out of both x and y via least squares, then the residuals are correlated.

    Args:
        x:        [m] variable of interest (e.g. E*).
        y:        [m] target (e.g. label).
        controls: [m] or [m, c] nuisance columns to partial out.

    Returns:
        (partial_spearman_rho, two_sided_p_value).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    controls = np.asarray(controls, dtype=np.float64)
    if controls.ndim == 1:
        controls = controls.reshape(-1, 1)
    if controls.shape[0] != len(x):  # allow [c, m] passed transposed
        controls = controls.T

    m = len(x)
    xr = _rankdata(x)
    yr = _rankdata(y)
    if controls.size == 0:
        cr = np.empty((m, 0))
    else:
        # drop degenerate (zero-variance) control columns to keep lstsq stable
        keep = [c for c in range(controls.shape[1])
                if np.std(controls[:, c]) > 1e-12]
        cr = np.column_stack([_rankdata(controls[:, c]) for c in keep]) \
            if keep else np.empty((m, 0))

    rx = _regress_out(xr, cr)
    ry = _regress_out(yr, cr)

    sx, sy = np.std(rx), np.std(ry)
    if sx < 1e-12 or sy < 1e-12:
        return (0.0, 1.0)

    rho = float(np.corrcoef(rx, ry)[0, 1])
    rho = max(min(rho, 1.0), -1.0)

    # t-based p-value with dof reduced by number of controls.
    dof = m - 2 - cr.shape[1]
    if dof <= 0:
        return (rho, float("nan"))
    denom = max(1.0 - rho * rho, 1e-12)
    t = rho * np.sqrt(dof / denom)
    p = 2.0 * stats.t.sf(abs(t), dof)
    return (rho, float(p))


# ----------------------------------------------------------------------------
# Confound battery
# ----------------------------------------------------------------------------
# Confounds the contract requires E* to survive.
CONFOUND_KEYS = ("mean_unpaired_prob", "frac_nan_dms", "n_candidate_edges", "length")


def confound_report(estar: np.ndarray, labels: np.ndarray,
                    confound_dict: Dict[str, np.ndarray]) -> dict:
    """Confound battery for the heterogeneity energy `E*`.

    Reports the bare AUROC of E* vs the multistate label, then the partial
    Spearman correlation of E* vs label after regressing out each confound
    individually and all confounds jointly. A real signal survives the joint
    control; a trivial one collapses to ~0.

    Args:
        estar:         [m] per-sample sheaf energy.
        labels:        [m] binary multistate label.
        confound_dict: maps confound name -> [m] array. Recognized keys:
                       mean_unpaired_prob, frac_nan_dms, n_candidate_edges, length.

    Returns:
        json-able dict with bare AUROC + per-confound and joint partial corr.
    """
    estar = np.asarray(estar, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(int)

    auc, lo, hi = bootstrap_auroc_ci(estar, labels, n=1000, seed=0)

    present = [k for k in CONFOUND_KEYS if k in confound_dict]
    raw_rho, raw_p = partial_corr(estar, labels, np.empty((len(estar), 0)))

    per_confound = {}
    for k in present:
        col = np.asarray(confound_dict[k], dtype=np.float64).ravel()
        rho, p = partial_corr(estar, labels, col)
        per_confound[k] = {"partial_spearman": rho, "p": p}

    if present:
        joint_controls = np.column_stack(
            [np.asarray(confound_dict[k], dtype=np.float64).ravel() for k in present]
        )
        j_rho, j_p = partial_corr(estar, labels, joint_controls)
    else:
        j_rho, j_p = raw_rho, raw_p

    return {
        "n_samples": int(len(estar)),
        "bare_auroc": auc,
        "auroc_ci_lo": lo,
        "auroc_ci_hi": hi,
        "spearman_estar_label": {"rho": raw_rho, "p": raw_p},
        "confounds_used": list(present),
        "partial_corr_per_confound": per_confound,
        "partial_corr_joint": {"partial_spearman": j_rho, "p": j_p},
    }
