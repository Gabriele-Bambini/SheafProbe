"""Real-data loaders for SheafProbe.

These parse genuine chemical-mapping datasets into the frozen :class:`Sample` type.
They NEVER fabricate data: if a dataset cannot be obtained they raise
:class:`DataUnavailable` carrying the URL / Kaggle slug so the CLI can skip
gracefully.

* ``load_openknot``  : no-auth GitHub fetch of eternagame/OpenKnotAIDesignData
                       (git clone, else raw-file download via ``requests``).
* ``load_ribonanza`` : Kaggle (needs creds); parsed only if files already present.
* ``load_openvaccine``: Kaggle; parsed only if files already present.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
from typing import Dict, List, Optional

import numpy as np

from .schema import Dataset, DataUnavailable, Sample, base_mask

# ---------------------------------------------------------------------------
# OpenKnot (GitHub, no auth)
# ---------------------------------------------------------------------------
OPENKNOT_REPO = "https://github.com/eternagame/OpenKnotAIDesignData"
OPENKNOT_GIT = OPENKNOT_REPO + ".git"
OPENKNOT_RAW = "https://raw.githubusercontent.com/eternagame/OpenKnotAIDesignData/main/"

# Brackets that denote a pseudoknot / crossing pair in dot-bracket notation.
_PK_BRACKETS = set("[]{}<>AaBbCcDd")
_NEST_OPEN = "("
_NEST_CLOSE = ")"


def _candidate_edges_from_structure(structure: str) -> List[tuple]:
    """Parse a (possibly multi-bracket) dot-bracket string into base pairs.

    Handles nested ``()`` plus pseudoknot bracket families ``[] {} <>`` and the
    letter convention ``Aa Bb ...``. Returns a list of (i, j) with i < j.
    """
    pairs: List[tuple] = []
    # Map each closing symbol to its opening symbol, per bracket family.
    families = ["()", "[]", "{}", "<>"]
    # Letter families: uppercase = open, matching lowercase = close.
    for L in "ABCD":
        families.append(L + L.lower())
    open_to_close = {f[0]: f[1] for f in families}
    close_to_open = {f[1]: f[0] for f in families}
    stacks: Dict[str, List[int]] = {o: [] for o in open_to_close}
    for k, ch in enumerate(structure):
        if ch in open_to_close:
            stacks[ch].append(k)
        elif ch in close_to_open:
            o = close_to_open[ch]
            if stacks.get(o):
                i = stacks[o].pop()
                pairs.append((i, k) if i < k else (k, i))
    return pairs


def _is_pseudoknot(structure: str) -> int:
    """Return 1 if the dot-bracket string contains any pseudoknot brackets, else 0."""
    return int(any(c in _PK_BRACKETS for c in structure))


def _bpp_candidate_edges(seq: str, min_prob: float = 0.05,
                         max_per_nt: float = 2.0) -> Optional[List[tuple]]:
    """STRUCTURE-BLIND candidate base pairs from a ViennaRNA partition-function fold.

    Returns (i, j, prob) pairs whose base-pair probability exceeds ``min_prob``,
    capped at ``max_per_nt * n`` strongest pairs. Returns ``None`` if ViennaRNA is
    not installed (caller falls back). This is the key anti-leakage step: candidate
    edges come from sequence alone, never from the experimental structure that also
    defines the label.
    """
    try:
        import RNA  # ViennaRNA
    except ImportError:
        return None
    n = len(seq)
    fc = RNA.fold_compound(seq.replace("T", "U"))
    fc.pf()
    bpp = fc.bpp()  # 1-indexed [n+1][n+1] upper triangle
    out = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            p = bpp[i][j]
            if p > min_prob:
                out.append((i - 1, j - 1, float(p)))
    out.sort(key=lambda t: t[2], reverse=True)
    cap = int(max_per_nt * n)
    return out[:cap] if cap > 0 else out


def _sample_from_record(rec_id: str, seq: str,
                        shape: np.ndarray,
                        structure: Optional[str]) -> Optional[Sample]:
    """Build a :class:`Sample` from a parsed OpenKnot-style record (SHAPE only).

    Candidate base-pair edges come from a STRUCTURE-BLIND ViennaRNA base-pair-
    probability fold (no leakage); the experimental ``structure`` is used ONLY to set
    ``label_multistate`` (pseudoknot vs not). DMS is all-nan (single-reagent set).
    Returns ``None`` if the record is unusable (empty / length mismatch).
    """
    n = len(seq)
    if n < 4 or shape.shape[0] != n:
        return None
    structure = structure if structure else ""

    bpp_pairs = _bpp_candidate_edges(seq)
    if bpp_pairs:
        pairs = [(i, j) for (i, j, _p) in bpp_pairs]
        weights = np.array([0.5 + 0.5 * p for (_i, _j, p) in bpp_pairs], dtype=np.float64)
    else:
        # ViennaRNA unavailable: structure-blind backbone-distant decoys (low weight).
        pairs = [(i, min(i + 8, n - 1)) for i in range(0, max(1, n - 8), 8)]
        weights = np.full(len(pairs), 0.5, dtype=np.float64)
    if not pairs:
        pairs = [(0, min(3, n - 1))]
        weights = np.array([0.5], dtype=np.float64)

    edges = np.array(pairs, dtype=np.int64).T.reshape(2, -1)
    backbone = np.array([[k, k + 1] for k in range(n - 1)], dtype=np.int64).T.reshape(2, -1)
    react_dms = np.full(n, np.nan, dtype=np.float64)  # OpenKnot = SHAPE-only
    label = _is_pseudoknot(structure)
    return Sample(
        id=str(rec_id),
        seq=seq,
        n=n,
        edges=edges,
        edge_weight=weights,
        backbone=backbone,
        react_shape=np.asarray(shape, dtype=np.float64),
        react_dms=react_dms,
        label_multistate=int(label),
        true_n_states=-1,
        meta={"source": "openknot", "has_structure": bool(structure)},
    )


def _coerce_shape(value) -> Optional[np.ndarray]:
    """Coerce a JSON/csv reactivity field into a float array, nan for blanks."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p for p in value.replace(";", ",").split(",") if p.strip() != ""]
        try:
            return np.array([float(p) for p in parts], dtype=np.float64)
        except ValueError:
            return None
    if isinstance(value, (list, tuple)):
        out = []
        for p in value:
            try:
                out.append(float(p))
            except (TypeError, ValueError):
                out.append(np.nan)
        return np.array(out, dtype=np.float64)
    return None


def _parse_openknot_dir(root: str) -> Dataset:
    """Parse whatever SHAPE/structure files exist under ``root`` into Samples.

    Tolerant of layout: scans for ``*.json``/``*.csv`` and pulls the first
    sequence + reactivity + (optional) structure columns it recognises. Designed to
    extract *some* usable molecules rather than to perfectly mirror one schema.
    """
    samples: Dataset = []
    seq_keys = ("sequence", "seq")
    shape_keys = ("reactivity", "shape", "shape_reactivity", "data")
    # Real OpenKnot CSVs annotate structure under several columns. The label we want
    # is whether the molecule *experimentally* folds into a pseudoknot, which lives in
    # ``M2_structure`` (the chemical-mapping-derived structure). ``target_structure`` is
    # the DESIGN intent (a pseudoknot for every row -> useless single-class label), so we
    # deliberately prefer M2_structure / generic structure columns and avoid target_*.
    struct_keys = ("m2_structure", "structure", "secstruct", "dotbracket", "dot_bracket")

    json_files = glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
        except (OSError, ValueError):
            continue
        records = blob if isinstance(blob, list) else [blob]
        for ri, rec in enumerate(records):
            if not isinstance(rec, dict):
                continue
            low = {k.lower(): v for k, v in rec.items()}
            seq = next((low[k] for k in seq_keys if k in low), None)
            shape_raw = next((low[k] for k in shape_keys if k in low), None)
            struct = next((low[k] for k in struct_keys if k in low), None)
            if not isinstance(seq, str):
                continue
            shape = _coerce_shape(shape_raw)
            if shape is None:
                shape = np.full(len(seq), np.nan, dtype=np.float64)
            rec_id = low.get("id", f"{os.path.basename(fp)}:{ri}")
            s = _sample_from_record(rec_id, seq, shape,
                                    struct if isinstance(struct, str) else None)
            if s is not None:
                samples.append(s)

    # CSV fallback (pandas only if csv files exist, to avoid a hard import cost).
    csv_files = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    if csv_files and not samples:
        try:
            import pandas as pd
        except ImportError:
            pd = None
        if pd is not None:
            for fp in csv_files:
                try:
                    df = pd.read_csv(fp)
                except (OSError, ValueError):
                    continue
                cols = {c.lower(): c for c in df.columns}
                seq_col = next((cols[k] for k in seq_keys if k in cols), None)
                if seq_col is None:
                    continue
                # SHAPE = the reactivity_NNNN columns, in file order.
                shape_cols = [c for c in df.columns
                              if c.lower().startswith("reactivity")
                              and "error" not in c.lower()]
                struct_col = next((cols[k] for k in struct_keys if k in cols), None)
                for ri, row in df.iterrows():
                    seq = row[seq_col]
                    if not isinstance(seq, str) or len(seq) < 4:
                        continue
                    n = len(seq)
                    if shape_cols:
                        shape = row[shape_cols].to_numpy(dtype=np.float64)
                        # Reactivity vectors are padded to a fixed width; trim/pad to n.
                        if shape.shape[0] >= n:
                            shape = shape[:n]
                        else:
                            shape = np.concatenate(
                                [shape, np.full(n - shape.shape[0], np.nan)])
                    else:
                        shape = np.full(n, np.nan, dtype=np.float64)
                    # Skip rows whose SHAPE is entirely missing: they carry no signal
                    # for E* and would only dilute the real-data evaluation.
                    if not np.any(np.isfinite(shape)):
                        continue
                    struct = row[struct_col] if struct_col else None
                    struct = struct if isinstance(struct, str) else None
                    s = _sample_from_record(f"{os.path.basename(fp)}:{ri}", seq, shape,
                                            struct)
                    if s is None:
                        continue
                    if struct is not None:
                        s.meta["label_source"] = struct_col
                    samples.append(s)
    return samples


def load_openknot(root: str = "data/openknot", download: bool = True) -> Dataset:
    """Load the OpenKnot AI design data (eternagame/OpenKnotAIDesignData) as Samples.

    Attempts a no-auth GitHub fetch when ``download`` is True and ``root`` is empty:
    first ``git clone``, then a best-effort raw-file download via ``requests``. SHAPE
    is taken as ``react_shape``; ``react_dms`` is all-nan (single-reagent set);
    ``label_multistate`` is set from the pseudoknot annotation when available.

    Raises:
        DataUnavailable: if the data cannot be obtained or parsed (offline / changed
        layout), with the repository URL in the message.
    """
    have_files = os.path.isdir(root) and any(
        glob.glob(os.path.join(root, "**", ext), recursive=True)
        for ext in ("*.json", "*.csv")
    )
    if not have_files and download:
        os.makedirs(os.path.dirname(root) or ".", exist_ok=True)
        cloned = _try_git_clone(OPENKNOT_GIT, root)
        if not cloned:
            _try_requests_download(root)

    if os.path.isdir(root):
        try:
            samples = _parse_openknot_dir(root)
        except Exception as exc:  # parsing should never hard-crash the pipeline
            raise DataUnavailable(
                f"OpenKnot present at {root!r} but could not be parsed ({exc}). "
                f"See {OPENKNOT_REPO}"
            ) from exc
        if samples:
            return samples

    raise DataUnavailable(
        "OpenKnot data unavailable (offline or empty after fetch). "
        f"Clone manually: git clone {OPENKNOT_GIT} {root}"
    )


def _try_git_clone(git_url: str, dest: str) -> bool:
    """Attempt a shallow ``git clone``; return True on success, False otherwise."""
    if os.path.isdir(dest) and os.listdir(dest):
        return True
    try:
        res = subprocess.run(
            ["git", "clone", "--depth", "1", git_url, dest],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        )
        return res.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _try_requests_download(dest: str) -> None:
    """Best-effort fetch of a few candidate raw files via ``requests`` (no auth)."""
    try:
        import requests
    except ImportError:
        return
    os.makedirs(dest, exist_ok=True)
    # Probe a small set of plausible top-level data file names.
    candidates = [
        "data.json", "dataset.json", "openknot.json",
        "data.csv", "dataset.csv", "README.md",
    ]
    for name in candidates:
        url = OPENKNOT_RAW + name
        try:
            r = requests.get(url, timeout=30)
        except requests.RequestException:
            continue
        if r.status_code == 200 and r.content:
            with open(os.path.join(dest, name), "wb") as fh:
                fh.write(r.content)


# ---------------------------------------------------------------------------
# Ribonanza & OpenVaccine (Kaggle; parse only if files already present)
# ---------------------------------------------------------------------------
RIBONANZA_SLUG = "stanford-ribonanza-rna-folding"
OPENVACCINE_SLUG = "stanford-covid-vaccine"


def _kaggle_hint(slug: str) -> str:
    """Standard ``kaggle datasets download`` hint string for a competition slug."""
    return (f"Kaggle dataset/competition slug: {slug}. "
            f"Download with: kaggle competitions download -c {slug}  "
            f"(or: kaggle datasets download {slug}). Requires Kaggle credentials.")


def _csv_paths(root: str) -> List[str]:
    """All CSV file paths under ``root`` (recursive)."""
    return glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)


def _parse_two_reagent_csv(fp: str, source: str) -> Dataset:
    """Parse a Ribonanza/OpenVaccine-style CSV into Samples (SHAPE + DMS where present).

    Pulls a ``sequence`` column plus any ``reactivity*`` (SHAPE) and ``deg_*`` /
    ``dms`` columns. Missing values become ``np.nan``. ``label_multistate`` is taken
    from a pseudoknot/structure annotation if present, else 0.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise DataUnavailable(
            f"{source}: pandas required to parse {fp}. {_kaggle_hint(source)}"
        ) from exc
    df = pd.read_csv(fp)
    cols = {c.lower(): c for c in df.columns}
    seq_col = cols.get("sequence") or cols.get("seq")
    if seq_col is None:
        return []
    shape_cols = [c for c in df.columns
                  if c.lower().startswith("reactivity") and "error" not in c.lower()]
    dms_cols = [c for c in df.columns if c.lower().startswith("dms")]
    struct_col = cols.get("structure") or cols.get("secstruct")

    samples: Dataset = []
    for ri, row in df.iterrows():
        seq = row[seq_col]
        if not isinstance(seq, str) or len(seq) < 4:
            continue
        n = len(seq)
        shape = (row[shape_cols].to_numpy(dtype=np.float64)[:n]
                 if shape_cols else np.full(n, np.nan))
        if shape.shape[0] < n:
            shape = np.concatenate([shape, np.full(n - shape.shape[0], np.nan)])
        if dms_cols:
            dms = row[dms_cols].to_numpy(dtype=np.float64)[:n]
            if dms.shape[0] < n:
                dms = np.concatenate([dms, np.full(n - dms.shape[0], np.nan)])
            # Enforce DMS observed only on A/C, per the contract's reagent semantics.
            dms = np.where(base_mask(seq, "AC"), dms, np.nan)
        else:
            dms = np.full(n, np.nan)
        struct = row[struct_col] if struct_col else None
        pairs = (_candidate_edges_from_structure(struct)
                 if isinstance(struct, str) else [])
        if not pairs:
            pairs = [(i, min(i + 8, n - 1)) for i in range(0, max(1, n - 8), 8)]
            weights = np.full(len(pairs), 0.5, dtype=np.float64)
            label = 0
        else:
            weights = np.full(len(pairs), 0.9, dtype=np.float64)
            label = _is_pseudoknot(struct) if isinstance(struct, str) else 0
        edges = np.array(pairs, dtype=np.int64).T.reshape(2, -1)
        backbone = np.array([[k, k + 1] for k in range(n - 1)],
                            dtype=np.int64).T.reshape(2, -1)
        samples.append(Sample(
            id=f"{source}:{ri}",
            seq=seq, n=n, edges=edges, edge_weight=weights, backbone=backbone,
            react_shape=np.asarray(shape, dtype=np.float64),
            react_dms=np.asarray(dms, dtype=np.float64),
            label_multistate=int(label), true_n_states=-1,
            meta={"source": source},
        ))
    return samples


def load_ribonanza(root: str = "data/ribonanza") -> Dataset:
    """Load the Stanford Ribonanza RNA-folding data if present at ``root``.

    Raises:
        DataUnavailable: if no CSV files are found at ``root`` (needs Kaggle creds),
        with the exact slug and a ``kaggle ... download`` hint.
    """
    csvs = _csv_paths(root)
    if not csvs:
        raise DataUnavailable(_kaggle_hint(RIBONANZA_SLUG))
    samples: Dataset = []
    for fp in csvs:
        samples.extend(_parse_two_reagent_csv(fp, RIBONANZA_SLUG))
    if not samples:
        raise DataUnavailable(
            f"Ribonanza CSVs at {root!r} had no parseable records. {_kaggle_hint(RIBONANZA_SLUG)}"
        )
    return samples


def load_openvaccine(root: str = "data/openvaccine") -> Dataset:
    """Load the Stanford OpenVaccine (COVID mRNA) data if present at ``root``.

    Raises:
        DataUnavailable: if no CSV files are found at ``root`` (needs Kaggle creds),
        with the exact slug and a ``kaggle ... download`` hint.
    """
    csvs = _csv_paths(root)
    if not csvs:
        raise DataUnavailable(_kaggle_hint(OPENVACCINE_SLUG))
    samples: Dataset = []
    for fp in csvs:
        samples.extend(_parse_two_reagent_csv(fp, OPENVACCINE_SLUG))
    if not samples:
        raise DataUnavailable(
            f"OpenVaccine CSVs at {root!r} had no parseable records. {_kaggle_hint(OPENVACCINE_SLUG)}"
        )
    return samples
