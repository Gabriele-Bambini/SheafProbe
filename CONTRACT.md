# SheafProbe — Build Contract (frozen interfaces)

**Goal:** a runnable, GitHub-ready pipeline demonstrating that the **sheaf Dirichlet
(inconsistency) energy `E*`** of an mRNA, computed from two distinct chemical-probing
views (SHAPE + DMS), separates **multi-state / pseudoknotted** molecules from
**single-state** ones — *by construction*, beating a non-learned gold-standard
(entropy-of-pairing) and a plain graph-diffusion baseline. CPU-only. Small.

Repo root: `G:\My Drive\progetti GNN\SheafProbe`. Package: `sheafprobe`.

## Thesis encoded in code
- Two reagents = two **distinct linear projections** of one latent pairing state.
  `react_shape` ≈ all-four-base backbone flexibility; `react_dms` ≈ A/C Watson–Crick only.
- A single rigid molecule = a globally consistent **0-cochain** (section) of a cellular
  sheaf on the nucleotide graph → low Dirichlet energy.
- A multi-state molecule has **two competing candidate pairings present as edges**; no
  single section glues both → **irreducible Dirichlet energy `E*` stays high**. `E*` IS the
  heterogeneity signal. A plain GNN (identity restriction maps) cannot represent the
  bond-frame transport, so its energy does not separate the classes as well.

## Frozen data type (already written): `sheafprobe/data/schema.py`
```python
@dataclass
class Sample:
    id: str
    seq: str                 # RNA over {A,C,G,U}
    n: int
    edges: np.ndarray        # [2,E] int64 candidate base-pair edges (i<j)
    edge_weight: np.ndarray  # [E] float in (0,1], BPP-like prior
    backbone: np.ndarray     # [2,n-1] int64 (i,i+1) edges
    react_shape: np.ndarray  # [n] float; np.nan where missing
    react_dms: np.ndarray    # [n] float; np.nan where missing (typically non-A/C)
    label_multistate: int    # 1 multi-state/pseudoknot, 0 single-state (VALIDATION label)
    true_n_states: int       # ground-truth #states (synthetic); -1 if unknown
    meta: dict
Dataset = List[Sample]
```
Helper in schema.py: `base_mask(seq, bases="AC") -> np.ndarray[bool]` (per-position).

## Module APIs (each agent implements exactly these signatures)

### `sheafprobe/data/synthetic.py`  (Agent A1)
```python
def generate_dataset(n_samples=600, length=68, frac_multistate=0.5, noise=0.15,
                     seed=0) -> Dataset
```
Construction rules (CRITICAL — this is what makes the experiment valid):
- Single-state sample: sample ONE nested pairing (a few stems). Candidate `edges` =
  the true stem pairs + a few decoys. Paired positions → LOW reactivity, unpaired → HIGH,
  plus Gaussian `noise`. `react_dms` = same latent but observed only where base ∈ {A,C}
  (else np.nan); `react_shape` observed on all bases. `label_multistate=0`, `true_n_states=1`.
- Multi-state sample: sample TWO mutually-incompatible pairings (e.g. a hairpin vs a
  pseudoknot-style crossing). BOTH pairings' pairs go into candidate `edges`. Reactivity =
  a 50/50 population average of the two states (so each competing stem looks "half-paired").
  `label_multistate=1`, `true_n_states=2`.
- `edge_weight` ~ a soft pairing prior (0.5..1.0) per candidate edge.
- Reactivities scaled to ~[0,1]; keep DMS missingness realistic (~50% non-A/C → nan).
Also expose `confounds(sample) -> dict` returning per-sample scalars used by the confound
battery: `mean_unpaired_prob`, `frac_nan_dms`, `n_candidate_edges`, `length` (so E* can be
partial-correlation–controlled against trivial explanations).

### `sheafprobe/data/loaders.py`  (Agent A1)
```python
def load_openknot(root="data/openknot", download=True) -> Dataset   # GitHub: eternagame/OpenKnotAIDesignData
def load_ribonanza(root="data/ribonanza") -> Dataset                # Kaggle (needs creds); raise clear error if absent
def load_openvaccine(root="data/openvaccine") -> Dataset            # Kaggle; clear error if absent
```
`load_openknot` MUST attempt a no-auth download from GitHub (raw files / git clone via
`requests`/`subprocess`) and parse whatever SHAPE/structure columns exist into `Sample`s
(SHAPE present, DMS = all-nan, label_multistate from pseudoknot annotation if available, else
skip). If download fails (offline), raise a clear, catchable `DataUnavailable` with the URL.
Ribonanza/OpenVaccine: parse if files present at `root`, else raise `DataUnavailable` with
the exact Kaggle dataset slug and `kaggle datasets download` hint. Never fabricate data.

### `sheafprobe/models/sheaf.py`  (Agent A2)
Adapt the proven pattern in `../sheaf-epistasis/src/sheafgi/model.py` (`SheafDiffusion`,
`inconsistency_energy`).
```python
class SheafProbe(nn.Module):
    def __init__(self, k=8, n_layers=4, eps=0.3, mode="general"): ...
        # mode in {"general","diagonal","identity"}; "identity" = plain graph diffusion baseline.
    def forward(self, sample) -> dict
        # returns {"x":Tensor[n,k], "pred_shape":Tensor[n], "pred_dms":Tensor[n],
        #          "estar":Tensor scalar, "edge_energy":Tensor[E]}
        # estar = sheaf Dirichlet energy of x over candidate base-pair edges (the heterogeneity score).
```
- Node encoder: base one-hot (+ positional) -> x0 [n,k]; then `n_layers` sheaf-diffusion
  steps over backbone+candidate edges using restriction maps selected by `mode`:
  general = learned k×k per edge (MLP on base pair, init near identity);
  diagonal = learned length-k vector (init ~1); identity = ones.
- Probe heads: `pred_shape = W_shape(x)`; `pred_dms = W_dms(x)` with DMS head applied only on
  A/C positions (use base_mask). Reactivity reconstruction is the data term.
- `estar` = Σ_e w_e · ||F_dst·x_i − F_src·x_j||² over candidate edges (edge-weighted Dirichlet).
```python
def train_sheafprobe(model, dataset, epochs=150, lr=1e-2, lam=1.0, seed=0) -> model
    # loss per sample = recon_mse(pred vs observed react, ignoring nan) + lam*estar; Adam.
def score_dataset(model, dataset) -> np.ndarray   # E* per sample (np.float, len = |dataset|)
def per_edge_energy(model, sample) -> np.ndarray   # for the competing-stem localization figure
```
Also write `tests/test_sheaf_math.py`: (i) identity-mode estar == plain graph Dirichlet energy;
(ii) a perfectly single-state hand-built sample has lower estar than a hand-built two-state one
after a few steps; (iii) estar ≥ 0; gradients flow.

### `sheafprobe/models/baselines.py` + `sheafprobe/experiments/metrics.py`  (Agent A3)
```python
# baselines.py
def entropy_bpp_score(sample) -> float
    # GOLD STANDARD non-learned baseline: per-position pairing prob p_i from reactivity
    # (low react -> high p_paired) blended with edge_weight; return Σ binary-entropy(p_i).
class TransformerRecon(nn.Module): ...   # small per-position transformer; reconstructs both
                                         # reactivity channels; heterogeneity score = recon residual.
def transformer_scores(dataset, epochs=150, seed=0) -> np.ndarray
# metrics.py
def auroc(scores, labels) -> float                       # via sklearn
def bootstrap_auroc_ci(scores, labels, n=1000, seed=0) -> tuple  # (auroc, lo, hi)
def partial_corr(x, y, controls: np.ndarray) -> tuple    # (partial_spearman, p) controlling for columns
def confound_report(estar, labels, confound_dict) -> dict
    # bare AUROC + partial-correlation of E* vs label after regressing out the confounds
    # (mean_unpaired_prob, frac_nan_dms, n_candidate_edges, length). Returns json-able dict.
```

### `sheafprobe/experiments/{killer,ablations,run,plots}.py`  (Agent A4)
```python
# killer.py
def run_killer(dataset, out_dir, seed=0) -> dict
    # trains SheafProbe(mode='general'); computes E* AUROC (with bootstrap CI) for
    # multistate separation vs: entropy_bpp (gold), SheafProbe(identity), TransformerRecon.
    # runs confound_report. Writes results/killer.json. Returns the dict.
# ablations.py
def run_ablations(dataset, out_dir, seed=0) -> dict
    # mode in {general,diagonal,identity} AUROC; stalk-dim k in {1,4,8}; single vs dual reagent
    # (mask DMS). Writes results/ablations.json.
# plots.py
def make_plots(results_dir) -> list[str]   # AUROC bar (sheaf vs baselines), ablation curves,
                                           # per-edge energy heatmap on one multistate example.
# run.py  — CLI
#   python -m sheafprobe.experiments.run --task {killer,ablations,all,real-openknot} [--out results] [--seed 0]
#   --quick for a tiny fast config (smoke). Real data tasks catch DataUnavailable and skip gracefully.
```
Also `tests/test_pipeline_smoke.py`: `run.py --task all --quick` produces non-empty
results json with finite AUROC values.

## Integration (Agent A5)
After A1–A4: create missing `__init__.py`, run `python -m sheafprobe.experiments.run --task all`
on synthetic; FIX any breakage by editing files; attempt `--task real-openknot`; run `pytest -q`;
generate figures into `results/figures/`; write the REAL `README.md` with the actual numbers
obtained (no invented results), an honest "what runs today vs what needs Kaggle" section, the
thesis, repro instructions, and a results table. Keep everything CPU-fast (minutes).

## Hard rules for every agent
- CPU only. Keep configs small (length≈68, ≤600 samples, k≤8, ≤200 epochs). It must finish in minutes.
- Never fabricate experimental numbers. Report whatever the run produces, including negatives.
- Match these signatures EXACTLY so modules compose. Read this file before coding.
- Style: type hints, short docstrings, numpy/torch idioms like the sheaf-epistasis repo.
