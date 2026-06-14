"""Tests for the synthetic dataset generator (the construction is the experiment)."""
from __future__ import annotations

import numpy as np
import pytest

from sheafprobe.data import (
    Sample,
    base_mask,
    confounds,
    generate_dataset,
)
from sheafprobe.config import QUICK


@pytest.fixture(scope="module")
def quick_dataset():
    """A small balanced dataset built from the QUICK config (fast, deterministic)."""
    return generate_dataset(
        n_samples=QUICK.N_SAMPLES,
        length=QUICK.LENGTH,
        frac_multistate=QUICK.FRAC_MULTISTATE,
        noise=QUICK.NOISE,
        seed=0,
    )


def test_dataset_size_and_type(quick_dataset):
    assert len(quick_dataset) == QUICK.N_SAMPLES
    assert all(isinstance(s, Sample) for s in quick_dataset)


def test_shapes_consistent(quick_dataset):
    for s in quick_dataset:
        assert s.n == QUICK.LENGTH
        assert len(s.seq) == s.n
        assert s.edges.shape[0] == 2
        assert s.edges.shape[1] == s.edge_weight.shape[0]
        assert s.backbone.shape == (2, s.n - 1)
        assert s.react_shape.shape == (s.n,)
        assert s.react_dms.shape == (s.n,)
        # candidate edges are valid (i < j, in range)
        i, j = s.edges
        assert np.all(i < j)
        assert np.all(i >= 0) and np.all(j < s.n)
        # edge weights are a soft prior in (0, 1]
        assert np.all(s.edge_weight > 0.0) and np.all(s.edge_weight <= 1.0)


def test_labels_valid_and_balanced(quick_dataset):
    labels = np.array([s.label_multistate for s in quick_dataset])
    assert set(np.unique(labels)).issubset({0, 1})
    frac_multi = labels.mean()
    # ~50/50 by construction; allow a small slack for rounding.
    assert 0.4 <= frac_multi <= 0.6
    # n_states matches label semantics.
    for s in quick_dataset:
        if s.label_multistate == 1:
            assert s.true_n_states == 2
        else:
            assert s.true_n_states == 1


def test_reactivities_finite_where_present(quick_dataset):
    for s in quick_dataset:
        # SHAPE observed on every base -> all finite and within [0, 1].
        assert np.all(np.isfinite(s.react_shape))
        assert np.all(s.react_shape >= 0.0) and np.all(s.react_shape <= 1.0)
        # DMS: finite exactly on A/C positions, nan elsewhere.
        ac = base_mask(s.seq, "AC")
        assert np.all(np.isfinite(s.react_dms[ac]))
        assert np.all(np.isnan(s.react_dms[~ac]))
        # where finite, DMS is in [0, 1].
        finite_dms = s.react_dms[np.isfinite(s.react_dms)]
        if finite_dms.size:
            assert np.all(finite_dms >= 0.0) and np.all(finite_dms <= 1.0)


def test_dms_missingness_realistic(quick_dataset):
    """~50% of positions should be non-A/C -> nan DMS (kept loose for randomness)."""
    fracs = [np.mean(~np.isfinite(s.react_dms)) for s in quick_dataset]
    assert 0.25 <= float(np.mean(fracs)) <= 0.75


def _edge_set(sample: Sample) -> set:
    return {(int(i), int(j)) for i, j in zip(*sample.edges)}


def test_multistate_has_two_competing_stems(quick_dataset):
    """Multi-state samples must carry pairs from BOTH incompatible states.

    Concretely: at least two candidate edges must *cross* (i<k<j<l), which is the
    graph signature of two mutually incompatible (nested vs pseudoknot) pairings.
    """
    multi = [s for s in quick_dataset if s.label_multistate == 1]
    assert multi, "expected some multi-state samples"
    checked = 0
    for s in multi:
        edges = sorted(_edge_set(s))
        crossing_found = False
        for a in range(len(edges)):
            i, j = edges[a]
            for b in range(a + 1, len(edges)):
                k, l = edges[b]
                if i < k < j < l:  # crossing pair = pseudoknot frustration
                    crossing_found = True
                    break
            if crossing_found:
                break
        # the molecule must have >= 2 competing (crossing) stems in its edge set
        assert crossing_found, f"{s.id} lacks crossing candidate stems"
        checked += 1
    assert checked == len(multi)


def test_confounds_keys_and_values(quick_dataset):
    keys = {"mean_unpaired_prob", "frac_nan_dms", "n_candidate_edges", "length"}
    for s in quick_dataset[:10]:
        c = confounds(s)
        assert keys.issubset(c.keys())
        assert np.isfinite(c["mean_unpaired_prob"])
        assert 0.0 <= c["frac_nan_dms"] <= 1.0
        assert c["n_candidate_edges"] == s.edges.shape[1]
        assert c["length"] == s.n


def test_multistate_reactivity_more_intermediate(quick_dataset):
    """Population-averaged multi-state reactivities should be more 'half-paired'.

    Single-state positions are pushed toward the paired/unpaired anchors; multi-state
    averaging pulls competing-stem positions toward the middle. We check the mean
    absolute deviation from 0.5 is smaller for multi-state on average.
    """
    def mad_from_half(s):
        return float(np.mean(np.abs(s.react_shape - 0.5)))

    single = [mad_from_half(s) for s in quick_dataset if s.label_multistate == 0]
    multi = [mad_from_half(s) for s in quick_dataset if s.label_multistate == 1]
    # not a strict per-sample guarantee, but should hold in aggregate by construction
    assert np.mean(multi) <= np.mean(single) + 1e-6
