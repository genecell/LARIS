"""compareLARISMatched: anchor-standardised cross-condition comparison.

The design history matters here: two earlier "matched background" forms
failed their negative controls (86% and 63% null false-positive rates at
p<0.05 - see the module docstring of laris.tools._compareMatched). These
tests pin the properties that version failed on, so a refactor cannot
quietly reintroduce either trap:

- calibration on a null with real per-subject biological variability
- EXACT invariance to per-sample scale factors, at 90% sparsity
- no false calls from a pure cell-state composition shift
- recovery of a genuine effect
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import anndata as ad
import scipy.sparse as sp

import laris as la


def _simulate(effect=0.0, scale_confound=False, composition_shift=False,
              n_subjects=5, cells=400, n_lr=40, seed=0):
    """Two conditions x n subjects with per-subject baselines, two cell
    states at different embedding locations, and 90% zeros."""
    rng = np.random.default_rng(seed)
    frames, embs, Xs = [], [], []
    for cond in ("ref", "alt"):
        for s in range(n_subjects):
            frac_state1 = 0.8 if (composition_shift and cond == "alt") else 0.4
            n1 = int(cells * frac_state1)
            emb = np.vstack([rng.normal(0, 1, (n1, 10)),
                             rng.normal(3, 1, (cells - n1, 10))])
            base = np.exp(rng.normal(-2, 1, n_lr))       # per-subject profile
            X = base * rng.lognormal(0, .5, (cells, n_lr))
            X[:n1, ::2] *= 3.0            # state-linked biology, both conditions
            X[rng.random((cells, n_lr)) < 0.9] = 0
            if effect and cond == "alt":
                X[:, :8] *= (1 + effect)
            if scale_confound and cond == "alt":
                X *= 3.0
            frames.append(pd.DataFrame({
                "condition": cond, "sample": f"{cond}{s}",
            }, index=[f"{cond}{s}_c{i}" for i in range(cells)]))
            embs.append(emb)
            Xs.append(X)
    lr_data = ad.AnnData(
        X=sp.csr_matrix(np.vstack(Xs)),
        obs=pd.concat(frames),
        var=pd.DataFrame(index=[f"L{i}::R{i}" for i in range(n_lr)]),
    )
    lr_data.obsm["X_pca"] = np.vstack(embs)
    return lr_data


def _run(lr_data, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return la.tl.compareLARISMatched(
            lr_data, conditionKey="condition", referenceCondition="ref",
            sampleKey="sample", **kwargs)


class TestCalibration:
    def test_null_with_subject_variability(self):
        """The v2 trap: per-subject baselines must stay in the noise."""
        cmp_, _ = _run(_simulate(seed=1))
        tested = cmp_[cmp_.pvalue.notna()]
        assert (tested.pvalue < 0.05).mean() < 0.15
        assert (tested.pvalue_fdr < 0.05).sum() == 0

    def test_composition_shift_is_not_an_effect(self):
        """A condition with more of a high-scoring cell state is NOT a
        change in LR signalling at matched states."""
        cmp_, _ = _run(_simulate(composition_shift=True, seed=2))
        tested = cmp_[cmp_.pvalue.notna()]
        assert (tested.pvalue_fdr < 0.05).sum() == 0

    def test_exact_scale_invariance_at_high_sparsity(self):
        """The equivariant centring must survive the anchor averaging."""
        plain = _simulate(seed=3)
        scaled = plain.copy()
        alt = (scaled.obs["condition"] == "alt").to_numpy()
        X = scaled.X.toarray(); X[alt] *= 3.0
        scaled.X = sp.csr_matrix(X)
        a, _ = _run(plain)
        b, _ = _run(scaled)
        m = a.merge(b, on="interaction_name", suffixes=("_a", "_b"))
        assert np.nanmax(np.abs(m.log_diff_a - m.log_diff_b)) < 1e-10
        assert np.nanmax(np.abs(m.pvalue_a - m.pvalue_b)) < 1e-10


class TestRecovery:
    def test_effect_is_recovered(self):
        cmp_, _ = _run(_simulate(effect=4.0, cells=800, seed=4))
        true = cmp_[cmp_.interaction_name.isin([f"L{i}::R{i}" for i in range(8)])]
        null = cmp_[~cmp_.interaction_name.isin([f"L{i}::R{i}" for i in range(8)])]
        assert (true.pvalue < 0.05).mean() >= 0.5
        assert (null.pvalue_fdr < 0.05).sum() <= 1

    def test_direction_of_the_effect(self):
        cmp_, _ = _run(_simulate(effect=4.0, cells=800, seed=4))
        true = cmp_[cmp_.interaction_name.isin([f"L{i}::R{i}" for i in range(8)])]
        assert (true.log_diff > 0).mean() >= 0.75


class TestStructure:
    def test_profiles_are_per_subject(self):
        lr_data = _simulate(seed=5)
        cmp_, profiles = _run(lr_data)
        assert profiles.shape == (10, lr_data.n_vars)
        assert set(profiles.index) == set(lr_data.obs["sample"])

    def test_detection_and_fisher_route_apply(self):
        """An LR absent in one condition routes to Fisher, like compareLARIS."""
        lr_data = _simulate(seed=6)
        X = lr_data.X.toarray()
        ref = (lr_data.obs["condition"] == "ref").to_numpy()
        X[ref, 0] = 0.0                       # L0::R0 never detected in ref
        X[~ref, 0] = np.abs(X[~ref, 0]) + 0.5
        lr_data.X = sp.csr_matrix(X)
        cmp_, _ = _run(lr_data)
        row = cmp_[cmp_.interaction_name == "L0::R0"].iloc[0]
        assert row.test_method == "fisher_detection"
        assert row.n_detected_ref == 0 and row.n_detected_alt == 5

    def test_subject_pooling_of_slices(self):
        """Two slices per subject collapse to one profile per subject."""
        lr_data = _simulate(seed=7)
        lr_data.obs["subject"] = [s.split("_")[0][:-1] + s[3] for s in lr_data.obs_names]
        # simpler: map pairs of samples onto shared subjects
        mapping = {f"{c}{i}": f"{c}_m{i // 2}" for c in ("ref", "alt") for i in range(5)}
        lr_data.obs["subject"] = lr_data.obs["sample"].map(mapping)
        cmp_, profiles = _run(lr_data, subjectKey="subject")
        assert set(profiles.index) == set(mapping.values())
        assert profiles.shape[0] == 6          # ceil(5/2) per condition

    def test_k_anchor_extremes_run(self):
        lr_data = _simulate(seed=8, cells=150)
        for k in (1, 1000):                    # 1000 > cells: clipped
            cmp_, _ = _run(lr_data, k_anchor=k, n_anchors=20)
            assert len(cmp_) == lr_data.n_vars


class TestGuards:
    def test_requires_anndata(self):
        with pytest.raises(TypeError, match="AnnData"):
            la.tl.compareLARISMatched(
                {"a": 1}, conditionKey="c", referenceCondition="x",
                sampleKey="s")

    def test_missing_obs_key(self):
        with pytest.raises(ValueError, match="not found in lr_data.obs"):
            _run(_simulate(seed=9).copy(), subjectKey="mouse")

    def test_missing_embedding(self):
        lr_data = _simulate(seed=9)
        del lr_data.obsm["X_pca"]
        with pytest.raises(ValueError, match="JOINT embedding"):
            _run(lr_data)

    def test_unknown_reference(self):
        lr_data = _simulate(seed=9)
        with pytest.raises(ValueError, match="referenceCondition"):
            la.tl.compareLARISMatched(
                lr_data, conditionKey="condition", referenceCondition="WT",
                sampleKey="sample")

    def test_subject_in_two_conditions_is_rejected(self):
        """A donor contributing sections to both conditions breaks the
        between-subject design; the function must say so, not guess."""
        lr_data = _simulate(seed=9)
        lr_data.obs["subject"] = "same_mouse"
        with pytest.raises(ValueError, match="more than one\\s+condition"):
            _run(lr_data, subjectKey="subject")


class TestMemoryStreaming:
    def test_sparse_input_is_never_fully_densified(self):
        """Peak traced memory must stay near one subject's dense block, not
        the whole cohort's. 20 subjects x 2,000 cells x 300 LR in float64 is
        96 MB dense (plus a centred copy); one subject is 4.8 MB."""
        import tracemalloc
        rng = np.random.default_rng(0)
        n_subj, cells, n_lr = 20, 2000, 300
        frames, embs, Xs = [], [], []
        for cond in ("ref", "alt"):
            for s in range(n_subj // 2):
                X = sp.random(cells, n_lr, density=0.05, random_state=s,
                              format="csr", dtype=np.float64)
                frames.append(pd.DataFrame(
                    {"condition": cond, "sample": f"{cond}{s}"},
                    index=[f"{cond}{s}_c{i}" for i in range(cells)]))
                embs.append(rng.normal(0, 1, (cells, 10)))
                Xs.append(X)
        lr_data = ad.AnnData(X=sp.vstack(Xs).tocsr(), obs=pd.concat(frames),
                             var=pd.DataFrame(index=[f"L{i}::R{i}"
                                                     for i in range(n_lr)]))
        lr_data.obsm["X_pca"] = np.vstack(embs)
        full_dense_bytes = n_subj * cells * n_lr * 8

        tracemalloc.start()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            la.tl.compareLARISMatched(
                lr_data, conditionKey="condition", referenceCondition="ref",
                sampleKey="sample", n_anchors=30)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # generous bound: half the full dense size (the old code needed
        # 2x full dense; the streamed path needs ~2 subject blocks)
        assert peak < full_dense_bytes / 2, (
            f"peak {peak/1e6:.0f} MB vs full dense {full_dense_bytes/1e6:.0f} MB"
        )
