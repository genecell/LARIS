"""v0.15.0: the FDR family, the consistency of the null with the observed
score, and the low-count ranking penalty.

1. With the exact matched-gene null, Benjamini-Hochberg runs within each
   sender-receiver pair over every LR pair whose null has a non-zero
   pseudo-pair. Restricting it to rows with a positive observed score (the
   v0.14.1 default) leaves each null row's p-value too small by the share
   of non-zero pseudo-pairs, and inflates the FDR.
2. The null sees exactly what the observed score sees: pseudo-pairs below
   score_threshold count as zero, and the observed pair is scored through
   the same tables (a mismatch with the pipeline raises).
3. n_cells_expressed_threshold penalises the ranking only; the cell-type
   score uses the actual spatial specificity.
4. With p-values every LR pair is tested; n_top_lr only limits the output.
5. n_matched_genes='auto' sizes k to the database.
6. A gene without k/2 peers on one side, even after the side repair of the
   pool, gets a symmetrically truncated matched set that still straddles it;
   its pairs stay in the FDR family, flagged ``edge_limited`` (v0.15.0).
"""

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest
import anndata as ad
import scipy.sparse as sp

import laris as la
from laris.tools._utils import _bh_within_pairs, _fdr_family
from laris.tools._background import _resolve_n_matched_genes


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def planted():
    """A spatially coherent A->B interaction G000::G001 plus noise pairs."""
    rng = np.random.default_rng(12)
    n, g = 1200, 60
    genes = [f"G{i:03d}" for i in range(g)]
    X = sp.random(n, g, density=0.2, random_state=2,
                  data_rvs=lambda k: rng.poisson(2, k) + 1).tolil()
    coords = rng.random((n, 2)) * 100
    blob = np.linalg.norm(coords - [50, 50], axis=1) < 25
    ct = np.where(rng.random(n) < 0.5, "A", "B")
    X[np.flatnonzero(blob & (ct == "A")), 0] = 30
    X[np.flatnonzero(blob & (ct == "B")), 1] = 30
    # ten dense non-LR genes above the planted pair in abundance, so that
    # every LR gene has peers on both sides (the most abundant gene of a
    # matrix has none above it and gets an empty, truncated null)
    X[:, 50:] = rng.poisson(6, (n, 10)) + 1
    A = ad.AnnData(X=X.tocsr(), var=pd.DataFrame(index=genes))
    A.obsm["X_spatial"] = coords
    A.obs["ct"] = pd.Categorical(ct)
    lr_df = pd.DataFrame({
        "ligand": [f"G{i:03d}" for i in range(0, 20, 2)],
        "receptor": [f"G{i:03d}" for i in range(1, 20, 2)],
    })
    lr = _quiet(la.tl.prepareLRInteraction, A, lr_df,
                use_rep_spatial="X_spatial")
    bg = _quiet(la.tl.prepareLRBackground, A, lr_df, n_pool=40,
                n_matched_genes=6, use_rep_spatial="X_spatial", verbosity=0)
    return A, lr_df, lr, bg


def _run(planted, **kw):
    A, lr_df, lr, bg = planted
    kw.setdefault("background", bg)
    kw.setdefault("n_cells_expressed_threshold", 5)
    _, res = _quiet(la.tl.runLARIS, lr.copy(), A, use_rep="X_spatial",
                    use_rep_spatial="X_spatial", groupby="ct",
                    specificity_reference="all", **kw)
    return res


# ---------------------------------------------------------------- 1. family
def _simulate_null_table(rng, n_fam=200, m=150, N=400):
    """Results table in which every row is null (exchangeable with its
    pseudo-pairs) and most pseudo-pairs score zero, as on real tissue."""
    pi = np.where(rng.random(n_fam * m) < 0.6, 0.0,
                  rng.beta(0.6, 4.0, n_fam * m))
    K = rng.binomial(N + 1, pi)
    pos = rng.random(pi.size) < K / (N + 1)
    exceed = np.where(pos, np.floor(rng.random(pi.size) * np.maximum(K, 1)), N)
    support = np.where(pos, K - 1, K)
    p = np.where(pos & (support > 0), (exceed + 1) / (N + 1), 1.0)
    return pd.DataFrame({
        "sender": np.repeat(np.arange(n_fam), m).astype(str),
        "receiver": "r",
        "interaction_score": pos.astype(float),
        "p_value": p,
        "null_support": support,
    })


class TestFdrFamily:
    def test_new_family_controls_fdr_old_one_does_not(self):
        rng = np.random.default_rng(0)
        new_rate, old_rate = [], []
        for _ in range(5):
            t = _simulate_null_table(rng)
            new = _bh_within_pairs(t, _fdr_family(t, True, 0, True, 0.0))
            old = _bh_within_pairs(
                t, t.p_value.notna() & (t.interaction_score > 0))
            fam = t.sender
            new_rate.append((new < 0.05).groupby(fam).any().mean())
            old_rate.append((old < 0.05).groupby(fam).any().mean())
        # all rows are null, so any call is false: the share of families
        # with a call must stay near the nominal 5% under the new family
        assert np.mean(new_rate) <= 0.08
        assert np.mean(old_rate) >= 0.2

    def test_family_is_defined_by_the_null(self):
        t = pd.DataFrame({"sender": ["s"] * 4, "receiver": ["r"] * 4,
                          "interaction_score": [0.0, 1.0, 0.0, 1.0],
                          "p_value": [1.0, 0.01, 1.0, 0.2],
                          "null_support": [0, 5, 3, 0]})
        fam = _fdr_family(t, True, 0, True, 0.0)
        # zero-score rows are in when their null has support; a positive
        # row whose null is all zero is out (it has p = 1)
        assert fam.tolist() == [False, True, True, False]
        assert _fdr_family(t, True, 4, True, 0.0).tolist() == \
            [False, True, False, False]

    def test_runlaris_fdr_is_bh_over_the_null_defined_family(self, planted):
        k = planted[3].params["n_matched_genes"]
        for floor, kw in ((2 * k, {}), (1, {"min_null_support": 0})):
            res = _run(planted, **kw)
            self._check_family(res, floor)

    @staticmethod
    def _check_family(res, floor):
        for _, g in res.groupby(["sender", "receiver"]):
            # null-defined ('auto' floor: 2k non-zero pseudo-pairs);
            # edge-limited rows are tested too (section 6)
            fam = g.null_support >= floor
            if fam.any():
                from statsmodels.stats.multitest import multipletests
                exp = multipletests(g.p_value[fam].round(6), method="fdr_bh")[1]
                assert np.allclose(g.p_value_fdr[fam].to_numpy(), exp)
            assert (g.p_value_fdr[~fam] == 1.0).all()

    def test_planted_interaction_is_still_called(self, planted):
        res = _run(planted)
        hit = res[(res.interaction_name == "G000::G001")
                  & (res.sender == "A") & (res.receiver == "B")]
        assert hit.p_value.iloc[0] <= 3.0 / 37


# ------------------------------------------------------ 2. consistent null
class TestConsistentNull:
    def test_mismatched_background_raises(self, planted):
        A, lr_df, lr, bg = planted
        # a background diffused on a different graph than the observed
        # scores: every p-value would compare unlike quantities. The LR
        # object records its diffusion settings, so this fails up front.
        bg_bad = _quiet(la.tl.prepareLRBackground, A, lr_df, n_pool=40,
                        n_matched_genes=6, use_rep_spatial="X_spatial",
                        number_nearest_neighbors=5, verbosity=0)
        with pytest.raises(ValueError, match="different diffusion graph"):
            _run(planted, background=bg_bad)

    def test_mismatch_without_record_caught_by_table_guard(self, planted):
        A, lr_df, lr, bg = planted
        # LR objects from before v0.15.0 carry no diffusion record; the
        # table-score guard still catches the mismatch
        bg_bad = _quiet(la.tl.prepareLRBackground, A, lr_df, n_pool=40,
                        n_matched_genes=6, use_rep_spatial="X_spatial",
                        number_nearest_neighbors=5, verbosity=0)
        old = lr.copy()
        old.uns.pop("laris_diffusion", None)
        with pytest.raises(ValueError, match="background tables"):
            _quiet(la.tl.runLARIS, old, A, use_rep="X_spatial",
                   use_rep_spatial="X_spatial", groupby="ct",
                   specificity_reference="all", background=bg_bad,
                   n_cells_expressed_threshold=5)

    def test_null_support_counts_only_values_reaching_the_threshold(
            self, planted):
        lo = _run(planted, score_threshold=1e-12)
        hi = _run(planted, score_threshold=1e-3)
        key = ["sender", "receiver", "interaction_name"]
        m = lo.merge(hi, on=key, suffixes=("_lo", "_hi"))
        assert (m.null_support_hi <= m.null_support_lo).all()
        assert (m.null_support_hi < m.null_support_lo).any()

    def test_rust_and_numpy_agree_with_threshold(self, planted, monkeypatch):
        pytest.importorskip("laris._laris")
        monkeypatch.setenv("LARIS_NO_RUST", "1")
        a = _run(planted, score_threshold=1e-3)
        monkeypatch.delenv("LARIS_NO_RUST")
        b = _run(planted, score_threshold=1e-3)
        key = ["sender", "receiver", "interaction_name"]
        m = a.merge(b, on=key, suffixes=("_n", "_r"))
        assert (m.null_support_n == m.null_support_r).all()
        # the two paths build the co-localization numerator with different
        # arithmetic (float32 sparse vs float64 dense), so an exact tie can
        # flip by one lattice step, as in test_end_to_end_parity
        d = (m.p_value_n - m.p_value_r).abs()
        assert d.max() <= 1.0 / 37 + 1e-12
        assert (d < 1e-12).mean() > 0.95


# ------------------------------------------------- 3. ranking penalty only
class TestLowCountPenalty:
    def test_penalty_does_not_reach_the_score(self, planted):
        A, lr_df, lr, bg = planted
        lr_ = lr.copy()
        laris_lr, res = _quiet(la.tl.runLARIS, lr_, A, use_rep="X_spatial",
                               use_rep_spatial="X_spatial", groupby="ct",
                               specificity_reference="all",
                               calculate_pvalues=False,
                               n_cells_expressed_threshold=10_000)
        # every pair is below the threshold: the ranking puts them all
        # last, but each keeps its own spatial specificity ...
        spec = lr_.var["LR_SpatialSpecificity"]
        assert np.allclose(laris_lr["score"].to_numpy(),
                           spec.loc[laris_lr.index].to_numpy())
        # ... so the planted interaction still scores
        hit = res[(res.interaction_name == "G000::G001")
                  & (res.sender == "A") & (res.receiver == "B")]
        assert hit.interaction_score.iloc[0] > 0

    def test_scores_do_not_depend_on_the_count_threshold(self, planted):
        A, lr_df, lr, bg = planted
        kw = dict(use_rep="X_spatial", use_rep_spatial="X_spatial",
                  groupby="ct", specificity_reference="all",
                  calculate_pvalues=False)
        _, a = _quiet(la.tl.runLARIS, lr.copy(), A,
                      n_cells_expressed_threshold=0, **kw)
        _, b = _quiet(la.tl.runLARIS, lr.copy(), A,
                      n_cells_expressed_threshold=10_000, **kw)
        key = ["sender", "receiver", "interaction_name"]
        m = a.merge(b, on=key, suffixes=("_a", "_b"))
        assert np.allclose(m.interaction_score_a, m.interaction_score_b)


# ----------------------------------------------- 4. all pairs are tested
class TestTopLrOnlyLimitsOutput:
    def test_fdr_is_computed_over_every_pair(self, planted):
        full = _run(planted)
        top = _run(planted, n_top_lr=3)
        assert top.interaction_name.nunique() == 3
        key = ["sender", "receiver", "interaction_name"]
        m = top.merge(full, on=key, suffixes=("_t", "_f"))
        assert len(m) == len(top)
        assert np.allclose(m.p_value_fdr_t, m.p_value_fdr_f)
        assert np.allclose(m.p_value_t, m.p_value_f)


# --------------------------------------------------------------- 5. auto k
class TestAutoMatchedGenes:
    def _db(self, n):
        genes = [f"g{i}" for i in range(2 * n)]
        filler = [f"other{i}" for i in range(20_000)]   # a whole transcriptome
        return (pd.DataFrame({"ligand": genes[:n], "receptor": genes[n:]}),
                genes + filler)

    @pytest.mark.parametrize("n_pairs,k", [(12, 100), (777, 125),
                                           (1175, 154), (1985, 200),
                                           (5000, 200)])
    def test_sized_to_the_database(self, n_pairs, k):
        db, genes = self._db(n_pairs)
        assert _resolve_n_matched_genes("auto", db, genes) == k
        # at the resolved k an isolated hit at the floor can reach FDR 0.05
        # in a family of n_pairs, unless k was capped at 200
        if k < 200:
            assert 1 / (k * k + 1) <= 0.05 / n_pairs

    def test_only_pairs_present_count(self):
        db, genes = self._db(1985)
        assert _resolve_n_matched_genes("auto", db,
                                        genes[2 * 1985:]) == 100

    def test_explicit_value_and_bad_input(self):
        db, genes = self._db(10)
        assert _resolve_n_matched_genes(37, db, genes) == 37
        with pytest.raises(ValueError):
            _resolve_n_matched_genes("many", db, genes)

    def test_capped_by_the_candidate_pool(self):
        db, genes = self._db(1985)
        assert _resolve_n_matched_genes("auto", db, genes, n_pool=40) == 39

    def test_background_records_resolved_k(self, planted):
        A, lr_df, lr, bg = planted
        bg_auto = _quiet(la.tl.prepareLRBackground, A, lr_df, n_pool=40,
                         use_rep_spatial="X_spatial", verbosity=0)
        # 10 pairs would ask for 100, but the 40-gene pool caps it
        assert bg_auto.params["n_matched_genes"] == 39


# --------------------------------------------------------- legacy path
class TestLegacyPath:
    def test_prefiltered_family_uses_the_conditional_pvalue(self, planted):
        res = _run(planted, background=None, n_permutations=200)
        zero = res.interaction_score == 0
        assert (res.p_value[zero] == 1.0).all()

    def test_unconditional_with_prefilter_warns(self, planted):
        A, lr_df, lr, bg = planted
        with pytest.warns(UserWarning, match="conditions on the score"):
            with contextlib.redirect_stdout(io.StringIO()):
                la.tl.runLARIS(lr.copy(), A, use_rep="X_spatial",
                               use_rep_spatial="X_spatial", groupby="ct",
                               specificity_reference="all",
                               n_cells_expressed_threshold=5,
                               n_permutations=200,
                               use_conditional_pvalue=False)


# ------------------------------------------------------- 6. edge-limited genes
class TestEdgeLimited:
    """A gene with fewer than k/2 candidates on one side by mean, even after
    the side repair of the pool, gets a symmetrically truncated matched set
    that still straddles it; its pairs stay in the FDR family and carry the
    ``edge_limited`` flag (v0.15.0; the tag backfilled and excluded them)."""

    def test_mask_agrees_with_balanced_matching(self):
        from laris.tools._background import (_balanced_sets,
                                             _edge_limited_mask, _self_index)
        rng = np.random.default_rng(3)
        for k in (6, 7, 20):
            m = rng.gamma(1.0, 1.0, 300)
            v = m * (1 + rng.random(300))
            feats = np.column_stack([m, v])
            pool = np.sort(rng.choice(300, 120, replace=False))
            query = rng.choice(300, 60, replace=False)
            for qpi in (None, _self_index(pool, query)):
                idx, pool_sorted, count = _balanced_sets(
                    feats, pool, query, k, query_pool_index=qpi)
                mask = _edge_limited_mask(m, pool, query, k,
                                          query_pool_index=qpi)
                assert int(mask.sum()) == count
                # the extremes of the pool are edge-limited
                top = np.argsort(m[query])[-1]
                assert mask[top] == (np.sum(m[pool] > m[query][top]) < k - k // 2)
                for gi, sel, edge in zip(query, idx, mask):
                    means = m[pool_sorted[sel]]
                    below = int((means <= m[gi]).sum())
                    if edge:
                        # symmetric: t below and t above, t < k/2
                        assert len(sel) == 2 * below < k
                        assert int((means > m[gi]).sum()) == below
                    else:
                        assert len(sel) == k and below == k // 2

    def test_non_edge_sets_unchanged(self):
        """Genes that can be straddled keep exactly their v0.15.0-tag sets."""
        from laris.tools._background import _balanced_sets, _edge_limited_mask
        rng = np.random.default_rng(5)
        m = rng.gamma(1.0, 1.0, 400)
        feats = np.column_stack([m, m * (1 + rng.random(400))])
        pool = np.sort(rng.choice(400, 200, replace=False))
        query = rng.choice(400, 50, replace=False)
        idx, pool_sorted, _ = _balanced_sets(feats, pool, query, 10)
        mask = _edge_limited_mask(m, pool, query, 10)
        mu, sd = feats.mean(0), feats.std(0)
        z = (feats - mu) / sd
        for gi, sel, edge in zip(query, idx, mask):
            if edge:
                continue
            ms = m[pool_sorted]
            cut = int(np.searchsorted(ms, m[gi], side="right"))
            want = []
            for side in (np.arange(cut), np.arange(cut, len(ms))):
                if side.size <= 5:
                    want.append(side)
                    continue
                d = np.linalg.norm(z[pool_sorted[side]] - z[gi], axis=1)
                want.append(side[np.argpartition(d, 5)[:5]])
            assert np.array_equal(np.sort(sel), np.sort(np.concatenate(want)))

    def test_side_repair(self):
        from laris.tools._background import _side_repair_pool
        rng = np.random.default_rng(1)
        n, k = 500, 20
        means = np.sort(rng.gamma(1.0, 1.0, n))
        means[:15] = 0.0                       # 15 genes with no counts
        detected = means > 0
        query = np.array([16, 20, 250, 480, 495])
        # a pool that misses the bottom and the top of the range
        pool = np.arange(100, 400)
        out = _side_repair_pool(means, detected, pool, query, k)
        assert detected[out].all()             # no zero-count candidates
        assert set(pool) <= set(out)
        added = np.setdiff1d(out, pool)
        # the bottom tail: k/2 per side at most, shared by the low genes
        assert ((added < 100).sum()) <= k // 2
        assert ((added >= 400).sum()) <= k // 2
        for g in query:
            below = out[(means[out] <= means[g]) & (out != g)]
            above = out[means[out] > means[g]]
            n_below_all = int((detected & (means <= means[g])).sum()) - 1
            n_above_all = int((detected & (means > means[g])).sum())
            assert len(below) >= min(k // 2, n_below_all)
            assert len(above) >= min(k - k // 2, n_above_all)

    def test_zero_count_genes_leave_the_pool(self):
        from laris.tools._background import _side_repair_pool
        means = np.array([0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        detected = means > 0
        out = _side_repair_pool(means, detected, np.arange(8), [4], 4)
        assert not set(out) & {0, 1}

    def test_background_records_edge_genes(self, planted):
        _, lr_df, _, bg = planted
        assert isinstance(bg.edge_genes, frozenset)
        assert bg.params["truncation"] == "symmetric"
        genes = set(lr_df.ligand) | set(lr_df.receptor)
        assert bg.edge_genes <= genes
        k = bg.params["n_matched_genes"]
        for d in (bg.matched_ligand, bg.matched_receptor):
            for g, s in d.items():
                assert (len(s) < k) == (g in bg.edge_genes)
                # every set straddles its gene (or is empty)
                f = bg.match_frac_below[g]
                if len(s) == 0:
                    assert np.isnan(f)
                else:
                    assert abs(f - (len(s) // 2) / len(s)) <= 1.0 / len(s)

    def test_family_keeps_edge_rows(self):
        t = pd.DataFrame({"sender": ["s"] * 3, "receiver": ["r"] * 3,
                          "p_value": [0.001, 0.001, 0.5],
                          "null_support": [10, 10, 10],
                          "edge_limited": [False, True, False]})
        assert _fdr_family(t, True, 0, True, 0.0).tolist() == [True, True, True]

    def test_results_flag_edge_rows_and_test_them(self, planted):
        res = _run(planted)
        assert {"edge_limited", "null_size", "null_support"} <= set(res)
        k = planted[3].params["n_matched_genes"]
        assert (res.null_size <= k * k).all()
        if res.edge_limited.any():
            edge = res[res.edge_limited]
            assert (edge.null_size < k * k).all()
            tested = edge.null_support >= 1
            # tested like every other row: FDR from BH, not forced to 1
            assert edge.loc[tested, "p_value"].notna().all()

    def test_legacy_background_warns(self, planted):
        import copy
        A, lr_df, lr, bg = planted
        old = copy.copy(bg)
        old.params = {k_: v_ for k_, v_ in bg.params.items()
                      if k_ != "truncation"}
        old.edge_genes = None
        with pytest.warns(UserWarning, match="before the v0.15.0 edge repair"):
            with contextlib.redirect_stdout(io.StringIO()):
                _, res = la.tl.runLARIS(
                    lr.copy(), A, use_rep="X_spatial",
                    use_rep_spatial="X_spatial", groupby="ct",
                    background=old, n_cells_expressed_threshold=5,
                    specificity_reference="all")
        assert not res.edge_limited.any()

    def test_knn_matching_is_deprecated(self, planted):
        A, lr_df, _, _ = planted
        with pytest.warns(FutureWarning, match="matching='knn' is deprecated"):
            with contextlib.redirect_stdout(io.StringIO()):
                la.tl.prepareLRBackground(A, lr_df, n_pool=40,
                                          n_matched_genes=6, matching="knn",
                                          use_rep_spatial="X_spatial",
                                          verbosity=0)

    def test_support_per_pair_is_deprecated(self, planted):
        bg = planted[3]
        with pytest.warns(FutureWarning, match="null_size"):
            assert bg.support_per_pair == bg.params["n_matched_genes"] ** 2


class TestEmptyMatchedSet:
    """An LR gene with no counts at all has no detected peers below it, so
    its truncated set is empty: its rows get an empty null (null_size 0,
    p = 1) instead of crashing anything."""

    @pytest.fixture(scope="class")
    def silent_gene(self):
        rng = np.random.default_rng(4)
        n, g = 600, 50
        genes = [f"G{i:03d}" for i in range(g)]
        X = sp.random(n, g, density=0.25, random_state=3,
                      data_rvs=lambda k: rng.poisson(2, k) + 1).tolil()
        X[:, 2] = 0                       # ligand G002 is never detected
        A = ad.AnnData(X=X.tocsr(), var=pd.DataFrame(index=genes))
        A.obsm["X_spatial"] = rng.random((n, 2)) * 100
        A.obs["ct"] = pd.Categorical(np.where(rng.random(n) < 0.5, "A", "B"))
        lr_df = pd.DataFrame({"ligand": ["G000", "G002", "G004"],
                              "receptor": ["G001", "G003", "G005"]})
        lr = _quiet(la.tl.prepareLRInteraction, A, lr_df,
                    use_rep_spatial="X_spatial")
        bg = _quiet(la.tl.prepareLRBackground, A, lr_df, n_pool=40,
                    n_matched_genes=6, use_rep_spatial="X_spatial",
                    verbosity=0)
        return A, lr_df, lr, bg

    def test_empty_set_and_rows(self, silent_gene):
        A, lr_df, lr, bg = silent_gene
        assert len(bg.matched_ligand["G002"]) == 0
        # NaN, not None: the dict stays numeric (comparisons, np.array)
        assert np.isnan(bg.match_frac_below["G002"])
        vals = np.array(list(bg.match_frac_below.values()))
        assert vals.dtype == float and not (vals >= 0.99).any()
        assert "G002" in bg.edge_genes
        _, res = _quiet(la.tl.runLARIS, lr.copy(), A, use_rep="X_spatial",
                        use_rep_spatial="X_spatial", groupby="ct",
                        specificity_reference="all", background=bg,
                        n_cells_expressed_threshold=5)
        rows = res[res.ligand == "G002"]
        assert len(rows) and (rows.null_size == 0).all()
        assert (rows.p_value == 1.0).all() and (rows.p_value_fdr == 1.0).all()
        assert res.loc[res.ligand != "G002", "p_value"].notna().all()


class TestBackgroundGraphCheck:
    """runLARIS's spatial-specificity graph must be the background's."""

    def test_mismatched_sigma_fails_fast(self, planted):
        with pytest.raises(ValueError, match=r"prepareLRBackground\(sigma_gsp="):
            _run(planted, sigma=100)

    def test_mismatched_k_fails_fast(self, planted):
        with pytest.raises(ValueError,
                           match=r"prepareLRBackground\(n_nearest_neighbors_gsp="):
            _run(planted, n_nearest_neighbors=7)

    def test_matching_settings_pass(self, planted):
        A, lr_df, lr, _ = planted
        bg = _quiet(la.tl.prepareLRBackground, A, lr_df, n_pool=40,
                    n_matched_genes=6, use_rep_spatial="X_spatial",
                    sigma_gsp=100, verbosity=0)
        res = _run(planted, background=bg, sigma=100)
        assert res.p_value.notna().all()


class TestEdgeRowsPureNull:
    """Under a pure null (Poisson counts, random coordinates and labels) the
    rows of truncated (edge-limited) genes must not be inflated. The v0.15.0
    tag backfilled such sets from one side, which was 2.9x nominal at
    P <= 0.05 on the Visium HD pure null (k = 200)."""

    def test_edge_rows_calibrated(self):
        from scipy.stats import binom
        rng = np.random.default_rng(11)
        n, g = 1500, 160
        lam = np.exp(rng.uniform(np.log(0.03), np.log(25), g))
        X = sp.csr_matrix(rng.poisson(lam, (n, g)).astype(np.float32))
        genes = [f"G{i:03d}" for i in range(g)]
        A = ad.AnnData(X=X, var=pd.DataFrame(index=genes))
        A.obsm["X_spatial"] = rng.random((n, 2)) * 100
        A.obs["ct"] = pd.Categorical(rng.choice(list("ABC"), n))
        order = np.argsort(lam)
        ends = np.concatenate([order[:6], order[-6:]])
        middle = order[20:-20]
        # every gene at either end of the abundance range, in four pairs
        lig, rec = [], []
        for j, e in enumerate(ends):
            for m in rng.choice(middle, 4, replace=False):
                a, b = (e, m) if j % 2 == 0 else (m, e)
                lig.append(genes[a]); rec.append(genes[b])
        lr_df = pd.DataFrame({"ligand": lig, "receptor": rec}).drop_duplicates()
        lr = _quiet(la.tl.prepareLRInteraction, A, lr_df,
                    use_rep_spatial="X_spatial")
        bg = _quiet(la.tl.prepareLRBackground, A, lr_df, n_pool=100,
                    n_matched_genes=10, use_rep_spatial="X_spatial",
                    verbosity=0)
        assert bg.edge_genes, "the panel must force truncated sets"
        _, res = _quiet(la.tl.runLARIS, lr.copy(), A, use_rep="X_spatial",
                        use_rep_spatial="X_spatial", groupby="ct",
                        specificity_reference="all", background=bg,
                        n_cells_expressed_threshold=5)
        edge = res[res.edge_limited & (res.null_support >= 1)]
        assert len(edge) >= 50
        # At k = 10 a truncated set has 2-8 genes, so these nulls are tiny
        # and boundary bias remains (measured over 8 seeds: 0.071 at
        # P <= 0.05, 0.099 at P <= 0.10, the same as the backfill on this
        # synthetic panel). The guard is against gross inflation; the
        # Fig. S6 pure nulls at k = 200 are the calibration check.
        assert (edge.p_value <= 0.05).mean() <= 0.10
        x = int((edge.p_value <= 0.1).sum())
        assert binom.sf(x - 1, len(edge), 0.1) > 1e-3, (x, len(edge))
