"""v0.15.0 significance defaults: the 2k support floor, the partner-rank
columns and the ordering of the results.

1. ``min_null_support='auto'`` is 2 x n_matched_genes: rows whose null has
   fewer non-zero pseudo-pairs get p = 1 and leave the FDR family; every
   other row keeps exactly its p-value.
2. ``ligand_rank`` / ``receptor_rank``: the observed pair's rank among its
   ligand's (receptor's) matched genes paired with the observed partner.
3. Results are ordered by interaction score by default (as before v0.15.0);
   ``sort_by='p_value'`` orders them by p-value, ties by score.
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


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def planted():
    """A spatially coherent A->B interaction G000::G001, noise pairs, and
    dense non-LR genes above them (so every LR gene has peers both sides)."""
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


KEY = ["sender", "receiver", "interaction_name"]


class TestAutoFloor:
    def test_auto_is_two_k(self, planted):
        from laris.tools._utils import _resolve_min_null_support
        bg = planted[3]
        k = bg.params["n_matched_genes"]
        assert _resolve_min_null_support("auto", bg) == 2 * k
        assert _resolve_min_null_support("auto", None) == 0
        assert _resolve_min_null_support(5, bg) == 5
        with pytest.raises(ValueError):
            _resolve_min_null_support("2k", bg)
        with pytest.raises(ValueError):
            _resolve_min_null_support(-1, bg)

    def test_floor_moves_thin_rows_out_only(self, planted):
        k = planted[3].params["n_matched_genes"]
        auto = _run(planted)
        none = _run(planted, min_null_support=0)
        m = auto.merge(none, on=KEY, suffixes=("_a", "_0"))
        assert len(m) == len(auto) == len(none)
        thin = m.null_support_0 < 2 * k
        assert thin.any() and (~thin).any()
        # thin rows: p = 1 and outside the family
        assert (m.loc[thin, "p_value_a"] == 1.0).all()
        assert (m.loc[thin, "p_value_fdr_a"] == 1.0).all()
        # every other row keeps its p-value exactly
        assert np.array_equal(m.loc[~thin, "p_value_a"],
                              m.loc[~thin, "p_value_0"])
        # and BH runs over the smaller family
        from statsmodels.stats.multitest import multipletests
        for _, g in m[~thin].groupby(["sender", "receiver"]):
            exp = multipletests(g.p_value_a.round(6), method="fdr_bh")[1]
            assert np.allclose(g.p_value_fdr_a.to_numpy(), exp)


class TestPartnerRanks:
    def test_columns_and_bounds(self, planted):
        res = _run(planted)
        k = planted[3].params["n_matched_genes"]
        for col in ("ligand_rank", "receptor_rank"):
            assert col in res
            v = res[col].to_numpy()
            assert ((v > 0) & (v <= 1)).all()
            assert (v[res.interaction_score.to_numpy() == 0] == 1.0).all()
            # on the lattice (1 + j) / (1 + n_valid), n_valid <= k
            pos = res.interaction_score > 0
            ok = np.zeros(int(pos.sum()), dtype=bool)
            for n_valid in range(1, k + 1):
                s = v[pos.to_numpy()] * (n_valid + 1)
                ok |= np.abs(s - np.round(s)) < 1e-9
            assert ok.all()

    @pytest.mark.parametrize("side", ["ligand", "receptor"])
    def test_rank_is_the_p_value_against_the_partner_pairs(self, planted,
                                                          side):
        """Independent check through the main P-value machinery: give the
        partner gene a matched set containing only itself, and the cross
        null of a row becomes exactly that row's partner pairs, so its
        exact p-value must equal the partner rank."""
        import copy
        A, lr_df, lr, bg = planted
        res = _run(planted)
        bg1 = copy.copy(bg)
        bg1.matched_ligand = dict(bg.matched_ligand)
        bg1.matched_receptor = dict(bg.matched_receptor)
        pos = {g: i for i, g in enumerate(bg.gene_index)}
        if side == "ligand":           # the receptor's set -> itself
            for r_ in lr_df.receptor:
                bg1.matched_receptor[r_] = np.array([pos[r_]])
        else:
            for l_ in lr_df.ligand:
                bg1.matched_ligand[l_] = np.array([pos[l_]])
        alt = _run(planted, background=bg1, min_null_support=0)
        m = res.merge(alt, on=KEY, suffixes=("", "_alt"))
        tested = (m.interaction_score > 0) & (m.null_support_alt > 0)
        assert tested.sum() >= 10
        np.testing.assert_allclose(m.loc[tested, f"{side}_rank"],
                                   m.loc[tested, "p_value_alt"], rtol=0,
                                   atol=1e-12)

    def test_planted_receptor_carries_the_call(self, planted):
        """In this toy the planted ligand sits in a gap of the abundance
        range, so its matched set mixes sparse and dense genes, and dense
        ligand-side genes paired with the planted receptor score higher
        (detection fractions): the call leans on the receptor, which the
        ranks show."""
        res = _run(planted)
        k = planted[3].params["n_matched_genes"]
        hit = res[(res.interaction_name == "G000::G001")
                  & (res.sender == "A") & (res.receiver == "B")].iloc[0]
        assert hit.p_value <= 3.0 / (k * k + 1)
        assert hit.receptor_rank <= 2.0 / (k + 1)
        assert hit.ligand_rank > hit.receptor_rank

    def test_fused_and_numpy_paths_agree(self, planted, monkeypatch):
        monkeypatch.setenv("LARIS_NO_RUST", "1")
        res_np = _run(planted)
        monkeypatch.delenv("LARIS_NO_RUST")
        res_rs = _run(planted)
        k = planted[3].params["n_matched_genes"]
        m = res_rs.merge(res_np, on=KEY, suffixes=("_r", "_n"))
        assert len(m) == len(res_rs)
        assert np.array_equal(m.null_size_r, m.null_size_n)
        for col in ("ligand_rank", "receptor_rank"):
            d = (m[f"{col}_r"] - m[f"{col}_n"]).abs()
            # float32 summation order only: at most one lattice step
            assert d.max() <= 1.0 / (k + 1) + 1e-12
            assert (d < 1e-12).mean() > 0.95


class TestOrdering:
    def test_default_ordered_by_score(self, planted):
        res = _run(planted)
        assert res.p_value.notna().any()
        assert (np.diff(res.interaction_score.to_numpy()) <= 0).all()
        assert res.index.equals(pd.RangeIndex(len(res)))

    def test_sort_by_p_value(self, planted):
        res = _run(planted, sort_by="p_value")
        p = res.p_value.to_numpy()
        s = res.interaction_score.to_numpy()
        assert (np.diff(p) >= 0).all()
        tie = np.diff(p) == 0
        assert (np.diff(s)[tie] <= 0).all()
        assert res.index.equals(pd.RangeIndex(len(res)))

    def test_without_pvalues_ordered_by_score(self, planted):
        res = _run(planted, calculate_pvalues=False, background=None)
        assert (np.diff(res.interaction_score.to_numpy()) <= 0).all()

    def test_order_changes_no_value(self, planted):
        a = _run(planted)
        b = _run(planted, sort_by="p_value")
        cols = KEY + ["interaction_score", "p_value", "p_value_fdr",
                      "ligand_rank", "receptor_rank"]
        a = a[cols].sort_values(KEY, ignore_index=True)
        b = b[cols].sort_values(KEY, ignore_index=True)
        pd.testing.assert_frame_equal(a, b)

    def test_sort_by_checked(self, planted):
        with pytest.raises(ValueError, match="sort_by must be"):
            _run(planted, sort_by="score")
        with pytest.raises(ValueError, match="needs calculate_pvalues"):
            _run(planted, sort_by="p_value", calculate_pvalues=False,
                 background=None)

    def test_top_n_plots_still_use_the_score(self, planted):
        """The plotting fallbacks take the n_top highest scores whatever the
        row order (sort_by='p_value' puts the significant rows first)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        res = _run(planted, sort_by="p_value")
        shuffled = res.sample(frac=1.0, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()):
                for frame in (res, shuffled):
                    fig = la.pl.plotCCCHeatmap(
                        frame, n_top=5, filter_significant=False,
                        filter_by_interaction_score=False, return_fig=True)
                    plt.close("all")
        # the same five interactions either way
        top = res.nlargest(5, "interaction_score")
        assert set(top.index) == set(
            shuffled.sort_values("interaction_score", ascending=False,
                                 kind="mergesort").index[:5])
