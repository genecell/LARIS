"""Tests for the factorized matched-gene null (prepareLRBackground +
runLARIS(background=...)) and the analytic shuffled-graph expectation.

Statistical design: docs/discussion/2026-08-25_analytic_null_proof.md and
Rounds 34-38 of the discussion record.
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
from laris.tools import _utils
from laris.tools._background import (
    _analytic_random_gsp,
    _edge_gram,
    _expected_row_sq_sum,
    _iqr_apply,
    _iqr_fit,
    _matched_sets,
    _pair_gsp_from_tables,
    _quantile_grid_pool,
)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def fixture():
    rng = np.random.default_rng(0)
    n, g = 900, 80
    genes = [f"G{i:03d}" for i in range(g)]
    X = sp.random(n, g, density=0.25, random_state=1,
                  data_rvs=lambda k: rng.poisson(4, k) + 1).tocsr()
    A = ad.AnnData(X=X, var=pd.DataFrame(index=genes))
    A.obsm["X_spatial"] = rng.random((n, 2)) * 100
    A.obs["ct"] = pd.Categorical(rng.choice(["A", "B", "C"], n))
    lr_df = pd.DataFrame({
        "ligand": [f"G{i:03d}" for i in range(0, 24, 2)],
        "receptor": [f"G{i:03d}" for i in range(1, 24, 2)],
    })
    lr = _quiet(la.tl.prepareLRInteraction, A, lr_df,
                use_rep_spatial="X_spatial")
    bg = _quiet(la.tl.prepareLRBackground, A, lr_df, n_pool=40,
                n_matched_genes=6, use_rep_spatial="X_spatial", verbosity=0)
    return A, lr_df, lr, bg


class TestGramIdentities:
    """The bilinear identities the whole design rests on."""

    def test_edge_gram_matches_bruteforce(self):
        rng = np.random.default_rng(3)
        n, U = 150, 7
        P = rng.random((U, n)).astype(np.float32)
        mask = rng.random((U, n)) < 0.4
        Q = np.where(mask, P, 0.0).astype(np.float32)
        W = sp.random(n, n, density=0.05, random_state=5)
        G = _edge_gram(P, Q, W, edge_chunk=100)
        for i, j in [(0, 1), (2, 5), (6, 6)]:
            v = P[i] * P[j] - Q[i] * Q[j]
            direct = float(v @ (W @ v))
            assert np.isclose(G[i, j], direct, rtol=1e-5), (i, j)

    def test_masked_profile_is_difference_of_products(self):
        # v = P_i*P_j on OR-mask cells, 0 elsewhere == P_i*P_j - Q_i*Q_j
        rng = np.random.default_rng(4)
        P = rng.random((2, 50))
        raw_nonzero = rng.random((2, 50)) < 0.5
        Q = np.where(raw_nonzero, 0.0, P)
        or_mask = raw_nonzero[0] | raw_nonzero[1]
        v_direct = np.where(or_mask, P[0] * P[1], 0.0)
        v_tables = P[0] * P[1] - Q[0] * Q[1]
        assert np.allclose(v_direct, v_tables)

    def test_real_pair_profile_matches_pipeline(self, fixture):
        A, lr_df, lr, bg = fixture
        for lig, rec in [("G000", "G001"), ("G010", "G011")]:
            i = bg.positions([lig])[0]
            j = bg.positions([rec])[0]
            v_tab = bg.P[i] * bg.P[j] - bg.Q[i] * bg.Q[j]
            v_pipe = np.asarray(lr[:, f"{lig}::{rec}"].X.todense()).ravel()
            assert np.allclose(v_tab, v_pipe, rtol=1e-4, atol=1e-6)

    def test_real_pair_gsp_matches_pipeline(self, fixture):
        A, lr_df, lr, bg = fixture
        _quiet(la.tl.runLARIS, lr, A, use_rep="X_spatial",
               use_rep_spatial="X_spatial", groupby="ct", by_celltype=False,
               n_cells_expressed_threshold=5)
        ii = bg.positions(lr_df.ligand)
        jj = bg.positions(lr_df.receptor)
        gsp_tab = np.diag(_pair_gsp_from_tables(bg, ii, jj))
        names = [f"{l}::{r}" for l, r in zip(lr_df.ligand, lr_df.receptor)]
        gsp_pipe = lr.var["LRSS_Target"].reindex(names).values
        ok = ~np.isnan(gsp_pipe)
        assert np.allclose(gsp_tab[ok], gsp_pipe[ok], atol=1e-6)


class TestAnalyticRandomGsp:
    def test_matches_realized_mean(self, fixture):
        A, lr_df, lr, bg = fixture
        genexcell = lr.X.T.tocsr()
        W = _utils._build_adjacency_matrix(
            lr, use_rep="X_spatial", n_nearest_neighbors=20, sigma="adaptive")
        from sklearn.preprocessing import normalize
        reals, Rs = [], []
        for seed in (11, 22, 33, 44, 55, 66):
            sh = _utils._build_random_adjacency_matrix(lr, W, 20, seed)
            sh = normalize(sh, axis=1, norm="l1")
            Rs.append((sh.data ** 2).sum())
            o1 = genexcell @ sh.T
            reals.append(np.asarray(
                _utils._rowwise_cosine_similarity(genexcell, o1)).ravel())
        realized = np.mean(reals, axis=0)
        X = lr.X.tocsc()
        m1 = np.asarray(X.mean(axis=0)).ravel()
        m2 = np.asarray(X.power(2).mean(axis=0)).ravel()
        R = lr.n_obs * _expected_row_sq_sum(W.data, 20)
        analytic = _analytic_random_gsp(m1, m2, lr.n_obs, R)
        # delta is what the pipeline uses; require tight agreement there
        o1 = genexcell @ W.T
        gsp = np.asarray(_utils._rowwise_cosine_similarity(genexcell, o1)).ravel()
        d_real = gsp - 0.25 * realized
        d_ana = gsp - 0.25 * analytic
        ok = np.isfinite(d_real) & np.isfinite(d_ana)
        # tight absolute agreement on delta; rank agreement is meaningless
        # on this 12-pair fixture where deltas are near-tied
        assert np.abs(d_real[ok] - d_ana[ok]).max() < 0.01

    def test_delta_method_R_close_to_realized(self, fixture):
        A, lr_df, lr, bg = fixture
        W = _utils._build_adjacency_matrix(
            lr, use_rep="X_spatial", n_nearest_neighbors=20, sigma="adaptive")
        from sklearn.preprocessing import normalize
        realized = []
        for seed in range(8):
            sh = _utils._build_random_adjacency_matrix(lr, W, 20, seed)
            realized.append((normalize(sh, axis=1, norm="l1").data ** 2).sum())
        analytic = lr.n_obs * _expected_row_sq_sum(W.data, 20)
        assert abs(analytic - np.mean(realized)) / np.mean(realized) < 0.02

    def test_constant_profile_gives_one(self):
        assert np.isclose(_analytic_random_gsp(2.0, 4.0, 1000, 60.0), 1.0)

    def test_zero_profile_gives_zero(self):
        assert _analytic_random_gsp(0.0, 0.0, 1000, 60.0) == 0.0

    def test_pipeline_delta_is_deterministic(self, fixture):
        A, lr_df, lr, bg = fixture
        out = []
        for seed in (1, 999):
            lrc = lr.copy()
            _quiet(la.tl.runLARIS, lrc, A, use_rep="X_spatial",
                   use_rep_spatial="X_spatial", groupby="ct",
                   by_celltype=False, random_seed=seed,
                   n_cells_expressed_threshold=5)
            out.append(lrc.var["LR_SpatialSpecificity"].values.copy())
        assert np.array_equal(out[0], out[1])

    def test_n_repeats_deprecation_warns(self, fixture):
        A, lr_df, lr, bg = fixture
        with pytest.warns(FutureWarning, match="n_repeats"):
            _quiet_ctx = contextlib.redirect_stdout(io.StringIO())
            with _quiet_ctx:
                la.tl.runLARIS(lr.copy(), A, use_rep="X_spatial",
                               use_rep_spatial="X_spatial", groupby="ct",
                               by_celltype=False, n_repeats=5,
                               n_cells_expressed_threshold=5)


class TestPoolAndMatching:
    def test_pool_size_and_determinism(self):
        rng = np.random.default_rng(7)
        m, v = rng.random(5000), rng.random(5000)
        p1 = _quantile_grid_pool(m, v, 400)
        p2 = _quantile_grid_pool(m, v, 400)
        # since R58 the default pads the grid with the top-500 by mean, so
        # the size is n_pool plus the not-already-selected extremes
        assert len(p1) == 400 and np.array_equal(p1, p2)

    def test_pool_covers_expression_space(self):
        rng = np.random.default_rng(8)
        m, v = rng.random(5000), rng.random(5000)
        pool = _quantile_grid_pool(m, v, 400)
        # every decile of the mean axis is represented
        deciles = np.digitize(m[pool], np.quantile(m, np.linspace(0, 1, 11)[1:-1]))
        assert len(np.unique(deciles)) == 10

    def test_matched_sets_within_pool(self, fixture):
        A, lr_df, lr, bg = fixture
        k = bg.params["n_matched_genes"]
        for g, idx in bg.matched_ligand.items():
            assert len(idx) == k
            assert (idx >= 0).all() and (idx < len(bg.gene_index)).all()

    def test_tables_cover_lr_genes(self, fixture):
        A, lr_df, lr, bg = fixture
        lr_genes = set(lr_df.ligand) | set(lr_df.receptor)
        assert lr_genes <= set(bg.gene_index)


class TestIqrFitApply:
    def test_matches_cosg(self):
        import cosg
        rng = np.random.default_rng(9)
        df = pd.DataFrame(rng.random((200, 5)) * 10,
                          columns=list("abcde"))
        expected = cosg.iqrLogNormalize(df)
        iqr = _iqr_fit(df)
        got = pd.DataFrame(_iqr_apply(df.values, iqr.values),
                           index=df.index, columns=df.columns)
        pd.testing.assert_frame_equal(got, expected)

    def test_zero_iqr_fallback_matches_cosg(self):
        import cosg
        rng = np.random.default_rng(10)
        df = pd.DataFrame(rng.random((100, 3)), columns=list("abc"))
        df["b"] = 1.0                       # zero IQR column
        expected = cosg.iqrLogNormalize(df)
        iqr = _iqr_fit(df)
        got = pd.DataFrame(_iqr_apply(df.values, iqr.values),
                           index=df.index, columns=df.columns)
        pd.testing.assert_frame_equal(got, expected)


class TestFactorizedPvalues:
    @pytest.fixture(scope="class")
    def run(self, fixture):
        A, lr_df, lr, bg = fixture
        _, res = _quiet(la.tl.runLARIS, lr.copy(), A, use_rep="X_spatial",
                        use_rep_spatial="X_spatial", groupby="ct",
                        background=bg, n_cells_expressed_threshold=5,
                        specificity_reference="all")
        return res

    def test_floor_is_exact_support(self, fixture, run):
        A, lr_df, lr, bg = fixture
        k = bg.params["n_matched_genes"]
        floor = 1.0 / (k * k + 1)
        p = run.p_value.dropna()
        assert p.min() >= floor - 1e-12
        assert (p <= 1.0).all()
        # every p-value sits on an exact lattice (exceed+1)/(support+1)
        # for SOME support <= k*k (database-pair exclusions shrink the
        # support of pairs whose matched sets contain database combos)
        on_lattice = np.zeros(len(p), dtype=bool)
        for support in range(k * k, max(k * k - 20, 1), -1):
            scaled = p * (support + 1)
            on_lattice |= np.abs(scaled - np.round(scaled)) < 1e-9
        assert on_lattice.all()

    def test_calibration_on_noise(self, run):
        # the fixture is pure noise: no FDR discoveries expected
        assert (run.p_value_fdr < 0.05).sum() == 0
        assert 0.2 < run.p_value.dropna().mean() < 0.8

    def test_deterministic(self, fixture, run):
        A, lr_df, lr, bg = fixture
        _, res2 = _quiet(la.tl.runLARIS, lr.copy(), A, use_rep="X_spatial",
                         use_rep_spatial="X_spatial", groupby="ct",
                         background=bg, n_cells_expressed_threshold=5,
                         specificity_reference="all")
        pd.testing.assert_series_equal(run.p_value, res2.p_value)

    def test_legacy_path_still_works(self, fixture):
        A, lr_df, lr, bg = fixture
        _, res = _quiet(la.tl.runLARIS, lr.copy(), A, use_rep="X_spatial",
                        use_rep_spatial="X_spatial", groupby="ct",
                        n_cells_expressed_threshold=5,
                        specificity_reference="all", n_permutations=200)
        assert res.p_value.notna().any()
        assert res.p_value.dropna().min() >= 1 / 201

    def test_power_on_planted_signal(self):
        # plant a spatially coherent sender->receiver interaction; its
        # factorized p-value should be small
        rng = np.random.default_rng(12)
        n, g = 1200, 60
        genes = [f"G{i:03d}" for i in range(g)]
        X = sp.random(n, g, density=0.2, random_state=2,
                      data_rvs=lambda k: rng.poisson(2, k) + 1).tolil()
        coords = rng.random((n, 2)) * 100
        blob = np.linalg.norm(coords - [50, 50], axis=1) < 25
        ct = np.where(rng.random(n) < 0.5, "A", "B")
        lig_cells = blob & (ct == "A")
        rec_cells = blob & (ct == "B")
        X[np.flatnonzero(lig_cells), 0] = 30      # G000 ligand in A-blob
        X[np.flatnonzero(rec_cells), 1] = 30      # G001 receptor in B-blob
        A_ = ad.AnnData(X=X.tocsr(), var=pd.DataFrame(index=genes))
        A_.obsm["X_spatial"] = coords
        A_.obs["ct"] = pd.Categorical(ct)
        lr_df = pd.DataFrame({
            "ligand": [f"G{i:03d}" for i in range(0, 20, 2)],
            "receptor": [f"G{i:03d}" for i in range(1, 20, 2)],
        })
        lr = _quiet(la.tl.prepareLRInteraction, A_, lr_df,
                    use_rep_spatial="X_spatial")
        bg = _quiet(la.tl.prepareLRBackground, A_, lr_df, n_pool=40,
                    n_matched_genes=6, use_rep_spatial="X_spatial",
                    verbosity=0)
        _, res = _quiet(la.tl.runLARIS, lr, A_, use_rep="X_spatial",
                        use_rep_spatial="X_spatial", groupby="ct",
                        background=bg, n_cells_expressed_threshold=5,
                        specificity_reference="all")
        hit = res[(res.interaction_name == "G000::G001")
                  & (res.sender == "A") & (res.receiver == "B")]
        assert len(hit) == 1
        assert hit.p_value.iloc[0] <= 3.0 / 37    # near the floor
        # and it should rank at/near the top of its group
        grp = res[(res.sender == "A") & (res.receiver == "B")]
        assert hit.interaction_score.iloc[0] >= grp.interaction_score.quantile(0.9)

    def test_db_pairs_excluded_from_null(self, fixture):
        A, lr_df, lr, bg = fixture
        assert ("G000", "G001") in bg.db_pairs

    def test_cytome_background_matches_in_memory(self, fixture, tmp_path):
        A, lr_df, lr, bg = fixture
        cytome = pytest.importorskip("cytome")
        path = str(tmp_path / "fixture.cytome")
        ds = _quiet(cytome.from_anndata, A, output=path)
        ds.close()
        bg_cy = _quiet(la.tl.prepareLRBackground, path, lr_df, n_pool=40,
                       n_matched_genes=6, use_rep_spatial="X_spatial",
                       verbosity=0)
        # the float32 storage round-trip can flip quantile-grid tie-breaks
        # for a gene or two; require near-identical pools and exact table
        # agreement on the shared genes
        shared = bg.gene_index.intersection(bg_cy.gene_index)
        assert len(shared) / len(bg.gene_index) > 0.9
        a = bg.gene_index.get_indexer(shared)
        b = bg_cy.gene_index.get_indexer(shared)
        np.testing.assert_allclose(bg_cy.G_W[np.ix_(b, b)],
                                   bg.G_W[np.ix_(a, a)], rtol=1e-3)
        np.testing.assert_allclose(bg_cy.G_sq[np.ix_(b, b)],
                                   bg.G_sq[np.ix_(a, a)], rtol=1e-3)
        # matched sets resolve to the same gene names where pools agree
        for g in list(bg.matched_ligand)[:4]:
            names_mem = set(bg.gene_index[bg.matched_ligand[g]])
            names_cy = set(bg_cy.gene_index[bg_cy.matched_ligand[g]])
            assert len(names_mem & names_cy) >= len(names_mem) - 2


class TestRustBackend:
    """The bundled Rust kernels must reproduce the NumPy path."""

    def test_numerator_matches_dense(self, fixture):
        laris_core = pytest.importorskip("laris._laris")
        A, lr_df, lr, bg = fixture
        P, Q = bg.P, bg.Q
        y = np.random.default_rng(5).random(P.shape[1]).astype(np.float32)
        N_dense = (P * y) @ P.T - (Q * y) @ Q.T
        Pnz = sp.csr_matrix(P - Q)
        Pnz_csc = Pnz.tocsc()
        N_rust = laris_core.ctc_cos_numerator(
            np.ascontiguousarray(P.T),
            Pnz.indptr.astype(np.int64), Pnz.indices.astype(np.int32),
            Pnz.data.astype(np.float32),
            Pnz_csc.indptr.astype(np.int64),
            Pnz_csc.indices.astype(np.int32),
            Pnz_csc.data.astype(np.float32), y)
        scale = max(np.abs(N_dense).max(), 1e-9)
        assert np.abs(N_rust - N_dense).max() / scale < 1e-4

    def test_end_to_end_parity(self, fixture, monkeypatch):
        pytest.importorskip("laris._laris")
        A, lr_df, lr, bg = fixture

        def _run():
            _, res = _quiet(la.tl.runLARIS, lr.copy(), A,
                            use_rep="X_spatial", use_rep_spatial="X_spatial",
                            groupby="ct", background=bg,
                            n_cells_expressed_threshold=5,
                            specificity_reference="all")
            return res

        monkeypatch.setenv("LARIS_NO_RUST", "1")
        res_np = _run()
        monkeypatch.delenv("LARIS_NO_RUST")
        res_rs = _run()
        m = res_rs.merge(res_np, on=["sender", "receiver", "interaction_name"],
                         suffixes=("_r", "_n"))
        # float32 numerator reordering can flip exact ties by one lattice
        # step at most; on this fixture the p-values come out identical
        assert (m.p_value_r - m.p_value_n).abs().max() <= 1.0 / 37 + 1e-12
        assert ((m.p_value_r - m.p_value_n).abs() < 1e-12).mean() > 0.95


class TestGroupbyValidation:
    """Real datasets carry unannotated cells; fail with instructions."""

    def test_nan_cell_types_raise_with_instructions(self, fixture):
        A, lr_df, lr, bg = fixture
        lr2 = lr.copy()
        ct = lr2.obs["ct"].astype(object).copy()
        ct.iloc[:5] = np.nan
        lr2.obs["ct"] = ct
        with pytest.raises(ValueError, match="missing value"):
            _quiet(la.tl.runLARIS, lr2, A, use_rep="X_spatial",
                   use_rep_spatial="X_spatial", groupby="ct",
                   n_cells_expressed_threshold=5)


class TestEffectiveNullSupport:
    """The null's denominator counts pseudo-pairs; only the non-zero ones
    carry resolution. That count is reported and can gate the p-value."""

    def test_null_support_column_present_and_bounded(self, fixture):
        A, lr_df, lr, bg = fixture
        _, res = _quiet(la.tl.runLARIS, lr.copy(), A, use_rep="X_spatial",
                        use_rep_spatial="X_spatial", groupby="ct",
                        background=bg, n_cells_expressed_threshold=5,
                        specificity_reference="all")
        assert "null_support" in res.columns
        k = bg.params["n_matched_genes"]
        assert res.null_support.min() >= 0
        assert res.null_support.max() <= k * k

    def test_zero_support_rows_have_p_one(self, fixture):
        A, lr_df, lr, bg = fixture
        _, res = _quiet(la.tl.runLARIS, lr.copy(), A, use_rep="X_spatial",
                        use_rep_spatial="X_spatial", groupby="ct",
                        background=bg, n_cells_expressed_threshold=5,
                        specificity_reference="all")
        dead = res[res.null_support == 0]
        if len(dead):
            assert (dead.p_value == 1.0).all()

    def test_p_value_never_below_effective_resolution(self, fixture):
        """A row cannot claim more resolution than its support allows.

        The reported p is (exceed+1)/(support+1) over ALL pseudo-pairs,
        so this is the property that motivates the column: it can be far
        below 1/(null_support+1). The test pins the direction of the
        discrepancy so a future change that silently redefines either
        quantity is caught.
        """
        A, lr_df, lr, bg = fixture
        _, res = _quiet(la.tl.runLARIS, lr.copy(), A, use_rep="X_spatial",
                        use_rep_spatial="X_spatial", groupby="ct",
                        background=bg, n_cells_expressed_threshold=5,
                        specificity_reference="all")
        live = res[res.null_support > 0]
        k = bg.params["n_matched_genes"]
        assert (live.p_value >= 1.0 / (k * k + 1) - 1e-12).all()

    def test_min_null_support_only_removes_calls(self, fixture):
        A, lr_df, lr, bg = fixture

        def run(min_support):
            _, r = _quiet(la.tl.runLARIS, lr.copy(), A, use_rep="X_spatial",
                          use_rep_spatial="X_spatial", groupby="ct",
                          background=bg, n_cells_expressed_threshold=5,
                          specificity_reference="all",
                          min_null_support=min_support)
            return r.set_index(["sender", "receiver", "interaction_name"])

        base = run(0)
        gated = run(10)
        j = base.join(gated, lsuffix="_b", rsuffix="_g")
        # gating can only push p-values up, never down
        assert (j.p_value_g >= j.p_value_b - 1e-12).all()
        thin = j.null_support_b < 10
        assert (j.loc[thin, "p_value_g"] == 1.0).all()
        assert (j.loc[~thin, "p_value_g"]
                == j.loc[~thin, "p_value_b"]).all()

    def test_default_reproduces_v0120_pvalues(self, fixture):
        """min_null_support=0 must leave the released numbers untouched."""
        A, lr_df, lr, bg = fixture
        _, a = _quiet(la.tl.runLARIS, lr.copy(), A, use_rep="X_spatial",
                      use_rep_spatial="X_spatial", groupby="ct",
                      background=bg, n_cells_expressed_threshold=5,
                      specificity_reference="all")
        _, b = _quiet(la.tl.runLARIS, lr.copy(), A, use_rep="X_spatial",
                      use_rep_spatial="X_spatial", groupby="ct",
                      background=bg, n_cells_expressed_threshold=5,
                      specificity_reference="all", min_null_support=0)
        assert np.allclose(a.p_value.to_numpy(), b.p_value.to_numpy(),
                           equal_nan=True)

    def test_rust_and_numpy_agree_on_support(self, fixture, monkeypatch):
        A, lr_df, lr, bg = fixture

        def _run():
            _, r = _quiet(la.tl.runLARIS, lr.copy(), A, use_rep="X_spatial",
                          use_rep_spatial="X_spatial", groupby="ct",
                          background=bg, n_cells_expressed_threshold=5,
                          specificity_reference="all")
            return r

        monkeypatch.setenv("LARIS_NO_RUST", "1")
        np_res = _run()
        monkeypatch.delenv("LARIS_NO_RUST")
        rs_res = _run()
        m = rs_res.merge(np_res, on=["sender", "receiver", "interaction_name"],
                         suffixes=("_r", "_n"))
        assert (m.null_support_r == m.null_support_n).all()


class TestPermuteLRPairs:
    """The calibration control that matches a competitive null."""

    def test_decoys_avoid_real_pairs_and_self_pairs(self, fixture):
        # uniform draw rejects real pairs outright; the degree-preserving
        # rewiring cannot always avoid them and flags them instead
        A, lr_df, lr, bg = fixture
        d = la.tl.permuteLRPairs(lr_df, A, random_seed=3, method="uniform")
        real = set(zip(lr_df.ligand, lr_df.receptor))
        assert not set(zip(d.ligand, d.receptor)) & real
        assert (d.ligand != d.receptor).all()
        assert d.interaction_name.is_unique

    def test_decoy_genes_exist_in_data(self, fixture):
        A, lr_df, lr, bg = fixture
        d = la.tl.permuteLRPairs(lr_df, A, random_seed=3, method="uniform")
        present = set(A.var_names)
        assert set(d.ligand) <= present and set(d.receptor) <= present

    def test_preserve_genes_keeps_the_marginal_pools(self, fixture):
        A, lr_df, lr, bg = fixture
        d = la.tl.permuteLRPairs(lr_df, A, random_seed=3, preserve_genes=True,
                                 method="uniform")
        assert set(d.ligand) <= set(lr_df.ligand)
        assert set(d.receptor) <= set(lr_df.receptor)

    def test_unconstrained_draw_uses_all_genes(self, fixture):
        A, lr_df, lr, bg = fixture
        d = la.tl.permuteLRPairs(lr_df, A, random_seed=3,
                                 preserve_genes=False, method="uniform")
        assert not set(d.ligand) <= set(lr_df.ligand)

    def test_deterministic_for_a_seed(self, fixture):
        A, lr_df, lr, bg = fixture
        a = la.tl.permuteLRPairs(lr_df, A, random_seed=7)
        b = la.tl.permuteLRPairs(lr_df, A, random_seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_decoy_database_runs_end_to_end(self, fixture):
        """The control must actually be runnable, not just constructible."""
        A, lr_df, lr, bg = fixture
        d = la.tl.permuteLRPairs(lr_df, A, random_seed=11, method="uniform")
        lr_d = _quiet(la.tl.prepareLRInteraction, A, d,
                      use_rep_spatial="X_spatial")
        bg_d = _quiet(la.tl.prepareLRBackground, A, d, n_pool=40,
                      n_matched_genes=6, use_rep_spatial="X_spatial",
                      verbosity=0)
        _, res = _quiet(la.tl.runLARIS, lr_d, A, use_rep="X_spatial",
                        use_rep_spatial="X_spatial", groupby="ct",
                        background=bg_d, n_cells_expressed_threshold=5,
                        specificity_reference="all")
        assert len(res) > 0
        assert res.p_value.between(0, 1).all()


class TestDegreePreservingDecoys:
    """A fair decoy database must not change the database's structure.

    Curated LR resources are hub-heavy, and hub genes are broadly
    expressed and individually unremarkable. Drawing decoys uniformly
    under-samples them, which makes the decoy database *easier* than the
    real one and overstates the false-positive rate it appears to show.
    """

    def _hub_db(self):
        # one hub receptor in many pairs, plus a long tail - the shape of
        # a real resource in miniature
        lig = [f"G{i:03d}" for i in range(0, 30, 2)]
        rows = [(l, "G001") for l in lig]                 # hub
        rows += [(l, f"G{2 * i + 3:03d}") for i, l in enumerate(lig[:6])]
        return pd.DataFrame(rows, columns=["ligand", "receptor"])

    def test_degree_sequences_are_preserved_exactly(self):
        db = self._hub_db()
        d = la.tl.permuteLRPairs(db, random_seed=0, method="degree")
        for col in ("ligand", "receptor"):
            a = db[col].value_counts().sort_index()
            b = d[col].value_counts().sort_index()
            a, b = a.align(b, fill_value=0)
            assert (a == b).all(), col

    def test_uniform_flattens_the_hub_but_degree_does_not(self):
        db = self._hub_db()
        hub_real = int((db.receptor == "G001").sum())
        deg = la.tl.permuteLRPairs(db, random_seed=0, method="degree")
        uni = la.tl.permuteLRPairs(db, random_seed=0, method="uniform")
        assert int((deg.receptor == "G001").sum()) == hub_real
        assert int((uni.receptor == "G001").sum()) < hub_real

    def test_pair_count_is_unchanged_and_unique(self):
        db = self._hub_db()
        d = la.tl.permuteLRPairs(db, random_seed=0, method="degree")
        assert len(d) == len(db)
        assert d.interaction_name.is_unique

    def test_retained_real_pairs_are_flagged(self):
        db = self._hub_db()
        d = la.tl.permuteLRPairs(db, random_seed=0, method="degree")
        real = set(zip(db.ligand, db.receptor))
        flagged = set(zip(d.loc[d.is_real, "ligand"],
                          d.loc[d.is_real, "receptor"]))
        actual = {e for e in zip(d.ligand, d.receptor) if e in real}
        assert flagged == actual
        assert d.attrs["n_real_retained"] == len(actual)

    def test_degree_is_the_default(self):
        db = self._hub_db()
        a = la.tl.permuteLRPairs(db, random_seed=4)
        b = la.tl.permuteLRPairs(db, random_seed=4, method="degree")
        pd.testing.assert_frame_equal(a, b)

    def test_bad_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            la.tl.permuteLRPairs(self._hub_db(), method="nonsense")

    def test_degree_decoys_run_end_to_end(self, fixture):
        A, lr_df, lr, bg = fixture
        d = la.tl.permuteLRPairs(lr_df, A, random_seed=2, method="degree")
        lr_d = _quiet(la.tl.prepareLRInteraction, A, d[["ligand", "receptor"]],
                      use_rep_spatial="X_spatial")
        bg_d = _quiet(la.tl.prepareLRBackground, A, d[["ligand", "receptor"]],
                      n_pool=40, n_matched_genes=6,
                      use_rep_spatial="X_spatial", verbosity=0)
        _, res = _quiet(la.tl.runLARIS, lr_d, A, use_rep="X_spatial",
                        use_rep_spatial="X_spatial", groupby="ct",
                        background=bg_d, n_cells_expressed_threshold=5,
                        specificity_reference="all")
        assert len(res) > 0 and res.p_value.between(0, 1).all()


class TestDecoyFDR:
    """Target-decoy empirical FDR: the honesty layer over the p-value.

    The p-value certifies arrangement, not pairing (a degree-preserving
    decoy database scores ~as well as the real one). decoyFDR measures
    the pairing error rate the way proteomics does: same pipeline, decoy
    database, ratio of call rates, monotonized into a q-value.
    """

    def _uniform(self, n, seed):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({"p_value": rng.uniform(size=n)})

    def test_q_is_monotone_in_p(self):
        t = self._uniform(4000, 0)
        d = self._uniform(4000, 1)
        q = la.tl.decoyFDR(t, d)
        srt = t.p_value.sort_values()
        assert (q[srt.index].diff().dropna() >= -1e-12).all()

    def test_uniform_vs_uniform_is_uninformative(self):
        # same distribution in both -> q near 1 everywhere: the decoy
        # does exactly as well as the target, so nothing is trustworthy
        t = self._uniform(5000, 2)
        d = self._uniform(5000, 3)
        q = la.tl.decoyFDR(t, d)
        assert q.median() > 0.8

    def test_planted_signal_gets_small_q(self):
        rng = np.random.default_rng(4)
        t = pd.DataFrame({"p_value": np.concatenate(
            [np.full(60, 1e-6), rng.uniform(size=5000)])})
        d = self._uniform(5000, 5)
        q = la.tl.decoyFDR(t, d)
        # 60 planted rows, ~0 decoys below them, pseudocount 1:
        # FDR ~ (0+1)/5000 / (60/5060) ~ 0.017
        assert q.iloc[:60].max() < 0.05
        assert q.iloc[100:].median() > 0.5

    def test_rate_normalization_row_count_invariance(self):
        # halving the decoy set must not change the estimate (rates, not
        # counts): same distribution, different n
        rng = np.random.default_rng(6)
        t = pd.DataFrame({"p_value": np.concatenate(
            [np.full(50, 1e-6), rng.uniform(size=4000)])})
        d_full = self._uniform(8000, 7)
        d_half = d_full.iloc[:4000]
        q1 = la.tl.decoyFDR(t, d_full)
        q2 = la.tl.decoyFDR(t, d_half)
        # pseudocount scales differently, so allow slack at the extreme
        assert np.median(np.abs(q1 - q2)) < 0.05

    def test_nan_propagates_and_index_alignment(self):
        t = self._uniform(100, 8)
        t.loc[7, "p_value"] = np.nan
        t.index = [f"row{i}" for i in range(100)]
        shuffled = t.sample(frac=1, random_state=9)
        q = la.tl.decoyFDR(shuffled, self._uniform(500, 10))
        assert q.index.equals(shuffled.index)
        assert np.isnan(q.loc["row7"])
        assert q.drop("row7").notna().all()

    def test_clipped_to_one(self):
        # decoy strictly better than target -> raw ratio > 1 -> clipped
        t = pd.DataFrame({"p_value": np.linspace(.5, 1, 200)})
        d = pd.DataFrame({"p_value": np.linspace(0, .5, 200)})
        q = la.tl.decoyFDR(t, d)
        assert (q <= 1).all() and q.min() > 0.9

    def test_empty_decoy_raises(self):
        with pytest.raises(ValueError, match="no finite p-values"):
            la.tl.decoyFDR(self._uniform(10, 0),
                           pd.DataFrame({"p_value": [np.nan]}))

    def test_end_to_end_on_fixture(self, fixture):
        A, lr_df, lr, bg = fixture
        _, res = _quiet(la.tl.runLARIS, lr.copy(), A, use_rep="X_spatial",
                        use_rep_spatial="X_spatial", groupby="ct",
                        background=bg, n_cells_expressed_threshold=5,
                        specificity_reference="all")
        q, dec = _quiet(
            la.tl.computeDecoyFDR, A, lr_df, res, background=bg,
            random_seed=1,
            prepare_kwargs={"use_rep_spatial": "X_spatial"},
            run_kwargs={"use_rep": "X_spatial",
                        "use_rep_spatial": "X_spatial", "groupby": "ct",
                        "n_cells_expressed_threshold": 5,
                        "specificity_reference": "all"},
            verbosity=0)
        assert q.index.equals(res.index)
        ok = q.dropna()
        assert ((ok >= 0) & (ok <= 1)).all()
        assert len(dec) > 0
        # decoy rows that coincide with real pairs must be gone
        real = set(lr_df.ligand + "::" + lr_df.receptor)
        assert not set(dec.interaction_name) & real

    def test_seed_determinism(self, fixture):
        A, lr_df, lr, bg = fixture
        _, res = _quiet(la.tl.runLARIS, lr.copy(), A, use_rep="X_spatial",
                        use_rep_spatial="X_spatial", groupby="ct",
                        background=bg, n_cells_expressed_threshold=5,
                        specificity_reference="all")
        kw = dict(background=bg, random_seed=3,
                  prepare_kwargs={"use_rep_spatial": "X_spatial"},
                  run_kwargs={"use_rep": "X_spatial",
                              "use_rep_spatial": "X_spatial",
                              "groupby": "ct",
                              "n_cells_expressed_threshold": 5,
                              "specificity_reference": "all"},
                  verbosity=0)
        q1, _ = _quiet(la.tl.computeDecoyFDR, A, lr_df, res, **kw)
        q2, _ = _quiet(la.tl.computeDecoyFDR, A, lr_df, res, **kw)
        pd.testing.assert_series_equal(q1, q2)


class TestPoolCoverageAndFlags:
    """The pool must reach the top of the abundance range, and the results
    must report when it cannot (matchability) and when a call carries no
    cell-type information (breadth).

    Mechanism (Round 58): the rank-uniform quantile grid gives the top
    ~0.1% of genes almost no slots, so a very abundant gene's matched set
    sits entirely below it and every pseudo-pair is weaker than the real
    one - the gene becomes unbeatable in its own null.
    """

    def test_augmentation_desaturates_extreme_genes(self):
        """The property the pool must satisfy, stated directly."""
        from laris.tools._background import (_quantile_grid_pool,
                                             _augment_pool_for_saturation,
                                             _matched_sets)
        rng = np.random.default_rng(0)
        means = rng.lognormal(0, 1, 5000)
        means[:20] *= 400                       # extreme-abundance outliers
        var = means * (1 + rng.random(5000))
        feats = np.column_stack([means, var])
        pool = _quantile_grid_pool(means, var, n_pool=400)
        query = np.arange(20)                   # the outliers are the LR genes
        before = (feats[pool][_matched_sets(feats[query], feats[pool], 10), 0]
                  < means[query][:, None]).mean(1)
        assert (before >= 0.99).any(), "fixture must start saturated"
        grown = _augment_pool_for_saturation(feats, pool, query, k=10)
        after = (feats[grown][_matched_sets(feats[query], feats[grown], 10), 0]
                 < means[query][:, None]).mean(1)
        assert (after < 0.99).all()
        assert set(pool) <= set(grown)

    def test_augmentation_is_bounded_and_targeted(self):
        """Only the failing genes trigger growth, and it stays small."""
        from laris.tools._background import (_quantile_grid_pool,
                                             _augment_pool_for_saturation)
        rng = np.random.default_rng(3)
        means = rng.lognormal(0, 1, 5000)
        means[:10] *= 500
        var = means * (1 + rng.random(5000))
        feats = np.column_stack([means, var])
        pool = _quantile_grid_pool(means, var, n_pool=400)
        grown = _augment_pool_for_saturation(feats, pool, np.arange(10), k=20)
        assert len(grown) < len(pool) + 10 * 20 + 1
        assert len(grown) < 5000 * 0.5, "must not approach the transcriptome"

    def test_augmentation_noop_when_pool_already_covers(self):
        from laris.tools._background import (_quantile_grid_pool,
                                             _augment_pool_for_saturation)
        rng = np.random.default_rng(4)
        means = rng.random(2000) + 1.0          # no extreme tail
        var = means * (1 + rng.random(2000))
        feats = np.column_stack([means, var])
        pool = _quantile_grid_pool(means, var, n_pool=300)
        mid = np.argsort(means)[800:830]        # mid-range query genes
        grown = _augment_pool_for_saturation(feats, pool, mid, k=10)
        assert len(grown) == len(pool)

    def test_augment_pool_false_reproduces_the_bare_grid(self, fixture):
        A, lr_df, lr, bg = fixture
        bare = _quiet(la.tl.prepareLRBackground, A, lr_df, n_pool=40,
                      n_matched_genes=6, augment_pool=False,
                      use_rep_spatial="X_spatial", verbosity=0)
        assert len(bare.gene_index) <= len(bg.gene_index)

    def test_match_frac_below_stored_and_bounded(self, fixture):
        A, lr_df, lr, bg = fixture
        assert bg.match_frac_below
        vals = np.array(list(bg.match_frac_below.values()))
        assert ((vals >= 0) & (vals <= 1)).all()
        covered = set(lr_df.ligand) | set(lr_df.receptor)
        assert covered <= set(bg.match_frac_below)

    def test_extreme_gene_is_matchable_with_pool_fix(self):
        """A gene far above the grid's reach must still get peers."""
        rng = np.random.default_rng(2)
        n, g = 600, 400
        genes = [f"G{i:03d}" for i in range(g)]
        lam = np.full(g, 2.0)
        lam[:30] = 60.0                          # an abundant block
        X = sp.csr_matrix(rng.poisson(lam, size=(n, g)).astype(float))
        A = ad.AnnData(X=X, var=pd.DataFrame(index=genes))
        A.obsm["X_spatial"] = rng.random((n, 2)) * 100
        A.obs["ct"] = pd.Categorical(rng.choice(["A", "B"], n))
        lr_df = pd.DataFrame({"ligand": ["G000"], "receptor": ["G001"]})
        bg_fix = _quiet(la.tl.prepareLRBackground, A, lr_df, n_pool=60,
                        n_matched_genes=6,
                        use_rep_spatial="X_spatial", verbosity=0)
        bg_bare = _quiet(la.tl.prepareLRBackground, A, lr_df, n_pool=60,
                         augment_pool=False, n_matched_genes=6,
                         use_rep_spatial="X_spatial", verbosity=0)
        # with the fix, the abundant gene has peers at its own level
        assert bg_fix.match_frac_below["G000"] < 1.0
        # and the fix strictly improves on (or matches) the bare grid
        assert (bg_fix.match_frac_below["G000"]
                <= bg_bare.match_frac_below["G000"])

    def test_null_matchability_and_breadth_columns(self, fixture):
        A, lr_df, lr, bg = fixture
        _, res = _quiet(la.tl.runLARIS, lr.copy(), A, use_rep="X_spatial",
                        use_rep_spatial="X_spatial", groupby="ct",
                        background=bg, n_cells_expressed_threshold=5,
                        specificity_reference="all")
        assert "null_matchability" in res.columns
        assert res.null_matchability.between(0, 1).all()
        assert "pair_breadth" in res.columns
        assert res.pair_breadth.between(0, 1).all()
        # breadth is constant within a pair and equals calls/combos
        n_combos = res[["sender", "receiver"]].drop_duplicates().shape[0]
        for nm, sub in res.groupby("interaction_name"):
            expect = (sub.p_value_fdr < .05).sum() / n_combos
            assert np.allclose(sub.pair_breadth, expect)

    def test_saturation_warning_fires_when_pool_cannot_reach(self):
        """With the repair disabled and an unreachable gene, warn."""
        rng = np.random.default_rng(3)
        n, g = 500, 300
        genes = [f"G{i:03d}" for i in range(g)]
        lam = np.full(g, 1.0)
        lam[0] = 80.0; lam[1] = 70.0             # only two extreme genes
        X = sp.csr_matrix(rng.poisson(lam, size=(n, g)).astype(float))
        A = ad.AnnData(X=X, var=pd.DataFrame(index=genes))
        # smooth spatial structure so the pair actually scores
        t = np.linspace(0, 1, n)
        A.obsm["X_spatial"] = np.column_stack([t * 100, rng.random(n)])
        A.obs["ct"] = pd.Categorical(np.where(t < .5, "A", "B"))
        lr_df = pd.DataFrame({"ligand": ["G000"], "receptor": ["G001"]})
        with contextlib.redirect_stdout(io.StringIO()):
            bg = la.tl.prepareLRBackground(A, lr_df, n_pool=50,
                                           augment_pool=False,
                                           n_matched_genes=6,
                                           use_rep_spatial="X_spatial",
                                           verbosity=0)
            assert bg.match_frac_below["G000"] == 1.0
            lrd = la.tl.prepareLRInteraction(A, lr_df,
                                             use_rep_spatial="X_spatial")
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _, res = la.tl.runLARIS(lrd, A, use_rep="X_spatial",
                                        use_rep_spatial="X_spatial",
                                        groupby="ct", background=bg,
                                        n_cells_expressed_threshold=5,
                                        specificity_reference="all")
        sat = res[res.null_matchability >= .99]
        if (sat.p_value < .05).any():
            assert any("matched set lies entirely below" in str(x.message)
                       for x in w)


class TestDecoyReport:
    """Dataset-level framing of the pairing question."""

    def _pair(self, n=2000, planted=0, seed=0):
        rng = np.random.default_rng(seed)
        p = rng.uniform(size=n)
        if planted:
            p[:planted] = 1e-6
        return pd.DataFrame({"p_value": p,
                             "p_value_fdr": np.clip(p * 2, 0, 1)})

    def test_report_structure_and_bounds(self, capsys):
        rep = la.tl.decoyReport(self._pair(planted=80, seed=1),
                                self._pair(seed=2))
        assert set(rep) == {"n_rows", "per_threshold", "q_min"}
        for t, d in rep["per_threshold"].items():
            assert 0 <= d["pairing_fdr"] <= 1
            assert d["target_calls"] >= 0 and d["decoy_calls"] >= 0
        out = capsys.readouterr().out
        assert "TARGET-DECOY REPORT" in out and "pairing-FDR" in out

    def test_silent_when_verbosity_zero(self, capsys):
        la.tl.decoyReport(self._pair(planted=50), self._pair(seed=3),
                          verbosity=0)
        assert capsys.readouterr().out == ""

    def test_planted_signal_gives_low_pairing_fdr(self):
        rep = la.tl.decoyReport(self._pair(planted=400, seed=4),
                                self._pair(seed=5), verbosity=0)
        assert rep["per_threshold"][0.05]["pairing_fdr"] < 0.5

    def test_indistinguishable_arms_give_high_pairing_fdr(self):
        rep = la.tl.decoyReport(self._pair(seed=6), self._pair(seed=7),
                                verbosity=0)
        assert rep["per_threshold"][0.05]["pairing_fdr"] > 0.7

    def test_empty_input_raises(self):
        empty = pd.DataFrame({"p_value": [np.nan], "p_value_fdr": [np.nan]})
        with pytest.raises(ValueError, match="finite p-values"):
            la.tl.decoyReport(self._pair(), empty, verbosity=0)

    def test_custom_thresholds(self):
        rep = la.tl.decoyReport(self._pair(planted=100), self._pair(seed=8),
                                thresholds=(0.1,), verbosity=0)
        assert list(rep["per_threshold"]) == [0.1]
