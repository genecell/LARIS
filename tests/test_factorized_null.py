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
