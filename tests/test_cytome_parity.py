"""In-memory and cytome inputs give the same exact null and the same calls.

Up to v0.14.0 they did not (Xenium benchmark, 10,000 cells: 817 calls in
memory, 780 from a cytome with streaming COSG, 1,024 with in-memory COSG):

1. the cytome background re-ran matching on its loaded pool + LR subset,
   standardising the (mean, variance) features on that subset rather than
   on the whole matrix - ~70% of matched sets differed by 1-5 genes;
2. it ignored ``matching``, ``exclude_self`` and ``layer``;
3. gene-subset cytome reads went through a dense float32 buffer, rounding
   float64 data;
4. ``runLARIS(cosg_backend='memory')`` on a cytome loaded only the LR genes,
   so every matched gene got specificity 0 (silently anti-conservative).

Everything here is compared bit for bit, not to a tolerance: the two paths
are meant to compute the same numbers, not similar ones.
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

cytome = pytest.importorskip("cytome")

KW = dict(n_pool=40, n_matched_genes=6, use_rep_spatial="X_spatial",
          verbosity=0)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*args, **kwargs)


def _make(dtype, reverse_genes, n=900, g=80):
    """Planted ligand/receptor signal, non-integer values like normalised
    data; optionally a var order that is NOT alphabetical."""
    rng = np.random.default_rng(0)
    genes = [f"G{i:03d}" for i in range(g)]
    X = sp.random(n, g, density=0.25, random_state=1,
                  data_rvs=lambda k: rng.poisson(4, k) + 1).tolil()
    coords = rng.random((n, 2)) * 100
    blob = np.linalg.norm(coords - [50, 50], axis=1) < 25
    ct = np.where(rng.random(n) < 0.5, "A", "B")
    X[np.flatnonzero(blob & (ct == "A")), 0] = 30
    X[np.flatnonzero(blob & (ct == "B")), 1] = 30
    # dense non-LR genes above the planted pair: every LR gene then has
    # matched peers on both sides (no truncated set for the planted pair)
    X[:, g - 10:] = rng.poisson(6, (n, 10)) + 1
    X = X.tocsr()
    X.data = np.log1p(X.data * 1.7)
    order = list(range(g))[::-1] if reverse_genes else list(range(g))
    A = ad.AnnData(X=sp.csr_matrix(X[:, order], dtype=dtype),
                   var=pd.DataFrame(index=[genes[i] for i in order]))
    A.obsm["X_spatial"] = coords
    A.obs["ct"] = pd.Categorical(ct)
    lr_df = pd.DataFrame({
        "ligand": [f"G{i:03d}" for i in range(0, 24, 2)],
        "receptor": [f"G{i:03d}" for i in range(1, 24, 2)],
    })
    return A, lr_df


@pytest.fixture(scope="module",
                params=[(np.float32, False), (np.float32, True),
                        (np.float64, False)],
                ids=["float32", "float32-unsorted-genes", "float64"])
def pair(request, tmp_path_factory):
    """(in-memory AnnData, cytome path, lr_df) holding identical data.

    The in-memory side is read back from the cytome, so both sides start
    from exactly the stored values and any difference is the pipeline's.
    """
    dtype, reverse = request.param
    A, lr_df = _make(dtype, reverse)
    path = str(tmp_path_factory.mktemp("cytome") / "fixture.cytome")
    _quiet(cytome.from_anndata, A, output=path).close()
    A_mem = la.pp.readCytome(path)
    assert A_mem.X.dtype == dtype
    return A_mem, path, lr_df


def _assert_same_background(a, b):
    assert a.gene_index.equals(b.gene_index)          # same genes, same order
    for side in ("matched_ligand", "matched_receptor"):
        ma, mb = getattr(a, side), getattr(b, side)
        assert list(ma) == list(mb)
        for gene in ma:
            np.testing.assert_array_equal(ma[gene], mb[gene],
                                          err_msg=f"{side}[{gene}]")
    for table in ("P", "Q", "G_W", "G_W2", "G_dot", "G_sq"):
        np.testing.assert_array_equal(getattr(a, table), getattr(b, table),
                                      err_msg=table)
    assert a.R == b.R
    assert list(a.match_frac_below) == list(b.match_frac_below)
    np.testing.assert_array_equal(list(a.match_frac_below.values()),
                                  list(b.match_frac_below.values()))
    assert a.edge_limited == b.edge_limited
    assert a.db_pairs == b.db_pairs


class TestBackgroundParity:

    def test_cytome_background_identical_to_in_memory(self, pair):
        A, path, lr_df = pair
        _assert_same_background(
            _quiet(la.tl.prepareLRBackground, A, lr_df, **KW),
            _quiet(la.tl.prepareLRBackground, path, lr_df, **KW))

    @pytest.mark.parametrize("matching,exclude_self",
                             [("knn", True), ("balanced", False)])
    def test_matching_parameters_reach_the_cytome_path(
            self, pair, matching, exclude_self):
        A, path, lr_df = pair
        kw = dict(KW, matching=matching, exclude_self=exclude_self)
        bg_mem = _quiet(la.tl.prepareLRBackground, A, lr_df, **kw)
        bg_cy = _quiet(la.tl.prepareLRBackground, path, lr_df, **kw)
        assert bg_cy.params["matching"] == matching
        assert bg_cy.params["exclude_self"] == exclude_self
        _assert_same_background(bg_mem, bg_cy)
        # ...and they change the null, i.e. they were not silently dropped
        default = _quiet(la.tl.prepareLRBackground, path, lr_df, **KW)
        assert any(
            set(bg_cy.matched_ligand[g]) != set(default.matched_ligand[g])
            or set(bg_cy.matched_receptor.get(g, [])) !=
            set(default.matched_receptor.get(g, []))
            for g in bg_cy.matched_ligand)

    def test_q_is_bitwise_the_v0140_formula(self, pair):
        """Q is now filled block-wise in place; v0.14.0 built it in one
        np.where(...).astype(float32) at ~3.25x P peak memory."""
        A, _, lr_df = pair
        bg = _quiet(la.tl.prepareLRBackground, A, lr_df,
                    **dict(KW, block_size=4))            # several blocks
        cols = A.var_names.get_indexer(bg.gene_index)
        raw = sp.csc_matrix(A.X)[:, cols]
        old_q = np.where(np.asarray((raw != 0).todense()), 0.0,
                         bg.P).astype(np.float32)
        np.testing.assert_array_equal(bg.Q, old_q)


class TestRunLARISParity:

    @staticmethod
    def _run(lr, data, bg, **kw):
        _, res = _quiet(la.tl.runLARIS, lr, data, use_rep="X_spatial",
                        use_rep_spatial="X_spatial", groupby="ct",
                        background=bg, n_cells_expressed_threshold=5, **kw)
        return res.set_index(["sender", "receiver",
                              "interaction_name"]).sort_index()

    @pytest.fixture(scope="class")
    def inputs(self, pair):
        A, path, lr_df = pair
        lr = _quiet(la.tl.prepareLRInteraction, A, lr_df,
                    use_rep_spatial="X_spatial")
        return (A, path, lr,
                _quiet(la.tl.prepareLRBackground, A, lr_df, **KW),
                _quiet(la.tl.prepareLRBackground, path, lr_df, **KW))

    @pytest.mark.parametrize("reference", ["lr", "all"])
    def test_memory_cosg_on_a_cytome_equals_anndata(self, inputs, reference):
        A, path, lr, bg_mem, bg_cy = inputs
        ref = self._run(lr, A, bg_mem, specificity_reference=reference)
        got = self._run(lr, path, bg_cy, cosg_backend="memory",
                        specificity_reference=reference)
        assert ref.index.equals(got.index)
        # the test has teeth: the planted pair sits at the null's floor
        # (k = 6 matched genes -> 36 pseudo-pairs -> p >= 1/37)
        assert ref.loc[("A", "B", "G000::G001"), "p_value"] <= 3.0 / 37
        for col in ("interaction_score", "p_value", "p_value_fdr"):
            np.testing.assert_array_equal(ref[col].to_numpy(),
                                          got[col].to_numpy(), err_msg=col)

    def test_streaming_cosg_agrees_to_float32_precision(self, inputs):
        """cosg.run_cosg_cytome accumulates in float32 (cosg package), so
        the streamed specificity matches in-memory COSG to ~1e-7, not
        bitwise; the background itself is identical."""
        A, path, lr, bg_mem, bg_cy = inputs
        ref = self._run(lr, A, bg_mem)
        got = self._run(lr, path, bg_cy, cosg_backend="stream")
        assert ref.index.equals(got.index)
        np.testing.assert_allclose(got.interaction_score.to_numpy(),
                                   ref.interaction_score.to_numpy(),
                                   rtol=1e-5, atol=1e-9)

    def test_expression_missing_background_genes_raises(self, inputs):
        """No silent zero-fill of the null's specificity any more."""
        A, _, lr, bg_mem, _ = inputs
        lr_genes = pd.unique(np.concatenate(
            [lr.var_names.str.split("::").str[0],
             lr.var_names.str.split("::").str[1]]))
        with pytest.raises(ValueError, match="background gene"):
            self._run(lr, A[:, list(lr_genes)].copy(), bg_mem)


class TestReader:

    def test_gene_subset_read_keeps_stored_dtype_and_values(self, pair):
        A, path, _ = pair
        req = list(A.var_names[[7, 3, 50, 12]])          # not file order
        sub = la.pp.readCytome(path, genes=req)
        assert sub.X.dtype == A.X.dtype
        assert list(sub.var_names) == req
        assert sp.issparse(sub.X)
        assert (abs(sub.X - A[:, req].X) > 0).nnz == 0

    def test_gene_subset_read_restores_obs_dtypes(self, pair):
        """Categorical labels came back as plain strings from subset reads;
        COSG then differed from a whole-file read by up to 1.2e-14."""
        A, path, _ = pair
        sub = la.pp.readCytome(path, genes=list(A.var_names[:3]))
        assert isinstance(sub.obs["ct"].dtype, pd.CategoricalDtype)
        pd.testing.assert_series_equal(sub.obs["ct"], A.obs["ct"],
                                       check_names=False)
