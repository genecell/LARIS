"""v0.15.0 phase 3: the background Gram tables in one pass over the cells.

background_grams computes G_W, G_W2, G_sq and G_dot from P, its detected
part P_det and the row-normalised graph W through the per-cell identity

    M_u = sum_c W[u, c] V(c),  G_W = sum_u V(u) * M_u,  G_W2 = sum_u M_u^2,
    G_sq = sum_u V(u)^2,       G_dot = sum_u V(u),

with V(c)[i, j] = R_i(c) P_j(c) + Q_i(c) R_j(c) (all terms >= 0). These
tests pin it to a float64 reference of the same identity, to the edge
formula LARIS used up to v0.14, and pin the background object that now
stores P_det instead of the dense Q.
"""

import contextlib
import io
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
import anndata as ad
import scipy.sparse as sp

import laris as la
from laris.tools._background import _edge_gram, _numpy_grams

rust = pytest.importorskip("laris._laris")


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*args, **kwargs)


def _case(seed, n, U, k, density=0.12, self_loops=False, empty_rows=0):
    rng = np.random.default_rng(seed)
    P = (rng.random((n, U)) * (rng.random((n, U)) < 0.6)).astype(np.float32)
    det = (rng.random((n, U)) < density) & (P > 0)
    P_det = sp.csr_matrix(np.where(det, P, 0).astype(np.float32))
    rows = np.repeat(np.arange(n), k)
    cols = np.array([rng.choice(np.delete(np.arange(n), i), k, replace=False)
                     for i in range(n)]).ravel()
    w = rng.random(n * k) + 0.1
    if self_loops:
        rows = np.concatenate([rows, np.arange(n)])
        cols = np.concatenate([cols, np.arange(n)])
        w = np.concatenate([w, rng.random(n) + 0.1])
    W = sp.csr_matrix((w, (rows, cols)), shape=(n, n))
    if empty_rows:                       # cells without neighbours
        W = sp.diags((np.arange(n) >= empty_rows).astype(float)) @ W
    s = np.asarray(W.sum(1)).ravel()
    W = (sp.diags(np.where(s > 0, 1 / np.where(s > 0, s, 1), 0)) @ W).tocsr()
    W.eliminate_zeros()
    return P, P_det, W


def _args(P, P_det, W, order=None):
    if order is None:
        order = np.arange(P.shape[0])
    return (P, P_det.indptr.astype(np.int64), P_det.indices.astype(np.int32),
            P_det.data.astype(np.float32), W.indptr.astype(np.int64),
            W.indices.astype(np.int32), W.data.astype(np.float64),
            np.ascontiguousarray(order, np.int64))


def _reference(P, P_det, W):
    """The per-cell identity in plain float64 NumPy."""
    Pd = P.astype(np.float64)
    Rd = P_det.toarray().astype(np.float64)
    Qd = Pd - Rd
    U = P.shape[1]
    out = {k: np.zeros((U, U)) for k in ("G_W", "G_W2", "G_sq", "G_dot")}
    for u in range(P.shape[0]):
        nb, wu = W[u].indices, W[u].data
        M = Rd[nb].T @ (wu[:, None] * Pd[nb]) + Qd[nb].T @ (wu[:, None] * Rd[nb])
        V = np.outer(Rd[u], Pd[u]) + np.outer(Qd[u], Rd[u])
        out["G_W"] += V * M
        out["G_W2"] += M * M
        out["G_sq"] += V * V
        out["G_dot"] += V
    return out


@pytest.mark.parametrize("n, U, k, kw", [
    (120, 9, 6, {}),
    (150, 300, 8, {}),                         # three tiles, ragged last
    (90, 129, 5, dict(self_loops=True)),       # one gene past a tile
    (80, 140, 4, dict(empty_rows=7, density=0.3)),
])
def test_tables_match_float64_reference(n, U, k, kw):
    P, P_det, W = _case(n + U, n, U, k, **kw)
    got = dict(zip(("G_W", "G_W2", "G_sq", "G_dot"),
                   rust.background_grams(*_args(P, P_det, W))))
    ref = _reference(P, P_det, W)
    for name, g in got.items():
        r = ref[name]
        assert np.array_equal(g, g.T), name                  # exactly symmetric
        assert np.array_equal(g == 0, r == 0), name          # same exact zeros
        ok = r > 0
        assert (np.abs(g - r)[ok] / r[ok]).max() < 1e-12, name


def test_cell_order_changes_only_rounding():
    P, P_det, W = _case(3, 200, 150, 7)
    a = rust.background_grams(*_args(P, P_det, W))
    order = np.random.default_rng(0).permutation(200)
    b = rust.background_grams(*_args(P, P_det, W, order))
    for x, y in zip(a, b):
        ok = x > 0
        assert (np.abs(x - y)[ok] / x[ok]).max() < 1e-12


def test_agrees_with_the_v014_edge_formula():
    """The four-term edge expansion (float32 edge products) is accurate to
    ~1e-7 relative away from cancelling entries."""
    P, P_det, W = _case(4, 160, 60, 6)
    G_W, G_W2, G_sq, G_dot = rust.background_grams(*_args(P, P_det, W))
    o_W, o_W2, o_dot, o_sq = _numpy_grams(P, P_det, W)
    for new, old in ((G_W, o_W), (G_W2, o_W2), (G_sq, o_sq), (G_dot, o_dot)):
        big = new > 1e-3 * new.max()
        assert (np.abs(new - old)[big] / new[big]).max() < 1e-5


def test_rejects_bad_inputs():
    P, P_det, W = _case(5, 50, 20, 4)
    args = list(_args(P, P_det, W))
    bad = list(args)
    bad[7] = np.zeros(50, np.int64)                        # not a permutation
    with pytest.raises(ValueError):
        rust.background_grams(*bad)
    bad = list(args)
    idx = args[2].copy()
    s = args[1][np.flatnonzero(np.diff(args[1]) >= 2)[0]]
    idx[s], idx[s + 1] = idx[s + 1], idx[s]                # unsorted genes
    bad[2] = idx
    with pytest.raises(ValueError):
        rust.background_grams(*bad)
    bad = list(args)
    bad[5] = args[5].copy()
    bad[5][0] = 50                                         # graph out of range
    with pytest.raises(ValueError):
        rust.background_grams(*bad)
    with pytest.raises(ValueError):                        # Fortran-order P
        rust.background_grams(np.asfortranarray(P), *args[1:])


@pytest.fixture(scope="module")
def small():
    rng = np.random.default_rng(0)
    n, g = 700, 70
    genes = [f"G{i:03d}" for i in range(g)]
    X = sp.random(n, g, density=0.25, random_state=1,
                  data_rvs=lambda k: rng.poisson(4, k) + 1).tocsr()
    A = ad.AnnData(X=X, var=pd.DataFrame(index=genes))
    A.obsm["X_spatial"] = rng.random((n, 2)) * 100
    A.obs["ct"] = pd.Categorical(rng.choice(["A", "B", "C"], n))
    lr_df = pd.DataFrame({
        "ligand": [f"G{i:03d}" for i in range(0, 20, 2)],
        "receptor": [f"G{i:03d}" for i in range(1, 20, 2)],
    })
    bg = _quiet(la.tl.prepareLRBackground, A, lr_df, n_pool=40,
                n_matched_genes=6, use_rep_spatial="X_spatial", verbosity=0)
    return A, lr_df, bg


def test_background_stores_the_detected_part(small):
    A, _, bg = small
    assert sp.issparse(bg.P_det)
    assert "Q" not in vars(bg)                 # a property, not stored
    cols = A.var_names.get_indexer(bg.gene_index)
    raw = sp.csr_matrix(A.X)[:, cols]
    raw.eliminate_zeros()
    det = bg.P_det.copy()
    assert np.array_equal(det.indptr, raw.indptr)
    assert np.array_equal(det.indices, raw.indices)
    assert np.array_equal(det.toarray(), np.where(raw.toarray() != 0, bg.P, 0))
    old_q = np.where(raw.toarray() != 0, 0.0, bg.P).astype(np.float32)
    assert np.array_equal(bg.Q, old_q)


def test_legacy_pickle_with_dense_q_loads(small):
    _, _, bg = small
    state = dict(bg.__dict__)
    state["Q"] = bg.Q
    del state["P_det"]
    legacy = object.__new__(type(bg))
    legacy.__setstate__(state)
    assert np.array_equal(legacy.P_det.toarray(), bg.P_det.toarray())
    assert np.array_equal(legacy.Q, bg.Q)
    again = pickle.loads(pickle.dumps(bg))
    assert np.array_equal(again.P_det.toarray(), bg.P_det.toarray())


def test_rust_and_numpy_backgrounds_agree(small, monkeypatch):
    A, lr_df, bg = small
    monkeypatch.setenv("LARIS_NO_RUST", "1")
    bg_np = _quiet(la.tl.prepareLRBackground, A, lr_df, n_pool=40,
                   n_matched_genes=6, use_rep_spatial="X_spatial", verbosity=0)
    assert np.array_equal(bg.P, bg_np.P)
    assert np.array_equal(bg.P_det.toarray(), bg_np.P_det.toarray())
    for name in ("G_W", "G_W2", "G_sq", "G_dot"):
        new, old = getattr(bg, name), getattr(bg_np, name)
        big = new > 1e-3 * new.max()
        assert (np.abs(new - old)[big] / new[big]).max() < 1e-5, name
