"""v0.15.0 phase 2: the fused Rust kernels of the exact P-value step.

Each kernel replaces a chain of NumPy statements and must reproduce it
bitwise (same IEEE operations, same order), so P values do not move:

* cos_sq_accumulate  ==  S += c * c; S += c.T * c.T
* coloc_x + log1p    ==  _iqr_apply(lam * c, iqr) of the NumPy pass B
* assembly_sweep     ==  per-row null assembly of v0.15.0 phase 1
* _dense_delta_weight == the per-pair delta blocks
"""

import contextlib
import io
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
import anndata as ad
import scipy.sparse as sp

import laris as la
from laris.tools._background import (
    _dense_delta_weight,
    _iqr_apply,
    _log1p_inplace,
    _pair_gsp_from_tables,
    _pair_random_gsp_from_tables,
)

rust = pytest.importorskip("laris._laris")


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*args, **kwargs)


def _tables(U, seed):
    """Numerator buffers A, B (as the kernels store them), the N they
    encode, and v, S."""
    rng = np.random.default_rng(seed)
    a = (rng.random((U, U)) * 0.5).astype(np.float32)
    b = (rng.random((U, U)) * 0.5).astype(np.float32)   # some N < 0
    a[rng.random((U, U)) < 0.1] = 0.0
    # N[i, j] = A[i, j] + A[j, i] - B[i, j], A[i, j] stored at a[j, i]
    N = (a.T + a) - b.T
    v = rng.random((U, U)) * 3
    v[rng.random((U, U)) < 0.05] = 0.0                   # den = 0 branch
    S = rng.random((U, U)) * 5
    S[rng.random((U, U)) < 0.05] = 0.0
    return a, b, N, v, S


def _numpy_cos(N, v, y_norm):
    den = v * y_norm
    out = np.zeros_like(N, dtype=np.float64)
    np.divide(N, den, out=out, where=den > 0)
    return out


def _numerator_reference(P, Q, y):
    """The v0.14 column-at-a-time numerator in NumPy float32: every entry
    of A and B accumulated over cells in increasing order, w = y_c * Pnz."""
    n, u = P.shape
    Pnz = sp.csr_matrix(P - Q)
    A = np.zeros((u, u), np.float32)
    B = np.zeros((u, u), np.float32)
    for c in range(n):
        if y[c] == 0:
            continue
        det = Pnz.indices[Pnz.indptr[c]:Pnz.indptr[c + 1]]
        val = Pnz.data[Pnz.indptr[c]:Pnz.indptr[c + 1]]
        for j, pj in zip(det, val):
            w = np.float32(y[c]) * pj
            A[:, j] += w * P[c]
            B[det, j] += w * val
    return (A + A.T) - B


@pytest.mark.parametrize("n, u, density, frac_y", [
    (60, 7, 0.4, 1.0),
    (50, 600, 0.08, 0.7),       # several row tiles and gene blocks
    (40, 513, 0.15, 0.1),       # a ragged last tile
])
def test_blocked_numerator_is_bitwise_the_column_kernel(n, u, density, frac_y):
    rng = np.random.default_rng(u)
    P = rng.random((n, u)).astype(np.float32)
    Q = np.where(rng.random((n, u)) < 1 - density, P, 0.0).astype(np.float32)
    y = (rng.random(n) * (rng.random(n) < frac_y)).astype(np.float32)
    Pnz = sp.csr_matrix(P - Q)
    got = rust.ctc_cos_numerator(P, Pnz.indptr.astype(np.int64),
                                 Pnz.indices.astype(np.int32),
                                 Pnz.data.astype(np.float32), y)
    assert np.array_equal(got, _numerator_reference(P, Q, y))


def test_numerator_buffers_are_refilled():
    """Buffers are reused across cell-type pairs: a second call must not
    see the first one's values."""
    rng = np.random.default_rng(9)
    n, u = 30, 40
    P = rng.random((n, u)).astype(np.float32)
    Q = np.where(rng.random((n, u)) < 0.7, P, 0.0).astype(np.float32)
    Pnz = sp.csr_matrix(P - Q)
    args = (P, Pnz.indptr.astype(np.int64), Pnz.indices.astype(np.int32),
            Pnz.data.astype(np.float32))
    a = np.full((u, u), 7.0, np.float32)
    b = np.full((u, u), 7.0, np.float32)
    y = rng.random(n).astype(np.float32)
    rust.ctc_numerator_ab(*args, y, a, b)
    assert np.array_equal((a.T + a) - b.T, rust.ctc_cos_numerator(*args, y))
    with pytest.raises(ValueError):            # unsorted genes in a cell
        bad = Pnz.indices.astype(np.int32).copy()
        s = Pnz.indptr[0]
        bad[s], bad[s + 1] = bad[s + 1], bad[s]
        rust.ctc_numerator_ab(P, args[1], bad, args[3], y, a, b)


@pytest.mark.parametrize("U", [1, 37, 300])
@pytest.mark.parametrize("mirror", [True, False])
def test_cos_sq_accumulate_is_bitwise(U, mirror):
    a, b, N, v, S = _tables(U, 1)
    y_norm = 1.7
    ref = S.copy()
    c = _numpy_cos(N, v, y_norm)
    ref += c * c
    if mirror:
        ref += c.T * c.T
    got = S.copy()
    rust.cos_sq_accumulate(a, b, v, y_norm, got, mirror)
    assert np.array_equal(got, ref)


def test_cos_sq_accumulate_rejects_bad_shapes():
    a, b, N, v, S = _tables(20, 2)
    with pytest.raises(ValueError):
        rust.cos_sq_accumulate(a, b, v[:10], 1.0, S, True)
    with pytest.raises(ValueError):
        rust.cos_sq_accumulate(a, b, v, 1.0, np.asfortranarray(S), True)


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@pytest.mark.parametrize("lam_mu", [0.0, 0.3, 1.0])
def test_coloc_x_then_log1p_is_bitwise(lam_mu):
    U = 211
    a, b, N, v, S = _tables(U, 3)
    y_norm, iqr = 2.3, 0.37
    # the NumPy pass B statements
    cosU = _numpy_cos(N, v, y_norm)
    cos2 = cosU * cosU
    den = (1.0 - lam_mu) * cos2 + lam_mu * S
    lam = np.zeros_like(cosU)
    np.divide(cos2, den, out=lam, where=den != 0)
    ref = _iqr_apply(lam * cosU, iqr)
    x = np.full((U, U), np.nan)
    rust.coloc_x(a, b, v, y_norm, S, lam_mu, iqr, x)
    assert np.array_equal(x, lam * cosU / iqr)
    with ThreadPoolExecutor(4) as pool, np.errstate(invalid="ignore"):
        _log1p_inplace(x, pool, 4)
    # these synthetic "cosines" leave [-1, 1], so log1p meets x < -1 (NaN)
    assert np.array_equal(x, ref, equal_nan=True)


def test_log1p_inplace_matches_one_call():
    x = np.random.default_rng(4).random((1003, 517)) * 5
    ref = np.log1p(x)
    with ThreadPoolExecutor(7) as pool:
        _log1p_inplace(x, pool, 7)
    assert np.array_equal(x, ref)


def _sweep_case(seed, U=60, n_pairs=9, k=7, thr=0.0, with_mirror=True):
    rng = np.random.default_rng(seed)
    x = rng.random((U, U)) * 2
    fs = rng.random((U, U)).astype(np.float32)
    fr = rng.random((U, U)).astype(np.float32)
    fs[rng.random((U, U)) < 0.3] = 0
    scale = 0.83
    dw = rng.random((U, U)) ** 3
    dw[rng.random((U, U)) < 0.2] = 0
    db = rng.random((U, U)) < 0.03
    dwx = dw.copy()
    dwx[db] = np.nan
    sa = rng.random(U)
    sb = rng.random(U)
    sa[rng.random(U) < 0.4] = 0
    sb[rng.random(U) < 0.4] = 0
    lig = [rng.choice(U, k) for _ in range(n_pairs)]   # duplicates allowed
    rec = [np.sort(rng.choice(U, k + 2)) for _ in range(n_pairs)]
    M = (x * (fs.astype(np.float64) * fr)) * scale

    def null(p, s1, s2):
        ii, jj = lig[p], rec[p]
        v = ((s1[ii][:, None] * s2[jj][None, :]) * dw[np.ix_(ii, jj)]) \
            * M[np.ix_(ii, jj)]
        return v[~db[np.ix_(ii, jj)]]

    # observed values: some taken from the null itself, to exercise ties
    obs1 = np.array([null(p, sa, sb)[p % 5] if p % 2 else rng.random() * 0.1
                     for p in range(n_pairs)])
    obs2 = np.array([null(p, sb, sa)[p % 3] if p % 3 else rng.random() * 0.1
                     for p in range(n_pairs)])
    ref = []
    for p in range(n_pairs):
        row = []
        for s1, s2, o in ((sa, sb, obs1[p]), (sb, sa, obs2[p])):
            v = null(p, s1, s2)
            pos = (v >= thr) if thr > 0 else (v > 0)
            row += [int((v >= o).sum()), int(pos.sum())]
        ref.append(row)
    ref = np.array(ref)
    lig_rows = np.concatenate(lig)
    lig_pair = np.repeat(np.arange(n_pairs, dtype=np.int32), k)
    order = np.argsort(lig_rows, kind="stable")
    row_ptr = np.zeros(U + 1, np.int64)
    row_ptr[1:] = np.cumsum(np.bincount(lig_rows, minlength=U))
    col_ptr = np.zeros(n_pairs + 1, np.int64)
    col_ptr[1:] = np.cumsum([len(r) for r in rec])
    args = (x, fs, fr, scale, dwx, sa, sb, row_ptr,
            np.ascontiguousarray(lig_pair[order]), col_ptr,
            np.concatenate(rec).astype(np.int32), obs1,
            obs2 if with_mirror else np.empty(0), thr)
    return args, ref


@pytest.mark.parametrize("thr", [0.0, 1e-3])
def test_assembly_sweep_counts_match_reference(thr):
    args, ref = _sweep_case(5, thr=thr)
    ex1, np1, ex2, np2 = rust.assembly_sweep(*args)
    assert np.array_equal(ex1, ref[:, 0]) and np.array_equal(np1, ref[:, 1])
    assert np.array_equal(ex2, ref[:, 2]) and np.array_equal(np2, ref[:, 3])


def test_assembly_sweep_without_mirror():
    args, ref = _sweep_case(6, with_mirror=False)
    ex1, np1, ex2, np2 = rust.assembly_sweep(*args)
    assert np.array_equal(ex1, ref[:, 0]) and np.array_equal(np1, ref[:, 1])
    assert not ex2.any() and not np2.any()


def test_assembly_sweep_rejects_bad_indices():
    args, _ = _sweep_case(7)
    args = list(args)
    bad = args[10].copy()
    bad[0] = args[0].shape[0]               # receptor gene out of range
    args[10] = bad
    with pytest.raises(ValueError):
        rust.assembly_sweep(*args)


def test_mirror_cell_type_pairs_are_bitwise_symmetric():
    """A::B and B::A must share storage order, so their norms and cosine
    columns agree to the bit and the P-value step can share their tables."""
    from sklearn.metrics.pairwise import cosine_similarity
    from laris.tools._utils import _pairwise_row_multiply
    # neighbourhood composition the way LARIS builds it: a sparse product,
    # whose rows come out with unsorted indices
    rng = np.random.default_rng(0)
    member = sp.csr_matrix((np.ones(400), (rng.integers(0, 5, 400),
                                           np.arange(400))), shape=(5, 400))
    graph = sp.random(400, 400, density=0.05, random_state=1, format="csr")
    M = member @ graph.T
    assert not M.has_sorted_indices
    Y, names = _pairwise_row_multiply(M, list("abcde"))
    pos = {n: i for i, n in enumerate(names)}
    norms = np.sqrt(np.asarray(Y.multiply(Y).sum(axis=1)).ravel())
    X = sp.random(400, 30, density=0.2, random_state=4, format="csr")
    cos = cosine_similarity(X.T, Y, dense_output=False).toarray()
    for a in "abcde":
        for b in "abcde":
            i, j = pos[f"{a}::{b}"], pos[f"{b}::{a}"]
            assert np.array_equal(Y[i].indices, Y[j].indices)
            assert np.array_equal(Y[i].data, Y[j].data)
            assert norms[i] == norms[j]
            assert np.array_equal(cos[:, i], cos[:, j])


@pytest.fixture(scope="module")
def small():
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


@pytest.mark.parametrize("spatial_weight", [0.0, 1.0, 3.0])
def test_dense_delta_weight_equals_pair_blocks(small, spatial_weight):
    _, _, _, bg = small
    mu = 1.0
    dense = _dense_delta_weight(bg, mu, spatial_weight, block=7)
    for lig, rec in list(zip(bg.matched_ligand, bg.matched_receptor))[:6]:
        ii, jj = bg.matched_ligand[lig], bg.matched_receptor[rec]
        d = (_pair_gsp_from_tables(bg, ii, jj)
             - mu * _pair_random_gsp_from_tables(bg, ii, jj))
        blk = (np.ones_like(d) if spatial_weight == 0
               else np.clip(d, 0.0, None) ** spatial_weight)
        assert np.array_equal(dense[np.ix_(ii, jj)], blk)


def test_fused_and_numpy_paths_agree(small, monkeypatch):
    """End to end: the fused path against the NumPy reference path."""
    A, _, lr, bg = small

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
    assert len(m) == len(res_np) == len(res_rs)
    # the numerators differ in float32 summation order only
    assert np.array_equal(m.null_support_r, m.null_support_n)
    assert (m.p_value_r - m.p_value_n).abs().max() <= 1.0 / 37 + 1e-12
    assert ((m.p_value_r - m.p_value_n).abs() < 1e-12).mean() > 0.95
