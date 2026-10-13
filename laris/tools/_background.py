"""Matched-gene background for factorized null p-values, and the analytic
shuffled-graph expectation.

The statistical design and its validation are documented in
``docs/discussion/2026-08-25_analytic_null_proof.md`` (closed form) and
Rounds 34-38 of the discussion record (factorized null). In brief:

- The p-value floor of a resampled background equals one over its support
  size. LARIS's original pair-background has support <= n_neighbors + 1
  (~31), so p-values below ~1/31 were artifacts of the sampling
  convention, and the database itself caps any pair-background at ~2k.
- The factorized null draws a matched gene for the ligand and for the
  receptor independently (k x k support, ~1e4 at k=100), with every
  pseudo-pair scored through the full pipeline composition - its own
  spatial specificity, per-cell-type averages and co-localization - via
  Gram tables over the diffused background genes, so nothing is
  conditioned away and no independence between score components is
  assumed.
- The shuffled-graph side of the spatial specificity has a closed form
  under the L1-normalized random graph (see the proof document), so no
  random graphs are built anywhere in the null.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Dict, Union

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import KDTree, kneighbors_graph

from .._compat import _UNSET, resolve_data_arg


_RUST_THREADS_SET = False


def _rust_kernels():
    """Return LARIS's compiled Rust kernels (``laris._laris``), or None.

    The extension ships inside the wheel; the NumPy path stays as a
    fallback for source installs without a Rust toolchain and for
    debugging. Both compute the identical quantities (parity-tested).
    Set ``LARIS_NO_RUST=1`` to force the NumPy path.
    """
    import os
    global _RUST_THREADS_SET
    if os.environ.get('LARIS_NO_RUST'):
        return None
    try:
        from .. import _laris
    except ImportError:
        return None
    if not _RUST_THREADS_SET:
        # rayon sizes its pool once per process, on first use: give it the
        # count the Python side uses before any kernel runs.
        _RUST_THREADS_SET = True
        init = getattr(_laris, 'init_threads', None)
        if init is not None:
            init(_n_threads())
    return _laris


# ---------------------------------------------------------------------------
# Analytic shuffled-graph expectation (see the proof document, eq. 4-5)
# ---------------------------------------------------------------------------

def _expected_row_sq_sum(weights: np.ndarray, k: int) -> float:
    """E[sum of squared L1-normalized weights of one random-graph row].

    Rows draw ``k`` weights from the observed kernel-weight population and
    L1-normalize; this returns the second-order delta-method expansion of
    ``E[S2 / S1^2]`` in the population moments (proof document, eq. 4).
    """
    w = np.asarray(weights, dtype=np.float64)
    mu1 = w.mean()
    mu2 = (w ** 2).mean()
    mu3 = (w ** 3).mean()
    var = mu2 - mu1 ** 2
    e_s2 = k * mu2
    e_s1sq = k ** 2 * mu1 ** 2 + k * var
    cov_s2_s1sq = 2 * k ** 2 * mu1 * (mu3 - mu2 * mu1)
    var_s1sq = 4 * k ** 3 * mu1 ** 2 * var
    r = (e_s2 / e_s1sq
         - cov_s2_s1sq / e_s1sq ** 2
         + e_s2 * var_s1sq / e_s1sq ** 3)
    # r is a normalized sum of squares of k positive weights: 1/k <= r <= 1.
    return float(np.clip(r, 1.0 / k, 1.0))


def _analytic_random_gsp(m1, m2, n_cells: int, R: float) -> np.ndarray:
    """Closed form for E[cos(v, W_random v)] (proof document, eq. 5).

    ``m1``/``m2`` are the profile's per-cell mean and mean-square; both may
    be arrays (vectorized over profiles). Profiles with ``m2 == 0`` (all
    zero) return 0.
    """
    m1 = np.asarray(m1, dtype=np.float64)
    m2 = np.asarray(m2, dtype=np.float64)
    num = n_cells * m1 ** 2
    den_sq = n_cells * m2 * (R * m2 + (n_cells - R) * m1 ** 2)
    out = np.zeros(np.broadcast(m1, m2).shape, dtype=np.float64)
    ok = den_sq > 0
    np.divide(num, np.sqrt(den_sq, where=ok, out=np.ones_like(den_sq)),
              out=out, where=ok)
    return out


# ---------------------------------------------------------------------------
# Pool selection and matched sets
# ---------------------------------------------------------------------------

def _quantile_grid_pool(means: np.ndarray, variances: np.ndarray,
                        n_pool: int) -> np.ndarray:
    """Deterministic candidate pool covering the (mean, variance) space.

    Genes are rank-transformed on both axes and hashed onto a G x G grid;
    from each occupied cell the gene nearest the cell centre is taken, then
    remaining slots are filled round-robin by within-cell rank. This keeps
    the pool's *coverage* of expression space fixed while bounding its
    size - the Gram tables downstream are quadratic in the pool size, and
    the union of unconstrained per-gene kNN sets spans most of the
    transcriptome (measured: 92% on tonsil at k=100).

    """
    n_genes = means.shape[0]
    if n_pool >= n_genes:
        return np.arange(n_genes)
    rank_m = np.argsort(np.argsort(means)) / max(n_genes - 1, 1)
    rank_v = np.argsort(np.argsort(variances)) / max(n_genes - 1, 1)
    G = int(np.ceil(np.sqrt(n_pool)))
    cell = (np.minimum((rank_m * G).astype(int), G - 1) * G
            + np.minimum((rank_v * G).astype(int), G - 1))
    # distance to the centre of the gene's own grid cell, in rank units
    cx = (cell // G + 0.5) / G
    cy = (cell % G + 0.5) / G
    d2 = (rank_m - cx) ** 2 + (rank_v - cy) ** 2
    order = np.lexsort((d2, cell))          # within each cell: centre-first
    cell_sorted = cell[order]
    # within-cell rank: 0 for the centre-most gene of each cell, 1 next, ...
    first = np.r_[True, cell_sorted[1:] != cell_sorted[:-1]]
    idx_in_cell = np.arange(n_genes) - np.maximum.accumulate(
        np.where(first, np.arange(n_genes), 0))
    # take all rank-0 genes, then rank-1, ... until the pool is full
    pick_order = np.lexsort((cell_sorted, idx_in_cell))
    return np.sort(order[pick_order][:n_pool])


def _augment_pool_for_saturation(
    feats: np.ndarray, pool_pos: np.ndarray, query_pos: np.ndarray,
    k: int, max_rounds: int = 4, tol: float = 0.99,
) -> np.ndarray:
    """Grow the pool until no query gene's matched set is saturated.

    A gene whose matched set lies entirely *below* it in expression has no
    peers in the pool: every pseudo-pair built from it is weaker than the
    real pair, so it beats its own null arithmetically rather than
    biologically. The quantile grid is rank-uniform and so starves the
    extreme top of the abundance distribution, where the most-detected
    genes of every tissue live (measured on tonsil: 485 genes detected in
    over half the cells, 15 of them in the grid pool).

    Rather than padding the pool by a fixed count - which needs a constant
    nobody can choose correctly for an unseen tissue, and which only
    covers the mean axis - this states the property directly and lets the
    pool satisfy it: any gene whose matched set is saturated contributes
    its own `k` nearest neighbours **from the whole transcriptome**, and
    matching is recomputed. Only the genes that actually failed trigger
    growth, and their neighbourhoods overlap heavily, so the pool grows by
    far less than the unconstrained per-gene kNN union (which spans 92% of
    the transcriptome and was rejected for that reason).

    Returns the augmented pool positions, sorted.
    """
    mu = feats.mean(axis=0)
    sd = feats.std(axis=0)
    sd[sd == 0] = 1.0
    zfeat = (feats - mu) / sd
    kdt_all = KDTree(zfeat)
    pool = np.asarray(pool_pos)
    for _ in range(max_rounds):
        # A gene is never its own peer when deciding whether it HAS peers:
        # the saturation test here always excludes self, independently of
        # prepareLRBackground(exclude_self=...), which governs the matched
        # sets actually used for the null. Counting a gene as its own
        # above-neighbour would mask exactly the saturation this repairs.
        qpi = _self_index(pool, query_pos)
        idx = _matched_sets(feats[query_pos], feats[pool], k,
                            query_pool_index=qpi)
        frac_below = (feats[pool][idx, 0] < feats[query_pos, 0][:, None]).mean(1)
        sat = np.flatnonzero(frac_below >= tol)
        if sat.size == 0:
            break
        _, add = kdt_all.query(zfeat[query_pos[sat]],
                               k=min(k, zfeat.shape[0]))
        grown = np.unique(np.concatenate([pool, np.asarray(add).ravel()]))
        if grown.size == pool.size:
            break                      # nothing new to add; genuinely at the top
        pool = grown
    return np.sort(pool)


def _side_repair_pool(means: np.ndarray, detected: np.ndarray,
                      pool_pos: np.ndarray, query_pos: np.ndarray,
                      k: int) -> np.ndarray:
    """Give every LR gene k/2 detected candidates on each side by mean.

    Balanced matching (:func:`_balanced_sets`) takes k/2 peers below an LR
    gene's mean and k/2 above it. The quantile grid can leave fewer than
    that on one side of the least and most abundant LR genes, and the
    saturation repair only looks at the top. So each LR gene, taken from
    the least abundant up, receives the missing genes immediately below it
    by mean; taken from the most abundant down, those immediately above
    it. Neighbouring LR genes share the additions, so a tail of the pool
    grows by about k/2 genes. A gene that the whole transcriptome cannot
    serve on one side is then truncated symmetrically by
    :func:`_balanced_sets`.

    Only genes detected in at least one cell are candidates, and genes
    with no counts leave the pool: a pseudo-pair built from such a gene can
    never score above zero, so a detected gene would beat it for free
    (without this rule the repair adds exactly those genes below the
    scarcest LR genes: 101 on Visium HD). Every other gene keeps its pool.

    Returns the repaired pool positions, sorted.
    """
    means = np.asarray(means)
    detected = np.asarray(detected, dtype=bool)
    pool = np.asarray(pool_pos)
    in_pool = np.zeros(len(means), dtype=bool)
    in_pool[pool[detected[pool]]] = True
    order = np.argsort(means, kind="stable")
    order = order[detected[order]]
    ms = means[order]
    half = k // 2
    q = [int(g) for g in np.asarray(query_pos)]
    for gi in sorted(q, key=lambda g: means[g]):
        below = order[:int(np.searchsorted(ms, means[gi], side="right"))]
        below = below[below != gi]
        need = min(half, below.size) - int(in_pool[below].sum())
        if need > 0:
            in_pool[below[~in_pool[below]][::-1][:need]] = True
    for gi in sorted(q, key=lambda g: -means[g]):
        above = order[int(np.searchsorted(ms, means[gi], side="right")):]
        need = min(k - half, above.size) - int(in_pool[above].sum())
        if need > 0:
            in_pool[above[~in_pool[above]][:need]] = True
    return np.flatnonzero(in_pool)


def _self_index(pool_pos: np.ndarray, query_pos: np.ndarray) -> np.ndarray:
    """For each query gene, its own position inside `pool_pos`, else -1."""
    where = {int(g): i for i, g in enumerate(np.asarray(pool_pos))}
    return np.array([where.get(int(g), -1) for g in np.asarray(query_pos)],
                    dtype=np.int64)


def _matched_sets(query_feats: np.ndarray, pool_feats: np.ndarray,
                  k: int, query_pool_index: Optional[np.ndarray] = None
                  ) -> np.ndarray:
    """k nearest pool genes per query gene in (mean, variance) space.

    Features are standardized on the pool's statistics so neither axis
    dominates. Same KDTree matcher as the rest of the ecosystem, restricted
    to the bounded candidate pool.

    `query_pool_index` gives, for each query gene, its own position inside
    the pool, or -1 when it is not a pool member. When supplied, a gene is
    never its own match. This matters: the candidate pool is drawn by
    quantiles over the whole transcriptome, so LR genes land in it (246 of
    962 on tonsil), and a k-NN query returns the gene itself first. Its
    pseudo-pairs then share the observed pair's exact profile on one side
    and score systematically high, inflating the null.
    """
    mu = pool_feats.mean(axis=0)
    sd = pool_feats.std(axis=0)
    sd[sd == 0] = 1.0
    kdt = KDTree((pool_feats - mu) / sd)
    want = min(k + (1 if query_pool_index is not None else 0),
               pool_feats.shape[0])
    _, idx = kdt.query((query_feats - mu) / sd, k=want)
    idx = np.atleast_2d(idx)
    if query_pool_index is None:
        return idx[:, :k]
    out = np.empty((idx.shape[0], k), dtype=idx.dtype)
    for i in range(idx.shape[0]):
        row = idx[i]
        if query_pool_index[i] >= 0:
            row = row[row != query_pool_index[i]]
        out[i] = row[:k]
    return out


def _edge_limited_mask(means: np.ndarray, pool_pos: np.ndarray,
                       query_pos: np.ndarray, k: int,
                       query_pool_index: Optional[np.ndarray] = None
                       ) -> np.ndarray:
    """Query genes whose matched set is truncated.

    True when fewer than ``k // 2`` candidates (itself excluded) have a mean
    at or below the gene's, or fewer than ``k - k // 2`` above it - the rule
    by which :func:`_balanced_sets` truncates the set symmetrically. After
    the side repair of the pool (:func:`_side_repair_pool`) this happens
    only to genes the whole transcriptome cannot serve on one side: the
    few most and least abundant genes. A property of the gene's position
    in the pool, so it is defined for ``matching='knn'`` too.
    """
    pool_means = np.sort(np.asarray(means)[np.asarray(pool_pos)])
    npool = len(pool_means)
    k = min(k, npool if query_pool_index is None else max(npool - 1, 1))
    half = k // 2
    cut = np.searchsorted(pool_means, np.asarray(means)[np.asarray(query_pos)],
                          side="right")
    below = cut.astype(np.int64)
    if query_pool_index is not None:
        # a query gene in the pool has its own mean: counted at-or-below
        below = below - (np.asarray(query_pool_index) >= 0)
    above = npool - cut
    return (above < k - half) | (below < half)


def _balanced_sets(feats: np.ndarray, pool_pos: np.ndarray,
                   query_pos: np.ndarray, k: int,
                   query_pool_index: Optional[np.ndarray] = None):
    """Matched sets that **straddle** the gene: k/2 below, k/2 above.

    Plain k-NN suffers boundary bias - at the edge of a covariate
    distribution every neighbour lies on one side - so an abundant gene's
    matched set is systematically weaker than the gene and its null is too
    easy. Splitting the nearest neighbours either side of the gene on the
    mean axis makes ``match_frac_below`` = 0.5 by construction, at no cost
    in pool size.

    Distances are still taken in standardized (mean, variance) space: a
    pure rank window on the mean was measured and rejected, since it
    degrades the variance match 5.8-fold for the same mean balance.

    A gene with fewer than k/2 candidates on one side gets a **symmetric
    truncation**: t = min(k/2, candidates below, candidates above) on each
    side, a smaller set that still straddles it (v0.15.0). Up to the
    v0.15.0 tag the short side was backfilled from the other side, which
    biases the null in no fixed direction (measured: 2.9x nominal at 0.05
    on a pure null), and such pairs were then left untested.
    t = 0 gives an empty set, an empty null and p = 1. Every other gene
    gets exactly the set it had before.

    Returns ``(idx, pool_sorted, n_truncated)``: ``idx`` is a list with
    one array of positions into ``pool_sorted`` per query gene (length k,
    or 2t for a truncated gene), and ``n_truncated`` counts the truncated
    genes.
    """
    mu = feats.mean(axis=0)
    sd = feats.std(axis=0)
    sd[sd == 0] = 1.0
    z = (feats - mu) / sd
    pool_pos = np.asarray(pool_pos)
    order = np.argsort(feats[pool_pos, 0], kind="stable")
    pool_sorted = pool_pos[order]
    zp = z[pool_sorted]
    means_sorted = feats[pool_sorted, 0]
    npool = len(pool_sorted)
    k = min(k, npool if query_pool_index is None else max(npool - 1, 1))
    half = k // 2
    out = []
    truncated = 0
    # position of each query gene inside pool_sorted, or -1
    where = {g: i for i, g in enumerate(pool_sorted)}
    for gi in query_pos:
        cut = int(np.searchsorted(means_sorted, feats[gi, 0], side="right"))
        below = np.arange(0, cut)
        above = np.arange(cut, npool)
        self_i = where.get(int(gi), -1)
        if query_pool_index is not None and self_i >= 0:
            below = below[below != self_i]
            above = above[above != self_i]
        need_lo, need_hi = half, k - half
        if above.size < need_hi or below.size < need_lo:
            truncated += 1
            need_lo = need_hi = min(below.size, above.size, half)
        pick = []
        for side, need in ((below, need_lo), (above, need_hi)):
            if need <= 0:
                continue
            if side.size <= need:
                pick.append(side)
            else:
                d = np.linalg.norm(zp[side] - z[gi], axis=1)
                pick.append(side[np.argpartition(d, need)[:need]])
        out.append(np.concatenate(pick).astype(np.int64) if pick
                   else np.empty(0, dtype=np.int64))
    return out, pool_sorted, truncated


def _edge_gram(Pt: np.ndarray, Qt: np.ndarray, graph: sp.spmatrix,
               chunk_bytes: int = 256 << 20,
               assume_symmetric: Optional[bool] = None) -> np.ndarray:
    """Edge Gram of the MASKED pseudo-pair profiles.

    ``prepareLRInteraction`` masks each pair score to cells where the
    ligand or the receptor has nonzero raw counts. With ``Q = P * (raw ==
    0)`` the masked profile is the difference of two separable products,
    ``v_ij = P_i*P_j - Q_i*Q_j`` (on masked-out cells P equals Q, so the
    difference vanishes there and equals P_i*P_j elsewhere). Any quadratic
    form over v therefore expands into four per-gene edge Grams:

        v(u) v(c) =  [P(u)P(c)]_i [P(u)P(c)]_j  -  [P(u)Q(c)]_i [P(u)Q(c)]_j
                   - [Q(u)P(c)]_i [Q(u)P(c)]_j  +  [Q(u)Q(c)]_i [Q(u)Q(c)]_j

    each term separable per gene, hence a ``(U x nnz) @ (nnz x U)``
    product. The row end and the column end are coupled through a per-gene
    product, so this does **not** collapse to ``P W P.T``; the
    ``U^2 x nnz`` cost is inherent (checked, R88.3).

    Layout
    ------
    ``Pt`` and ``Qt`` are **cells x genes**, C-contiguous, so the per-edge
    gather ``Pt[rows]`` is a contiguous row copy rather than a strided
    column gather. That is what this routine is bound by: measured at
    tonsil's shapes (U = 4,824, n = 5,695, 358,877 edges) the transposed
    layout runs in 175 s against 450 s for genes x cells, with checksums
    agreeing to every printed digit.

    Memory
    ------
    The chunk is sized from ``chunk_bytes`` rather than fixed, and the
    four terms are built one at a time. The previous fixed 200,000-edge
    chunk held four ``U x 200_000`` float32 arrays at once - 14.4 GB at
    tonsil's U - for no speed benefit at all (450 s at 200k against 449 s
    at 20k).

    Symmetry
    --------
    When the graph is symmetric - ``W.T @ W`` is, bitwise, on real data -
    each unordered edge appears twice with an identical contribution,
    because ``P_i(u)P_i(c)`` is unchanged by swapping ``u`` and ``c``.
    ``assume_symmetric=True`` halves the edge list. ``W`` itself is
    row-normalised and is **not** symmetric, so it must not take this
    path; ``None`` detects the symmetry instead of assuming it.

    A sparser identity exists and was implemented, measured and rejected.
    Splitting P = Q + R (R nonzero only on raw-detected cells) and lifting
    A=Q(u)Q(c), B=Q(u)R(c), C=R(u)Q(c), D=R(u)R(c) makes every pure-A,
    pure-B and pure-C term cancel, leaving

        G = <A,D>+<D,A> + <B,C>+<C,B> + <B,D>+<D,B> + <C,D>+<D,C> + <D,D>

    in which every term carries an R factor. R is ~7x sparser than P
    (0.048 vs 0.344 on tonsil), giving roughly 400x fewer multiply-adds.
    A threaded Rust kernel over R's nonzeros nevertheless benchmarked
    **2x SLOWER** than the four dense products: the dominant <A,D> term is
    a scatter over all U rows per nonzero, memory-bound and cache-hostile,
    while the dense path runs at near-peak BLAS throughput. Float32 sgemm
    was also tried: 1.44x faster at 1.8e-7 relative error, rejected
    because the tail counts are exact comparisons.
    """
    g = sp.coo_matrix(graph)
    U = Pt.shape[1]
    rows, cols, w = g.row, g.col, g.data.astype(np.float64)

    if assume_symmetric is None:
        gc = g.tocsr()
        assume_symmetric = (gc.shape[0] == gc.shape[1]
                            and (abs(gc - gc.T) > 0).nnz == 0)
    if assume_symmetric:
        keep = rows <= cols
        rows, cols = rows[keep], cols[keep]
        w = w[keep].copy()
        w[rows != cols] *= 2.0

    nnz = len(rows)
    # Per chunk we hold E as float32 (4 B) and E * w as float64 (8 B).
    # The weights stay float64 ON PURPOSE: that upcast is what makes the
    # product a dgemm. Downcasting them turns this into the float32 sgemm
    # that was measured and rejected above, and the chunk-invariance test
    # catches it.
    edge_chunk = max(1024, min(nnz, int(chunk_bytes // max(12 * U, 1))))

    G = np.zeros((U, U), dtype=np.float64)
    for start in range(0, nnz, edge_chunk):
        stop = min(start + edge_chunk, nnz)
        r = rows[start:stop]
        c = cols[start:stop]
        ww = w[start:stop][:, None]                 # float64, deliberately
        for A, B, sign in ((Pt, Pt, 1.0), (Pt, Qt, -1.0),
                           (Qt, Pt, -1.0), (Qt, Qt, 1.0)):
            E = A[r] * B[c]                       # (chunk, U), contiguous
            if sign > 0:
                G += (E * ww).T @ E
            else:
                G -= (E * ww).T @ E
            del E
    return G


# ---------------------------------------------------------------------------
# The background object
# ---------------------------------------------------------------------------

@dataclass
class LRBackground:
    """Product of :func:`prepareLRBackground`; consumed by ``runLARIS``.

    ``gene_index`` lists the U genes covered by the tables (candidate pool
    plus every LR gene, so real pairs resolve against the same tables).
    ``P`` is their diffused matrix (**cells x U**, dense float32,
    C-contiguous) using the same kernel graph as
    ``prepareLRInteraction``. The cells-major layout is deliberate: every
    consumer gathers cells, and that is a contiguous copy here. The Gram tables are
    over the *gsp* graph that ``runLARIS`` builds for spatial specificity.

    ``P_det`` is ``P`` on the entries whose raw count is non-zero (sparse
    CSR, cells x U, genes ascending): the detected part of every profile,
    which is all the masked pseudo-pair profiles need. It replaces the dense
    ``Q = P - P_det`` that versions up to 0.14 stored (a second cells x U
    float32 array: 7.9 GB at 500,000 cells and 4,240 genes); ``Q`` is still
    available as a property, rebuilt exactly on demand.

    Built by :func:`prepareLRBackground`; not meant to be constructed
    directly.

    Attributes
    ----------
    gene_index : pandas.Index
        The U table genes: the candidate pool plus every LR gene.
    P : numpy.ndarray
        Diffused profiles, cells x U, float32, C-contiguous.
    P_det : scipy.sparse.csr_matrix
        ``P`` on the entries whose raw count is non-zero.
    G_W, G_W2 : numpy.ndarray
        Edge Gram tables (U x U, float64) over the spatial-specificity
        graph W and over W'W.
    G_dot, G_sq : numpy.ndarray
        Moment Gram tables (U x U, float64): sums over cells of products of
        profiles and of squared profiles.
    R : float
        Analytic expectation of a random-graph row's squared weights,
        times the number of cells (the shuffled-graph baseline).
    n_cells : int
        Number of cells.
    matched_ligand, matched_receptor : dict of str to numpy.ndarray
        Every ligand's and receptor's matched genes, as positions in
        ``gene_index``: k genes, or fewer for a gene whose set is
        truncated (possibly none).
    db_pairs : frozenset of tuple
        The database's (ligand, receptor) pairs; never used as pseudo-pairs.
    match_frac_below : dict of str to float
        Per LR gene, the fraction of its matched set with a lower mean
        (1.0: every match is weaker, the null is easy; NaN: empty set).
    edge_limited : int
        Number of LR genes whose matched set is truncated (fewer than k/2
        candidates on one side even after the side repair).
    params : dict
        The settings the background was built with.
    edge_genes : frozenset of str or None
        The LR genes whose matched set is truncated (None for backgrounds
        built before v0.15.0).
    """
    gene_index: pd.Index                     # U gene names
    P: np.ndarray                            # (n_cells, U) diffused, float32
    P_det: sp.csr_matrix                     # P on raw-detected entries
    G_W: np.ndarray                          # edge Gram, observed gsp graph
    G_W2: np.ndarray                         # edge Gram, W'W of the gsp graph
    G_dot: np.ndarray                        # P.T @ P      (=> m1 * n)
    G_sq: np.ndarray                         # (P^2).T @ P^2 (=> m2 * n, ||v||^2)
    R: float                                 # analytic E[sum row w'^2] * n
    n_cells: int
    matched_ligand: Dict[str, np.ndarray] = field(default_factory=dict)
    matched_receptor: Dict[str, np.ndarray] = field(default_factory=dict)
    db_pairs: frozenset = frozenset()        # gene-name pairs in the database
    # per LR gene: fraction of its matched set with a lower mean (1.0 =
    # saturated; every pseudo-pair is weaker than the real one)
    match_frac_below: Dict[str, float] = field(default_factory=dict)
    # genes whose matched set is truncated: fewer than k/2 candidates on
    # one side under balanced matching, even after the side repair
    edge_limited: int = 0
    params: dict = field(default_factory=dict)
    # the LR genes with a truncated matched set (see _edge_limited_mask).
    # Their sets still straddle them, so their pairs are tested as
    # usual; the flag is reported. None for backgrounds built earlier.
    edge_genes: Optional[frozenset] = None

    def positions(self, genes) -> np.ndarray:
        pos = self.gene_index.get_indexer(genes)
        if (pos < 0).any():
            missing = [g for g, p in zip(genes, pos) if p < 0]
            raise KeyError(f"genes not in background: {missing[:5]}")
        return pos

    @property
    def support_per_pair(self) -> int:
        """Deprecated: the largest possible support, k**2.

        Since v0.15.0 a truncated matched set gives a row fewer
        pseudo-pairs; each row's own support is ``runLARIS``'s
        ``null_size`` column.
        """
        warnings.warn(
            "LRBackground.support_per_pair is deprecated: since v0.15.0 a "
            "row's support depends on its matched sets. Use the null_size "
            "column of runLARIS's results.", FutureWarning, stacklevel=2)
        k = self.params.get('n_matched_genes', 0)
        return k * k

    @property
    def Q(self) -> np.ndarray:
        """``P`` masked to raw-zero entries (dense; built on each access)."""
        Q = self.P.copy()
        det = self.P_det.tocoo()
        Q[det.row, det.col] = 0.0
        return Q

    def __setstate__(self, state):
        # backgrounds pickled by LARIS <= 0.14 carry the dense Q
        if 'Q' in state and 'P_det' not in state:
            Q = state.pop('Q')
            state['P_det'] = sp.csr_matrix(state['P'] - Q)
        self.__dict__.update(state)



# ---------------------------------------------------------------------------
# One definition of the background, shared by the in-memory and cytome paths
# ---------------------------------------------------------------------------

def _gene_moments(X):
    """Per-gene mean and (population) variance, as v0.14.0 computed them.

    Column-wise on CSC, in the matrix's own dtype, so each gene's value
    depends on its own column only. The cytome path calls this on gene
    blocks of the same stored columns and so gets bit-identical values -
    which is what makes both paths rank near-tied genes the same way and
    therefore draw the same pool and matched sets.
    """
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocsc()
    m = np.asarray(X.mean(axis=0)).ravel()
    m2 = np.asarray(X.power(2).mean(axis=0)).ravel()
    return m, m2 - m ** 2


def _gene_detected(X) -> np.ndarray:
    """Per gene: True if it is non-zero in at least one cell.

    Counts stored values that are non-zero (an explicitly stored 0 is not
    a detection), column by column, so gene blocks of a cytome give the
    same answer as the whole matrix.
    """
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocsc()
    nz = np.concatenate([[0], np.cumsum(X.data != 0)])
    return (nz[X.indptr[1:]] - nz[X.indptr[:-1]]) > 0


@dataclass
class _MatchingSelection:
    """Table genes and matched sets chosen by :func:`_select_matching`."""
    gene_pos: np.ndarray                     # table genes, sorted positions in var_names
    gene_index: pd.Index                     # their names, in var_names order
    matched_ligand: Dict[str, np.ndarray]    # positions into gene_index
    matched_receptor: Dict[str, np.ndarray]
    match_frac_below: Dict[str, float]
    edge_limited: int
    edge_genes: frozenset = frozenset()


def _select_matching(var_names, m, v, lr_df, n_pool, augment_pool,
                     n_matched_genes, matching, exclude_self, say,
                     detected=None) -> _MatchingSelection:
    """Candidate pool, table genes and matched sets from WHOLE-matrix gene
    statistics ``m`` / ``v`` (one value per gene of ``var_names``).

    This is the only place matching is defined. Both input paths call it on
    statistics over every gene, so they cannot disagree. Up to v0.14.0 the
    cytome path re-ran matching on its loaded pool + LR subset, where
    ``_balanced_sets`` standardised the (mean, variance) features over the
    subset instead of the whole matrix: ~70% of matched sets differed by
    1-5 of 100 genes and FDR<0.05 calls moved (817 vs 780 on 10,000 Xenium
    cells).

    ``detected`` (one bool per gene: non-zero in at least one cell) limits
    the side repair to detected genes; None takes it from ``m > 0``, which
    is the same thing for non-negative expression.
    """
    var_names = pd.Index(var_names)
    if not var_names.is_unique:
        dup = list(var_names[var_names.duplicated()].unique()[:5])
        raise ValueError(
            f"Gene names must be unique to define matched sets; duplicated: "
            f"{dup}. Make them unique (e.g. var_names_make_unique()).")
    lr_genes = pd.Index(pd.unique(np.concatenate([
        np.asarray(lr_df['ligand']), np.asarray(lr_df['receptor'])])))
    lr_genes = lr_genes[lr_genes.isin(var_names)]
    lig_genes = pd.Index(pd.unique(np.asarray(lr_df['ligand'])))
    lig_genes = lig_genes[lig_genes.isin(var_names)]
    rec_genes = pd.Index(pd.unique(np.asarray(lr_df['receptor'])))
    rec_genes = rec_genes[rec_genes.isin(var_names)]

    pool_pos = _quantile_grid_pool(m, v, n_pool)
    if augment_pool:
        _feats_all = np.column_stack([m, v])
        _q = var_names.get_indexer(lr_genes)
        _q = _q[_q >= 0]
        _before = len(pool_pos)
        pool_pos = _augment_pool_for_saturation(
            _feats_all, pool_pos, _q, n_matched_genes)
        if len(pool_pos) > _before:
            say(f"  - pool grown {_before:,} -> {len(pool_pos):,} genes "
                "so every LR gene has matched peers above it")
        if detected is None:
            detected = np.asarray(m) > 0
        _before = len(pool_pos)
        pool_pos = _side_repair_pool(m, detected, pool_pos, _q,
                                     n_matched_genes)
        if len(pool_pos) != _before:
            say(f"  - side repair: pool {_before:,} -> {len(pool_pos):,} "
                f"genes (k/2 detected peers on each side of every LR gene "
                f"where the transcriptome has them)")
    say(f"  - candidate pool: {len(pool_pos):,} genes "
        f"(quantile grid over mean/variance)")

    # tables cover pool genes + every LR gene (so real pairs resolve too)
    lr_pos = var_names.get_indexer(lr_genes)
    gene_pos = np.unique(np.concatenate([pool_pos, lr_pos]))
    gene_index = var_names[gene_pos]
    pos_in_tables = pd.Index(gene_index).get_indexer

    # matched sets: k nearest *pool* genes per LR gene (kNN-within-pool)
    feats = np.column_stack([m, v])
    # positions of the pool genes inside the table gene set
    def _qpi(pos_):
        return _self_index(pool_pos, pos_) if exclude_self else None

    if matching not in ('knn', 'balanced'):
        raise ValueError(f"matching must be 'knn' or 'balanced', got {matching!r}")
    lig_pos = var_names.get_indexer(lig_genes)
    rec_pos = var_names.get_indexer(rec_genes)
    edge_limited = 0
    if matching == 'balanced':
        lig_idx, lig_pool, e1 = _balanced_sets(
            feats, pool_pos, lig_pos, n_matched_genes,
            query_pool_index=_qpi(lig_pos))
        rec_idx, rec_pool, e2 = _balanced_sets(
            feats, pool_pos, rec_pos, n_matched_genes,
            query_pool_index=_qpi(rec_pos))
        edge_limited = e1 + e2
    else:
        lig_idx = _matched_sets(feats[lig_pos], feats[pool_pos],
                                n_matched_genes,
                                query_pool_index=_qpi(lig_pos))
        rec_idx = _matched_sets(feats[rec_pos], feats[pool_pos],
                                n_matched_genes,
                                query_pool_index=_qpi(rec_pos))
        lig_pool = rec_pool = np.asarray(pool_pos)
    lig_tab = pos_in_tables(var_names[lig_pool])
    rec_tab = pos_in_tables(var_names[rec_pool])
    matched_ligand = {g: lig_tab[lig_idx[i]]
                      for i, g in enumerate(lig_genes)}
    matched_receptor = {g: rec_tab[rec_idx[i]]
                        for i, g in enumerate(rec_genes)}
    # Matchability: fraction of a gene's matched set with a LOWER mean
    # than the gene itself. 1.0 means every match is weaker - the null
    # for any pair containing this gene is systematically easy, and its
    # p-values overstate. With the n_extreme pool repair this should be
    # rare; the value is reported so the residual cases are visible.
    # NaN for an empty (fully truncated) set, so the values stay floats.
    match_frac_below = {}
    for genes_, idx_, pl in ((lig_genes, lig_idx, lig_pool),
                             (rec_genes, rec_idx, rec_pool)):
        pool_means = m[np.asarray(pl)]
        gm = m[var_names.get_indexer(genes_)]
        for i, g in enumerate(genes_):
            sel_ = np.asarray(idx_[i])
            match_frac_below[g] = (float((pool_means[sel_] < gm[i]).mean())
                                   if sel_.size else float('nan'))

    # LR genes whose matched set is truncated (either role: same pool,
    # same mean, so the flag is a property of the gene)
    edge_genes = set()
    for genes_, pos_ in ((lig_genes, lig_pos), (rec_genes, rec_pos)):
        mask = _edge_limited_mask(m, pool_pos, pos_, n_matched_genes,
                                  query_pool_index=_qpi(pos_))
        edge_genes.update(np.asarray(genes_)[mask].tolist())

    return _MatchingSelection(
        gene_pos=gene_pos, gene_index=pd.Index(gene_index),
        matched_ligand=matched_ligand, matched_receptor=matched_receptor,
        match_frac_below=match_frac_below, edge_limited=edge_limited,
        edge_genes=frozenset(edge_genes))


def _build_tables(adata, Xg, n_cells, number_nearest_neighbors,
                  use_rep_spatial, use_rep_gsp, n_nearest_neighbors_gsp,
                  sigma_gsp, section_key, block_size, say, sigma='adaptive'):
    """Diffused table-gene profiles ``P``, their detected part ``P_det``
    and the Gram tables.

    ``Xg`` is cells x table genes (raw values; columns in table order);
    ``adata`` supplies ``obs`` / ``obsm`` for the spatial graphs. ``sigma``
    is the diffusion kernel's bandwidth, as in ``prepareLRInteraction``.
    """
    from . import _utils

    Xg = sp.csr_matrix(Xg)
    U = Xg.shape[1]
    say(f"  - diffusing {U:,} genes through the "
        f"{number_nearest_neighbors}-NN kernel graph...")
    coords = np.asarray(adata.obsm[use_rep_spatial])
    sections = _utils._resolve_sections(adata, section_key, n_cells)
    # identical graph construction to prepareLRInteraction, so real pairs'
    # profiles reconstructed from P/Q match the pipeline's columns
    knn = _utils._sectioned_kneighbors_graph(
        coords, n_neighbors=number_nearest_neighbors, sections=sections,
        mode='distance', include_self=True)
    knn = sp.csr_matrix(_utils._apply_knn_kernel(knn, sigma=sigma))
    # cells x genes, C-contiguous: every downstream consumer gathers
    # CELLS, and in this layout that is a contiguous row copy rather than
    # a strided column gather. Measured 2.6x on the edge Grams (R89.8).
    #
    # prepareLRInteraction keeps a pair's score only in cells where ligand
    # OR receptor raw counts are nonzero. With P_det = P on the raw-nonzero
    # entries and Q = P - P_det, the masked pseudo-pair profile is
    # P_i*P_j - Q_i*Q_j (see _edge_gram), which keeps every Gram identity.
    #
    # P_det is sparse (the raw matrix's pattern, values from P). v0.14
    # stored Q densely instead - a second cells x genes float32 array.
    P = np.empty((n_cells, U), dtype=np.float32)
    step = block_size * 32
    for start in range(0, n_cells, step):
        stop = min(start + step, n_cells)
        P[start:stop, :] = np.asarray(
            (knn[start:stop, :] @ Xg).todense())
    del knn
    det = Xg.copy()
    det.eliminate_zeros()
    det.sort_indices()
    rows = np.repeat(np.arange(n_cells), np.diff(det.indptr))
    P_det = sp.csr_matrix((P[rows, det.indices], det.indices.copy(),
                           det.indptr.copy()), shape=(n_cells, U))
    del det, rows

    # ---- the gsp graph and its Gram tables ------------------------------
    say("  - building spatial-specificity graph and Gram tables...")
    # identical to the graph runLARIS builds for spatial specificity
    W = sp.csr_matrix(_utils._build_adjacency_matrix(
        adata, use_rep=use_rep_gsp,
        n_nearest_neighbors=n_nearest_neighbors_gsp,
        sigma=sigma_gsp, sections=sections))

    _rust = _rust_kernels()
    if _rust is not None and hasattr(_rust, 'background_grams'):
        # One pass over the cells gives all four tables, from P_det's
        # non-zeros, in float64 with no cancelling terms (src/lib.rs).
        # Cells are visited in reverse Cuthill-McKee order of the graph,
        # which keeps their neighbours' profiles in cache; the order moves
        # the sums by float64 rounding only.
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        order = reverse_cuthill_mckee(sp.csr_matrix(W + W.T),
                                      symmetric_mode=True)
        G_W, G_W2, G_sq, G_dot = _rust.background_grams(
            P, P_det.indptr.astype(np.int64),
            P_det.indices.astype(np.int32), P_det.data.astype(np.float32),
            W.indptr.astype(np.int64), W.indices.astype(np.int32),
            W.data.astype(np.float64), np.ascontiguousarray(order, np.int64))
    else:
        G_W, G_W2, G_dot, G_sq = _numpy_grams(P, P_det, W)

    R = n_cells * _expected_row_sq_sum(W.data, n_nearest_neighbors_gsp)

    return P, P_det, G_W, G_W2, G_dot, G_sq, R


def _numpy_grams(P, P_det, W):
    """The four Gram tables without the compiled kernel (reference path):
    the four-term edge expansion of ``_edge_gram`` and the moment Grams."""
    n_cells, U = P.shape
    Q = P.copy()
    det = P_det.tocoo()
    Q[det.row, det.col] = 0.0
    del det
    # W is row-normalised and therefore NOT symmetric; W'W is (bitwise,
    # checked on real data). Only the latter may halve its edge list.
    G_W = _edge_gram(P, Q, W, assume_symmetric=False)
    G_W2 = _edge_gram(P, Q, (W.T @ W).tocoo(), assume_symmetric=True)

    # Accumulate over cell blocks instead of materialising four full
    # float64 copies of a cells x genes array at once (149 GB at one
    # million cells and U = 5,000; R88.2).
    G_dot = np.zeros((U, U), dtype=np.float64)
    G_sq = np.zeros((U, U), dtype=np.float64)
    cell_block = max(1, min(n_cells, int((256 << 20) // max(8 * U, 1))))
    for start in range(0, n_cells, cell_block):
        stop = min(start + cell_block, n_cells)
        Pb = P[start:stop].astype(np.float64)
        Qb = Q[start:stop].astype(np.float64)
        G_dot += Pb.T @ Pb
        G_dot -= Qb.T @ Qb
        Pb *= Pb                      # in place: Pb is now P^2
        Qb *= Qb
        G_sq += Pb.T @ Pb
        G_sq -= Qb.T @ Qb
        del Pb, Qb
    return G_W, G_W2, G_dot, G_sq


def _assemble_background(sel, tables, n_cells, lr_df, n_matched_genes,
                         params, say) -> "LRBackground":
    P, P_det, G_W, G_W2, G_dot, G_sq, R = tables
    db_pairs = frozenset(zip(np.asarray(lr_df['ligand']),
                             np.asarray(lr_df['receptor'])))
    # matched sets of edge genes are truncated symmetrically (v0.15.0);
    # runLARIS warns about backgrounds built with the earlier backfill
    params = dict(params, truncation='symmetric')
    sizes = [len(s) for d in (sel.matched_ligand, sel.matched_receptor)
             for s in d.values()]
    n_short = sum(s < n_matched_genes for s in sizes)
    n_empty = sum(s == 0 for s in sizes)
    say(f"  - done: {len(sel.gene_index):,} genes, support per pair up to "
        f"{n_matched_genes ** 2:,} (floor "
        f"{1.0 / (n_matched_genes ** 2 + 1):.2e}); {len(sel.edge_genes)} "
        f"LR gene(s) with a truncated matched set"
        + (f" ({n_empty} matched set(s) empty)" if n_empty else ""))
    del n_short
    return LRBackground(
        gene_index=pd.Index(sel.gene_index), P=P, P_det=P_det,
        G_W=G_W, G_W2=G_W2, G_dot=G_dot, G_sq=G_sq,
        R=R, n_cells=n_cells,
        matched_ligand=sel.matched_ligand,
        matched_receptor=sel.matched_receptor,
        db_pairs=db_pairs, match_frac_below=sel.match_frac_below,
        edge_limited=sel.edge_limited, params=params,
        edge_genes=sel.edge_genes)



def _resolve_n_matched_genes(n_matched_genes, lr_df, var_names,
                             n_pool: Optional[int] = None,
                             fdr: float = 0.05, k_min: int = 100,
                             k_max: int = 200) -> int:
    """Matched genes per side; 'auto' sizes k to the LR database.

    A sender-receiver pair has at most one row per LR pair, so with
    k**2 >= n_pairs / fdr an interaction at the floor 1/(k**2+1) can reach
    FDR ``fdr`` on its own. Clipped to [k_min, k_max]: below 100 the null
    is coarse anyway, and above 200 the matched sets widen in expression
    for little gain. 'auto' never asks for more genes than the candidate
    pool can supply (``min(n_pool, number of genes) - 1``).
    """
    if isinstance(n_matched_genes, str):
        if n_matched_genes != 'auto':
            raise ValueError(
                f"n_matched_genes must be an int or 'auto', got "
                f"{n_matched_genes!r}")
        present = pd.Index(var_names)
        ok = (lr_df['ligand'].isin(present)
              & lr_df['receptor'].isin(present))
        n_pairs = int(lr_df.loc[ok, ['ligand', 'receptor']]
                      .drop_duplicates().shape[0])
        k = int(min(max(int(np.ceil(np.sqrt(max(n_pairs, 1) / fdr))),
                        k_min), k_max))
        n_cand = len(present) if n_pool is None else min(n_pool, len(present))
        return int(max(1, min(k, n_cand - 1)))
    k = int(n_matched_genes)
    if k < 1:
        raise ValueError(f"n_matched_genes must be >= 1, got {k}")
    return k

def prepareLRBackground(
    data=_UNSET,
    lr_df: Optional[pd.DataFrame] = None,
    n_pool: int = 4000,
    augment_pool: bool = True,
    n_matched_genes='auto',
    matching: str = 'balanced',
    exclude_self: bool = True,
    number_nearest_neighbors: int = 20,
    use_rep_spatial: str = 'X_spatial',
    sigma: Union[float, str] = 'adaptive',
    use_rep_gsp: Optional[str] = None,
    n_nearest_neighbors_gsp: int = 20,
    sigma_gsp='adaptive',
    layer: Optional[str] = None,
    section_key: Optional[str] = None,
    block_size: int = 512,
    verbosity: int = 1,
    adata=_UNSET,
) -> LRBackground:
    """Build the matched-gene background for factorized null p-values.

    One background serves every ``runLARIS`` call on the same cells: the
    Gram tables depend only on the spatial graphs and the gene pool, not on
    the cell-type grouping or on ``mu`` / ``spatial_weight``, so parameter
    sweeps and different ``groupby`` choices reuse it unchanged.

    Parameters
    ----------
    data : AnnData, str or cytome dataset
        The same expression object passed to ``prepareLRInteraction``
        (``layer`` selects a layer): an AnnData, or a cytome path or open
        dataset.
    lr_df : DataFrame
        The LR database subset used for the analysis ('ligand'/'receptor'
        columns). Needed to know which genes require matched sets and which
        gene combinations are real database pairs (excluded from nulls).
    n_pool : int
        Candidate pool size U. The Gram tables are U x U; the default keeps
        them ~130 MB each in float64.
    augment_pool : bool, default=True
        Repair the candidate pool so that every ligand and receptor has
        peers on both sides by expression. Two rules:

        - *saturation*: genes whose matched set would lie entirely below
          them contribute their own nearest neighbours from the whole
          transcriptome (the quantile grid is rank-uniform and starves the
          extreme top of the abundance range; such a gene would beat its
          own null arithmetically);
        - *side repair* (v0.15.0): every LR gene gets k/2 candidates below
          and k/2 above it by mean, taken from the genes nearest to it, and
          only genes detected in at least one cell are candidates. A gene
          the whole transcriptome cannot serve on one side gets a smaller
          set that still straddles it (see ``matching``).

        Set False to reproduce the v0.12.0 pool exactly.
    n_matched_genes : int or 'auto', default='auto'
        Matched genes per side (k). The factorized support per pair is at
        most k**2 and the exact p-value floor is 1/(k**2+1) before
        exclusions; a row's own support is reported in its ``null_size``
        column.
        Within a sender-receiver pair BH can call an isolated interaction
        at the floor only if 1/(k**2+1) <= 0.05/m with m tested LR
        pairs, i.e. k**2 >= 20 m. 'auto' sizes k to the database: the
        smallest k with k**2 >= 20 x (number of LR pairs present in the
        data), clipped to [100, 200] (v0.15.0; earlier versions used 100,
        which cannot call an isolated hit among more than 500 tested
        pairs). Only the null assembly grows with k (as k**2); the Gram
        tables do not depend on it.
    matching : {'balanced', 'knn'}, default='balanced'
        How the k matched genes are chosen around each LR gene in (mean,
        variance) space. 'balanced' takes k/2 nearest pool genes below the
        gene's mean and k/2 above, so the matched set straddles it. A gene
        with fewer than k/2 candidates on one side, even after the side
        repair, gets t = min(k/2, below, above) on each side: a smaller
        set that still straddles it, reported as ``edge_limited`` (up to
        the v0.15.0 tag the short side was backfilled from the other side,
        and such pairs were left untested). 'knn' takes the k
        nearest regardless of side (the pre-v0.13 behaviour); it cannot
        straddle a gene at the edge of the pool and is **deprecated**.
    exclude_self : bool, default=True
        Never put an LR gene into its own matched set.
    number_nearest_neighbors, use_rep_spatial, sigma : int, str, float or 'adaptive'
        The diffusion graph: must match the ``prepareLRInteraction`` call
        (defaults 20, 'X_spatial', 'adaptive'). ``sigma`` is the diffusion
        kernel's bandwidth, 'adaptive' or a number in coordinate units; up
        to v0.15.0 the background always used 'adaptive'. ``runLARIS``
        compares these with the settings ``prepareLRInteraction`` records
        on its output and refuses a mismatch before any work.
    use_rep_gsp, n_nearest_neighbors_gsp, sigma_gsp : str, int, float or 'adaptive'
        The spatial-specificity graph: must match ``runLARIS``'s
        ``use_rep`` / ``n_nearest_neighbors`` / ``sigma`` (defaults: as
        ``use_rep_spatial``, 20, 'adaptive'). ``runLARIS`` checks them
        before any work.
    layer : str, optional
        Layer holding the expression values (default: ``.X``); must match
        what ``prepareLRInteraction`` read. For a cytome source it names
        the cytome layer; None reads whichever matrix the file records as
        ``adata.X``.
    section_key : str, optional
        Column of ``.obs`` naming each cell's tissue section, as passed to
        ``prepareLRInteraction``: the neighbour graphs are then built within
        sections.
    block_size : int, default=512
        Cells per block when diffusing the table genes (bounds peak
        memory; results do not depend on it).
    verbosity : int, default=1
        1 prints progress; 0 is silent.
    adata : deprecated
        Former name of ``data``.

    Returns
    -------
    LRBackground
        The candidate pool, every LR gene's matched sets, and the diffused
        profiles and Gram tables that score any pseudo-pair; pass it to
        ``runLARIS(..., background=)``. ``params`` records the settings it
        was built with.

    Notes
    -----
    An AnnData and a cytome of the same data give the **same background,
    bit for bit**: the pool and matched sets are chosen once, by one
    function, from per-gene statistics over every gene, and the tables are
    built by one builder. The cytome path only reads the data in bounded
    gene blocks (v0.14.1; up to v0.14.0 the two could differ, see the
    changelog).
    """
    from . import _utils

    adata = resolve_data_arg(data, 'prepareLRBackground', adata=adata)
    if lr_df is None:
        raise TypeError("prepareLRBackground() requires lr_df")
    if use_rep_gsp is None:
        use_rep_gsp = use_rep_spatial
    if matching == 'knn':
        warnings.warn(
            "matching='knn' is deprecated and will be removed: plain "
            "nearest neighbours cannot straddle a gene at the edge of the "
            "candidate pool, so the null of every pair with such a gene is "
            "biased. Use the default matching='balanced'.",
            FutureWarning, stacklevel=2)

    say = print if verbosity else (lambda *a, **k: None)

    if _is_cytome_source(adata):
        # Pool selection needs per-gene mean/variance over ALL genes, which
        # is computed in gene blocks (bounded memory); only the selected
        # pool + LR genes are then loaded, so the memory profile matches
        # the streaming pipeline.
        return _prepare_background_from_cytome(
            adata, lr_df, n_pool=n_pool, augment_pool=augment_pool,
            n_matched_genes=n_matched_genes, matching=matching,
            exclude_self=exclude_self,
            number_nearest_neighbors=number_nearest_neighbors,
            use_rep_spatial=use_rep_spatial, use_rep_gsp=use_rep_gsp,
            n_nearest_neighbors_gsp=n_nearest_neighbors_gsp,
            sigma_gsp=sigma_gsp, layer=layer, section_key=section_key,
            block_size=block_size, verbosity=verbosity, sigma=sigma)

    X = adata.layers[layer] if layer is not None else adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocsc()
    n_cells = adata.n_obs

    n_matched_genes = _resolve_n_matched_genes(n_matched_genes, lr_df,
                                               adata.var_names, n_pool)

    # ---- gene statistics, pool and matched sets (shared definition) -----
    m, v = _gene_moments(X)
    sel = _select_matching(adata.var_names, m, v, lr_df, n_pool,
                           augment_pool, n_matched_genes, matching,
                           exclude_self, say, detected=_gene_detected(X))

    # ---- diffusion and Gram tables (shared builder) ---------------------
    tables = _build_tables(
        adata, sp.csr_matrix(X[:, sel.gene_pos]), n_cells,
        number_nearest_neighbors, use_rep_spatial, use_rep_gsp,
        n_nearest_neighbors_gsp, sigma_gsp, section_key, block_size, say,
        sigma=sigma)
    return _assemble_background(
        sel, tables, n_cells, lr_df, n_matched_genes,
        dict(n_pool=n_pool, n_matched_genes=n_matched_genes,
             matching=matching, exclude_self=exclude_self,
             number_nearest_neighbors=number_nearest_neighbors,
             use_rep_spatial=use_rep_spatial, sigma=sigma,
             use_rep_gsp=use_rep_gsp,
             n_nearest_neighbors_gsp=n_nearest_neighbors_gsp,
             sigma_gsp=sigma_gsp, section_key=section_key, layer=layer),
        say)


def _pair_gsp_from_tables(bg: LRBackground, ii: np.ndarray,
                          jj: np.ndarray) -> np.ndarray:
    """cos(v, Wv) for pseudo-pairs v = P[i] * P[j], from the Gram tables."""
    num = bg.G_W[np.ix_(ii, jj)]
    den = np.sqrt(bg.G_sq[np.ix_(ii, jj)] * bg.G_W2[np.ix_(ii, jj)])
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=den > 0)
    return out


def _pair_random_gsp_from_tables(bg: LRBackground, ii: np.ndarray,
                                 jj: np.ndarray) -> np.ndarray:
    """Analytic E[cos(v, W_rand v)] for pseudo-pairs, from the Gram tables."""
    n = bg.n_cells
    m1 = bg.G_dot[np.ix_(ii, jj)] / n
    m2 = bg.G_sq[np.ix_(ii, jj)] / n
    return _analytic_random_gsp(m1, m2, n, bg.R)


# ---------------------------------------------------------------------------
# iqrLogNormalize replicated with separable fit/apply (fit on real pairs,
# apply to pseudo-pairs) - mirrors cosg.iqrLogNormalize exactly.
# ---------------------------------------------------------------------------

def _iqr_fit(raw: pd.DataFrame, fit_rows=None,
             q_upper: float = 0.95, q_lower: float = 0.75) -> pd.Series:
    pop = raw if fit_rows is None else raw.loc[fit_rows]
    iqr = pop.quantile(q_upper) - pop.quantile(q_lower)
    nonzero = iqr[iqr > 0]
    fallback = nonzero.min() if len(nonzero) else 1e-6
    return iqr.mask(iqr == 0, fallback)


def _iqr_apply(values, iqr):
    return np.log1p(values / np.asarray(iqr))


# ---------------------------------------------------------------------------
# Helpers of the fused (Rust) P-value path
# ---------------------------------------------------------------------------

def _usable_cpus() -> int:
    """CPUs this process may run on.

    ``os.cpu_count()`` counts every CPU of the machine. On a cluster node
    where the scheduler grants 8 of 128 cores (cpuset / CPU affinity), that
    would start 128 workers on 8 cores. The affinity mask is the right
    count: ``os.process_cpu_count()`` (Python >= 3.13), else
    ``os.sched_getaffinity`` (Linux), else ``os.cpu_count()`` (macOS and
    Windows, which have no affinity mask).
    """
    import os
    count = getattr(os, 'process_cpu_count', None)
    if count is not None:
        n = count()
        if n:
            return int(n)
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _n_threads() -> int:
    """Threads for the fused path and the background Grams:
    RAYON_NUM_THREADS, else OMP_NUM_THREADS, else the CPUs this process may
    use (:func:`_usable_cpus`). The Rust pool is sized to the same number
    (:func:`_rust_kernels`)."""
    import os
    for var in ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS'):
        value = os.environ.get(var, '').strip()
        if value.isdigit() and int(value) > 0:
            return int(value)
    return _usable_cpus()


def _log1p_inplace(x: np.ndarray, pool, n_blocks: int) -> np.ndarray:
    """``np.log1p(x, out=x)`` on row blocks in parallel threads.

    NumPy releases the GIL inside the ufunc, and the result is element-wise,
    so the blocks give exactly the values of one call (checked bitwise for
    the SIMD log1p this NumPy uses).
    """
    n_rows = x.shape[0]
    if pool is None or n_blocks <= 1 or n_rows < 2 * n_blocks:
        return np.log1p(x, out=x)
    bounds = np.linspace(0, n_rows, n_blocks + 1).astype(int)

    def _block(ab):
        a, b = ab
        np.log1p(x[a:b], out=x[a:b])

    list(pool.map(_block, zip(bounds[:-1], bounds[1:])))
    return x


def _dense_delta_weight(bg: 'LRBackground', mu_gsp: float,
                        spatial_weight: float, block: int = 512) -> np.ndarray:
    """``max(delta, 0) ** spatial_weight`` for every pair of table genes.

    The same element-wise arithmetic as the per-pair blocks
    (``_pair_gsp_from_tables`` - ``mu_gsp`` * ``_pair_random_gsp_from_tables``),
    evaluated on row blocks of the full tables, so every entry is bitwise
    the value the per-pair path computes. U x U float64.
    """
    U = len(bg.gene_index)
    n = bg.n_cells
    out = np.empty((U, U), dtype=np.float64)
    for a in range(0, U, block):
        b = min(a + block, U)
        num = bg.G_W[a:b]
        den = np.sqrt(bg.G_sq[a:b] * bg.G_W2[a:b])
        gsp = np.zeros_like(num)
        np.divide(num, den, out=gsp, where=den > 0)
        d = gsp - mu_gsp * _analytic_random_gsp(
            bg.G_dot[a:b] / n, bg.G_sq[a:b] / n, n, bg.R)
        if spatial_weight == 0:
            out[a:b] = 1.0
        else:
            out[a:b] = np.clip(d, 0.0, None) ** spatial_weight
    return out


# Relative difference above which an observed score and the same pair scored
# through the background tables are taken to come from different data or
# settings. The two paths sum in different orders and precisions: the
# pipeline accumulates the co-localization numerator in float64, the table
# kernels in float32 over every cell of a group. On a 14,149-cell cluster
# (Atera, 10x k-means) that alone differed by 1.04e-4, and v0.15.0's 1e-4
# raised. A different graph setting or expression object moves scores by
# far more than 1%, and runLARIS checks every graph setting before this
# point, so the guard is a backstop for a different expression object.
_TABLE_REL_TOL = 1e-2


def _table_score_mismatch(table: np.ndarray, pipe: np.ndarray,
                          zero_tol: float,
                          rel_tol: float = _TABLE_REL_TOL) -> np.ndarray:
    """Rows whose pipeline score and table score disagree beyond rounding:
    a relative difference above ``rel_tol``, or a zero pipeline score whose
    table score clears the cleanup threshold (``zero_tol``)."""
    table = np.asarray(table, dtype=np.float64)
    pipe = np.asarray(pipe, dtype=np.float64)
    return (((pipe > 0) & (np.abs(table - pipe) > rel_tol * pipe))
            | ((pipe == 0) & (table > zero_tol)))


# Cells per block when building the detection-fraction tables. Each block
# holds ~10 bytes per cell x table gene at once (boolean masks and their
# float32 copies): 0.4 GB at 8,192 cells and U = 5,111, where one whole
# group used to take 2.9 GB (Atera's largest type, 62,815 cells).
_FRAC_CELL_BLOCK = 8192


def _detection_fraction_tables(P, P_det, group_labels, groups_order,
                               cell_block: Optional[int] = None):
    """Per group g, the U x U table frac_g[i, j]: the share of g's cells in
    which both diffused profiles are positive and at least one of the two
    genes is detected - the masked pseudo-pair's detection fraction.

    It is ``D'D - E'E`` over the group's cells, with D = (P > 0) and E = D
    on raw-zero entries (Q > 0), divided by the group size. The two Grams
    are integer counts, exact in float32 below 2**24, so building them in
    cell blocks gives the same tables bit for bit, whatever the block
    size, while bounding peak memory (v0.15.0).
    """
    U = P.shape[1]
    block = int(cell_block or _FRAC_CELL_BLOCK)
    frac = {}
    for g in groups_order:
        cells = np.flatnonzero(group_labels == g)
        acc = np.zeros((U, U), dtype=np.float32)
        for start in range(0, len(cells), block):
            cb = cells[start:start + block]
            Dg = P[cb] > 0
            Eg = Dg.copy()
            det_g = P_det[cb].tocoo()
            Eg[det_g.row, det_g.col] = False
            del det_g
            Dg = Dg.astype(np.float32)
            Eg = Eg.astype(np.float32)
            acc += Dg.T @ Dg
            acc -= Eg.T @ Eg
            del Dg, Eg
        if len(cells):
            acc /= np.float32(len(cells))
        frac[g] = acc
    return frac


# ---------------------------------------------------------------------------
# Factorized null p-values
# ---------------------------------------------------------------------------

def compute_factorized_pvalues(
    res_laris: pd.DataFrame,
    bg: LRBackground,
    laris_lr: pd.DataFrame,
    spec_ext: pd.DataFrame,
    group_labels: np.ndarray,
    groups_order,
    ctc_internals: dict,
    scale_factor: float,
    spatial_weight: float,
    mu_gsp: float,
    mu_celltype: float,
    min_null_support: int = 0,
    verbosity: int = 1,
    score_threshold: float = 0.0,
) -> pd.DataFrame:
    """Exact p-values against the factorized matched-gene null.

    For each tested row (sender s, receiver r, pair lig::rec) the null
    support is the k x k pseudo-pairs (i, j) drawn from the ligand's and
    receptor's matched gene sets, each scored through the full pipeline
    composition:

        null[i,j] = spec[i|s] * spec[j|r]
                    * max(delta_ij, 0) ** spatial_weight
                    * frac_s[i,j] * frac_r[i,j]
                    * scale_factor
                    * coloc_ij(s::r)

    where delta_ij combines the Gram-table cosine with the analytic
    shuffled-graph expectation, frac_g is the detection fraction of the
    pseudo-pair in group g (boolean-AND Gram), and coloc applies the
    co-localization COSG transform with the IQR normalization *fitted on
    the real pairs* and applied to pseudo-pairs. Pseudo-pairs that are
    themselves database pairs are excluded. The p-value is the exact tail
    (exceed + 1) / (n_valid + 1); a degenerate all-zero support gives 1.0.

    The null sees exactly what the observed score sees (v0.15.0):

    * a pseudo-pair counts as non-zero only if it reaches
      ``score_threshold``, the cleanup that sets small observed scores to
      0, so ``null_support`` is the number of pseudo-pairs that could have
      been a non-zero observed score;
    * the observed pair is scored through the same tables as its
      pseudo-pairs, so the comparison is between numbers computed with
      identical arithmetic. That score is checked against the pipeline's
      ``interaction_score`` and a difference above 1% raises: it means
      the background and the observed scores were built from different
      data or settings, which would make every p-value wrong. (The two
      paths sum in different orders and precisions, so they agree to
      float32 accuracy, not bitwise.)

    Many pseudo-pairs score exactly zero, because their matched genes are
    not co-detected in the two groups. The p-value counts them, which is
    correct for an unconditional test, *provided* the interactions that
    enter Benjamini-Hochberg are not chosen by the observed score.
    ``_fdr_family`` therefore chooses them by the null: every row whose
    ``null_support`` reaches ``max(1, min_null_support)``. Rows below
    that get p = 1.0 here.

    Returns
    -------
    pandas.DataFrame
        Indexed like ``res_laris``, with columns ``p_value``,
        ``null_support`` (the number of pseudo-pairs scoring at least
        ``score_threshold``, or above zero when it is 0), ``null_size``
        (the row's valid pseudo-pairs) and the descriptive partner ranks
        ``ligand_rank`` and ``receptor_rank``.
    """
    say = print if verbosity else (lambda *a, **k: None)
    n_groups = len(groups_order)
    group_col = {g: i for i, g in enumerate(groups_order)}
    U = len(bg.gene_index)
    P = bg.P

    # --- spec matrix aligned to table genes (genes x groups) -------------
    # Every table gene must have been scored. A missing gene would be
    # zero-filled, its pseudo-pairs would score 0 and the null would be
    # too easy - silently anti-conservative. That happens when the
    # expression passed to runLARIS lacks genes the background was built
    # from (a gene subset, or a cytome read for the LR genes only).
    missing = pd.Index(bg.gene_index).difference(spec_ext.index)
    if len(missing):
        raise ValueError(
            f"{len(missing)} background gene(s) have no cell type "
            f"specificity (e.g. {list(missing[:5])}): the expression data "
            f"passed to runLARIS does not contain every gene the background "
            f"was built from. Pass the same expression object (all genes) "
            f"that was given to prepareLRBackground.")
    spec_tab = spec_ext.reindex(bg.gene_index).fillna(0.0)
    spec_tab = spec_tab.loc[:, list(groups_order)].to_numpy(dtype=np.float64)

    # --- detection-fraction Grams per group -------------------------------
    say("  - detection-fraction tables per cell type...")
    # masked pseudo-pair is nonzero iff both diffused values are nonzero
    # AND the raw-OR mask holds: D_i D_j - E_i E_j with E = D on raw-zero
    # cells (same difference structure as the score itself). Both masks
    # are built per group: D from P (cells x genes, so P[cells] is a
    # contiguous row gather), E as D with the group's detected entries
    # cleared - exactly Q > 0, without a whole-data Q or mask in memory.
    P_det = bg.P_det
    frac = _detection_fraction_tables(P, P_det, group_labels, groups_order)

    # --- co-localization: pass A accumulates the row-sum of cos^2 ---------
    ctXct_cell = ctc_internals['ctXct_cell']         # (n_ctc, n_cells) sparse
    ctc_names = list(ctc_internals['ctc_names'])
    y_norms = ctc_internals['y_norms']
    iqr = ctc_internals['iqr']                       # per ctc column, real-fitted
    v_norm = np.sqrt(bg.G_sq)                        # ||v_ij||, (U, U)

    say(f"  - co-localization pass A over {len(ctc_names)} cell-type pairs...")
    S = np.zeros((U, U), dtype=np.float64)

    _rust = _rust_kernels()
    if _rust is not None and not hasattr(_rust, 'assembly_sweep'):
        warnings.warn(
            "The compiled LARIS kernels (laris._laris) predate this version "
            "of the Python code; using the slower NumPy path. Reinstall LARIS "
            "(or rebuild the extension) to restore them.",
            RuntimeWarning, stacklevel=2)
        _rust = None
    if _rust is not None:
        # threaded sparse decomposition (see src/lib.rs): flops scale
        # with the raw data's sparsity. The kernel reads P (cells x genes,
        # as stored) and Pnz = P - Q = P_det per cell (each cell's detected
        # genes, ascending).
        if not P_det.has_sorted_indices:
            P_det = P_det.sorted_indices()
        _rust_args = (P,
                      P_det.indptr.astype(np.int64),
                      P_det.indices.astype(np.int32),
                      P_det.data.astype(np.float32))
        # the numerator's A and B, refilled for every cell-type pair; the
        # pass A/B kernels read N = A + A' - B from them directly
        _A = np.empty((U, U), dtype=np.float32)
        _B = np.empty((U, U), dtype=np.float32)
        Q = None
    else:
        # NumPy reference path: the masked numerator as two dense products
        # over the co-locating cells, which needs the dense Q
        Q = bg.Q

    def _ctc_y(idx):
        return np.asarray(ctXct_cell[idx].todense()).ravel().astype(np.float32)

    def _ctc_numerator(idx):
        """Fill _A, _B for cell-type pair ``idx``; False if y is all zero."""
        y = _ctc_y(idx)
        if not y.any():
            return False
        # the kernel skips zero-y cells internally (see src/lib.rs)
        _rust.ctc_numerator_ab(*_rust_args, y, _A, _B)
        return True

    def _ctc_cos(idx):
        # NumPy path. y = o_s * o_r is exactly zero wherever the two cell
        # types do not both have neighbours. At fine annotation the median
        # pair co-locates in under 1% of cells, so the zeros are the work.
        y = _ctc_y(idx)
        nz = np.flatnonzero(y)
        if nz.size == 0:
            return np.zeros((U, U), dtype=np.float64)
        # Equivalent arithmetic to the Rust numerator, not bitwise:
        # restricting the product changes the dgemm's internal blocking.
        # P is cells-major, so P[nz] is a contiguous row gather.
        yc = y[nz][:, None]
        Pn, Qn = P[nz], Q[nz]
        N = (Pn * yc).T @ Pn - (Qn * yc).T @ Qn
        den = v_norm * y_norms[idx]
        out = np.zeros_like(N, dtype=np.float64)
        np.divide(N, den, out=out, where=den > 0)
        return out

    # Pass A exploits the symmetry cos[(r,s)] = cos[(s,r)].T (the column
    # vector y = o_s * o_r is symmetric in s and r), so only the upper
    # triangle of cell-type pairs is computed.
    _ctc_pos = {name: i for i, name in enumerate(ctc_names)}
    done = set()
    for idx, name in enumerate(ctc_names):
        if idx in done:
            continue
        a, b = name.split('::')
        mirror = _ctc_pos.get(f"{b}::{a}")
        with_mirror = (mirror is not None and mirror != idx
                       and mirror not in done)
        if _rust is not None:
            # S += c * c (+ c.T * c.T), fused and threaded, bitwise the
            # NumPy statements below; an all-zero y adds exactly nothing
            if _ctc_numerator(idx):
                _rust.cos_sq_accumulate(_A, _B, v_norm, float(y_norms[idx]),
                                        S, with_mirror)
        else:
            c = _ctc_cos(idx)
            S += c * c
            if with_mirror:
                S += c.T * c.T
        done.add(idx)
        if with_mirror:
            done.add(mirror)

    # --- delta blocks are (s, r)-independent: cache per pair --------------
    say("  - assembling per-pair nulls...")
    # database pairs as a sparse boolean U x U matrix (row = ligand gene,
    # col = receptor gene), for vectorized exclusion masks per block
    _db_rows, _db_cols = [], []
    _gene_pos = {g_: i_ for i_, g_ in enumerate(bg.gene_index)}
    for _l, _r in bg.db_pairs:
        li = _gene_pos.get(_l)
        ri = _gene_pos.get(_r)
        if li is not None and ri is not None:
            _db_rows.append(li)
            _db_cols.append(ri)
    _db_bool = sp.csr_matrix(
        (np.ones(len(_db_rows), dtype=np.int8), (_db_rows, _db_cols)),
        shape=(U, U))
    pair_cache = {}

    def _delta_block(lig, rec):
        # Per pair, cached: everything (sender, receiver)-independent,
        # pre-flattened to the valid (non-database) block entries so the
        # per-row work in pass B is a single gather plus a compare - the
        # naive per-row np.ix_ extraction of four U x U tables was the
        # dominant cost of the whole computation (450 s of 745 s on
        # tonsil; the matmuls were only 218 s).
        key = (lig, rec)
        if key not in pair_cache:
            ii = bg.matched_ligand[lig]
            jj = bg.matched_receptor[rec]
            d = (_pair_gsp_from_tables(bg, ii, jj)
                 - mu_gsp * _pair_random_gsp_from_tables(bg, ii, jj))
            if spatial_weight == 0:
                dw = np.ones_like(d)
            else:
                dw = np.clip(d, 0.0, None) ** spatial_weight
            db_mask = np.asarray(
                _db_bool[np.ix_(ii, jj)].todense()).astype(bool)
            valid = ~db_mask.ravel()
            k_i, k_j = len(ii), len(jj)
            flat = (ii[:, None] * U + jj[None, :]).ravel()[valid]
            rowmap = np.repeat(np.arange(k_i), k_j)[valid]
            colmap = np.tile(np.arange(k_j), k_i)[valid]
            pair_cache[key] = (ii, jj, dw.ravel()[valid],
                               flat, rowmap, colmap)
        return pair_cache[key]

    # Fused (Rust) path: no per-pair blocks. The delta weight depends on the
    # gene pair only, so one dense U x U table serves every LR pair, with
    # NaN on database pairs (a NaN score is never counted). Each pair is
    # then just its two matched sets: the ligand side as, per table row,
    # the pairs whose matched set holds that gene (row_ptr/row_pair), and
    # the receptor side as a sorted gene list (col_ptr/col_gene). At
    # k = 200 this replaces ~4 GB of flattened per-pair arrays (tonsil).
    if _rust is not None:
        dwx = _dense_delta_weight(bg, mu_gsp, spatial_weight)
        if _db_rows:
            dwx[np.asarray(_db_rows), np.asarray(_db_cols)] = np.nan
        uniq = list(dict.fromkeys(zip(res_laris['ligand'],
                                      res_laris['receptor'])))
        _pair_index = {pr: q for q, pr in enumerate(uniq)}
        n_pairs = len(uniq)
        lig_sets = [np.asarray(bg.matched_ligand[l_], dtype=np.int64)
                    for l_, _ in uniq]
        rec_sets = [np.sort(np.asarray(bg.matched_receptor[r_], np.int64))
                    for _, r_ in uniq]
        col_ptr = np.zeros(n_pairs + 1, dtype=np.int64)
        col_ptr[1:] = np.cumsum([len(c_) for c_ in rec_sets])
        col_gene = (np.concatenate(rec_sets).astype(np.int32)
                    if n_pairs else np.empty(0, np.int32))
        lig_rows = (np.concatenate(lig_sets) if n_pairs
                    else np.empty(0, np.int64))
        lig_pair = np.repeat(np.arange(n_pairs, dtype=np.int32),
                             [len(i_) for i_ in lig_sets])
        _order = np.argsort(lig_rows, kind='stable')
        row_pair = np.ascontiguousarray(lig_pair[_order])
        row_ptr = np.zeros(U + 1, dtype=np.int64)
        row_ptr[1:] = np.cumsum(np.bincount(lig_rows, minlength=U))
        # valid (non-database) pseudo-pairs per LR pair: the denominator
        sizes = np.array(
            [len(i_) * len(j_) - _db_bool[i_][:, j_].count_nonzero()
             for i_, j_ in zip(lig_sets, rec_sets)], dtype=np.int64)
        del lig_rows, lig_pair, _order
        _threads = _n_threads()

    # --- the observed pairs, scored through the same tables --------------
    # Same formula as _delta_block, element by element, so an observed pair
    # and its pseudo-pairs share their arithmetic exactly.
    _obs_pairs = list(dict.fromkeys(zip(res_laris['ligand'],
                                        res_laris['receptor'])))
    _li = bg.positions([l_ for l_, _ in _obs_pairs])
    _ri = bg.positions([r_ for _, r_ in _obs_pairs])
    _gsq = bg.G_sq[_li, _ri]
    _den = np.sqrt(_gsq * bg.G_W2[_li, _ri])
    _cos = np.zeros_like(_den)
    np.divide(bg.G_W[_li, _ri], _den, out=_cos, where=_den > 0)
    _d = _cos - mu_gsp * _analytic_random_gsp(
        bg.G_dot[_li, _ri] / bg.n_cells, _gsq / bg.n_cells, bg.n_cells, bg.R)
    if spatial_weight == 0:
        _dw_obs = np.ones_like(_d)
    else:
        _dw_obs = np.clip(_d, 0.0, None) ** spatial_weight
    _obs_of = {pr: (i_, j_, w_) for pr, i_, j_, w_ in
               zip(_obs_pairs, _li, _ri, _dw_obs)}
    thr = float(score_threshold)
    _zero_tol = max(thr, 1e-12) * (1 + 1e-4)

    def _observed_through_tables(gdf, spec_s, spec_r, m_at):
        """Observed scores from the tables, cleaned like the pipeline's.

        ``m_at(flat)`` returns the group's combined table M at flat indices.
        Raises if the scores disagree with the pipeline's interaction_score:
        the background and the observed scores were then built differently.
        """
        trip = [_obs_of[pr] for pr in zip(gdf['ligand'], gdf['receptor'])]
        i_ = np.fromiter((t[0] for t in trip), np.int64, len(trip))
        j_ = np.fromiter((t[1] for t in trip), np.int64, len(trip))
        w_ = np.fromiter((t[2] for t in trip), np.float64, len(trip))
        table = spec_s[i_] * spec_r[j_] * w_ * m_at(i_ * U + j_)  # kernel order
        if thr > 0:
            table = np.where(table < thr, 0.0, table)
        pipe = gdf['interaction_score'].to_numpy(np.float64)
        bad = _table_score_mismatch(table, pipe, _zero_tol)
        if bad.any():
            k_ = np.flatnonzero(bad)[:3]
            ex = [f"{gdf['ligand'].iloc[q]}::{gdf['receptor'].iloc[q]} "
                  f"pipeline {pipe[q]:.6g} vs tables {table[q]:.6g}"
                  for q in k_]
            raise ValueError(
                f"{int(bad.sum())} observed interaction score(s) in "
                f"{gdf['sender'].iloc[0]} -> {gdf['receiver'].iloc[0]} "
                f"differ by more than {_TABLE_REL_TOL:.0%} from the same pair "
                f"scored through the background tables (e.g. "
                f"{'; '.join(ex)}). The background was built from different "
                f"data or settings than the observed scores: pass the same "
                f"expression object (and layer) to prepareLRInteraction and "
                f"prepareLRBackground, and the same graph settings "
                f"(number_nearest_neighbors, use_rep_spatial, sigma, "
                f"section_key).")
        return table

    # --- pass B: one ctc at a time, all its rows scored -------------------
    p_vals = np.full(len(res_laris), np.nan)
    n_pos = np.zeros(len(res_laris), dtype=np.int64)
    n_size = np.zeros(len(res_laris), dtype=np.int64)
    pos_of = {ix: i for i, ix in enumerate(res_laris.index)}
    grouped = res_laris.groupby(['sender', 'receiver'], observed=True)
    lam_mu = mu_celltype

    def _same_colocation(i1, i2):
        # the mirror (r, s) of (s, r) has the same co-location vector, norm
        # and IQR, so the same coloc table - checked bitwise, not assumed
        return (y_norms[i1] == y_norms[i2]
                and iqr[ctc_names[i1]] == iqr[ctc_names[i2]]
                and np.array_equal(ctXct_cell[i1].toarray(),
                                   ctXct_cell[i2].toarray()))

    def _store(gdf, ex, npos):
        pid = np.fromiter((_pair_index[pr] for pr in
                           zip(gdf['ligand'], gdf['receptor'])),
                          dtype=np.int64, count=len(gdf))
        ex, npos, size = ex[pid], npos[pid], sizes[pid]
        degenerate = (size == 0) | (npos == 0) | (npos < min_null_support)
        rows = np.fromiter((pos_of[ix] for ix in gdf.index),
                           dtype=np.int64, count=len(gdf))
        p_vals[rows] = np.where(degenerate, 1.0, (ex + 1) / (size + 1))
        n_pos[rows] = npos
        n_size[rows] = size

    def _per_pair(gdf, obs):
        out = np.full(n_pairs, np.inf)      # pairs without a row: never count
        out[[_pair_index[pr] for pr in
             zip(gdf['ligand'], gdf['receptor'])]] = obs
        return out

    # --- partner ranks (descriptive) --------------------------------------
    # The cross design compares the observed pair with pseudo-pairs that
    # share neither of its genes. These two columns look at the pairs that
    # share one: the observed receptor with each of the ligand's matched
    # genes (ligand_rank), and the observed ligand with each of the
    # receptor's (receptor_rank). (1 + #{peer pair >= observed}) /
    # (1 + #{valid peer pairs}), so 1/(k + 1) means the gene beats all its
    # peers given its partner; a call that leans on one partner has a
    # small rank on that side only. Resolution 1/(k + 1): descriptive, not
    # a test. Rows whose observed score is 0 keep 1.0.
    lig_rank = np.ones(len(res_laris))
    rec_rank = np.ones(len(res_laris))

    def _dw_dense(gi, gj):
        return dwx[gi, gj]

    def _dw_elementwise(gi, gj):
        # the same arithmetic as _dense_delta_weight, entry by entry, with
        # database pairs excluded (NaN), as in the fused path's dense table
        num = bg.G_W[gi, gj]
        den = np.sqrt(bg.G_sq[gi, gj] * bg.G_W2[gi, gj])
        gsp = np.zeros_like(num)
        np.divide(num, den, out=gsp, where=den > 0)
        d = gsp - mu_gsp * _analytic_random_gsp(
            bg.G_dot[gi, gj] / bg.n_cells, bg.G_sq[gi, gj] / bg.n_cells,
            bg.n_cells, bg.R)
        if spatial_weight == 0:
            dw = np.ones_like(d)
        else:
            dw = np.clip(d, 0.0, None) ** spatial_weight
        dw[np.asarray(_db_bool[gi, gj]).ravel().astype(bool)] = np.nan
        return dw

    def _partner_ranks(gdf, obs_rows, spec_l, spec_r, m_at, dw_at):
        obs_rows = np.asarray(obs_rows, dtype=np.float64)
        sel = np.flatnonzero(obs_rows > 0)
        if sel.size == 0:
            return
        rows = np.fromiter((pos_of[ix] for ix in gdf.index[sel]),
                           dtype=np.int64, count=sel.size)
        ligs = gdf['ligand'].to_numpy()[sel]
        recs = gdf['receptor'].to_numpy()[sel]
        o = obs_rows[sel]
        trip = [_obs_of[pr] for pr in zip(ligs, recs)]
        iL = np.fromiter((t[0] for t in trip), np.int64, sel.size)
        iR = np.fromiter((t[1] for t in trip), np.int64, sel.size)
        for sets, out, ligand_side in (
                ([bg.matched_ligand[l_] for l_ in ligs], lig_rank, True),
                ([bg.matched_receptor[r_] for r_ in recs], rec_rank, False)):
            lens = np.fromiter((len(s_) for s_ in sets), np.int64, sel.size)
            n_valid = np.zeros(sel.size)
            n_ge = np.zeros(sel.size)
            if lens.sum():
                owner = np.repeat(np.arange(sel.size), lens)
                peers = np.concatenate(
                    [np.asarray(s_, dtype=np.int64) for s_ in sets])
                gi, gj = ((peers, iR[owner]) if ligand_side
                          else (iL[owner], peers))
                # kernel order: ((spec * spec) * delta weight) * M
                v = ((spec_l[gi] * spec_r[gj]) * dw_at(gi, gj)) * m_at(
                    gi * U + gj)
                valid = ~np.isnan(v)
                n_valid = np.bincount(owner, weights=valid,
                                      minlength=sel.size)
                n_ge = np.bincount(owner, weights=valid & (v >= o[owner]),
                                   minlength=sel.size)
            out[rows] = (1.0 + n_ge) / (1.0 + n_valid)

    if _rust is not None:
        # One cell-type pair and its mirror at a time: they share the
        # numerator, the coloc table and frac_s * frac_r (symmetric in s
        # and r), so M is built once and one sweep counts both directions.
        groups = {key: gdf for key, gdf in grouped}
        handled = set()
        x = np.empty((U, U), dtype=np.float64)     # the coloc table, reused
        with ThreadPoolExecutor(max_workers=_threads) as pool:
            for (sender, receiver), gdf in groups.items():
                if (sender, receiver) in handled:
                    continue
                handled.add((sender, receiver))
                idx = _ctc_pos.get(f"{sender}::{receiver}")
                if idx is None:
                    continue
                mirror = None
                if sender != receiver and (receiver, sender) in groups:
                    midx = _ctc_pos.get(f"{receiver}::{sender}")
                    if midx is not None and _same_colocation(idx, midx):
                        mirror = groups[(receiver, sender)]
                        handled.add((receiver, sender))
                if _ctc_numerator(idx):
                    # lam * c / iqr fused; log1p in NumPy (see src/lib.rs)
                    _rust.coloc_x(_A, _B, v_norm, float(y_norms[idx]), S,
                                  float(lam_mu), float(iqr[ctc_names[idx]]),
                                  x)
                    _log1p_inplace(x, pool, _threads)
                else:                            # y all zero: coloc = 0
                    x.fill(0.0)
                fs, fr = frac[sender], frac[receiver]
                x_flat, fs_flat, fr_flat = x.ravel(), fs.ravel(), fr.ravel()

                def m_at(f):
                    # M = coloc * (frac_s * frac_r) * scale at flat indices
                    return ((x_flat[f] * (fs_flat[f].astype(np.float64)
                                          * fr_flat[f])) * scale_factor)

                spec_a = np.ascontiguousarray(spec_tab[:, group_col[sender]])
                spec_b = np.ascontiguousarray(spec_tab[:, group_col[receiver]])
                obs_rows1 = _observed_through_tables(gdf, spec_a, spec_b, m_at)
                obs1 = _per_pair(gdf, obs_rows1)
                if mirror is None:
                    obs_rows2, obs2 = None, np.empty(0)
                else:
                    obs_rows2 = _observed_through_tables(
                        mirror, spec_b, spec_a, m_at)
                    obs2 = _per_pair(mirror, obs_rows2)
                ex1, np1, ex2, np2 = _rust.assembly_sweep(
                    x, fs, fr, float(scale_factor), dwx, spec_a, spec_b,
                    row_ptr, row_pair, col_ptr, col_gene, obs1, obs2, thr)
                _store(gdf, ex1, np1)
                _partner_ranks(gdf, obs_rows1, spec_a, spec_b, m_at,
                               _dw_dense)
                if mirror is not None:
                    _store(mirror, ex2, np2)
                    _partner_ranks(mirror, obs_rows2, spec_b, spec_a, m_at,
                                   _dw_dense)
        grouped = ()                    # every group was handled above

    for (sender, receiver), gdf in grouped:
        ctc = f"{sender}::{receiver}"
        if ctc not in ctc_names:
            continue
        idx = ctc_names.index(ctc)
        cosU = _ctc_cos(idx)
        cos2 = cosU * cosU
        den = (1.0 - lam_mu) * cos2 + lam_mu * S
        lam = np.zeros_like(cosU)
        np.divide(cos2, den, out=lam, where=den != 0)
        colocU = _iqr_apply(lam * cosU, iqr[ctc])
        s_col = group_col[sender]
        r_col = group_col[receiver]
        # one combined table per (sender, receiver): the three U x U
        # factors that vary with the group, multiplied once
        M = (colocU
             * (frac[sender].astype(np.float64) * frac[receiver])
             * scale_factor).ravel()
        spec_s = spec_tab[:, s_col]
        spec_r = spec_tab[:, r_col]
        obs = _observed_through_tables(gdf, spec_s, spec_r, M.__getitem__)
        _partner_ranks(gdf, obs, spec_s, spec_r, M.__getitem__,
                       _dw_elementwise)
        for row, o in zip(gdf.itertuples(), obs):
            ii, jj, dw_flat, flat, rowmap, colmap = _delta_block(
                row.ligand, row.receptor)
            n_size[pos_of[row.Index]] = len(flat)
            if len(flat) == 0:
                p_vals[pos_of[row.Index]] = 1.0
                continue
            null = (spec_s[ii][rowmap] * spec_r[jj][colmap]
                    * dw_flat * M[flat])
            npos_row = int(np.count_nonzero(
                (null >= thr) if thr > 0 else (null > 0)))
            n_pos[pos_of[row.Index]] = npos_row
            if npos_row == 0 or npos_row < min_null_support:
                p_vals[pos_of[row.Index]] = 1.0
                continue
            exceed = int(np.count_nonzero(null >= o))
            p_vals[pos_of[row.Index]] = (exceed + 1) / (len(flat) + 1)

    if len(p_vals) and not np.isfinite(p_vals).any():
        # Every row was skipped, which means no (sender, receiver) group in
        # the results matched a cell-type pair the co-localization step
        # built. Returning a column of NaN looks like a p-value of "no
        # evidence" and silently propagates into the FDR; it is a broken
        # run, so say so. Reported by a user on degenerate input.
        raise ValueError(
            "No interaction could be tested: none of the "
            f"{len(res_laris):,} sender-receiver rows matched a cell-type "
            f"pair among the {len(ctc_names):,} the co-localization step "
            "produced. This usually means `groupby` labels changed between "
            "prepareLRInteraction and runLARIS, or that every group was "
            "dropped as too small. Check that the groupby column is the "
            "same in both calls and has at least two non-empty groups.")

    floor = max(1, int(min_null_support))
    tested = n_pos >= floor
    n_tested = int(np.count_nonzero(tested))
    if n_tested:
        thin = int(np.count_nonzero(tested & (n_pos < 100)))
        if thin > 0.05 * n_tested:
            warnings.warn(
                f"{thin:,} of {n_tested:,} tested interactions "
                f"({100 * thin / n_tested:.1f}%) have a null with fewer "
                "than 100 non-zero pseudo-pairs. Their p-values are "
                "reported against a denominator of "
                f"{int(np.median(n_size[tested])):,} but can only "
                "resolve about 1/(effective support). Inspect the "
                "'null_support' column; the default min_null_support='auto' "
                "leaves out rows below 2 x n_matched_genes.",
                UserWarning, stacklevel=2)
    return pd.DataFrame({'p_value': p_vals, 'null_support': n_pos,
                         'null_size': n_size, 'ligand_rank': lig_rank,
                         'receptor_rank': rec_rank}, index=res_laris.index)


# ---------------------------------------------------------------------------
# Cytome-source support
# ---------------------------------------------------------------------------

def _is_cytome_source(obj) -> bool:
    from pathlib import Path as _Path
    if isinstance(obj, (str, _Path)):
        return str(obj).endswith('.cytome')
    return type(obj).__module__.startswith('cytome')


def _prepare_background_from_cytome(
    source, lr_df, n_pool, augment_pool, n_matched_genes, matching,
    exclude_self, number_nearest_neighbors,
    use_rep_spatial, use_rep_gsp, n_nearest_neighbors_gsp, sigma_gsp,
    layer, section_key, block_size, verbosity,
    stats_block_genes: int = 2048, sigma='adaptive',
) -> "LRBackground":
    """Cytome path: the in-memory definition, computed out of core.

    1. Per-gene mean/variance over ALL genes, in gene blocks read sparse
       and in the stored dtype. Each gene's statistic depends only on its
       own column, so the values equal the in-memory ones bit for bit.
    2. :func:`_select_matching` on those whole-matrix statistics - the
       same function on the same inputs as the in-memory path, so the
       pool and every matched set are identical.
    3. One sparse read of the table genes, in file order, then the shared
       :func:`_build_tables`.

    Memory stays bounded by the table genes (as in v0.14.0); only the
    gene blocks of step 1 and the table genes of step 3 are ever loaded.

    Up to v0.14.0 step 2 was re-run on the loaded pool + LR subset, so the
    matching features were standardised on that subset rather than on the
    whole matrix; ``matching``, ``exclude_self`` and ``layer`` were not
    forwarded; and reads went through a dense float32 buffer. Together
    these moved 3-37 FDR<0.05 calls against the in-memory path on the
    Xenium benchmark (817 vs 780 at 10,000 cells).
    """
    from ..preprocessing._io import (_open_cytome, _resolve_x_layer,
                                     _cytome_gene_names,
                                     _read_cytome_columns, readCytome)

    say = print if verbosity else (lambda *a, **k: None)
    modality = 'RNA'
    ds, opened_here = _open_cytome(source)
    try:
        cy_layer = _resolve_x_layer(ds, modality) if layer is None else layer
        all_genes = _cytome_gene_names(ds)
        n_genes = len(all_genes)
        n_matched_genes = _resolve_n_matched_genes(n_matched_genes, lr_df,
                                                   all_genes, n_pool)
        say(f"  - cytome stats pass over {n_genes:,} genes "
            f"(sparse gene blocks, stored dtype)...")
        m = v = None
        detected = np.zeros(n_genes, dtype=bool)
        for start in range(0, n_genes, stats_block_genes):
            cols = np.arange(start, min(start + stats_block_genes, n_genes))
            block = _read_cytome_columns(ds, modality, cy_layer, cols)
            mb, vb = _gene_moments(block)
            if m is None:            # the dtype the in-memory path would get
                m = np.empty(n_genes, dtype=mb.dtype)
                v = np.empty(n_genes, dtype=vb.dtype)
            m[cols] = mb
            v[cols] = vb
            detected[cols] = _gene_detected(block)
            del block

        sel = _select_matching(all_genes, m, v, lr_df, n_pool, augment_pool,
                               n_matched_genes, matching, exclude_self, say,
                               detected=detected)

        say(f"  - loading {len(sel.gene_index):,} table genes "
            f"(pool + LR) from cytome...")
        # file order (gene_pos is sorted), exactly the in-memory column order
        sub = readCytome(ds, genes=list(sel.gene_index), modality=modality,
                         layer=cy_layer)
    finally:
        if opened_here:
            ds.close()
    if not sub.var_names.equals(sel.gene_index):
        raise RuntimeError("cytome table-gene read returned a different gene "
                           "set or order than selected")

    n_cells = sub.n_obs
    tables = _build_tables(
        sub, sub.X, n_cells, number_nearest_neighbors, use_rep_spatial,
        use_rep_gsp, n_nearest_neighbors_gsp, sigma_gsp, section_key,
        block_size, say, sigma=sigma)
    return _assemble_background(
        sel, tables, n_cells, lr_df, n_matched_genes,
        dict(n_pool=n_pool, n_matched_genes=n_matched_genes,
             matching=matching, exclude_self=exclude_self,
             number_nearest_neighbors=number_nearest_neighbors,
             use_rep_spatial=use_rep_spatial, sigma=sigma,
             use_rep_gsp=use_rep_gsp,
             n_nearest_neighbors_gsp=n_nearest_neighbors_gsp,
             sigma_gsp=sigma_gsp, section_key=section_key, layer=cy_layer),
        say)

# ---------------------------------------------------------------------------
# Calibration control
# ---------------------------------------------------------------------------

def _degree_preserving_decoys(lr_df, rng, data=None, n_swap_rounds=20):
    """Rewire the database by double-edge swaps, preserving every degree.

    Each swap replaces (L1,R1),(L2,R2) with (L1,R2),(L2,R1), which leaves
    both degree sequences untouched. Swaps that would duplicate an
    existing decoy pair are rejected. Pairs that survive as real pairs
    are reported in ``.attrs['n_real_retained']``: with a hub-heavy
    database a perfect rewiring is not always reachable, and the caller
    should exclude those rows rather than count them as decoys.
    """
    lig = lr_df['ligand'].astype(str).to_numpy()
    rec = lr_df['receptor'].astype(str).to_numpy()
    if data is not None:
        present = set(np.asarray(data.var_names, dtype=str))
        keep = np.array([l in present and r in present
                         for l, r in zip(lig, rec)])
        lig, rec = lig[keep], rec[keep]
    real = set(zip(lig, rec))
    edges = list(zip(lig, rec))
    current = set(edges)
    n = len(edges)
    target = n_swap_rounds * n
    done = 0
    for _ in range(target * 12):          # bounded: rejections are common
        if done >= target:
            break
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue
        l1, r1 = edges[i]
        l2, r2 = edges[j]
        if l1 == l2 or r1 == r2:
            continue
        if (l1, r2) in current or (l2, r1) in current:
            continue
        current.discard((l1, r1)); current.discard((l2, r2))
        current.add((l1, r2)); current.add((l2, r1))
        edges[i] = (l1, r2); edges[j] = (l2, r1)
        done += 1

    decoy = pd.DataFrame(edges, columns=['ligand', 'receptor'])
    decoy['interaction_name'] = decoy['ligand'] + '::' + decoy['receptor']
    retained = int(sum(1 for e in edges if e in real))
    decoy.attrs['n_real_retained'] = retained
    decoy.attrs['n_swaps'] = done
    if retained:
        warnings.warn(
            f"{retained:,} of {len(decoy):,} rewired pairs coincide with a "
            "real database pair; a hub-heavy database cannot always be "
            "fully rewired. Exclude them before using this as a null "
            "(they are marked in the 'is_real' column).", UserWarning)
    decoy['is_real'] = [e in real for e in edges]
    return decoy


def permuteLRPairs(
    lr_df: pd.DataFrame,
    data=None,
    random_seed: int = 0,
    preserve_genes: bool = True,
    method: str = 'degree',
) -> pd.DataFrame:
    """Build a decoy LR database: the calibration control for this test.

    LARIS's p-value is a *competitive* test in the sense of Goeman &
    Bühlmann (2007): it asks whether a real ligand-receptor pair beats
    expression-matched genes. The null distribution of a competitive test
    is generated by permuting **gene identity**, not sample labels.

    Shuffling cell-type labels or spatial coordinates is the calibration
    procedure for a *self-contained* test (the design CellPhoneDB and
    CellChat use), and it is not this test's null. Under a label shuffle
    every group converges to the tissue average, so the cell-type factors
    become a constant that cancels between the observed score and the
    null, leaving a comparison of spatial co-expression that real
    database pairs win on merit. Such a run can therefore return *more*
    calls than the real one, and that is not evidence of a defect. See
    tutorial 07.

    This function builds the control that does apply: a database of the
    same size in which the pairings are random. Every returned pair is a
    decoy, so a correctly calibrated run over it should yield p-values
    that are close to uniform and essentially no FDR-significant calls,
    at any dataset size.

    Parameters
    ----------
    lr_df : pandas.DataFrame
        The real database, with 'ligand' and 'receptor' columns. Its size
        sets the size of the decoy database so that the multiple-testing
        burden matches.
    data : AnnData, optional
        If given, decoy genes are drawn from ``data.var_names``. Without
        it they are drawn from the genes appearing in ``lr_df``.
    random_seed : int, default=0
        Seed for the pairing.
    preserve_genes : bool, default=True
        Draw ligands from the real ligand pool and receptors from the
        real receptor pool, re-pairing them at random. This keeps the
        marginal expression distribution of each side realistic and
        changes only the pairing, which is the thing being tested. Set
        False to draw both sides from all available genes. Ignored when
        ``method='degree'``, which always reuses the real genes.
    method : {'degree', 'uniform'}, default='degree'
        How the decoys are drawn.

        ``'degree'`` rewires the real database by repeated double-edge
        swaps: pick two pairs (L1,R1) and (L2,R2) and replace them with
        (L1,R2) and (L2,R1). Every gene therefore appears in **exactly as
        many decoy pairs as real ones**, and only the pairing changes.

        This matters more than it sounds. Curated databases are strongly
        hub-structured - in CellChatDB's human table restricted to a
        tonsil, ITGB1 appears in 55 pairs and the median gene in 2, with
        the top 10% of genes carrying ~40% of the pairs. Drawing decoys
        uniformly flattens that: it under-samples the broadly expressed
        integrin and collagen hubs, which are individually unremarkable
        and dilute the real database's hit rate, and over-samples rare
        cell-type-restricted genes, which are not. A uniform decoy is
        therefore an *easier* database than the real one and will
        overstate the false-positive rate.

        ``'uniform'`` is the naive version, kept because it is the
        obvious thing to try and the comparison between the two is
        informative.

    Returns
    -------
    pandas.DataFrame
        Columns 'ligand', 'receptor' and 'interaction_name', containing no
        pair present in ``lr_df``.

    Examples
    --------
    >>> decoy = la.tl.permuteLRPairs(lr_df, adata, random_seed=0)
    >>> bg_d = la.tl.prepareLRBackground(adata, decoy,
    ...                                  use_rep_spatial="X_spatial")
    >>> lr_d = la.tl.prepareLRInteraction(adata, decoy,
    ...                                   use_rep_spatial="X_spatial")
    >>> _, res_d = la.tl.runLARIS(lr_d, adata, groupby="cell_type",
    ...                           background=bg_d)
    >>> (res_d.p_value_fdr < 0.05).sum()      # should be ~0
    """
    rng = np.random.default_rng(random_seed)
    n_target = len(lr_df)
    real = set(zip(lr_df['ligand'].astype(str), lr_df['receptor'].astype(str)))

    if method not in ('degree', 'uniform'):
        raise ValueError("method must be 'degree' or 'uniform'")
    if method == 'degree':
        return _degree_preserving_decoys(lr_df, rng, data)

    if preserve_genes:
        lig_pool = pd.unique(lr_df['ligand'].astype(str))
        rec_pool = pd.unique(lr_df['receptor'].astype(str))
    else:
        if data is None:
            raise ValueError(
                "preserve_genes=False needs `data` to draw genes from.")
        allg = np.asarray(data.var_names, dtype=str)
        lig_pool = rec_pool = allg

    if data is not None:
        present = set(np.asarray(data.var_names, dtype=str))
        lig_pool = np.array([g for g in lig_pool if g in present])
        rec_pool = np.array([g for g in rec_pool if g in present])
    lig_pool = np.asarray(lig_pool, dtype=str)
    rec_pool = np.asarray(rec_pool, dtype=str)
    if len(lig_pool) == 0 or len(rec_pool) == 0:
        raise ValueError(
            "No database genes are present in `data`; cannot build decoys.")

    # Sample with rejection so that no decoy coincides with a real pair
    # (a real pair among the decoys would not be a decoy) and no decoy is
    # a gene paired with itself.
    max_unique = len(lig_pool) * len(rec_pool)
    if n_target > max_unique // 2:
        n_target = max(1, max_unique // 2)
    seen, out = set(), []
    for _ in range(200):
        if len(out) >= n_target:
            break
        need = (n_target - len(out)) * 2 + 16
        li = lig_pool[rng.integers(0, len(lig_pool), need)]
        ri = rec_pool[rng.integers(0, len(rec_pool), need)]
        for a, b in zip(li, ri):
            if len(out) >= n_target:
                break
            if a == b or (a, b) in real or (a, b) in seen:
                continue
            seen.add((a, b))
            out.append((a, b))
    if len(out) < n_target:
        warnings.warn(
            f"Only {len(out):,} decoy pairs could be drawn (asked for "
            f"{n_target:,}); the gene pools are small.", UserWarning)

    decoy = pd.DataFrame(out, columns=['ligand', 'receptor'])
    decoy['interaction_name'] = decoy['ligand'] + '::' + decoy['receptor']
    return decoy


# ---------------------------------------------------------------------------
# Target-decoy empirical FDR
# ---------------------------------------------------------------------------

def decoyFDR(
    target_res: pd.DataFrame,
    decoy_res: pd.DataFrame,
    p_col: str = 'p_value',
    pseudocount: float = 1.0,
) -> pd.Series:
    """Empirical pairing-FDR for each target row, by target-decoy estimation.

    The factorized p-value certifies that a pair's expression is arranged
    with respect to the tested cell types beyond expression-matched
    chance. It does not certify the *pairing*: a degree-preserving
    rewiring of the database reproduces part of the calls (tonsil,
    v0.15.0: 50 against the real database's 95 at FDR < 0.05), because
    for marker-structured pairs the pairing is unidentifiable from
    expression and coordinates alone. See tutorial 07.

    This function measures that error rate empirically, the way
    proteomics has done for two decades (target-decoy search, Elias &
    Gygi 2007): run the same pipeline on a decoy database whose pairings
    are scrambled but whose genes and degrees are identical
    (:func:`permuteLRPairs` with ``method='degree'``), and estimate, at
    each threshold t,

        FDR(t) = (decoy rows with p <= t, +pseudocount) / n_decoy
                 -------------------------------------------------
                 (target rows with p <= t)               / n_target

    monotonized into a q-value (minimum over all looser thresholds) and
    clipped to 1. The +1 pseudocount keeps the estimate conservative
    when decoy counts are small.

    Because the degree decoy re-pairs real genes, a fraction of decoys
    may be genuine unannotated interactions; on tonsil this contamination
    measures 0.3% (pathway-sharing check), so the estimate is effectively
    unbiased there, and in general it errs conservative.

    Parameters
    ----------
    target_res, decoy_res : pandas.DataFrame
        Results from :func:`runLARIS` on the real and decoy databases,
        run with the same settings and (ideally) the same background.
        Rows of ``decoy_res`` whose pair coincides with a real database
        pair should be excluded first (``computeDecoyFDR`` does this).
    p_col : str, default='p_value'
        Column holding the raw p-values. Raw, not BH-adjusted: the two
        runs have different multiplicity structures, and the raw p is the
        comparable scale.
    pseudocount : float, default=1.0
        Added to the decoy count at every threshold.

    Returns
    -------
    pandas.Series
        ``q_decoy``, aligned to ``target_res.index``; NaN where the
        target p is NaN.
    """
    pt = pd.to_numeric(target_res[p_col], errors='coerce').to_numpy(float)
    pdec = pd.to_numeric(decoy_res[p_col], errors='coerce').to_numpy(float)
    pdec = pdec[np.isfinite(pdec)]
    n_d = len(pdec)
    if n_d == 0:
        raise ValueError("decoy_res has no finite p-values; run the decoy "
                         "database through runLARIS first.")
    finite = np.isfinite(pt)
    n_t = int(finite.sum())
    if n_t == 0:
        return pd.Series(np.nan, index=target_res.index, name='q_decoy')

    pt_sorted = np.sort(pt[finite])
    pdec_sorted = np.sort(pdec)
    # counts at each distinct target p (ties handled by side='right')
    uniq = np.unique(pt_sorted)
    t_cnt = np.searchsorted(pt_sorted, uniq, side='right')
    d_cnt = np.searchsorted(pdec_sorted, uniq, side='right')
    fdr = ((d_cnt + pseudocount) / n_d) / (t_cnt / n_t)
    # q-value: minimum over all thresholds at least as loose
    q_at_uniq = np.minimum.accumulate(np.clip(fdr, 0.0, 1.0)[::-1])[::-1]
    q = np.full(len(pt), np.nan)
    q[finite] = q_at_uniq[np.searchsorted(uniq, pt[finite])]
    return pd.Series(q, index=target_res.index, name='q_decoy')


def computeDecoyFDR(
    data,
    lr_df: pd.DataFrame,
    target_res: pd.DataFrame,
    background: "LRBackground",
    random_seed: int = 0,
    prepare_kwargs: Optional[dict] = None,
    run_kwargs: Optional[dict] = None,
    verbosity: int = 1,
):
    """One-call target-decoy FDR: build the decoy, run it, return q_decoy.

    Builds a degree-preserving decoy database, runs it through
    ``prepareLRInteraction`` + ``runLARIS`` against the *same* background
    (valid because the rewiring preserves the exact gene multiset, so
    every per-gene matched set is identical), drops decoy rows whose
    pair coincides with a real one, and returns
    ``(q_decoy, decoy_res)``.

    ``prepare_kwargs`` and ``run_kwargs`` must repeat the settings of the
    target run (``use_rep_spatial``, ``groupby``, ...); they are passed
    through verbatim.

    Parameters
    ----------
    data : AnnData, str or cytome dataset
        The expression data of the target run.
    lr_df : pandas.DataFrame
        The real LR database of the target run ('ligand'/'receptor'
        columns); it is rewired into the decoy.
    target_res : pandas.DataFrame
        The target run's results (``runLARIS``'s cell-type table), whose
        raw ``p_value`` the q-values are computed for.
    background : LRBackground
        The target run's background. The decoy keeps every gene and degree,
        so the same matched sets and tables apply.
    random_seed : int, default=0
        Seed of the rewiring.
    prepare_kwargs : dict, optional
        Keyword arguments for ``prepareLRInteraction`` on the decoy
        database; repeat the target run's settings.
    run_kwargs : dict, optional
        Keyword arguments for ``runLARIS`` on the decoy database; repeat the
        target run's settings.
    verbosity : int, default=1
        1 prints the decoy's size; 0 is silent.

    Returns
    -------
    q_decoy : pandas.Series
        The target-decoy q-value of every target row
        (:func:`decoyFDR`), aligned to ``target_res.index``.
    decoy_res : pandas.DataFrame
        The decoy run's results, without rows whose pair coincides with a
        real database pair.

    Examples
    --------
    >>> q, dec = la.tl.computeDecoyFDR(
    ...     adata, lr_df, res, background=bg,
    ...     prepare_kwargs={"use_rep_spatial": "X_spatial"},
    ...     run_kwargs={"use_rep": "X_spatial",
    ...                 "use_rep_spatial": "X_spatial",
    ...                 "groupby": "cell_type"})
    >>> res["q_decoy"] = q
    """
    from ._prepare import prepareLRInteraction
    from ._runLARIS import runLARIS

    say = print if verbosity else (lambda *a, **k: None)
    decoy = permuteLRPairs(lr_df, data, random_seed=random_seed,
                           method='degree')
    n_real = int(decoy['is_real'].sum())
    say(f"  - decoy database: {len(decoy):,} rewired pairs "
        f"({n_real:,} coincide with real pairs and will be excluded)")

    prep = dict(prepare_kwargs or {})
    runk = dict(run_kwargs or {})
    lr_d = prepareLRInteraction(data, decoy[['ligand', 'receptor']], **prep)
    out = runLARIS(lr_d, data, background=background, **runk)
    decoy_res = out[1] if isinstance(out, tuple) else out
    real_names = set(decoy.loc[decoy['is_real'], 'interaction_name'])
    decoy_res = decoy_res[
        ~decoy_res['interaction_name'].isin(real_names)].copy()
    say(f"  - decoy rows tested: {len(decoy_res):,}")

    q = decoyFDR(target_res, decoy_res)
    return q, decoy_res


def decoyReport(
    target_res: pd.DataFrame,
    decoy_res: pd.DataFrame,
    thresholds=(0.05, 0.01),
    p_col: str = 'p_value',
    fdr_col: str = 'p_value_fdr',
    verbosity: int = 1,
) -> dict:
    """Dataset-level summary of what the database contributes over chance.

    ``p_value`` answers: *is this pair's expression arranged specifically
    with respect to these two cell types, beyond expression-matched
    chance?* It does not answer *does this ligand bind this receptor
    here?* - the pairing is asserted by the database, not measured from
    the data. A degree-preserving rewiring of the database, scored
    identically, quantifies the difference: whatever it recovers is what
    chance pairing alone achieves on this dataset.

    This returns (and by default prints) that comparison at the dataset
    level, which is the scale at which it is informative - the per-row
    q-value from :func:`decoyFDR` is close to constant within a dataset,
    so a single figure carries nearly all of the signal.

    Parameters
    ----------
    target_res, decoy_res : pandas.DataFrame
        ``runLARIS`` results for the real and decoy databases, same
        settings, ideally the same background. See
        :func:`computeDecoyFDR`, which produces ``decoy_res``.
    thresholds : tuple of float, default=(0.05, 0.01)
        FDR thresholds to report.
    p_col : str, default='p_value'
        Column holding the raw p-values (for the q-values).
    fdr_col : str, default='p_value_fdr'
        Column holding each run's own FDR (for the call counts).
    verbosity : int, default=1
        1 prints the report; 0 returns it silently.

    Returns
    -------
    dict
        ``per_threshold`` (target calls, decoy calls, and the estimated
        pairing-FDR at each threshold), ``q_min``, and ``n_rows``.

    Examples
    --------
    >>> q, dec = la.tl.computeDecoyFDR(adata, lr_df, res, background=bg,
    ...                                prepare_kwargs=..., run_kwargs=...)
    >>> rep = la.tl.decoyReport(res, dec)
    """
    n_t = int(target_res[p_col].notna().sum())
    n_d = int(decoy_res[p_col].notna().sum())
    if n_t == 0 or n_d == 0:
        raise ValueError("target_res and decoy_res must both contain "
                         "finite p-values.")
    out = {"n_rows": {"target": n_t, "decoy": n_d}, "per_threshold": {}}
    for t in thresholds:
        a = int((target_res[fdr_col] < t).sum())
        b = int((decoy_res[fdr_col] < t).sum())
        est = ((b + 1) / n_d) / max(a / n_t, 1e-12)
        out["per_threshold"][t] = {
            "target_calls": a, "decoy_calls": b,
            "pairing_fdr": round(float(min(est, 1.0)), 3)}
    q = decoyFDR(target_res, decoy_res, p_col=p_col)
    out["q_min"] = None if q.isna().all() else round(float(q.min()), 3)

    if verbosity:
        print("\n" + "=" * 66)
        print("TARGET-DECOY REPORT — what the database contributes")
        print("=" * 66)
        print("  The p-value tests how the pair's expression is ARRANGED "
              "with respect\n  to the cell types. The PAIRING itself comes "
              "from the database, not\n  from the data. A rewired database "
              "scored the same way shows how much\n  of the result chance "
              "pairing alone reproduces here.\n")
        for t, d in out["per_threshold"].items():
            print(f"  FDR < {t:<5}  real database {d['target_calls']:>7,} calls"
                  f"   |  rewired {d['decoy_calls']:>7,}"
                  f"   |  estimated pairing-FDR {d['pairing_fdr']:.2f}")
        print(f"\n  Best attainable pairing-FDR on this dataset: {out['q_min']}")
        print("  Read as: of the calls at this threshold, roughly this "
              "fraction are\n  matched by a database whose pairings are "
              "random. High values do not\n  mean the interactions are "
              "absent - they mean the DATA cannot\n  distinguish this "
              "pairing from a comparable one, so the evidence for\n  the "
              "pairing rests on the database. See tutorial 07.")
        print("=" * 66 + "\n")
    return out
