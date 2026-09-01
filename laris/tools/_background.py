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

from dataclasses import dataclass, field
from typing import Optional, Dict

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import KDTree, kneighbors_graph

from .._compat import _UNSET, resolve_data_arg


def _rust_kernels():
    """Return LARIS's compiled Rust kernels (``laris._laris``), or None.

    The extension ships inside the wheel; the NumPy path stays as a
    fallback for source installs without a Rust toolchain and for
    debugging. Both compute the identical quantities (parity-tested).
    Set ``LARIS_NO_RUST=1`` to force the NumPy path.
    """
    import os
    if os.environ.get('LARIS_NO_RUST'):
        return None
    try:
        from .. import _laris
        return _laris
    except ImportError:
        return None


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
        idx = _matched_sets(feats[query_pos], feats[pool], k)
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


def _matched_sets(query_feats: np.ndarray, pool_feats: np.ndarray,
                  k: int) -> np.ndarray:
    """k nearest pool genes per query gene in (mean, variance) space.

    Features are standardized on the pool's statistics so neither axis
    dominates. Same KDTree matcher as the rest of the ecosystem, restricted
    to the bounded candidate pool.
    """
    mu = pool_feats.mean(axis=0)
    sd = pool_feats.std(axis=0)
    sd[sd == 0] = 1.0
    kdt = KDTree((pool_feats - mu) / sd)
    k = min(k, pool_feats.shape[0])
    _, idx = kdt.query((query_feats - mu) / sd, k=k)
    return idx


# ---------------------------------------------------------------------------
# Gram tables
# ---------------------------------------------------------------------------

def _edge_gram(P: np.ndarray, Q: np.ndarray, graph: sp.spmatrix,
               edge_chunk: int = 200_000) -> np.ndarray:
    """Edge Gram of the MASKED pseudo-pair profiles.

    ``prepareLRInteraction`` masks each pair score to cells where the
    ligand or the receptor has nonzero raw counts. With ``Q = P * (raw ==
    0)`` the masked profile is the difference of two separable products,
    ``v_ij = P_i*P_j - Q_i*Q_j`` (on masked-out cells P equals Q, so the
    difference vanishes there and equals P_i*P_j elsewhere). Any quadratic
    form over v therefore expands into four per-gene edge Grams:

        v(u) v(c) =  [P(u)P(c)]_i [P(u)P(c)]_j  -  [P(u)Q(c)]_i [P(u)Q(c)]_j
                   - [Q(u)P(c)]_i [Q(u)P(c)]_j  +  [Q(u)Q(c)]_i [Q(u)Q(c)]_j

    each term separable per gene, hence a (U x nnz) @ (nnz x U) product.
    Chunked over edges so the U x chunk intermediates stay bounded.

    A sparser identity exists and was implemented, measured and rejected.
    Splitting P = Q + R (R nonzero only on raw-detected cells) and lifting
    A=Q(u)Q(c), B=Q(u)R(c), C=R(u)Q(c), D=R(u)R(c) makes every pure-A,
    pure-B and pure-C term cancel, leaving

        G = <A,D>+<D,A> + <B,C>+<C,B> + <B,D>+<D,B> + <C,D>+<D,C> + <D,D>

    in which every term carries an R factor. R is ~7x sparser than P
    (0.048 vs 0.344 on tonsil), giving roughly 400x fewer multiply-adds.
    A threaded Rust kernel over R's nonzeros nevertheless benchmarked
    **2x SLOWER** than the four dense products (172 s vs 85 s on tonsil's
    W graph): the dominant <A,D> term is a scatter over all U rows per
    nonzero, which is memory-bound and cache-hostile, while the dense
    path runs at near-peak BLAS throughput. Same lesson as the
    co-localization pass. Float32 sgemm was also tried: 1.44x faster at
    1.8e-7 relative error, rejected because the tail counts are exact
    comparisons. The dense form below is the fastest correct version we
    have.
    """
    g = sp.coo_matrix(graph)
    U = P.shape[0]

    G = np.zeros((U, U), dtype=np.float64)
    for start in range(0, g.nnz, edge_chunk):
        stop = min(start + edge_chunk, g.nnz)
        rows = g.row[start:stop]
        cols = g.col[start:stop]
        w = g.data[start:stop]
        Epp = P[:, rows] * P[:, cols]
        Epq = P[:, rows] * Q[:, cols]
        Eqp = Q[:, rows] * P[:, cols]
        Eqq = Q[:, rows] * Q[:, cols]
        G += (Epp * w) @ Epp.T
        G -= (Epq * w) @ Epq.T
        G -= (Eqp * w) @ Eqp.T
        G += (Eqq * w) @ Eqq.T
    return G


# ---------------------------------------------------------------------------
# The background object
# ---------------------------------------------------------------------------

@dataclass
class LRBackground:
    """Product of :func:`prepareLRBackground`; consumed by ``runLARIS``.

    ``gene_index`` lists the U genes covered by the tables (candidate pool
    plus every LR gene, so real pairs resolve against the same tables).
    ``P`` is their diffused matrix (U x cells, dense float32) using the
    same kernel graph as ``prepareLRInteraction``. The Gram tables are
    over the *gsp* graph that ``runLARIS`` builds for spatial specificity.
    """
    gene_index: pd.Index                     # U gene names
    P: np.ndarray                            # (U, n_cells) diffused, float32
    Q: np.ndarray                            # P masked to raw-zero cells
    G_W: np.ndarray                          # edge Gram, observed gsp graph
    G_W2: np.ndarray                         # edge Gram, W'W of the gsp graph
    G_dot: np.ndarray                        # P @ P.T      (=> m1 * n)
    G_sq: np.ndarray                         # P^2 @ (P^2).T (=> m2 * n, ||v||^2)
    R: float                                 # analytic E[sum row w'^2] * n
    n_cells: int
    matched_ligand: Dict[str, np.ndarray] = field(default_factory=dict)
    matched_receptor: Dict[str, np.ndarray] = field(default_factory=dict)
    db_pairs: frozenset = frozenset()        # gene-name pairs in the database
    # per LR gene: fraction of its matched set with a lower mean (1.0 =
    # saturated; every pseudo-pair is weaker than the real one)
    match_frac_below: Dict[str, float] = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    def positions(self, genes) -> np.ndarray:
        pos = self.gene_index.get_indexer(genes)
        if (pos < 0).any():
            missing = [g for g, p in zip(genes, pos) if p < 0]
            raise KeyError(f"genes not in background: {missing[:5]}")
        return pos

    @property
    def support_per_pair(self) -> int:
        k = self.params.get('n_matched_genes', 0)
        return k * k


def prepareLRBackground(
    data=_UNSET,
    lr_df: Optional[pd.DataFrame] = None,
    n_pool: int = 4000,
    augment_pool: bool = True,
    n_matched_genes: int = 100,
    number_nearest_neighbors: int = 20,
    use_rep_spatial: str = 'X_spatial',
    use_rep_gsp: Optional[str] = None,
    n_nearest_neighbors_gsp: int = 20,
    sigma_gsp='adaptive',
    layer: Optional[str] = None,
    section_key: Optional[str] = None,
    block_size: int = 512,
    verbosity: int = 1,
    adata=_UNSET,
    _pool_genes=None,
) -> LRBackground:
    """Build the matched-gene background for factorized null p-values.

    One background serves every ``runLARIS`` call on the same cells: the
    Gram tables depend only on the spatial graphs and the gene pool, not on
    the cell-type grouping or on ``mu`` / ``spatial_weight``, so parameter
    sweeps and different ``groupby`` choices reuse it unchanged.

    Parameters
    ----------
    data : AnnData
        The same expression object passed to ``prepareLRInteraction``
        (raw counts; ``layer`` selects a layer).
    lr_df : DataFrame
        The LR database subset used for the analysis ('ligand'/'receptor'
        columns). Needed to know which genes require matched sets and which
        gene combinations are real database pairs (excluded from nulls).
    n_pool : int
        Candidate pool size U. The Gram tables are U x U; the default keeps
        them ~130 MB each in float64.
    augment_pool : bool, default=True
        Grow the candidate pool until no ligand or receptor has a matched
        set lying entirely below it in expression. The quantile grid is
        rank-uniform and so starves the extreme top of the abundance
        range; a gene left there has no peers, every pseudo-pair built
        from it is weaker than the real pair, and it beats its own null
        arithmetically. Genes that fail contribute their own nearest
        neighbours from the whole transcriptome, and matching is
        recomputed (see ``match_frac_below``). Set False to reproduce the
        v0.12.0 pool exactly.
    n_matched_genes : int
        Matched genes per side (k). The factorized support per pair is
        k**2 and the exact p-value floor is 1/(k**2+1) before exclusions.
    number_nearest_neighbors, use_rep_spatial, layer, section_key
        Must match the ``prepareLRInteraction`` call so the diffusion is
        identical.
    use_rep_gsp, n_nearest_neighbors_gsp, sigma_gsp
        Must match ``runLARIS``'s ``use_rep`` / ``n_nearest_neighbors`` /
        ``sigma`` (the spatial-specificity graph). ``use_rep_gsp`` defaults
        to ``use_rep_spatial``.
    """
    from . import _utils

    adata = resolve_data_arg(data, 'prepareLRBackground', adata=adata)
    if lr_df is None:
        raise TypeError("prepareLRBackground() requires lr_df")
    if use_rep_gsp is None:
        use_rep_gsp = use_rep_spatial

    say = print if verbosity else (lambda *a, **k: None)

    if _is_cytome_source(adata):
        # Pool selection needs per-gene mean/variance over ALL genes, which
        # is computed in gene blocks (bounded memory); only the selected
        # pool + LR genes are then loaded, so the memory profile matches
        # the streaming pipeline.
        return _prepare_background_from_cytome(
            adata, lr_df, n_pool=n_pool, augment_pool=augment_pool,
            n_matched_genes=n_matched_genes,
            number_nearest_neighbors=number_nearest_neighbors,
            use_rep_spatial=use_rep_spatial, use_rep_gsp=use_rep_gsp,
            n_nearest_neighbors_gsp=n_nearest_neighbors_gsp,
            sigma_gsp=sigma_gsp, section_key=section_key,
            block_size=block_size, verbosity=verbosity)

    X = adata.layers[layer] if layer is not None else adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocsc()
    n_cells = adata.n_obs

    # ---- gene features and pool -----------------------------------------
    m = np.asarray(X.mean(axis=0)).ravel()
    m2 = np.asarray(X.power(2).mean(axis=0)).ravel()
    v = m2 - m ** 2
    var_names = pd.Index(adata.var_names)

    lr_genes = pd.Index(pd.unique(np.concatenate([
        np.asarray(lr_df['ligand']), np.asarray(lr_df['receptor'])])))
    lr_genes = lr_genes[lr_genes.isin(var_names)]
    lig_genes = pd.Index(pd.unique(np.asarray(lr_df['ligand'])))
    lig_genes = lig_genes[lig_genes.isin(var_names)]
    rec_genes = pd.Index(pd.unique(np.asarray(lr_df['receptor'])))
    rec_genes = rec_genes[rec_genes.isin(var_names)]

    if _pool_genes is not None:
        # pool decided upstream (cytome path selects it from full-matrix
        # stats before loading the gene subset)
        pool_pos = var_names.get_indexer(pd.Index(_pool_genes))
        pool_pos = np.sort(pool_pos[pool_pos >= 0])
    else:
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
    say(f"  - candidate pool: {len(pool_pos):,} genes "
        f"(quantile grid over mean/variance)")

    # tables cover pool genes + every LR gene (so real pairs resolve too)
    lr_pos = var_names.get_indexer(lr_genes)
    gene_pos = np.unique(np.concatenate([pool_pos, lr_pos]))
    gene_index = var_names[gene_pos]
    U = len(gene_pos)
    pos_in_tables = pd.Index(gene_index).get_indexer

    # matched sets: k nearest *pool* genes per LR gene (kNN-within-pool)
    feats = np.column_stack([m, v])
    # positions of the pool genes inside the table gene set
    pool_tab_pos = pos_in_tables(var_names[pool_pos])
    lig_idx = _matched_sets(feats[var_names.get_indexer(lig_genes)],
                            feats[pool_pos], n_matched_genes)
    rec_idx = _matched_sets(feats[var_names.get_indexer(rec_genes)],
                            feats[pool_pos], n_matched_genes)
    matched_ligand = {g: pool_tab_pos[lig_idx[i]]
                      for i, g in enumerate(lig_genes)}
    matched_receptor = {g: pool_tab_pos[rec_idx[i]]
                        for i, g in enumerate(rec_genes)}
    # Matchability: fraction of a gene's matched set with a LOWER mean
    # than the gene itself. 1.0 means every match is weaker - the null
    # for any pair containing this gene is systematically easy, and its
    # p-values overstate. With the n_extreme pool repair this should be
    # rare; the value is reported so the residual cases are visible.
    pool_means = m[pool_pos]
    match_frac_below = {}
    for genes_, idx_ in ((lig_genes, lig_idx), (rec_genes, rec_idx)):
        gm = m[var_names.get_indexer(genes_)]
        for i, g in enumerate(genes_):
            match_frac_below[g] = float((pool_means[idx_[i]] < gm[i]).mean())

    # ---- diffusion of the table genes (same kernel as prepare) ----------
    say(f"  - diffusing {U:,} genes through the "
        f"{number_nearest_neighbors}-NN kernel graph...")
    coords = np.asarray(adata.obsm[use_rep_spatial])
    sections = _utils._resolve_sections(adata, section_key, n_cells)
    # identical graph construction to prepareLRInteraction, so real pairs'
    # profiles reconstructed from P/Q match the pipeline's columns
    knn = _utils._sectioned_kneighbors_graph(
        coords, n_neighbors=number_nearest_neighbors, sections=sections,
        mode='distance', include_self=True)
    knn = sp.csr_matrix(_utils._apply_knn_kernel(knn, sigma='adaptive'))
    genexcell = sp.csr_matrix(X[:, gene_pos]).T
    P = np.empty((U, n_cells), dtype=np.float32)
    for start in range(0, n_cells, block_size * 32):
        stop = min(start + block_size * 32, n_cells)
        P[:, start:stop] = np.asarray(
            (genexcell @ knn[start:stop, :].T).todense())
    # prepareLRInteraction keeps a pair's score only in cells where ligand
    # OR receptor raw counts are nonzero. Q carries each gene's diffused
    # values on its raw-zero cells; the masked pseudo-pair profile is then
    # P_i*P_j - Q_i*Q_j (see _edge_gram), which keeps every Gram identity.
    raw_nonzero = np.asarray((X[:, gene_pos] != 0).todense()).T
    Q = np.where(raw_nonzero, 0.0, P).astype(np.float32)
    del raw_nonzero

    # ---- the gsp graph and its Gram tables ------------------------------
    say("  - building spatial-specificity graph and Gram tables...")
    # identical to the graph runLARIS builds for spatial specificity
    W = _utils._build_adjacency_matrix(
        adata, use_rep=use_rep_gsp,
        n_nearest_neighbors=n_nearest_neighbors_gsp,
        sigma=sigma_gsp, sections=sections)

    G_W = _edge_gram(P, Q, W)
    G_W2 = _edge_gram(P, Q, (W.T @ W).tocoo())
    P64 = P.astype(np.float64)
    Q64 = Q.astype(np.float64)
    G_dot = P64 @ P64.T - Q64 @ Q64.T
    Psq = P64 ** 2
    Qsq = Q64 ** 2
    G_sq = Psq @ Psq.T - Qsq @ Qsq.T
    del P64, Q64, Psq, Qsq

    R = n_cells * _expected_row_sq_sum(W.data, n_nearest_neighbors_gsp)

    db_pairs = frozenset(zip(np.asarray(lr_df['ligand']),
                             np.asarray(lr_df['receptor'])))

    say(f"  - done: {U:,} genes, support per pair "
        f"{n_matched_genes ** 2:,} (floor "
        f"{1.0 / (n_matched_genes ** 2 + 1):.2e})")

    return LRBackground(
        gene_index=pd.Index(gene_index), P=P, Q=Q,
        G_W=G_W, G_W2=G_W2, G_dot=G_dot, G_sq=G_sq,
        R=R, n_cells=n_cells,
        matched_ligand=matched_ligand, matched_receptor=matched_receptor,
        db_pairs=db_pairs, match_frac_below=match_frac_below,
        params=dict(n_pool=n_pool, n_matched_genes=n_matched_genes,
                    number_nearest_neighbors=number_nearest_neighbors,
                    use_rep_spatial=use_rep_spatial, use_rep_gsp=use_rep_gsp,
                    n_nearest_neighbors_gsp=n_nearest_neighbors_gsp,
                    sigma_gsp=sigma_gsp, section_key=section_key),
    )


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

    Many pseudo-pairs score exactly zero, because their matched genes are
    not co-detected in the two groups. A zero can never exceed a positive
    observed score, so it enlarges the denominator without contributing
    resolution: a row whose support is 9,900 zeros and 100 positive
    entries can still report p = 1e-4 while the null can only really
    resolve 1e-2. The count of positive entries is therefore the null's
    *effective support*, and it is returned alongside the p-value so the
    caller can see it. ``min_null_support`` sets p = 1.0 for rows whose
    effective support falls below it; the default of 0 leaves every
    p-value exactly as v0.12.0 computed it.

    Returns
    -------
    pandas.DataFrame
        Indexed like ``res_laris``, with columns ``p_value`` and
        ``null_support`` (the number of pseudo-pairs scoring above zero).
    """
    say = print if verbosity else (lambda *a, **k: None)
    n_groups = len(groups_order)
    group_col = {g: i for i, g in enumerate(groups_order)}
    U = len(bg.gene_index)
    P = bg.P

    # --- spec matrix aligned to table genes (genes x groups) -------------
    spec_tab = spec_ext.reindex(bg.gene_index).fillna(0.0)
    spec_tab = spec_tab.loc[:, list(groups_order)].to_numpy(dtype=np.float64)

    # --- detection-fraction Grams per group -------------------------------
    say("  - detection-fraction tables per cell type...")
    # masked pseudo-pair is nonzero iff both diffused values are nonzero
    # AND the raw-OR mask holds: D_i D_j - E_i E_j with E = D on raw-zero
    # cells (same difference structure as the score itself).
    D = (P > 0).astype(np.float32)
    E = (bg.Q > 0).astype(np.float32)
    frac = {}
    for g in groups_order:
        cells = np.flatnonzero(group_labels == g)
        if len(cells) == 0:
            frac[g] = np.zeros((U, U), dtype=np.float32)
        else:
            Dg = D[:, cells]
            Eg = E[:, cells]
            frac[g] = (Dg @ Dg.T - Eg @ Eg.T) / np.float32(len(cells))
    del D, E

    # --- co-localization: pass A accumulates the row-sum of cos^2 ---------
    ctXct_cell = ctc_internals['ctXct_cell']         # (n_ctc, n_cells) sparse
    ctc_names = list(ctc_internals['ctc_names'])
    y_norms = ctc_internals['y_norms']
    iqr = ctc_internals['iqr']                       # per ctc column, real-fitted
    v_norm = np.sqrt(bg.G_sq)                        # ||v_ij||, (U, U)

    say(f"  - co-localization pass A over {len(ctc_names)} cell-type pairs...")
    S = np.zeros((U, U), dtype=np.float64)

    # Masked numerator as two dense BLAS matmuls. A sparse decomposition
    # via Pnz = P - Q has ~20x fewer flops but benchmarks 10x SLOWER on
    # real data: scipy's sparse kernels are single-threaded while the
    # dense products use threaded BLAS (measured on tonsil, U=4,718,
    # 5,695 cells, Pnz density 4.8%: dense 0.70 s/column, sparse chain
    # 7.4 s/column). Dense wins until a threaded sparse backend exists;
    # that backend (Rust/MKL) is the natural home for a revisit.
    Q = bg.Q
    _rust = _rust_kernels()
    if _rust is not None:
        # threaded sparse decomposition (see src/lib.rs): flops scale
        # with the raw data's sparsity; measured 2.1x over threaded BLAS
        # at tonsil's 4.8% density, growing as density falls
        _Pnz = sp.csr_matrix(P - Q)
        _Pnz_csc = _Pnz.tocsc()
        _Pt = np.ascontiguousarray(P.T)
        _rust_args = (_Pt,
                      _Pnz.indptr.astype(np.int64),
                      _Pnz.indices.astype(np.int32),
                      _Pnz.data.astype(np.float32),
                      _Pnz_csc.indptr.astype(np.int64),
                      _Pnz_csc.indices.astype(np.int32),
                      _Pnz_csc.data.astype(np.float32))

    def _ctc_cos(idx):
        y = np.asarray(ctXct_cell[idx].todense()).ravel().astype(np.float32)
        if _rust is not None:
            N = _rust.ctc_cos_numerator(*_rust_args, y)
        else:
            N = (P * y) @ P.T - (Q * y) @ Q.T
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
        c = _ctc_cos(idx)
        S += c * c
        done.add(idx)
        if mirror is not None and mirror != idx and mirror not in done:
            S += c.T * c.T
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

    # concatenated per-pair arrays for the fused Rust assembly kernel
    _cat = None
    _pair_index = {}

    def _pair_id(lig, rec):
        return _pair_index[(lig, rec)]

    if _rust is not None:
        uniq = list(dict.fromkeys(zip(res_laris['ligand'],
                                      res_laris['receptor'])))
        flats, rgs, cgs, dws, offs = [], [], [], [], [0]
        for lig, rec in uniq:
            ii, jj, dw_flat, flat, rowmap, colmap = _delta_block(lig, rec)
            _pair_index[(lig, rec)] = len(offs) - 1
            flats.append(flat)
            rgs.append(ii[rowmap].astype(np.int32))
            cgs.append(jj[colmap].astype(np.int32))
            dws.append(dw_flat)
            offs.append(offs[-1] + len(flat))
        _cat = dict(
            flat=np.concatenate(flats).astype(np.int64)
            if flats else np.empty(0, np.int64),
            row_gene=np.concatenate(rgs).astype(np.int32)
            if rgs else np.empty(0, np.int32),
            col_gene=np.concatenate(cgs).astype(np.int32)
            if cgs else np.empty(0, np.int32),
            dw=np.concatenate(dws).astype(np.float64)
            if dws else np.empty(0, np.float64),
            offsets=np.asarray(offs, dtype=np.int64),
        )

    # --- pass B: one ctc at a time, all its rows scored -------------------
    p_vals = np.full(len(res_laris), np.nan)
    n_pos = np.zeros(len(res_laris), dtype=np.int64)
    pos_of = {ix: i for i, ix in enumerate(res_laris.index)}
    grouped = res_laris.groupby(['sender', 'receiver'], observed=True)
    lam_mu = mu_celltype

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
        if _rust is not None:
            pair_ids = np.fromiter(
                (_pair_id(lig, rec) for lig, rec in
                 zip(gdf['ligand'], gdf['receptor'])),
                dtype=np.int64, count=len(gdf))
            exceed, npos_grp = _rust.assembly_counts(
                M, np.ascontiguousarray(spec_s),
                np.ascontiguousarray(spec_r),
                _cat['flat'], _cat['row_gene'], _cat['col_gene'],
                _cat['dw'], _cat['offsets'], pair_ids,
                gdf['interaction_score'].to_numpy(np.float64))
            sizes = (_cat['offsets'][pair_ids + 1]
                     - _cat['offsets'][pair_ids])
            degenerate = ((sizes == 0) | (npos_grp == 0)
                          | (npos_grp < min_null_support))
            p_grp = np.where(degenerate, 1.0, (exceed + 1) / (sizes + 1))
            for ix, pv, nz in zip(gdf.index, p_grp, npos_grp):
                p_vals[pos_of[ix]] = pv
                n_pos[pos_of[ix]] = nz
            continue
        for row in gdf.itertuples():
            ii, jj, dw_flat, flat, rowmap, colmap = _delta_block(
                row.ligand, row.receptor)
            if len(flat) == 0:
                p_vals[pos_of[row.Index]] = 1.0
                continue
            null = (spec_s[ii][rowmap] * spec_r[jj][colmap]
                    * dw_flat * M[flat])
            npos_row = int(np.count_nonzero(null > 0))
            n_pos[pos_of[row.Index]] = npos_row
            if npos_row == 0 or npos_row < min_null_support:
                p_vals[pos_of[row.Index]] = 1.0
                continue
            exceed = int(np.count_nonzero(null >= row.interaction_score))
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

    n_tested = int(np.count_nonzero(n_pos > 0))
    if n_tested:
        thin = int(np.count_nonzero((n_pos > 0) & (n_pos < 100)))
        if thin > 0.05 * n_tested:
            warnings.warn(
                f"{thin:,} of {n_tested:,} tested interactions "
                f"({100 * thin / n_tested:.1f}%) have a null with fewer "
                "than 100 non-zero pseudo-pairs. Their p-values are "
                "reported against a denominator of "
                f"{int(np.median(n_pos[n_pos > 0])):,} but can only "
                "resolve about 1/(effective support). Inspect the "
                "'null_support' column, and consider "
                "min_null_support= to drop them.",
                UserWarning, stacklevel=2)
    return pd.DataFrame({'p_value': p_vals, 'null_support': n_pos},
                        index=res_laris.index)


# ---------------------------------------------------------------------------
# Cytome-source support
# ---------------------------------------------------------------------------

def _is_cytome_source(obj) -> bool:
    from pathlib import Path as _Path
    if isinstance(obj, (str, _Path)):
        return str(obj).endswith('.cytome')
    return type(obj).__module__.startswith('cytome')


def _prepare_background_from_cytome(
    source, lr_df, n_pool, augment_pool, n_matched_genes,
    number_nearest_neighbors,
    use_rep_spatial, use_rep_gsp, n_nearest_neighbors_gsp, sigma_gsp,
    section_key, block_size, verbosity,
) -> "LRBackground":
    """Cytome path: gene-blocked stats pass, then a subset load.

    Two disk passes with bounded memory: (1) per-gene mean/variance over
    all genes, read in blocks via ``readCytome(genes=block)``, to select
    the candidate pool; (2) one load of the pool + LR genes into an
    in-memory AnnData, after which the standard builder runs unchanged.
    The tables and matched sets are identical to what the in-memory path
    would produce from ``to_anndata`` of the same file (tested).
    """
    from ..preprocessing._io import _open_cytome, readCytome

    say = print if verbosity else (lambda *a, **k: None)
    ds, opened_here = _open_cytome(source)
    try:
        genes_df = ds.genes.to_pandas()
        # same resolution as readCytome(gene_name_column='auto'): symbol
        # where present, gene_id filling the gaps
        symbols = genes_df.get('symbol')
        gene_ids = genes_df.get('gene_id')
        if symbols is None and gene_ids is None:
            raise KeyError(
                "cytome genes table has neither 'symbol' nor 'gene_id'")
        if symbols is None:
            symbols = gene_ids.astype(str)
        elif gene_ids is not None:
            symbols = symbols.where(symbols.notna(), gene_ids).astype(str)
        else:
            symbols = symbols.astype(str)
        all_genes = pd.Index(symbols)

        say(f"  - cytome stats pass over {len(all_genes):,} genes "
            f"(blocked reads)...")
        means = np.empty(len(all_genes))
        variances = np.empty(len(all_genes))
        blk = 2000
        for start in range(0, len(all_genes), blk):
            sub = readCytome(ds, genes=list(all_genes[start:start + blk]))
            Xb = sub.X if sp.issparse(sub.X) else sp.csr_matrix(sub.X)
            pos = all_genes.get_indexer(sub.var_names)
            mb = np.asarray(Xb.mean(axis=0)).ravel()
            m2b = np.asarray(Xb.power(2).mean(axis=0)).ravel()
            means[pos] = mb
            variances[pos] = m2b - mb ** 2

        pool_pos = _quantile_grid_pool(means, variances, n_pool)
        lr_genes = pd.Index(pd.unique(np.concatenate([
            np.asarray(lr_df['ligand']), np.asarray(lr_df['receptor'])])))
        lr_genes = lr_genes[lr_genes.isin(all_genes)]
        if augment_pool:
            _q = pd.Index(all_genes).get_indexer(lr_genes)
            _q = _q[_q >= 0]
            _before = len(pool_pos)
            pool_pos = _augment_pool_for_saturation(
                np.column_stack([means, variances]), pool_pos, _q,
                n_matched_genes)
            if len(pool_pos) > _before:
                say(f"  - pool grown {_before:,} -> {len(pool_pos):,} genes "
                    "so every LR gene has matched peers above it")
        needed = pd.Index(all_genes[pool_pos]).union(lr_genes)

        say(f"  - loading {len(needed):,} genes (pool + LR) from cytome...")
        sub = readCytome(ds, genes=list(needed))
    finally:
        if opened_here:
            ds.close()

    # the subset AnnData has obs/obsm from the file; run the standard
    # builder with a pool that covers (almost) the whole loaded gene set
    return prepareLRBackground(
        sub, lr_df, n_pool=len(pool_pos), augment_pool=False,
        n_matched_genes=n_matched_genes,
        number_nearest_neighbors=number_nearest_neighbors,
        use_rep_spatial=use_rep_spatial, use_rep_gsp=use_rep_gsp,
        n_nearest_neighbors_gsp=n_nearest_neighbors_gsp,
        sigma_gsp=sigma_gsp, section_key=section_key,
        block_size=block_size, verbosity=verbosity,
        _pool_genes=list(all_genes[pool_pos]))


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
    rewiring of the database scores almost as well (tonsil: 947 vs 1,195
    calls at FDR<0.05), because for marker-structured pairs the pairing
    is unidentifiable from expression and coordinates alone. See
    tutorial 07 and the discussion record.

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
