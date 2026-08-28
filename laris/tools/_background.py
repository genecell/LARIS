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
            adata, lr_df, n_pool=n_pool, n_matched_genes=n_matched_genes,
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
        db_pairs=db_pairs,
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
    verbosity: int = 1,
) -> pd.Series:
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
            exceed, anypos = _rust.assembly_counts(
                M, np.ascontiguousarray(spec_s),
                np.ascontiguousarray(spec_r),
                _cat['flat'], _cat['row_gene'], _cat['col_gene'],
                _cat['dw'], _cat['offsets'], pair_ids,
                gdf['interaction_score'].to_numpy(np.float64))
            sizes = (_cat['offsets'][pair_ids + 1]
                     - _cat['offsets'][pair_ids])
            p_grp = np.where((sizes == 0) | (anypos == 0), 1.0,
                             (exceed + 1) / (sizes + 1))
            for ix, pv in zip(gdf.index, p_grp):
                p_vals[pos_of[ix]] = pv
            continue
        for row in gdf.itertuples():
            ii, jj, dw_flat, flat, rowmap, colmap = _delta_block(
                row.ligand, row.receptor)
            if len(flat) == 0:
                p_vals[pos_of[row.Index]] = 1.0
                continue
            null = (spec_s[ii][rowmap] * spec_r[jj][colmap]
                    * dw_flat * M[flat])
            if not np.any(null > 0):
                p_vals[pos_of[row.Index]] = 1.0
                continue
            exceed = int(np.count_nonzero(null >= row.interaction_score))
            p_vals[pos_of[row.Index]] = (exceed + 1) / (len(flat) + 1)

    return pd.Series(p_vals, index=res_laris.index)


# ---------------------------------------------------------------------------
# Cytome-source support
# ---------------------------------------------------------------------------

def _is_cytome_source(obj) -> bool:
    from pathlib import Path as _Path
    if isinstance(obj, (str, _Path)):
        return str(obj).endswith('.cytome')
    return type(obj).__module__.startswith('cytome')


def _prepare_background_from_cytome(
    source, lr_df, n_pool, n_matched_genes, number_nearest_neighbors,
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
        needed = pd.Index(all_genes[pool_pos]).union(lr_genes)

        say(f"  - loading {len(needed):,} genes (pool + LR) from cytome...")
        sub = readCytome(ds, genes=list(needed))
    finally:
        if opened_here:
            ds.close()

    # the subset AnnData has obs/obsm from the file; run the standard
    # builder with a pool that covers (almost) the whole loaded gene set
    return prepareLRBackground(
        sub, lr_df, n_pool=len(pool_pos), n_matched_genes=n_matched_genes,
        number_nearest_neighbors=number_nearest_neighbors,
        use_rep_spatial=use_rep_spatial, use_rep_gsp=use_rep_gsp,
        n_nearest_neighbors_gsp=n_nearest_neighbors_gsp,
        sigma_gsp=sigma_gsp, section_key=section_key,
        block_size=block_size, verbosity=verbosity,
        _pool_genes=list(all_genes[pool_pos]))
