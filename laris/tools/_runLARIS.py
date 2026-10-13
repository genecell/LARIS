"""The LARIS algorithm: spatial specificity and cell-type interaction scores."""

import warnings
from typing import Optional, Union, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize

from . import _utils
from . import _background as _background
from .._compat import _UNSET, resolve_data_arg
from ..preprocessing._io import (
    _ensure_expression_anndata,
    _ensure_lr_anndata,
    _is_cytome_source,
)


def _check_background_graph(background, use_rep, n_nearest_neighbors, sigma,
                            section_key, diffusion=None) -> None:
    """The background must be built on the graphs the observed scores use,
    so the check fails here, before any work:

    * the spatial-specificity graph runLARIS builds (``use_rep``,
      ``n_nearest_neighbors``, ``sigma``, ``section_key``): the
      pseudo-pairs' spatial scores come from its Gram tables, the observed
      pairs' from runLARIS's own;
    * the diffusion graph of ``prepareLRInteraction``
      (``number_nearest_neighbors``, ``use_rep_spatial``, ``sigma``,
      ``section_key``), which LR objects record from v0.15.0
      (``diffusion``). Older LR objects carry no record; a mismatch there
      is still caught later, by scoring each observed pair through the
      background tables.

    A mismatch makes every P value compare unlike quantities."""
    p = getattr(background, 'params', None) or {}

    def same(a, b):
        if isinstance(a, str) or isinstance(b, str) or a is None or b is None:
            return a == b
        return bool(np.isclose(float(a), float(b), rtol=1e-12, atol=0.0))

    checks = (
        ('use_rep', use_rep, 'use_rep_gsp',
         p.get('use_rep_gsp') or p.get('use_rep_spatial')),
        ('n_nearest_neighbors', n_nearest_neighbors, 'n_nearest_neighbors_gsp',
         p.get('n_nearest_neighbors_gsp')),
        ('sigma', sigma, 'sigma_gsp', p.get('sigma_gsp')),
        ('section_key', section_key, 'section_key', p.get('section_key')),
    )
    bad = [(a, v, b, bv) for a, v, b, bv in checks
           if (b in p or (b == 'use_rep_gsp' and 'use_rep_spatial' in p))
           and not same(v, bv)]
    if bad:
        raise ValueError(
            "The background was built for a different spatial-specificity "
            "graph than runLARIS uses: "
            + "; ".join(f"runLARIS({a}={v!r}) vs "
                        f"prepareLRBackground({b}={bv!r})"
                        for a, v, b, bv in bad)
            + ". Pass matching values; otherwise the null's spatial scores "
            "come from another graph and every P value is wrong.")
    if not diffusion or 'number_nearest_neighbors' not in p:
        return
    # backgrounds before v0.15.0's ``sigma`` parameter always used 'adaptive'
    dchecks = (
        ('number_nearest_neighbors', diffusion.get('number_nearest_neighbors'),
         p.get('number_nearest_neighbors')),
        ('use_rep_spatial', diffusion.get('use_rep_spatial'),
         p.get('use_rep_spatial')),
        ('sigma', diffusion.get('sigma', 'adaptive'), p.get('sigma', 'adaptive')),
        ('section_key', diffusion.get('section_key'), p.get('section_key')),
    )
    dbad = [(k, a, b) for k, a, b in dchecks if not same(a, b)]
    if dbad:
        raise ValueError(
            "The background was built for a different diffusion graph than "
            "the LR scores: "
            + "; ".join(f"prepareLRInteraction({k}={a!r}) vs "
                        f"prepareLRBackground({k}={b!r})" for k, a, b in dbad)
            + ". Pass the same values to both; otherwise the null's profiles "
            "are diffused differently from the observed scores and every P "
            "value is wrong.")

def runLARIS(
    lr_data=_UNSET,
    data=_UNSET,
    use_rep: str = 'X_spatial',
    n_nearest_neighbors: int = 20,
    random_seed: int = 27,
    n_repeats: Optional[int] = None,   # deprecated: delta is now analytic
    mu: float = 0.25,
    sigma: Union[float, str] = 'adaptive',
    remove_lowly_expressed: bool = True,
    expressed_pct: float = 0.1,
    n_cells_expressed_threshold: Union[int, float] = 100,
    n_top_lr: int = 4000,
    # Cell Type & Statistical Testing Parameters
    by_celltype: bool = True,
    groupby: str = 'CellTypes',
    use_rep_spatial: str = 'X_spatial',
    number_nearest_neighbors: int = 20,
    mu_celltype: float = 100,
    sigma_celltype: Union[float, str] = 'adaptive',
    expressed_pct_celltype: float = 0.1,
    remove_lowly_expressed_celltype: bool = True,
    mask_threshold: float = 1e-6,
    calculate_pvalues: bool = True,
    layer_celltype: Optional[str] = None,
    n_neighbors_permutation: int = 30,
    n_permutations=_UNSET,
    chunk_size: int = 50000,
    prefilter_fdr: bool = True,
    prefilter_threshold: float = 0.0,
    score_threshold: float = 1e-6,
    spatial_weight: float = 3.0,
    use_conditional_pvalue: Optional[bool] = None,
    rescale: bool = True,
    cosg_backend: str = 'auto',
    specificity_reference: str = 'lr',
    section_key: Optional[str] = None,
    background=None,
    min_null_support: Union[int, str] = 'auto',
    sort_by: str = 'interaction_score',
    lr_adata=_UNSET,
    adata=_UNSET
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Identify spatially-specific ligand-receptor interactions using LARIS algorithm.

    LARIS (Ligand And Receptor Interaction in Spatial transcriptomics) identifies LR pairs
    that show spatial specificity by comparing observed spatial correlation patterns
    with randomized null distributions. When `by_celltype=True`, the function also
    computes cell type-specific interaction scores with optional statistical testing.

    This is the main analytical function of the LARIS package, providing:

    1. **Spatial Specificity Analysis**: Identifies LR pairs that show non-random
       spatial co-localization patterns (higher scores = stronger spatial organization)

    2. **Cell type-specific Scores**: Integrates spatial specificity with cell type
       expression specificity and spatial co-localization to identify which sender-
       receiver cell type pairs are communicating via which LR pairs

    3. **Statistical Testing**: Optional permutation-based P values with FDR correction
       to identify statistically significant interactions

    Parameters
    ----------
    lr_data : AnnData, str or cytome dataset
        LR interaction scores from `prepareLRInteraction()`: an AnnData, or
        an LR-score cytome (path or open dataset).

        Required contents:
        - `.X`: Diffused LR scores (cells × LR pairs)
        - `.var_names`: LR pair names ("ligand::receptor")
        - `.obsm[use_rep]`: Spatial coordinates or other representation

    data : AnnData, str or cytome dataset, optional
        The expression data the LR scores were computed from, with spatial
        information (an AnnData, or a cytome path or open dataset).
        **Required when `by_celltype=True`**.

        Required contents (when by_celltype=True):
        - `.obs[groupby]`: Cell type annotations
        - `.obsm[use_rep_spatial]`: Spatial coordinates
        - `.X` or `.layers[layer_celltype]`: Gene expression

    use_rep : str, default='X_spatial'
        Key in `lr_adata.obsm` for coordinates to use in spatial specificity analysis.
        Typically spatial coordinates, but could be other representations.

    n_nearest_neighbors : int, default=20
        Number of spatial neighbors for building the adjacency matrix in the
        spatial specificity analysis. Larger values capture broader spatial
        patterns. Note the physical area spanned by k neighbors differs
        between platforms (a k of 20 covers a much smaller area at
        8 um Visium HD than at Stereo-seq bin50); rankings are robust across
        k = 3-50, but top-hit identity shifts at very small k.
        (Changed from 10 to 20 in v0.10.0 to match the published analyses;
        pass n_nearest_neighbors=10 to reproduce pre-0.10 results.)

    random_seed : int, default=27
        Random seed for reproducibility of permutation tests.

    n_repeats : int, default=3
        Number of random permutations to generate the null distribution for
        spatial specificity scoring. More repeats give more stable estimates.

    mu : float, default=0.25
        Strength of the null subtraction in the spatial specificity score:
        ``score = cos_observed - mu * cos_shuffled``. Larger values subtract
        the randomized-background signal more strongly and are more
        conservative; mu=1 corresponds to the plain difference
        Delta = cos_observed - cos_shuffled. The default 0.25 matches the
        value used in the LARIS manuscript analyses. Not to be confused with
        ``mu_celltype``, which is the (unrelated) COSG regularization
        parameter for cell type specificity.
        (Changed from 1 to 0.25 in v0.10.0; pass mu=1 to reproduce pre-0.10
        results.)

    sigma : float or 'adaptive', default='adaptive'
        Bandwidth of the exponential kernel converting spatial k-NN distances
        to edge weights in the spatial specificity step
        (``exp(-d / sigma)``). 'adaptive' uses half the mean k-NN edge
        distance, which makes results independent of coordinate units (chip
        pixels, image pixels, or micrometres) and platform spot spacing, and
        matches the kernel used by prepareLRInteraction and the cell-type
        co-localization step. A numeric value is an absolute distance in
        coordinate units.
        (Changed from a fixed 100 to 'adaptive' in v0.10.0; pass sigma=100 to
        reproduce pre-0.10 results, which assumed roughly-micrometre
        coordinates.)

    remove_lowly_expressed : bool, default=True
        Whether to filter out LR pairs with low expression before ranking.

    expressed_pct : float, default=0.1
        Minimum fraction of cells expressing an LR pair (if remove_lowly_expressed=True).

    n_cells_expressed_threshold : int or float, default=100
        Minimum number of cells expressing an LR pair for it to be ranked.
        Pairs below this threshold receive a penalty in ranking.
        A value >= 1 is an absolute cell count; a float in (0, 1) is
        interpreted as a fraction of the total number of cells
        (e.g. 0.001 = 0.1% of cells), which scales better across platforms
        with very different cell/bin counts (e.g. 8 um Visium HD bins).
        The penalty affects the ranking only. The cell-type interaction
        score always uses the pair's actual spatial specificity (v0.15.0;
        earlier versions also fed the penalised ranking value into the
        score, which zeroed every pair below the threshold).

    n_top_lr : int, default=4000
        Number of top-ranked spatially-specific LR pairs to return.
        With ``background=`` and ``calculate_pvalues=True`` every LR pair is
        still *tested*: the cell-type analysis and the FDR run over all
        pairs, and only the returned tables are restricted to the top
        ``n_top_lr``. Selecting pairs by their own spatial specificity
        before testing would bias the p-values (the selection would be on
        the outcome), so ``n_top_lr`` only limits what is reported.

    by_celltype : bool, default=True
        Whether to compute cell type-specific interaction scores. If False,
        only returns spatial specificity results (much faster). If True,
        `data` must be provided. The Other Parameters configure this
        cell-type analysis and its P values; they are used only when
        ``by_celltype=True``.

    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
        If `by_celltype=False`:
            Single DataFrame with spatial specificity results:

            Columns:
            - 'ligand': Ligand gene name
            - 'receptor': Receptor gene name  
            - 'score': LARIS spatial specificity score
            - 'Rank': Rank (0 = highest scoring)

            Index: LR pair names ("ligand::receptor")
            Sorted by score (descending)

        If `by_celltype=True`:
            Tuple of (laris_lr, celltype_results) where:

            - **laris_lr**: DataFrame as described above

            - **celltype_results**: DataFrame with cell type-specific scores:

              Columns:
              - 'sender': Cell type sending the ligand
              - 'receiver': Cell type receiving the signal
              - 'ligand': Ligand gene name
              - 'receptor': Receptor gene name
              - 'interaction_name': "ligand::receptor"
              - 'interaction_score': Integrated LARIS score
              - 'p_value': P value (if calculate_pvalues=True). With
                ``background=``, the exact tail of the row's score among
                its expression-matched pseudo-pairs in the same
                sender-receiver pair.
              - 'p_value_fdr': Benjamini-Hochberg FDR within each
                sender-receiver pair (if calculate_pvalues=True). With
                ``background=``, a row with ``p_value_fdr < 0.05`` is an
                interaction whose score exceeds that of expression-matched
                gene pairs in the same sender-receiver context, at an FDR
                of 5% within that context. Pairs of the two cell types'
                marker genes are part of that null, so a call certifies
                specificity beyond cell-type identity; it does not test
                whether the two proteins bind (the pairing comes from the
                database).
              - 'nlog10_p_value_fdr': -log10(FDR) for visualization

              With ``background=``, these diagnostic columns are added:

              - 'null_support': how many of the row's pseudo-pairs scored
                above zero. This is the null's *effective* resolution: a
                row backed by 100 non-zero pseudo-pairs cannot really
                resolve the 1e-4 its p-value may report. See
                ``min_null_support``.
              - 'null_size': how many pseudo-pairs the row's null has: k**2
                minus the database pairs inside it, fewer when a gene's
                matched set is truncated. The p-value floor is
                1/(null_size + 1).
              - 'edge_limited': True when the ligand or the receptor has a
                truncated matched set (one of the few most or least
                abundant genes, without k/2 peers on one side; see
                ``prepareLRBackground(matching=)``). The set still
                straddles the gene, so the row is tested as usual.
              - 'ligand_rank', 'receptor_rank': descriptive partner ranks.
                ``ligand_rank`` is the observed pair's rank among the
                ligand's matched genes, each paired with the observed
                receptor: (1 + number of those pairs scoring at least as
                high) / (1 + number of valid pairs). ``receptor_rank`` is
                the same with the receptor's matched genes and the observed
                ligand. 1/(k + 1) means the gene beats all of its peers
                given its partner; a call that leans on one partner (a
                specific ligand acting through a broadly expressed
                receptor) has a small rank on that side only. Resolution
                1/(k + 1), so not a test; 1.0 for a row whose score is 0.
              - 'null_matchability': the larger of the two genes'
                saturation values - the fraction of a gene's matched set
                whose mean lies below the gene itself. 0.5 is ideal; 1.0
                means every matched gene is weaker, so every pseudo-pair
                is weaker than the real pair and the p-value overstates.
                The pool augmentation in ``prepareLRBackground`` normally
                prevents this; values near 1 among called rows mean it was
                disabled or could not reach that gene.
              - 'pair_breadth': the fraction of tested sender-receiver
                combinations in which this pair is called at FDR < 0.05.
                Genuine cell-type-specific results are narrow (medians of
                1-2% measured across four datasets); a pair called across
                more than ~25% of the grid carries no cell-type
                information, however real its expression - typically a
                tissue-ubiquitous pair.

              Sorted by ``interaction_score`` (descending) by default; with
              ``sort_by='p_value'``, by ``p_value`` (ascending), ties by
              ``interaction_score`` (descending).

    Other Parameters
    ----------------
    groupby : str, default='CellTypes'
        Column in `adata.obs` defining cell type groups.
    use_rep_spatial : str, default='X_spatial'
        Key in `adata.obsm` for spatial coordinates (for cell type analysis).
    number_nearest_neighbors : int, default=20
        Number of spatial neighbors for cell type co-localization analysis.
        (Changed from 10 to 20 in v0.10.0 for consistency with
        n_nearest_neighbors; pass number_nearest_neighbors=10 to reproduce
        pre-0.10 results.)
    mu_celltype : float, default=100
        Regularization parameter for COSG cell type specificity calculation.
        Higher values more strongly penalize broadly expressed genes.
        This is COSG's lambda-style regularizer and is unrelated to ``mu``
        (the null-subtraction strength in the spatial specificity step).
    sigma_celltype : float or 'adaptive', default='adaptive'
        Kernel bandwidth for the cell-type co-localization spatial graph,
        with the same semantics as ``sigma``. 'adaptive' reproduces the
        pre-0.10 behaviour of this step.
    expressed_pct_celltype : float, default=0.1
        Minimum expression fraction for cell type analysis.
    remove_lowly_expressed_celltype : bool, default=True
        Whether to filter lowly expressed genes in cell type analysis.
    mask_threshold : float, default=1e-6
        Numerical threshold for masking near-zero values.
    calculate_pvalues : bool, default=True
        Whether to perform permutation testing for statistical significance.
        Set to False for faster exploratory analysis.
    layer_celltype : str, optional
        Layer in `adata.layers` to use for expression data. If None, uses `adata.X`.
    n_neighbors_permutation : int, default=30
        Number of similar interactions to use as background controls for
        permutation testing. These are selected based on similarity of
        diffused score profiles.
    n_permutations : int, default=1000
        **Legacy path only, and ignored when ``background=`` is given.**

        Without a background, p-values come from sampling
        ``n_permutations`` draws from a resampled-pair null, and the
        floor is ``1/(n_permutations+1)``.

        With a background the null is *enumerated exactly* - every one of
        the ``n_matched_genes ** 2`` pseudo-pairs is scored and the
        p-value is the exact tail ``(exceed + 1) / (support + 1)``. There
        is no sampling, no seed and nothing for this parameter to do:
        the floor is ``1/(n_matched_genes ** 2 + 1)`` and
        ``n_matched_genes`` is the only knob that moves it. Passing this
        parameter together with ``background=`` raises a
        ``FutureWarning``.
    chunk_size : int, default=50000
        Number of interactions to process simultaneously during permutation.
        Larger values are faster but use more memory.
    prefilter_fdr : bool, default=True
        **Legacy (no-background) path only.** With ``background=`` the
        interactions that Benjamini-Hochberg covers are fixed by the null
        (see ``min_null_support``) and this parameter is ignored.

        In the legacy path, True restricts BH to rows whose score exceeds
        ``prefilter_threshold``. Restricting the tested rows by the observed
        score is only valid with a p-value that conditions on the same
        event, so the legacy path then uses the conditional p-value
        (``use_conditional_pvalue``).
    prefilter_threshold : float, default=0.0
        Minimum interaction score for FDR testing in the legacy path
        (if prefilter_fdr=True). Ignored with ``background=``.
    score_threshold : float, default=1e-6
        Scores below this are set to exactly 0.0. With ``background=`` the
        same rule is applied to every pseudo-pair, so a pseudo-pair counts
        as non-zero (``null_support``) under exactly the condition an
        observed score does.
    spatial_weight : float, default=3.0
        Exponent applied to the (non-negative-clamped) spatial specificity
        score before it multiplies the cell-type interaction score:
        ``interaction_score *= max(delta, 0) ** spatial_weight``.
        **Despite the name this is a power, not a multiplicative weight** -
        raising it changes the score non-linearly.
        - 0: Ignore spatial specificity entirely
        - 1: Linear influence
        - >1: Stronger emphasis on spatially specific LR pairs (the default 3
          is the generally recommended setting and matches the tutorial);
          increase it further to sharpen the ranking toward pairs with strong
          spatial structure, decrease toward 1 when spatial specificity
          estimates are noisy (e.g. very sparse data)
        - <1: Weaker emphasis
        Negative spatial specificity scores are clamped to 0 before powering
        (v0.10.0): a pair whose spatial pattern is indistinguishable from or
        weaker than the random background contributes no interaction score,
        instead of producing NaN (fractional weights) or a sign flip (even
        weights).
        (Changed from 1.0 to 3.0 in v0.10.0; pass spatial_weight=1.0 to
        reproduce pre-0.10 results.)
    use_conditional_pvalue : bool or None, default=None
        **Legacy (no-background) path only.** When True:
        - Interactions with score=0 get p-value=1.0
        - Non-zero scores are compared only to non-zero background scores
        None (default) uses the conditional p-value exactly when
        ``prefilter_fdr=True``, which is the combination that keeps the FDR
        valid: a test restricted to non-zero scores needs a p-value that
        conditions on the score being non-zero. Passing False together with
        ``prefilter_fdr=True`` warns and uses the conditional p-value.
    rescale : bool, default=True
        Rescale interaction scores so that the mean of the top-100 scores is
        0.1. The applied factor is dataset-dependent and is recorded in
        ``lr_adata.uns['laris_scale_factor']`` and
        ``celltype_results.attrs['laris_scale_factor']`` (1.0 when
        rescale=False), so separate runs can be put back on a common scale
        by dividing scores by their respective factors. Because the factor
        varies between runs (and with subsampling), never compare or subtract
        rescaled ``interaction_score`` values across runs directly.
    cosg_backend : {'auto', 'memory', 'stream'}, default='auto'
        How to compute cell type specificity. 'stream' reads the expression
        in chunks from a cytome via ``cosg.run_cosg_cytome``, so no
        cells x LR-genes AnnData is built; 'memory' materialises the
        ligand/receptor gene subset. 'auto' streams for cytome sources and
        uses memory otherwise, so AnnData workflows are unaffected. The two
        agree to float32 precision.
    specificity_reference : {'lr', 'all'}, default='lr'
        Which genes set the scale when cell type specificity is normalised.
        ``cosg.iqrLogNormalize`` divides each cell type's scores by that
        cell type's ``q0.95 - q0.75`` spread over the genes it is given, so
        the reference set matters. 'lr' uses the ligand/receptor genes,
        which makes results independent of whatever *other* genes are in
        the object - subsetting to highly variable genes changes nothing.
        'all' uses every gene, which instead makes results independent of
        which LR database you look up. Gene rankings within a cell type are
        identical either way (the transform is monotone); what changes is
        the relative scale between cell types, and therefore the ranking of
        sender-receiver pairs. 'lr' is the default and matches the
        published results; prefer 'all' when very few database genes are
        present, as on targeted panels, where the spread over a handful of
        LR genes is unstable (a warning fires below 100).
    section_key : str, optional
        Column in ``.obs`` identifying which tissue section each cell
        belongs to. Several sections tiled into one coordinate system
        otherwise share a spatial k-NN, so cells near a tile edge acquire
        neighbours from the adjacent section. With ``section_key`` every
        neighbour graph in the pipeline - the spatial specificity step, its
        randomised background, and the cell-type co-localization step - is
        built independently within each section (GitHub issue #8). Pass the
        same column to :func:`prepareLRInteraction`.
    background : LRBackground, optional
        From :func:`prepareLRBackground`, built on the same cells and graph
        settings. With ``calculate_pvalues=True`` it gives every LR pair
        an exact P value against its expression-matched null (the
        factorized matched-gene test): ``p_value``, ``p_value_fdr`` and the
        diagnostic columns described under Returns. Its diffusion and
        spatial-specificity graphs must be the ones the LR scores and this
        call use; a mismatch raises before any work. Without it, P values
        come from the legacy permutation path.
    min_null_support : int or 'auto', default='auto'
        With ``background=``: within each sender-receiver pair,
        Benjamini-Hochberg covers every LR pair whose matched-gene null has
        at least ``max(1, min_null_support)`` non-zero pseudo-pairs
        (``null_support``). Rows below that get p-value 1.0 and are not
        tested. 'auto' (v0.15.0) is 2 x ``n_matched_genes`` of the
        background (400 at k = 200): below that, the non-zero part of the
        null can come from a single matched gene's row or column of the
        k x k cross, and a tiny score beats an otherwise all-zero null - the
        test would measure whether the pair clears the 10% detection gate,
        not its strength. Pass 0 to test every row with a non-zero null.

        The p-value is the exact tail ``(exceed + 1) / (n_valid + 1)`` over
        all valid pseudo-pairs, zeros included. That is a valid p-value
        only if the set of tested rows is not chosen by the observed score,
        so that set is defined by the null alone: a row whose null is
        identically zero cannot reach significance and is left out, every
        other row is in, whatever its own score (v0.15.0; earlier versions
        tested only the rows with a positive score, which inflated the FDR).
        ``min_null_support`` is a property of the null, fixed in advance,
        so any value keeps the FDR valid.
    sort_by : {'interaction_score', 'p_value'}, default='interaction_score'
        Order of the cell-type results. 'interaction_score' ranks by the
        score, descending, as in earlier versions. 'p_value' ranks by the
        p-value, ascending, ties by the score, so the head of the table is
        the strongest evidence; it needs ``calculate_pvalues=True``. The
        order changes no value.
    lr_adata, adata : deprecated
        Former names of ``lr_data`` and ``data``.

    Raises
    ------
    ValueError
        If by_celltype=True but adata is not provided, or if required data
        is missing from adata or lr_adata.

    ImportError
        If required helper functions are not available.

    See Also
    --------
    prepareLRInteraction : Prepare LR scores (prerequisite for this function)

    Examples
    --------
    **Example 1: Quick spatial specificity analysis (no cell types)**

    >>> import laris as la
    >>> 
    >>> # Prepare LR scores
    >>> lr_adata = la.tl.prepareLRInteraction(adata, lr_df)
    >>> 
    >>> # Identify spatially-specific LR pairs only
    >>> laris_lr = la.tl.runLARIS(
    ...     lr_adata,
    ...     by_celltype=False,
    ...     n_top_lr=1000
    ... )
    >>> 
    >>> print(laris_lr.head())

    **Example 2: Full analysis with cell type-specific scores**

    >>> # Full LARIS analysis with cell types
    >>> laris_lr, celltype_results = la.tl.runLARIS(
    ...     lr_adata,
    ...     adata,
    ...     by_celltype=True,
    ...     groupby='cell_type',
    ...     calculate_pvalues=True,
    ...     n_permutations=5000
    ... )
    >>> 
    >>> # Filter for significant interactions
    >>> sig_results = celltype_results[
    ...     celltype_results['p_value_fdr'] < 0.05
    ... ]
    >>> 
    >>> print(f"Found {len(sig_results)} significant interactions")

    **Example 3: Fast exploratory analysis (no p-values)**

    >>> laris_lr, celltype_results = la.tl.runLARIS(
    ...     lr_adata,
    ...     adata,
    ...     by_celltype=True,
    ...     calculate_pvalues=False  # Much faster!
    ... )

    **Example 4: Conservative testing for sparse data**

    >>> laris_lr, celltype_results = la.tl.runLARIS(
    ...     lr_adata,
    ...     adata,
    ...     by_celltype=True,
    ...     use_conditional_pvalue=True,  # Robust for sparse data
    ...     n_permutations=5000,
    ...     prefilter_fdr=True,
    ...     prefilter_threshold=0.01  # Only test score > 0.01
    ... )

    **Example 5: Emphasize spatial specificity**

    >>> laris_lr, celltype_results = la.tl.runLARIS(
    ...     lr_adata,
    ...     adata,
    ...     by_celltype=True,
    ...     spatial_weight=2.0  # Square the spatial scores
    ... )
    """
    # Validate inputs
    lr_data = resolve_data_arg(lr_data, 'runLARIS', canonical='lr_data',
                               lr_adata=lr_adata)
    data = resolve_data_arg(data, 'runLARIS', canonical='data',
                            required=False, adata=adata)
    if specificity_reference not in ('lr', 'all'):
        raise ValueError(
            f"specificity_reference must be 'lr' or 'all', got "
            f"{specificity_reference!r}"
        )
    if cosg_backend not in ('auto', 'memory', 'stream'):
        raise ValueError(
            f"cosg_backend must be 'auto', 'memory' or 'stream', got "
            f"{cosg_backend!r}"
        )

    # The LR scores may arrive as an AnnData or as an LR cytome written by
    # prepareLRInteraction(return_type='cytome').
    lr_adata = _ensure_lr_anndata(lr_data)

    # Cell type specificity is the only step that touches the expression
    # object. When it is a cytome we can hand it to COSG's streaming reader
    # and never build an expression AnnData at all; 'memory' forces the old
    # behaviour of materialising the ligand/receptor gene subset.
    expression_is_cytome = data is not None and not isinstance(data, ad.AnnData) \
        and _is_cytome_source(data)
    if cosg_backend == 'stream' and not expression_is_cytome:
        raise ValueError(
            "cosg_backend='stream' requires a cytome expression source; "
            "pass data='sample.cytome' or use cosg_backend='memory'."
        )
    stream_cosg = expression_is_cytome and cosg_backend in ('auto', 'stream')

    if by_celltype and data is None:
        raise ValueError(
            "Parameter 'data' must be provided when by_celltype=True. "
            "It should contain gene expression and cell type annotations."
        )
    
    if use_rep not in lr_adata.obsm:
        raise KeyError(
            f"Representation '{use_rep}' not found in lr_adata.obsm. "
            f"Available keys: {list(lr_adata.obsm.keys())}"
        )
    
    print("\n" + "="*70)
    print("LARIS ANALYSIS")
    print("="*70)
    print(f"\nInput data: {lr_adata.shape[0]} cells × {lr_adata.shape[1]} LR pairs")
    print(f"Mode: {'Cell type-specific analysis' if by_celltype else 'Spatial specificity only'}")

    # With streaming COSG the expression object stays on disk. Otherwise a
    # cytome source is materialised for exactly the genes the cell-type
    # step scores: the ligands and receptors, plus - with a background -
    # the background's table genes, whose specificity the exact null needs
    # (v0.14.0 loaded the LR genes only, so every matched gene got
    # specificity 0, its pseudo-pairs scored 0 and the null was too easy:
    # 1,024 calls against 817 in memory on 10,000 Xenium cells).
    # specificity_reference='all' normalises over every gene, so it needs
    # them all.
    cytome_source = data if stream_cosg else None
    if stream_cosg:
        adata = None
    elif data is not None and not isinstance(data, ad.AnnData):
        if specificity_reference == 'all':
            cell_type_genes = None
        else:
            cell_type_genes = list(pd.unique(pd.Series(
                [g for name in lr_adata.var_names for g in name.split('::')]
            )))
            if background is not None:
                cell_type_genes = list(pd.Index(cell_type_genes).union(
                    pd.Index(background.gene_index), sort=False))
        adata = _ensure_expression_anndata(data, genes=cell_type_genes)
    else:
        adata = data
    
    # Import helper functions
    try:
        from . import _utils
    except ImportError:
        raise ImportError(
            "Could not import _utils module. Please ensure all LARIS "
            "dependencies are properly installed."
        )
    
    if background is not None and calculate_pvalues:
        _check_background_graph(background, use_rep=use_rep,
                                n_nearest_neighbors=n_nearest_neighbors,
                                sigma=sigma, section_key=section_key,
                                diffusion=lr_adata.uns.get('laris_diffusion'))

    # =========================================================================
    # STEP 1: Calculate LARIS Spatial Specificity Scores
    # =========================================================================
    if sort_by not in ('interaction_score', 'p_value'):
        raise ValueError(
            f"sort_by must be 'interaction_score' or 'p_value', got {sort_by!r}")
    if sort_by == 'p_value' and by_celltype and not calculate_pvalues:
        raise ValueError("sort_by='p_value' needs calculate_pvalues=True")

    print("\n--- Step 1: Calculating spatial specificity scores ---")
    print(f"  - Using {n_nearest_neighbors} nearest neighbors")
    print(f"  - Regularization parameter μ = {mu}")
    print(f"  - Random repeats: {n_repeats}")
    
    # Build spatial adjacency matrix
    if n_repeats is not None:
        warnings.warn(
            "n_repeats is deprecated and ignored: the shuffled-graph "
            "baseline is now computed analytically (closed form of the "
            "random-graph expectation), so no realized random graphs are "
            "built and the spatial specificity is deterministic.",
            FutureWarning, stacklevel=2,
        )

    # n_permutations drives only the legacy resampled-pair null. With a
    # background the null is enumerated exactly (every n_matched_genes**2
    # pseudo-pair is scored), so the parameter has nothing to do and
    # silently ignoring it invites the misreading that raising it buys
    # precision - which is exactly how the sampled-null description
    # outlived the sampled null.
    if n_permutations is _UNSET:
        n_permutations = 1000
    elif background is not None:
        warnings.warn(
            "n_permutations is ignored when background= is given: the "
            "factorized null is enumerated exactly, so its floor is "
            "1/(n_matched_genes**2 + 1) and only n_matched_genes moves "
            "it. Set n_matched_genes in prepareLRBackground instead.",
            FutureWarning, stacklevel=2,
        )
    sections = _utils._resolve_sections(lr_adata, section_key, lr_adata.n_obs)
    if sections is not None:
        print(f"  - Neighbour graphs built within {len(pd.unique(sections))} "
              f"section(s) from '{section_key}'")
    cellxcell = _utils._build_adjacency_matrix(
        lr_adata,
        use_rep=use_rep,
        n_nearest_neighbors=n_nearest_neighbors,
        sigma=sigma,
        sections=sections
    )
    
    # Calculate observed spatial correlation
    genexcell = lr_adata.X.T
    order1 = genexcell @ cellxcell.T
    gsp = _utils._rowwise_cosine_similarity(genexcell, order1)
    
    # Shuffled-graph baseline, analytic. The pipeline previously averaged
    # cos(v, W_rand v) over n_repeats realized random graphs; under the
    # L1-normalized random graph that mean estimates a closed-form
    # expectation in the profile's first two moments and one graph scalar
    # (docs/discussion/2026-08-25_analytic_null_proof.md). Computing the
    # expectation directly removes the Monte Carlo noise and the seed
    # dependence; on tonsil the resulting delta agrees with the realized
    # computation at Spearman 0.99995 (max abs diff 0.028).
    print("  - Computing analytic shuffled-graph baseline...")
    _X_pairs = lr_adata.X.tocsc() if sp.issparse(lr_adata.X) else np.asarray(lr_adata.X)
    if sp.issparse(_X_pairs):
        _m1 = np.asarray(_X_pairs.mean(axis=0)).ravel()
        _m2 = np.asarray(_X_pairs.power(2).mean(axis=0)).ravel()
    else:
        _m1 = _X_pairs.mean(axis=0)
        _m2 = (_X_pairs ** 2).mean(axis=0)
    _R = lr_adata.n_obs * _background._expected_row_sq_sum(
        cellxcell.data, n_nearest_neighbors)
    random_gsp = _background._analytic_random_gsp(
        _m1, _m2, lr_adata.n_obs, _R)

    # The removed random-graph builder used to seed the global RNG as a side
    # effect, and the legacy permutation p-value sampler downstream consumes
    # the global RNG. Seed it explicitly so that path stays deterministic.
    np.random.seed(random_seed)
    
    # Calculate spatial specificity score
    gsp_score = gsp - mu * random_gsp
    gsp_score = np.array(gsp_score).ravel()
    
    # Store in lr_adata
    lr_adata.var['LRSS_Target'] = np.array(gsp).ravel()
    lr_adata.var['LRSS_Random'] = np.array(random_gsp).ravel()
    lr_adata.var['LR_SpatialSpecificity'] = gsp_score
    
    # Calculate QC metrics. percent_top values must not exceed the number of
    # features, otherwise scanpy raises IndexError (e.g. < 100 LR pairs).
    if lr_adata.shape[1] < 500:
        percent_top = [p for p in (50, 100) if p <= lr_adata.shape[1]] or None
        sc.pp.calculate_qc_metrics(lr_adata, inplace=True, percent_top=percent_top)
    else:
        sc.pp.calculate_qc_metrics(lr_adata, inplace=True)
    
    # Rank LR pairs by spatial specificity
    lr_var = lr_adata.var.sort_values(
        by='LR_SpatialSpecificity', 
        ascending=False
    ).copy()
    
    n_cells_expressed = lr_var['n_cells_by_counts'].values.copy()
    spatial_specificity = lr_var['LR_SpatialSpecificity'].values.copy()
    gsp_score_for_ranking = spatial_specificity.copy()

    # Penalize LR pairs with low cell counts IN THE RANKING. A float
    # threshold in (0, 1) is interpreted as a fraction of the number of
    # cells (sklearn-style), a value >= 1 as an absolute cell count.
    if 0 < n_cells_expressed_threshold < 1:
        cells_expressed_cutoff = int(np.ceil(
            n_cells_expressed_threshold * lr_adata.shape[0]
        ))
    else:
        cells_expressed_cutoff = int(n_cells_expressed_threshold)
    min_score = np.min(gsp_score_for_ranking)
    low_count_mask = n_cells_expressed < cells_expressed_cutoff
    gsp_score_for_ranking[low_count_mask] = min_score - 0.001

    def _lr_table(indices):
        names = lr_var.index.values[indices]
        table = pd.DataFrame({
            'ligand': [name.split('::')[0] for name in names],
            'receptor': [name.split('::')[1] for name in names],
            # the pair's actual spatial specificity. Up to v0.14.1 this held
            # the penalised ranking value, and the cell-type score used it
            # as its spatial factor: every pair below the cell-count
            # threshold scored 0 everywhere (1,325 of 1,985 pairs on the
            # tonsil) although the penalty is documented as a ranking one.
            'score': spatial_specificity[indices],
            'n_cells_expressed': n_cells_expressed[indices],
        })
        table.index = [f"{l}::{r}" for l, r in zip(table.ligand, table.receptor)]
        table['Rank'] = np.arange(len(table))
        return table

    # Select top N LR pairs (what is returned)
    top_indices = _utils._select_top_n(gsp_score_for_ranking, n_top_lr)
    laris_lr = _lr_table(top_indices)
    # With p-values every pair is tested, whatever n_top_lr: choosing pairs
    # by their own spatial specificity before testing selects on the
    # outcome. n_top_lr then only restricts the returned tables.
    if calculate_pvalues and len(top_indices) < len(gsp_score_for_ranking):
        laris_lr_tested = _lr_table(_utils._select_top_n(
            gsp_score_for_ranking, len(gsp_score_for_ranking)))
    else:
        laris_lr_tested = laris_lr

    print(f"  ✓ Identified {len(laris_lr)} top spatially-specific LR pairs")
    print(f"  - Score range: [{laris_lr['score'].min():.4f}, "
          f"{laris_lr['score'].max():.4f}]")
    if len(laris_lr_tested) > len(laris_lr):
        print(f"  - Testing all {len(laris_lr_tested):,} pairs; results are "
              f"restricted to the top {len(laris_lr):,} afterwards")
    
    # =========================================================================
    # STEP 2: Calculate Cell Type-Specific Interactions (Optional)
    # =========================================================================
    if by_celltype:
        print("\n" + "="*70)
        print("CELL TYPE-SPECIFIC ANALYSIS")
        print("="*70)
        
        # lr_adata carries the same cells (and therefore the same obs) as
        # the expression object, and is present under every backend - with
        # streaming COSG the expression object is never opened here.
        celltype_obs = lr_adata.obs
        if groupby not in celltype_obs:
            raise ValueError(
                f"Cell type column '{groupby}' not found in the cell "
                f"metadata. Available columns: {list(celltype_obs.columns)}"
            )
        
        n_missing = celltype_obs[groupby].isna().sum()
        if n_missing:
            # Real datasets carry unannotated cells (NaN); they cannot be a
            # sender or receiver, and mixing them into the label list also
            # used to crash the summary below on str-vs-float comparison.
            raise ValueError(
                f"Cell type column '{groupby}' has {n_missing:,} missing "
                f"value(s). Drop or label those cells before running, e.g. "
                f"data = data[data.obs['{groupby}'].notna()].copy()."
            )
        n_cell_types = celltype_obs[groupby].nunique()
        labels = sorted(map(str, pd.unique(celltype_obs[groupby].dropna())))
        print(f"\nAnalyzing {n_cell_types} cell types from '{groupby}'")
        print(f"Cell types: {labels[:10]}"
              f"{'...' if n_cell_types > 10 else ''}")
        
        celltype_results = _utils._calculate_laris_score_by_celltype(
            adata=adata,
            cytome_source=cytome_source,
            specificity_reference=specificity_reference,
            section_key=section_key,
            lr_adata=lr_adata,
            laris_lr=laris_lr_tested,
            groupby=groupby,
            use_rep_spatial=use_rep_spatial,
            number_nearest_neighbors=number_nearest_neighbors,
            mu=mu_celltype,
            sigma=sigma_celltype,
            expressed_pct=expressed_pct_celltype,
            remove_lowly_expressed=remove_lowly_expressed_celltype,
            mask_threshold=mask_threshold,
            calculate_pvalues=calculate_pvalues,
            layer=layer_celltype,
            n_nearest_neighbors=n_neighbors_permutation,
            n_permutations=n_permutations,
            chunk_size=chunk_size,
            prefilter_fdr=prefilter_fdr,
            prefilter_threshold=prefilter_threshold,
            score_threshold=score_threshold,
            spatial_weight=spatial_weight,
            use_conditional_pvalue=use_conditional_pvalue,
            rescale=rescale,
            background=background,
            mu_gsp=mu,
            min_null_support=min_null_support,
            sort_by=sort_by,
        )
        if len(laris_lr_tested) > len(laris_lr):
            # tested (and FDR-adjusted) over every pair; report the top ones
            _attrs = dict(celltype_results.attrs)
            celltype_results = celltype_results[
                celltype_results['interaction_name'].isin(laris_lr.index)
            ].reset_index(drop=True)
            celltype_results.attrs.update(_attrs)

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nResults summary:")
        print(f"  - Spatially-specific LR pairs: {len(laris_lr)}")
        print(f"  - Cell type combinations: {len(celltype_results):,}")
        
        if calculate_pvalues:
            n_sig = (celltype_results['p_value_fdr'] < 0.05).sum()
            print(f"  - Significant interactions (FDR < 0.05): {n_sig:,}")
        
        return laris_lr, celltype_results
    
    else:
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nReturning {len(laris_lr)} spatially-specific LR pairs")
        
        return laris_lr
