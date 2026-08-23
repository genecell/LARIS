"""
LARIS Tools Module (laris.tl)

Core analytical tools for ligand-receptor interaction analysis in spatial transcriptomics data.

This module contains public functions for:
- Preparing ligand-receptor integration scores with spatial diffusion
- Running the LARIS algorithm to identify spatially-specific LR interactions
- Computing cell type-specific interaction scores

Main Functions:
- prepareLRInteraction: Calculate LR interaction scores using spatial neighborhoods
- runLARIS: Identify spatially-specific LR pairs and compute cell type interactions
"""

import pandas as pd
import scanpy as sc
import numpy as np
import anndata as ad
import scipy.sparse as sp
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from typing import Optional, Union, List, Tuple

from . import _utils
from ._compare import compare_laris_internal
from ._io import readCytome, _ensure_expression_anndata, _is_cytome_source


def prepareLRInteraction(
    adata: ad.AnnData,
    lr_df: pd.DataFrame,
    number_nearest_neighbors: int = 10,
    use_rep_spatial: str = 'X_spatial',
    unmatched: str = 'drop',
    sigma: Union[float, str] = 'adaptive'
) -> ad.AnnData:
    """
    Calculate ligand-receptor integration scores using spatial neighborhood information.
    
    This function computes diffused ligand-receptor interaction scores by considering
    the spatial context of each cell. It uses k-nearest neighbors to create a spatial
    neighborhood graph and calculates element-wise multiplication of diffused ligand
    and receptor expression levels.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing gene expression and spatial coordinates.
        Must have `.obsm[use_rep_spatial]` with spatial coordinates.
    lr_df : pd.DataFrame
        DataFrame containing ligand-receptor pairs with columns 'ligand' and 'receptor'.
    number_nearest_neighbors : int, default=10
        Number of nearest neighbors to consider for spatial diffusion.
    use_rep_spatial : str, default='X_spatial'
        Key in adata.obsm containing spatial coordinates.
    unmatched : {'drop', 'error'}, default='drop'
        How to handle ligand or receptor names in `lr_df` that are absent from
        `adata.var_names`. With 'drop', unmatched pairs are removed and a
        UserWarning summarises how many were dropped and lists example missing
        names; this is convenient for targeted panels (e.g. Xenium, MERFISH)
        where most database genes are legitimately absent. With 'error', a
        ValueError is raised instead, which is safer when supplying a custom
        database where a missing name likely indicates a typo.
    sigma : float or 'adaptive', default='adaptive'
        Bandwidth of the exponential kernel converting spatial k-NN
        distances into diffusion weights (``exp(-d / sigma)``). 'adaptive'
        uses half the mean k-NN edge distance, making the result independent
        of coordinate units (pixels vs micrometres) and platform spot
        spacing. Pass a number for an absolute bandwidth in coordinate
        units.
        
    Returns
    -------
    AnnData
        New AnnData object containing ligand-receptor interaction scores.
        - `.X`: Sparse matrix of LR interaction scores (cells × LR pairs)
        - `.var_names`: Ligand-receptor pair names in format "ligand::receptor"
        - `.obs`: Cell metadata copied from input adata
        - `.obsm`: Spatial and other representations copied from input adata
        
    Examples
    --------
    >>> import laris as la
    >>> import pandas as pd
    >>> 
    >>> # Define ligand-receptor pairs
    >>> lr_df = pd.DataFrame({
    ...     'ligand': ['Tgfb1', 'Vegfa'],
    ...     'receptor': ['Tgfbr1', 'Kdr']
    ... })
    >>> 
    >>> # Calculate LR integration scores
    >>> lr_adata = la.tl.prepareLRInteraction(adata, lr_df)
    >>> print(lr_adata.shape)  # (n_cells, n_lr_pairs)

    """
    # Accept a cytome source: stream just the ligand/receptor gene subset
    # into an in-memory AnnData, then run the standard path unchanged.
    if not isinstance(adata, ad.AnnData):
        lr_genes = list(pd.unique(pd.concat([
            lr_df['ligand'].astype(str), lr_df['receptor'].astype(str)
        ])))
        adata = _ensure_expression_anndata(adata, genes=lr_genes)

    # Deduplicate lr_df to prevent duplicate var_names in lr_adata, which
    # causes identical background sets and duplicate p-values downstream.
    n_before = len(lr_df)
    lr_df = lr_df.drop_duplicates(subset=['ligand', 'receptor']).reset_index(drop=True)
    n_dropped = n_before - len(lr_df)
    if n_dropped > 0:
        import warnings
        warnings.warn(
            f"prepareLRInteraction: removed {n_dropped} duplicate ligand-receptor "
            f"pair(s) from lr_df. Duplicate pairs cause identical p-values in "
            f"runLARIS. Ensure each (ligand, receptor) pair is unique.",
            UserWarning,
            stacklevel=2,
        )

    # Validate ligand/receptor names against adata.var_names BEFORE indexing.
    # np.searchsorted below returns insertion positions, so an unmatched name
    # would otherwise silently resolve to a neighbouring gene with no error.
    if unmatched not in ('drop', 'error'):
        raise ValueError(
            f"unmatched must be 'drop' or 'error', got {unmatched!r}"
        )
    var_names_set = set(adata.var_names)
    ligand_present = lr_df['ligand'].isin(var_names_set)
    receptor_present = lr_df['receptor'].isin(var_names_set)
    matched_mask = ligand_present & receptor_present
    if not matched_mask.all():
        missing_names = sorted(
            set(lr_df.loc[~ligand_present, 'ligand'])
            | set(lr_df.loc[~receptor_present, 'receptor'])
        )
        n_unmatched_pairs = int((~matched_mask).sum())
        example_names = ', '.join(missing_names[:10])
        if len(missing_names) > 10:
            example_names += ', ...'
        message = (
            f"prepareLRInteraction: {n_unmatched_pairs} ligand-receptor pair(s) "
            f"reference {len(missing_names)} gene name(s) absent from "
            f"adata.var_names (e.g. {example_names})."
        )
        if unmatched == 'error':
            raise ValueError(
                message + " Filter lr_df with .isin(adata.var_names) or pass "
                "unmatched='drop' to remove these pairs automatically."
            )
        import warnings
        warnings.warn(
            message + " These pairs were dropped. Pass unmatched='error' to "
            "raise instead (recommended when using a custom database, where a "
            "missing name may be a typo).",
            UserWarning,
            stacklevel=2,
        )
        lr_df = lr_df.loc[matched_mask].reset_index(drop=True)
    if len(lr_df) == 0:
        raise ValueError(
            "prepareLRInteraction: no ligand-receptor pairs remain after "
            "matching against adata.var_names."
        )

    X_spatial = adata.obsm[use_rep_spatial].copy()

    # Ensure adata.X is sparse so downstream .multiply() / .maximum() work.
    # If adata.X is a dense numpy array, convert to CSR sparse matrix.
    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)

    # Create the diffused ligand/receptor matrix
    cellxcell = kneighbors_graph(
        X_spatial,
        n_neighbors=number_nearest_neighbors,
        mode='distance',
        include_self=True
    )
    cellxcell = _utils._apply_knn_kernel(cellxcell, sigma=sigma)

    genexcell = X.copy().T
    order1 = genexcell @ cellxcell.T

    # Estimate diffused ligand-receptor activity
    sorter = np.argsort(adata.var_names)
    ligand_idx = sorter[np.searchsorted(adata.var_names, lr_df['ligand'], sorter=sorter)]
    receptor_idx = sorter[np.searchsorted(adata.var_names, lr_df['receptor'], sorter=sorter)]
    
    # Element-wise multiplication
    lrxcell = order1[ligand_idx, :].multiply(order1[receptor_idx, :])

    # Create an AnnData object
    lr_names = lr_df['ligand'].astype(str) + '::' + lr_df['receptor'].astype(str)
    lr_adata = sc.AnnData(lrxcell.T)
    lr_adata.obs = adata.obs.copy()
    lr_adata.obsm = adata.obsm.copy()
    lr_adata.var_names = lr_names

    # Carry tissue images along so plotCCCSpatial can overlay from lr_adata
    # alone (no adata= argument needed). Cheap: the same array objects.
    if 'spatial' in getattr(adata, 'uns', {}):
        lr_adata.uns['spatial'] = adata.uns['spatial']

    # Filter to only include cells where ligand or receptor is expressed
    ligand_idx = sorter[np.searchsorted(adata.var_names, lr_df['ligand'], sorter=sorter)]
    receptor_idx = sorter[np.searchsorted(adata.var_names, lr_df['receptor'], sorter=sorter)]

    ligand_mask = X[:, ligand_idx] != 0   # True where ligand is expressed
    receptor_mask = X[:, receptor_idx] != 0  # True where receptor is expressed

    non_zero_mask = ligand_mask.maximum(receptor_mask)  # elementwise OR for sparse matrices
    lr_adata.X = lr_adata.X.multiply(non_zero_mask)
    
    return lr_adata

def runLARIS(
    lr_adata: ad.AnnData,
    adata: Optional[ad.AnnData] = None,
    use_rep: str = 'X_spatial',
    n_nearest_neighbors: int = 20,
    random_seed: int = 27,
    n_repeats: int = 3,
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
    n_permutations: int = 1000,
    chunk_size: int = 50000,
    prefilter_fdr: bool = True,
    prefilter_threshold: float = 0.0,
    score_threshold: float = 1e-6,
    spatial_weight: float = 3.0,
    use_conditional_pvalue: bool = False,
    rescale: bool = True
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
    lr_adata : AnnData
        AnnData object containing LR interaction scores from `prepareLRInteraction()`.
        
        Required contents:
        - `.X`: Diffused LR scores (cells × LR pairs)
        - `.var_names`: LR pair names ("ligand::receptor")
        - `.obsm[use_rep]`: Spatial coordinates or other representation
        
    adata : AnnData, optional
        Original annotated data matrix with gene expression and spatial information.
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
        
    n_top_lr : int, default=4000
        Number of top-ranked spatially-specific LR pairs to return.
        
    by_celltype : bool, default=True
        Whether to compute cell type-specific interaction scores. If False,
        only returns spatial specificity results (much faster). If True,
        `adata` must be provided.
    
    Cell Type Analysis Parameters (only used if by_celltype=True)
    -------------------------------------------------------------
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
        Number of permutations for statistical testing. Common values:
        - 1000: Quick testing
        - 5000: More precise p-values
        - 10000: Publication-quality precision
        
    chunk_size : int, default=50000
        Number of interactions to process simultaneously during permutation.
        Larger values are faster but use more memory.
        
    prefilter_fdr : bool, default=True
        If True, only test interactions with scores > prefilter_threshold for
        significance. Others get FDR p-value = 1.0. This reduces multiple
        testing burden and focuses power on high-scoring interactions.
        
    prefilter_threshold : float, default=0.0
        Minimum interaction score for FDR testing (if prefilter_fdr=True).
        
    score_threshold : float, default=1e-6
        Numerical precision threshold. Scores below this are set to exactly 0.0.
        
    spatial_weight : float, default=3.0
        Exponent applied to the (non-negative-clamped) spatial specificity
        score before it multiplies the cell-type interaction score:
        ``interaction_score *= max(delta, 0) ** spatial_weight``.
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
        
    use_conditional_pvalue : bool, default=False
        Use conditional p-value calculation for zero-inflated data. **Recommended
        for sparse datasets**. When True:
        - Interactions with score=0 get p-value=1.0
        - Non-zero scores compared only to non-zero background

    rescale : bool, default=True
        Rescale interaction scores so that the mean of the top-100 scores is
        0.1. The applied factor is dataset-dependent and is recorded in
        ``lr_adata.uns['laris_scale_factor']`` and
        ``celltype_results.attrs['laris_scale_factor']`` (1.0 when
        rescale=False), so separate runs can be put back on a common scale
        by dividing scores by their respective factors. Because the factor
        varies between runs (and with subsampling), never compare or subtract
        rescaled ``interaction_score`` values across runs directly.
        - Prevents spurious significance from sparse null distributions
        
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
              - 'p_value': Raw permutation p-value (if calculate_pvalues=True)
              - 'p_value_fdr': FDR-corrected p-value (if calculate_pvalues=True)
              - 'nlog10_p_value_fdr': -log10(FDR) for visualization
              
              Sorted by interaction_score (descending)
    
    Raises
    ------
    ValueError
        If by_celltype=True but adata is not provided, or if required data
        is missing from adata or lr_adata.
        
    ImportError
        If required helper functions are not available.
    
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
    
    
    See Also
    --------
    prepareLRInteraction : Prepare LR scores (prerequisite for this function)
    
    """
    # Validate inputs
    if by_celltype and adata is None:
        raise ValueError(
            "Parameter 'adata' must be provided when by_celltype=True. "
            "adata should contain gene expression and cell type annotations."
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

    # Accept a cytome source for the expression adata: only the ligand and
    # receptor genes are needed downstream (the cell-type specificity step
    # subsets to them before COSG), so stream just that subset.
    if adata is not None and not isinstance(adata, ad.AnnData):
        lr_genes = list(pd.unique(pd.Series(
            [g for name in lr_adata.var_names for g in name.split('::')]
        )))
        adata = _ensure_expression_anndata(adata, genes=lr_genes)
    
    # Import helper functions
    try:
        from . import _utils
    except ImportError:
        raise ImportError(
            "Could not import _utils module. Please ensure all LARIS "
            "dependencies are properly installed."
        )
    
    # =========================================================================
    # STEP 1: Calculate LARIS Spatial Specificity Scores
    # =========================================================================
    print("\n--- Step 1: Calculating spatial specificity scores ---")
    print(f"  - Using {n_nearest_neighbors} nearest neighbors")
    print(f"  - Regularization parameter μ = {mu}")
    print(f"  - Random repeats: {n_repeats}")
    
    # Build spatial adjacency matrix
    cellxcell = _utils._build_adjacency_matrix(
        lr_adata,
        use_rep=use_rep,
        n_nearest_neighbors=n_nearest_neighbors,
        sigma=sigma
    )
    
    # Calculate observed spatial correlation
    genexcell = lr_adata.X.T
    order1 = genexcell @ cellxcell.T
    gsp = _utils._rowwise_cosine_similarity(genexcell, order1)
    
    # Generate random background
    print("  - Generating random permutations...")
    random_gsp_list = _utils._generate_random_background(
        lr_adata, cellxcell, genexcell,
        n_nearest_neighbors=n_nearest_neighbors,
        n_repeats=n_repeats,
        random_seed=random_seed
    )
    
    random_gsp = np.mean(random_gsp_list, axis=0)
    
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
    gsp_score_for_ranking = lr_var['LR_SpatialSpecificity'].values.copy()
    
    # Penalize LR pairs with low cell counts. A float threshold in (0, 1) is
    # interpreted as a fraction of the number of cells (sklearn-style),
    # a value >= 1 as an absolute cell count.
    if 0 < n_cells_expressed_threshold < 1:
        cells_expressed_cutoff = int(np.ceil(
            n_cells_expressed_threshold * lr_adata.shape[0]
        ))
    else:
        cells_expressed_cutoff = int(n_cells_expressed_threshold)
    min_score = np.min(gsp_score_for_ranking)
    low_count_mask = n_cells_expressed < cells_expressed_cutoff
    gsp_score_for_ranking[low_count_mask] = min_score - 0.001
    
    # Select top N LR pairs
    top_indices = _utils._select_top_n(gsp_score_for_ranking, n_top_lr)
    spag_list = lr_var.index.values[top_indices]
    spag_list_ligand = [name.split('::')[0] for name in spag_list]
    spag_list_receptor = [name.split('::')[1] for name in spag_list]
    spag_list_score = gsp_score_for_ranking[top_indices]
    
    # Create results DataFrame
    laris_lr = pd.DataFrame({
        'ligand': spag_list_ligand,
        'receptor': spag_list_receptor,
        'score': spag_list_score
    })
    laris_lr.index = [f"{l}::{r}" for l, r in zip(spag_list_ligand, spag_list_receptor)]
    laris_lr['Rank'] = np.arange(len(laris_lr))
    
    print(f"  ✓ Identified {len(laris_lr)} top spatially-specific LR pairs")
    print(f"  - Score range: [{laris_lr['score'].min():.4f}, "
          f"{laris_lr['score'].max():.4f}]")
    
    # =========================================================================
    # STEP 2: Calculate Cell Type-Specific Interactions (Optional)
    # =========================================================================
    if by_celltype:
        print("\n" + "="*70)
        print("CELL TYPE-SPECIFIC ANALYSIS")
        print("="*70)
        
        if groupby not in adata.obs:
            raise ValueError(
                f"Cell type column '{groupby}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        
        n_cell_types = adata.obs[groupby].nunique()
        print(f"\nAnalyzing {n_cell_types} cell types from '{groupby}'")
        print(f"Cell types: {sorted(adata.obs[groupby].unique())[:10]}"
              f"{'...' if n_cell_types > 10 else ''}")
        
        celltype_results = _utils._calculate_laris_score_by_celltype(
            adata=adata,
            lr_adata=lr_adata,
            laris_lr=laris_lr,
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
            rescale=rescale
        )
        
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



def compareLARIS(
    results: dict,
    conditionMap: dict,
    referenceCondition: str,
    sampleToSubject: Optional[dict] = None,
    scoreCol: str = 'interaction_score',
    minSubjectsObserved: int = 2,
    minCellTypePairs: int = 3,
    pseudocount: float = 1e-6,
    fdrMethod: str = 'fdr_bh',
    level: str = 'both',
):
    """
    Compare LARIS cell-cell communication results across experimental conditions.

    Method: each sample's interaction scores are log-transformed and
    centred on the sample's median log-score (removing any per-sample
    multiplicative factor - the step-3.5 rescaling, sequencing depth,
    batch - exactly); all rows belonging to a subject (cell-type pairs
    and technical-replicate slices) are averaged to one value per
    (subject, LR pair), making the subject the unit of inference; and
    conditions are compared with an empirical-Bayes moderated t-test
    (limma-style variance shrinkage across LR pairs), which keeps power
    at the 3-5 subjects per condition typical of spatial cohorts.
    Calibration was verified by simulation (3-6% false-positive rate at
    nominal 5% under every null, including condition-confounded rescale
    drift) and on real multi-condition datasets; see
    docs/discussion for the validation study.

    Parameters
    ----------
    results : dict
        {sample_name: celltype_results_df} from runLARIS(by_celltype=True).
    conditionMap : dict
        {sample_name: condition_label} mapping each sample to its condition.
    referenceCondition : str
        The reference condition for pairwise comparisons (e.g., 'Healthy').
    sampleToSubject : dict, optional
        {sample_name: subject_id} mapping samples to biological replicates.
        Required when samples include technical replicates (e.g., multiple
        slices per mouse): slices of one subject are averaged before
        testing. If None, each sample is treated as an independent subject
        (a warning is emitted).
    scoreCol : str
        Column name for interaction scores in celltype_results. Default
        'interaction_score'. Scores may be rescaled or raw - the
        per-sample centring makes the test invariant to any per-sample
        scale factor.
    minSubjectsObserved : int
        Minimum subjects per condition for a result to be flagged
        estimable (Level 2; Level-2 FDR is computed over estimable rows
        only). Tests need at least 2 subjects per condition regardless.
    minCellTypePairs : int
        Retired (the aggregated design has no per-pair model to gate);
        accepted for backward compatibility and ignored.
    pseudocount : float
        Added to raw scores for the descriptive log2FC columns.
    fdrMethod : str
        Method for multiple testing correction (default 'fdr_bh').
        Level 1 is corrected globally; Level 2 within each LR pair.
    level : {'both', 'lr', 'triple'}, default 'both'
        'lr' computes only the per-LR-pair table (fast - recommended as a
        first pass on large cohorts), 'triple' only the per-triple table,
        'both' computes both.

    Returns
    -------
    lr_comparison : pd.DataFrame
        Level 1 - one row per (LR pair x comparison). Columns include
        interaction_name, comparison, log_diff (difference of centred
        log scores, alt minus reference), pvalue, pvalue_fdr, log2fc.
    triple_comparison : pd.DataFrame
        Level 2 - one row per (sender, receiver, LR pair x comparison),
        with the same statistics plus estimable and per-condition
        descriptive means.
    """
    return compare_laris_internal(
        results=results,
        conditionMap=conditionMap,
        referenceCondition=referenceCondition,
        sampleToSubject=sampleToSubject,
        scoreCol=scoreCol,
        minSubjectsObserved=minSubjectsObserved,
        minCellTypePairs=minCellTypePairs,
        pseudocount=pseudocount,
        fdrMethod=fdrMethod,
        level=level,
    )


# Define public API for the tools module
__all__ = [
    'prepareLRInteraction',
    'runLARIS',
    'compareLARIS',
    'readCytome',
]
