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


def prepareLRInteraction(
    adata: ad.AnnData,
    lr_df: pd.DataFrame,
    number_nearest_neighbors: int = 10,
    use_rep_spatial: str = 'X_spatial'
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
    X_spatial = adata.obsm[use_rep_spatial].copy()
    
    # Create the diffused ligand/receptor matrix
    cellxcell = kneighbors_graph(
        X_spatial, 
        n_neighbors=number_nearest_neighbors, 
        mode='distance', 
        include_self=True
    )
    cellxcell.data = 1 / np.exp(cellxcell.data / (np.mean(cellxcell.data) / 2))

    genexcell = adata.X.copy().T
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

    # Filter to only include cells where ligand or receptor is expressed
    ligand_idx = sorter[np.searchsorted(adata.var_names, lr_df['ligand'], sorter=sorter)]
    receptor_idx = sorter[np.searchsorted(adata.var_names, lr_df['receptor'], sorter=sorter)]

    ligand_mask = adata.X[:, ligand_idx] != 0   # True where ligand is expressed
    receptor_mask = adata.X[:, receptor_idx] != 0  # True where receptor is expressed

    non_zero_mask = ligand_mask.maximum(receptor_mask)  # elementwise OR for sparse matrices
    lr_adata.X = lr_adata.X.multiply(non_zero_mask)
    
    return lr_adata

def runLARIS(
    lr_adata: ad.AnnData,
    adata: Optional[ad.AnnData] = None,
    use_rep: str = 'X_spatial',
    n_nearest_neighbors: int = 10,
    random_seed: int = 27,
    n_repeats: int = 3,
    mu: float = 1,
    sigma: float = 100,
    remove_lowly_expressed: bool = True,
    expressed_pct: float = 0.1,
    n_cells_expressed_threshold: int = 100,
    n_top_lr: int = 4000,
    # Cell Type & Statistical Testing Parameters
    by_celltype: bool = True,
    groupby: str = 'CellTypes',
    use_rep_spatial: str = 'X_spatial',
    number_nearest_neighbors: int = 10,
    mu_celltype: float = 100,
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
    spatial_weight: float = 1.0,
    use_conditional_pvalue: bool = False
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
        
    n_nearest_neighbors : int, default=10
        Number of spatial neighbors for building the adjacency matrix in the
        spatial specificity analysis. Larger values capture broader spatial patterns.
        
    random_seed : int, default=27
        Random seed for reproducibility of permutation tests.
        
    n_repeats : int, default=3
        Number of random permutations to generate the null distribution for
        spatial specificity scoring. More repeats give more stable estimates.
        
    mu : float, default=1
        Regularization parameter for spatial specificity. Higher values penalize
        interactions that look similar to random background more strongly.
        
    sigma : float, default=100
        Bandwidth parameter for Gaussian distance kernel in adjacency matrix.
        Controls how quickly spatial weights decay with distance.
        
    remove_lowly_expressed : bool, default=True
        Whether to filter out LR pairs with low expression before ranking.
        
    expressed_pct : float, default=0.1
        Minimum fraction of cells expressing an LR pair (if remove_lowly_expressed=True).
        
    n_cells_expressed_threshold : int, default=100
        Minimum number of cells expressing an LR pair for it to be ranked.
        Pairs below this threshold receive a penalty in ranking.
        
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
        
    number_nearest_neighbors : int, default=10
        Number of spatial neighbors for cell type co-localization analysis.
        
    mu_celltype : float, default=100
        Regularization parameter for COSG cell type specificity calculation.
        Higher values more strongly penalize broadly expressed genes.
        
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
        
    spatial_weight : float, default=1.0
        Exponent applied to spatial specificity scores. Controls influence on
        final interaction scores:
        - 0: Ignore spatial specificity
        - 1: Linear influence (default)
        - >1: Stronger emphasis on spatial specificity
        - <1: Weaker emphasis on spatial specificity
        
    use_conditional_pvalue : bool, default=False
        Use conditional p-value calculation for zero-inflated data. **Recommended
        for sparse datasets**. When True:
        - Interactions with score=0 get p-value=1.0
        - Non-zero scores compared only to non-zero background
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
    
    # Calculate QC metrics
    if lr_adata.shape[1] < 500:
        sc.pp.calculate_qc_metrics(lr_adata, inplace=True, percent_top=[50, 100])
    else:
        sc.pp.calculate_qc_metrics(lr_adata, inplace=True)
    
    # Rank LR pairs by spatial specificity
    lr_var = lr_adata.var.sort_values(
        by='LR_SpatialSpecificity', 
        ascending=False
    ).copy()
    
    n_cells_expressed = lr_var['n_cells_by_counts'].values
    gsp_score_for_ranking = lr_var['LR_SpatialSpecificity'].values
    
    # Penalize LR pairs with low cell counts
    min_score = np.min(gsp_score_for_ranking)
    low_count_mask = n_cells_expressed < n_cells_expressed_threshold
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
            use_conditional_pvalue=use_conditional_pvalue
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



# Define public API for the tools module
__all__ = [
    'prepareLRInteraction',
    'runLARIS',
]
