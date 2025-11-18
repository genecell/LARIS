"""
LARIS Tools Module (laris.tl)

Core analytical tools for ligand-receptor interaction analysis in spatial transcriptomics data.

This module contains public functions for:
- Preparing ligand-receptor integration scores with spatial diffusion
- Running the LARIS algorithm to identify spatially-specific LR interactions
- Computing cell type-specific interaction scores

All functions follow the AnnData structure commonly used in single-cell analysis.

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
    
    Notes
    -----
    The function performs the following steps:
    1. Creates a spatial neighborhood graph using k-NN
    2. Applies distance-based weighting to the graph
    3. Diffuses gene expression through the spatial neighborhood
    4. Computes element-wise multiplication of ligand and receptor scores
    5. Filters interactions to only include cells where L or R is expressed
    
    The spatial diffusion captures local microenvironment effects, where ligands
    produced by nearby cells can interact with receptors on the focal cell.
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
    # --- Cell Type & P-Value Parameters ---
    by_celltype: bool = True,
    groupby: str = 'CellTypes',
    use_rep_spatial: str = 'X_spatial',
    number_nearest_neighbors: int = 10,
    mu_celltype: float = 100,
    expressed_pct_celltype: float = 0.1,
    remove_lowly_expressed_celltype: bool = True,
    mask_threshold: float = 1e-6,
    calculate_pvalues: bool = True,
    layer_celltype: str = None,
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
    
    LARIS (Ligand And Receptor Interaction Spatial analysis) identifies LR pairs
    that show spatial specificity by comparing observed spatial correlation patterns
    with randomized null distributions. Higher scores indicate stronger spatial
    co-localization of the ligand-receptor interaction.
    
    When `by_celltype=True`, this function also computes cell type-specific interaction
    scores that integrate spatial specificity, cell type specificity, and spatial
    co-localization of sender-receiver cell type pairs, with optional
    statistical significance testing.
    
    Parameters
    ----------
    lr_adata : AnnData
        AnnData object containing ligand-receptor interaction scores (from 
        `prepareLRInteraction()`).
    adata : AnnData, optional
        Original annotated data matrix with gene expression and spatial information.
        Required when `by_celltype=True`.
    use_rep : str, default='X_spatial'
        Key in lr_adata.obsm for representation to use (typically spatial coordinates).
    n_nearest_neighbors : int, default=10
        Number of nearest neighbors for spatial graph construction (LARIS step 1).
    random_seed : int, default=27
        Random seed for reproducibility.
    n_repeats : int, default=3
        Number of random permutations for null distribution.
    mu : float, default=1
        Regularization parameter. Higher values penalize random background more strongly.
    sigma : float, default=100
        Bandwidth for distance kernel in adjacency matrix.
    remove_lowly_expressed : bool, default=True
        Whether to filter lowly expressed LR pairs.
    expressed_pct : float, default=0.1
        Minimum fraction of cells that must express an LR pair (if remove_lowly_expressed=True).
    n_cells_expressed_threshold : int, default=100
        Minimum number of cells expressing an LR pair for ranking.
    n_top_lr : int, default=4000
        Number of top-ranked LR pairs to return.
    by_celltype : bool, default=True
        Whether to compute cell type-specific interaction scores. If True, `adata` must be provided.
    
    Parameters (only used if by_celltype=True):
    --------------------------------------------
    groupby : str, default='CellTypes'
        Column in adata.obs defining cell type groups.
    use_rep_spatial : str, default='X_spatial'
        Key in adata.obsm for spatial coordinates.
    number_nearest_neighbors : int, default=10
        Number of neighbors for spatial graph in cell type neighborhood analysis.
    mu_celltype : float, default=100
        Regularization parameter for cell type specificity calculation (COSG).
    expressed_pct_celltype : float, default=0.1
        Minimum expression fraction for cell type analysis.
    remove_lowly_expressed_celltype : bool, default=True
        Whether to filter lowly expressed genes in cell type analysis.
    mask_threshold : float, default=1e-6
        Threshold for masking low values in cell type analysis.
    calculate_pvalues : bool, default=True
        Whether to calculate p-values via permutation testing.
    layer_celltype : str, optional
        Layer in `adata` to use for p-value calculation (if None, uses adata.X).
    n_neighbors_permutation : int, default=30
        Number of nearest neighbors for creating the background interaction set
        for permutation testing.
    n_permutations : int, default=1000
        Number of permutations to run for statistical testing.
    chunk_size : int, default=50000
        Number of interactions to process at once during permutation testing.
    prefilter_fdr : bool, default=True
        If True, only test interactions with score > prefilter_threshold.
    prefilter_threshold : float, default=0.0
        Minimum interaction score threshold for FDR testing.
    score_threshold : float, default=1e-6
        Minimum threshold for interaction scores. Values below this are set to 0.0.
    spatial_weight : float, default=1.0
        Exponent applied to the spatial specificity score from runLARIS().
    use_conditional_pvalue : bool, default=False
        If True, use conditional p-value logic to handle zero-inflated nulls.
        
    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
        If by_celltype=False:
            DataFrame with columns:
            - 'ligand': Ligand gene name
            - 'receptor': Receptor gene name
            - 'score': LARIS spatial specificity score
            - 'Rank': Rank of the LR pair (0 = highest score)
            Index is in format "ligand::receptor"
        
        If by_celltype=True:
            Tuple of (laris_lr, celltype_results) where:
            - laris_lr: DataFrame as described above
            - celltype_results: DataFrame with cell type-specific scores and p-values
              (see `_calculate_laris_score_by_celltype` docstring for details).
    
    Raises
    ------
    ValueError
        If by_celltype=True but adata is not provided.
    ImportError
        If helper functions (`_build_adjacency_matrix`, etc.) are not found.
    """
    # Validate inputs
    if by_celltype and adata is None:
        raise ValueError("adata must be provided when by_celltype=True")
    
    # Check for helper functions (assuming they are in _utils)
    # This is a placeholder; you'll need your actual import system
    try:
        from . import _utils 
    except ImportError:
        print("Warning: Could not import _utils. Assuming helper functions are available.")
        # As a fallback, create a dummy object if not found, to avoid crashing
        # This is NOT a permanent fix, just for making the code runnable
        if '_utils' not in locals():
            class DummyUtils:
                def _build_adjacency_matrix(self, *args, **kwargs):
                    raise ImportError("Missing _utils._build_adjacency_matrix")
                def _generate_random_background(self, *args, **kwargs):
                    raise ImportError("Missing _utils._generate_random_background")
                def _rowwise_cosine_similarity(self, *args, **kwargs):
                    raise ImportError("Missing _utils._rowwise_cosine_similarity")
                def _select_top_n(self, *args, **kwargs):
                    raise ImportError("Missing _utils._select_top_n")
            _utils = DummyUtils()
            
    # Step 1: Calculate LARIS spatial specificity scores
    cellxcell = _utils._build_adjacency_matrix(
        lr_adata,
        use_rep=use_rep,
        n_nearest_neighbors=n_nearest_neighbors,
        sigma=sigma,
    )
    
    genexcell = lr_adata.X.T
    order1 = genexcell @ cellxcell.T   
    gsp = _utils._rowwise_cosine_similarity(genexcell, order1)

    # Build the random background by sampling
    random_gsp_list = _utils._generate_random_background(
        lr_adata, cellxcell, genexcell,
        n_nearest_neighbors=n_nearest_neighbors,
        n_repeats=n_repeats,
        random_seed=random_seed
    )
    
    random_gsp = np.mean(random_gsp_list, axis=0)

    lr_adata.var['LRSS_Target'] = np.array(gsp).ravel()
    lr_adata.var['LRSS_Random'] = np.array(random_gsp).ravel()
    
    gsp_score = gsp - mu * random_gsp
    gsp_score = np.array(gsp_score).ravel()

    if remove_lowly_expressed:
        lr_adata.var['LR_SpatialSpecificity'] = gsp_score
    else:
        lr_adata.var['LR_SpatialSpecificity'] = gsp_score

    # Calculate QC metrics
    if lr_adata.shape[1] < 500:
        sc.pp.calculate_qc_metrics(lr_adata, inplace=True, percent_top=[50, 100])
    else:
        sc.pp.calculate_qc_metrics(lr_adata, inplace=True)
        
    lr_var = lr_adata.var.sort_values(by='LR_SpatialSpecificity', ascending=False).copy()
    n_cells_expressed = lr_var['n_cells_by_counts'].values
    
    gsp_score_for_ranking = lr_var['LR_SpatialSpecificity'].values
    gsp_score_for_ranking[np.where(n_cells_expressed < n_cells_expressed_threshold)[0]] = np.min(gsp_score_for_ranking) - 0.001

    spag_list = lr_var.index.values[_utils._select_top_n(gsp_score_for_ranking, n_top_lr)]
    spag_list_ligand = [np.str_.split(i, '::')[0] for i in spag_list]
    spag_list_receptor = [np.str_.split(i, '::')[1] for i in spag_list]
    spag_list_score = gsp_score_for_ranking[_utils._select_top_n(gsp_score_for_ranking, n_top_lr)]
    
    laris_lr = pd.DataFrame({
        'ligand': spag_list_ligand, 
        'receptor': spag_list_receptor, 
        'score': spag_list_score
    })
    laris_lr.index = laris_lr['ligand'] + '::' + laris_lr['receptor']
    laris_lr['Rank'] = np.arange(laris_lr.shape[0])
    
    # Step 2: Calculate cell type-specific interactions if requested
    if by_celltype:
        # Use the imported function
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
            # --- Pass all new p-value parameters ---
            calculate_pvalues=calculate_pvalues,
            layer=layer_celltype,
            n_nearest_neighbors=n_neighbors_permutation, # Pass renamed parameter
            n_permutations=n_permutations,
            chunk_size=chunk_size,
            prefilter_fdr=prefilter_fdr,
            prefilter_threshold=prefilter_threshold,
            score_threshold=score_threshold,
            spatial_weight=spatial_weight,
            use_conditional_pvalue=use_conditional_pvalue
        )
        return laris_lr, celltype_results
    else:
        return laris_lr



# Define public API for the tools module
__all__ = [
    'prepareLRInteraction',
    'runLARIS',
]
