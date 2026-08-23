"""Ligand-receptor interaction score preparation (prepareLRInteraction)."""

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
from ..preprocessing._io import _ensure_expression_anndata

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
