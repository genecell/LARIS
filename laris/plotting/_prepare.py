"""Data preparation helpers for LARIS plots."""

import math
import itertools
import textwrap
import warnings
from collections import OrderedDict
from typing import Optional, Union, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy
from scipy.sparse import csr_matrix, issparse, hstack

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects
from matplotlib import cm
from matplotlib import colors, colorbar
from matplotlib import colors as colors_mod
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
import networkx as nx

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ._utils import _log_message
from ..preprocessing._io import _ensure_expression_anndata, _ensure_lr_anndata
from .._compat import _UNSET, resolve_data_arg

def prepareDotPlotAdata(
    lr_data=_UNSET,
    data=_UNSET,
    verbosity: int = 2,
    lr_adata=_UNSET,
    adata=_UNSET
) -> ad.AnnData:
    """
    Prepare combined AnnData for dot plot visualizations.
    
    Concatenates LR interaction scores with original gene expression data 
    horizontally to create a unified AnnData object for plotting.
    
    Parameters
    ----------
    lr_adata : anndata.AnnData
        AnnData object containing LR interaction scores
        
    adata : anndata.AnnData, str or cytome.CytomeDataset
        Expression object; a ``.cytome``/``.db`` path or an open cytome
        dataset is read automatically.
        Original AnnData object containing gene expression data
        
    verbosity : int, default=2
        Verbosity level
    
    Returns
    -------
    adata_dotplot : anndata.AnnData
        Combined AnnData object
    
    Examples
    --------
    >>> adata_combined = la.pl.prepareDotPlotAdata(lr_adata, adata)
    >>> la.pl.plotLRDotPlot(adata_combined, interactions, groupby='cell_type')
    """
    lr_adata = _ensure_lr_anndata(
        resolve_data_arg(lr_data, 'prepareDotPlotAdata', canonical='lr_data',
                         lr_adata=lr_adata))
    adata = resolve_data_arg(data, 'prepareDotPlotAdata', canonical='data',
                             adata=adata)

    # Ensure sparse format
    # Accept a cytome source for the expression object: only the genes that
    # end up in the dot plot are needed, but the combined object keeps every
    # gene, so read the full matrix here.
    if not isinstance(adata, ad.AnnData):
        adata = _ensure_expression_anndata(adata)

    lr_X = lr_adata.X
    if not issparse(lr_X):
        lr_X = csr_matrix(lr_X)
    elif not isinstance(lr_X, csr_matrix):
        lr_X = lr_X.tocsr()

    adata_X = adata.X
    if not issparse(adata_X):
        adata_X = csr_matrix(adata_X)
    elif not isinstance(adata_X, csr_matrix):
        adata_X = adata_X.tocsr()

    # Concatenate horizontally
    combined_X = hstack([lr_X, adata_X], format='csr')
    _log_message("Combined matrices in sparse format.", 3, verbosity, 'debug')

    # Combine variable names
    combined_var_names = np.concatenate(
        [lr_adata.var_names, adata.var_names], axis=0
    )

    # Create new AnnData
    adata_dotplot = sc.AnnData(X=combined_X)
    adata_dotplot.var_names = combined_var_names.copy()
    adata_dotplot.obs = adata.obs.copy()
    adata_dotplot.obsm = adata.obsm.copy()

    _log_message(
        f"Created combined AnnData: {adata_dotplot.shape[0]} cells × "
        f"{adata_dotplot.shape[1]} features",
        2, verbosity, 'info'
    )

    return adata_dotplot


def _compute_max_fraction(
    adata: ad.AnnData,
    genes: List[str],
    groupby: str
) -> float:
    """
    Compute maximum expression fraction across groups.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing expression data
    genes : list of str
        List of gene names to check
    groupby : str
        Column in adata.obs to group by
    
    Returns
    -------
    max_frac : float
        Maximum fraction of expressing cells
    """
    max_frac = 0

    # Validate genes
    valid_genes = [g for g in genes if g in adata.var_names]
    if len(valid_genes) != len(genes):
        missing = set(genes) - set(valid_genes)
        warnings.warn(f"Genes not found in adata: {missing}")

    if not valid_genes:
        return 0.0

    groups = adata.obs[groupby].unique()

    for gene in valid_genes:
        for group in groups:
            subset = adata[adata.obs[groupby] == group]
            n_cells = subset.n_obs
            
            if n_cells == 0:
                frac = 0
            else:
                n_expressing = (subset[:, gene].X > 0).sum()
                frac = n_expressing / n_cells

            max_frac = max(max_frac, frac)

    return max_frac
