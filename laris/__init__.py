"""
LARIS - Ligand And Receptor Interaction analysis in Spatial transcriptomic data

A Python package for analyzing ligand-receptor interactions in spatial transcriptomics data.

LARIS provides tools for:
- Calculating ligand-receptor interaction scores
- Identifying spatially-specific ligand-receptor interactions
- Inferring cell-cell communications

The package follows a modular structure:
- laris.tl: Core analytical tools
- laris.pp: Preprocessing functions
- laris.pl: Plotting and visualization functions

Example usage:
    >>> import laris as la
    >>> import scanpy as sc
    >>> 
    >>> # Calculate ligand-receptor interaction scores
    >>> lr_adata = la.tl.prepareLRInteraction(adata, lr_df)
    >>> 
    >>> # Identify spatially-specific LR interactions
    >>> laris_results, celltype_results = la.tl.runLARIS(lr_adata, adata)
    >>> 
    >>> # View top interactions
    >>> print(laris_results.head(10))
    >>> print(celltype_results.head(10))

Authors: Min Dai, Tivadar Török, Dawei Sun, et al.
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Min Dai, Tivadar Török, Dawei Sun, et al."

# Import submodules
from . import tools as tl
from . import preprocessing as pp
from . import plotting as pl

# Define public API
__all__ = [
    'tl',
    'pp', 
    'pl',
]
