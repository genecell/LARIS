"""
LARIS - Ligand And Receptor Interaction analysis in Spatial transcriptomic data

A Python package for analyzing ligand-receptor interactions in spatial transcriptomics data.

LARIS provides tools for:
- Calculating ligand-receptor interaction scores
- Identifying spatially-specific ligand-receptor interactions
- Inferring cell-cell communications

The package follows a modular structure:
- laris.tl: Core analysis (prepareLRInteraction, runLARIS, compareLARIS)
- laris.pp: Readers and preprocessing (readCytome, spatialOffsetMultisample)
- laris.pl: Plotting and visualization
- laris.datasets: Bundled ligand-receptor databases

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
"""

__version__ = "0.11.0"
__author__ = "Min Dai, Tivadar Török, Dawei Sun, et al."

# Import submodules
from . import tools as tl
from . import preprocessing as pp
from . import plotting as pl
from . import datasets

# Define public API
__all__ = [
    'tl',
    'pp',
    'pl',
    'datasets',
]
