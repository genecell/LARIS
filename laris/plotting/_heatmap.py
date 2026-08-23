"""Cell-cell communication heatmaps."""

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

from ._colors import _get_cmap, pos_cmap, _resolve_cell_type_colors
from ._utils import _log_message, _save_figure, _compute_bubble_sizes_log10, _create_pvalue_legend_log10, _create_edge_thickness_legend

def plotCCCHeatmap(
    laris_results: pd.DataFrame,
    cmap: Union[str, colors.Colormap] = 'Purples',
    n_top: int = 3000,
    figsize: Tuple[float, float] = (6, 5),
    axis_label_fontsize: int = 16,
    tick_fontsize: int = 12,
    cbar_label_fontsize: int = 16,
    cbar_tick_fontsize: int = 12,
    filter_significant: bool = True,
    p_value_col: str = 'p_value_fdr',
    threshold: float = 0.05,
    show_borders: bool = True,
    cluster: bool = False,
    filter_by_interaction_score: bool = True,
    threshold_interaction_score: float = 0.01,
    save: Optional[str] = None,
    verbosity: int = 2,
    return_fig: bool = False
) -> Optional[plt.Figure]:
    """
    Create a heatmap showing the number of cell-cell communication interactions.
    
    This function visualizes the frequency of interactions between sending and 
    receiving cell types as a heatmap. It ensures the row and column orders
    are identical to maintain the self-interaction diagonal.
    
    Parameters
    ----------
    laris_results : pd.DataFrame
        DataFrame containing LARIS results with columns:
        - 'sender' : str, cell type sending the signal
        - 'receiver' : str, cell type receiving the signal
        - 'significant' : bool (optional), fallback significance flag
        - p_value_col : str (optional), column name for p-values
        
    cmap : str or matplotlib.colors.Colormap, default='Purples'
        Colormap to use for the heatmap (e.g., 'viridis', 'RdBu_r')
        
    n_top : int, default=3000
        Number of top-ranked interactions to include if no filters are applied
        
    figsize : tuple, default=(6, 5)
        Figure size as (width, height) in inches
        
    axis_label_fontsize : int, default=16
        Font size for axis labels
        
    tick_fontsize : int, default=12
        Font size for tick labels
        
    cbar_label_fontsize : int, default=16
        Font size for colorbar label
        
    cbar_tick_fontsize : int, default=12
        Font size for colorbar tick labels
        
    filter_significant : bool, default=True
        If True, filters interactions based on p_value_col and threshold
        
    p_value_col : str, default='p_value_fdr'
        Column name to use for p-value filtering
        
    threshold : float, default=0.05
        P-value cutoff for filtering
        
    show_borders : bool, default=True
        If True, draws light grey border lines between heatmap cells
        
    cluster : bool, default=False
        If True, performs hierarchical clustering on cell types
        
    filter_by_interaction_score : bool, default=True
        If True, filters by interaction_score > threshold_interaction_score
        
    threshold_interaction_score : float, default=0.01
        Cutoff for interaction score filtering
        
    save : str, optional
        Path to save the figure (e.g., 'heatmap.pdf'). If None, figure is not saved
        
    verbosity : int, default=2
        Verbosity level (0=silent, 1=errors, 2=warnings/info, 3=debug)
        
    return_fig : bool, default=False
        If True, return the figure object instead of just displaying
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure object if return_fig=True, otherwise None
    
    Examples
    --------
    >>> la.pl.plotCCCHeatmap(
    ...     laris_results,
    ...     filter_significant=True,
    ...     cluster=True,
    ...     save='heatmap.pdf'
    ... )
    """
    laris_results_subset = laris_results.copy()
    did_filter = False

    # Apply significance filter
    if filter_significant:
        if p_value_col in laris_results_subset.columns:
            laris_results_subset = laris_results_subset[
                laris_results_subset[p_value_col] < threshold
            ]
            did_filter = True
            _log_message(f"Filtered by {p_value_col} < {threshold}", 3, verbosity, 'debug')
        elif 'significant' in laris_results_subset.columns:
            laris_results_subset = laris_results_subset[
                laris_results_subset['significant']
            ]
            did_filter = True
            _log_message("Filtered by 'significant' column", 3, verbosity, 'debug')
        else:
            _log_message(
                f"'filter_significant' is True but '{p_value_col}' and 'significant' "
                "columns are missing. Skipping significance filter.",
                2, verbosity, 'warning'
            )

    # Apply interaction score filter
    if filter_by_interaction_score:
        score_col = 'interaction_score'
        if score_col in laris_results_subset.columns:
            laris_results_subset = laris_results_subset[
                laris_results_subset[score_col] > threshold_interaction_score
            ]
            did_filter = True
            _log_message(
                f"Filtered by {score_col} > {threshold_interaction_score}", 
                3, verbosity, 'debug'
            )
        else:
            _log_message(
                f"'filter_by_interaction_score' is True but '{score_col}' column "
                "is missing. Skipping score filter.",
                2, verbosity, 'warning'
            )

    # Fallback to n_top
    if not did_filter:
        _log_message(
            f"No filters applied. Using top {n_top} interactions.",
            2, verbosity, 'info'
        )
        laris_results_subset = laris_results_subset.iloc[:n_top]

    if laris_results_subset.empty:
        _log_message(
            "No interactions found matching criteria. Cannot plot heatmap.",
            1, verbosity, 'error'
        )
        return None

    _log_message(
        f"Plotting {len(laris_results_subset)} interactions",
        3, verbosity, 'debug'
    )

    # Create pivot table
    heatmap_data_raw = laris_results_subset.pivot_table(
        index='sender',
        columns='receiver',
        aggfunc='size',
        fill_value=0
    )

    # Ensure square matrix
    all_cell_types = sorted(
        list(set(heatmap_data_raw.index) | set(heatmap_data_raw.columns))
    )
    
    heatmap_data = heatmap_data_raw.reindex(
        index=all_cell_types,
        columns=all_cell_types,
        fill_value=0
    )

    # Apply clustering if requested
    if cluster:
        if SCIPY_AVAILABLE:
            try:
                row_linkage = linkage(
                    pdist(heatmap_data, metric='euclidean'), 
                    method='average'
                )
                new_order_indices = leaves_list(row_linkage)
                new_order_labels = heatmap_data.index[new_order_indices]
                
                heatmap_data = heatmap_data.reindex(
                    index=new_order_labels,
                    columns=new_order_labels
                )
                _log_message("Applied hierarchical clustering", 3, verbosity, 'debug')
            except Exception as e:
                _log_message(
                    f"Clustering failed: {e}. Using alphabetical order.",
                    2, verbosity, 'warning'
                )
        else:
            _log_message(
                "Clustering requires scipy. Using alphabetical order.",
                2, verbosity, 'warning'
            )

    # Set border parameters
    line_width_val = 0.5 if show_borders else 0
    line_color_val = 'lightgrey' if show_borders else 'none'

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = sns.heatmap(
        heatmap_data,
        cmap=cmap,
        annot=False,
        cbar=True,
        linewidths=line_width_val,
        linecolor=line_color_val,
        square=True
    )

    # Remove tick lines and grid lines from ticks
    ax.tick_params(axis='both', which='both', length=0)
    # Turn off grid lines that extend from ticks
    ax.grid(False)

    # Set labels
    plt.xlabel('Receiver', fontsize=axis_label_fontsize)
    plt.ylabel('Sender', fontsize=axis_label_fontsize)
    plt.xticks(fontsize=tick_fontsize, rotation=90)
    plt.yticks(fontsize=tick_fontsize)

    # Set colorbar properties
    cbar = ax.collections[0].colorbar
    cbar.set_label("Number of interactions", fontsize=cbar_label_fontsize)
    cbar.ax.tick_params(labelsize=cbar_tick_fontsize)

    plt.tight_layout()
    
    # Save figure
    _save_figure(fig, save, verbosity)
    
    plt.show()
    
    if return_fig:
        return fig
    return None
