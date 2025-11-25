"""
LARIS Plotting Module (laris.pl)

Visualization functions for LARIS analysis results.

This module provides comprehensive plotting functions for visualizing:
- Cell-cell communication (CCC) networks and heatmaps
- Ligand-receptor interaction patterns
- Spatial distributions of interactions
- Dot plots and faceted visualizations

Key Functions
-------------
plotCCCHeatmap : Create heatmap of cell-cell communication interactions
plotCCCNetwork : Plot interaction network for specific cell type
plotCCCNetworkCumulative : Plot cumulative interaction network across all cell types
plotCCCDotPlot : Create bubble/dot plot for selected cell type pairs
plotCCCDotPlotFacet : Create faceted dot plot by sending or receiving cell type
plotLRDotPlot : Create side-by-side dot plots for LR pairs, ligands, and receptors
plotCCCSpatial : Plot spatial distribution of interactions
prepareDotPlotAdata : Prepare combined AnnData object for dot plot visualizations

Examples
--------
>>> import laris as lr
>>> import matplotlib.pyplot as plt
>>> 
>>> # Plot heatmap of top interactions
>>> lr.pl.plotCCCHeatmap(
...     laris_results, 
...     n_top=1000,
...     filter_significant=True
... )
>>> 
>>> # Plot network for a specific cell type
>>> fig, ax = lr.pl.plotCCCNetwork(
...     laris_results,
...     cell_type_of_interest='B_cell',
...     interaction_direction='sending',
...     adata=adata,
...     return_fig=True
... )
>>> 
>>> # Create spatial plot of an interaction
>>> lr.pl.plotCCCSpatial(
...     lr_adata,
...     basis='X_spatial',
...     interaction='CXCL13::CXCR5',
...     cell_type='cell_type',
...     selected_cell_types=['B_cell', 'T_cell']
... )
"""

# Import required packages
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy
from matplotlib import cm
from matplotlib import colors, colorbar
from matplotlib.lines import Line2D
import math
import anndata as ad
import networkx as nx
import itertools
from matplotlib.patches import FancyArrowPatch
from collections import OrderedDict
from scipy.sparse import csr_matrix, issparse, hstack
from typing import Optional, Union, List, Tuple
import warnings
import textwrap
import matplotlib.patheffects as path_effects

# Import for clustering
try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================================
# CUSTOM COLORMAP
# ============================================================================

# Define custom colormap for interaction scores
cmap_own = cm.get_cmap('magma_r', 256)
newcolors = cmap_own(np.linspace(0, 0.75, 256))
Greys = cm.get_cmap('Greys_r', 256)
newcolors[:10, :] = Greys(np.linspace(0.8125, 0.8725, 10))
pos_cmap = colors.ListedColormap(newcolors)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _log_message(message: str, level: int, verbosity: int, msg_type: str = "info"):
    """
    Print message based on verbosity level.
    
    Parameters
    ----------
    message : str
        Message to print
    level : int
        Required verbosity level for this message (1=error, 2=warning/info, 3=debug)
    verbosity : int
        Current verbosity setting
    msg_type : str
        Type of message ('error', 'warning', 'info', 'debug')
    """
    if verbosity >= level:
        prefix = {
            'error': '✗ ERROR: ',
            'warning': '⚠ Warning: ',
            'info': '',
            'debug': '  [DEBUG] '
        }.get(msg_type, '')
        print(f"{prefix}{message}")


def _save_figure(fig, save: Optional[str], verbosity: int = 2):
    """
    Save figure to file if save path is provided.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    save : str or None
        File path to save figure. If None, figure is not saved.
    verbosity : int
        Verbosity level for messages
    """
    if save is not None:
        try:
            fig.savefig(save, bbox_inches='tight', dpi=300)
            _log_message(f"Figure saved to: {save}", 2, verbosity, 'info')
        except Exception as e:
            _log_message(f"Failed to save figure: {e}", 1, verbosity, 'error')


def _compute_bubble_sizes_log10(p_values: np.ndarray, 
                                 bubble_size: float,
                                 n_permutations: int = 1000) -> np.ndarray:
    """
    Compute bubble sizes based on -log10(p_value).
    
    This provides a more informative scaling than discrete categories,
    especially for permutation-based p-values where minimum p = 1/(n_perm+1).
    
    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values
    bubble_size : float
        Maximum bubble size for most significant p-values
    n_permutations : int
        Number of permutations used (determines minimum possible p-value)
        
    Returns
    -------
    np.ndarray
        Array of bubble sizes scaled by -log10(p_value)
    """
    # Minimum p-value from permutation test
    min_p = 1.0 / (n_permutations + 1)
    
    # Clip p-values to avoid log(0)
    p_clipped = np.clip(p_values, min_p, 1.0)
    
    # Calculate -log10(p)
    neg_log10_p = -np.log10(p_clipped)
    
    # Normalize to [0, 1] range based on possible range
    # Max -log10(p) occurs at min_p
    max_neg_log10 = -np.log10(min_p)  # e.g., 3.0 for 1000 permutations
    min_neg_log10 = 0  # -log10(1) = 0
    
    # Normalized values
    normalized = (neg_log10_p - min_neg_log10) / (max_neg_log10 - min_neg_log10)
    
    # Scale to bubble size (minimum size is 10% of max for p=1)
    sizes = bubble_size * (0.1 + 0.9 * normalized)
    
    return sizes


def _create_pvalue_legend_log10(ax, bubble_size: float, 
                                 n_permutations: int = 1000,
                                 loc: str = 'upper left',
                                 bbox_to_anchor: tuple = (1.05, 1.0),
                                 frameon: bool = False,
                                 title_fontsize: int = 16):
    """
    Create legend for -log10(p_value) based bubble sizes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add legend to
    bubble_size : float
        Maximum bubble size
    n_permutations : int
        Number of permutations (for min p-value calculation)
    loc : str
        Legend location
    bbox_to_anchor : tuple
        Legend anchor position
    frameon : bool
        Whether to draw frame around legend
    title_fontsize : int
        Font size for legend title
        
    Returns
    -------
    legend : matplotlib.legend.Legend
        The created legend object
    """
    min_p = 1.0 / (n_permutations + 1)
    max_neg_log10 = -np.log10(min_p)
    
    # Create representative p-values for legend
    p_values_legend = [0.001, 0.01, 0.05, 0.1, 1.0]
    
    # Filter to only show achievable p-values
    p_values_legend = [p for p in p_values_legend if p >= min_p]
    if min_p not in p_values_legend and min_p < 0.001:
        p_values_legend = [min_p] + p_values_legend
    
    legend_handles = []
    for p in p_values_legend:
        neg_log10_p = -np.log10(p)
        normalized = neg_log10_p / max_neg_log10
        size = bubble_size * (0.1 + 0.9 * normalized)
        
        # Format label with italic P
        if p < 0.001:
            label = f'$\it{{P}}$ ≤ {p:.0e}'
        elif p < 0.01:
            label = f'$\it{{P}}$ = {p:.3f}'
        else:
            label = f'$\it{{P}}$ = {p:.2f}'
        
        handle = Line2D([0], [0], marker='o', color='w', label=label,
                       markerfacecolor='gray', markersize=np.sqrt(size),
                       markeredgecolor='black', markeredgewidth=0.5)
        legend_handles.append(handle)
    
    # Use italic P in title
    legend = ax.legend(handles=legend_handles, title="$\it{P}$ value", 
                      loc=loc, bbox_to_anchor=bbox_to_anchor, 
                      frameon=frameon, framealpha=0.9,
                      labelspacing=1.2,  # Add more space between labels
                      handletextpad=1.5)  # Add more space between marker and text
    
    # Set title font size
    legend.get_title().set_fontsize(title_fontsize)
    
    return legend


def _create_edge_thickness_legend(ax, edge_values: list, edge_widths: list,
                                   title: str = "Edge Thickness",
                                   loc: str = 'upper left',
                                   bbox_to_anchor: tuple = (1.05, 1.0)):
    """
    Create legend for edge thickness in network plots.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add legend to
    edge_values : list
        Representative values for the legend
    edge_widths : list
        Corresponding line widths
    title : str
        Legend title
    loc : str
        Legend location
    bbox_to_anchor : tuple
        Legend anchor position
        
    Returns
    -------
    legend : matplotlib.legend.Legend
        The created legend object
    """
    legend_handles = []
    for val, width in zip(edge_values, edge_widths):
        if val >= 1:
            label = f'{val:.1f}'
        else:
            label = f'{val:.2f}'
        
        handle = Line2D([0], [0], color='gray', linewidth=width, label=label)
        legend_handles.append(handle)
    
    legend = ax.legend(handles=legend_handles, title=title,
                      loc=loc, bbox_to_anchor=bbox_to_anchor,
                      frameon=False, framealpha=0.9,
                      handletextpad=2.0,  # More space between stroke and label
                      labelspacing=1.0)
    
    return legend


# ============================================================================
# HEATMAP FUNCTIONS
# ============================================================================

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


# ============================================================================
# NETWORK VISUALIZATION FUNCTIONS
# ============================================================================

def plotCCCNetwork(
    laris_results: pd.DataFrame,
    cell_type_of_interest: str,
    interaction_direction: str = "sending",
    adata: Optional[ad.AnnData] = None,
    n_top: int = 3000,
    edge_width_scale: float = 30,
    interaction_cutoff: float = 0.0,
    groupby: str = "cell_type",
    cell_type_color_key: str = "cell_type_colors",
    custom_color_mapping: Optional[dict] = None,
    figsize: Tuple[float, float] = (12, 10),
    margins: float = 0.2,
    label_font_size: int = 16,
    node_size: int = 1100,
    p_value_col: str = 'p_value_fdr',
    threshold: float = 0.05,
    filter_by_interaction_score: bool = True,
    threshold_interaction_score: float = 0.01,
    filter_significant: bool = True,
    label_border: bool = True,
    label_border_color: str = 'white',
    label_border_width: float = 3.0,
    save: Optional[str] = None,
    verbosity: int = 2,
    return_fig: bool = False
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot an interaction network for a specific cell type.
    
    Creates a directed network graph showing aggregated interactions where a 
    specific cell type is either sending or receiving signals. Edge thickness 
    represents the cumulative interaction strength.
    
    Parameters
    ----------
    laris_results : pd.DataFrame
        DataFrame containing LARIS results
        
    cell_type_of_interest : str
        Cell type to focus on (e.g., "B_cell", "T_cell")
        
    interaction_direction : {'sending', 'receiving'}, default='sending'
        Direction to filter:
        - 'sending' : Show outgoing interactions from cell_type_of_interest
        - 'receiving' : Show incoming interactions to cell_type_of_interest
        
    adata : anndata.AnnData, optional
        AnnData object containing cell type information for coloring nodes
        
    n_top : int, default=3000
        Number of top interactions if no filters applied
        
    edge_width_scale : float, default=30
        Scaling factor for edge thickness based on interaction score
        
    interaction_cutoff : float, default=0.0
        Minimum interaction score threshold
        
    groupby : str, default='cell_type'
        Column name in adata.obs containing cell type labels
        
    cell_type_color_key : str, default='cell_type_colors'
        Key in adata.uns containing cell type colors
        
    custom_color_mapping : dict, optional
        Custom mapping of cell types to colors
        
    figsize : tuple, default=(12, 10)
        Figure size in inches
        
    margins : float, default=0.2
        Margin space around the plot
        
    label_font_size : int, default=16
        Font size for node labels
        
    node_size : int, default=1100
        Size of network nodes
        
    p_value_col : str, default='p_value_fdr'
        Column name for p-value filtering
        
    threshold : float, default=0.05
        P-value cutoff for significance
        
    filter_by_interaction_score : bool, default=True
        If True, filter by interaction_score > threshold_interaction_score
        
    threshold_interaction_score : float, default=0.01
        Cutoff for interaction score
        
    filter_significant : bool, default=True
        If True, apply significance filtering
        
    label_border : bool, default=True
        If True, add border/outline to cell type labels for better visibility
        
    label_border_color : str, default='white'
        Color of the label border/outline
        
    label_border_width : float, default=3.0
        Width of the label border/outline
        
    save : str, optional
        Path to save figure
        
    verbosity : int, default=2
        Verbosity level
        
    return_fig : bool, default=False
        If True, return the figure and axes objects
    
    Returns
    -------
    tuple or None
        (fig, ax) if return_fig=True, otherwise None
    
    Examples
    --------
    >>> la.pl.plotCCCNetwork(
    ...     laris_results,
    ...     cell_type_of_interest='B_cell',
    ...     interaction_direction='sending',
    ...     adata=adata,
    ...     filter_significant=True,
    ...     label_border=True,
    ...     label_border_width=4.0,
    ...     save='network.pdf'
    ... )
    """
    # Apply filters
    laris_results_subset = laris_results.copy()
    did_filter = False

    if filter_significant:
        if p_value_col in laris_results_subset.columns:
            laris_results_subset = laris_results_subset[
                laris_results_subset[p_value_col] < threshold
            ]
            did_filter = True
        elif 'significant' in laris_results_subset.columns:
            laris_results_subset = laris_results_subset[
                laris_results_subset['significant']
            ]
            did_filter = True
        else:
            _log_message(
                f"'{p_value_col}' and 'significant' columns missing. "
                "Skipping significance filter.",
                2, verbosity, 'warning'
            )

    if filter_by_interaction_score:
        score_col = 'interaction_score'
        if score_col in laris_results_subset.columns:
            laris_results_subset = laris_results_subset[
                laris_results_subset[score_col] > threshold_interaction_score
            ]
            did_filter = True
        else:
            _log_message(
                f"'{score_col}' column missing. Skipping score filter.",
                2, verbosity, 'warning'
            )

    if not did_filter:
        _log_message(
            f"No filters applied. Using top {n_top} interactions.",
            2, verbosity, 'info'
        )
        laris_results_subset = laris_results_subset.iloc[:n_top]

    df_subset = laris_results_subset

    # Filter for cell type of interest
    if interaction_direction == "sending":
        df_filtered = df_subset[df_subset['sender'] == cell_type_of_interest]
    elif interaction_direction == "receiving":
        df_filtered = df_subset[df_subset['receiver'] == cell_type_of_interest]
    else:
        raise ValueError("interaction_direction must be 'sending' or 'receiving'")

    # Apply interaction cutoff
    df_filtered = df_filtered[df_filtered['interaction_score'] >= interaction_cutoff]

    if df_filtered.empty:
        _log_message(
            "No interactions found matching criteria.",
            1, verbosity, 'error'
        )
        fig, ax = plt.subplots(figsize=figsize)
        plt.show()
        if return_fig:
            return fig, ax
        return None

    # Group by cell type pairs
    df_grouped = df_filtered.groupby(
        ['sender', 'receiver'],
        as_index=False
    ).agg({'interaction_score': 'sum'})

    # Build network graph
    G = nx.from_pandas_edgelist(
        df_grouped,
        source='sender',
        target='receiver',
        edge_attr='interaction_score',
        create_using=nx.DiGraph()
    )

    # Add all cell type nodes
    if adata is not None:
        all_nodes = adata.obs[groupby].unique()
        for node in all_nodes:
            if node not in G:
                G.add_node(node)
    elif custom_color_mapping is not None:
        for node in custom_color_mapping.keys():
            if node not in G:
                G.add_node(node)

    # Compute layout
    pos = nx.circular_layout(G)

    # Define node colors
    if custom_color_mapping is not None:
        cell_type_to_color = custom_color_mapping
    elif adata is not None:
        cell_type_labels = adata.obs[groupby].unique()
        cell_type_colors = adata.uns[cell_type_color_key]
        cell_type_to_color = dict(zip(cell_type_labels, cell_type_colors))
    else:
        cell_type_to_color = {}

    node_colors = [cell_type_to_color.get(node, 'gray') for node in G.nodes()]

    # Create figure with GridSpec for proper layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
    
    # Main network plot (square)
    ax = fig.add_subplot(gs[0])
    ax.set_aspect('equal')
    ax.margins(margins)

    # Draw nodes
    node_collection = nx.draw_networkx_nodes(
        G, pos, node_size=node_size, node_color=node_colors, ax=ax
    )
    node_collection.set_zorder(1)

    # Draw labels with optional border
    labels = nx.draw_networkx_labels(
        G, pos, font_size=label_font_size, font_family="sans-serif", ax=ax
    )
    for label in labels.values():
        label.set_zorder(3)
        if label_border:
            label.set_path_effects([
                path_effects.Stroke(linewidth=label_border_width, foreground=label_border_color),
                path_effects.Normal()
            ])

    # Draw edges and collect edge scores for legend
    edge_scores = []
    for u, v, data in G.edges(data=True):
        sender_color = cell_type_to_color.get(u, 'gray')
        score = data['interaction_score']
        linewidth = score * edge_width_scale
        edge_scores.append(score)

        posA = pos[u]
        posB = pos[v]

        arrow = FancyArrowPatch(
            posA=posA,
            posB=posB,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            mutation_scale=40,
            color=sender_color,
            linewidth=linewidth,
            shrinkA=10,
            shrinkB=10,
            zorder=2
        )
        ax.add_patch(arrow)

    ax.set_title(
        f"Interaction Network for {cell_type_of_interest} "
        f"({interaction_direction.capitalize()} Interactions)"
    )
    ax.axis('off')

    # Create edge thickness legend in separate axes
    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis('off')
    
    if edge_scores:
        min_score = min(edge_scores)
        max_score = max(edge_scores)
        
        # Create representative values for legend
        if max_score > min_score:
            legend_values = [min_score, (min_score + max_score) / 2, max_score]
        else:
            legend_values = [max_score]
        
        legend_widths = [v * edge_width_scale for v in legend_values]
        
        _create_edge_thickness_legend(
            legend_ax, legend_values, legend_widths,
            title="Interaction\nScore",
            loc='center left',
            bbox_to_anchor=(0, 0.5)
        )

    plt.tight_layout()

    _save_figure(fig, save, verbosity)
    plt.show()

    if return_fig:
        return fig, ax
    return None


def plotCCCNetworkCumulative(
    laris_results: pd.DataFrame,
    adata: ad.AnnData,
    cutoff: float = 0,
    n_top: int = 3000,
    groupby: str = "cell_type",
    cell_type_color_key: str = "cell_type_colors",
    custom_color_mapping: Optional[dict] = None,
    figsize: Tuple[float, float] = (12, 10),
    margins: float = 0.2,
    label_font_size: int = 16,
    node_size: int = 1100,
    edge_width_scale: float = 5,
    p_value_col: str = 'p_value_fdr',
    threshold: float = 0.05,
    filter_by_interaction_score: bool = True,
    threshold_interaction_score: float = 0.01,
    filter_significant: bool = True,
    edge_thickness_by_numbers: bool = False,
    total_edge_thickness: float = 100,
    label_border: bool = True,
    label_border_color: str = 'white',
    label_border_width: float = 3.0,
    save: Optional[str] = None,
    verbosity: int = 2,
    return_fig: bool = False
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot a cumulative interaction network across all cell types.
    
    Creates a comprehensive network showing aggregated interactions between all 
    cell type pairs. Edge thickness can represent either cumulative interaction 
    scores or total interaction counts.
    
    Parameters
    ----------
    laris_results : pd.DataFrame
        DataFrame containing LARIS interaction results
        
    adata : anndata.AnnData
        AnnData object with cell type information
        
    cutoff : float, default=0
        Minimum threshold for including an edge
        
    n_top : int, default=3000
        Number of top interactions if no filters applied
        
    groupby : str, default='cell_type'
        Column in adata.obs containing cell type labels
        
    cell_type_color_key : str, default='cell_type_colors'
        Key in adata.uns for cell type colors
        
    custom_color_mapping : dict, optional
        Custom cell type to color mapping
        
    figsize : tuple, default=(12, 10)
        Figure dimensions in inches
        
    margins : float, default=0.2
        Plot margin size
        
    label_font_size : int, default=16
        Font size for node labels
        
    node_size : int, default=1100
        Size of network nodes
        
    edge_width_scale : float, default=5
        Scaling factor for edge thickness (when using scores)
        
    p_value_col : str, default='p_value_fdr'
        Column name for p-value filtering
        
    threshold : float, default=0.05
        P-value cutoff for significance
        
    filter_by_interaction_score : bool, default=True
        If True, filter by interaction_score > threshold_interaction_score
        
    threshold_interaction_score : float, default=0.01
        Cutoff for interaction score
        
    filter_significant : bool, default=True
        If True, apply significance filtering
        
    edge_thickness_by_numbers : bool, default=False
        If True, edge thickness represents interaction count
        
    total_edge_thickness : float, default=100
        Total thickness budget when edge_thickness_by_numbers=True
        
    label_border : bool, default=True
        If True, add border/outline to cell type labels for better visibility
        
    label_border_color : str, default='white'
        Color of the label border/outline
        
    label_border_width : float, default=3.0
        Width of the label border/outline
        
    save : str, optional
        Path to save figure
        
    verbosity : int, default=2
        Verbosity level
        
    return_fig : bool, default=False
        If True, return the figure and axes objects
    
    Returns
    -------
    tuple or None
        (fig, ax) if return_fig=True, otherwise None
    
    Examples
    --------
    >>> la.pl.plotCCCNetworkCumulative(
    ...     laris_results,
    ...     adata=adata,
    ...     filter_significant=True,
    ...     label_border=True,
    ...     label_border_color='white',
    ...     label_border_width=4.0,
    ...     save='cumulative_network.pdf'
    ... )
    """
    # Apply filters
    laris_results_subset = laris_results.copy()
    did_filter = False

    if filter_significant:
        if p_value_col in laris_results_subset.columns:
            laris_results_subset = laris_results_subset[
                laris_results_subset[p_value_col] < threshold
            ]
            did_filter = True
        elif 'significant' in laris_results_subset.columns:
            laris_results_subset = laris_results_subset[
                laris_results_subset['significant']
            ]
            did_filter = True
        else:
            _log_message(
                f"'{p_value_col}' and 'significant' columns missing. "
                "Skipping significance filter.",
                2, verbosity, 'warning'
            )

    if filter_by_interaction_score:
        score_col = 'interaction_score'
        if score_col in laris_results_subset.columns:
            laris_results_subset = laris_results_subset[
                laris_results_subset[score_col] > threshold_interaction_score
            ]
            did_filter = True
        else:
            _log_message(
                f"'{score_col}' column missing. Skipping score filter.",
                2, verbosity, 'warning'
            )

    if not did_filter:
        _log_message(
            f"No filters applied. Using top {n_top} interactions.",
            2, verbosity, 'info'
        )
        laris_results_subset = laris_results_subset.iloc[:n_top]

    df_subset = laris_results_subset

    # Aggregate data
    if edge_thickness_by_numbers:
        df_agg = (
            df_subset
            .groupby(['sender', 'receiver'])
            .size()
            .reset_index(name='interaction_count')
        )
        df_agg = df_agg[df_agg['interaction_count'] >= cutoff]
        total_interaction_count = df_agg['interaction_count'].sum()
        edge_attr_field = 'interaction_count'
    else:
        df_agg = (
            df_subset
            .groupby(['sender', 'receiver'], as_index=False)
            ['interaction_score']
            .sum()
        )
        df_agg = df_agg[df_agg['interaction_score'] >= cutoff]
        edge_attr_field = 'interaction_score'

    if df_agg.empty:
        _log_message(
            "No interactions found matching criteria.",
            1, verbosity, 'error'
        )
        fig, ax = plt.subplots(figsize=figsize)
        plt.show()
        if return_fig:
            return fig, ax
        return None

    # Build network graph
    G = nx.from_pandas_edgelist(
        df_agg,
        source='sender',
        target='receiver',
        edge_attr=edge_attr_field,
        create_using=nx.DiGraph()
    )

    # Remove self-loops
    G.remove_edges_from(list(nx.selfloop_edges(G)))

    # Add all cell types as nodes
    unique_cell_types = adata.obs[groupby].unique()
    for ctype in unique_cell_types:
        if ctype not in G:
            G.add_node(ctype)

    # Determine node colors
    if custom_color_mapping is not None:
        cell_type_to_color = custom_color_mapping
    elif adata is not None:
        cell_type_labels = adata.obs[groupby].unique()
        cell_type_colors = adata.uns[cell_type_color_key]
        cell_type_to_color = dict(zip(cell_type_labels, cell_type_colors))
    else:
        cell_type_to_color = {}

    node_colors = [cell_type_to_color.get(node, 'gray') for node in G.nodes()]

    # Create figure with GridSpec for proper layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
    
    # Main network plot (square)
    ax = fig.add_subplot(gs[0])
    ax.set_aspect('equal')
    ax.margins(margins)
    
    pos = nx.circular_layout(G)

    # Draw nodes
    node_collection = nx.draw_networkx_nodes(
        G, pos, node_size=node_size, node_color=node_colors, ax=ax
    )
    node_collection.set_zorder(1)

    # Draw labels with optional border
    labels = nx.draw_networkx_labels(
        G, pos, font_size=label_font_size, font_family="sans-serif", ax=ax
    )
    for label in labels.values():
        label.set_zorder(3)
        if label_border:
            label.set_path_effects([
                path_effects.Stroke(linewidth=label_border_width, foreground=label_border_color),
                path_effects.Normal()
            ])

    # Draw edges and collect values for legend
    edge_values = []
    edge_widths = []
    
    for u, v, data in G.edges(data=True):
        sender_color = cell_type_to_color.get(u, 'gray')
        posA = pos[u]
        posB = pos[v]

        if edge_thickness_by_numbers:
            if total_interaction_count > 0:
                count = data['interaction_count']
                linewidth = (count / total_interaction_count) * total_edge_thickness
                edge_values.append(count)
            else:
                linewidth = 0
                edge_values.append(0)
        else:
            score = data['interaction_score']
            linewidth = score * edge_width_scale
            edge_values.append(score)
        
        edge_widths.append(linewidth)

        arrow = FancyArrowPatch(
            posA=posA,
            posB=posB,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            mutation_scale=10,
            color=sender_color,
            linewidth=linewidth,
            shrinkA=10,
            shrinkB=10,
            zorder=2
        )
        ax.add_patch(arrow)

    title_text = (
        "Interaction Network by " +
        ("Interaction Count" if edge_thickness_by_numbers else "Cumulative Score")
    )
    ax.set_title(title_text)
    ax.axis('off')

    # Create edge thickness legend in separate axes
    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis('off')
    
    if edge_values:
        min_val = min(edge_values)
        max_val = max(edge_values)
        
        # Create representative values for legend
        if max_val > min_val:
            legend_values = [min_val, (min_val + max_val) / 2, max_val]
        else:
            legend_values = [max_val]
        
        if edge_thickness_by_numbers:
            legend_widths = [(v / total_interaction_count) * total_edge_thickness 
                            for v in legend_values]
            legend_title = "Interaction\nCount"
        else:
            legend_widths = [v * edge_width_scale for v in legend_values]
            legend_title = "Cumulative\nScore"
        
        _create_edge_thickness_legend(
            legend_ax, legend_values, legend_widths,
            title=legend_title,
            loc='center left',
            bbox_to_anchor=(0, 0.5)
        )

    plt.tight_layout()

    _save_figure(fig, save, verbosity)
    plt.show()

    if return_fig:
        return fig, ax
    return None


# ============================================================================
# DOT PLOT FUNCTIONS
# ============================================================================

def plotCCCDotPlot(
    laris_results: pd.DataFrame,
    interactions_to_plot: List[str],
    senders: Optional[List[str]] = None,
    receivers: Optional[List[str]] = None,
    sender_receiver_pairs: Optional[List[str]] = None,
    delimiter_pair: str = "-->",
    n_top: int = 3000,
    cmap: Union[str, colors.Colormap] = None,
    bubble_size: float = 250,
    p_value_col: str = 'p_value_fdr',
    threshold: float = 0.05,
    filter_by_interaction_score: bool = True,
    threshold_interaction_score: float = 0.01,
    filter_significant: bool = True,
    n_permutations: int = 1000,
    legend_fontsize: int = 16,
    show_grid: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    save: Optional[str] = None,
    verbosity: int = 2,
    return_fig: bool = False
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Create a bubble plot for selected cell type pairs and interactions.
    
    Visualizes interaction strengths between specific sender-receiver cell type 
    pairs using bubbles where size represents -log10(p-value) and color 
    represents interaction score.
    
    Parameters
    ----------
    laris_results : pd.DataFrame
        DataFrame containing LARIS results
        
    interactions_to_plot : list of str
        List of interaction names to include (e.g., ['CXCL13::CXCR5', ...])
        
    senders : list of str, optional
        List of sending cell types to plot (used with receivers)
        
    receivers : list of str, optional
        List of receiving cell types (must match length of senders)
        
    sender_receiver_pairs : list of str, optional
        Alternative to senders/receivers. List of pairs in format 
        'sender-->receiver' (or using custom delimiter_pair)
        
    delimiter_pair : str, default='-->'
        Delimiter used in sender_receiver_pairs to separate sender and receiver
        
    n_top : int, default=3000
        Number of top interactions if no filters applied
        
    cmap : colormap, optional
        Colormap for bubble colors. If None, uses pos_cmap
        
    bubble_size : float, default=250
        Maximum bubble size for most significant p-values
        
    p_value_col : str, default='p_value_fdr'
        Column name for p-value filtering and sizing
        
    threshold : float, default=0.05
        P-value cutoff for significance filtering
        
    filter_by_interaction_score : bool, default=True
        If True, filter by interaction_score > threshold_interaction_score
        
    threshold_interaction_score : float, default=0.01
        Cutoff for interaction score
        
    filter_significant : bool, default=True
        If True, apply significance filtering
        
    n_permutations : int, default=1000
        Number of permutations used (for p-value scaling)
        
    legend_fontsize : int, default=16
        Font size for legend titles ("Interaction Score" and "P value")
        
    show_grid : bool, default=False
        If True, show grid lines in the plot
        
    figsize : tuple, optional
        Figure size. If None, automatically calculated
        
    save : str, optional
        Path to save figure
        
    verbosity : int, default=2
        Verbosity level
        
    return_fig : bool, default=False
        If True, return the figure and axes objects
    
    Returns
    -------
    tuple or None
        (fig, ax) if return_fig=True, otherwise None
    
    Examples
    --------
    >>> # Using senders and receivers
    >>> senders = ['B_cell', 'B_cell', 'T_cell']
    >>> receivers = ['T_cell', 'Macrophage', 'B_cell']
    >>> interactions = ['CXCL13::CXCR5', 'CD40LG::CD40']
    >>> 
    >>> la.pl.plotCCCDotPlot(
    ...     laris_results,
    ...     interactions_to_plot=interactions,
    ...     senders=senders,
    ...     receivers=receivers,
    ...     filter_significant=True,
    ...     legend_fontsize=16,
    ...     show_grid=False,
    ...     save='dotplot.pdf'
    ... )
    >>> 
    >>> # Using sender_receiver_pairs
    >>> pairs = ['B_cell-->T_cell', 'B_cell-->Macrophage', 'T_cell-->B_cell']
    >>> la.pl.plotCCCDotPlot(
    ...     laris_results,
    ...     interactions_to_plot=interactions,
    ...     sender_receiver_pairs=pairs,
    ...     save='dotplot.pdf'
    ... )
    
    Notes
    -----
    Bubble sizes are scaled by -log10(p_value), providing continuous scaling
    that better represents the range of significance values. The minimum
    possible p-value is determined by 1/(n_permutations + 1).
    """
    if cmap is None:
        cmap = pos_cmap

    # Parse sender_receiver_pairs if provided
    if sender_receiver_pairs is not None:
        senders = []
        receivers = []
        for pair in sender_receiver_pairs:
            parts = pair.split(delimiter_pair)
            if len(parts) != 2:
                _log_message(
                    f"Invalid pair format: '{pair}'. Expected format: 'sender{delimiter_pair}receiver'",
                    1, verbosity, 'error'
                )
                return None
            senders.append(parts[0].strip())
            receivers.append(parts[1].strip())
    
    # Validate inputs
    if senders is None or receivers is None:
        _log_message(
            "Must provide either (senders, receivers) or sender_receiver_pairs",
            1, verbosity, 'error'
        )
        return None
    
    if len(senders) != len(receivers):
        _log_message(
            "Length of senders and receivers must match",
            1, verbosity, 'error'
        )
        return None

    # Apply filters
    laris_results_subset = laris_results.copy()
    did_filter = False

    if filter_significant:
        if p_value_col in laris_results_subset.columns:
            laris_results_subset = laris_results_subset[
                laris_results_subset[p_value_col] < threshold
            ]
            did_filter = True
        elif 'significant' in laris_results_subset.columns:
            laris_results_subset = laris_results_subset[
                laris_results_subset['significant']
            ]
            did_filter = True
        else:
            _log_message(
                f"'{p_value_col}' and 'significant' columns missing.",
                2, verbosity, 'warning'
            )

    if filter_by_interaction_score:
        score_col = 'interaction_score'
        if score_col in laris_results_subset.columns:
            laris_results_subset = laris_results_subset[
                laris_results_subset[score_col] > threshold_interaction_score
            ]
            did_filter = True
        else:
            _log_message(
                f"'{score_col}' column missing. Skipping score filter.",
                2, verbosity, 'warning'
            )

    if not did_filter:
        if n_top is not None:
            _log_message(
                f"No filters applied. Using top {n_top} interactions.",
                2, verbosity, 'info'
            )
            laris_results_subset = laris_results_subset.sort_values(
                'interaction_score', ascending=False
            ).iloc[:n_top]

    # Build mask for specified sender-receiver pairs
    mask_cell_pairs = None
    for sender, receiver in zip(senders, receivers):
        current_mask = (
            (laris_results_subset['sender'] == sender) &
            (laris_results_subset['receiver'] == receiver)
        )
        if mask_cell_pairs is None:
            mask_cell_pairs = current_mask
        else:
            mask_cell_pairs |= current_mask

    if mask_cell_pairs is None:
        _log_message("No cell pairs specified or found.", 1, verbosity, 'error')
        fig, ax = plt.subplots()
        plt.show()
        if return_fig:
            return fig, ax
        return None

    # Filter for selected interactions and cell pairs
    df_filtered = laris_results_subset[
        laris_results_subset['interaction_name'].isin(interactions_to_plot) & 
        mask_cell_pairs
    ].copy()

    # Remove missing scores
    df_filtered = df_filtered[df_filtered['interaction_score'].notna()]

    # Handle p-value based bubble sizing using -log10
    bubble_legend = False
    if 'p_value' in df_filtered.columns:
        # Compute bubble sizes using -log10(p_value)
        df_filtered['bubble_size_plot'] = _compute_bubble_sizes_log10(
            df_filtered['p_value'].values,
            bubble_size,
            n_permutations
        )
        bubble_legend = True
        _log_message(
            f"Using -log10(p_value) for bubble sizing (n_perm={n_permutations})",
            3, verbosity, 'debug'
        )
    else:
        _log_message(
            "p_value not found. Using constant bubble size.",
            2, verbosity, 'info'
        )
        df_filtered['bubble_size_plot'] = bubble_size

    # Create cell type pair labels
    df_filtered['cell_type_pair'] = (
        df_filtered['sender'] + ' → ' + df_filtered['receiver']
    )

    # Define expected cell pairs in order
    all_cell_pairs = [
        f"{s} → {r}" for s, r in zip(senders, receivers)
    ]

    # Force categorical order
    df_filtered['cell_type_pair'] = pd.Categorical(
        df_filtered['cell_type_pair'],
        categories=all_cell_pairs,
        ordered=True
    )
    df_filtered['interaction_name'] = pd.Categorical(
        df_filtered['interaction_name'],
        categories=interactions_to_plot,
        ordered=True
    )

    # Only plot non-zero scores
    df_nonzero = df_filtered[df_filtered['interaction_score'] > 0]

    if df_nonzero.empty:
        _log_message(
            "No non-zero interactions found to plot.",
            2, verbosity, 'warning'
        )

    # Calculate figure size
    num_cell_pairs = len(all_cell_pairs)
    num_interactions = len(interactions_to_plot)
    
    if figsize is None:
        fig_width = max(8, num_cell_pairs * 1.5 + 4)  # Extra space for legends
        fig_height = max(6, num_interactions * 0.8)
        figsize = (fig_width, fig_height)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot bubbles
    scatter = None
    if not df_nonzero.empty:
        scatter = ax.scatter(
            x=df_nonzero['cell_type_pair'].cat.codes,
            y=df_nonzero['interaction_name'].cat.codes,
            c=df_nonzero['interaction_score'],
            s=df_nonzero['bubble_size_plot'],
            cmap=cmap,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )

    # Set ticks and labels
    ax.set_xticks(range(len(all_cell_pairs)))
    ax.set_xticklabels(all_cell_pairs, rotation=45, ha='right')
    ax.set_yticks(range(len(interactions_to_plot)))
    ax.set_yticklabels(interactions_to_plot)
    ax.set_xlabel('Cell type pairs')
    ax.set_ylabel('Interactions')

    # Set limits with padding
    ax.set_xlim(-0.5, len(all_cell_pairs) - 0.5)
    ax.set_ylim(-0.5, len(interactions_to_plot) - 0.5)

    # Control grid lines
    if not show_grid:
        ax.grid(False)

    # Add colorbar with doubled width and label on left side
    if scatter is not None:
        # Get the actual max score for colorbar
        if not df_nonzero.empty:
            max_score = df_nonzero['interaction_score'].max()
        else:
            max_score = 1.0
        
        # Set color limits to start from 0
        scatter.set_clim(0, max_score)
        
        # Create colorbar with doubled width and moved further right (pad=0.15 to account for left label)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.15, aspect=20)
        # Set label on the left side of colorbar
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.set_label('Interaction Score', fontsize=legend_fontsize)

    # Add p-value legend with more space from colorbar
    if bubble_legend:
        _create_pvalue_legend_log10(
            ax, bubble_size, n_permutations,
            loc='upper left',
            bbox_to_anchor=(1.35, 1.0),  # Moved further right for more spacing
            frameon=False,
            title_fontsize=legend_fontsize
        )

    plt.tight_layout()

    _save_figure(fig, save, verbosity)
    plt.show()

    if return_fig:
        return fig, ax
    return None


def plotCCCDotPlotFacet(
    laris_results: pd.DataFrame,
    cmap: Union[str, colors.Colormap] = None,
    bubble_size: float = 400,
    height_single_panel: float = 4.5,
    width_single_panel: float = 4.5,
    x_padding: float = 0.4,
    y_padding: float = 0.4,
    n_top: Optional[int] = None,
    senders: Optional[List[str]] = None,
    receivers: Optional[List[str]] = None,
    interactions_to_plot: Optional[List[str]] = None,
    p_value_col: str = 'p_value_fdr',
    threshold: float = 0.05,
    filter_by_interaction_score: bool = True,
    threshold_interaction_score: float = 0.01,
    filter_significant: bool = True,
    n_permutations: int = 1000,
    ncol: int = 3,
    facet_by: str = 'sender',
    legend_fontsize: int = 16,
    show_grid: bool = True,
    save: Optional[str] = None,
    verbosity: int = 2,
    return_fig: bool = False
) -> Optional[plt.Figure]:
    """
    Create faceted bubble plots organized by sending or receiving cell type.
    
    Generates a grid of bubble plots where each facet represents a different 
    sending or receiving cell type, showing interactions to/from other cell types.
    
    Parameters
    ----------
    laris_results : pd.DataFrame
        DataFrame containing LARIS interaction results
        
    cmap : colormap, optional
        Colormap for bubble colors. If None, uses pos_cmap
        
    bubble_size : float, default=400
        Maximum bubble size
        
    height_single_panel : float, default=4.5
        Height of each facet panel in inches
        
    width_single_panel : float, default=4.5
        Width of each facet panel in inches
        
    x_padding : float, default=0.4
        Extra space on x-axis
        
    y_padding : float, default=0.4
        Extra space on y-axis
        
    n_top : int, optional
        Number of top interactions if no filters applied
        
    senders : list of str, optional
        Specific sending cell types. If None, all are included
        
    receivers : list of str, optional
        Specific receiving cell types. If None, all are included
        
    interactions_to_plot : list of str, optional
        Specific interactions. If None, all are included
        
    p_value_col : str, default='p_value_fdr'
        Column name for p-value filtering
        
    threshold : float, default=0.05
        P-value cutoff for significance
        
    filter_by_interaction_score : bool, default=True
        If True, filter by interaction_score > threshold_interaction_score
        
    threshold_interaction_score : float, default=0.01
        Cutoff for interaction score
        
    filter_significant : bool, default=True
        If True, apply significance filtering
        
    n_permutations : int, default=1000
        Number of permutations (for p-value scaling)
        
    ncol : int, default=3
        Number of columns per row
        
    facet_by : str, default='sender'
        How to organize facets:
        - 'sender' : Each facet shows a different sending cell type
        - 'receiver' : Each facet shows a different receiving cell type
        
    legend_fontsize : int, default=16
        Font size for legend titles ("Interaction Score" and "P value")
        
    show_grid : bool, default=True
        If True, show grid lines in the plots
        
    save : str, optional
        Path to save figure
        
    verbosity : int, default=2
        Verbosity level
        
    return_fig : bool, default=False
        If True, return the Figure object
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The Figure object if return_fig=True, otherwise None
    
    Examples
    --------
    >>> # Facet by sender (default)
    >>> la.pl.plotCCCDotPlotFacet(
    ...     laris_results,
    ...     senders=['B_cell', 'T_cell'],
    ...     receivers=['Macrophage', 'NK_cell'],
    ...     interactions_to_plot=['CXCL13::CXCR5', 'CD40LG::CD40'],
    ...     filter_significant=True,
    ...     ncol=2,
    ...     facet_by='sender',
    ...     show_grid=False,
    ...     save='facet_plot_sender.pdf'
    ... )
    >>> 
    >>> # Facet by receiver
    >>> la.pl.plotCCCDotPlotFacet(
    ...     laris_results,
    ...     senders=['B_cell', 'T_cell'],
    ...     receivers=['Macrophage', 'NK_cell'],
    ...     interactions_to_plot=['CXCL13::CXCR5', 'CD40LG::CD40'],
    ...     filter_significant=True,
    ...     ncol=2,
    ...     facet_by='receiver',
    ...     save='facet_plot_receiver.pdf'
    ... )
    """
    if cmap is None:
        cmap = pos_cmap
    
    # Validate facet_by parameter
    if facet_by not in ['sender', 'receiver']:
        _log_message(
            f"facet_by must be 'sender' or 'receiver', got '{facet_by}'",
            1, verbosity, 'error'
        )
        return None

    # Apply filters
    laris_results_subset = laris_results.copy()
    did_filter = False

    if filter_significant:
        if p_value_col in laris_results_subset.columns:
            laris_results_subset = laris_results_subset[
                laris_results_subset[p_value_col] < threshold
            ]
            did_filter = True
        elif 'significant' in laris_results_subset.columns:
            laris_results_subset = laris_results_subset[
                laris_results_subset['significant']
            ]
            did_filter = True
        else:
            _log_message(
                f"'{p_value_col}' and 'significant' columns missing.",
                2, verbosity, 'warning'
            )

    if filter_by_interaction_score:
        score_col = 'interaction_score'
        if score_col in laris_results_subset.columns:
            laris_results_subset = laris_results_subset[
                laris_results_subset[score_col] > threshold_interaction_score
            ]
            did_filter = True
        else:
            _log_message(
                f"'{score_col}' column missing. Skipping score filter.",
                2, verbosity, 'warning'
            )

    if not did_filter:
        if n_top is not None:
            _log_message(
                f"No filters applied. Using top {n_top} interactions.",
                2, verbosity, 'info'
            )
            laris_results_subset = laris_results_subset.iloc[:n_top]

    data = laris_results_subset

    # Subset based on user selections
    if senders is not None:
        data = data[data["sender"].isin(senders)]

    if receivers is not None:
        data = data[data["receiver"].isin(receivers)]

    if interactions_to_plot is not None:
        data = data[data["interaction_name"].isin(interactions_to_plot)]

    if data.empty:
        _log_message(
            "No interactions found matching criteria.",
            1, verbosity, 'error'
        )
        return None

    # Determine categories for axes based on facet_by
    if senders is not None:
        sender_cats = senders
    else:
        sender_cats = sorted(data["sender"].unique())
    
    if receivers is not None:
        receiver_cats = receivers
    else:
        receiver_cats = sorted(data["receiver"].unique())
    
    if interactions_to_plot is not None:
        interaction_cats = interactions_to_plot
    else:
        interaction_cats = sorted(data["interaction_name"].unique())

    # Determine facet categories and x-axis categories based on facet_by
    if facet_by == 'sender':
        facet_cats = sender_cats
        x_cats = receiver_cats
        facet_col = 'sender'
        x_col = 'receiver'
        x_label = "Receiver"
    else:  # facet_by == 'receiver'
        facet_cats = receiver_cats
        x_cats = sender_cats
        facet_col = 'receiver'
        x_col = 'sender'
        x_label = "Sender"

    # Convert to categorical with specified order
    data["sender"] = pd.Categorical(
        data["sender"],
        categories=sender_cats,
        ordered=True
    )
    data["receiver"] = pd.Categorical(
        data["receiver"],
        categories=receiver_cats,
        ordered=True
    )
    data["interaction_name"] = pd.Categorical(
        data["interaction_name"],
        categories=interaction_cats,
        ordered=True
    )

    # Filter for non-zero scores
    data_plot = data[data["interaction_score"] != 0].copy()

    # Handle p-value based bubble sizing using -log10
    bubble_legend = False
    if 'p_value' in data_plot.columns:
        data_plot['bubble_size_plot'] = _compute_bubble_sizes_log10(
            data_plot['p_value'].values,
            bubble_size,
            n_permutations
        )
        bubble_legend = True
    else:
        data_plot['bubble_size_plot'] = bubble_size

    if data_plot.empty:
        _log_message(
            "No non-zero interactions found.",
            2, verbosity, 'warning'
        )

    # Determine number of facets
    n_facets = len(facet_cats)
    
    # Calculate number of rows
    nrow = math.ceil(n_facets / ncol)

    # Calculate figure size based on panel dimensions
    fig_width = width_single_panel * ncol + 2.5  # Extra space for legends
    fig_height = height_single_panel * nrow
    
    # Create figure and axes manually for better control
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create GridSpec for the panels
    gs = gridspec.GridSpec(nrow, ncol, figure=fig, wspace=0.3, hspace=0.4)
    
    axes = []
    for idx in range(n_facets):
        row_idx = idx // ncol
        col_idx = idx % ncol
        ax = fig.add_subplot(gs[row_idx, col_idx])
        axes.append(ax)
    
    # Calculate global min and max for consistent color scaling
    if len(data_plot) > 0:
        vmin = 0  # Start from 0 for interaction scores
        vmax = data_plot["interaction_score"].max()
    else:
        vmin, vmax = 0, 1
    
    # Plot each facet
    for idx, facet_val in enumerate(facet_cats):
        ax = axes[idx]
        row_idx = idx // ncol
        col_idx = idx % ncol
        
        # Get data for this facet
        facet_data = data_plot[data_plot[facet_col] == facet_val]
        
        if not facet_data.empty:
            scatter = ax.scatter(
                x=facet_data[x_col].cat.codes,
                y=facet_data["interaction_name"].cat.codes,
                c=facet_data["interaction_score"],
                s=facet_data["bubble_size_plot"],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5
            )
        
        # Set title
        ax.set_title(facet_val)
        
        # Set x-axis ticks
        ax.set_xticks(range(len(x_cats)))
        
        # Determine if this is the bottom row
        is_bottom_row = (row_idx == nrow - 1) or (idx >= n_facets - ncol)
        
        # Only show x-axis labels for bottom row panels
        if is_bottom_row or nrow == 1:
            ax.set_xticklabels(x_cats, rotation=45, ha='right')
            ax.set_xlabel(x_label)
        else:
            ax.set_xticklabels([])
        
        ax.set_xlim(-0.5 - x_padding, len(x_cats) - 0.5 + x_padding)
        
        # Set y-axis ticks and labels
        ax.set_yticks(range(len(interaction_cats)))
        
        # Always show y-axis labels for first column
        if col_idx == 0:
            ax.set_yticklabels(interaction_cats)
            ax.set_ylabel("Interactions")
        else:
            ax.set_yticklabels([])
        
        ax.set_ylim(-0.5 - y_padding, len(interaction_cats) - 0.5 + y_padding)
        
        # Show spines
        for spine in ax.spines.values():
            spine.set_visible(True)
        
        # Control grid lines
        if not show_grid:
            ax.grid(False)

    # Adjust layout to make room for legends
    fig.subplots_adjust(right=0.82)
    
    # Position legends based on number of rows
    if nrow == 1:
        # Single row: colorbar and p-value legend side by side on the right
        # Doubled width: 0.016 instead of 0.008, moved right to 0.88 to account for left label
        cbar_ax = fig.add_axes([0.88, 0.15, 0.016, 0.5])
        
        if len(data_plot) > 0:
            max_score = data['interaction_score'].max()
            norm = plt.Normalize(0, max_score)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=cbar_ax, ticks=[0, max_score])
            cbar.ax.set_yticklabels(['0', f'{max_score:.3f}'])
            # Set label on the left side
            cbar.ax.yaxis.set_label_position('left')
            cbar.ax.yaxis.set_ticks_position('right')
            cbar.set_label("Interaction Score", fontsize=legend_fontsize)

        # P-value legend to the right of colorbar with more spacing
        if bubble_legend:
            legend_ax = fig.add_axes([0.93, 0.15, 0.10, 0.5], frameon=False)
            legend_ax.axis('off')
            
            min_p = 1.0 / (n_permutations + 1)
            max_neg_log10 = -np.log10(min_p)
            
            p_values_legend = [0.001, 0.01, 0.05, 0.1, 1.0]
            p_values_legend = [p for p in p_values_legend if p >= min_p]
            
            legend_handles = []
            for p in p_values_legend:
                neg_log10_p = -np.log10(p)
                normalized = neg_log10_p / max_neg_log10
                size = bubble_size * (0.1 + 0.9 * normalized)
                
                if p < 0.001:
                    label = f'$\it{{P}}$ ≤ {p:.0e}'
                elif p < 0.01:
                    label = f'$\it{{P}}$ = {p:.3f}'
                else:
                    label = f'$\it{{P}}$ = {p:.2f}'
                
                handle = Line2D([0], [0], marker='o', color='w', label=label,
                               markerfacecolor='gray', markersize=np.sqrt(size),
                               markeredgecolor='black', markeredgewidth=0.5)
                legend_handles.append(handle)
            
            legend = legend_ax.legend(
                handles=legend_handles, 
                title="$\it{P}$ value",
                loc='center left',
                frameon=False,
                labelspacing=1.2,
                handletextpad=1.5
            )
            legend.get_title().set_fontsize(legend_fontsize)
    else:
        # Multiple rows: p-value legend above colorbar with more spacing
        # Doubled width: 0.016 instead of 0.008, moved right to 0.88 to account for left label
        cbar_ax = fig.add_axes([0.88, 0.10, 0.016, 0.25])
        
        if len(data_plot) > 0:
            max_score = data['interaction_score'].max()
            norm = plt.Normalize(0, max_score)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=cbar_ax, ticks=[0, max_score])
            cbar.ax.set_yticklabels(['0', f'{max_score:.3f}'])
            # Set label on the left side
            cbar.ax.yaxis.set_label_position('left')
            cbar.ax.yaxis.set_ticks_position('right')
            cbar.set_label("Interaction Score", fontsize=legend_fontsize)

        # P-value legend above colorbar with more spacing
        if bubble_legend:
            legend_ax = fig.add_axes([0.88, 0.42, 0.15, 0.30], frameon=False)
            legend_ax.axis('off')
            
            min_p = 1.0 / (n_permutations + 1)
            max_neg_log10 = -np.log10(min_p)
            
            p_values_legend = [0.001, 0.01, 0.05, 0.1, 1.0]
            p_values_legend = [p for p in p_values_legend if p >= min_p]
            
            legend_handles = []
            for p in p_values_legend:
                neg_log10_p = -np.log10(p)
                normalized = neg_log10_p / max_neg_log10
                size = bubble_size * (0.1 + 0.9 * normalized)
                
                if p < 0.001:
                    label = f'$\it{{P}}$ ≤ {p:.0e}'
                elif p < 0.01:
                    label = f'$\it{{P}}$ = {p:.3f}'
                else:
                    label = f'$\it{{P}}$ = {p:.2f}'
                
                handle = Line2D([0], [0], marker='o', color='w', label=label,
                               markerfacecolor='gray', markersize=np.sqrt(size),
                               markeredgecolor='black', markeredgewidth=0.5)
                legend_handles.append(handle)
            
            legend = legend_ax.legend(
                handles=legend_handles, 
                title="$\it{P}$ value",
                loc='upper left',
                frameon=False,
                labelspacing=1.2,
                handletextpad=1.5
            )
            legend.get_title().set_fontsize(legend_fontsize)

    # Save figure
    if save is not None:
        fig.savefig(save, bbox_inches='tight', dpi=300)
        _log_message(f"Figure saved to: {save}", 2, verbosity, 'info')

    plt.show()
    
    if return_fig:
        return fig
    return None


def plotLRDotPlot(
    adata_dotplot: ad.AnnData,
    interactions_to_plot: List[str],
    groupby: str,
    delimiter: str = '::',
    cmap_interaction: str = 'Spectral_r',
    cmap_ligand: str = 'Blues',
    cmap_receptor: str = 'Purples',
    standard_scale_interaction: Optional[str] = 'var',
    standard_scale_ligand: Optional[str] = 'var',
    standard_scale_receptor: Optional[str] = 'var',
    orientation: str = 'horizontal',
    row_height: Optional[float] = None,
    max_height: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save: Optional[str] = None,
    verbosity: int = 2,
    return_fig: bool = False
) -> Optional[Tuple[plt.Figure, np.ndarray]]:
    """
    Create three side-by-side (or stacked) dot plots for LR pairs, ligands, and receptors.
    
    Visualizes expression patterns of ligand-receptor pairs alongside individual 
    ligand and receptor expression across cell types.
    
    Parameters
    ----------
    adata_dotplot : anndata.AnnData
        AnnData object prepared with prepareDotPlotAdata() containing both 
        LR pair scores and individual gene expression
        
    interactions_to_plot : list of str
        List of LR pairs in format 'ligand::receptor' (or using custom delimiter)
        
    groupby : str
        Column in adata_dotplot.obs to group by
        
    delimiter : str, default='::'
        Delimiter used to separate ligand and receptor in interaction names
        
    cmap_interaction : str, default='Spectral_r'
        Colormap for the LR interaction scores plot
        
    cmap_ligand : str, default='Blues'
        Colormap for the ligand expression plot
        
    cmap_receptor : str, default='Purples'
        Colormap for the receptor expression plot
        
    standard_scale_interaction : str or None, default='var'
        Scaling method for interaction scores plot ('var', 'group', or None)
        
    standard_scale_ligand : str or None, default='var'
        Scaling method for ligand expression plot ('var', 'group', or None)
        
    standard_scale_receptor : str or None, default='var'
        Scaling method for receptor expression plot ('var', 'group', or None)
        
    orientation : str, default='horizontal'
        Layout orientation: 'horizontal' for side-by-side, 'vertical' for stacked
        
    row_height : float, optional
        Height per interaction row in inches
        
    max_height : float, optional
        Maximum figure height in inches
        
    figsize : tuple, optional
        Overall figure size (width, height). Overrides row_height if provided
        
    save : str, optional
        Path to save figure
        
    verbosity : int, default=2
        Verbosity level
        
    return_fig : bool, default=False
        If True, return the figure and axes objects
    
    Returns
    -------
    tuple or None
        (fig, axes) if return_fig=True, otherwise None
    
    Examples
    --------
    >>> adata_combined = la.pl.prepareDotPlotAdata(lr_adata, adata)
    >>> la.pl.plotLRDotPlot(
    ...     adata_combined,
    ...     interactions_to_plot=['CXCL13::CXCR5', 'CD40LG::CD40'],
    ...     groupby='cell_type',
    ...     delimiter='::',
    ...     orientation='vertical',
    ...     standard_scale_interaction='var',
    ...     standard_scale_ligand=None,
    ...     save='lr_dotplot.pdf'
    ... )
    """
    # Split interactions into ligands and receptors using delimiter
    ligands = [interaction.split(delimiter)[0] for interaction in interactions_to_plot]
    receptors = [interaction.split(delimiter)[1] for interaction in interactions_to_plot]

    # Compute maximum fractions
    max_frac_ligands = math.ceil(
        _compute_max_fraction(adata_dotplot, ligands, groupby) * 10
    ) / 10.0
    max_frac_receptors = math.ceil(
        _compute_max_fraction(adata_dotplot, receptors, groupby) * 10
    ) / 10.0
    max_frac_interactions = math.ceil(
        _compute_max_fraction(adata_dotplot, interactions_to_plot, groupby) * 10
    ) / 10.0

    common_dot_max = max(max_frac_ligands, max_frac_receptors, max_frac_interactions)

    # Determine figure size
    n_interactions = len(interactions_to_plot)
    
    titles = ["LR interaction score", "Ligands", "Receptors"]
    
    if figsize is not None:
        fig_width, fig_height = figsize
    else:
        if orientation == 'horizontal':
            fig_width = 18
            fig_height = n_interactions * (row_height or 1.0)
            if max_height is not None:
                fig_height = min(fig_height, max_height)
        else:  # vertical
            fig_width = 8
            fig_height = n_interactions * (row_height or 1.2) * 3 + 2
            if max_height is not None:
                fig_height = min(fig_height, max_height)

    # Create figure with appropriate layout
    if orientation == 'horizontal':
        fig, axes = plt.subplots(ncols=3, figsize=(fig_width, fig_height))
        
        # Plot all three with updated legend titles
        sc.pl.dotplot(
            adata_dotplot,
            var_names=interactions_to_plot,
            groupby=groupby,
            standard_scale=standard_scale_interaction,
            cmap=cmap_interaction,
            swap_axes=True,
            dot_max=common_dot_max,
            ax=axes[0],
            show=False,
            colorbar_title='Mean interaction\nscore',
            size_title='Fraction of\ncells (%)'
        )

        sc.pl.dotplot(
            adata_dotplot,
            var_names=ligands,
            groupby=groupby,
            standard_scale=standard_scale_ligand,
            cmap=cmap_ligand,
            swap_axes=True,
            dot_max=common_dot_max,
            ax=axes[1],
            show=False,
            colorbar_title='Mean ligand\nexpression',
            size_title='Fraction of\ncells (%)'
        )

        sc.pl.dotplot(
            adata_dotplot,
            var_names=receptors,
            groupby=groupby,
            standard_scale=standard_scale_receptor,
            cmap=cmap_receptor,
            swap_axes=True,
            dot_max=common_dot_max,
            ax=axes[2],
            show=False,
            colorbar_title='Mean receptor\nexpression',
            size_title='Fraction of\ncells (%)'
        )

        # Add titles above each plot
        for ax, title in zip(axes, titles):
            pos = ax.get_position()
            x_center = pos.x0 + pos.width / 2
            y_top = pos.y1 + 0.02
            fig.text(x_center, y_top, title, ha='center', va='bottom', 
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
    else:  # vertical
        # Create figure with subplots
        fig, axes = plt.subplots(nrows=3, figsize=(fig_width, fig_height))
        
        cmaps = [cmap_interaction, cmap_ligand, cmap_receptor]
        var_names_list = [interactions_to_plot, ligands, receptors]
        standard_scales = [standard_scale_interaction, standard_scale_ligand, standard_scale_receptor]
        colorbar_titles = ['Mean interaction\nscore', 'Mean ligand\nexpression', 'Mean receptor\nexpression']
        
        for i in range(3):
            sc.pl.dotplot(
                adata_dotplot,
                var_names=var_names_list[i],
                groupby=groupby,
                standard_scale=standard_scales[i],
                cmap=cmaps[i],
                swap_axes=True,
                dot_max=common_dot_max,
                ax=axes[i],
                show=False,
                colorbar_title=colorbar_titles[i],
                size_title='Fraction of\ncells (%)'
            )
        
        # Add titles in the upper left corner of each row
        for ax, title in zip(axes, titles):
            ax.text(-0.15, 1.05, title, transform=ax.transAxes,
                   fontsize=12, fontweight='bold',
                   ha='left', va='bottom')
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4, left=0.15)

    _save_figure(fig, save, verbosity)
    plt.show()

    if return_fig:
        return fig, axes
    return None


# ============================================================================
# SPATIAL VISUALIZATION FUNCTIONS
# ============================================================================

def plotCCCSpatial(
    lr_adata: ad.AnnData,
    basis: str,
    interaction: str,
    cell_type: str,
    selected_cell_types: Optional[List[str]] = None,
    highlight_all_expressing: bool = False,
    background_color: str = 'lightgrey',
    colors: Optional[List[str]] = None,
    size: float = 120,
    fig_width: Optional[float] = 6,
    fig_height: Optional[float] = None,
    max_title_chars: int = 60,
    save: Optional[str] = None,
    verbosity: int = 2,
    return_fig: bool = False
) -> Optional[plt.Figure]:
    """
    Plot spatial distribution of ligand-receptor interactions.
    
    Creates a spatial plot highlighting cells expressing a specific interaction, 
    with options to highlight specific cell types or all expressing cells.
    
    Parameters
    ----------
    lr_adata : anndata.AnnData
        AnnData object containing LR interaction scores and spatial coordinates
        
    basis : str
        Key for spatial coordinates in lr_adata.obsm
        
    interaction : str
        Interaction name to visualize (must be in lr_adata.var_names)
        
    cell_type : str
        Column name in lr_adata.obs containing cell type annotations
        
    selected_cell_types : list of str, optional
        Specific cell types to highlight
        
    highlight_all_expressing : bool, default=False
        If True, highlight all cells expressing the interaction
        
    background_color : str, default='lightgrey'
        Color for non-expressing cells
        
    colors : list of str, optional
        Colors for selected cell types
        
    size : float, default=120
        Point size for spatial plot
        
    fig_width : float, default=6
        Figure width in inches. Height is calculated to maintain data aspect ratio
        
    fig_height : float, optional
        Figure height in inches. If provided, overrides aspect ratio calculation
        
    max_title_chars : int, default=60
        Maximum characters per line in title before wrapping
        
    save : str, optional
        Path to save figure
        
    verbosity : int, default=2
        Verbosity level
        
    return_fig : bool, default=False
        If True, return the figure object
    
    Returns
    -------
    fig or None
        The figure object if return_fig=True, otherwise None
    
    Examples
    --------
    >>> la.pl.plotCCCSpatial(
    ...     lr_adata,
    ...     basis='X_spatial',
    ...     interaction='CXCL13::CXCR5',
    ...     cell_type='cell_type',
    ...     selected_cell_types=['B_cell', 'T_cell'],
    ...     colors=['green', 'orange'],
    ...     fig_width=10,
    ...     save='spatial.pdf'
    ... )
    """
    # Save original rcParams to restore later
    original_figsize = plt.rcParams['figure.figsize'].copy()
    
    try:
        # Check if interaction exists
        if interaction not in lr_adata.var_names:
            _log_message(
                f"{interaction} not found in lr_adata.var_names.",
                1, verbosity, 'error'
            )
            return None

        # Compute expression mask
        gene_idx = lr_adata.var_names.get_loc(interaction)
        mask = lr_adata.X[:, gene_idx] != 0
        if scipy.sparse.issparse(mask):
            mask = mask.toarray().flatten()

        # Mode 1: Highlight all expressing cells
        if highlight_all_expressing:
            lr_adata.obs['interaction_highlight'] = 'background'
            lr_adata.obs.loc[mask, 'interaction_highlight'] = lr_adata.obs.loc[mask, cell_type]

            full_categories = (
                lr_adata.obs[cell_type].cat.categories
                if pd.api.types.is_categorical_dtype(lr_adata.obs[cell_type])
                else pd.Categorical(lr_adata.obs[cell_type]).categories
            )

            lr_adata.obs['interaction_highlight'] = pd.Categorical(
                lr_adata.obs['interaction_highlight'],
                categories=['background'] + list(full_categories),
                ordered=True
            )

            sorted_idx = lr_adata.obs.sort_values('interaction_highlight').index
            lr_adata_sorted = lr_adata[sorted_idx].copy()

            palette = {'background': background_color}
            for i, ct in enumerate(full_categories):
                try:
                    palette[ct] = lr_adata.uns[f'{cell_type}_colors'][i]
                except (KeyError, IndexError):
                    _log_message(
                        f"Color for {ct} not found. Using default.",
                        2, verbosity, 'warning'
                    )
                    default_colors = plt.cm.get_cmap('tab10')(
                        np.linspace(0, 1, len(full_categories))
                    )
                    palette[ct] = default_colors[i]

            color_column = 'interaction_highlight'
            
            # Build informative title with wrapping
            n_expressing = mask.sum()
            title_text = f"{interaction}\nExpressing cells by cell type (n={n_expressing})"

        # Mode 2: Highlight specific cell types
        else:
            if selected_cell_types is None:
                _log_message(
                    "Either provide selected_cell_types or set highlight_all_expressing=True",
                    1, verbosity, 'error'
                )
                return None

            lr_adata.obs['custom_color'] = 'other'

            for ct in selected_cell_types:
                condition = (lr_adata.obs[cell_type] == ct) & mask
                lr_adata.obs.loc[condition, 'custom_color'] = ct

            order = ['other'] + selected_cell_types
            lr_adata.obs['custom_color'] = pd.Categorical(
                lr_adata.obs['custom_color'],
                categories=order,
                ordered=True
            )

            lr_adata_sorted = lr_adata[
                lr_adata.obs.sort_values('custom_color').index
            ].copy()

            if colors is None:
                colors = plt.cm.tab10(np.linspace(0, 1, len(selected_cell_types)))
            elif len(selected_cell_types) != len(colors):
                _log_message(
                    "Length of selected_cell_types and colors must match.",
                    1, verbosity, 'error'
                )
                return None

            palette = {'other': background_color}
            for ct, col in zip(selected_cell_types, colors):
                palette[ct] = col

            color_column = 'custom_color'
            
            # Build informative title with wrapping
            ct_counts = []
            for ct in selected_cell_types:
                condition = (lr_adata.obs[cell_type] == ct) & mask
                ct_counts.append(f"{ct}: {condition.sum()}")
            
            details = ', '.join(ct_counts)
            # Wrap long details
            wrapped_details = textwrap.fill(details, width=max_title_chars)
            title_text = f"{interaction}\nExpressing cells: {wrapped_details}"

        # Compute figure size based on data aspect ratio
        x_coords = lr_adata.obsm[basis][:, 0]
        y_coords = lr_adata.obsm[basis][:, 1]
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        
        if x_range == 0 or y_range == 0:
            # Default to square if data has no range
            aspect_ratio = 1.0
        else:
            aspect_ratio = y_range / x_range
        
        # Calculate figure size
        if fig_width is not None and fig_height is not None:
            # Both provided - use as is
            figsize = (fig_width, fig_height)
        elif fig_width is not None:
            # Only width provided - calculate height
            figsize = (fig_width, fig_width * aspect_ratio)
        elif fig_height is not None:
            # Only height provided - calculate width
            figsize = (fig_height / aspect_ratio, fig_height)
        else:
            # Neither provided - default width of 6
            figsize = (6, 6 * aspect_ratio)

        plt.rcParams['figure.figsize'] = figsize

        # Plot
        sc.pl.embedding(
            lr_adata_sorted,
            basis=basis,
            color=color_column,
            palette=palette,
            size=size,
            frameon=False,
            ncols=1,
            sort_order=False,
            title=title_text,
            show=False
        )

        # Get current figure
        fig = plt.gcf()

        # Save figure
        if save is not None:
            plt.savefig(save, bbox_inches='tight', dpi=300)
            _log_message(f"Figure saved to: {save}", 2, verbosity, 'info')

        plt.show()

        if return_fig:
            return fig
        return None
        
    finally:
        # Restore original rcParams
        plt.rcParams['figure.figsize'] = original_figsize


# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

def prepareDotPlotAdata(
    lr_adata: ad.AnnData,
    adata: ad.AnnData,
    verbosity: int = 2
) -> ad.AnnData:
    """
    Prepare combined AnnData for dot plot visualizations.
    
    Concatenates LR interaction scores with original gene expression data 
    horizontally to create a unified AnnData object for plotting.
    
    Parameters
    ----------
    lr_adata : anndata.AnnData
        AnnData object containing LR interaction scores
        
    adata : anndata.AnnData
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
    # Ensure sparse format
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


# ============================================================================
# MODULE API
# ============================================================================

__all__ = [
    # Main plotting functions
    'plotCCCHeatmap',
    'plotCCCNetwork',
    'plotCCCNetworkCumulative',
    'plotCCCDotPlot',
    'plotCCCDotPlotFacet',
    'plotLRDotPlot',
    'plotCCCSpatial',
    
    # Utility functions
    'prepareDotPlotAdata',
    
    # Custom colormap
    'pos_cmap',
]
