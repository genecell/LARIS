"""Dot plots of ligand-receptor interactions."""

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
                raise ValueError(
                    f"Invalid pair format: '{pair}'. Expected format: "
                    f"'sender{delimiter_pair}receiver'"
                )
            senders.append(parts[0].strip())
            receivers.append(parts[1].strip())

    # Validate inputs. Raise instead of print-and-return-None so failures
    # cannot be silently overlooked in batch scripts.
    if senders is None or receivers is None:
        raise ValueError(
            "Must provide either (senders, receivers) or sender_receiver_pairs"
        )

    if len(senders) != len(receivers):
        raise ValueError(
            f"Length of senders ({len(senders)}) and receivers "
            f"({len(receivers)}) must match: plotCCCDotPlot pairs them "
            f"element-wise. To plot every sender-receiver combination "
            f"(Cartesian product), use plotCCCDotPlotFacet instead."
        )

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
