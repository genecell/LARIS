"""Cell-cell communication network plots."""

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
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
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
from ..preprocessing._io import _ensure_expression_anndata, _is_cytome_source
from .._compat import _UNSET, resolve_data_arg

def _draw_edge_arrow(ax, pos, u, v, *, color, linewidth, mutation_scale,
                     node_size=1100, zorder=2, rad=0.1, shrink=10,
                     loop_radius=0.75):
    """Draw one network edge, including the self-interaction case.

    ``FancyArrowPatch`` cannot build a path between two identical points,
    so an edge with ``u == v`` collapses to zero length and disappears
    (issue #9). Self-interactions are meaningful in LARIS - autocrine and
    within-cell-type signalling is common - so they are drawn as a ring
    just outside the node, ending in an arrowhead that points back into
    it.

    The ring is an explicit circular arc rather than a heavily bowed
    ``arc3`` connection: a short chord with a large ``rad`` renders as a
    thin tendril, while an arc path gives a clean circle. It is sized in
    *data* units from the layout's own extent, because the axes limits
    are still being set while these patches are added, so a
    transform-based radius would evaluate against the wrong view.
    """
    if u != v:
        ax.add_patch(FancyArrowPatch(
            posA=pos[u], posB=pos[v], arrowstyle="-|>",
            connectionstyle=f"arc3,rad={rad}", mutation_scale=mutation_scale,
            color=color, linewidth=linewidth, shrinkA=shrink, shrinkB=shrink,
            zorder=zorder))
        return

    xy = np.asarray(pos[u], dtype=float)
    coords = np.asarray([pos[n] for n in pos], dtype=float)
    span = float(np.ptp(coords, axis=0).max()) if len(coords) > 1 else 1.0
    span = span or 1.0
    # Node markers are an area in points squared and the loop is in data
    # units, so the marker radius is approximated as a fraction of the
    # layout span, scaled by sqrt(node_size) so the loop tracks the marker
    # when the caller changes it. Calibrated on the default spring layout.
    r_node = 0.085 * span * float(np.sqrt(max(node_size, 1.0) / 1100.0))
    r_loop = loop_radius * r_node

    # place the ring away from the layout centre, clear of the interior
    away = xy - coords.mean(axis=0) if len(coords) > 1 else np.zeros(2)
    norm = float(np.hypot(*away))
    away = away / norm if norm > 1e-12 else np.array([0.0, 1.0])
    centre = xy + away * (r_node + r_loop * 0.85)

    # sweep almost the full circle, finishing next to the node so the
    # arrowhead points back at it
    to_node = np.degrees(np.arctan2(-away[1], -away[0]))
    arc = Path.arc(to_node + 30.0, to_node + 330.0)
    loop = Affine2D().scale(r_loop).translate(*centre).transform_path(arc)
    ax.add_patch(FancyArrowPatch(
        path=loop, arrowstyle="-|>", mutation_scale=mutation_scale,
        color=color, linewidth=linewidth, fill=False, zorder=zorder))


def plotCCCNetwork(
    laris_results: pd.DataFrame,
    cell_type_of_interest: str,
    interaction_direction: str = "sending",
    data=_UNSET,
    n_top: int = 3000,
    edge_width_scale: float = 30,
    interaction_cutoff: float = 0.0,
    groupby: str = "cell_type",
    cell_type_color_key: Optional[str] = None,
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
    return_fig: bool = False,
    adata=_UNSET,
    include_self_interactions: bool = True
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
        
    cell_type_color_key : str, optional
        Key in ``adata.uns`` holding the cell-type palette. Defaults to
        ``f"{groupby}_colors"`` (the scanpy convention). When the key is
        missing, a palette is generated automatically instead of raising.
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
        
    include_self_interactions : bool, default=True
        Draw interactions where the sender and the receiver are the same
        cell type, as a loop on that node. Autocrine and within-cell-type
        signalling is common, so these are shown by default; set False to
        restrict the plot to interactions between different cell types.
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
    adata = resolve_data_arg(data, 'plotCCCNetwork', canonical='data',
                             required=False, adata=adata)

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
    if not include_self_interactions:
        G.remove_edges_from(list(nx.selfloop_edges(G)))

    # Add all cell type nodes
    if adata is not None and not isinstance(adata, ad.AnnData):
        # cell-type labels and colours only: a cytome source is read here
        adata = _ensure_expression_anndata(adata)
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
        cell_type_to_color = _resolve_cell_type_colors(
            adata, groupby, cell_type_color_key
        )
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

        _draw_edge_arrow(ax, pos, u, v, color=sender_color,
                         linewidth=linewidth, mutation_scale=40,
                         node_size=node_size)

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
    data=_UNSET,
    cutoff: float = 0,
    n_top: int = 3000,
    groupby: str = "cell_type",
    cell_type_color_key: Optional[str] = None,
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
    return_fig: bool = False,
    adata=_UNSET,
    include_self_interactions: bool = True
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
        
    cell_type_color_key : str, optional
        Key in ``adata.uns`` holding the cell-type palette. Defaults to
        ``f"{groupby}_colors"`` (the scanpy convention). When the key is
        missing, a palette is generated automatically instead of raising.
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
    adata = resolve_data_arg(data, 'plotCCCNetworkCumulative',
                             canonical='data', adata=adata)

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

    # Self-interactions are kept: autocrine and within-cell-type signalling
    # is real, and _draw_edge_arrow renders those edges as loops. They used
    # to be deleted here, which (together with FancyArrowPatch collapsing on
    # posA == posB) is why they never appeared - issue #9. Pass
    # ``include_self_interactions=False`` to drop them.
    if not include_self_interactions:
        G.remove_edges_from(list(nx.selfloop_edges(G)))

    # Add all cell types as nodes (cytome sources are read here)
    if adata is not None and not isinstance(adata, ad.AnnData):
        adata = _ensure_expression_anndata(adata)
    unique_cell_types = adata.obs[groupby].unique()
    for ctype in unique_cell_types:
        if ctype not in G:
            G.add_node(ctype)

    # Determine node colors
    if custom_color_mapping is not None:
        cell_type_to_color = custom_color_mapping
    elif adata is not None:
        cell_type_to_color = _resolve_cell_type_colors(
            adata, groupby, cell_type_color_key
        )
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

        _draw_edge_arrow(ax, pos, u, v, color=sender_color,
                         linewidth=linewidth, mutation_scale=10,
                         node_size=node_size)

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
