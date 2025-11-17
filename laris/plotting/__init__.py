"""
LARIS Plotting Module (laris.pl)

Visualization functions for LARIS (Ligand-Receptor Analysis in Resolved Imaging Systems) analysis results.

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
plotCCCDotPlotFacet : Create faceted dot plot by sending cell type
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
...     cmap='viridis',
...     n_top=1000,
...     filter_significant=True
... )
>>> 
>>> # Plot network for a specific cell type
>>> fig, ax = lr.pl.plotCCCNetwork(
...     laris_results,
...     cell_type_of_interest='B_cell',
...     interaction_direction='sending',
...     adata=adata
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

Author: LARIS Development Team
"""

# Import required packages
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
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
from scipy.sparse import csr_matrix, issparse
from typing import Optional, Union, List, Tuple


# Define custom colormap
cmap_own = cm.get_cmap('magma_r', 256)
newcolors = cmap_own(np.linspace(0, 0.75, 256))
Greys = cm.get_cmap('Greys_r', 256)
newcolors[:10, :] = Greys(np.linspace(0.8125, 0.8725, 10))
pos_cmap = colors.ListedColormap(newcolors)


# ============================================================================
# HEATMAP FUNCTIONS
# ============================================================================

# Added imports for clustering
try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def plotCCCHeatmap(
    laris_results, 
    cmap, 
    n_top=3000,
    fig_size=(12, 10),
    axis_label_fontsize=16,
    tick_fontsize=12,
    cbar_label_fontsize=16,
    cbar_tick_fontsize=12,
    filter_significant=False,
    p_value_col='p_value',
    threshold=0.05,
    show_borders=False, # --- DEFAULT CHANGED TO False ---
    cluster=False,      # --- NEW PARAMETER ---
    # --- New Parameters Below ---
    filter_by_interaction_score=True,
    threshold_interaction_score=0.01
):
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
        
    cmap : str or matplotlib.colors.Colormap
        Colormap to use for the heatmap (e.g., 'viridis', 'RdBu_r')
        
    n_top : int, default=3000
        Number of top-ranked interactions to include if filter_significant=False
        or if significance columns are missing.
        
    fig_size : tuple, default=(12, 10)
        Figure size as (width, height) in inches
        
    axis_label_fontsize : int, default=16
        Font size for axis labels
        
    tick_fontsize : int, default=12
        Font size for tick labels
        
    cbar_label_fontsize : int, default=16
        Font size for colorbar label
        
    cbar_tick_fontsize : int, default=12
        Font size for colorbar tick labels
        
    filter_significant : bool, default=False
        If True, filters interactions based on p_value_col and threshold.
        Falls back to 'significant' column if p_value_col is missing.
        
    p_value_col : str, default='p_value'
        Column name to use for p-value filtering when filter_significant=True.
        
    threshold : float, default=0.05
        P-value cutoff to use for filtering (interactions with p < threshold are kept).
        
    show_borders : bool, default=False
        If True, draws light grey border lines between each heatmap cell.
        Default is False (no lines).
        
    cluster : bool, default=False
        If True, performs hierarchical clustering on senders and applies the
        same order to receivers to maintain the self-interaction diagonal.
        Requires scipy.
    
    filter_by_interaction_score : bool, default=True
        If True, filters interactions based on the 'interaction_score' column.
        
    threshold_interaction_score : float, default=0.01
        Cutoff to use for score filtering. Assumes keeping interactions
        where 'interaction_score' > threshold_interaction_score.
    
    Returns
    -------
    None
        Displays the plot using plt.show()
    
    Examples
    --------
    >>> # Example with clustering (requires scipy)
    >>> lr.pl.plotCCCHeatmap(
    ...     laris_results,
    ...     cmap='viridis',
    ...     filter_significant=True,
    ...     cluster=True
    ... )
    
    >>> # Example with alphabetical order and no borders
    >>> lr.pl.plotCCCHeatmap(
    ...     laris_results,
    ...     cmap='RdBu_r'
    ... )
    
    Notes
    -----
    - Filtering logic:
      - If `filter_significant` is True, filters by `p_value_col` or 'significant' column.
      - If `filter_by_interaction_score` is True, filters by 'interaction_score' > `threshold_interaction_score`.
      - Both filters can be applied.
      - If *no* filters are applied (either set to False or if specified
        columns are missing), the plot will default to using `n_top` interactions.
    """
    
    laris_results_subset = laris_results.copy()
    did_filter = False

    # --- Updated filtering logic ---
    
    # 1. Try significance filter
    if filter_significant:
        if p_value_col in laris_results_subset.columns:
            laris_results_subset = laris_results_subset[laris_results_subset[p_value_col] < threshold]
            did_filter = True
        elif 'significant' in laris_results_subset.columns:
            laris_results_subset = laris_results_subset[laris_results_subset['significant']]
            did_filter = True
        else:
            print(f"Warning: 'filter_significant' is True but '{p_value_col}' and 'significant' "
                  "columns are missing. Skipping significance filter.")

    # 2. Try interaction score filter
    if filter_by_interaction_score:
        score_col = 'interaction_score' # Assuming this column name
        if score_col in laris_results_subset.columns:
            laris_results_subset = laris_results_subset[laris_results_subset[score_col] > threshold_interaction_score]
            did_filter = True
        else:
            print(f"Warning: 'filter_by_interaction_score' is True but "
                  f"'{score_col}' column is missing. Skipping score filter.")

    # 3. Fallback to n_top if no other filter was successfully applied
    if not did_filter:
        print(f"No filters were applied (or failed due to missing columns). "
              f"Defaulting to top {n_top} interactions.")
        laris_results_subset = laris_results_subset.iloc[:n_top]


    if laris_results_subset.empty:
        print("No interactions found matching the criteria. Cannot plot heatmap.")
        return None

    # --- Create pivot table ---
    heatmap_data_raw = laris_results_subset.pivot_table(
        index='sender',
        columns='receiver',
        aggfunc='size',
        fill_value=0
    )

    # --- Ensure square matrix with identical row/col order ---
    # 1. Get all unique cell types from both senders and receivers
    all_cell_types = sorted(
        list(
            set(heatmap_data_raw.index) | set(heatmap_data_raw.columns)
        )
    )
    
    # 2. Reindex the DataFrame to make it square (alphabetical order by default)
    heatmap_data = heatmap_data_raw.reindex(
        index=all_cell_types, 
        columns=all_cell_types, 
        fill_value=0
    )
    
    # --- NEW: Apply clustering if requested ---
    if cluster:
        if SCIPY_AVAILABLE:
            try:
                # Calculate linkage on rows (senders)
                row_linkage = linkage(pdist(heatmap_data, metric='euclidean'), method='average')
                # Get the order from the clustering
                new_order_indices = leaves_list(row_linkage)
                # Get the labels for that order
                new_order_labels = heatmap_data.index[new_order_indices]
                
                # Re-apply this order to BOTH rows and columns
                heatmap_data = heatmap_data.reindex(
                    index=new_order_labels, 
                    columns=new_order_labels
                )
            except Exception as e:
                print(f"Clustering failed: {e}. Falling back to alphabetical order.")
        else:
            print("Clustering requires scipy. Please install scipy to use this feature.")
            print("Falling back to alphabetical order.")
    # --- End of clustering logic ---


    # --- Set border parameters based on user request ---
    if show_borders:
        line_width_val = 0.5
        line_color_val = 'lightgrey'
    else:
        line_width_val = 0       # Default is now 0
        line_color_val = 'none'

    # --- Plot the heatmap ---
    plt.figure(figsize=fig_size)
    ax = sns.heatmap(
        heatmap_data, 
        cmap=cmap, 
        annot=False, 
        cbar=True,
        linewidths=line_width_val,  # Use new border width
        linecolor=line_color_val, # Use new border color
        square=True # Often looks better for this type of matrix
    )

    # Set axis labels
    plt.xlabel('Receiver', fontsize=axis_label_fontsize)
    plt.ylabel('Sender', fontsize=axis_label_fontsize)

    # Set tick label sizes
    plt.xticks(fontsize=tick_fontsize, rotation=90)
    plt.yticks(fontsize=tick_fontsize)

    # Set colorbar properties
    cbar = ax.collections[0].colorbar
    cbar.set_label("Number of interactions", fontsize=cbar_label_fontsize)
    cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
    
    plt.tight_layout() # Adjust plot to prevent labels from being cut off
    plt.show()
    return None

# ============================================================================
# NETWORK VISUALIZATION FUNCTIONS
# ============================================================================

def plotCCCNetwork(
    laris_results,
    cell_type_of_interest,
    interaction_direction="sending",
    adata=None,
    n_top=3000,
    interaction_multiplier=250,
    interaction_cutoff=0.0,
    cell_type="cell_type",
    cell_type_color_key="cell_type_colors",
    custom_color_mapping=None,
    figure_size=(10, 10),
    margins=0.2,
    label_font_size=16,
    node_size=1100,
    filter_significant=False
):
    """
    Plot an interaction network for a specific cell type.
    
    Creates a directed network graph showing aggregated interactions where a 
    specific cell type is either sending or receiving signals. Edge thickness 
    represents the cumulative interaction strength.
    
    Parameters
    ----------
    laris_results : pd.DataFrame
        DataFrame containing LARIS results with columns:
        - 'sender' : str
        - 'receiver' : str
        - 'interaction_score' : float
        
    cell_type_of_interest : str
        Cell type to focus on (e.g., "B_cell", "T_cell")
        
    interaction_direction : {'sending', 'receiving'}, default='sending'
        Direction to filter:
        - 'sending' : Show outgoing interactions from cell_type_of_interest
        - 'receiving' : Show incoming interactions to cell_type_of_interest
        
    adata : anndata.AnnData, optional
        AnnData object containing cell type information for coloring nodes
        
    n_top : int, default=3000
        Number of top interactions to consider
        
    interaction_multiplier : float, default=250
        Scaling factor for edge thickness based on interaction score
        
    interaction_cutoff : float, default=0.0
        Minimum interaction score threshold
        
    cell_type : str, default='cell_type'
        Column name in adata.obs containing cell type labels
        
    cell_type_color_key : str, default='cell_type_colors'
        Key in adata.uns containing cell type colors
        
    custom_color_mapping : dict, optional
        Custom mapping of cell types to colors, overrides adata colors
        
    figure_size : tuple, default=(10, 10)
        Figure size in inches
        
    margins : float, default=0.2
        Margin space around the plot
        
    label_font_size : int, default=16
        Font size for node labels
        
    node_size : int, default=1100
        Size of network nodes
        
    filter_significant : bool, default=False
        If True, only use significant interactions
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    
    Examples
    --------
    >>> fig, ax = lr.pl.plotCCCNetwork(
    ...     laris_results,
    ...     cell_type_of_interest='B_cell',
    ...     interaction_direction='sending',
    ...     adata=adata
    ... )
    
    Notes
    -----
    - Edges are drawn as curved arrows with thickness proportional to 
      cumulative interaction strength
    - Node colors match cell type colors from AnnData if available
    - Self-loops are automatically excluded
    """
    # Step 1: Subset and filter the DataFrame
    if filter_significant:
        if 'significant' in laris_results.columns:
            df_subset = laris_results[laris_results['significant']]
        else:
            print("'significant' column is missing. Run significance testing first. "
                  "Using manual cutoff for now.")
            df_subset = laris_results.iloc[:n_top]
    else:
        df_subset = laris_results.iloc[:n_top]
    
    # Filter for the cell type of interest based on direction
    if interaction_direction == "sending":
        df_filtered = df_subset[df_subset['sender'] == cell_type_of_interest]
    elif interaction_direction == "receiving":
        df_filtered = df_subset[df_subset['receiver'] == cell_type_of_interest]
    else:
        raise ValueError("interaction_direction must be either 'sending' or 'receiving'")
    
    # Apply interaction cutoff
    df_filtered = df_filtered[df_filtered['interaction_score'] >= interaction_cutoff]
    
    # Step 2: Group by cell type pairs and sum interaction scores
    df_grouped = df_filtered.groupby(
        ['sender', 'receiver'],
        as_index=False
    ).agg({'interaction_score': 'sum'})
    
    # Step 3: Build the network graph
    G = nx.from_pandas_edgelist(
        df_grouped,
        source='sender',
        target='receiver',
        edge_attr='interaction_score',
        create_using=nx.DiGraph()
    )
    
    # Add all cell type nodes if adata is provided
    if adata is not None:
        all_nodes = adata.obs[cell_type].unique()
        for node in all_nodes:
            if node not in G:
                G.add_node(node)
    elif custom_color_mapping is not None:
        for node in custom_color_mapping.keys():
            if node not in G:
                G.add_node(node)
    
    # Step 4: Compute layout
    pos = nx.circular_layout(G)
    
    # Define node colors
    if custom_color_mapping is not None:
        cell_type_to_color = custom_color_mapping
    elif adata is not None:
        cell_type_labels = adata.obs[cell_type].unique()
        cell_type_colors = adata.uns[cell_type_color_key]
        cell_type_to_color = dict(zip(cell_type_labels, cell_type_colors))
    else:
        cell_type_to_color = {}
    
    node_colors = [cell_type_to_color.get(node, 'gray') for node in G.nodes()]
    
    # Step 5: Create figure and draw nodes
    fig, ax = plt.subplots(figsize=figure_size)
    ax.margins(margins)
    
    node_collection = nx.draw_networkx_nodes(G, pos, node_size=node_size, 
                                              node_color=node_colors, ax=ax)
    node_collection.set_zorder(1)
    
    labels = nx.draw_networkx_labels(G, pos, font_size=label_font_size, 
                                      font_family="sans-serif", ax=ax)
    for label in labels.values():
        label.set_zorder(3)
    
    # Step 6: Draw edges
    for u, v, data in G.edges(data=True):
        sender_color = cell_type_to_color.get(u, 'gray')
        linewidth = data['interaction_score'] * interaction_multiplier
        
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
    
    ax.set_title(f"Interaction Network for {cell_type_of_interest} "
                 f"({interaction_direction.capitalize()} Interactions)")
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def plotCCCNetworkCumulative(
    laris_results=None,
    adata=None,
    cutoff=0,
    n_top=3000,
    cell_type="cell_type",
    cell_type_color_key="cell_type_colors",
    custom_color_mapping=None,
    figure_size=(10, 10),
    margins=0.2,
    label_font_size=16,
    node_size=1100,
    interaction_multiplier=5,
    filter_significant=False,
    edge_thickness_by_numbers=False,
    total_edge_thickness=100
):
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
        Minimum threshold for including an edge:
        - If edge_thickness_by_numbers=False: minimum cumulative score
        - If edge_thickness_by_numbers=True: minimum interaction count
        
    n_top : int, default=3000
        Number of top interactions to aggregate
        
    cell_type : str, default='cell_type'
        Column in adata.obs containing cell type labels
        
    cell_type_color_key : str, default='cell_type_colors'
        Key in adata.uns for cell type colors
        
    custom_color_mapping : dict, optional
        Custom cell type to color mapping
        
    figure_size : tuple, default=(10, 10)
        Figure dimensions in inches
        
    margins : float, default=0.2
        Plot margin size
        
    label_font_size : int, default=16
        Font size for node labels
        
    node_size : int, default=1100
        Size of network nodes
        
    interaction_multiplier : float, default=5
        Scaling factor for edge thickness (when using scores)
        
    filter_significant : bool, default=False
        If True, only include significant interactions
        
    edge_thickness_by_numbers : bool, default=False
        If True, edge thickness represents interaction count rather than 
        cumulative score
        
    total_edge_thickness : float, default=100
        Total thickness budget when edge_thickness_by_numbers=True
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    
    Examples
    --------
    >>> # Plot network with edge thickness by cumulative score
    >>> fig, ax = lr.pl.plotCCCNetworkCumulative(
    ...     laris_results,
    ...     adata=adata,
    ...     cutoff=0.5,
    ...     interaction_multiplier=10
    ... )
    >>> 
    >>> # Plot network with edge thickness by interaction count
    >>> fig, ax = lr.pl.plotCCCNetworkCumulative(
    ...     laris_results,
    ...     adata=adata,
    ...     edge_thickness_by_numbers=True,
    ...     total_edge_thickness=150
    ... )
    
    Notes
    -----
    - Self-loops (sender == receiver) are automatically removed
    - All cell types from adata are included as nodes, even if they have no edges
    - Edge colors match the sender cell type color
    """
    # Step 1: Subset and aggregate data
    if filter_significant:
        if 'significant' in laris_results.columns:
            df_subset = laris_results[laris_results['significant']]
        else:
            print("'significant' column is missing. Using manual cutoff.")
            df_subset = laris_results.iloc[:n_top]
    else:
        df_subset = laris_results.iloc[:n_top]

    if edge_thickness_by_numbers:
        # Aggregate by counting interactions
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
        # Aggregate by summing scores
        df_agg = (
            df_subset
            .groupby(['sender', 'receiver'], as_index=False)
            ['interaction_score']
            .sum()
        )
        df_agg = df_agg[df_agg['interaction_score'] >= cutoff]
        edge_attr_field = 'interaction_score'

    # Step 2: Build network graph
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
    unique_cell_types = adata.obs[cell_type].unique()
    for ctype in unique_cell_types:
        if ctype not in G:
            G.add_node(ctype)
    
    # Step 3: Determine node colors
    if custom_color_mapping is not None:
        cell_type_to_color = custom_color_mapping
    elif adata is not None:
        cell_type_labels = adata.obs[cell_type].unique()
        cell_type_colors = adata.uns[cell_type_color_key]
        cell_type_to_color = dict(zip(cell_type_labels, cell_type_colors))
    else:
        cell_type_to_color = {}
    
    node_colors = [cell_type_to_color.get(node, 'gray') for node in G.nodes()]
    
    # Step 4: Create figure and draw
    pos = nx.circular_layout(G)
    fig, ax = plt.subplots(figsize=figure_size)
    ax.margins(margins)
    
    node_collection = nx.draw_networkx_nodes(G, pos, node_size=node_size, 
                                              node_color=node_colors, ax=ax)
    node_collection.set_zorder(1)
    
    labels = nx.draw_networkx_labels(G, pos, font_size=label_font_size, 
                                      font_family="sans-serif", ax=ax)
    for label in labels.values():
        label.set_zorder(3)
    
    # Step 5: Draw edges
    for u, v, data in G.edges(data=True):
        sender_color = cell_type_to_color.get(u, 'gray')
        posA = pos[u]
        posB = pos[v]

        if edge_thickness_by_numbers:
            count = data['interaction_count']
            linewidth = (count / total_interaction_count) * total_edge_thickness
        else:
            linewidth = data['interaction_score'] * interaction_multiplier

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
    
    title_text = ("Interaction Network by " + 
                  ("Interaction Count" if edge_thickness_by_numbers 
                   else "Cumulative Interaction Score"))
    plt.title(title_text)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return fig, ax


# ============================================================================
# DOT PLOT FUNCTIONS
# ============================================================================

def plotCCCDotPlot(
    laris_results,
    sender_celltypes,
    receiver_celltypes,
    interactions_to_plot,
    n_top=3000,
    cmap=pos_cmap,
    bubble_size=250,
    filter_significant=False
):
    """
    Create a bubble plot for selected cell type pairs and interactions.
    
    Visualizes interaction strengths between specific sender-receiver cell type 
    pairs using bubbles where size can represent p-value significance and color 
    represents interaction score.
    
    Parameters
    ----------
    laris_results : pd.DataFrame
        DataFrame containing LARIS results
        
    sender_celltypes : list of str
        List of sending cell types to plot
        
    receiver_celltypes : list of str
        List of receiving cell types (must match length of sender_celltypes)
        
    interactions_to_plot : list of str
        List of interaction names to include (e.g., ['CXCL13::CXCR5', ...])
        
    n_top : int, default=3000
        Number of top interactions to consider
        
    cmap : matplotlib colormap, default=pos_cmap
        Colormap for bubble colors representing interaction scores
        
    bubble_size : float, default=250
        Base size for bubbles
        
    filter_significant : bool, default=False
        If True, only plot significant interactions
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    
    Examples
    --------
    >>> senders = ['B_cell', 'B_cell', 'T_cell']
    >>> receivers = ['T_cell', 'Macrophage', 'B_cell']
    >>> interactions = ['CXCL13::CXCR5', 'CD40LG::CD40']
    >>> 
    >>> fig, ax = lr.pl.plotCCCDotPlot(
    ...     laris_results,
    ...     sender_celltypes=senders,
    ...     receiver_celltypes=receivers,
    ...     interactions_to_plot=interactions
    ... )
    
    Notes
    -----
    - If 'p_value' column exists, bubble sizes are scaled:
      * p ≤ 0.01: full size
      * 0.01 < p ≤ 0.03: 70% size
      * 0.03 < p ≤ 0.05: 40% size
      * 0.05 < p ≤ 0.1: 10% size
    - Only non-zero interaction scores are plotted
    - Figure dimensions automatically adjust based on number of categories
    """
    # Filter for significant interactions if requested
    if filter_significant:
        if 'significant' in laris_results.columns:
            laris_results = laris_results[laris_results['significant']]
        else:
            print("'significant' column is missing. Run significance testing first. "
                  "Using manual cutoff for now.")
            if n_top is not None:
                laris_results = laris_results.sort_values(
                    'interaction_score', ascending=False).iloc[:n_top].copy()
    else:
        if n_top is not None:
            laris_results = laris_results.sort_values(
                'interaction_score', ascending=False).iloc[:n_top].copy()
    
    # Build mask for specified sender-receiver pairs
    mask_cell_pairs = None
    for sender, receiver in zip(sender_celltypes, receiver_celltypes):
        current_mask = ((laris_results['sender'] == sender) & 
                       (laris_results['receiver'] == receiver))
        if mask_cell_pairs is None:
            mask_cell_pairs = current_mask
        else:
            mask_cell_pairs |= current_mask

    # Filter for selected interactions and cell pairs
    df_filtered = laris_results[
        laris_results['interaction_name'].isin(interactions_to_plot) & mask_cell_pairs
    ].copy()

    # Remove missing scores
    df_filtered = df_filtered[df_filtered['interaction_score'].notna()]

    # Handle p-value based bubble sizing
    bubble_legend = False
    if 'p_value' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['p_value'] <= 0.1].copy()
        conditions = [
            (df_filtered['p_value'] <= 0.01),
            (df_filtered['p_value'] > 0.01) & (df_filtered['p_value'] <= 0.03),
            (df_filtered['p_value'] > 0.03) & (df_filtered['p_value'] <= 0.05),
            (df_filtered['p_value'] > 0.05) & (df_filtered['p_value'] <= 0.1)
        ]
        choices = [bubble_size, bubble_size * 0.7, bubble_size * 0.4, bubble_size * 0.1]
        df_filtered['bubble_size_plot'] = np.select(conditions, choices, default=bubble_size)
        bubble_legend = True
    else:
        print("p_value not found in results. Plotting with constant bubble size.")
        df_filtered['bubble_size_plot'] = bubble_size

    # Create cell type pair labels
    df_filtered['cell_type_pair'] = (df_filtered['sender'] + ' -> ' + 
                                      df_filtered['receiver'])

    # Define expected cell pairs in order
    all_cell_pairs = [f"{s} -> {r}" for s, r in zip(sender_celltypes, receiver_celltypes)]

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

    # Calculate figure size
    num_cell_pairs = len(all_cell_pairs)
    num_interactions = len(interactions_to_plot)
    fig_width = num_cell_pairs * 1.5
    fig_height = num_interactions * 1.1
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Plot bubbles
    sns.scatterplot(
        ax=ax,
        data=df_nonzero,
        x='cell_type_pair',
        y='interaction_name',
        hue='interaction_score',
        palette=cmap,
        s=df_nonzero['bubble_size_plot'],
        edgecolor='black',
        legend=False,
        clip_on=True,
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

    # Adjust layout
    plt.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.10)

    # Add colorbar
    if not df_nonzero.empty:
        max_score = df_nonzero['interaction_score'].max()
        norm = plt.Normalize(0, max_score)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, ticks=[0, max_score])
        cbar.ax.set_yticklabels(['0', 'max'])
        cbar.set_label('Interaction Score')

    # Add bubble legend if p-value scaling used
    if bubble_legend:
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', label='p_value ≤ 0.01',
                   markerfacecolor='gray', markersize=np.sqrt(bubble_size)),
            Line2D([0], [0], marker='o', color='w', label='0.01 < p_value ≤ 0.03',
                   markerfacecolor='gray', markersize=np.sqrt(bubble_size * 0.7)),
            Line2D([0], [0], marker='o', color='w', label='0.03 < p_value ≤ 0.05',
                   markerfacecolor='gray', markersize=np.sqrt(bubble_size * 0.4)),
            Line2D([0], [0], marker='o', color='w', label='0.05 < p_value ≤ 0.1',
                   markerfacecolor='gray', markersize=np.sqrt(bubble_size * 0.1))
        ]
        ax.legend(handles=legend_handles, title="", loc='upper right', 
                  bbox_to_anchor=(1.3, 1.0), frameon=False)

    plt.show()
    
    return fig, ax


def plotCCCDotPlotFacet(
    laris_results,
    cmap='pos_cmap',
    bubble_size=200,
    height=5,
    aspect_ratio=0.7,
    x_padding=0.4,
    y_padding=0.4,
    n_top=None,
    sender_celltypes=None,
    receiver_celltypes=None,
    interactions_to_plot=None,
    filter_significant=False
):
    """
    Create faceted bubble plots organized by sending cell type.
    
    Generates a grid of bubble plots where each facet represents a different 
    sending cell type, showing interactions to receiving cell types.
    
    Parameters
    ----------
    laris_results : pd.DataFrame
        DataFrame containing LARIS interaction results
        
    cmap : str or matplotlib colormap, default='pos_cmap'
        Colormap for bubble colors
        
    bubble_size : float, default=200
        Base size for bubbles
        
    height : float, default=5
        Height of each facet panel in inches
        
    aspect_ratio : float, default=0.7
        Aspect ratio (width = aspect × height) for each facet
        
    x_padding : float, default=0.4
        Extra space on x-axis to prevent clipping
        
    y_padding : float, default=0.4
        Extra space on y-axis to prevent clipping
        
    n_top : int, optional
        Number of top interactions to include
        
    sender_celltypes : list of str, optional
        Specific sending cell types to include. If None, all are included
        
    receiver_celltypes : list of str, optional
        Specific receiving cell types to include. If None, all are included
        
    interactions_to_plot : list of str, optional
        Specific interactions to include. If None, all are included
        
    filter_significant : bool, default=False
        If True, only plot significant interactions
    
    Returns
    -------
    g : seaborn.FacetGrid
        The FacetGrid object containing the plots
    
    Examples
    --------
    >>> g = lr.pl.plotCCCDotPlotFacet(
    ...     laris_results,
    ...     sender_celltypes=['B_cell', 'T_cell'],
    ...     receiver_celltypes=['Macrophage', 'NK_cell'],
    ...     interactions_to_plot=['CXCL13::CXCR5', 'CD40LG::CD40'],
    ...     filter_significant=True
    ... )
    
    Notes
    -----
    - Each facet shows one sending cell type's interactions
    - X-axis shows receiving cell types
    - Y-axis shows interaction names
    - Bubble color represents interaction score
    - Bubble size represents p-value significance (if available)
    - Colorbar and bubble size legend are added automatically
    """
    # Copy to avoid modifying original
    data = laris_results.copy()
    
    # Apply significance filtering if requested
    if filter_significant:
        if 'significant' in data.columns:
            data = data[data['significant']]
        else:
            print("'significant' column is missing. Run significance testing first. "
                  "Using manual cutoff for now.")
            if n_top is not None:
                data = data.iloc[:n_top]
    else:
        if n_top is not None:
            data = data.iloc[:n_top]
    
    # Subset based on user selections
    if sender_celltypes is not None:
        data = data[data["sender"].isin(sender_celltypes)]
    
    if receiver_celltypes is not None:
        data = data[data["receiver"].isin(receiver_celltypes)]
    
    if interactions_to_plot is not None:
        data = data[data["interaction_name"].isin(interactions_to_plot)]
    
    # Convert to categorical with specified order
    if sender_celltypes is not None:
        data["sender"] = pd.Categorical(
            data["sender"],
            categories=sender_celltypes,
            ordered=False
        )
    else:
        data["sender"] = data["sender"].astype("category")
    
    if receiver_celltypes is not None:
        data["receiver"] = pd.Categorical(
            data["receiver"],
            categories=receiver_celltypes,
            ordered=False
        )
    else:
        data["receiver"] = data["receiver"].astype("category")
    
    if interactions_to_plot is not None:
        data["interaction_name"] = pd.Categorical(
            data["interaction_name"],
            categories=interactions_to_plot,
            ordered=False
        )
    else:
        data["interaction_name"] = data["interaction_name"].astype("category")
    
    # Remove zero scores
    data_plot = data[data["interaction_score"] != 0].copy()
    
    # Handle p-value based bubble sizing
    if 'p_value' in data_plot.columns:
        data_plot = data_plot[data_plot["p_value"] <= 0.1].copy()
        conditions = [
            (data_plot['p_value'] <= 0.01),
            (data_plot['p_value'] > 0.01) & (data_plot['p_value'] <= 0.03),
            (data_plot['p_value'] > 0.03) & (data_plot['p_value'] <= 0.05),
            (data_plot['p_value'] > 0.05) & (data_plot['p_value'] <= 0.1)
        ]
        choices = [bubble_size, bubble_size * 0.7, bubble_size * 0.4, bubble_size * 0.1]
        data_plot['bubble_size_plot'] = np.select(conditions, choices, default=bubble_size)
        bubble_legend = True
    else:
        data_plot['bubble_size_plot'] = bubble_size
        bubble_legend = False
    
    # Create FacetGrid
    g = sns.FacetGrid(
        data_plot,
        col="sender",
        col_order=sender_celltypes,
        sharey=True,
        sharex=False,
        height=height,
        aspect=aspect_ratio,
    )
    
    # Define scatter plot for each facet
    def facet_scatter(data, **kwargs):
        sns.scatterplot(
            data=data,
            x="receiver",
            y="interaction_name",
            hue="interaction_score",
            palette=cmap,
            s=data["bubble_size_plot"],
            alpha=0.8,
            legend=False,
            edgecolor='black'
        )
    
    g.map_dataframe(facet_scatter)
    
    # Add padding
    xcats = data["receiver"].cat.categories
    ycats = data["interaction_name"].cat.categories
    
    for ax in g.axes.flatten():
        ax.set_xlim(-0.5 - x_padding, len(xcats) - 0.5 + x_padding)
        ax.set_ylim(-0.5 - y_padding, len(ycats) - 0.5 + y_padding)
    
    # Add colorbar
    if len(data_plot) > 0:
        vmin, vmax = data_plot["interaction_score"].min(), data_plot["interaction_score"].max()
    else:
        vmin, vmax = 0, 1
    
    g.fig.subplots_adjust(right=0.75)
    cbar_ax = g.fig.add_axes([0.78, 0.15, 0.02, 0.7])
    max_score = data['interaction_score'].max()
    norm = plt.Normalize(0, max_score)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, ticks=[0, max_score])
    cbar.ax.set_yticklabels(['0', 'max'])
    cbar.set_label("Interaction Score")
    
    # Add bubble legend if applicable
    if bubble_legend:
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', label='p_value ≤ 0.01',
                   markerfacecolor='gray', markersize=np.sqrt(bubble_size)),
            Line2D([0], [0], marker='o', color='w', label='0.01 < p_value ≤ 0.03',
                   markerfacecolor='gray', markersize=np.sqrt(bubble_size * 0.7)),
            Line2D([0], [0], marker='o', color='w', label='0.03 < p_value ≤ 0.05',
                   markerfacecolor='gray', markersize=np.sqrt(bubble_size * 0.4)),
            Line2D([0], [0], marker='o', color='w', label='0.05 < p_value ≤ 0.1',
                   markerfacecolor='gray', markersize=np.sqrt(bubble_size * 0.1))
        ]
        g.fig.legend(handles=legend_handles, title="", loc='upper left',
                     bbox_to_anchor=(0.85, 0.9), frameon=False)
    
    # Set axis labels and titles
    g.set_axis_labels("Receiving Cell Type", "Interactions")
    g.set_titles(col_template="{col_name}")
    
    # Rotate x-tick labels
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=45)
        for spine in ax.spines.values():
            spine.set_visible(True)
    
    plt.show()
    return g


def plotLRDotPlot(
    adata_dotplot,
    interactions_to_plot,
    groupby,
    cmap='Spectral_r',
    row_height=None,
    max_height=None,
    figsize_x=18,
    figsize_y=6
):
    """
    Create three side-by-side dot plots for LR pairs, ligands, and receptors.
    
    Visualizes expression patterns of ligand-receptor pairs alongside individual 
    ligand and receptor expression across cell types.
    
    Parameters
    ----------
    adata_dotplot : anndata.AnnData
        AnnData object prepared with prepareDotPlotAdata() containing both 
        LR pair scores and individual gene expression
        
    interactions_to_plot : list of str
        List of LR pairs in format 'ligand::receptor'
        
    groupby : str
        Column in adata_dotplot.obs to group by (typically cell types)
        
    cmap : str, default='Spectral_r'
        Colormap for the dot plots
        
    row_height : float, optional
        Height per interaction row in inches. If None, uses automatic sizing
        
    max_height : float, optional
        Maximum figure height in inches
        
    figsize_x : float, default=18
        Total figure width in inches
        
    figsize_y : float, default=6
        Figure height in inches (overrides row_height if provided)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : array of matplotlib.axes.Axes
        Array of three axes objects [LR_score, ligands, receptors]
    
    Examples
    --------
    >>> # Prepare combined AnnData
    >>> adata_combined = lr.pl.prepareDotPlotAdata(lr_adata, adata)
    >>> 
    >>> # Create dot plots
    >>> fig, axes = lr.pl.plotLRDotPlot(
    ...     adata_combined,
    ...     interactions_to_plot=['CXCL13::CXCR5', 'CD40LG::CD40'],
    ...     groupby='cell_type',
    ...     cmap='viridis'
    ... )
    
    Notes
    -----
    - First panel: diffused LR interaction scores
    - Second panel: ligand expression only
    - Third panel: receptor expression only
    - All three panels use the same dot size scale for comparability
    - Dot size represents fraction of expressing cells
    - Dot color represents standardized expression level
    """
    # Split interactions into ligands and receptors
    ligands = [interaction.split("::")[0] for interaction in interactions_to_plot]
    receptors = [interaction.split("::")[1] for interaction in interactions_to_plot]
    
    # Compute maximum fractions and round up to nearest 10%
    max_frac_ligands = math.ceil(_compute_max_fraction(adata_dotplot, ligands, groupby) * 10) / 10.0
    max_frac_receptors = math.ceil(_compute_max_fraction(adata_dotplot, receptors, groupby) * 10) / 10.0
    max_frac_interactions = math.ceil(_compute_max_fraction(adata_dotplot, interactions_to_plot, groupby) * 10) / 10.0
    
    # Use common maximum for consistent dot sizing
    common_dot_max = max(max_frac_ligands, max_frac_receptors, max_frac_interactions)
    
    # Determine figure height
    if figsize_y is not None:
        fig_height = figsize_y
    else:
        n_interactions = len(interactions_to_plot)
        fig_height = n_interactions * (row_height or 1.0)
        if max_height is not None:
            fig_height = min(fig_height, max_height)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(ncols=3, figsize=(figsize_x, fig_height))
    
    # Plot 1: Diffused LR scores
    sc.pl.dotplot(
        adata_dotplot,
        var_names=interactions_to_plot,
        groupby=groupby,
        standard_scale='var',
        cmap=cmap,
        swap_axes=True,
        dot_max=common_dot_max,
        ax=axes[0],
        show=False,
    )
    
    # Plot 2: Ligands
    sc.pl.dotplot(
        adata_dotplot,
        var_names=ligands,
        groupby=groupby,
        standard_scale='var',
        cmap=cmap,
        swap_axes=True,
        dot_max=common_dot_max,
        ax=axes[1],
        show=False
    )

    # Plot 3: Receptors
    sc.pl.dotplot(
        adata_dotplot,
        var_names=receptors,
        groupby=groupby,
        standard_scale='var',
        cmap=cmap,
        swap_axes=True,
        dot_max=common_dot_max,
        ax=axes[2],
        show=False
    )
    
    # Add titles
    fig.canvas.draw()
    titles = ["Diffused LR score", "Ligands", "Receptors"]

    for ax, title in zip(axes, titles):
        bbox = ax.get_tightbbox(fig.canvas.get_renderer())
        inv = fig.transFigure.inverted()
        bbox_fig = inv.transform(bbox)
        x_center = (bbox_fig[0, 0] + bbox_fig[1, 0]) / 2
        y_top = bbox_fig[1, 1]
        fig.text(x_center, y_top, title, ha='center', va='bottom', fontsize=14)

    plt.tight_layout()
    plt.show()
    
    return fig, axes


# ============================================================================
# SPATIAL VISUALIZATION FUNCTIONS
# ============================================================================

def plotCCCSpatial(
    lr_adata,
    basis,
    interaction,
    cell_type,
    selected_cell_types=None,
    highlight_all_expressing=False,
    background_color='lightgrey',
    colors=None,
    size=120,
    fig_size=None
):
    """
    Plot spatial distribution of ligand-receptor interactions.
    
    Creates a spatial plot highlighting cells expressing a specific interaction, 
    with options to highlight specific cell types or all expressing cells.
    
    Parameters
    ----------
    lr_adata : anndata.AnnData
        AnnData object containing LR interaction scores and spatial coordinates
        
    basis : str
        Key for spatial coordinates in lr_adata.obsm (e.g., 'X_spatial', 'X_umap')
        
    interaction : str
        Interaction name to visualize (must be in lr_adata.var_names)
        
    cell_type : str
        Column name in lr_adata.obs containing cell type annotations
        
    selected_cell_types : list of str, optional
        Specific cell types to highlight. If None and highlight_all_expressing=True,
        all expressing cells are highlighted with original colors
        
    highlight_all_expressing : bool, default=False
        If True, highlight all cells expressing the interaction using their 
        original cell type colors. Overrides selected_cell_types behavior
        
    background_color : str, default='lightgrey'
        Color for non-expressing cells
        
    colors : list of str, optional
        Colors for selected cell types (must match length of selected_cell_types)
        Only used when selected_cell_types is provided and highlight_all_expressing=False
        
    size : float, default=120
        Point size for spatial plot
        
    fig_size : tuple, optional
        Figure size as (width, height). If None, automatically computed 
        based on spatial coordinates
    
    Returns
    -------
    None
        Displays the plot using plt.show()
    
    Examples
    --------
    >>> # Highlight specific cell types
    >>> lr.pl.plotCCCSpatial(
    ...     lr_adata,
    ...     basis='X_spatial',
    ...     interaction='CXCL13::CXCR5',
    ...     cell_type='cell_type',
    ...     selected_cell_types=['B_cell', 'T_cell'],
    ...     colors=['green', 'orange']
    ... )
    >>> 
    >>> # Highlight all expressing cells with original colors
    >>> lr.pl.plotCCCSpatial(
    ...     lr_adata,
    ...     basis='X_spatial',
    ...     interaction='CXCL13::CXCR5',
    ...     cell_type='cell_type',
    ...     highlight_all_expressing=True
    ... )
    
    Notes
    -----
    - Expression is defined as non-zero values in the LR interaction score matrix
    - Background cells (non-expressing) are always drawn first for proper layering
    - Figure dimensions automatically maintain spatial aspect ratio if not specified
    - Modifies lr_adata.obs in place (adds temporary color columns)
    """
    # Check if interaction exists
    if interaction not in lr_adata.var_names:
        raise ValueError(f"{interaction} not found in lr_adata.var_names.")

    # Compute expression mask
    gene_idx = lr_adata.var_names.get_loc(interaction)
    mask = lr_adata.X[:, gene_idx] != 0
    if scipy.sparse.issparse(mask):
        mask = mask.toarray().flatten()

    # Mode 1: Highlight all expressing cells with original cell type colors
    if highlight_all_expressing:
        # Add color group: default background, then cell_type where expressed
        lr_adata.obs['interaction_highlight'] = 'background'
        lr_adata.obs.loc[mask, 'interaction_highlight'] = lr_adata.obs.loc[mask, cell_type]

        # Set categorical order so background is plotted first
        full_categories = (lr_adata.obs[cell_type].cat.categories 
                          if pd.api.types.is_categorical_dtype(lr_adata.obs[cell_type])
                          else pd.Categorical(lr_adata.obs[cell_type]).categories)

        lr_adata.obs['interaction_highlight'] = pd.Categorical(
            lr_adata.obs['interaction_highlight'],
            categories=['background'] + list(full_categories),
            ordered=True
        )

        # Reorder AnnData for proper layer rendering
        sorted_idx = lr_adata.obs.sort_values('interaction_highlight').index
        lr_adata_sorted = lr_adata[sorted_idx].copy()

        # Build color palette: original colors + grey for background
        palette = {'background': background_color}
        for i, ct in enumerate(full_categories):
            palette[ct] = lr_adata.uns[f'{cell_type}_colors'][i]

        color_column = 'interaction_highlight'

    # Mode 2: Highlight specific selected cell types
    else:
        if selected_cell_types is None:
            raise ValueError("Either provide selected_cell_types or set "
                           "highlight_all_expressing=True")
        
        # Create custom color column
        lr_adata.obs['custom_color'] = 'other'
        
        # For each selected cell type, assign label if gene is expressed
        for ct in selected_cell_types:
            condition = (lr_adata.obs[cell_type] == ct) & mask
            lr_adata.obs.loc[condition, 'custom_color'] = ct

        # Force categorical order so "other" cells are drawn first
        order = ['other'] + selected_cell_types
        lr_adata.obs['custom_color'] = pd.Categorical(
            lr_adata.obs['custom_color'],
            categories=order,
            ordered=True
        )
        
        # Reorder AnnData
        lr_adata_sorted = lr_adata[lr_adata.obs.sort_values('custom_color').index].copy()
        
        # Build custom palette
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_cell_types)))
        elif len(selected_cell_types) != len(colors):
            raise ValueError("Length of selected_cell_types and colors must match.")
        
        palette = {'other': background_color}
        for ct, col in zip(selected_cell_types, colors):
            palette[ct] = col

        color_column = 'custom_color'

    # Compute figure size if not provided
    if fig_size is None:
        x_coords = lr_adata.obsm[basis][:, 0]
        y_coords = lr_adata.obsm[basis][:, 1]
        x_width = x_coords.max() - x_coords.min()
        y_width = y_coords.max() - y_coords.min()
        fig_size = (10, 10 * y_width / x_width)
    
    plt.rcParams['figure.figsize'] = fig_size

    # Plot the spatial embedding
    sc.pl.embedding(
        lr_adata_sorted,
        basis=basis,
        color=color_column,
        palette=palette,
        size=size,
        frameon=False,
        ncols=6,
        sort_order=False,
        title=interaction,
        show=False
    )
    
    # Reset default figsize
    plt.rcParams['figure.figsize'] = (4, 4)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def prepareDotPlotAdata(lr_adata, adata):
    """
    Prepare combined AnnData for dot plot visualizations.
    
    Concatenates LR interaction scores with original gene expression data 
    horizontally to create a unified AnnData object for plotting.
    
    Parameters
    ----------
    lr_adata : anndata.AnnData
        AnnData object containing LR interaction scores (from LARIS analysis)
        
    adata : anndata.AnnData
        Original AnnData object containing gene expression data
    
    Returns
    -------
    adata_dotplot : anndata.AnnData
        Combined AnnData object with:
        - X matrix: concatenated [lr_adata.X, adata.X]
        - var_names: concatenated variable names (LR pairs + genes)
        - obs: copied from adata
        - obsm: copied from adata
        - Sparse format for memory efficiency
    
    Examples
    --------
    >>> # After running LARIS analysis
    >>> lr_adata = lr.analyse_interaction(adata, ligand_receptor_pairs)
    >>> 
    >>> # Prepare for dot plot visualization
    >>> adata_combined = lr.pl.prepareDotPlotAdata(lr_adata, adata)
    >>> 
    >>> # Now can plot both LR scores and individual genes
    >>> lr.pl.plotLRDotPlot(
    ...     adata_combined,
    ...     interactions_to_plot=['CXCL13::CXCR5'],
    ...     groupby='cell_type'
    ... )
    
    Notes
    -----
    - Both input matrices are converted to dense format before concatenation
    - Result is converted back to sparse format for efficiency
    - obs and obsm must be identical between lr_adata and adata
    - Typically used after LARIS analysis and before creating dot plots
    """
    # Ensure both X matrices are dense
    dense_lr_X = lr_adata.X if isinstance(lr_adata.X, np.ndarray) else lr_adata.X.toarray()
    dense_adata_X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()

    # Concatenate horizontally
    combined_X = np.concatenate([dense_lr_X, dense_adata_X], axis=1)

    # Combine variable names
    combined_var_names = np.concatenate([lr_adata.var_names, adata.var_names], axis=0)

    # Create new AnnData object
    adata_dotplot = sc.AnnData(X=combined_X)
    adata_dotplot.var_names = combined_var_names.copy()

    # Copy obs and obsm from original AnnData
    adata_dotplot.obs = adata.obs.copy()
    adata_dotplot.obsm = adata.obsm.copy()

    # Convert to sparse format
    if not issparse(adata_dotplot.X):
        adata_dotplot.X = csr_matrix(adata_dotplot.X)
        print("Converted adata_dotplot.X to sparse format.")

    return adata_dotplot


def _compute_max_fraction(adata, genes, groupby):
    """
    Compute maximum expression fraction across groups.
    
    Internal utility function to calculate the maximum fraction of cells 
    expressing any gene across all groups. Used for dot plot size scaling.
    
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
        Maximum fraction of expressing cells (between 0 and 1)
    
    Notes
    -----
    - Expression is defined as value > 0
    - Used internally by plotLRDotPlot for consistent dot sizing
    - Private function (prefixed with _) not intended for direct use
    """
    max_frac = 0
    groups = adata.obs[groupby].unique()
    for gene in genes:
        for group in groups:
            subset = adata[adata.obs[groupby] == group]
            # Fraction of cells with expression > 0
            frac = (subset[:, gene].X > 0).mean()
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
