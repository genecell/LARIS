# Import packages

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib import colors, colorbar

cmap_own = cm.get_cmap('magma_r', 256)
newcolors = cmap_own(np.linspace(0,0.75 , 256))
Greys = cm.get_cmap('Greys_r', 256)
newcolors[:10, :] = Greys(np.linspace(0.8125, 0.8725, 10))
pos_cmap = colors.ListedColormap(newcolors)

import networkx as nx
import itertools
from matplotlib.patches import FancyArrowPatch
from collections import OrderedDict
from scipy.sparse import csr_matrix, issparse



# Define plotting functions



def plot_cell_interaction_heatmap(
    res_zonetalk, 
    pos_cmap, 
    n_top=3000,
    fig_size=(12, 10),
    axis_label_fontsize=16,
    tick_fontsize=12,
    cbar_label_fontsize=16,
    cbar_tick_fontsize=12,
    filter_significant=False  # optional parameter to filter by the 'significant' column
):
    """
    Plots a heatmap of cell type interactions from res_zonetalk DataFrame with customizable figure and font sizes.
    
    Parameters:
    - res_zonetalk: pandas DataFrame containing at least 'sending_celltype' and 'receiving_celltype' columns.
    - pos_cmap: colormap to be used for the heatmap.
    - n_top: number of top rows to subset from the DataFrame (default is 3000).
    - fig_size: tuple indicating the figure size (width, height) (default is (10, 10)).
    - axis_label_fontsize: font size for the x and y axis labels.
    - tick_fontsize: font size for the x and y tick labels.
    - cbar_label_fontsize: font size for the colorbar label.
    - cbar_tick_fontsize: font size for the colorbar tick labels.
    - filter_significant: bool, if True, checks for a 'significant' column in res_zonetalk and only uses rows where it's True.
                            If the column is absent, the heatmap is plotted without filtering.
    """
    # Check if significant filtering is requested
    if filter_significant:
        if 'significant' in res_zonetalk.columns:
            # Use only the significant rows (ignore n_top)
            res_zonetalk_subset = res_zonetalk[res_zonetalk['significant']]
        else:
            print("'significant' column is missing, run significance testing first to plot significant interactions, running with manual cutoff for now")
            res_zonetalk_subset = res_zonetalk.iloc[:n_top]
    else:
        res_zonetalk_subset = res_zonetalk.iloc[:n_top]

    # Create a pivot table to count occurrences of interactions
    heatmap_data = res_zonetalk_subset.pivot_table(
        index='sending_celltype',
        columns='receiving_celltype',
        aggfunc='size',
        fill_value=0
    )

    # Plot the heatmap
    plt.figure(figsize=fig_size)
    ax = sns.heatmap(heatmap_data, cmap=pos_cmap, annot=False, cbar=True)

    # Set axis labels with the specified font size
    plt.xlabel('Receiving Cell Type', fontsize=axis_label_fontsize)
    plt.ylabel('Sending Cell Type', fontsize=axis_label_fontsize)

    # Set tick labels for x and y axes with the specified font size
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    # Set the colorbar label and tick label font sizes
    cbar = ax.collections[0].colorbar
    cbar.set_label("Number of interactions", fontsize=cbar_label_fontsize)
    cbar.ax.tick_params(labelsize=cbar_tick_fontsize)

    plt.show()



def plot_interaction_network_interaction_of_interest(
    res_zonetalk,
    interaction_of_interest="MIF::CD74",
    adata=None,
    n_top=3000,
    interaction_multiplier=250,
    interaction_cutoff=0.0,
    cell_type_obs_field="cell_type_2",
    cell_type_color_key="cell_type_2_colors",
    custom_color_mapping=None,
    figure_size=(10, 10),
    margins=0.2,
    label_font_size=16,
    node_size=1100,
    filter_significant=False
):
    """
    Plot an interaction network graph with customizable options, always including all nodes.

    Parameters
    ----------
    res_zonetalk : pandas.DataFrame
        DataFrame containing interaction data. Must include columns:
        'interaction_name', 'sending_celltype', 'receiving_celltype', and 'interaction_score'.
        
    interaction_of_interest : str, optional
        The specific interaction to filter on (default is "MIF::CD74").
        
    adata : AnnData object, optional
        AnnData object containing cell type information. It should have:
            - adata.obs[cell_type_obs_field]: cell type labels.
            - adata.uns[cell_type_color_key]: list of colors corresponding to cell types.
        If `custom_color_mapping` is provided, this will be ignored.
        
    n_top : int, optional
        Number of rows to subset from the input DataFrame (default is 3000). This is only applied
        when filtering by significance is disabled or the 'significant' column is missing.
        
    interaction_multiplier : float, optional
        Multiplier applied to the interaction_score to set edge linewidth (default is 250).
        
    interaction_cutoff : float, optional
        Minimum interaction score to include an edge in the network (default is 0.0).
        
    cell_type_obs_field : str, optional
        The field name in adata.obs containing the cell type labels (default is "cell_type_2").
        
    cell_type_color_key : str, optional
        The key in adata.uns containing the list of colors (default is "cell_type_2_colors").
        
    custom_color_mapping : dict, optional
        Dictionary mapping cell types to colors. If provided, this mapping will override 
        the colors extracted from `adata`.
        
    figure_size : tuple, optional
        The size of the figure (width, height) in inches (default is (10, 10)).
        
    margins : float, optional
        The margin size for the axes (default is 0.2).
        
    label_font_size : int, optional
        Font size for the node labels (default is 16).
        
    node_size : int, optional
        Size of the nodes in the network (default is 1100).
        
    filter_significant : bool, optional
        If True, check for a 'significant' column in res_zonetalk and only use rows where it's True.
        If the column is missing, print a message and subset using the first n_top rows.
        
    Returns
    -------
    fig, ax : tuple
        The matplotlib Figure and Axes objects.
    """
    # Step 1: Subset and filter the DataFrame
    if filter_significant:
        if 'significant' in res_zonetalk.columns:
            # Use only the rows where 'significant' is True (ignoring n_top)
            df_subset = res_zonetalk[res_zonetalk['significant']]
        else:
            print("significant column is missing, run significance testing first to plot significant interactions, running with manual cutoff for now")
            df_subset = res_zonetalk.iloc[:n_top]
    else:
        df_subset = res_zonetalk.iloc[:n_top]
    
    # Filter for the interaction of interest
    df_filtered = df_subset[df_subset['interaction_name'] == interaction_of_interest]
    
    # Apply cutoff filter: remove interactions with a score below the threshold.
    df_filtered = df_filtered[df_filtered['interaction_score'] >= interaction_cutoff]
    
    # Step 2: Build the Network Graph from the filtered DataFrame
    G = nx.from_pandas_edgelist(
        df_filtered,
        source='sending_celltype',
        target='receiving_celltype',
        edge_attr='interaction_score',
        create_using=nx.DiGraph()
    )
    
    # Always add all nodes.
    # If an AnnData object is provided, use its cell type labels.
    if adata is not None:
        all_nodes = adata.obs[cell_type_obs_field].unique()
        for node in all_nodes:
            if node not in G:
                G.add_node(node)
    # Otherwise, if a custom color mapping is provided, add those nodes.
    elif custom_color_mapping is not None:
        for node in custom_color_mapping.keys():
            if node not in G:
                G.add_node(node)
    
    # Step 3: Compute layout for the graph
    pos = nx.circular_layout(G)
    
    # Define node colors using a custom mapping or from the AnnData object.
    if custom_color_mapping is not None:
        cell_type_to_color = custom_color_mapping
    elif adata is not None:
        cell_type_labels = adata.obs[cell_type_obs_field].unique()
        cell_type_colors = adata.uns[cell_type_color_key]
        cell_type_to_color = dict(zip(cell_type_labels, cell_type_colors))
    else:
        cell_type_to_color = {}  # Fallback to empty mapping
    
    # For nodes not found in the mapping, default to gray.
    node_colors = [cell_type_to_color.get(node, 'gray') for node in G.nodes()]
    
    # Step 4: Create figure and draw nodes and labels.
    fig, ax = plt.subplots(figsize=figure_size)
    ax.margins(margins)  # Extra padding so labels are not cut off
    
    node_collection = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, ax=ax)
    node_collection.set_zorder(1)
    
    labels = nx.draw_networkx_labels(G, pos, font_size=label_font_size, font_family="sans-serif", ax=ax)
    for label in labels.values():
        label.set_zorder(3)
    
    # Step 5: Draw edges (arrows) with proper curvature and scaled linewidths.
    for u, v, data in G.edges(data=True):
        sender_color = cell_type_to_color.get(u, 'gray')
        linewidth = data['interaction_score'] * interaction_multiplier
        
        posA = pos[u]
        posB = pos[v]
        connection_style = "arc3,rad=0.1"  # Minimal curvature.
        shrinkA = 10  # Adjust arrow endpoints near node boundaries.
        shrinkB = 10
        
        arrow = FancyArrowPatch(
            posA=posA,
            posB=posB,
            arrowstyle="-|>",           # Full, filled arrowhead.
            connectionstyle=connection_style,
            mutation_scale=40,          # Controls arrowhead size.
            color=sender_color,
            linewidth=linewidth,
            shrinkA=shrinkA,
            shrinkB=shrinkB,
            zorder=2                   # Draw arrows above nodes but below labels.
        )
        ax.add_patch(arrow)
    
    ax.set_title(f"Interaction Network for {interaction_of_interest}")
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    return fig, ax



def plot_interaction_network_for_cell_type(
    res_zonetalk,
    cell_type_of_interest,
    interaction_direction="sending",  # "sending" or "receiving"
    adata=None,
    n_top=3000,
    interaction_multiplier=250,
    interaction_cutoff=0.0,
    cell_type_obs_field="cell_type_2",
    cell_type_color_key="cell_type_2_colors",
    custom_color_mapping=None,
    figure_size=(10, 10),
    margins=0.2,
    label_font_size=16,
    node_size=1100,
    filter_significant=False  # New parameter for significance filtering
):
    """
    Plot an interaction network graph for a specific cell type, where for each pair of cell types 
    the interaction strengths are summed to determine the edge thickness.
    
    When a given cell type is chosen as the cell_type_of_interest, only interactions where that cell type 
    is either the sender or receiver (based on interaction_direction) are considered. For each pair of cell 
    types, all interaction scores are summed and the result sets the thickness of the connecting edge.
    
    Parameters
    ----------
    res_zonetalk : pandas.DataFrame
        DataFrame containing interaction data. Must include columns:
        'sending_celltype', 'receiving_celltype', and 'interaction_score'.
        
    cell_type_of_interest : str
        The cell type for which to plot interactions (e.g., "B_germinal_center").
        
    interaction_direction : str, optional
        Direction to filter interactions. Either "sending" (default) or "receiving".
        - "sending": Only interactions where cell_type_of_interest is the sender are used.
        - "receiving": Only interactions where cell_type_of_interest is the receiver are used.
        
    adata : AnnData object, optional
        AnnData object containing cell type information. It should have:
            - adata.obs[cell_type_obs_field]: cell type labels.
            - adata.uns[cell_type_color_key]: list of colors corresponding to cell types.
        If `custom_color_mapping` is provided, this mapping will override the colors from adata.
        
    n_top : int, optional
        Number of rows to subset from the input DataFrame (default is 3000). This is only applied when 
        significance filtering is disabled or the 'significant' column is missing.
        
    interaction_multiplier : float, optional
        Multiplier applied to the summed interaction_score to set the edge linewidth (default is 250).
        
    interaction_cutoff : float, optional
        Minimum interaction score for each individual interaction to be considered (default is 0.0).
        
    cell_type_obs_field : str, optional
        The field name in adata.obs containing the cell type labels (default is "cell_type_2").
        
    cell_type_color_key : str, optional
        The key in adata.uns containing the list of colors (default is "cell_type_2_colors").
        
    custom_color_mapping : dict, optional
        Dictionary mapping cell types to colors. If provided, this mapping will override the colors from adata.
        
    figure_size : tuple, optional
        The size of the figure (width, height) in inches (default is (10, 10)).
        
    margins : float, optional
        The margin size for the axes (default is 0.2).
        
    label_font_size : int, optional
        Font size for the node labels (default is 16).
        
    node_size : int, optional
        Size of the nodes in the network (default is 1100).
        
    filter_significant : bool, optional
        If True, check for a 'significant' column in df and only use rows where it's True.
        If the column is missing, print a message and subset using the first n_subset rows.
        
    Returns
    -------
    fig, ax : tuple
        The matplotlib Figure and Axes objects.
    """

    # Step 1: Subset and filter the DataFrame.
    if filter_significant:
        if 'significant' in res_zonetalk.columns:
            df_subset = res_zonetalk[v['significant']]
        else:
            print("significant column is missing, run significance testing first to plot significant interactions, running with manual cutoff for now")
            df_subset = res_zonetalk.iloc[:n_top]
    else:
        df_subset = res_zonetalk.iloc[:n_top]
    
    # Filter for the cell type of interest based on the interaction_direction.
    if interaction_direction == "sending":
        df_filtered = df_subset[df_subset['sending_celltype'] == cell_type_of_interest]
    elif interaction_direction == "receiving":
        df_filtered = df_subset[df_subset['receiving_celltype'] == cell_type_of_interest]
    else:
        raise ValueError("interaction_direction must be either 'sending' or 'receiving'")
    
    # Filter out interactions that do not meet the cutoff.
    df_filtered = df_filtered[df_filtered['interaction_score'] >= interaction_cutoff]
    
    # Step 2: Group interactions by cell type pairs and sum the interaction scores.
    df_grouped = df_filtered.groupby(
        ['sending_celltype', 'receiving_celltype'],
        as_index=False
    ).agg({'interaction_score': 'sum'})
    
    # Step 3: Build the network graph using the grouped data.
    G = nx.from_pandas_edgelist(
        df_grouped,
        source='sending_celltype',
        target='receiving_celltype',
        edge_attr='interaction_score',
        create_using=nx.DiGraph()
    )
    
    # Always add all nodes, if available from adata or custom mapping.
    if adata is not None:
        all_nodes = adata.obs[cell_type_obs_field].unique()
        for node in all_nodes:
            if node not in G:
                G.add_node(node)
    elif custom_color_mapping is not None:
        for node in custom_color_mapping.keys():
            if node not in G:
                G.add_node(node)
    
    # Step 4: Compute the layout.
    pos = nx.circular_layout(G)
    
    # Define node colors using either a custom mapping or from adata.
    if custom_color_mapping is not None:
        cell_type_to_color = custom_color_mapping
    elif adata is not None:
        cell_type_labels = adata.obs[cell_type_obs_field].unique()
        cell_type_colors = adata.uns[cell_type_color_key]
        cell_type_to_color = dict(zip(cell_type_labels, cell_type_colors))
    else:
        cell_type_to_color = {}
    
    # Default to gray if the node's color is not found.
    node_colors = [cell_type_to_color.get(node, 'gray') for node in G.nodes()]
    
    # Step 5: Create figure, draw nodes and labels.
    fig, ax = plt.subplots(figsize=figure_size)
    ax.margins(margins)
    
    node_collection = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, ax=ax)
    node_collection.set_zorder(1)
    
    labels = nx.draw_networkx_labels(G, pos, font_size=label_font_size, font_family="sans-serif", ax=ax)
    for label in labels.values():
        label.set_zorder(3)
    
    # Step 6: Draw edges using the summed interaction scores for linewidth.
    for u, v, data in G.edges(data=True):
        sender_color = cell_type_to_color.get(u, 'gray')
        # The summed interaction score determines the linewidth.
        linewidth = data['interaction_score'] * interaction_multiplier
        
        posA = pos[u]
        posB = pos[v]
        connection_style = "arc3,rad=0.1"
        shrinkA = 10
        shrinkB = 10
        
        arrow = FancyArrowPatch(
            posA=posA,
            posB=posB,
            arrowstyle="-|>",
            connectionstyle=connection_style,
            mutation_scale=40,
            color=sender_color,
            linewidth=linewidth,
            shrinkA=shrinkA,
            shrinkB=shrinkB,
            zorder=2
        )
        ax.add_patch(arrow)
    
    ax.set_title(f"Interaction Network for {cell_type_of_interest} ({interaction_direction.capitalize()} Interactions)")
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    return fig, ax



def plot_cumulative_interaction_network(
    res_zonetalk,
    adata,
    cutoff,
    n_top=3000,
    cell_type_col="cell_type_2",
    custom_color_dict=None,
    figure_size=(10, 10),
    margins=0.2,
    label_font_size=16,
    node_size=1100,
    interaction_multiplier=5,
    filter_significant=False 
):
    """
    Plots a cumulative interaction network based on cell-cell interaction scores,
    ensuring that all cell types (nodes) are visible even if they do not have edges
    above the cutoff threshold.

    Parameters:
        res_zonetalk (pd.DataFrame): DataFrame containing interaction data.
        adata (AnnData): Annotated data object containing cell type info in .obs and .uns.
        cutoff (float): Minimum cumulative interaction score required to plot an edge.
        n_top (int): Number of rows to use from res_zonetalk for aggregation. This is only
                        applied when significance filtering is disabled or the 'significant'
                        column is missing.
        cell_type_col (str): The key in adata.obs for cell type labels.
        custom_color_dict (dict, optional): Mapping from cell type labels to colors. Overrides
                                            colors from adata.uns if provided.
        figure_size (tuple): Size of the matplotlib figure.
        margins (float): Margin for the axes (extra padding to prevent labels from being cut off).
        label_font_size (int): Font size for node labels.
        node_size (int): Size of the network nodes.
        interaction_multiplier (float): Multiplier to scale the interaction score for arrow linewidth.
        filter_significant (bool): If True, checks for a 'significant' column in res_zonetalk and
                                   uses only rows where it's True. If the column is missing, a message
                                   is printed and n_top is applied.

    Returns:
        fig, ax: Matplotlib figure and axes objects.
    """

    # ---------------------------
    # Step 1: Subset and Aggregate Data
    # ---------------------------
    if filter_significant:
        if 'significant' in res_zonetalk.columns:
            df_subset = res_zonetalk[res_zonetalk['significant']]
        else:
            print("significant column is missing, run significance testing first to plot significant interactions, running with manual cutoff for now")
            df_subset = res_zonetalk.iloc[:n_top]
    else:
        df_subset = res_zonetalk.iloc[:n_top]

    # Aggregate the interaction scores for each sender/receiver cell type pair.
    df_agg = (
        df_subset
        .groupby(['sending_celltype', 'receiving_celltype'], as_index=False)['interaction_score']
        .sum()
    )
    
    # Filter out interactions with cumulative scores below the cutoff.
    df_agg = df_agg[df_agg['interaction_score'] >= cutoff]
    
    # ---------------------------
    # Step 2: Build the Network Graph Using the Aggregated Data
    # ---------------------------
    G = nx.from_pandas_edgelist(
        df_agg,
        source='sending_celltype',
        target='receiving_celltype',
        edge_attr='interaction_score',
        create_using=nx.DiGraph()
    )
    
    # Ensure all cell type nodes from adata.obs are added to the graph,
    # even if they have no edges after filtering.
    unique_cell_types = adata.obs[cell_type_col].unique()
    for cell_type in unique_cell_types:
        if cell_type not in G:
            G.add_node(cell_type)
    
    # ---------------------------
    # Step 3: Determine Node Colors
    # ---------------------------
    if custom_color_dict is not None:
        cell_type_to_color = custom_color_dict
    else:
        # Construct the expected key for colors in adata.uns.
        colors_key = f"{cell_type_col}_colors"
        if colors_key in adata.uns:
            cell_type_colors = adata.uns[colors_key]
            cell_type_to_color = dict(zip(unique_cell_types, cell_type_colors))
        else:
            # Fallback: generate a colormap.
            cmap = cm.get_cmap('tab20', len(unique_cell_types))
            cell_type_to_color = {ctype: cmap(i) for i, ctype in enumerate(unique_cell_types)}
    
    node_colors = [cell_type_to_color.get(node, 'gray') for node in G.nodes()]
    
    # ---------------------------
    # Step 4: Compute Layout and Draw Nodes & Labels
    # ---------------------------
    pos = nx.circular_layout(G)
    fig, ax = plt.subplots(figsize=figure_size)
    ax.margins(margins)  # Extra padding so labels are not cut off
    
    # Draw the nodes.
    node_collection = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, ax=ax)
    node_collection.set_zorder(1)
    
    # Draw the labels.
    labels = nx.draw_networkx_labels(G, pos, font_size=label_font_size, font_family="sans-serif", ax=ax)
    for label in labels.values():
        label.set_zorder(3)
    
    # ---------------------------
    # Step 5: Draw Edges (Arrows) with Special Handling for Self-loops
    # ---------------------------
    for u, v, data in G.edges(data=True):
        sender_color = cell_type_to_color.get(u, 'gray')
        # Scale the linewidth based on the cumulative interaction score.
        linewidth = data['interaction_score'] * interaction_multiplier

        posA = pos[u]
        posB = pos[v]
        connection_style = "arc3,rad=0.1"  # Minimal curvature for all edges.
        shrinkA = 10  # Adjust arrow endpoints so they start near node boundaries.
        shrinkB = 10

        arrow = FancyArrowPatch(
            posA=posA,
            posB=posB,
            arrowstyle="-|>",
            connectionstyle=connection_style,
            mutation_scale=10,  # Controls arrowhead size.
            color=sender_color,
            linewidth=linewidth,
            shrinkA=shrinkA,
            shrinkB=shrinkB,
            zorder=2  # Draw arrows above nodes but below labels.
        )
        ax.add_patch(arrow)
    
    plt.title("Cumulative Interaction Network")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return fig, ax



def cell_type_to_cell_type_dot_plot(sender_celltypes, receiver_celltypes, interactions_to_plot, 
                                      n_top=3000, res_zonetalk=None, cmap=pos_cmap, bubble_size=250,
                                      filter_significant=False):
    """
    Create a bubble plot for selected interactions and cell type pairs.
    
    Parameters:
        sender_celltypes (list): List of sender cell types.
        receiver_celltypes (list): List of receiver cell types.
        interactions_to_plot (list): List of interaction names to plot.
        n_top (int or None): Number of top hits to use from the original DataFrame 
                                  (sorted by interaction_score in descending order). 
                                  If None, the entire DataFrame is used. Default is 3000.
        res_zonetalk (pd.DataFrame): DataFrame containing the interaction data. 
                                     If None, the function will try to use a global variable.
        cmap: A matplotlib colormap to use for the bubble colors.
        filter_significant (bool): If True, checks for a 'significant' column in res_zonetalk 
                                   and uses only rows where it's True. If the column is missing, 
                                   prints a message and falls back to n_top subsetting.
        
    Returns:
        fig, ax: The matplotlib figure and axis objects.
    """
    # Use the provided DataFrame or look for a global variable
    if res_zonetalk is None:
        try:
            res_zonetalk = globals()['res_zonetalk']
        except KeyError:
            raise ValueError("No DataFrame provided and 'res_zonetalk' not found in globals.")
    
    # Filter for significant interactions if requested.
    if filter_significant:
        if 'significant' in res_zonetalk.columns:
            res_zonetalk = res_zonetalk[res_zonetalk['significant']]
        else:
            print("significant column is missing, run significance testing first to plot significant interactions, running with manual cutoff for now")
            if top_n_topx_hits is not None:
                res_zonetalk = res_zonetalk.sort_values('interaction_score', ascending=False).iloc[:n_top].copy()
    else:
        if n_top is not None:
            res_zonetalk = res_zonetalk.sort_values('interaction_score', ascending=False).iloc[:n_top].copy()
    
    # Build a mask for the specified sender-receiver pairs
    mask_cell_pairs = None
    for sender, receiver in zip(sender_celltypes, receiver_celltypes):
        current_mask = (res_zonetalk['sending_celltype'] == sender) & (res_zonetalk['receiving_celltype'] == receiver)
        if mask_cell_pairs is None:
            mask_cell_pairs = current_mask
        else:
            mask_cell_pairs |= current_mask

    # Filter the DataFrame for the selected interactions and cell type pairs
    df_filtered = res_zonetalk[
        res_zonetalk['interaction_name'].isin(interactions_to_plot) & mask_cell_pairs
    ].copy()

    # Remove rows with missing interaction scores
    df_filtered = df_filtered[df_filtered['interaction_score'].notna()]

    # Create a column for labeling the cell type pair
    df_filtered['cell_type_pair'] = df_filtered['sending_celltype'] + ' -> ' + df_filtered['receiving_celltype']

    # Define the full set of expected cell type pairs in the desired order
    all_cell_pairs = [f"{s} -> {r}" for s, r in zip(sender_celltypes, receiver_celltypes)]

    # Force the categorical order so that all labels appear on the axes
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

    # Only plot bubbles for nonzero interaction scores
    df_nonzero = df_filtered[df_filtered['interaction_score'] > 0]

    # Calculate the number of categories for setting the figure size
    num_cell_pairs = len(all_cell_pairs)
    num_interactions = len(interactions_to_plot)

    fig_width = num_cell_pairs * 1.5  
    fig_height = num_interactions * 1.1
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Plot the bubbles for nonzero interaction scores
    sns.scatterplot(
        ax=ax,
        data=df_nonzero,
        x='cell_type_pair',
        y='interaction_name',
        hue='interaction_score',  # Bubble color represents interaction_score
        palette=cmap,          # Colormap to use
        s=bubble_size,                     # Bubble size
        edgecolor='black',
        legend=False,
        clip_on=True,
    )

    # Set the x- and y-ticks to display all expected categories
    ax.set_xticks(range(len(all_cell_pairs)))
    ax.set_xticklabels(all_cell_pairs, rotation=45, ha='right')
    ax.set_yticks(range(len(interactions_to_plot)))
    ax.set_yticklabels(interactions_to_plot)

    # Add extra padding so that bubbles don't overlap with the boundary
    x_padding = 0  # Adjust as needed
    y_padding = 0
    ax.set_xlim(-0.5 - x_padding, len(all_cell_pairs) - 0.5 + x_padding)
    ax.set_ylim(-0.5 - y_padding, len(interactions_to_plot) - 0.5 + y_padding)

    # Adjust subplot parameters to reduce whitespace
    plt.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.10)

    # Set up a custom colorbar (only if there are nonzero values)
    if not df_nonzero.empty:
        max_score = df_nonzero['interaction_score'].max()
        norm = plt.Normalize(0, max_score)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, ticks=[0, max_score])
        cbar.ax.set_yticklabels(['0', 'max'])
        cbar.set_label('Interaction Score')

    plt.tight_layout()
    plt.show()
    
    return fig, ax



def create_adata_dotplot(lr_adata, adata):
    """
    Concatenates the .X matrices (and var_names) from lr_adata and adata horizontally,
    and copies over obs and obsm from adata. Returns a new AnnData object.

    Parameters
    ----------
    lr_adata : AnnData
        The LR AnnData object.
    adata : AnnData
        The original AnnData object.

    Returns
    -------
    AnnData
        A new AnnData object with combined .X matrices and var_names, 
        and obs/obsm inherited from `adata`. Used for plotting functions of ZoneTalk.
    """
    # Ensure both X matrices are dense
    dense_lr_X = lr_adata.X if isinstance(lr_adata.X, np.ndarray) else lr_adata.X.toarray()
    dense_adata_X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()

    # Concatenate the dense matrices horizontally
    combined_X = np.concatenate([dense_lr_X, dense_adata_X], axis=1)

    # Combine the variable names
    combined_var_names = np.concatenate([lr_adata.var_names, adata.var_names], axis=0)

    # Create a new AnnData object
    adata_dotplot = sc.AnnData(X=combined_X)
    adata_dotplot.var_names = combined_var_names.copy()

    # Copy obs and obsm from the second AnnData object
    adata_dotplot.obs = adata.obs.copy()
    adata_dotplot.obsm = adata.obsm.copy()

    # Convert to sparse if it's not already
    if not issparse(adata_dotplot.X):
        adata_dotplot.X = csr_matrix(adata_dotplot.X)
        print("Converted adata_dotplot.X to sparse format.")

    return adata_dotplot


def LR_ligand_receptor_expression_dotplots(adata_dotplot, interactions_to_plot, groupby, cmap='Spectral_r', row_height=1.0, max_height=None, figsize_x=18, figsize_y=None):
    """
    Create three dotplots (Diffused LR score, Ligands, and Receptors) side-by-side.
    The overall figure dimensions can be dynamically adjusted based on the number of interactions,
    or customized directly via figsize_y (height) and figsize_x (width).

    Parameters:
    -----------
    adata_dotplot : AnnData
        AnnData object containing the data to plot.
    interactions_to_plot : list of str
        List of interactions, each formatted as "ligand::receptor".
    groupby : str, optional
        Key in adata_dotplot.obs to group the data by (default is 'cell_type_2').
    cmap : str, optional
        Colormap used for the dotplots (default is 'Spectral_r').
    row_height : float, optional
        Height (in inches) allocated per interaction. Default is 1.0 inch per row.
    max_height : float, optional
        Maximum allowed figure height. If None, no maximum is applied.
    figsize_y : float, optional
        Custom overall figure height in inches. If provided, this value overrides the computed height.
    figsize_x : float, optional
        Custom overall figure width in inches. Default is 18 inches.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the dotplots.
    axes : array of matplotlib.axes.Axes
        The array of axes objects for the subplots.
    """
    # Split the interactions into ligands and receptors.
    ligands = [interaction.split("::")[0] for interaction in interactions_to_plot]
    receptors = [interaction.split("::")[1] for interaction in interactions_to_plot]
    
    # Determine figure height.
    if figsize_y is not None:
        fig_height = figsize_y
    else:
        n_interactions = len(interactions_to_plot)
        fig_height = n_interactions * row_height
        if max_height is not None:
            fig_height = min(fig_height, max_height)
    
    # Create a figure with 3 subplots side-by-side.
    fig, axes = plt.subplots(ncols=3, figsize=(figsize_x, fig_height))
    
    # Dotplot for Diffused LR score (full interactions)
    sc.pl.dotplot(
        adata_dotplot, 
        var_names=interactions_to_plot, 
        groupby=groupby, 
        standard_scale='var', 
        cmap=cmap, 
        swap_axes=True,
        ax=axes[0],
        show=False
    )
    
    # Dotplot for Ligands
    sc.pl.dotplot(
        adata_dotplot, 
        var_names=ligands, 
        groupby=groupby, 
        standard_scale='var', 
        cmap=cmap, 
        swap_axes=True,
        ax=axes[1],
        show=False
    )
    
    # Dotplot for Receptors
    sc.pl.dotplot(
        adata_dotplot, 
        var_names=receptors, 
        groupby=groupby, 
        standard_scale='var', 
        cmap=cmap, 
        swap_axes=True,
        ax=axes[2],
        show=False
    )
    
    # Remove default titles and add custom annotations.
    titles = ["Diffused LR score", "Ligands", "Receptors"]
    for ax, title in zip(axes, titles):
        ax.set_title("")
        ax.annotate(
            title, 
            xy=(0.5, 0.75), 
            xycoords='axes fraction', 
            ha='center', 
            va='top', 
            fontsize=14,
        )
    
    plt.tight_layout()
    plt.show()
    
    return fig, axes



def facet_plot_of_interactions(
    res_zonetalk,
    cmap='pos_cmap',
    dot_size=200,
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
    Facet-style dot plot for res_zonetalk:

      - Facet by 'sending_celltype'
      - X-axis = 'receiving_celltype'
      - Y-axis = 'interaction_name'
      - Dot color = 'interaction_score'
      - Fixed dot size
      - Colorbar on the right

    Parameters
    ----------
    res_zonetalk : pd.DataFrame
        Must contain columns:
          'sending_celltype', 'receiving_celltype',
          'interaction_score', 'ligand', 'receptor', 'interaction_name'
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap for coloring the dots by 'interaction_score'.
        Default is 'pos_cmap'.
    dot_size : float, optional
        Size of the dots. Default is 200.
    height : float, optional
        Height (in inches) of each facet. Default is 5.
    aspect_ratio : float, optional
        Aspect ratio (width = aspect * height). Default is 0.7.
    x_padding : float, optional
        Extra space on the x-axis (left/right) so dots don't get clipped.
    y_padding : float, optional
        Extra space on the y-axis (top/bottom) so dots don't get clipped.
    n_top : int, optional
        If provided, only the top `n_top` rows of `df` (in its current order) 
        will be taken before plotting.
    sender_celltypes : list, optional
        List of sending cell types to keep. If None, all are kept.
    receiver_celltypes : list, optional
        List of receiving cell types to keep. If None, all are kept.
    interactions_to_plot : list, optional
        List of interaction names to keep. If None, all are kept.
    filter_significant : bool, optional
        If True, checks for a 'significant' column in df and uses only rows where it's True.
        If the column is missing, prints a message and falls back to subsetting with n_top.

    Returns
    -------
    g : sns.FacetGrid
        The FacetGrid object containing the plot.
    """
    # Copy to avoid modifying the original DataFrame
    data = res_zonetalk.copy()

    # --- Apply significance filtering if requested ---
    if filter_significant:
        if 'significant' in data.columns:
            data = data[data['significant']]
        else:
            print("significant column is missing, run significance testing first to plot significant interactions, running with manual cutoff for now")
            if n_top is not None:
                data = data.iloc[:n_top]
    else:
        # --- Subset the data based on n_top (if provided) ---
        if n_top is not None:
            data = data.iloc[:n_top]

    # --- Subset the data based on user selections ---
    if sender_celltypes is not None:
        data = data[data["sending_celltype"].isin(sender_celltypes)]

    if receiver_celltypes is not None:
        data = data[data["receiving_celltype"].isin(receiver_celltypes)]

    if interactions_to_plot is not None:
        data = data[data["interaction_name"].isin(interactions_to_plot)]

    # ---- Convert columns to categorical and set specified categories explicitly ----
    # This ensures that all user-requested categories appear on axes/facets, 
    # even if some have no data.
    if sender_celltypes is not None:
        data["sending_celltype"] = pd.Categorical(
            data["sending_celltype"],
            categories=sender_celltypes,
            ordered=False
        )
    else:
        data["sending_celltype"] = data["sending_celltype"].astype("category")

    if receiver_celltypes is not None:
        data["receiving_celltype"] = pd.Categorical(
            data["receiving_celltype"],
            categories=receiver_celltypes,
            ordered=False
        )
    else:
        data["receiving_celltype"] = data["receiving_celltype"].astype("category")

    if interactions_to_plot is not None:
        data["interaction_name"] = pd.Categorical(
            data["interaction_name"],
            categories=interactions_to_plot,
            ordered=False
        )
    else:
        data["interaction_name"] = data["interaction_name"].astype("category")

    # ---- Remove rows with interaction_score == 0 to avoid plotting "zero" dots ----
    data_plot = data[data["interaction_score"] != 0].copy()

    # Create a FacetGrid for each sender (enforcing the sender order, if specified)
    g = sns.FacetGrid(
        data_plot,
        col="sending_celltype",
        col_order=sender_celltypes,  # ensures we include all specified sending cell types
        sharey=True,   # share the same y categories (interactions)
        sharex=False,  # set to True if you want identical order of receiving_celltypes
        height=height,
        aspect=aspect_ratio
    )

    # Define the scatterplot for each facet
    def facet_scatter(data, **kwargs):
        sns.scatterplot(
            data=data,
            x="receiving_celltype",
            y="interaction_name",
            hue="interaction_score",
            palette=cmap,
            s=dot_size,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
            legend=False
        )

    g.map_dataframe(facet_scatter)

    # Add padding around the x and y axis so dots don't hit the frame
    # Use the full categorical range from 'data', not just 'data_plot',
    # so empty categories still appear on the axes.
    xcats = data["receiving_celltype"].cat.categories
    ycats = data["interaction_name"].cat.categories

    for ax in g.axes.flatten():
        ax.set_xlim(-0.5 - x_padding, len(xcats) - 0.5 + x_padding)
        ax.set_ylim(-0.5 - y_padding, len(ycats) - 0.5 + y_padding)

    # Create a colorbar for interaction_score
    # If everything is zero or we have no data, we need a fallback:
    if len(data_plot) > 0:
        vmin, vmax = data_plot["interaction_score"].min(), data_plot["interaction_score"].max()
    else:
        # Fallback to 0..1 if we have no data
        vmin, vmax = 0, 1

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Make space on the right for the colorbar
    g.fig.subplots_adjust(right=0.75)
    cbar_ax = g.fig.add_axes([0.78, 0.2, 0.02, 0.6])  # left, bottom, width, height
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Interaction Score")

    # Set axis labels and facet titles
    g.set_axis_labels("Receiving Cell Type", "Ligand::Receptor")
    g.set_titles(col_template="{col_name}")

    # Rotate the x-ticks for clarity and ensure all spines are visible
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=45)
        for spine in ax.spines.values():
            spine.set_visible(True)

    plt.show()
    return g