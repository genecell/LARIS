# `laris.pl`: Plotting

<!-- Generated from the docstrings by tools/make_api_docs.py; do not edit by hand. -->

Heatmaps, networks, dot plots, spatial maps and comparison volcanoes.

| Function | Summary |
|---|---|
| [`plotCCCHeatmap`](#plotcccheatmap) | Create a heatmap showing the number of cell-cell communication interactions. |
| [`plotCCCNetwork`](#plotcccnetwork) | Plot an interaction network for a specific cell type. |
| [`plotCCCNetworkCumulative`](#plotcccnetworkcumulative) | Plot a cumulative interaction network across all cell types. |
| [`plotCCCDotPlot`](#plotcccdotplot) | Create a bubble plot for selected cell type pairs and interactions. |
| [`plotCCCDotPlotFacet`](#plotcccdotplotfacet) | Create faceted bubble plots organized by sending or receiving cell type. |
| [`plotLRDotPlot`](#plotlrdotplot) | Create three side-by-side (or stacked) dot plots for LR pairs, ligands, and receptors. |
| [`plotCCCSpatial`](#plotcccspatial) | Plot spatial distribution of ligand-receptor interactions. |
| [`plotCompareLARIS`](#plotcomparelaris) | Volcano plot of a `laris.tl.compareLARIS` result. |
| [`prepareDotPlotAdata`](#preparedotplotadata) | Prepare combined AnnData for dot plot visualizations. |

---

<a id="plotcccheatmap"></a>

### `la.pl.plotCCCHeatmap`

```python
la.pl.plotCCCHeatmap(
    laris_results,
    cmap='Purples',
    n_top=3000,
    figsize=(6, 5),
    axis_label_fontsize=16,
    tick_fontsize=12,
    cbar_label_fontsize=16,
    cbar_tick_fontsize=12,
    filter_significant=True,
    p_value_col='p_value_fdr',
    threshold=0.05,
    show_borders=True,
    cluster=False,
    filter_by_interaction_score=True,
    threshold_interaction_score=0.01,
    save=None,
    verbosity=2,
    return_fig=False,
)
```

Create a heatmap showing the number of cell-cell communication interactions.

This function visualizes the frequency of interactions between sending and 
receiving cell types as a heatmap. It ensures the row and column orders
are identical to maintain the self-interaction diagonal.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `laris_results` | pd.DataFrame | DataFrame containing LARIS results with columns:<br>• 'sender' : str, cell type sending the signal<br>• 'receiver' : str, cell type receiving the signal<br>• 'significant' : bool (optional), fallback significance flag<br>• p_value_col : str (optional), column name for p-values |
| `cmap` | str or matplotlib.colors.Colormap, default='Purples' | Colormap to use for the heatmap (e.g., 'viridis', 'RdBu_r') |
| `n_top` | int, default=3000 | Number of top-scoring interactions (by `interaction_score`) to include if no filters are applied |
| `figsize` | tuple, default=(6, 5) | Figure size as (width, height) in inches |
| `axis_label_fontsize` | int, default=16 | Font size for axis labels |
| `tick_fontsize` | int, default=12 | Font size for tick labels |
| `cbar_label_fontsize` | int, default=16 | Font size for colorbar label |
| `cbar_tick_fontsize` | int, default=12 | Font size for colorbar tick labels |
| `filter_significant` | bool, default=True | If True, filters interactions based on p_value_col and threshold |
| `p_value_col` | str, default='p_value_fdr' | Column name to use for p-value filtering |
| `threshold` | float, default=0.05 | P-value cutoff for filtering |
| `show_borders` | bool, default=True | If True, draws light grey border lines between heatmap cells |
| `cluster` | bool, default=False | If True, performs hierarchical clustering on cell types |
| `filter_by_interaction_score` | bool, default=True | If True, filters by interaction_score > threshold_interaction_score |
| `threshold_interaction_score` | float, default=0.01 | Cutoff for interaction score filtering |
| `save` | str, optional | Path to save the figure (e.g., 'heatmap.pdf'). If None, figure is not saved |
| `verbosity` | int, default=2 | Verbosity level (0=silent, 1=errors, 2=warnings/info, 3=debug) |
| `return_fig` | bool, default=False | If True, return the figure object instead of just displaying |

**Returns**

- **`fig` : matplotlib.figure.Figure or None**

  The figure object if return_fig=True, otherwise None

**Examples**

```python
>>> la.pl.plotCCCHeatmap(
...     laris_results,
...     filter_significant=True,
...     cluster=True,
...     save='heatmap.pdf'
... )
```

---

<a id="plotcccnetwork"></a>

### `la.pl.plotCCCNetwork`

```python
la.pl.plotCCCNetwork(
    laris_results,
    cell_type_of_interest,
    interaction_direction='sending',
    data,
    n_top=3000,
    edge_width_scale=30,
    interaction_cutoff=0.0,
    groupby='cell_type',
    cell_type_color_key=None,
    custom_color_mapping=None,
    figsize=(12, 10),
    margins=0.2,
    label_font_size=16,
    node_size=1100,
    p_value_col='p_value_fdr',
    threshold=0.05,
    filter_by_interaction_score=True,
    threshold_interaction_score=0.01,
    filter_significant=True,
    label_border=True,
    label_border_color='white',
    label_border_width=3.0,
    save=None,
    verbosity=2,
    return_fig=False,
    include_self_interactions=True,
)
```

Plot an interaction network for a specific cell type.

Creates a directed network graph showing aggregated interactions where a 
specific cell type is either sending or receiving signals. Edge thickness 
represents the cumulative interaction strength.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `laris_results` | pd.DataFrame | DataFrame containing LARIS results |
| `cell_type_of_interest` | str | Cell type to focus on (e.g., "B_cell", "T_cell") |
| `interaction_direction` | {'sending', 'receiving'}, default='sending' | Direction to filter:<br>• 'sending' : Show outgoing interactions from cell_type_of_interest<br>• 'receiving' : Show incoming interactions to cell_type_of_interest |
| `data` | AnnData, str or cytome dataset, optional | Expression object whose cell-type colours (`uns['<groupby>_colors']`) colour the nodes. |
| `n_top` | int, default=3000 | Number of top-scoring interactions (by `interaction_score`) if no filters are applied |
| `edge_width_scale` | float, default=30 | Scaling factor for edge thickness based on interaction score |
| `interaction_cutoff` | float, default=0.0 | Minimum interaction score threshold |
| `groupby` | str, default='cell_type' | Column name in adata.obs containing cell type labels |
| `cell_type_color_key` | str, optional | Key in `adata.uns` holding the cell-type palette. Defaults to `f"{groupby}_colors"` (the scanpy convention). When the key is missing, a palette is generated automatically instead of raising. Key in adata.uns containing cell type colors |
| `custom_color_mapping` | dict, optional | Custom mapping of cell types to colors |
| `figsize` | tuple, default=(12, 10) | Figure size in inches |
| `margins` | float, default=0.2 | Margin space around the plot |
| `label_font_size` | int, default=16 | Font size for node labels |
| `node_size` | int, default=1100 | Size of network nodes |
| `p_value_col` | str, default='p_value_fdr' | Column name for p-value filtering |
| `threshold` | float, default=0.05 | P-value cutoff for significance |
| `filter_by_interaction_score` | bool, default=True | If True, filter by interaction_score > threshold_interaction_score |
| `threshold_interaction_score` | float, default=0.01 | Cutoff for interaction score |
| `filter_significant` | bool, default=True | If True, apply significance filtering |
| `label_border` | bool, default=True | If True, add border/outline to cell type labels for better visibility |
| `label_border_color` | str, default='white' | Color of the label border/outline |
| `label_border_width` | float, default=3.0 | Width of the label border/outline |
| `save` | str, optional | Path to save figure |
| `verbosity` | int, default=2 | Verbosity level |
| `return_fig` | bool, default=False | If True, return the figure and axes objects. |
| `adata` | deprecated | Former name of `data`. |
| `include_self_interactions` | bool, default=True | Draw interactions where the sender and the receiver are the same cell type, as a loop on that node. Autocrine and within-cell-type signalling is common, so these are shown by default; set False to restrict the plot to interactions between different cell types. |

**Returns**

- **tuple or None**

  (fig, ax) if return_fig=True, otherwise None.

**Examples**

```python
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
```

---

<a id="plotcccnetworkcumulative"></a>

### `la.pl.plotCCCNetworkCumulative`

```python
la.pl.plotCCCNetworkCumulative(
    laris_results,
    data,
    cutoff=0,
    n_top=3000,
    groupby='cell_type',
    cell_type_color_key=None,
    custom_color_mapping=None,
    figsize=(12, 10),
    margins=0.2,
    label_font_size=16,
    node_size=1100,
    edge_width_scale=5,
    p_value_col='p_value_fdr',
    threshold=0.05,
    filter_by_interaction_score=True,
    threshold_interaction_score=0.01,
    filter_significant=True,
    edge_thickness_by_numbers=False,
    total_edge_thickness=100,
    label_border=True,
    label_border_color='white',
    label_border_width=3.0,
    save=None,
    verbosity=2,
    return_fig=False,
    include_self_interactions=True,
)
```

Plot a cumulative interaction network across all cell types.

Creates a comprehensive network showing aggregated interactions between all 
cell type pairs. Edge thickness can represent either cumulative interaction 
scores or total interaction counts.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `laris_results` | pd.DataFrame | DataFrame containing LARIS interaction results |
| `data` | AnnData, str or cytome dataset, optional | Expression object whose cell-type colours (`uns['<groupby>_colors']`) colour the nodes. |
| `cutoff` | float, default=0 | Minimum threshold for including an edge |
| `n_top` | int, default=3000 | Number of top-scoring interactions (by `interaction_score`) if no filters are applied |
| `groupby` | str, default='cell_type' | Column in adata.obs containing cell type labels |
| `cell_type_color_key` | str, optional | Key in `adata.uns` holding the cell-type palette. Defaults to `f"{groupby}_colors"` (the scanpy convention). When the key is missing, a palette is generated automatically instead of raising. Key in adata.uns for cell type colors |
| `custom_color_mapping` | dict, optional | Custom cell type to color mapping |
| `figsize` | tuple, default=(12, 10) | Figure dimensions in inches |
| `margins` | float, default=0.2 | Plot margin size |
| `label_font_size` | int, default=16 | Font size for node labels |
| `node_size` | int, default=1100 | Size of network nodes |
| `edge_width_scale` | float, default=5 | Scaling factor for edge thickness (when using scores) |
| `p_value_col` | str, default='p_value_fdr' | Column name for p-value filtering |
| `threshold` | float, default=0.05 | P-value cutoff for significance |
| `filter_by_interaction_score` | bool, default=True | If True, filter by interaction_score > threshold_interaction_score |
| `threshold_interaction_score` | float, default=0.01 | Cutoff for interaction score |
| `filter_significant` | bool, default=True | If True, apply significance filtering |
| `edge_thickness_by_numbers` | bool, default=False | If True, edge thickness represents interaction count |
| `total_edge_thickness` | float, default=100 | Total thickness budget when edge_thickness_by_numbers=True |
| `label_border` | bool, default=True | If True, add border/outline to cell type labels for better visibility |
| `label_border_color` | str, default='white' | Color of the label border/outline |
| `label_border_width` | float, default=3.0 | Width of the label border/outline |
| `save` | str, optional | Path to save figure |
| `verbosity` | int, default=2 | Verbosity level |
| `return_fig` | bool, default=False | If True, return the figure and axes objects. |
| `adata` | deprecated | Former name of `data`. |
| `include_self_interactions` | bool, default=True | Draw interactions where the sender and the receiver are the same cell type, as a loop on that node. Autocrine and within-cell-type signalling is common, so these are shown by default; set False to restrict the plot to interactions between different cell types. |

**Returns**

- **tuple or None**

  (fig, ax) if return_fig=True, otherwise None.

**Examples**

```python
>>> la.pl.plotCCCNetworkCumulative(
...     laris_results,
...     adata=adata,
...     filter_significant=True,
...     label_border=True,
...     label_border_color='white',
...     label_border_width=4.0,
...     save='cumulative_network.pdf'
... )
```

---

<a id="plotcccdotplot"></a>

### `la.pl.plotCCCDotPlot`

```python
la.pl.plotCCCDotPlot(
    laris_results,
    interactions_to_plot,
    senders=None,
    receivers=None,
    sender_receiver_pairs=None,
    delimiter_pair='-->',
    n_top=3000,
    cmap=None,
    bubble_size=250,
    p_value_col='p_value_fdr',
    threshold=0.05,
    filter_by_interaction_score=True,
    threshold_interaction_score=0.01,
    filter_significant=True,
    n_permutations=1000,
    legend_fontsize=16,
    show_grid=False,
    figsize=None,
    save=None,
    verbosity=2,
    return_fig=False,
    score_max=None,
)
```

Create a bubble plot for selected cell type pairs and interactions.

Visualizes interaction strengths between specific sender-receiver cell type 
pairs using bubbles where size represents -log10(p-value) and color 
represents interaction score.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `laris_results` | pd.DataFrame | DataFrame containing LARIS results |
| `interactions_to_plot` | list of str | List of interaction names to include (e.g., ['CXCL13::CXCR5', ...]) |
| `senders` | list of str, optional | List of sending cell types to plot (used with receivers) |
| `receivers` | list of str, optional | List of receiving cell types (must match length of senders) |
| `sender_receiver_pairs` | list of str, optional | Alternative to senders/receivers. List of pairs in format 'sender-->receiver' (or using custom delimiter_pair) |
| `delimiter_pair` | str, default='-->' | Delimiter used in sender_receiver_pairs to separate sender and receiver |
| `n_top` | int, default=3000 | Number of top-scoring interactions (by `interaction_score`) if no filters are applied |
| `cmap` | colormap, optional | Colormap for bubble colors. If None, uses pos_cmap |
| `bubble_size` | float, default=250 | Maximum bubble size for most significant p-values |
| `p_value_col` | str, default='p_value_fdr' | Column name for p-value filtering and sizing |
| `threshold` | float, default=0.05 | P-value cutoff for significance filtering |
| `filter_by_interaction_score` | bool, default=True | If True, filter by interaction_score > threshold_interaction_score |
| `threshold_interaction_score` | float, default=0.01 | Cutoff for interaction score |
| `filter_significant` | bool, default=True | If True, apply significance filtering |
| `n_permutations` | int, default=1000 | Number of permutations used (for p-value scaling) |
| `legend_fontsize` | int, default=16 | Font size for legend titles ("Interaction Score" and "P value") |
| `show_grid` | bool, default=False | If True, show grid lines in the plot |
| `figsize` | tuple, optional | Figure size. If None, automatically calculated |
| `save` | str, optional | Path to save figure |
| `verbosity` | int, default=2 | Verbosity level |
| `return_fig` | bool, default=False | If True, return the figure and axes objects |
| `score_max` | float, optional | Upper end of the colour scale. By default the largest plotted score, so two plots of different rows are coloured on different scales; pass the same value to both (for example the same view at `threshold=0.05` and `threshold=0.10`) to make their colours comparable. |

**Returns**

- **tuple or None**

  (fig, ax) if return_fig=True, otherwise None

**Notes**

Bubble sizes are scaled by -log10(p_value), providing continuous scaling
that better represents the range of significance values. The minimum
possible p-value is determined by 1/(n_permutations + 1).

**Examples**

```python
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
```

---

<a id="plotcccdotplotfacet"></a>

### `la.pl.plotCCCDotPlotFacet`

```python
la.pl.plotCCCDotPlotFacet(
    laris_results,
    cmap=None,
    bubble_size=400,
    height_single_panel=4.5,
    width_single_panel=4.5,
    x_padding=0.4,
    y_padding=0.4,
    n_top=None,
    senders=None,
    receivers=None,
    interactions_to_plot=None,
    p_value_col='p_value_fdr',
    threshold=0.05,
    filter_by_interaction_score=True,
    threshold_interaction_score=0.01,
    filter_significant=True,
    n_permutations=1000,
    ncol=3,
    facet_by='sender',
    legend_fontsize=16,
    show_grid=True,
    save=None,
    verbosity=2,
    return_fig=False,
)
```

Create faceted bubble plots organized by sending or receiving cell type.

Generates a grid of bubble plots where each facet represents a different 
sending or receiving cell type, showing interactions to/from other cell types.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `laris_results` | pd.DataFrame | DataFrame containing LARIS interaction results |
| `cmap` | colormap, optional | Colormap for bubble colors. If None, uses pos_cmap |
| `bubble_size` | float, default=400 | Maximum bubble size |
| `height_single_panel` | float, default=4.5 | Height of each facet panel in inches |
| `width_single_panel` | float, default=4.5 | Width of each facet panel in inches |
| `x_padding` | float, default=0.4 | Extra space on x-axis |
| `y_padding` | float, default=0.4 | Extra space on y-axis |
| `n_top` | int, optional | Number of top-scoring interactions (by `interaction_score`) if no filters are applied |
| `senders` | list of str, optional | Specific sending cell types. If None, all are included |
| `receivers` | list of str, optional | Specific receiving cell types. If None, all are included |
| `interactions_to_plot` | list of str, optional | Specific interactions. If None, all are included |
| `p_value_col` | str, default='p_value_fdr' | Column name for p-value filtering |
| `threshold` | float, default=0.05 | P-value cutoff for significance |
| `filter_by_interaction_score` | bool, default=True | If True, filter by interaction_score > threshold_interaction_score |
| `threshold_interaction_score` | float, default=0.01 | Cutoff for interaction score |
| `filter_significant` | bool, default=True | If True, apply significance filtering |
| `n_permutations` | int, default=1000 | Number of permutations (for p-value scaling) |
| `ncol` | int, default=3 | Number of columns per row |
| `facet_by` | str, default='sender' | How to organize facets:<br>• 'sender' : Each facet shows a different sending cell type<br>• 'receiver' : Each facet shows a different receiving cell type |
| `legend_fontsize` | int, default=16 | Font size for legend titles ("Interaction Score" and "P value") |
| `show_grid` | bool, default=True | If True, show grid lines in the plots |
| `save` | str, optional | Path to save figure |
| `verbosity` | int, default=2 | Verbosity level |
| `return_fig` | bool, default=False | If True, return the Figure object |

**Returns**

- **`fig` : matplotlib.figure.Figure or None**

  The Figure object if return_fig=True, otherwise None

**Examples**

```python
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
```

---

<a id="plotlrdotplot"></a>

### `la.pl.plotLRDotPlot`

```python
la.pl.plotLRDotPlot(
    adata_dotplot,
    interactions_to_plot,
    groupby,
    delimiter='::',
    cmap_interaction='Spectral_r',
    cmap_ligand='Blues',
    cmap_receptor='Purples',
    standard_scale_interaction='var',
    standard_scale_ligand='var',
    standard_scale_receptor='var',
    orientation='horizontal',
    row_height=None,
    max_height=None,
    figsize=None,
    save=None,
    verbosity=2,
    return_fig=False,
)
```

Create three side-by-side (or stacked) dot plots for LR pairs, ligands, and receptors.

Visualizes expression patterns of ligand-receptor pairs alongside individual 
ligand and receptor expression across cell types.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `adata_dotplot` | anndata.AnnData | AnnData object prepared with prepareDotPlotAdata() containing both LR pair scores and individual gene expression |
| `interactions_to_plot` | list of str | List of LR pairs in format 'ligand::receptor' (or using custom delimiter) |
| `groupby` | str | Column in adata_dotplot.obs to group by |
| `delimiter` | str, default='::' | Delimiter used to separate ligand and receptor in interaction names |
| `cmap_interaction` | str, default='Spectral_r' | Colormap for the LR interaction scores plot |
| `cmap_ligand` | str, default='Blues' | Colormap for the ligand expression plot |
| `cmap_receptor` | str, default='Purples' | Colormap for the receptor expression plot |
| `standard_scale_interaction` | str or None, default='var' | Scaling method for interaction scores plot ('var', 'group', or None) |
| `standard_scale_ligand` | str or None, default='var' | Scaling method for ligand expression plot ('var', 'group', or None) |
| `standard_scale_receptor` | str or None, default='var' | Scaling method for receptor expression plot ('var', 'group', or None) |
| `orientation` | str, default='horizontal' | Layout orientation: 'horizontal' for side-by-side, 'vertical' for stacked |
| `row_height` | float, optional | Height per interaction row in inches |
| `max_height` | float, optional | Maximum figure height in inches |
| `figsize` | tuple, optional | Overall figure size (width, height). Overrides row_height if provided |
| `save` | str, optional | Path to save figure |
| `verbosity` | int, default=2 | Verbosity level |
| `return_fig` | bool, default=False | If True, return the figure and axes objects |

**Returns**

- **tuple or None**

  (fig, axes) if return_fig=True, otherwise None

**Examples**

```python
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
```

---

<a id="plotcccspatial"></a>

### `la.pl.plotCCCSpatial`

```python
la.pl.plotCCCSpatial(
    lr_data,
    basis='spatial',
    interaction=None,
    cell_type=None,
    selected_cell_types=None,
    highlight_all_expressing=False,
    background_color='lightgrey',
    colors=None,
    size=120,
    fig_width=6,
    fig_height=None,
    max_title_chars=60,
    save=None,
    verbosity=2,
    return_fig=False,
    color_by='cell_type',
    data,
    library_id=None,
    img_key='hires',
    library_key=None,
    img=None,
    scale_factor=None,
    alpha_img=1.0,
    cmap=None,
    vmax_quantile=0.995,
    score_threshold=0.0,
    crop=True,
    margin=0.05,
    colorbar=True,
)
```

Plot spatial distribution of ligand-receptor interactions.

Creates a spatial plot highlighting cells expressing a specific interaction, 
with options to highlight specific cell types or all expressing cells.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `lr_data` | AnnData, str or cytome dataset | LR interaction scores with spatial coordinates (from `prepareLRInteraction`): an AnnData, or an LR cytome. |
| `basis` | str | Key for spatial coordinates in lr_adata.obsm |
| `interaction` | str | Interaction name to visualize (must be in lr_adata.var_names) |
| `cell_type` | str | Column name in lr_adata.obs containing cell type annotations |
| `selected_cell_types` | list of str, optional | Specific cell types to highlight |
| `highlight_all_expressing` | bool, default=False | If True, highlight all cells expressing the interaction |
| `background_color` | str, default='lightgrey' | Color for non-expressing cells |
| `colors` | list of str, optional | Colors for selected cell types |
| `size` | float, default=120 | Point size for spatial plot |
| `fig_width` | float, default=6 | Figure width in inches. Height is calculated to maintain data aspect ratio |
| `fig_height` | float, optional | Figure height in inches. If provided, overrides aspect ratio calculation |
| `max_title_chars` | int, default=60 | Maximum characters per line in title before wrapping |
| `save` | str, optional | Path to save figure |
| `verbosity` | int, default=2 | Verbosity level |
| `return_fig` | bool, default=False | If True, return the figure object. |
| `color_by` | {'cell_type', 'score'}, default='cell_type' | 'cell_type' highlights the expressing cells by their cell type (the modes above); 'score' colours every cell by its interaction score, optionally over a tissue image. |
| `data` | AnnData, str or cytome dataset, optional | The expression object, searched for a tissue image (scanpy-style `uns['spatial']` or a cytome's stored images) before `lr_data`. |
| `library_id` | str, optional | Which library's image to draw when several have one. |
| `img_key` | str, default='hires' | Which stored image of the library to draw ('hires', 'lowres', ...). |
| `library_key` | str, optional | Column of `.obs` naming each cell's library; when several libraries have images, the one the plotted cells belong to is used. |
| `img` | numpy.ndarray, optional | An explicit image to draw instead of a stored one; needs `scale_factor`. |
| `scale_factor` | float, optional | Image pixels per coordinate unit for `img` (1.0 when the coordinates are already image pixels). |
| `alpha_img` | float, default=1.0 | Opacity of the tissue image. |
| `cmap` | str or matplotlib.colors.Colormap, optional | Colormap of the score mode (LARIS's own by default). |
| `vmax_quantile` | float, default=0.995 | The colour scale saturates at this quantile of the positive scores. |
| `score_threshold` | float, default=0.0 | Cells scoring at or below this are drawn as background. |
| `crop` | bool, default=True | Crop the axes to the cells (plus `margin`). |
| `margin` | float, default=0.05 | Margin around the cells when cropping, as a fraction of their extent. |
| `colorbar` | bool, default=True | Draw a colour bar in the score mode. |
| `lr_adata`, `adata` | deprecated | Former names of `lr_data` and `data`. |

**Returns**

- **matplotlib.figure.Figure or None**

  The figure object if return_fig=True, otherwise None.

**Examples**

```python
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
```

---

<a id="plotcomparelaris"></a>

### `la.pl.plotCompareLARIS`

```python
la.pl.plotCompareLARIS(
    comparison,
    comparison_name=None,
    effect_col='log_diff',
    pvalue_col='pvalue_fdr',
    fdr_threshold=0.05,
    effect_threshold=0.0,
    n_labels=10,
    label=None,
    label_col='interaction_name',
    sender=None,
    receiver=None,
    condition_labels=None,
    colors=None,
    point_size=14,
    highlight_size=30,
    label_fontsize=8,
    leader_lines=True,
    figsize=(5.4, 4.6),
    title=None,
    show_counts=True,
    show_thresholds=True,
    xlim=None,
    ylim=None,
    ax=None,
    save=None,
    verbosity=2,
    return_fig=False,
)
```

Volcano plot of a `laris.tl.compareLARIS` result.

Plots the effect size against significance for every tested
ligand-receptor pair, so up- and down-regulated interactions and the
bulk of unchanged ones are visible at once.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `comparison` | pd.DataFrame | Either table returned by `laris.tl.compareLARIS` - the per-LR-pair one, or the per-triple one (use `sender`/`receiver` to pick a cell-type pair from it). |
| `comparison_name` | str, optional | Which contrast to draw when the table holds several (the `comparison` column, e.g. `'cKO_vs_WT'`). Required if there is more than one and none is given. |
| `effect_col` | str, default='log_diff' | Column for the x axis. `log_diff` is the difference of per-sample-centred log scores, which is invariant to LARIS's per-run rescaling; `log2fc` is the descriptive fold change on raw scores and is *not* scale-invariant, so prefer the default. |
| `pvalue_col` | str, default='pvalue_fdr' | Column for the y axis, plotted as `-log10`. |
| `fdr_threshold` | float, default=0.05 | Significance cut-off for colouring and the horizontal rule. |
| `effect_threshold` | float, default=0.0 | Optional minimum \|effect\| for an interaction to count as changed. Vertical rules are drawn when this is non-zero. |
| `n_labels` | int, default=10 | How many significant interactions to name, balanced across the two directions. Set 0 for none. Labels that cannot be placed without overlapping are dropped, and the count is reported at `verbosity >= 2`. |
| `label` | sequence of str, optional | Name these interactions explicitly instead of choosing by rank. |
| `label_col` | str, default='interaction_name' | Column supplying the label text. |
| `sender`, `receiver` | str, optional | Restrict a per-triple table to one sender-receiver pair. |
| `condition_labels` | (str, str), optional | `(reference, alternative)` names used to annotate the direction of the x axis, e.g. `('WT', 'cKO')`. Inferred from `comparison_name` of the form `'alt_vs_ref'` when possible. |
| `colors` | (down, ns, up), optional | Override the default blue / grey / orange. The default avoids red-green, which a volcano cannot afford. |
| `point_size`, `highlight_size` | float | Marker areas for unchanged and significant interactions. |
| `label_fontsize` | float, default=8 | Font size of the interaction labels. |
| `leader_lines` | bool, default=True | Draw a hairline from each label to the point it names. Worth keeping: displaced labels are otherwise ambiguous in dense regions. |
| `figsize` | (float, float), default=(5.4, 4.6) | Figure size in inches; ignored when `ax` is given. |
| `title` | str, optional | Plot title. A contrast-derived default is used when omitted. |
| `show_counts` | bool, default=True | Annotate how many interactions are significant in each direction. |
| `show_thresholds` | bool, default=True | Draw the significance (and effect) cut-off rules. |
| `xlim`, `ylim` | (float, float), optional | Axis limits. The x axis is symmetric about 0 by default, so the two directions are visually comparable. |
| `ax` | matplotlib.axes.Axes, optional | Draw into an existing axes instead of creating a figure. |
| `save` | str, optional | Path to save to. Use a `.pdf` or `.svg` extension for vector output suitable for a manuscript figure. |
| `verbosity` | int, default=2 | Verbosity level. |
| `return_fig` | bool, default=False | Return the figure instead of showing it. |

**Returns**

- **matplotlib.figure.Figure or None**

  The figure with `return_fig=True`, otherwise None.

**Examples**

```python
>>> lr_cmp, triple_cmp = la.tl.compareLARIS(...)
>>> la.pl.plotCompareLARIS(lr_cmp, condition_labels=("WT", "cKO"))
>>>
>>> # one cell-type pair from the per-triple table, saved as vector art
>>> la.pl.plotCompareLARIS(triple_cmp, sender="Astro", receiver="L2_3",
...                        save="astro_to_l23.pdf")
```

---

<a id="preparedotplotadata"></a>

### `la.pl.prepareDotPlotAdata`

```python
la.pl.prepareDotPlotAdata(
    lr_data,
    data,
    verbosity=2,
)
```

Prepare combined AnnData for dot plot visualizations.

Concatenates LR interaction scores with original gene expression data 
horizontally to create a unified AnnData object for plotting.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `lr_data` | AnnData, str or cytome dataset | LR interaction scores (from `prepareLRInteraction`): an AnnData, or an LR cytome. |
| `data` | AnnData, str or cytome dataset | The gene expression object; a `.cytome`/`.db` path or an open cytome dataset is read automatically. |
| `verbosity` | int, default=2 | Verbosity level. |
| `lr_adata`, `adata` | deprecated | Former names of `lr_data` and `data`. |

**Returns**

- **`adata_dotplot` : anndata.AnnData**

  Combined AnnData object

**Examples**

```python
>>> adata_combined = la.pl.prepareDotPlotAdata(lr_adata, adata)
>>> la.pl.plotLRDotPlot(adata_combined, interactions, groupby='cell_type')
```
