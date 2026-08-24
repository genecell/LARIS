"""Spatial plots of ligand-receptor interactions."""

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
from ._utils import _log_message, _save_figure
from ._spatial_image import (_render_score_overlay, _resolve_background_image,
                             _draw_image_overlay, _image_axis_limits)
from .._compat import _UNSET, resolve_data_arg
from ..preprocessing._io import _ensure_lr_anndata

def plotCCCSpatial(
    lr_data=_UNSET,
    basis: str = 'spatial',
    interaction: Optional[str] = None,
    cell_type: Optional[str] = None,
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
    return_fig: bool = False,
    # --- score mode + tissue-image overlay (v0.10.0, GitHub issue #1) ---
    color_by: str = 'cell_type',
    data=_UNSET,
    library_id: Optional[str] = None,
    img_key: str = 'hires',
    library_key: Optional[str] = None,
    img: Optional[np.ndarray] = None,
    scale_factor: Optional[float] = None,
    alpha_img: float = 1.0,
    cmap: Union[str, colors_mod.Colormap, None] = None,
    vmax_quantile: float = 0.995,
    score_threshold: float = 0.0,
    crop: bool = True,
    margin: float = 0.05,
    colorbar: bool = True,
    lr_adata=_UNSET,
    adata=_UNSET,
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
    lr_adata = _ensure_lr_anndata(
        resolve_data_arg(lr_data, 'plotCCCSpatial', canonical='lr_data',
                         lr_adata=lr_adata))
    adata = resolve_data_arg(data, 'plotCCCSpatial', canonical='data',
                             required=False, adata=adata)
    if interaction is None:
        raise TypeError(
            "plotCCCSpatial() missing required argument: 'interaction'")

    # Save original rcParams to restore later
    original_figsize = plt.rcParams['figure.figsize'].copy()
    
    try:
        # Check if interaction exists
        if interaction not in lr_adata.var_names:
            # Raise, not print-and-return-None: silent failures hide in
            # batch scripts (same rationale as the v0.9.4 dot-plot fix).
            raise ValueError(
                f"Interaction {interaction!r} not found in "
                f"lr_adata.var_names."
            )

        if color_by not in ('cell_type', 'score'):
            raise ValueError(
                f"color_by must be 'cell_type' or 'score', got {color_by!r}"
            )
        if color_by == 'cell_type' and cell_type is None:
            raise ValueError(
                "cell_type is required with color_by='cell_type' "
                "(pass color_by='score' for a continuous score overlay)."
            )

        ctx = _resolve_background_image(
            lr_adata, adata=adata, library_id=library_id,
            img_key=img_key, img=img, scale_factor=scale_factor,
            library_values=(lr_adata.obs[library_key].to_numpy()
                            if library_key is not None
                            and library_key in lr_adata.obs else None))

        # --- continuous score mode: matplotlib renderer, optional image ---
        if color_by == 'score':
            return _render_score_overlay(
                lr_adata, basis, interaction, ctx,
                size=size, alpha_img=alpha_img, cmap=cmap,
                vmax_quantile=vmax_quantile, score_threshold=score_threshold,
                crop=crop, margin=margin, colorbar=colorbar,
                fig_width=fig_width, fig_height=fig_height,
                save=save, verbosity=verbosity, return_fig=return_fig)

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
                    default_colors = _get_cmap('tab10')(
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

        # --- categorical over a tissue image: matplotlib renderer ---
        if ctx is not None:
            fig, ax = plt.subplots(figsize=figsize)
            _draw_image_overlay(ax, ctx, alpha_img=alpha_img)
            px = np.asarray(lr_adata_sorted.obsm[basis])[:, :2]
            cats = lr_adata_sorted.obs[color_column]
            point_colors = [palette.get(c, background_color) for c in cats]
            ax.scatter(px[:, 0], px[:, 1], c=point_colors,
                       s=size * 0.1, linewidths=0)
            if crop and len(px):
                _image_axis_limits(ax, px, ctx, pad_frac=margin)
            handles = [Line2D([0], [0], marker='o', linestyle='',
                              markerfacecolor=palette[c], markeredgewidth=0,
                              label=str(c))
                       for c in palette if c not in ('background', 'other')]
            if handles:
                ax.legend(handles=handles, loc='center left',
                          bbox_to_anchor=(1.0, 0.5), frameon=False)
            ax.set_title(title_text)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if save is not None:
                fig.savefig(save, bbox_inches='tight', dpi=300)
                _log_message(f"Figure saved to: {save}", 2, verbosity, 'info')
            plt.show()
            if return_fig:
                return fig
            return None

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
