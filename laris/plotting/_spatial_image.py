"""Tissue-image resolution and overlay for LARIS spatial plots.

The image is placed in COORDINATE units via ``extent`` rather than
scaling the coordinates, so axis limits, crops and spot diameters all
live in one unit system - the same units cytome's ``crop()`` and
``cells_in_region()`` speak. Adapted from
``piaso.plotting._spatial_image`` (PIASO 1.2.2); copied rather than
imported so LARIS installs without piaso-tools.
"""

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

from ._colors import _get_cmap, pos_cmap
from ._utils import _log_message

def _render_score_overlay(lr_adata, basis, interaction, ctx,
                          size, alpha_img, cmap, vmax_quantile,
                          score_threshold, crop, margin, colorbar,
                          fig_width, fig_height, save, verbosity, return_fig):
    """Continuous per-spot interaction-score rendering, optionally over a
    tissue image (plotCCCSpatial's color_by='score' mode)."""
    if interaction not in lr_adata.var_names:
        raise ValueError(
            f"Interaction {interaction!r} not found in lr_adata.var_names."
        )
    col = lr_adata[:, interaction]
    values = col.X
    scores = np.asarray(
        values.toarray() if issparse(values) else values
    ).ravel().astype(float)
    coords = np.asarray(lr_adata.obsm[basis])[:, :2].astype(float)

    w = fig_width if fig_width is not None else 8
    h = fig_height if fig_height is not None else w
    fig, ax = plt.subplots(figsize=(w, h))

    # The image is drawn in COORDINATE units (extent), so spots are never
    # scaled and y-inversion comes from the extent alone.
    px = coords
    if ctx is not None:
        _draw_image_overlay(ax, ctx, alpha_img=alpha_img)
    else:
        ax.set_aspect('equal')
        ax.invert_yaxis()

    if cmap is None:
        cmap = pos_cmap
    elif isinstance(cmap, str):
        cmap = _get_cmap(cmap)

    positive = scores > score_threshold
    vmax = (np.quantile(scores[positive], vmax_quantile)
            if positive.any() else 1.0)
    if vmax <= 0:
        vmax = scores.max() if scores.max() > 0 else 1.0

    marker = size * 0.1
    ax.scatter(px[~positive, 0], px[~positive, 1],
               s=marker * 0.4, c='lightgrey', alpha=0.25, linewidths=0)
    order = np.argsort(scores[positive])
    sc_plot = ax.scatter(px[positive, 0][order], px[positive, 1][order],
                         c=scores[positive][order], cmap=cmap,
                         vmin=0, vmax=vmax, s=marker, linewidths=0)

    if crop and len(px):
        if ctx is not None:
            _image_axis_limits(ax, px, ctx, pad_frac=margin)
        else:
            x0, x1 = px[:, 0].min(), px[:, 0].max()
            y0, y1 = px[:, 1].min(), px[:, 1].max()
            dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
            ax.set_xlim(x0 - dx, x1 + dx)
            ax.set_ylim(y1 + dy, y0 - dy)

    if colorbar and positive.any():
        cb = fig.colorbar(sc_plot, ax=ax, shrink=0.6, pad=0.02)
        cb.set_label('interaction score')

    ax.set_title(interaction)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if save is not None:
        fig.savefig(save, bbox_inches='tight', dpi=300)
        _log_message(f"Figure saved to: {save}", 2, verbosity, 'info')
    if return_fig:
        return fig
    return None


def _resolve_background_image(lr_adata, adata=None, library_id=None,
                              img_key='hires', img=None, scale_factor=None,
                              library_values=None):
    """Resolve a tissue image and place it in COORDINATE units.

    Waterfall: explicit ``img`` (requires ``scale_factor``) -> a cytome
    ``spatial_images`` accessor or ``uns['spatial']`` on ``adata`` -> the
    same on ``lr_adata`` -> no image.

    Returns ``{'img', 'extent', 'scalef', 'spot_diameter', 'library'}`` or
    ``None``. ``extent`` is the matplotlib 4-tuple
    ``(0, W/scalef, H/scalef, 0)``: the image is scaled UP into
    full-resolution coordinate units with y increasing downward, so spot
    coordinates are drawn unscaled and no ``invert_yaxis()`` is needed
    (calling it as well would mirror the spots against the tissue). Axis
    limits, crops and spot diameters then all live in one unit system -
    the same units as the spatial embedding.

    Adapted from ``piaso.plotting._spatial_image`` (PIASO 1.2.2); copied
    rather than imported so LARIS installs without piaso-tools. Keep the
    two in step if the overlay contract changes.
    """
    from ..preprocessing._io import _spatial_uns_from

    if img is not None:
        if scale_factor is None:
            raise ValueError(
                "scale_factor is required with an explicit img= (pass 1.0 "
                "when coordinates are already in image pixels)."
            )
        arr = np.asarray(img)
        sf = float(scale_factor)
        h, w = arr.shape[0], arr.shape[1]
        return {'img': arr, 'extent': (0.0, w / sf, h / sf, 0.0),
                'scalef': sf, 'spot_diameter': None, 'library': None}

    spatial = {}
    for source in (adata, lr_adata):
        if source is None:
            continue
        spatial = _spatial_uns_from(source)
        if spatial:
            break
    if not spatial:
        return None

    # Which library to draw: explicit argument, else the only one, else the
    # single library the plotted cells belong to; never silently the first.
    libs = sorted(spatial)
    if library_id is not None:
        if library_id not in spatial:
            raise KeyError(
                f"library {library_id!r} has no stored image; available: {libs}")
        library = library_id
    elif len(libs) == 1:
        library = libs[0]
    else:
        narrowed = None
        if library_values is not None:
            present = {str(v) for v in np.asarray(library_values)}
            candidates = [l for l in libs if l in present]
            if len(candidates) == 1:
                narrowed = candidates[0]
        if narrowed is None:
            raise ValueError(
                f"Multiple libraries have images ({libs}); pass library_id= "
                f"or subset to one library."
            )
        library = narrowed

    entry = spatial[library]
    images = entry.get('images', {})
    if img_key not in images:
        raise KeyError(
            f"library {library!r} has no image {img_key!r}; available: "
            f"{sorted(images)}")
    arr = np.asarray(images[img_key])
    sfs = entry.get('scalefactors', {})
    sf = float(scale_factor) if scale_factor is not None else \
        float(sfs.get(f'tissue_{img_key}_scalef', 1.0))
    spot_d = sfs.get('spot_diameter_fullres')
    h, w = arr.shape[0], arr.shape[1]
    return {'img': arr, 'extent': (0.0, w / sf, h / sf, 0.0), 'scalef': sf,
            'spot_diameter': (float(spot_d) if spot_d is not None else None),
            'library': library}


def _draw_image_overlay(ax, ctx, alpha_img=1.0):
    """Draw the resolved image under the scatter, in coordinate units."""
    img = ctx['img']
    kwargs = {'cmap': 'gray'} if img.ndim == 2 else {}
    ax.imshow(img, extent=ctx['extent'], origin='upper', zorder=0,
              alpha=alpha_img, aspect='equal', interpolation='antialiased',
              **kwargs)


def _image_axis_limits(ax, coords, ctx, pad_frac=0.02):
    """Crop to the plotted cells (coord bbox padded by one spot diameter),
    y inverted to match the image's top-down orientation."""
    coords = np.asarray(coords, dtype=float)
    x0, x1 = float(np.min(coords[:, 0])), float(np.max(coords[:, 0]))
    y0, y1 = float(np.min(coords[:, 1])), float(np.max(coords[:, 1]))
    pad = (ctx.get('spot_diameter') if ctx else None) or 0.0
    pad = max(pad, pad_frac * max(x1 - x0, y1 - y0, 1.0))
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y1 + pad, y0 - pad)
