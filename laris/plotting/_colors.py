"""Colormaps and cell-type palette resolution for LARIS plots."""

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


def _get_cmap(name, lutsize=None):
    """Look up a colormap compatibly across matplotlib versions.

    matplotlib.cm.get_cmap was removed in matplotlib 3.9; the
    matplotlib.colormaps registry (with .resampled) is available from 3.6.
    """
    try:
        cmap = matplotlib.colormaps[name]
        return cmap.resampled(lutsize) if lutsize is not None else cmap
    except AttributeError:  # matplotlib < 3.6
        return cm.get_cmap(name, lutsize)


def _resolve_cell_type_colors(adata, groupby, cell_type_color_key=None):
    """Map cell-type labels to colors.

    Uses ``adata.uns[cell_type_color_key]`` when available (default key:
    ``f"{groupby}_colors"``, the scanpy convention), pairing colors with the
    categorical *categories* order (scanpy stores palettes in that order, not
    in appearance order). Falls back to a generated palette when no stored
    palette fits, instead of raising KeyError.
    """
    col = adata.obs[groupby]
    if hasattr(col, 'cat'):
        labels = list(col.cat.categories)
    else:
        labels = list(pd.unique(col))
    key = cell_type_color_key if cell_type_color_key is not None else f"{groupby}_colors"
    if key in adata.uns and len(adata.uns[key]) >= len(labels):
        palette = list(adata.uns[key])[:len(labels)]
    else:
        cmap = _get_cmap('tab20')
        palette = [
            colors.to_hex(cmap(i % cmap.N)) for i in range(len(labels))
        ]
    return dict(zip(labels, palette))


# Define custom colormap for interaction scores
cmap_own = _get_cmap('magma_r', 256)
newcolors = cmap_own(np.linspace(0, 0.75, 256))
Greys = _get_cmap('Greys_r', 256)
newcolors[:10, :] = Greys(np.linspace(0.8125, 0.8725, 10))
pos_cmap = colors.ListedColormap(newcolors)
