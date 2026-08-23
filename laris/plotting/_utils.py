"""Shared helpers for LARIS plotting: logging, saving, legends."""

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
