"""Visualization of cross-condition comparisons (``laris.tl.compareLARIS``)."""

import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ._utils import _log_message, _save_figure

# Direction colours. Deliberately not red/green: the pair is indistinguishable
# for the ~8% of male readers with a red-green deficiency, and a volcano is
# exactly the figure where direction must survive that.
_DOWN = '#2a78d6'
_UP = '#eb6834'
_NS = '#b8b8b2'
_INK = '#0b0b0b'
_INK_SOFT = '#52514e'
_RULE = '#8a8a85'


def _pick_labels(df, n_labels, label, label_col):
    """Which rows get a text label: an explicit list, or the strongest hits.

    "Strongest" balances the two sides rather than taking the global top-n,
    so a volcano with one dominant direction still names something on the
    other - the asymmetry should be visible in the points, not manufactured
    by the labelling.
    """
    if label is not None:
        wanted = set(map(str, label))
        return df[df[label_col].astype(str).isin(wanted)]
    if not n_labels:
        return df.iloc[:0]
    sig = df[df['_significant']]
    if sig.empty:
        return df.iloc[:0]
    per_side = max(1, n_labels // 2)
    up = sig[sig['_x'] > 0].nsmallest(per_side, '_pval')
    down = sig[sig['_x'] < 0].nsmallest(per_side, '_pval')
    picked = pd.concat([up, down])
    if len(picked) < n_labels:
        rest = sig.drop(index=picked.index).nsmallest(
            n_labels - len(picked), '_pval')
        picked = pd.concat([picked, rest])
    return picked


def _place_labels(ax, rows, label_col, fontsize, colour=_INK,
                  leader_lines=True):
    """Greedy non-overlapping label placement.

    Tries a ring of candidate offsets per label and keeps the first that
    collides with nothing already placed. Labels that cannot be placed are
    dropped rather than stacked, because an unreadable pile of overlapping
    gene names is worse than a few unnamed points.

    A hairline leader connects each label to its point: once a label is
    displaced to avoid a collision, adjacency alone no longer says which
    point it names, and in a dense volcano that is a real ambiguity.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed, dropped = [], 0
    candidates = [(9, 5), (9, -6), (-9, 5), (-9, -6), (9, 13), (-9, 13),
                  (9, -14), (-9, -14), (0, 15), (0, -17)]
    for _, row in rows.iterrows():
        best = None
        for dx, dy in candidates:
            text = ax.annotate(
                str(row[label_col]), (row['_x'], row['_y']),
                textcoords='offset points', xytext=(dx, dy),
                fontsize=fontsize, color=colour,
                ha='left' if dx > 0 else ('right' if dx < 0 else 'center'),
                va='bottom' if dy >= 0 else 'top', zorder=6,
                arrowprops=dict(arrowstyle='-', color=_RULE, lw=.6,
                                alpha=.75, shrinkA=1, shrinkB=4,
                                connectionstyle='arc3')
                if leader_lines else None,
            )
            bbox = text.get_window_extent(renderer).expanded(1.06, 1.20)
            if not any(bbox.overlaps(other) for other in placed):
                best = (text, bbox)
                break
            text.remove()
        if best is None:
            dropped += 1
        else:
            placed.append(best[1])
    return dropped


def plotCompareLARIS(
    comparison: pd.DataFrame,
    comparison_name: Optional[str] = None,
    effect_col: str = 'log_diff',
    pvalue_col: str = 'pvalue_fdr',
    fdr_threshold: float = 0.05,
    effect_threshold: float = 0.0,
    n_labels: int = 10,
    label: Optional[Sequence[str]] = None,
    label_col: str = 'interaction_name',
    sender: Optional[str] = None,
    receiver: Optional[str] = None,
    condition_labels: Optional[Tuple[str, str]] = None,
    colors: Optional[Tuple[str, str, str]] = None,
    point_size: float = 14,
    highlight_size: float = 30,
    label_fontsize: float = 8,
    leader_lines: bool = True,
    figsize: Tuple[float, float] = (5.4, 4.6),
    title: Optional[str] = None,
    show_counts: bool = True,
    show_thresholds: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    save: Optional[str] = None,
    verbosity: int = 2,
    return_fig: bool = False,
):
    """
    Volcano plot of a :func:`laris.tl.compareLARIS` result.

    Plots the effect size against significance for every tested
    ligand-receptor pair, so up- and down-regulated interactions and the
    bulk of unchanged ones are visible at once.

    Parameters
    ----------
    comparison : pd.DataFrame
        Either table returned by :func:`laris.tl.compareLARIS` - the
        per-LR-pair one, or the per-triple one (use `sender`/`receiver` to
        pick a cell-type pair from it).
    comparison_name : str, optional
        Which contrast to draw when the table holds several (the
        ``comparison`` column, e.g. ``'cKO_vs_WT'``). Required if there is
        more than one and none is given.
    effect_col : str, default='log_diff'
        Column for the x axis. ``log_diff`` is the difference of
        per-sample-centred log scores, which is invariant to LARIS's
        per-run rescaling; ``log2fc`` is the descriptive fold change on raw
        scores and is *not* scale-invariant, so prefer the default.
    pvalue_col : str, default='pvalue_fdr'
        Column for the y axis, plotted as ``-log10``.
    fdr_threshold : float, default=0.05
        Significance cut-off for colouring and the horizontal rule.
    effect_threshold : float, default=0.0
        Optional minimum |effect| for an interaction to count as changed.
        Vertical rules are drawn when this is non-zero.
    n_labels : int, default=10
        How many significant interactions to name, balanced across the two
        directions. Set 0 for none. Labels that cannot be placed without
        overlapping are dropped, and the count is reported at
        ``verbosity >= 2``.
    label : sequence of str, optional
        Name these interactions explicitly instead of choosing by rank.
    label_col : str, default='interaction_name'
        Column supplying the label text.
    leader_lines : bool, default=True
        Draw a hairline from each label to the point it names. Worth
        keeping: displaced labels are otherwise ambiguous in dense regions.
    sender, receiver : str, optional
        Restrict a per-triple table to one sender-receiver pair.
    condition_labels : (str, str), optional
        ``(reference, alternative)`` names used to annotate the direction
        of the x axis, e.g. ``('WT', 'cKO')``. Inferred from
        ``comparison_name`` of the form ``'alt_vs_ref'`` when possible.
    colors : (down, ns, up), optional
        Override the default blue / grey / orange. The default avoids
        red-green, which a volcano cannot afford.
    point_size, highlight_size : float
        Marker areas for unchanged and significant interactions.
    figsize : (float, float), default=(5.4, 4.6)
        Figure size in inches; ignored when `ax` is given.
    title : str, optional
        Plot title. A contrast-derived default is used when omitted.
    show_counts : bool, default=True
        Annotate how many interactions are significant in each direction.
    show_thresholds : bool, default=True
        Draw the significance (and effect) cut-off rules.
    xlim, ylim : (float, float), optional
        Axis limits. The x axis is symmetric about 0 by default, so the
        two directions are visually comparable.
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axes instead of creating a figure.
    save : str, optional
        Path to save to. Use a ``.pdf`` or ``.svg`` extension for vector
        output suitable for a manuscript figure.
    return_fig : bool, default=False
        Return the figure instead of showing it.

    Returns
    -------
    matplotlib.figure.Figure or None

    Examples
    --------
    >>> lr_cmp, triple_cmp = la.tl.compareLARIS(...)
    >>> la.pl.plotCompareLARIS(lr_cmp, condition_labels=("WT", "cKO"))
    >>>
    >>> # one cell-type pair from the per-triple table, saved as vector art
    >>> la.pl.plotCompareLARIS(triple_cmp, sender="Astro", receiver="L2_3",
    ...                        save="astro_to_l23.pdf")
    """
    if not isinstance(comparison, pd.DataFrame) or comparison.empty:
        raise ValueError(
            "comparison must be a non-empty DataFrame from compareLARIS."
        )
    df = comparison.copy()

    for column in (effect_col, pvalue_col, label_col):
        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found in the comparison table. "
                f"Available: {list(df.columns)}"
            )

    if sender is not None or receiver is not None:
        for key, value in (('sender', sender), ('receiver', receiver)):
            if value is None:
                continue
            if key not in df.columns:
                raise ValueError(
                    f"'{key}' is not a column here - {key}/receiver only "
                    f"apply to the per-triple table."
                )
            df = df[df[key].astype(str) == str(value)]
        if df.empty:
            raise ValueError(
                f"No rows for sender={sender!r}, receiver={receiver!r}."
            )

    if 'comparison' in df.columns:
        contrasts = list(pd.unique(df['comparison'].dropna()))
        if comparison_name is not None:
            if comparison_name not in contrasts:
                raise ValueError(
                    f"comparison_name={comparison_name!r} not found. "
                    f"Available: {contrasts}"
                )
            df = df[df['comparison'] == comparison_name]
        elif len(contrasts) > 1:
            raise ValueError(
                f"This table holds {len(contrasts)} contrasts {contrasts}; "
                f"pass comparison_name= to choose one."
            )
        else:
            comparison_name = contrasts[0] if contrasts else None

    df = df[df[effect_col].notna() & df[pvalue_col].notna()].copy()
    if df.empty:
        raise ValueError(
            "Nothing to plot: every row has a missing effect or p-value. "
            "This usually means no LR pair had enough subjects to test."
        )

    down_c, ns_c, up_c = colors if colors is not None else (_DOWN, _NS, _UP)

    df['_x'] = df[effect_col].astype(float)
    df['_pval'] = df[pvalue_col].astype(float)
    # A p-value of exactly 0 is a floating-point floor, not infinite
    # evidence; cap it at the smallest positive value present so the point
    # stays on the canvas instead of being drawn at infinity.
    floor = df.loc[df['_pval'] > 0, '_pval'].min() if (df['_pval'] > 0).any() else 1e-300
    df['_y'] = -np.log10(df['_pval'].clip(lower=floor))
    df['_significant'] = (df['_pval'] < fdr_threshold) & \
                         (df['_x'].abs() >= effect_threshold)

    n_up = int((df['_significant'] & (df['_x'] > 0)).sum())
    n_down = int((df['_significant'] & (df['_x'] < 0)).sum())

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ns = df[~df['_significant']]
    up = df[df['_significant'] & (df['_x'] > 0)]
    down = df[df['_significant'] & (df['_x'] < 0)]
    ax.scatter(ns['_x'], ns['_y'], s=point_size, c=ns_c, linewidths=0,
               alpha=.55, zorder=3, rasterized=True)
    ax.scatter(down['_x'], down['_y'], s=highlight_size, c=down_c,
               linewidths=0, zorder=4)
    ax.scatter(up['_x'], up['_y'], s=highlight_size, c=up_c,
               linewidths=0, zorder=4)

    if show_thresholds:
        ax.axhline(-np.log10(fdr_threshold), color=_RULE, lw=.9,
                   ls=(0, (4, 3)), zorder=2)
        if effect_threshold > 0:
            for sign in (-1, 1):
                ax.axvline(sign * effect_threshold, color=_RULE, lw=.9,
                           ls=(0, (4, 3)), zorder=2)
    ax.axvline(0, color=_RULE, lw=.7, alpha=.6, zorder=2)

    if xlim is None:
        span = float(np.nanmax(np.abs(df['_x']))) * 1.12 or 1.0
        ax.set_xlim(-span, span)          # symmetric: directions comparable
    else:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    reference = alternative = None
    if condition_labels is not None:
        reference, alternative = condition_labels
    elif comparison_name and '_vs_' in str(comparison_name):
        alternative, reference = str(comparison_name).split('_vs_', 1)

    if reference and alternative:
        xlabel = (f"effect size   log difference "
                  f"({alternative} − {reference})")
    else:
        xlabel = f"effect size   {effect_col}"
    ax.set_xlabel(xlabel, fontsize=9.5, color=_INK_SOFT)
    ylabel = ("significance   $-\\log_{10}$ FDR" if 'fdr' in pvalue_col
              else "significance   $-\\log_{10}$ p")
    ax.set_ylabel(ylabel, fontsize=9.5, color=_INK_SOFT)

    if title is None and comparison_name:
        title = str(comparison_name).replace('_vs_', ' vs ')
    if sender is not None or receiver is not None:
        pair = f"{sender or '*'} → {receiver or '*'}"
        title = f"{title} · {pair}" if title else pair
    if title:
        ax.set_title(title, fontsize=10.5, fontweight='bold', color=_INK)

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(_RULE)
    ax.spines[['left', 'bottom']].set_linewidth(.8)
    ax.tick_params(colors=_INK_SOFT, labelsize=8.5)
    ax.grid(True, alpha=.5, color='#e6e6e2', lw=.7)
    ax.set_axisbelow(True)

    if show_counts and (n_up or n_down):
        handles = [
            Line2D([], [], marker='o', ls='', color=down_c, markersize=5.5,
                   label=f"down  {n_down}"),
            Line2D([], [], marker='o', ls='', color=up_c, markersize=5.5,
                   label=f"up  {n_up}"),
        ]
        ax.legend(handles=handles, loc='upper left', frameon=False,
                  fontsize=8.5, handletextpad=.4, borderaxespad=.3,
                  labelcolor=_INK_SOFT)

    dropped = _place_labels(
        ax, _pick_labels(df, n_labels, label, label_col), label_col,
        label_fontsize, leader_lines=leader_lines)
    if dropped:
        _log_message(
            f"{dropped} label(s) omitted to avoid overlap; reduce n_labels "
            f"or pass label= to choose which interactions to name.",
            2, verbosity)

    if created:
        fig.tight_layout()
    if save:
        _save_figure(fig, save, verbosity)
    if return_fig:
        return fig
    if created and not save:
        plt.show()
    return None
