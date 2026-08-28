"""Self-interactions must render in the network plots (issue #9).

FancyArrowPatch cannot build a path between two identical points, so an
edge with sender == receiver used to collapse to zero length and vanish
even though it was present in the graph and passed every filter.
"""

import contextlib
import io
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import FancyArrowPatch

import anndata as ad
import scipy.sparse as sp

import laris as la
from laris.plotting._network import _draw_edge_arrow


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*args, **kwargs)


@pytest.fixture
def adata():
    """Minimal object supplying the cell-type column the plots colour by."""
    rng = np.random.default_rng(0)
    n = 90
    A = ad.AnnData(X=sp.random(n, 6, density=.5, random_state=1).tocsr(),
                   var=pd.DataFrame(index=[f"G{i}" for i in range(6)]))
    A.obs["cell_type"] = pd.Categorical(rng.choice(["A", "B", "C"], n))
    return A


@pytest.fixture
def res():
    """Results table where two cell types signal to themselves."""
    rows = []
    for s, r, name, score in [
        ("A", "A", "L1::R1", 0.9),      # self-interaction
        ("B", "B", "L2::R2", 0.7),      # self-interaction
        ("A", "B", "L3::R3", 0.6),
        ("B", "A", "L4::R4", 0.5),
        ("C", "A", "L5::R5", 0.4),
        ("A", "C", "L6::R6", 0.3),
    ]:
        rows.append(dict(sender=s, receiver=r, interaction_name=name,
                         ligand=name.split("::")[0],
                         receptor=name.split("::")[1],
                         interaction_score=score, p_value=1e-4,
                         p_value_fdr=0.001))
    return pd.DataFrame(rows)


def _arrows(fig):
    return [p for ax in fig.axes for p in ax.patches
            if isinstance(p, FancyArrowPatch)]


def _path_length(patch):
    verts = patch.get_path().vertices
    return float(np.abs(np.diff(verts, axis=0)).sum())


class TestSelfLoopHelper:
    def test_self_loop_has_nonzero_extent(self):
        fig, ax = plt.subplots()
        pos = {"A": np.array([0.0, 0.0]), "B": np.array([1.0, 0.0])}
        _draw_edge_arrow(ax, pos, "A", "A", color="k", linewidth=1,
                         mutation_scale=10)
        arrows = _arrows(fig)
        assert len(arrows) == 1
        assert _path_length(arrows[0]) > 0.01
        plt.close(fig)

    def test_self_loop_stays_near_its_node(self):
        fig, ax = plt.subplots()
        pos = {"A": np.array([0.0, 0.0]), "B": np.array([1.0, 0.0]),
               "C": np.array([0.5, 1.0])}
        _draw_edge_arrow(ax, pos, "A", "A", color="k", linewidth=1,
                         mutation_scale=10)
        verts = _arrows(fig)[0].get_path().vertices
        # the loop must not wander closer to another node than to its own
        d_own = np.hypot(*(verts - pos["A"]).T).min()
        d_other = min(np.hypot(*(verts - pos[n]).T).min() for n in ("B", "C"))
        assert d_own < d_other
        plt.close(fig)

    def test_regular_edge_unchanged(self):
        fig, ax = plt.subplots()
        pos = {"A": np.array([0.0, 0.0]), "B": np.array([1.0, 0.0])}
        _draw_edge_arrow(ax, pos, "A", "B", color="k", linewidth=1,
                         mutation_scale=10)
        arrows = _arrows(fig)
        assert len(arrows) == 1
        assert _path_length(arrows[0]) > 0.5
        plt.close(fig)

    def test_single_node_graph_does_not_divide_by_zero(self):
        fig, ax = plt.subplots()
        pos = {"A": np.array([0.0, 0.0])}
        _draw_edge_arrow(ax, pos, "A", "A", color="k", linewidth=1,
                         mutation_scale=10)
        verts = _arrows(fig)[0].get_path().vertices
        assert np.isfinite(verts).all()
        plt.close(fig)


class TestNetworkPlotsRenderSelfLoops:
    def test_plotCCCNetwork_draws_the_self_edge(self, res, adata):
        fig, _ = _quiet(la.pl.plotCCCNetwork, res,
                        cell_type_of_interest="A", data=adata,
                        interaction_direction="sending", return_fig=True)
        arrows = _arrows(fig)
        # A->A and A->B and A->C are all outgoing from A
        assert len(arrows) >= 2
        assert all(_path_length(a) > 0.01 for a in arrows), \
            "an edge rendered with zero extent (self-loop regression)"
        plt.close(fig)

    def test_plotCCCNetworkCumulative_draws_self_edges(self, res, adata):
        fig, _ = _quiet(la.pl.plotCCCNetworkCumulative, res, data=adata,
                        return_fig=True)
        arrows = _arrows(fig)
        assert len(arrows) >= 4
        assert all(_path_length(a) > 0.01 for a in arrows), \
            "an edge rendered with zero extent (self-loop regression)"
        plt.close(fig)

    def test_self_edge_count_matches_the_data(self, res, adata):
        fig, _ = _quiet(la.pl.plotCCCNetworkCumulative, res, data=adata,
                        return_fig=True)
        n_expected = res[["sender", "receiver"]].drop_duplicates().shape[0]
        assert len(_arrows(fig)) == n_expected
        plt.close(fig)
