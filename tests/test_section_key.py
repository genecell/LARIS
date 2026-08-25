"""Per-section neighbour graphs (GitHub issue #8).

When several tissue sections are tiled into one coordinate system, a global
spatial k-NN joins cells that are close on the slide but came from
different sections. ``section_key`` builds every neighbour graph in the
pipeline within sections instead.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

import laris as la
from laris.tools import _utils


@pytest.fixture
def tiled_adata():
    """Two sections placed side by side, touching at x = 100."""
    rng = np.random.default_rng(0)
    n = 150
    coords = np.vstack([
        np.column_stack([rng.uniform(0, 100, n), rng.uniform(0, 100, n)]),
        np.column_stack([rng.uniform(100, 200, n), rng.uniform(0, 100, n)]),
    ])
    sections = np.array(["A"] * n + ["B"] * n)
    n_genes = 60
    X = rng.poisson(0.5, (2 * n, n_genes)).astype(np.float32)
    # give each section its own expression offset, i.e. a batch effect that
    # cross-section neighbours would smear across the boundary
    X[n:, :20] += rng.poisson(4, (n, 20))
    adata = ad.AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame({
            "section": pd.Categorical(sections),
            "cell_type": pd.Categorical(rng.choice(list("AB"), 2 * n)),
        }, index=[f"c{i}" for i in range(2 * n)]),
        var=pd.DataFrame(index=[f"G{i}" for i in range(n_genes)]),
    )
    adata.obsm["X_spatial"] = coords
    return adata


@pytest.fixture
def lr_df():
    return pd.DataFrame({"ligand": [f"G{i}" for i in range(0, 30)],
                         "receptor": [f"G{i}" for i in range(30, 60)]})


def _cross_edges(graph, sections):
    coo = graph.tocoo()
    return int((sections[coo.row] != sections[coo.col]).sum())


class TestSectionedGraph:
    def test_no_edge_crosses_a_section(self, tiled_adata):
        sections = np.asarray(tiled_adata.obs["section"].astype(str))
        coords = tiled_adata.obsm["X_spatial"]
        globalg = _utils._sectioned_kneighbors_graph(coords, 15, sections=None)
        sectioned = _utils._sectioned_kneighbors_graph(coords, 15, sections=sections)
        assert _cross_edges(globalg, sections) > 0     # the problem is real here
        assert _cross_edges(sectioned, sections) == 0  # and this fixes it

    def test_neighbour_count_is_preserved(self, tiled_adata):
        """Sectioning must not quietly thin the graph."""
        sections = np.asarray(tiled_adata.obs["section"].astype(str))
        coords = tiled_adata.obsm["X_spatial"]
        a = _utils._sectioned_kneighbors_graph(coords, 15, sections=None)
        b = _utils._sectioned_kneighbors_graph(coords, 15, sections=sections)
        assert a.nnz == b.nnz

    def test_single_section_matches_the_global_graph(self, tiled_adata):
        """One section everywhere must reproduce the unsectioned result."""
        coords = tiled_adata.obsm["X_spatial"]
        one = np.array(["only"] * coords.shape[0])
        a = _utils._sectioned_kneighbors_graph(coords, 12, sections=None)
        b = _utils._sectioned_kneighbors_graph(coords, 12, sections=one)
        assert np.allclose(a.toarray(), b.toarray())

    def test_include_self_respected(self, tiled_adata):
        sections = np.asarray(tiled_adata.obs["section"].astype(str))
        coords = tiled_adata.obsm["X_spatial"]
        g = _utils._sectioned_kneighbors_graph(coords, 10, sections=sections,
                                               include_self=True)
        assert _cross_edges(g, sections) == 0
        assert g.nnz == coords.shape[0] * 10

    def test_small_section_warns_and_still_builds(self, tiled_adata):
        coords = tiled_adata.obsm["X_spatial"]
        sections = np.array(["A"] * (coords.shape[0] - 3) + ["tiny"] * 3)
        with pytest.warns(UserWarning, match="reduced neighbourhood"):
            g = _utils._sectioned_kneighbors_graph(coords, 15, sections=sections)
        assert _cross_edges(g, sections) == 0

    def test_single_cell_section_is_an_error(self, tiled_adata):
        coords = tiled_adata.obsm["X_spatial"]
        sections = np.array(["A"] * (coords.shape[0] - 1) + ["lonely"])
        with pytest.raises(ValueError, match="at least"):
            _utils._sectioned_kneighbors_graph(coords, 15, sections=sections)


class TestResolveSections:
    def test_missing_column_names_what_is_available(self, tiled_adata):
        with pytest.raises(ValueError, match="Available columns"):
            _utils._resolve_sections(tiled_adata, "slide_id", tiled_adata.n_obs)

    def test_none_disables(self, tiled_adata):
        assert _utils._resolve_sections(tiled_adata, None, tiled_adata.n_obs) is None


class TestPipeline:
    def test_prepare_changes_results_at_the_boundary(self, tiled_adata, lr_df):
        """Sectioning must actually reach the scores, not just the graph."""
        plain = la.tl.prepareLRInteraction(
            tiled_adata, lr_df, use_rep_spatial="X_spatial")
        sectioned = la.tl.prepareLRInteraction(
            tiled_adata, lr_df, use_rep_spatial="X_spatial", section_key="section")
        assert plain.shape == sectioned.shape
        assert not np.allclose(plain.X.toarray(), sectioned.X.toarray())

    def test_prepare_single_section_is_a_no_op(self, tiled_adata, lr_df):
        adata = tiled_adata.copy()
        adata.obs["one"] = pd.Categorical(["s"] * adata.n_obs)
        plain = la.tl.prepareLRInteraction(
            adata, lr_df, use_rep_spatial="X_spatial")
        sectioned = la.tl.prepareLRInteraction(
            adata, lr_df, use_rep_spatial="X_spatial", section_key="one")
        assert np.allclose(plain.X.toarray(), sectioned.X.toarray())

    def test_runlaris_accepts_section_key(self, tiled_adata, lr_df):
        lr_adata = la.tl.prepareLRInteraction(
            tiled_adata, lr_df, use_rep_spatial="X_spatial", section_key="section")
        _, res = la.tl.runLARIS(
            lr_adata, data=tiled_adata, use_rep="X_spatial",
            use_rep_spatial="X_spatial", groupby="cell_type",
            section_key="section", n_cells_expressed_threshold=1,
            n_permutations=30, random_seed=0)
        assert len(res) > 0

    def test_runlaris_rejects_an_unknown_column(self, tiled_adata, lr_df):
        lr_adata = la.tl.prepareLRInteraction(
            tiled_adata, lr_df, use_rep_spatial="X_spatial")
        with pytest.raises(ValueError, match="section_key"):
            la.tl.runLARIS(lr_adata, data=tiled_adata, use_rep="X_spatial",
                           groupby="cell_type", section_key="not_a_column")

    def test_random_background_stays_within_sections(self, tiled_adata):
        """The null must not cross boundaries the observed graph respects."""
        sections = np.asarray(tiled_adata.obs["section"].astype(str))
        coords = tiled_adata.obsm["X_spatial"]
        observed = _utils._sectioned_kneighbors_graph(coords, 10, sections=sections)
        shuffled = _utils._build_random_adjacency_matrix(
            tiled_adata, observed, n_nearest_neighbors=10, random_seed=0,
            sections=sections)
        assert _cross_edges(shuffled, sections) == 0

    def test_random_background_unsectioned_is_unchanged(self, tiled_adata):
        sections = np.asarray(tiled_adata.obs["section"].astype(str))
        coords = tiled_adata.obsm["X_spatial"]
        observed = _utils._sectioned_kneighbors_graph(coords, 10, sections=None)
        shuffled = _utils._build_random_adjacency_matrix(
            tiled_adata, observed, n_nearest_neighbors=10, random_seed=0)
        assert _cross_edges(shuffled, sections) > 0   # global null, as before
