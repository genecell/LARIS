"""End-to-end integration test based on the LARIS tutorial.

Uses the tonsil dataset (5,695 cells x 25,583 genes, 14 cell types) and
the bundled CellChatDB human database to run the full LARIS pipeline:

    load data → prepareLRInteraction → runLARIS → plotting functions

This mirrors the official tutorial at:
    https://genecell.github.io/LARIS/notebooks/04_LARIS_tutorial.html

Data location (checked in order):
    1. $LARIS_TEST_DATA environment variable (path to directory containing
       adata_tonsil.h5ad)
    2. Default: tests/data/ relative to the repo root

Set the environment variable to point at your local copy, e.g.:
    export LARIS_TEST_DATA=/data1/mdai/Result/single-cell/Methods/LARIS/tutorial

If the file is absent the entire module is skipped.
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

import laris as la

# ---------------------------------------------------------------------------
# Skip the entire module if the tutorial dataset is not available
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
_DEFAULT_DATA_DIR = os.path.join(_REPO_ROOT, "tests", "data")
_DATA_DIR = os.environ.get("LARIS_TEST_DATA", _DEFAULT_DATA_DIR)
_ADATA_PATH = os.path.join(_DATA_DIR, "adata_tonsil.h5ad")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(_ADATA_PATH),
    reason="Tutorial dataset not found (adata_tonsil.h5ad)",
)


# =========================================================================
# Fixtures — heavy computation, cached for the whole module
# =========================================================================

@pytest.fixture(scope="module")
def adata():
    """Load the tonsil AnnData."""
    return sc.read_h5ad(_ADATA_PATH)


@pytest.fixture(scope="module")
def lr_df():
    """Load the bundled human CellChatDB and filter to genes in adata."""
    df = la.datasets.lrDatabase("human")
    # We do the filtering lazily below once we have adata
    return df


@pytest.fixture(scope="module")
def lr_df_filtered(lr_df, adata):
    """LR database filtered to genes present in adata.var_names."""
    mask = lr_df["ligand"].isin(adata.var_names) & lr_df["receptor"].isin(
        adata.var_names
    )
    return lr_df.loc[mask].copy()


@pytest.fixture(scope="module")
def lr_adata(adata, lr_df_filtered):
    """Run prepareLRInteraction (k=20, tutorial parameters)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return la.tl.prepareLRInteraction(
            adata,
            lr_df_filtered,
            number_nearest_neighbors=20,
            use_rep_spatial="X_spatial",
        )


@pytest.fixture(scope="module")
def laris_results(lr_adata, adata):
    """Run runLARIS with tutorial parameters (returns laris_lr, res_LARIS)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        laris_lr, res = la.tl.runLARIS(
            lr_adata,
            adata,
            use_rep="X_spatial",
            n_nearest_neighbors=20,
            random_seed=27,
            n_repeats=5,
            mu=0.40,
            sigma=100,
            remove_lowly_expressed=False,
            expressed_pct=0.1,
            n_cells_expressed_threshold=100,
            n_top_lr=lr_adata.shape[1],
            by_celltype=True,
            groupby="cell_type",
            use_rep_spatial="X_spatial",
            mu_celltype=100,
            expressed_pct_celltype=0.1,
            remove_lowly_expressed_celltype=False,
            mask_threshold=1e-6,
            n_neighbors_permutation=30,
            score_threshold=1e-10,
            spatial_weight=3.0,
        )
    return laris_lr, res


@pytest.fixture(scope="module")
def laris_lr(laris_results):
    return laris_results[0]


@pytest.fixture(scope="module")
def res_LARIS(laris_results):
    return laris_results[1]


# =========================================================================
# 1. Data loading
# =========================================================================

class TestDataLoading:

    def test_adata_shape(self, adata):
        assert adata.shape == (5695, 25583)

    def test_adata_has_cell_types(self, adata):
        assert "cell_type" in adata.obs.columns
        assert adata.obs["cell_type"].nunique() == 14

    def test_adata_has_spatial(self, adata):
        assert "X_spatial" in adata.obsm
        assert adata.obsm["X_spatial"].shape == (5695, 2)

    def test_bundled_database_matches_tutorial(self, lr_df):
        assert len(lr_df) == 2951

    def test_filtered_lr_count(self, lr_df_filtered):
        # Tutorial keeps pairs where both ligand and receptor are in adata
        assert len(lr_df_filtered) > 1000


# =========================================================================
# 2. prepareLRInteraction
# =========================================================================

class TestPrepareLRInteraction:

    def test_lr_adata_shape(self, lr_adata, lr_df_filtered):
        assert lr_adata.shape[0] == 5695
        assert lr_adata.shape[1] == len(lr_df_filtered)

    def test_lr_adata_sparse(self, lr_adata):
        assert sp.issparse(lr_adata.X)

    def test_lr_adata_var_names_unique(self, lr_adata):
        assert lr_adata.var_names.is_unique

    def test_lr_adata_var_names_format(self, lr_adata):
        for name in lr_adata.var_names[:20]:
            assert "::" in name

    def test_lr_adata_preserves_obs(self, lr_adata, adata):
        assert "cell_type" in lr_adata.obs.columns
        assert len(lr_adata.obs) == len(adata.obs)

    def test_lr_adata_preserves_obsm(self, lr_adata):
        assert "X_spatial" in lr_adata.obsm


# =========================================================================
# 3. runLARIS
# =========================================================================

class TestRunLARIS:

    def test_laris_lr_is_dataframe(self, laris_lr):
        assert isinstance(laris_lr, pd.DataFrame)

    def test_laris_lr_columns(self, laris_lr):
        for col in ["ligand", "receptor", "score", "Rank"]:
            assert col in laris_lr.columns

    def test_laris_lr_sorted_descending(self, laris_lr):
        scores = laris_lr["score"].values
        assert np.all(scores[:-1] >= scores[1:])

    def test_res_LARIS_is_dataframe(self, res_LARIS):
        assert isinstance(res_LARIS, pd.DataFrame)

    def test_res_LARIS_has_expected_columns(self, res_LARIS):
        expected = [
            "sender", "receiver", "ligand", "receptor",
            "interaction_name", "interaction_score",
            "p_value", "p_value_fdr", "nlog10_p_value_fdr",
        ]
        for col in expected:
            assert col in res_LARIS.columns, f"Missing column: {col}"

    def test_res_LARIS_nonempty(self, res_LARIS):
        assert len(res_LARIS) > 0

    def test_pvalues_in_range(self, res_LARIS):
        for col in ["p_value", "p_value_fdr"]:
            vals = res_LARIS[col].dropna()
            assert (vals >= 0).all(), f"{col} has negative values"
            assert (vals <= 1).all(), f"{col} has values > 1"

    def test_significant_interactions_exist(self, res_LARIS):
        n_sig = (res_LARIS["p_value_fdr"] < 0.05).sum()
        assert n_sig > 0, "Expected some significant interactions"

    def test_pvalue_diversity(self, res_LARIS):
        """After dedup fix, p-values should not be massively duplicated."""
        pvals = res_LARIS["p_value"].dropna().values
        nontrivial = pvals[pvals < 1.0]
        n_unique = len(np.unique(nontrivial))
        assert n_unique > 50, (
            f"Only {n_unique} distinct non-trivial p-values — "
            "possible regression of dedup fix"
        )

    def test_all_cell_types_present(self, res_LARIS, adata):
        senders = set(res_LARIS["sender"].unique())
        receivers = set(res_LARIS["receiver"].unique())
        cell_types = set(adata.obs["cell_type"].unique())
        # Not all cell types need to appear, but most should
        assert len(senders & cell_types) >= 10
        assert len(receivers & cell_types) >= 10

    def test_interaction_name_format(self, res_LARIS):
        for name in res_LARIS["interaction_name"].head(20):
            assert "::" in name

    def test_no_fully_duplicated_rows(self, res_LARIS):
        n_total = len(res_LARIS)
        n_dedup = res_LARIS.drop_duplicates().shape[0]
        assert n_total == n_dedup


# =========================================================================
# 4. Plotting functions (non-interactive, just check they run)
# =========================================================================

class TestPlotting:
    """Verify that plotting functions execute without error.

    We use matplotlib Agg backend and close all figures to avoid display.
    """

    def test_plotCCCHeatmap_default(self, res_LARIS):
        la.pl.plotCCCHeatmap(res_LARIS)
        plt.close("all")

    def test_plotCCCHeatmap_filtered(self, res_LARIS):
        la.pl.plotCCCHeatmap(
            res_LARIS,
            cmap="Purples",
            filter_significant=True,
            p_value_col="p_value_fdr",
            threshold=0.05,
            filter_by_interaction_score=True,
            threshold_interaction_score=0.01,
            cluster=True,
        )
        plt.close("all")

    def test_plotCCCNetwork_sending(self, res_LARIS, adata):
        la.pl.plotCCCNetwork(
            res_LARIS,
            "B_germinal_center",
            interaction_direction="sending",
            adata=adata,
        )
        plt.close("all")

    def test_plotCCCNetwork_receiving_filtered(self, res_LARIS, adata):
        la.pl.plotCCCNetwork(
            res_LARIS,
            "B_germinal_center",
            interaction_direction="receiving",
            adata=adata,
            filter_significant=True,
            p_value_col="p_value_fdr",
            threshold=0.05,
            filter_by_interaction_score=True,
            threshold_interaction_score=0.01,
        )
        plt.close("all")

    def test_plotCCCNetworkCumulative(self, res_LARIS, adata):
        la.pl.plotCCCNetworkCumulative(
            res_LARIS,
            adata=adata,
            groupby="cell_type",
            filter_significant=True,
            p_value_col="p_value_fdr",
            threshold=0.05,
            filter_by_interaction_score=True,
            threshold_interaction_score=0.01,
            edge_width_scale=25,
        )
        plt.close("all")

    def test_plotCCCDotPlot(self, res_LARIS):
        la.pl.plotCCCDotPlot(
            res_LARIS,
            interactions_to_plot=["CCL21::CCR7", "CD40LG::CD40"],
            senders=["MRC", "MRC"],
            receivers=["T_CD4", "T_follicular_helper"],
        )
        plt.close("all")

    def test_plotCCCDotPlotFacet(self, res_LARIS):
        la.pl.plotCCCDotPlotFacet(
            res_LARIS,
            senders=["MRC", "FDC_LZDZ", "B_naive"],
            receivers=["B_naive", "B_germinal_center", "T_CD4"],
            interactions_to_plot=["COL1A1::CD44", "APP::CD74", "IL7::IL7R"],
            filter_significant=False,
            filter_by_interaction_score=False,
            n_top=3000,
        )
        plt.close("all")

    def test_prepareDotPlotAdata(self, lr_adata, adata):
        adata_dp = la.pl.prepareDotPlotAdata(lr_adata, adata)
        assert isinstance(adata_dp, type(adata))
        assert adata_dp.shape[0] == adata.shape[0]

    def test_plotLRDotPlot(self, lr_adata, adata):
        adata_dp = la.pl.prepareDotPlotAdata(lr_adata, adata)
        # Pick interactions that are guaranteed to exist in lr_adata.var_names
        interactions = [
            n for n in lr_adata.var_names[:3]
        ]
        la.pl.plotLRDotPlot(
            adata_dp,
            interactions,
            groupby="cell_type",
        )
        plt.close("all")

    def test_plotCCCSpatial(self, lr_adata, adata):
        # Copy cell_type_colors if present
        if "cell_type_colors" in adata.uns:
            lr_adata.uns["cell_type_colors"] = adata.uns["cell_type_colors"].copy()

        # Pick an interaction that exists
        interaction = lr_adata.var_names[0]
        la.pl.plotCCCSpatial(
            lr_adata,
            basis="X_spatial",
            interaction=interaction,
            cell_type="cell_type",
            highlight_all_expressing=True,
            size=40,
        )
        plt.close("all")


# =========================================================================
# 5. Spatial offset on real data
# =========================================================================

class TestSpatialOffsetRealData:
    """Test spatialOffsetMultisample on the tonsil data split into pseudo-samples."""

    def test_pseudo_multisample(self, adata):
        """Split tonsil into 2 pseudo-samples and verify offset works."""
        import anndata as ad

        adata_copy = adata.copy()
        n = adata_copy.n_obs
        adata_copy.obs["pseudo_sample"] = (
            ["sampleA"] * (n // 2) + ["sampleB"] * (n - n // 2)
        )

        la.pp.spatialOffsetMultisample(
            adata_copy,
            sampleKey="pseudo_sample",
            spatialKey="X_spatial",
        )

        assert "spatial_offset_info" in adata_copy.uns
        info = adata_copy.uns["spatial_offset_info"]
        assert set(info["samples"].keys()) == {"sampleA", "sampleB"}
