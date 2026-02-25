"""Tests for LARIS bugfixes: issue #2, #5, #7 and p-value dedup."""

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import pytest

import laris as la


# =========================================================================
# Issue #2: Dense adata.X should not crash prepareLRInteraction
# =========================================================================

class TestIssue2DenseInput:
    """prepareLRInteraction must accept dense numpy array X."""

    def test_dense_input_succeeds(self, synthetic_adata_dense, lr_df):
        assert isinstance(synthetic_adata_dense.X, np.ndarray)
        lr_adata = la.tl.prepareLRInteraction(
            synthetic_adata_dense, lr_df,
            number_nearest_neighbors=10,
            use_rep_spatial="spatial",
        )
        assert lr_adata.shape[1] == len(lr_df)
        assert sp.issparse(lr_adata.X)

    def test_dense_sparse_produce_same_result(
        self, synthetic_adata, synthetic_adata_dense, lr_df
    ):
        lr_sparse = la.tl.prepareLRInteraction(
            synthetic_adata, lr_df,
            number_nearest_neighbors=10, use_rep_spatial="spatial",
        )
        lr_dense = la.tl.prepareLRInteraction(
            synthetic_adata_dense, lr_df,
            number_nearest_neighbors=10, use_rep_spatial="spatial",
        )

        X_s = lr_sparse.X.toarray() if sp.issparse(lr_sparse.X) else lr_sparse.X
        X_d = lr_dense.X.toarray() if sp.issparse(lr_dense.X) else lr_dense.X
        np.testing.assert_allclose(X_s, X_d, atol=1e-6)

    def test_sparse_input_still_works(self, synthetic_adata, lr_df):
        assert sp.issparse(synthetic_adata.X)
        lr_adata = la.tl.prepareLRInteraction(
            synthetic_adata, lr_df,
            number_nearest_neighbors=10, use_rep_spatial="spatial",
        )
        assert lr_adata.shape[1] == len(lr_df)


# =========================================================================
# Issue #5 / p-value dedup: duplicate lr_df rows
# =========================================================================

class TestPvalueDedup:
    """prepareLRInteraction must deduplicate LR pairs and emit a warning."""

    def test_dedup_warning_emitted(self, synthetic_adata, lr_df_with_duplicates):
        with pytest.warns(UserWarning, match="duplicate"):
            la.tl.prepareLRInteraction(
                synthetic_adata, lr_df_with_duplicates,
                number_nearest_neighbors=10, use_rep_spatial="spatial",
            )

    def test_dedup_produces_unique_var_names(
        self, synthetic_adata, lr_df_with_duplicates
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr_adata = la.tl.prepareLRInteraction(
                synthetic_adata, lr_df_with_duplicates,
                number_nearest_neighbors=10, use_rep_spatial="spatial",
            )
        assert lr_adata.var_names.is_unique

    def test_dedup_column_count_matches_unique(
        self, synthetic_adata, lr_df, lr_df_with_duplicates
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr_adata = la.tl.prepareLRInteraction(
                synthetic_adata, lr_df_with_duplicates,
                number_nearest_neighbors=10, use_rep_spatial="spatial",
            )
        # Should equal the number of unique pairs, not the total input rows
        n_unique = lr_df_with_duplicates.drop_duplicates(
            subset=["ligand", "receptor"]
        ).shape[0]
        assert lr_adata.n_vars == n_unique
        assert lr_adata.n_vars == len(lr_df)

    def test_no_warning_when_no_duplicates(self, synthetic_adata, lr_df):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            la.tl.prepareLRInteraction(
                synthetic_adata, lr_df,
                number_nearest_neighbors=10, use_rep_spatial="spatial",
            )
        dedup_warnings = [w for w in caught if "duplicate" in str(w.message).lower()]
        assert len(dedup_warnings) == 0


# =========================================================================
# Issue #7: Read-only array in runLARIS ranking
# =========================================================================

class TestIssue7ReadOnlyArray:
    """runLARIS must not crash on read-only arrays from pandas .values."""

    def test_run_laris_no_celltype(self, lr_adata):
        result = la.tl.runLARIS(
            lr_adata,
            by_celltype=False,
            use_rep="spatial",
            n_nearest_neighbors=10,
            n_repeats=2,
            mu=0.2,
            sigma=100,
            remove_lowly_expressed=False,
            n_cells_expressed_threshold=5,
            n_top_lr=lr_adata.shape[1],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "score" in result.columns

    def test_run_laris_with_celltype(self, lr_adata, synthetic_adata):
        laris_lr, ct_results = la.tl.runLARIS(
            lr_adata,
            synthetic_adata,
            by_celltype=True,
            groupby="cell_type",
            use_rep="spatial",
            use_rep_spatial="spatial",
            n_nearest_neighbors=10,
            n_repeats=2,
            mu=0.2,
            sigma=100,
            remove_lowly_expressed=False,
            n_cells_expressed_threshold=5,
            n_top_lr=lr_adata.shape[1],
            calculate_pvalues=False,
        )
        assert isinstance(laris_lr, pd.DataFrame)
        assert isinstance(ct_results, pd.DataFrame)
        assert len(ct_results) > 0
        assert "interaction_score" in ct_results.columns

    def test_readonly_array_copy_workaround(self):
        """Directly verify that .copy() makes read-only arrays writable."""
        arr = np.array([1.0, 2.0, 3.0])
        arr.flags.writeable = False

        arr_copy = arr.copy()
        arr_copy[0] = 99.0
        assert arr_copy[0] == 99.0
        assert arr[0] == 1.0  # original unchanged


# =========================================================================
# General pipeline sanity
# =========================================================================

class TestPipelineSanity:
    """Basic pipeline sanity checks."""

    def test_prepare_lr_output_shape(self, synthetic_adata, lr_df):
        lr_adata = la.tl.prepareLRInteraction(
            synthetic_adata, lr_df,
            number_nearest_neighbors=10, use_rep_spatial="spatial",
        )
        assert lr_adata.shape == (synthetic_adata.n_obs, len(lr_df))

    def test_prepare_lr_var_names_format(self, lr_adata):
        for name in lr_adata.var_names:
            assert "::" in name, f"var_name '{name}' missing '::' delimiter"

    def test_prepare_lr_preserves_obs(self, synthetic_adata, lr_df):
        lr_adata = la.tl.prepareLRInteraction(
            synthetic_adata, lr_df,
            number_nearest_neighbors=10, use_rep_spatial="spatial",
        )
        assert list(lr_adata.obs.columns) == list(synthetic_adata.obs.columns)
        assert len(lr_adata.obs) == len(synthetic_adata.obs)

    def test_run_laris_pvalues_in_range(self, lr_adata, synthetic_adata):
        _, ct_results = la.tl.runLARIS(
            lr_adata,
            synthetic_adata,
            by_celltype=True,
            groupby="cell_type",
            use_rep="spatial",
            use_rep_spatial="spatial",
            n_nearest_neighbors=10,
            n_repeats=2,
            mu=0.2,
            sigma=100,
            remove_lowly_expressed=False,
            n_cells_expressed_threshold=5,
            n_top_lr=lr_adata.shape[1],
            calculate_pvalues=True,
            n_permutations=100,
        )
        pvals = ct_results["p_value"].dropna()
        assert (pvals >= 0).all()
        assert (pvals <= 1).all()

        fdr = ct_results["p_value_fdr"].dropna()
        assert (fdr >= 0).all()
        assert (fdr <= 1).all()
