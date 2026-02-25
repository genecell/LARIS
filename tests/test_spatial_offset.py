"""Tests for la.pp.spatialOffsetMultisample()."""

import numpy as np
import pandas as pd
import anndata as ad
import pytest
from sklearn.neighbors import kneighbors_graph

import laris as la


def _make_two_samples(rng, n1=200, n2=200, overlap=True):
    """Helper: create two sample AnnData objects with optional coordinate overlap."""
    coords1 = rng.standard_normal((n1, 2)) * 10
    if overlap:
        coords2 = rng.standard_normal((n2, 2)) * 10 + 5  # overlapping
    else:
        coords2 = rng.standard_normal((n2, 2)) * 10 + 1000  # far apart

    a1 = ad.AnnData(
        X=rng.random((n1, 50)).astype(np.float32),
        obsm={"X_spatial": coords1},
    )
    a1.obs["sample_id"] = "sample_A"

    a2 = ad.AnnData(
        X=rng.random((n2, 50)).astype(np.float32),
        obsm={"X_spatial": coords2},
    )
    a2.obs["sample_id"] = "sample_B"

    merged = ad.concat([a1, a2])
    merged.obs_names_make_unique()
    return merged


def _count_cross_sample_edges(adata, k=10):
    """Count kNN edges that connect cells from different samples."""
    knn = kneighbors_graph(adata.obsm["X_spatial"], n_neighbors=k, mode="connectivity")
    labels = adata.obs["sample_id"].values
    rows, cols = knn.nonzero()
    return sum(1 for r, c in zip(rows, cols) if labels[r] != labels[c])


class TestSpatialOffsetMultisample:
    """Tests for la.pp.spatialOffsetMultisample()."""

    def test_eliminates_cross_sample_knn(self, rng):
        adata = _make_two_samples(rng, overlap=True)
        assert _count_cross_sample_edges(adata) > 0, "Precondition: overlapping coords"

        la.pp.spatialOffsetMultisample(adata, sampleKey="sample_id")
        assert _count_cross_sample_edges(adata) == 0

    def test_inplace_modification(self, rng):
        adata = _make_two_samples(rng)
        result = la.pp.spatialOffsetMultisample(adata, sampleKey="sample_id")
        assert result is None
        assert "spatial_offset_info" in adata.uns

    def test_copy_mode(self, rng):
        adata = _make_two_samples(rng)
        original_coords = adata.obsm["X_spatial"].copy()

        adata_out = la.pp.spatialOffsetMultisample(
            adata, sampleKey="sample_id", copy=True
        )
        assert adata_out is not None
        # Original should be unchanged
        np.testing.assert_array_equal(adata.obsm["X_spatial"], original_coords)
        # Copy should be modified
        assert not np.array_equal(adata_out.obsm["X_spatial"], original_coords)

    def test_uns_metadata(self, rng):
        adata = _make_two_samples(rng)
        la.pp.spatialOffsetMultisample(adata, sampleKey="sample_id")

        info = adata.uns["spatial_offset_info"]
        assert info["sample_key"] == "sample_id"
        assert info["spatial_key"] == "X_spatial"
        assert info["offset_factor"] == 2.0
        assert "spacing" in info
        assert "max_diameter" in info
        assert set(info["samples"].keys()) == {"sample_A", "sample_B"}

        for sample_info in info["samples"].values():
            assert "grid_row" in sample_info
            assert "grid_col" in sample_info
            assert "offset" in sample_info

    def test_single_sample_warns(self, rng):
        adata = ad.AnnData(
            X=rng.random((50, 10)).astype(np.float32),
            obsm={"X_spatial": rng.random((50, 2))},
        )
        adata.obs["sample_id"] = "only_one"

        with pytest.warns(UserWarning, match="only 1 sample"):
            la.pp.spatialOffsetMultisample(adata, sampleKey="sample_id")

    def test_missing_spatial_key_raises(self, rng):
        adata = _make_two_samples(rng)
        with pytest.raises(KeyError, match="bad_key"):
            la.pp.spatialOffsetMultisample(
                adata, sampleKey="sample_id", spatialKey="bad_key"
            )

    def test_missing_sample_key_raises(self, rng):
        adata = _make_two_samples(rng)
        with pytest.raises(KeyError, match="nonexistent"):
            la.pp.spatialOffsetMultisample(adata, sampleKey="nonexistent")

    def test_three_samples_grid_layout(self, rng):
        a1 = ad.AnnData(X=rng.random((50, 10)).astype(np.float32),
                         obsm={"X_spatial": rng.standard_normal((50, 2)) * 10})
        a1.obs["sample_id"] = "s1"
        a2 = ad.AnnData(X=rng.random((50, 10)).astype(np.float32),
                         obsm={"X_spatial": rng.standard_normal((50, 2)) * 10})
        a2.obs["sample_id"] = "s2"
        a3 = ad.AnnData(X=rng.random((50, 10)).astype(np.float32),
                         obsm={"X_spatial": rng.standard_normal((50, 2)) * 10})
        a3.obs["sample_id"] = "s3"

        adata = ad.concat([a1, a2, a3])
        adata.obs_names_make_unique()
        la.pp.spatialOffsetMultisample(adata, sampleKey="sample_id")

        info = adata.uns["spatial_offset_info"]
        # ceil(sqrt(3)) = 2 columns
        assert info["grid_ncols"] == 2
        assert _count_cross_sample_edges(adata) == 0

    def test_custom_grid_ncols(self, rng):
        adata = _make_two_samples(rng)
        la.pp.spatialOffsetMultisample(
            adata, sampleKey="sample_id", gridNCols=1
        )
        info = adata.uns["spatial_offset_info"]
        assert info["grid_ncols"] == 1

    def test_custom_offset_factor(self, rng):
        adata = _make_two_samples(rng)
        la.pp.spatialOffsetMultisample(
            adata, sampleKey="sample_id", offsetFactor=5.0
        )
        info = adata.uns["spatial_offset_info"]
        assert info["offset_factor"] == 5.0
        assert _count_cross_sample_edges(adata) == 0

    def test_3d_coordinates(self, rng):
        n = 100
        a1 = ad.AnnData(X=rng.random((n, 10)).astype(np.float32),
                         obsm={"X_spatial": rng.standard_normal((n, 3)) * 10})
        a1.obs["sample_id"] = "s1"
        a2 = ad.AnnData(X=rng.random((n, 10)).astype(np.float32),
                         obsm={"X_spatial": rng.standard_normal((n, 3)) * 10})
        a2.obs["sample_id"] = "s2"

        adata = ad.concat([a1, a2])
        adata.obs_names_make_unique()
        la.pp.spatialOffsetMultisample(adata, sampleKey="sample_id")

        assert adata.obsm["X_spatial"].shape[1] == 3
        assert _count_cross_sample_edges(adata) == 0
