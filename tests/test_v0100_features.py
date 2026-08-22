"""Tests for v0.10.0 behavior changes and features.

- adaptive kernel default (sigma='adaptive') and numeric sigma equivalence
- new defaults: mu=0.25, k=20/20, spatial_weight=3.0
- spatial_weight negative-score clamp
- cytome input support across the pipeline (requires cytome installed)
- plotCCCSpatialOverlay
"""

import os
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import laris as la

cytome = pytest.importorskip("cytome", reason="cytome not installed")

import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Kernel semantics
# ---------------------------------------------------------------------------

class TestAdaptiveKernel:
    def test_adaptive_equals_explicit_bandwidth(self, synthetic_adata, lr_df):
        """sigma='adaptive' must equal sigma=mean(d)/2 computed manually."""
        from sklearn.neighbors import kneighbors_graph
        from laris.tools._utils import _apply_knn_kernel

        g1 = kneighbors_graph(
            synthetic_adata.obsm["spatial"], n_neighbors=10, mode="distance"
        )
        manual_bw = np.mean(g1.data) / 2
        g2 = g1.copy()
        _apply_knn_kernel(g1, sigma="adaptive")
        _apply_knn_kernel(g2, sigma=manual_bw)
        assert np.allclose(g1.data, g2.data)

    def test_numeric_sigma_reproduces_old_kernel(self, synthetic_adata, lr_df):
        """sigma=100 must reproduce the pre-0.10 fixed kernel 1/exp(d/100)."""
        from sklearn.neighbors import kneighbors_graph
        from laris.tools._utils import _apply_knn_kernel

        g = kneighbors_graph(
            synthetic_adata.obsm["spatial"], n_neighbors=10, mode="distance"
        )
        expected = 1 / np.exp(g.data / 100.0)
        _apply_knn_kernel(g, sigma=100)
        assert np.allclose(g.data, expected)

    def test_invalid_sigma(self):
        from laris.tools._utils import _apply_knn_kernel
        g = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
        with pytest.raises(ValueError):
            _apply_knn_kernel(g.copy(), sigma="banana")
        with pytest.raises(ValueError):
            _apply_knn_kernel(g.copy(), sigma=-1)

    def test_adaptive_is_unit_invariant(self, synthetic_adata, lr_df):
        """Rescaling coordinates (unit change) must not change results."""
        adata_scaled = synthetic_adata.copy()
        adata_scaled.obsm["spatial"] = adata_scaled.obsm["spatial"] * 1000.0
        a = la.tl.prepareLRInteraction(
            synthetic_adata, lr_df, use_rep_spatial="spatial"
        )
        b = la.tl.prepareLRInteraction(
            adata_scaled, lr_df, use_rep_spatial="spatial"
        )
        assert np.allclose(a.X.toarray(), b.X.toarray())


class TestNewDefaults:
    def test_defaults(self):
        import inspect
        sig = inspect.signature(la.tl.runLARIS)
        assert sig.parameters["mu"].default == 0.25
        assert sig.parameters["sigma"].default == "adaptive"
        assert sig.parameters["sigma_celltype"].default == "adaptive"
        assert sig.parameters["n_nearest_neighbors"].default == 20
        assert sig.parameters["number_nearest_neighbors"].default == 20
        assert sig.parameters["spatial_weight"].default == 3.0


class TestSpatialWeightClamp:
    def test_negative_scores_clamped_not_flipped(self, synthetic_adata, lr_df):
        """Even spatial_weight must not rank negative-delta pairs positively."""
        lr_adata = la.tl.prepareLRInteraction(
            synthetic_adata, lr_df, use_rep_spatial="spatial"
        )
        _, res = la.tl.runLARIS(
            lr_adata,
            adata=synthetic_adata,
            use_rep="spatial",
            use_rep_spatial="spatial",
            groupby="cell_type",
            n_cells_expressed_threshold=1,
            n_permutations=50,
            spatial_weight=2.0,   # even exponent: sign flip without clamp
            rescale=False,
        )
        # With random data many deltas are negative; after clamping, those
        # interactions must score exactly 0, never positive via sign flip,
        # and never NaN.
        assert not res["interaction_score"].isna().any()
        assert (res["interaction_score"] >= 0).all()

    def test_fractional_weight_no_nan(self, synthetic_adata, lr_df):
        lr_adata = la.tl.prepareLRInteraction(
            synthetic_adata, lr_df, use_rep_spatial="spatial"
        )
        _, res = la.tl.runLARIS(
            lr_adata,
            adata=synthetic_adata,
            use_rep="spatial",
            use_rep_spatial="spatial",
            groupby="cell_type",
            n_cells_expressed_threshold=1,
            n_permutations=50,
            spatial_weight=1.5,   # fractional: NaN without clamp
            rescale=False,
        )
        assert not res["interaction_score"].isna().any()


# ---------------------------------------------------------------------------
# cytome input support
# ---------------------------------------------------------------------------

@pytest.fixture
def cytome_path(tmp_path, synthetic_adata):
    path = str(tmp_path / "synthetic.cytome")
    ds = cytome.from_anndata(synthetic_adata, output=path)
    ds.close()
    return path


class TestCytomeSupport:
    def test_read_cytome_subset(self, cytome_path, synthetic_adata):
        genes = ["Gene0", "Gene5", "Gene10"]
        sub = la.tl.readCytome(cytome_path, genes=genes)
        assert list(sub.var_names) == genes
        assert sub.n_obs == synthetic_adata.n_obs
        assert "X_spatial" in sub.obsm
        expected = synthetic_adata[:, genes].X.toarray()
        assert np.allclose(sub.X.toarray(), expected)

    def test_prepare_equivalent_to_anndata(self, cytome_path, synthetic_adata, lr_df):
        """cytome input must give bit-identical lr_adata to AnnData input."""
        from_ad = la.tl.prepareLRInteraction(
            synthetic_adata, lr_df, use_rep_spatial="X_spatial"
        )
        from_ct = la.tl.prepareLRInteraction(
            cytome_path, lr_df, use_rep_spatial="X_spatial"
        )
        assert list(from_ad.var_names) == list(from_ct.var_names)
        assert np.allclose(from_ad.X.toarray(), from_ct.X.toarray())

    def test_full_pipeline_equivalent(self, cytome_path, synthetic_adata, lr_df):
        """runLARIS with a cytome adata source must equal the AnnData run."""
        lr_adata = la.tl.prepareLRInteraction(
            synthetic_adata, lr_df, use_rep_spatial="X_spatial"
        )
        kwargs = dict(
            use_rep="X_spatial",
            use_rep_spatial="X_spatial",
            groupby="cell_type",
            n_cells_expressed_threshold=1,
            n_permutations=50,
            random_seed=0,
        )
        np.random.seed(0)
        _, res_ad = la.tl.runLARIS(
            lr_adata.copy(), adata=synthetic_adata, **kwargs
        )
        np.random.seed(0)
        _, res_ct = la.tl.runLARIS(
            lr_adata.copy(), adata=cytome_path, **kwargs
        )
        merged = res_ad.merge(
            res_ct,
            on=["sender", "receiver", "interaction_name"],
            suffixes=("_ad", "_ct"),
        )
        assert len(merged) == len(res_ad)
        assert np.allclose(
            merged["interaction_score_ad"], merged["interaction_score_ct"]
        )

    def test_open_dataset_accepted_and_not_closed(self, cytome_path, lr_df):
        ds = cytome.open(cytome_path)
        try:
            lr_adata = la.tl.prepareLRInteraction(
                ds, lr_df, use_rep_spatial="X_spatial"
            )
            assert lr_adata.n_vars == len(lr_df)
            assert not ds.is_closed() if callable(getattr(ds, "is_closed", None)) else True
        finally:
            ds.close()

    def test_type_error_on_garbage(self, lr_df):
        with pytest.raises(TypeError):
            la.tl.prepareLRInteraction(42, lr_df)


# ---------------------------------------------------------------------------
# plotCCCSpatialOverlay
# ---------------------------------------------------------------------------

class TestSpatialOverlay:
    @pytest.fixture
    def lr_adata_with_visium_uns(self, synthetic_adata, lr_df):
        lr_adata = la.tl.prepareLRInteraction(
            synthetic_adata, lr_df, use_rep_spatial="spatial"
        )
        # Fabricate the scanpy Visium uns['spatial'] layout with a small image
        rng = np.random.default_rng(0)
        img = rng.random((120, 120, 3)).astype(np.float32)
        lr_adata.uns["spatial"] = {
            "lib0": {
                "images": {"hires": img},
                "scalefactors": {"tissue_hires_scalef": 1.0},
            }
        }
        return lr_adata

    def test_overlay_visium_convention(self, lr_adata_with_visium_uns, tmp_path):
        interaction = str(lr_adata_with_visium_uns.var_names[0])
        out = str(tmp_path / "overlay.png")
        fig = la.pl.plotCCCSpatialOverlay(
            lr_adata_with_visium_uns, interaction,
            basis="spatial", save=out, return_fig=True,
        )
        assert fig is not None
        assert os.path.getsize(out) > 0
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_overlay_explicit_image(self, synthetic_adata, lr_df, tmp_path):
        lr_adata = la.tl.prepareLRInteraction(
            synthetic_adata, lr_df, use_rep_spatial="spatial"
        )
        img = np.zeros((110, 110, 3), dtype=np.float32)
        fig = la.pl.plotCCCSpatialOverlay(
            lr_adata, str(lr_adata.var_names[1]),
            basis="spatial", img=img, scale_factor=1.0, return_fig=True,
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_overlay_no_image_fallback(self, synthetic_adata, lr_df):
        lr_adata = la.tl.prepareLRInteraction(
            synthetic_adata, lr_df, use_rep_spatial="spatial"
        )
        fig = la.pl.plotCCCSpatialOverlay(
            lr_adata, str(lr_adata.var_names[0]),
            basis="spatial", return_fig=True,
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_explicit_img_requires_scale_factor(self, synthetic_adata, lr_df):
        lr_adata = la.tl.prepareLRInteraction(
            synthetic_adata, lr_df, use_rep_spatial="spatial"
        )
        with pytest.raises(ValueError, match="scale_factor"):
            la.pl.plotCCCSpatialOverlay(
                lr_adata, str(lr_adata.var_names[0]),
                basis="spatial", img=np.zeros((10, 10, 3)),
            )

    def test_unknown_interaction_raises(self, synthetic_adata, lr_df):
        lr_adata = la.tl.prepareLRInteraction(
            synthetic_adata, lr_df, use_rep_spatial="spatial"
        )
        with pytest.raises(ValueError, match="not found"):
            la.pl.plotCCCSpatialOverlay(lr_adata, "NOPE::NOPE", basis="spatial")


class TestPIASOConventions:
    """LARIS cytome handling mirrors PIASO's cytome compatibility layer."""

    def test_db_suffix_accepted(self, tmp_path, synthetic_adata, lr_df):
        path = str(tmp_path / "synthetic.db")
        ds = cytome.from_anndata(synthetic_adata, output=path)
        ds.close()
        lr_adata = la.tl.prepareLRInteraction(
            path, lr_df, use_rep_spatial="X_spatial"
        )
        assert lr_adata.n_vars == len(lr_df)

    def test_closed_dataset_raises_actionable(self, cytome_path, lr_df):
        ds = cytome.open(cytome_path)
        ds.close()
        with pytest.raises(RuntimeError, match="closed"):
            la.tl.prepareLRInteraction(ds, lr_df, use_rep_spatial="X_spatial")

    def test_duck_typing_no_cytome_import_needed(self):
        from laris.tools._io import _looks_like_cytome_dataset
        class NotADataset:
            pass
        assert not _looks_like_cytome_dataset(NotADataset())
        assert not _looks_like_cytome_dataset("some_string")
