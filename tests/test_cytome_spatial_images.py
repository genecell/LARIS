"""Tissue-image overlay from cytome sources (cytome >= 0.2.6).

Implements the LARIS side of the cytome spatial-image spec: the
`spatial_images` accessor joins the image waterfall in plotCCCSpatial,
readCytome carries images into `uns['spatial']`, and the overlay math
places the IMAGE in coordinate units (extent) rather than scaling the
coordinates.
"""
import math
import warnings

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import laris as la
from laris.plotting import _resolve_background_image
from laris.tools._io import _spatial_uns_from, _strip_embedding_prefix


class _FakeSpatialImages:
    """Stand-in for cytome's accessor, matching the shipped contract."""

    def __init__(self, d, raises=None):
        self._d = d
        self._raises = raises

    def as_uns(self):
        if self._raises is not None:
            raise self._raises
        return self._d

    def keys(self):
        return [(l, k) for l, v in self._d.items() for k in v["images"]]

    def libraries(self):
        return list(self._d)

    def scalefactors(self, lib):
        return self._d[lib]["scalefactors"]

    def __getitem__(self, lk):
        return self._d[lk[0]]["images"][lk[1]]

    def info(self, lib, key):
        a = self[lib, key]
        return {"height": a.shape[0], "width": a.shape[1],
                "channels": (a.shape[2] if a.ndim == 3 else 1),
                "dtype": str(a.dtype), "format": "raw"}

    def crop(self, lib, key, x, y, units="fullres", pad=0.0):
        a = self[lib, key]
        sf = (self.scalefactors(lib).get(f"tissue_{key}_scalef", 1.0)
              if units == "fullres" else 1.0)
        x0, x1 = sorted(x); y0, y1 = sorted(y)
        x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
        c0 = max(0, math.floor(x0 * sf)); c1 = min(a.shape[1], math.ceil(x1 * sf))
        r0 = max(0, math.floor(y0 * sf)); r1 = min(a.shape[0], math.ceil(y1 * sf))
        return a[r0:r1, c0:c1], {"x_offset": c0, "y_offset": r0, "scalef": sf,
                                 "extent": (c0 / sf, c1 / sf, r1 / sf, r0 / sf)}


class _AccessorOnly:
    """Minimal object carrying only the accessor (no uns)."""
    def __init__(self, acc):
        self.spatial_images = acc


def _uns_dict(img, scalef=0.5, spot=None, lib="lib0", key="hires"):
    sfs = {f"tissue_{key}_scalef": scalef}
    if spot is not None:
        sfs["spot_diameter_fullres"] = spot
    return {lib: {"images": {key: img}, "scalefactors": sfs}}


@pytest.fixture
def lr_adata_small(synthetic_adata, lr_df):
    return la.tl.prepareLRInteraction(
        synthetic_adata, lr_df, use_rep_spatial="spatial")


# ---------------------------------------------------------------- waterfall

class TestWaterfall:
    def test_accessor_beats_uns(self, lr_adata_small):
        acc_img = np.zeros((10, 10, 3), dtype=np.uint8)
        uns_img = np.ones((20, 20, 3), dtype=np.uint8)
        src = _AccessorOnly(_FakeSpatialImages(_uns_dict(acc_img, 1.0)))
        lr_adata_small.uns["spatial"] = _uns_dict(uns_img, 1.0)
        ctx = _resolve_background_image(lr_adata_small, adata=src)
        assert ctx["img"].shape == acc_img.shape

    def test_explicit_img_beats_accessor(self, lr_adata_small):
        acc_img = np.zeros((10, 10, 3), dtype=np.uint8)
        explicit = np.ones((7, 7, 3), dtype=np.uint8)
        src = _AccessorOnly(_FakeSpatialImages(_uns_dict(acc_img, 1.0)))
        ctx = _resolve_background_image(lr_adata_small, adata=src,
                                        img=explicit, scale_factor=1.0)
        assert ctx["img"].shape == explicit.shape

    def test_no_image_returns_none(self, lr_adata_small):
        assert _resolve_background_image(lr_adata_small) is None

    def test_empty_accessor_is_no_image(self, lr_adata_small):
        src = _AccessorOnly(_FakeSpatialImages({}))
        assert _resolve_background_image(lr_adata_small, adata=src) is None


class TestGracefulDegradation:
    def test_import_error_warns_and_degrades(self, lr_adata_small):
        acc = _FakeSpatialImages({}, raises=ImportError("needs tifffile"))
        src = _AccessorOnly(acc)
        with pytest.warns(UserWarning, match="not decodable"):
            assert _spatial_uns_from(src) == {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = la.pl.plotCCCSpatial(
                lr_adata_small, "spatial", str(lr_adata_small.var_names[0]),
                color_by="score", adata=src, return_fig=True)
        assert fig is not None       # plot survives, imageless
        plt.close(fig)

    def test_other_exception_also_degrades(self, lr_adata_small):
        acc = _FakeSpatialImages({}, raises=ValueError("imagecodecs missing"))
        with pytest.warns(UserWarning, match="could not be read"):
            assert _spatial_uns_from(_AccessorOnly(acc)) == {}


# ------------------------------------------------------------ library rules

class TestLibrarySelection:
    def _two_libs(self):
        d = _uns_dict(np.zeros((8, 8, 3), np.uint8), 1.0, lib="A")
        d.update(_uns_dict(np.ones((8, 8, 3), np.uint8), 1.0, lib="B"))
        return _AccessorOnly(_FakeSpatialImages(d))

    def test_ambiguous_raises_naming_libraries(self, lr_adata_small):
        with pytest.raises(ValueError, match=r"A.*B|\['A', 'B'\]"):
            _resolve_background_image(lr_adata_small, adata=self._two_libs())

    def test_explicit_library_id(self, lr_adata_small):
        ctx = _resolve_background_image(lr_adata_small, adata=self._two_libs(),
                                        library_id="B")
        assert ctx["library"] == "B" and ctx["img"].max() == 1

    def test_unknown_library_raises(self, lr_adata_small):
        with pytest.raises(KeyError, match="no stored image"):
            _resolve_background_image(lr_adata_small, adata=self._two_libs(),
                                      library_id="Z")

    def test_cells_narrow_to_one_library(self, lr_adata_small):
        """Cells spanning exactly one library disambiguate automatically."""
        ctx = _resolve_background_image(
            lr_adata_small, adata=self._two_libs(),
            library_values=np.array(["B"] * lr_adata_small.n_obs))
        assert ctx["library"] == "B"

    def test_missing_img_key_raises(self, lr_adata_small):
        src = _AccessorOnly(_FakeSpatialImages(
            _uns_dict(np.zeros((8, 8, 3), np.uint8), 1.0)))
        with pytest.raises(KeyError, match="no image 'lowres'"):
            _resolve_background_image(lr_adata_small, adata=src,
                                      img_key="lowres")


# ------------------------------------------------------------- overlay math

class TestOverlayMath:
    def test_extent_is_image_in_coordinate_units(self, lr_adata_small):
        """scalef=0.5 with a 60x40 image (HxW) -> extent (0, 80, 120, 0).

        This is the test that catches the coordinates-scaled-down variant:
        the picture looks plausible at the wrong scale, but every ROI
        rectangle downstream is off by 2x.
        """
        img = np.zeros((60, 40, 3), dtype=np.uint8)   # H=60, W=40
        src = _AccessorOnly(_FakeSpatialImages(_uns_dict(img, 0.5)))
        ctx = _resolve_background_image(lr_adata_small, adata=src)
        assert ctx["extent"] == (0.0, 80.0, 120.0, 0.0)

    def test_spot_diameter_carried(self, lr_adata_small):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        src = _AccessorOnly(_FakeSpatialImages(_uns_dict(img, 1.0, spot=9.0)))
        ctx = _resolve_background_image(lr_adata_small, adata=src)
        assert ctx["spot_diameter"] == 9.0

    def test_orientation_top_of_image_is_top_of_plot(self, synthetic_adata, lr_df):
        """Red top half / blue bottom half: a spot near y=0 must render over
        RED. Every sign error here (inverted y, wrong extent order, a stray
        invert_yaxis) turns it blue."""
        adata = synthetic_adata.copy()
        rng = np.random.default_rng(1)
        coords = np.column_stack([                       # all spots near top
            rng.uniform(45, 55, adata.n_obs),
            rng.uniform(3, 7, adata.n_obs)])
        adata.obsm["spatial"] = coords
        lr_adata = la.tl.prepareLRInteraction(
            adata, lr_df, use_rep_spatial="spatial")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:50] = [255, 0, 0]      # top half red
        img[50:] = [0, 0, 255]      # bottom half blue
        src = _AccessorOnly(_FakeSpatialImages(_uns_dict(img, 1.0)))
        fig = la.pl.plotCCCSpatial(
            lr_adata, "spatial", str(lr_adata.var_names[0]),
            color_by="score", adata=src, crop=False, colorbar=False,
            score_threshold=1e12,          # draw no coloured spots
            return_fig=True)
        ax = fig.axes[0]
        ax.set_xlim(0, 100); ax.set_ylim(100, 0)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        # sample the rendered background at data point (50, 5)
        px, py = ax.transData.transform((50.0, 5.0))
        row = int(round(fig.canvas.get_width_height()[1] - py))
        col = int(round(px))
        pixel = buf[row, col, :3]
        assert pixel[0] > pixel[2], f"expected red-ish, got {pixel}"
        plt.close(fig)


# --------------------------------------------------------- readCytome path

class TestReadCytomePassthrough:
    def test_embedding_key_both_generations(self):
        assert _strip_embedding_prefix("RNA_spatial") == "spatial"
        assert _strip_embedding_prefix("RNA_obsm_X_spatial") == "X_spatial"
        assert _strip_embedding_prefix("ATAC_umap", "RNA") is None

    def test_images_travel_through_readcytome(self, tmp_path, synthetic_adata):
        cytome = pytest.importorskip("cytome")
        img = (np.random.default_rng(0).random((16, 12, 3)) * 255).astype(np.uint8)
        adata = synthetic_adata.copy()
        adata.uns["spatial"] = _uns_dict(img, 0.5, spot=6.0)
        path = str(tmp_path / "img.cytome")
        ds = cytome.from_anndata(adata, output=path)
        ds.close()
        out = la.tl.readCytome(path, genes=["Gene0", "Gene1"])
        assert "spatial" in out.uns
        stored = out.uns["spatial"]["lib0"]["images"]["hires"]
        assert np.array_equal(np.asarray(stored), img)
        assert "spatial" in out.obsm      # embedding key resolved


class TestDegenerateGeometry:
    def test_identical_coordinates_do_not_produce_nan(self, synthetic_adata, lr_df):
        """Adaptive bandwidth is mean(d)/2; all-identical coordinates make
        that zero, which would divide every weight to NaN."""
        adata = synthetic_adata.copy()
        adata.obsm["spatial"] = np.zeros((adata.n_obs, 2))
        with pytest.warns(UserWarning, match="all k-NN distances are zero"):
            lr_adata = la.tl.prepareLRInteraction(
                adata, lr_df, use_rep_spatial="spatial")
        assert not np.isnan(lr_adata.X.toarray()).any()


class TestLrAdataCarriesImages:
    def test_prepare_propagates_spatial_uns(self, synthetic_adata, lr_df):
        """lr_adata inherits uns['spatial'], so the overlay needs no adata=."""
        adata = synthetic_adata.copy()
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        adata.uns["spatial"] = _uns_dict(img, 1.0)
        lr_adata = la.tl.prepareLRInteraction(
            adata, lr_df, use_rep_spatial="spatial")
        assert "spatial" in lr_adata.uns
        ctx = _resolve_background_image(lr_adata)
        assert ctx is not None and ctx["img"].shape == img.shape
