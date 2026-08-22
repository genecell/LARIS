"""Tests for the v0.9.4 bugfix release.

Covers the issues reported in the 2026-08 user email:
1. Silent gene mis-mapping via np.searchsorted in prepareLRInteraction
2. runLARIS IndexError with < 100 LR pairs (percent_top)
3. matplotlib >= 3.9 import compatibility (_get_cmap helper)
6. pandas-3.0-safe fillna
7. plotCCCDotPlot raising on invalid input
8. Network-plot color key resolution
9. No spurious avg-expression warning
Plus the additive features: scale-factor storage / rescale flag,
fractional n_cells_expressed_threshold, FDR floor warning.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import laris as la


# ---------------------------------------------------------------------------
# 1. unmatched gene names in prepareLRInteraction
# ---------------------------------------------------------------------------

class TestUnmatchedGenes:
    def test_unmatched_dropped_with_warning(self, synthetic_adata, lr_df):
        bad = lr_df.copy()
        # 'Gene0005' does not exist; sorts between Gene0 and Gene1, which is
        # exactly the silent-mis-mapping case from the report.
        bad.loc[0, "ligand"] = "Gene0005"
        with pytest.warns(UserWarning, match="absent from"):
            lr_adata = la.tl.prepareLRInteraction(
                synthetic_adata, bad, use_rep_spatial="spatial"
            )
        assert lr_adata.shape[1] == len(lr_df) - 1
        assert "Gene0005::Gene1" not in lr_adata.var_names

    def test_unmatched_error_mode_raises(self, synthetic_adata, lr_df):
        bad = lr_df.copy()
        bad.loc[0, "receptor"] = "NOT_A_GENE"
        with pytest.raises(ValueError, match="NOT_A_GENE"):
            la.tl.prepareLRInteraction(
                synthetic_adata, bad, use_rep_spatial="spatial",
                unmatched="error",
            )

    def test_matched_result_identical_to_prefiltered(self, synthetic_adata, lr_df):
        """Dropping unmatched pairs must equal user pre-filtering."""
        bad = lr_df.copy()
        bad.loc[3, "ligand"] = "Gene0005"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auto = la.tl.prepareLRInteraction(
                synthetic_adata, bad, use_rep_spatial="spatial"
            )
        prefiltered = lr_df.drop(index=3).reset_index(drop=True)
        manual = la.tl.prepareLRInteraction(
            synthetic_adata, prefiltered, use_rep_spatial="spatial"
        )
        assert list(auto.var_names) == list(manual.var_names)
        assert np.allclose(auto.X.toarray(), manual.X.toarray())

    def test_no_warning_when_all_matched(self, synthetic_adata, lr_df):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            la.tl.prepareLRInteraction(
                synthetic_adata, lr_df, use_rep_spatial="spatial"
            )

    def test_all_unmatched_raises(self, synthetic_adata):
        bad = pd.DataFrame({"ligand": ["NOPE1"], "receptor": ["NOPE2"]})
        with pytest.raises(ValueError, match="no ligand-receptor pairs remain"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                la.tl.prepareLRInteraction(
                    synthetic_adata, bad, use_rep_spatial="spatial"
                )

    def test_invalid_unmatched_value(self, synthetic_adata, lr_df):
        with pytest.raises(ValueError, match="unmatched must be"):
            la.tl.prepareLRInteraction(
                synthetic_adata, lr_df, use_rep_spatial="spatial",
                unmatched="ignore",
            )


# ---------------------------------------------------------------------------
# 2. runLARIS with < 100 LR pairs
# ---------------------------------------------------------------------------

class TestFewLRPairs:
    @pytest.mark.parametrize("n_pairs", [30, 60, 99])
    def test_runlaris_small_database(self, synthetic_adata, lr_df, n_pairs):
        small = lr_df.iloc[:n_pairs].reset_index(drop=True)
        lr_adata = la.tl.prepareLRInteraction(
            synthetic_adata, small, use_rep_spatial="spatial"
        )
        # Must not raise IndexError from calculate_qc_metrics.
        res = la.tl.runLARIS(
            lr_adata,
            adata=synthetic_adata,
            use_rep="spatial",
            use_rep_spatial="spatial",
            groupby="cell_type",
            n_cells_expressed_threshold=1,
            n_permutations=50,
        )
        assert res is not None


# ---------------------------------------------------------------------------
# 3. matplotlib compatibility
# ---------------------------------------------------------------------------

class TestMatplotlibCompat:
    def test_import_defines_colormap(self):
        import laris.plotting as pl
        assert pl.pos_cmap.N == 256

    def test_get_cmap_helper(self):
        from laris.plotting import _get_cmap
        cmap = _get_cmap("magma_r", 256)
        assert cmap.N == 256
        cmap_nolut = _get_cmap("tab10")
        assert cmap_nolut(0) is not None

    def test_no_module_level_get_cmap_calls(self):
        """Guard against reintroducing cm.get_cmap (removed in mpl 3.9)."""
        import inspect
        import laris.plotting as pl
        src = inspect.getsource(pl)
        for line in src.splitlines():
            stripped = line.split("#")[0]
            if "cm.get_cmap(" in stripped and "def _get_cmap" not in stripped:
                # Only the guarded fallback inside _get_cmap may call it.
                assert "return cm.get_cmap" in stripped, (
                    f"unguarded cm.get_cmap call: {line!r}"
                )


# ---------------------------------------------------------------------------
# 7. plotCCCDotPlot input validation
# ---------------------------------------------------------------------------

class TestDotPlotValidation:
    @pytest.fixture
    def fake_results(self):
        return pd.DataFrame({
            "sender": ["A", "B"],
            "receiver": ["B", "A"],
            "interaction_name": ["G1::G2", "G3::G4"],
            "interaction_score": [0.5, 0.4],
            "p_value_fdr": [0.01, 0.02],
        })

    def test_length_mismatch_raises(self, fake_results):
        with pytest.raises(ValueError, match="plotCCCDotPlotFacet"):
            la.pl.plotCCCDotPlot(
                fake_results,
                interactions_to_plot=["G1::G2"],
                senders=["A", "B"],
                receivers=["B"],
            )

    def test_missing_both_raises(self, fake_results):
        with pytest.raises(ValueError, match="Must provide"):
            la.pl.plotCCCDotPlot(
                fake_results, interactions_to_plot=["G1::G2"]
            )

    def test_bad_pair_format_raises(self, fake_results):
        with pytest.raises(ValueError, match="Invalid pair format"):
            la.pl.plotCCCDotPlot(
                fake_results,
                interactions_to_plot=["G1::G2"],
                sender_receiver_pairs=["A=>B"],
            )


# ---------------------------------------------------------------------------
# 8. network plot color-key resolution
# ---------------------------------------------------------------------------

class TestColorKeyResolution:
    def test_default_uses_groupby_colors(self, synthetic_adata):
        from laris.plotting import _resolve_cell_type_colors
        synthetic_adata.uns["cell_type_colors"] = ["#ff0000", "#00ff00", "#0000ff"]
        mapping = _resolve_cell_type_colors(synthetic_adata, "cell_type")
        # Pairing must follow categories order, not appearance order.
        cats = list(synthetic_adata.obs["cell_type"].cat.categories)
        assert mapping == dict(zip(cats, ["#ff0000", "#00ff00", "#0000ff"]))

    def test_missing_key_generates_palette(self, synthetic_adata):
        from laris.plotting import _resolve_cell_type_colors
        assert "cell_type_colors" not in synthetic_adata.uns
        mapping = _resolve_cell_type_colors(synthetic_adata, "cell_type")
        assert set(mapping) == {"A", "B", "C"}
        assert len(set(mapping.values())) == 3  # distinct colors

    def test_explicit_key_still_honoured(self, synthetic_adata):
        from laris.plotting import _resolve_cell_type_colors
        synthetic_adata.uns["my_palette"] = ["#111111", "#222222", "#333333"]
        mapping = _resolve_cell_type_colors(
            synthetic_adata, "cell_type", "my_palette"
        )
        assert mapping["A"] == "#111111"


# ---------------------------------------------------------------------------
# 6 & 9 & additive features, via a full runLARIS pass
# ---------------------------------------------------------------------------

class TestRunLARISBehaviour:
    @pytest.fixture
    def full_run(self, synthetic_adata, lr_df):
        lr_adata = la.tl.prepareLRInteraction(
            synthetic_adata, lr_df, use_rep_spatial="spatial"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            laris_lr, res = la.tl.runLARIS(
                lr_adata,
                adata=synthetic_adata,
                use_rep="spatial",
                use_rep_spatial="spatial",
                groupby="cell_type",
                n_cells_expressed_threshold=1,
                n_permutations=50,
            )
        return lr_adata, res, caught

    def test_no_avg_expression_warning(self, full_run):
        _, _, caught = full_run
        assert not any(
            "No genes or groups specified" in str(w.message) for w in caught
        )

    def test_scale_factor_stored(self, full_run):
        lr_adata, res, _ = full_run
        assert "laris_scale_factor" in lr_adata.uns
        assert lr_adata.uns["laris_scale_factor"] > 0
        assert res.attrs.get("laris_scale_factor") == pytest.approx(
            lr_adata.uns["laris_scale_factor"]
        )

    def test_fdr_never_nan(self, full_run):
        _, res, _ = full_run
        assert not res["p_value_fdr"].isna().any()
        assert (res["p_value_fdr"] <= 1.0).all()

    def test_fdr_floor_warning_emitted_when_binding(self, full_run):
        # n_permutations=50 -> floor 1/51; any group testing > 2 interactions
        # has min achievable FDR > 0.05, so the warning must fire here.
        _, _, caught = full_run
        assert any(
            "minimum achievable FDR" in str(w.message) for w in caught
        )

    def test_rescale_false(self, synthetic_adata, lr_df):
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
            rescale=False,
        )
        assert lr_adata.uns["laris_scale_factor"] == 1.0
        assert res.attrs["laris_scale_factor"] == 1.0

    def test_rescale_scales_scores(self, synthetic_adata, lr_df):
        lr_adata = la.tl.prepareLRInteraction(
            synthetic_adata, lr_df, use_rep_spatial="spatial"
        )
        kwargs = dict(
            adata=synthetic_adata,
            use_rep="spatial",
            use_rep_spatial="spatial",
            groupby="cell_type",
            n_cells_expressed_threshold=1,
            n_permutations=50,
        )
        _, res_scaled = la.tl.runLARIS(lr_adata.copy(), **kwargs)
        _, res_raw = la.tl.runLARIS(lr_adata.copy(), rescale=False, **kwargs)
        factor = res_scaled.attrs["laris_scale_factor"]
        merged = res_scaled.merge(
            res_raw,
            on=["sender", "receiver", "interaction_name"],
            suffixes=("_scaled", "_raw"),
        )
        assert np.allclose(
            merged["interaction_score_scaled"],
            merged["interaction_score_raw"] * factor,
        )


class TestFractionalThreshold:
    def test_fraction_equivalent_to_count(self, synthetic_adata, lr_df):
        lr_adata = la.tl.prepareLRInteraction(
            synthetic_adata, lr_df, use_rep_spatial="spatial"
        )
        kwargs = dict(
            adata=synthetic_adata,
            use_rep="spatial",
            use_rep_spatial="spatial",
            groupby="cell_type",
            n_permutations=50,
        )
        # 0.25 of 200 cells == 50 cells: results must be identical.
        lr_frac, res_frac = la.tl.runLARIS(
            lr_adata.copy(), n_cells_expressed_threshold=0.25, **kwargs
        )
        lr_abs, res_abs = la.tl.runLARIS(
            lr_adata.copy(), n_cells_expressed_threshold=50, **kwargs
        )
        pd.testing.assert_frame_equal(
            res_frac.reset_index(drop=True), res_abs.reset_index(drop=True)
        )
