"""Tests for laris.datasets — bundled LR databases."""

import pandas as pd
import pytest
import laris as la


class TestLrDatabase:
    """Tests for la.datasets.lrDatabase()."""

    def test_human_loads(self):
        df = la.datasets.lrDatabase("human")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 2000
        assert "interaction_name" in df.columns
        assert "ligand" in df.columns
        assert "receptor" in df.columns

    def test_mouse_loads(self):
        df = la.datasets.lrDatabase("mouse")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 2000

    def test_human_shape(self):
        df = la.datasets.lrDatabase("human")
        assert df.shape == (2951, 28)

    def test_mouse_shape(self):
        df = la.datasets.lrDatabase("mouse")
        assert df.shape == (3105, 28)

    def test_filter_pathway(self):
        df_all = la.datasets.lrDatabase("human")
        df_wnt = la.datasets.lrDatabase("human", pathway="WNT")
        assert 0 < len(df_wnt) < len(df_all)
        assert (df_wnt["pathway_name"] == "WNT").all()

    def test_filter_annotation(self):
        df = la.datasets.lrDatabase("human", annotation="Secreted Signaling")
        assert len(df) > 0
        assert (df["annotation"] == "Secreted Signaling").all()

    def test_filter_combined(self):
        df = la.datasets.lrDatabase(
            "human", pathway="WNT", annotation="Secreted Signaling"
        )
        assert len(df) > 0
        assert (df["pathway_name"] == "WNT").all()
        assert (df["annotation"] == "Secreted Signaling").all()

    def test_invalid_species_raises(self):
        with pytest.raises(ValueError, match="Unknown species"):
            la.datasets.lrDatabase("zebrafish")

    def test_case_insensitive_species(self):
        df1 = la.datasets.lrDatabase("Human")
        df2 = la.datasets.lrDatabase("HUMAN")
        assert len(df1) == len(df2)

    def test_nonexistent_pathway_returns_empty(self):
        df = la.datasets.lrDatabase("human", pathway="NONEXISTENT_PATHWAY_XYZ")
        assert len(df) == 0

    def test_required_columns_present(self):
        df = la.datasets.lrDatabase("human")
        required = [
            "interaction_name", "pathway_name", "ligand", "receptor",
            "annotation", "evidence",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
