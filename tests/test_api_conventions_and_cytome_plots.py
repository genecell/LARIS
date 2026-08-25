"""Public API naming conventions, and cytome sources in plotting functions.

LARIS's public API is camelCase throughout (matching the PIASO ecosystem);
internal helpers are _snake_case. Rather than shipping snake_case aliases -
which would double the API surface and split the documentation - a module
__getattr__ points scanpy-habit users at the camelCase name.

Plotting functions that take a RAW EXPRESSION object accept a cytome
source, like the analysis functions do. Functions taking a results
DataFrame or a LARIS-derived AnnData do not (there is nothing to read).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import laris as la

cytome = pytest.importorskip("cytome", reason="cytome not installed")


class TestNamingConvention:
    def test_public_api_is_camel_case(self):
        """Every public callable is camelCase; no snake_case creep."""
        for mod in (la.tl, la.pp, la.pl, la.datasets):
            for name in mod.__all__:
                obj = getattr(mod, name)
                if callable(obj) and not isinstance(obj, type(la.pl.pos_cmap)):
                    assert "_" not in name, f"{name} is not camelCase"

    def test_private_helpers_stay_underscored(self):
        """The public surface must not export bare snake_case helpers."""
        for mod in (la.tl, la.pp, la.pl):
            for name in mod.__all__:
                assert not name.startswith("_"), f"{name} leaks a private name"

    @pytest.mark.parametrize("mod,wrong,right", [
        ("pp", "read_cytome", "readCytome"),
        ("tl", "run_LARIS", "runLARIS"),
        ("tl", "prepare_LR_interaction", "prepareLRInteraction"),
        ("pl", "plot_CCC_spatial", "plotCCCSpatial"),
    ])
    def test_snake_case_gets_a_pointer_not_a_bare_error(self, mod, wrong, right):
        with pytest.raises(AttributeError, match=right):
            getattr(getattr(la, mod), wrong)

    def test_unknown_name_still_plain_attribute_error(self):
        with pytest.raises(AttributeError) as exc:
            la.pp.definitely_not_a_function
        assert "did you mean" not in str(exc.value)

    def test_readcytome_alias_is_the_same_object(self):
        assert la.tl.readCytome is la.pp.readCytome


@pytest.fixture
def cytome_path(tmp_path, synthetic_adata):
    path = str(tmp_path / "plots.cytome")
    ds = cytome.from_anndata(synthetic_adata, output=path)
    ds.close()
    return path


@pytest.fixture
def laris_results(synthetic_adata, lr_df):
    """A small results table for the DataFrame-driven plots."""
    lr_adata = la.tl.prepareLRInteraction(
        synthetic_adata, lr_df, use_rep_spatial="spatial")
    _, res = la.tl.runLARIS(
        lr_adata, synthetic_adata, use_rep="spatial",
        use_rep_spatial="spatial", groupby="cell_type",
        n_cells_expressed_threshold=1, n_permutations=50)
    return res


class TestCytomeInPlots:
    def test_prepare_dotplot_adata_accepts_cytome(
            self, cytome_path, synthetic_adata, lr_df):
        """The expression argument may be a cytome; output must match."""
        lr_adata = la.tl.prepareLRInteraction(
            synthetic_adata, lr_df, use_rep_spatial="X_spatial")
        from_ad = la.pl.prepareDotPlotAdata(lr_adata, synthetic_adata,
                                            verbosity=0)
        from_ct = la.pl.prepareDotPlotAdata(lr_adata, cytome_path,
                                            verbosity=0)
        assert from_ad.shape == from_ct.shape
        assert list(from_ad.var_names) == list(from_ct.var_names)
        assert np.allclose(from_ad.X.toarray(), from_ct.X.toarray())

    def test_network_cumulative_accepts_cytome(self, laris_results, cytome_path,
                                               synthetic_adata):
        fig_ct = la.pl.plotCCCNetworkCumulative(
            laris_results, adata=cytome_path, groupby="cell_type",
            return_fig=True, verbosity=0)
        assert fig_ct is not None
        plt.close("all")

    def test_network_accepts_cytome(self, laris_results, cytome_path):
        fig = la.pl.plotCCCNetwork(
            laris_results, cell_type_of_interest="A", adata=cytome_path,
            groupby="cell_type", return_fig=True, verbosity=0)
        assert fig is not None
        plt.close("all")

    def test_dataframe_plots_need_no_cytome(self, laris_results):
        """Results-table plots have no expression argument at all."""
        import inspect
        for fn in (la.pl.plotCCCHeatmap, la.pl.plotCCCDotPlot,
                   la.pl.plotCCCDotPlotFacet):
            params = inspect.signature(fn).parameters
            assert "adata" not in params, f"{fn.__name__} unexpectedly takes adata"


class TestPlotLRDotPlotImports:
    def test_plot_lr_dotplot_runs(self):
        """Regression: the module split dropped the _compute_max_fraction
        import from _dotplot.py, so plotLRDotPlot raised NameError on any
        call. Caught by executing the tutorial code."""
        import numpy as np
        import pandas as pd
        import scipy.sparse as sp
        import anndata as ad
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import laris as la

        rng = np.random.default_rng(0)
        n = 120
        adata = ad.AnnData(
            X=sp.csr_matrix(rng.poisson(1.0, (n, 6)).astype(np.float32)),
            obs=pd.DataFrame({"ct": pd.Categorical(rng.choice(list("AB"), n))},
                             index=[f"c{i}" for i in range(n)]),
            var=pd.DataFrame(index=["L1", "R1", "L2", "R2", "L3", "R3"]),
        )
        adata.obsm["X_spatial"] = rng.random((n, 2))
        lr_df = pd.DataFrame({"ligand": ["L1", "L2", "L3"],
                              "receptor": ["R1", "R2", "R3"]})
        lr_data = la.tl.prepareLRInteraction(adata, lr_df,
                                             use_rep_spatial="X_spatial")
        adata_dot = la.pl.prepareDotPlotAdata(lr_data, adata, verbosity=0)
        out = la.pl.plotLRDotPlot(adata_dot,
                                  interactions_to_plot=["L1::R1", "L2::R2"],
                                  groupby="ct", return_fig=True)
        assert out is not None
        plt.close("all")
