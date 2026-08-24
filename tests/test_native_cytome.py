"""Native cytome support: data= naming, LR-gene subsetting, streaming, lr_cytome.

Covers the v0.10.0 work that made cytome a first-class input rather than a
converted-to-AnnData shim:

- ``data=``/``lr_data=`` canonical names with deprecated aliases
- diffusing only the ligand/receptor genes, and blocking over cells
- ``return_type='auto'`` following the input type
- the LR cytome round-trip (RNA modality, RNA_lrscore layer, no counts)
- COSG streamed from disk instead of materialising an expression AnnData
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

import laris as la

cytome = pytest.importorskip("cytome", reason="cytome not installed")

import matplotlib
matplotlib.use("Agg")


@pytest.fixture
def structured_adata():
    """Expression with genuine per-cell-type marker structure.

    COSG on random noise produces near-zero scores whose ranking is
    arbitrary, which makes equivalence assertions meaningless; real marker
    structure gives scores with signal to compare.
    """
    rng = np.random.default_rng(0)
    n_cells, n_genes = 300, 120
    cell_types = rng.choice(list("ABC"), n_cells)
    X = rng.poisson(0.4, (n_cells, n_genes)).astype(np.float32)
    for k, c in enumerate("ABC"):
        rows = cell_types == c
        cols = np.arange(k * 15, (k + 1) * 15)
        X[np.ix_(rows, cols)] += rng.poisson(6, (int(rows.sum()), len(cols)))
    adata = ad.AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame({"cell_type": pd.Categorical(cell_types)},
                         index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=[f"G{i}" for i in range(n_genes)]),
    )
    adata.obsm["X_spatial"] = rng.random((n_cells, 2)) * 200
    return adata


@pytest.fixture
def structured_lr_df():
    rng = np.random.default_rng(7)
    pairs = rng.choice(120, size=(40, 2), replace=True)
    df = pd.DataFrame({"ligand": [f"G{i}" for i in pairs[:, 0]],
                       "receptor": [f"G{i}" for i in pairs[:, 1]]})
    return df.drop_duplicates(subset=["ligand", "receptor"]).reset_index(drop=True)


@pytest.fixture
def expression_cytome(tmp_path, structured_adata):
    path = str(tmp_path / "sample.cytome")
    ds = cytome.from_anndata(structured_adata, output=path)
    ds.close()
    return path


# ---------------------------------------------------------------------------
# data= / lr_data= naming
# ---------------------------------------------------------------------------

class TestDataArgumentNaming:
    def test_canonical_names_in_signature(self):
        import inspect
        assert "data" in inspect.signature(la.tl.prepareLRInteraction).parameters
        run = inspect.signature(la.tl.runLARIS).parameters
        assert "lr_data" in run and "data" in run
        for fn in (la.pl.prepareDotPlotAdata, la.pl.plotCCCSpatial):
            assert "data" in inspect.signature(fn).parameters
        assert "data" in inspect.signature(la.pl.plotCCCNetwork).parameters
        assert "data" in inspect.signature(la.pl.plotCCCNetworkCumulative).parameters

    def test_adata_alias_still_works_with_warning(self, structured_adata, structured_lr_df):
        with pytest.warns(FutureWarning, match="adata=.*deprecated"):
            out = la.tl.prepareLRInteraction(
                adata=structured_adata, lr_df=structured_lr_df,
                use_rep_spatial="X_spatial")
        assert out.n_vars == len(structured_lr_df)

    def test_new_name_is_silent(self, structured_adata, structured_lr_df):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            la.tl.prepareLRInteraction(
                data=structured_adata, lr_df=structured_lr_df,
                use_rep_spatial="X_spatial")

    def test_both_names_is_an_error(self, structured_adata, structured_lr_df):
        with pytest.raises(TypeError, match="more than one"):
            la.tl.prepareLRInteraction(
                data=structured_adata, adata=structured_adata,
                lr_df=structured_lr_df, use_rep_spatial="X_spatial")

    def test_missing_argument_names_the_new_parameter(self, structured_lr_df):
        with pytest.raises(TypeError, match="'data'"):
            la.tl.prepareLRInteraction(lr_df=structured_lr_df)

    def test_positional_call_unchanged(self, structured_adata, structured_lr_df):
        out = la.tl.prepareLRInteraction(
            structured_adata, structured_lr_df, use_rep_spatial="X_spatial")
        assert out.n_vars == len(structured_lr_df)


# ---------------------------------------------------------------------------
# LR-gene subsetting and cell blocking
# ---------------------------------------------------------------------------

class TestSubsettingAndBlocking:
    @pytest.mark.parametrize("block_size", [3, 37, 299, 300, 5000])
    def test_block_size_does_not_change_results(
        self, structured_adata, structured_lr_df, block_size
    ):
        """Blocking must be an exact decomposition, not an approximation."""
        whole = la.tl.prepareLRInteraction(
            structured_adata, structured_lr_df, use_rep_spatial="X_spatial")
        blocked = la.tl.prepareLRInteraction(
            structured_adata, structured_lr_df, use_rep_spatial="X_spatial",
            block_size=block_size)
        assert np.array_equal(blocked.X.toarray(), whole.X.toarray())

    def test_extra_genes_do_not_change_results(self, structured_adata, structured_lr_df):
        """Genes outside the LR set must not influence the scores at all.

        This is what makes subsetting-before-diffusion safe: padding the
        object with unrelated genes changes nothing.
        """
        base = la.tl.prepareLRInteraction(
            structured_adata, structured_lr_df, use_rep_spatial="X_spatial")
        rng = np.random.default_rng(11)
        padding = sp.csr_matrix(
            rng.poisson(2.0, (structured_adata.n_obs, 50)).astype(np.float32))
        padded = ad.AnnData(
            X=sp.hstack([structured_adata.X, padding]).tocsr(),
            obs=structured_adata.obs.copy(),
            var=pd.DataFrame(index=list(structured_adata.var_names)
                             + [f"PAD{i}" for i in range(50)]),
        )
        padded.obsm["X_spatial"] = structured_adata.obsm["X_spatial"]
        out = la.tl.prepareLRInteraction(
            padded, structured_lr_df, use_rep_spatial="X_spatial")
        assert np.array_equal(out.X.toarray(), base.X.toarray())

    def test_invalid_block_size(self, structured_adata, structured_lr_df):
        with pytest.raises(ValueError, match="block_size"):
            la.tl.prepareLRInteraction(
                structured_adata, structured_lr_df,
                use_rep_spatial="X_spatial", block_size=0)


# ---------------------------------------------------------------------------
# return_type and the LR cytome
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_auto_follows_input(self, structured_adata, expression_cytome,
                                structured_lr_df):
        from_ad = la.tl.prepareLRInteraction(
            structured_adata, structured_lr_df, use_rep_spatial="X_spatial")
        from_ct = la.tl.prepareLRInteraction(
            expression_cytome, structured_lr_df, use_rep_spatial="X_spatial")
        assert isinstance(from_ad, ad.AnnData)
        assert isinstance(from_ct, str) and from_ct.endswith("sample.lr.cytome")

    def test_explicit_overrides(self, structured_adata, expression_cytome,
                                structured_lr_df, tmp_path):
        as_anndata = la.tl.prepareLRInteraction(
            expression_cytome, structured_lr_df, use_rep_spatial="X_spatial",
            return_type="anndata")
        assert isinstance(as_anndata, ad.AnnData)
        target = str(tmp_path / "explicit.lr.cytome")
        as_cytome = la.tl.prepareLRInteraction(
            structured_adata, structured_lr_df, use_rep_spatial="X_spatial",
            return_type="cytome", output=target)
        assert as_cytome == target

    def test_anndata_to_cytome_requires_output(self, structured_adata,
                                               structured_lr_df):
        with pytest.raises(ValueError, match="output"):
            la.tl.prepareLRInteraction(
                structured_adata, structured_lr_df,
                use_rep_spatial="X_spatial", return_type="cytome")

    def test_output_rejected_for_anndata_return(self, structured_adata,
                                                structured_lr_df, tmp_path):
        with pytest.raises(ValueError, match="only meaningful"):
            la.tl.prepareLRInteraction(
                structured_adata, structured_lr_df, use_rep_spatial="X_spatial",
                output=str(tmp_path / "x.lr.cytome"))

    def test_will_not_silently_overwrite(self, expression_cytome, structured_lr_df):
        first = la.tl.prepareLRInteraction(
            expression_cytome, structured_lr_df, use_rep_spatial="X_spatial")
        with pytest.raises(FileExistsError):
            la.tl.prepareLRInteraction(
                expression_cytome, structured_lr_df, use_rep_spatial="X_spatial")
        again = la.tl.prepareLRInteraction(
            expression_cytome, structured_lr_df, use_rep_spatial="X_spatial",
            overwrite=True)
        assert again == first

    def test_bad_return_type(self, structured_adata, structured_lr_df):
        with pytest.raises(ValueError, match="return_type"):
            la.tl.prepareLRInteraction(
                structured_adata, structured_lr_df,
                use_rep_spatial="X_spatial", return_type="hdf5")


class TestLRCytome:
    def test_layout_reuses_rna_modality_without_counts(
        self, expression_cytome, structured_lr_df
    ):
        path = la.tl.prepareLRInteraction(
            expression_cytome, structured_lr_df, use_rep_spatial="X_spatial")
        ds = cytome.open(path)
        try:
            # Reusing the registered RNA modality is what avoids needing a
            # new entry in cytome's MODALITY_REGISTRY; there are no counts
            # to store, so the only layer is the LR scores.
            assert ds.list_matrices() == ["RNA_lrscore"]
            pair_names = list(ds.genes.to_pandas()["gene_id"])
            assert all("::" in name for name in pair_names)
        finally:
            ds.close()

    def test_roundtrip_is_exact(self, structured_adata, expression_cytome,
                                structured_lr_df):
        reference = la.tl.prepareLRInteraction(
            structured_adata, structured_lr_df, use_rep_spatial="X_spatial")
        path = la.tl.prepareLRInteraction(
            expression_cytome, structured_lr_df, use_rep_spatial="X_spatial")
        back = la.pp.readLRCytome(path)
        assert list(back.var_names) == list(reference.var_names)
        assert np.array_equal(back.X.toarray(), reference.X.toarray())
        assert list(back.obs_names) == list(reference.obs_names)
        assert np.allclose(back.obsm["X_spatial"], reference.obsm["X_spatial"])

    def test_streamed_write_matches_single_block(self, expression_cytome,
                                                 structured_lr_df, tmp_path):
        small = la.tl.prepareLRInteraction(
            expression_cytome, structured_lr_df, use_rep_spatial="X_spatial",
            output=str(tmp_path / "small.lr.cytome"), block_size=17)
        big = la.tl.prepareLRInteraction(
            expression_cytome, structured_lr_df, use_rep_spatial="X_spatial",
            output=str(tmp_path / "big.lr.cytome"), block_size=100000)
        assert np.array_equal(la.pp.readLRCytome(small).X.toarray(),
                              la.pp.readLRCytome(big).X.toarray())

    def test_expression_cytome_rejected_as_lr_data(self, expression_cytome):
        """An LR cytome and an expression cytome must not be confused."""
        with pytest.raises(ValueError, match="does not contain an LR-score"):
            la.pp.readLRCytome(expression_cytome)

    def test_runlaris_accepts_an_lr_cytome(self, expression_cytome,
                                           structured_lr_df):
        lr_path = la.tl.prepareLRInteraction(
            expression_cytome, structured_lr_df, use_rep_spatial="X_spatial")
        _, res = la.tl.runLARIS(
            lr_path, data=expression_cytome, use_rep="X_spatial",
            use_rep_spatial="X_spatial", groupby="cell_type",
            n_cells_expressed_threshold=1, n_permutations=30, random_seed=0)
        assert len(res) > 0
        assert {"sender", "receiver", "interaction_name"} <= set(res.columns)

    def test_plot_accepts_an_lr_cytome(self, expression_cytome, structured_lr_df):
        lr_path = la.tl.prepareLRInteraction(
            expression_cytome, structured_lr_df, use_rep_spatial="X_spatial")
        interaction = str(la.pp.readLRCytome(lr_path).var_names[0])
        fig = la.pl.plotCCCSpatial(
            lr_path, "X_spatial", interaction, color_by="score",
            return_fig=True)
        assert fig is not None


# ---------------------------------------------------------------------------
# Streaming COSG
# ---------------------------------------------------------------------------

class TestStreamingCosg:
    def test_stream_matches_memory(self, structured_adata, expression_cytome,
                                   structured_lr_df):
        """Streaming COSG must agree with the in-memory path.

        Equality is to float32 precision, not bitwise: run_cosg_cytome
        returns float32 scores where cosg.cosg carries float64.
        """
        lr_adata = la.tl.prepareLRInteraction(
            structured_adata, structured_lr_df, use_rep_spatial="X_spatial")
        kwargs = dict(use_rep="X_spatial", use_rep_spatial="X_spatial",
                      groupby="cell_type", n_cells_expressed_threshold=1,
                      n_permutations=30, random_seed=0)
        np.random.seed(0)
        _, memory = la.tl.runLARIS(lr_adata.copy(), data=structured_adata,
                                   cosg_backend="memory", **kwargs)
        np.random.seed(0)
        _, streamed = la.tl.runLARIS(lr_adata.copy(), data=expression_cytome,
                                     cosg_backend="stream", **kwargs)
        merged = memory.merge(streamed, on=["sender", "receiver", "interaction_name"],
                              suffixes=("_mem", "_str"))
        assert len(merged) == len(memory)
        assert np.allclose(merged["interaction_score_mem"],
                           merged["interaction_score_str"], atol=1e-6)

    def test_layer_is_pinned_not_auto(self, expression_cytome):
        """The COSG layer must be the one that held adata.X.

        cosg's layer='auto' normalises counts on the fly, which silently
        diverges from the in-memory path; _anndata_x_layer must resolve to
        the recorded X layer instead.
        """
        from laris.tools._utils import _anndata_x_layer
        ds = cytome.open(expression_cytome)
        try:
            assert _anndata_x_layer(ds) == "counts"
        finally:
            ds.close()

    def test_stream_requires_a_cytome(self, structured_adata, structured_lr_df):
        lr_adata = la.tl.prepareLRInteraction(
            structured_adata, structured_lr_df, use_rep_spatial="X_spatial")
        with pytest.raises(ValueError, match="requires a cytome"):
            la.tl.runLARIS(lr_adata, data=structured_adata, use_rep="X_spatial",
                           use_rep_spatial="X_spatial", groupby="cell_type",
                           cosg_backend="stream")

    def test_bad_backend(self, structured_adata, structured_lr_df):
        lr_adata = la.tl.prepareLRInteraction(
            structured_adata, structured_lr_df, use_rep_spatial="X_spatial")
        with pytest.raises(ValueError, match="cosg_backend"):
            la.tl.runLARIS(lr_adata, data=structured_adata,
                           use_rep="X_spatial", cosg_backend="gpu")


# ---------------------------------------------------------------------------
# Embedding key convention
# ---------------------------------------------------------------------------

class TestEmbeddingKeys:
    def test_single_spatial_key_file_is_readable(self, tmp_path):
        """Regression: cytome drops the scanpy X_ prefix when writing.

        obsm['X_spatial'] is stored as RNA_spatial, so stripping only the
        modality prefix yielded obsm['spatial'] and LARIS's own default
        use_rep_spatial='X_spatial' raised KeyError. Both spellings must
        resolve.
        """
        rng = np.random.default_rng(2)
        adata = ad.AnnData(
            X=sp.csr_matrix(rng.poisson(1, (30, 10)).astype(np.float32)),
            obs=pd.DataFrame(index=[f"c{i}" for i in range(30)]),
            var=pd.DataFrame(index=[f"G{i}" for i in range(10)]),
        )
        adata.obsm["X_spatial"] = rng.random((30, 2))     # the only spatial key
        path = str(tmp_path / "one_key.cytome")
        ds = cytome.from_anndata(adata, output=path)
        ds.close()
        obsm = la.pp.readCytome(path, genes=["G0", "G1"]).obsm
        assert "X_spatial" in obsm and "spatial" in obsm
        assert np.allclose(obsm["X_spatial"], obsm["spatial"])

    def test_non_categorical_groupby_is_tolerated(self):
        """A cytome round-trip returns plain strings, not categoricals."""
        from laris.tools._utils import _compute_avg_expression
        rng = np.random.default_rng(4)
        adata = ad.AnnData(
            X=sp.csr_matrix(rng.random((20, 5)).astype(np.float32)),
            obs=pd.DataFrame({"ct": ["a", "b"] * 10},   # object dtype
                             index=[f"c{i}" for i in range(20)]),
            var=pd.DataFrame(index=[f"G{i}" for i in range(5)]),
        )
        out = _compute_avg_expression(adata, groupby="ct", warn=False)
        assert sorted(out.index) == ["a", "b"]
