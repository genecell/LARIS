"""Which cytome matrix holds the expression.

cytome < 0.3.0 stored ``adata.X`` as ``{modality}_counts`` whatever it
contained. cytome 0.3.0 reserves that name for genuine raw integer counts
and writes ``{modality}_data`` otherwise -- so a hard-coded 'counts'
raises KeyError on any file converted from normalised expression, which is
what LARIS is normally handed. These tests pin the resolution on whichever
cytome is installed.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

import laris as la
from laris.preprocessing._io import _resolve_x_layer
from laris.tools._utils import _anndata_x_layer

cytome = pytest.importorskip("cytome", reason="cytome not installed")


def _write(tmp_path, X, name="s.cytome"):
    rng = np.random.default_rng(0)
    n, g = X.shape
    adata = ad.AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame({"cell_type": pd.Categorical(rng.choice(list("AB"), n))},
                         index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]),
    )
    adata.obsm["X_spatial"] = rng.random((n, 2)) * 50
    path = str(tmp_path / name)
    ds = cytome.from_anndata(adata, output=path)
    ds.close()
    return path, adata


@pytest.fixture
def normalised_cytome(tmp_path):
    """A file converted from log-normalised expression - the usual case."""
    rng = np.random.default_rng(1)
    X = np.log1p(rng.poisson(2, (70, 30)).astype(np.float32) / 3.0)
    return _write(tmp_path, X, "norm.cytome")


@pytest.fixture
def counts_cytome(tmp_path):
    rng = np.random.default_rng(2)
    X = rng.poisson(2, (70, 30)).astype(np.float32)
    return _write(tmp_path, X, "counts.cytome")


class TestResolution:
    def test_reads_normalised_files_without_an_explicit_layer(self, normalised_cytome):
        """The regression: default layer must not be a hard-coded 'counts'."""
        path, adata = normalised_cytome
        sub = la.pp.readCytome(path, genes=["G0", "G5"])
        assert np.allclose(sub.X.toarray(), adata[:, ["G0", "G5"]].X.toarray())

    def test_reads_counts_files_too(self, counts_cytome):
        path, adata = counts_cytome
        sub = la.pp.readCytome(path, genes=["G1"])
        assert np.allclose(sub.X.toarray(), adata[:, ["G1"]].X.toarray())

    def test_resolver_follows_the_recorded_x_layer(self, normalised_cytome):
        path, _ = normalised_cytome
        ds = cytome.open(path)
        try:
            recorded = ds.metadata.get("_anndata_X_layer")
            resolved = _resolve_x_layer(ds)
            assert recorded is not None, "cytome no longer records _anndata_X_layer"
            assert recorded == f"RNA_{resolved}"
            assert f"RNA_{resolved}" in ds.list_matrices()
        finally:
            ds.close()

    def test_reader_and_cosg_path_agree(self, normalised_cytome):
        """Drift here would silently score a different matrix than it reads."""
        path, _ = normalised_cytome
        ds = cytome.open(path)
        try:
            assert _anndata_x_layer(ds) == _resolve_x_layer(ds)
        finally:
            ds.close()

    def test_explicit_layer_is_obeyed(self, counts_cytome):
        path, adata = counts_cytome
        ds = cytome.open(path)
        try:
            name = _resolve_x_layer(ds)
        finally:
            ds.close()
        sub = la.pp.readCytome(path, genes=["G2"], layer=name)
        assert np.allclose(sub.X.toarray(), adata[:, ["G2"]].X.toarray())

    def test_unknown_layer_still_raises(self, counts_cytome):
        path, _ = counts_cytome
        with pytest.raises(Exception):
            la.pp.readCytome(path, genes=["G0"], layer="not_a_layer")


class TestPipelineOnNormalisedFiles:
    def test_prepare_matches_the_anndata_path(self, normalised_cytome):
        path, adata = normalised_cytome
        lr_df = pd.DataFrame({"ligand": [f"G{i}" for i in range(0, 10)],
                              "receptor": [f"G{i}" for i in range(10, 20)]})
        reference = la.tl.prepareLRInteraction(adata, lr_df,
                                               use_rep_spatial="X_spatial")
        written = la.tl.prepareLRInteraction(path, lr_df,
                                             use_rep_spatial="X_spatial")
        back = la.pp.readLRCytome(written)
        assert np.array_equal(back.X.toarray(), reference.X.toarray())

    def test_streaming_cosg_matches_memory(self, normalised_cytome):
        path, adata = normalised_cytome
        lr_df = pd.DataFrame({"ligand": [f"G{i}" for i in range(0, 10)],
                              "receptor": [f"G{i}" for i in range(10, 20)]})
        lr_adata = la.tl.prepareLRInteraction(adata, lr_df,
                                              use_rep_spatial="X_spatial")
        kwargs = dict(use_rep="X_spatial", use_rep_spatial="X_spatial",
                      groupby="cell_type", n_cells_expressed_threshold=1,
                      n_permutations=30, random_seed=0)
        np.random.seed(0)
        _, memory = la.tl.runLARIS(lr_adata.copy(), data=adata,
                                   cosg_backend="memory", **kwargs)
        np.random.seed(0)
        _, streamed = la.tl.runLARIS(lr_adata.copy(), data=path,
                                     cosg_backend="stream", **kwargs)
        merged = memory.merge(streamed, on=["sender", "receiver", "interaction_name"],
                              suffixes=("_m", "_s"))
        assert len(merged) == len(memory)
        assert np.allclose(merged["interaction_score_m"],
                           merged["interaction_score_s"], atol=1e-6)
