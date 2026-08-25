"""buildJointEmbedding: the one-liner joint embedding for the matched estimator."""

import warnings

import numpy as np
import pandas as pd
import pytest
import anndata as ad
import scipy.sparse as sp

import laris as la


@pytest.fixture
def multi_sample_adata():
    rng = np.random.default_rng(0)
    frames, Xs = [], []
    for s in range(4):
        n = 150
        X = rng.poisson(1.5, (n, 120)).astype(np.float32)
        frames.append(pd.DataFrame(
            {"sample": f"s{s}",
             "ct": pd.Categorical(rng.choice(list("AB"), n))},
            index=[f"s{s}_c{i}" for i in range(n)]))
        Xs.append(sp.csr_matrix(X))
    return ad.AnnData(X=sp.vstack(Xs), obs=pd.concat(frames),
                      var=pd.DataFrame(index=[f"G{i}" for i in range(120)]))


class TestMethods:
    def test_pca_needs_no_optional_dependency(self, multi_sample_adata):
        out = la.tl.buildJointEmbedding(multi_sample_adata, batch_key="sample",
                                        method="pca", n_comps=10,
                                        n_top_genes=50)
        assert out is multi_sample_adata
        assert out.obsm["X_joint"].shape == (multi_sample_adata.n_obs, 10)
        assert np.isfinite(out.obsm["X_joint"]).all()

    def test_harmony(self, multi_sample_adata):
        pytest.importorskip("harmonypy", reason="harmonypy not installed")
        out = la.tl.buildJointEmbedding(multi_sample_adata, batch_key="sample",
                                        method="harmony", n_comps=10,
                                        n_top_genes=50)
        assert out.obsm["X_joint"].shape == (multi_sample_adata.n_obs, 10)

    def test_feeds_the_matched_estimator(self, multi_sample_adata):
        """The advertised workflow must actually connect end to end."""
        la.tl.buildJointEmbedding(multi_sample_adata, batch_key="sample",
                                  method="pca", n_comps=10, n_top_genes=50)
        rng = np.random.default_rng(1)
        lr = ad.AnnData(
            X=sp.csr_matrix(rng.random((multi_sample_adata.n_obs, 20))
                            * (rng.random((multi_sample_adata.n_obs, 20)) > .8)),
            obs=multi_sample_adata.obs.copy(),
            var=pd.DataFrame(index=[f"L{i}::R{i}" for i in range(20)]))
        lr.obs["condition"] = np.where(
            lr.obs["sample"].isin(["s0", "s1"]), "ref", "alt")
        lr.obsm["X_joint"] = multi_sample_adata.obsm["X_joint"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cmp_, _ = la.tl.compareLARIS(
                lr, conditionKey="condition", referenceCondition="ref",
                sampleKey="sample", use_rep="X_joint", n_anchors=20)
        assert len(cmp_) == 20

    def test_key_added(self, multi_sample_adata):
        la.tl.buildJointEmbedding(multi_sample_adata, batch_key="sample",
                                  method="pca", key_added="X_mine",
                                  n_comps=5, n_top_genes=50)
        assert "X_mine" in multi_sample_adata.obsm


class TestGuards:
    def test_requires_anndata(self):
        with pytest.raises(TypeError, match="AnnData"):
            la.tl.buildJointEmbedding({"x": 1}, batch_key="sample")

    def test_batch_key_required_and_checked(self, multi_sample_adata):
        with pytest.raises(ValueError, match="batch_key"):
            la.tl.buildJointEmbedding(multi_sample_adata)
        with pytest.raises(ValueError, match="Available"):
            la.tl.buildJointEmbedding(multi_sample_adata, batch_key="slide")

    def test_unknown_method(self, multi_sample_adata):
        with pytest.raises(ValueError, match="method"):
            la.tl.buildJointEmbedding(multi_sample_adata, batch_key="sample",
                                      method="umap")

    def test_gdr_requires_groupby(self, multi_sample_adata):
        pytest.importorskip("piaso", reason="piaso not installed")
        with pytest.raises(ValueError, match="groupby"):
            la.tl.buildJointEmbedding(multi_sample_adata, batch_key="sample",
                                      method="gdr")

    def test_adata_alias(self, multi_sample_adata):
        with pytest.warns(FutureWarning, match="adata=.*deprecated"):
            la.tl.buildJointEmbedding(adata=multi_sample_adata,
                                      batch_key="sample", method="pca",
                                      n_comps=5, n_top_genes=50)
