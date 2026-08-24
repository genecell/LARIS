"""specificity_reference: which genes set the scale for cell type specificity.

``cosg.iqrLogNormalize`` divides each cell-type column by that column's
q0.95-q0.75 spread over the rows it is handed, so the reference gene set
changes the result. The two choices buy mutually exclusive invariances,
and these tests pin both of them:

- 'lr'  is exactly invariant to which *other* genes are in the object
- 'all' is exactly invariant to which LR database you look up

They also pin the structural fact that makes the choice a pure rescaling:
within a cell type, the gene ranking is identical either way.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

import laris as la
from laris.tools import _utils


@pytest.fixture
def marker_adata():
    """Expression with real per-cell-type markers, plus non-marker filler."""
    rng = np.random.default_rng(0)
    n_cells, n_genes = 400, 400
    cell_types = rng.choice(list("ABC"), n_cells)
    X = rng.poisson(0.4, (n_cells, n_genes)).astype(np.float32)
    for k, c in enumerate("ABC"):
        rows = cell_types == c
        cols = np.arange(k * 20, (k + 1) * 20)
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
def laris_lr_frame():
    """A 'laris_lr'-shaped frame spanning 140 distinct genes.

    Deliberately above the 100-gene stability threshold so that the normal
    path does not trip the small-panel warning.
    """
    return pd.DataFrame({"ligand": [f"G{i}" for i in range(70)],
                         "receptor": [f"G{i}" for i in range(70, 140)]})


def specificity(adata, laris_lr, reference):
    return _utils._calculate_ligand_receptor_specificity(
        adata, laris_lr, groupby="cell_type", mu=100, expressed_pct=0.1,
        remove_lowly_expressed=False, specificity_reference=reference)


class TestReferenceSemantics:
    def test_gene_ranking_within_a_cell_type_is_unchanged(self, marker_adata,
                                                          laris_lr_frame):
        """The normalisation is monotone per column, so only scale changes."""
        a = specificity(marker_adata, laris_lr_frame, "lr")
        b = specificity(marker_adata, laris_lr_frame, "all").reindex(index=a.index)
        for column in a.columns:
            rho = a[column].rank().corr(b[column].rank(), method="spearman")
            assert rho == pytest.approx(1.0, abs=1e-12)

    def test_the_two_references_actually_differ(self, marker_adata, laris_lr_frame):
        a = specificity(marker_adata, laris_lr_frame, "lr")
        b = specificity(marker_adata, laris_lr_frame, "all").reindex(index=a.index)
        assert not np.allclose(a.values.astype(float), b.values.astype(float))

    def test_same_genes_and_shape(self, marker_adata, laris_lr_frame):
        a = specificity(marker_adata, laris_lr_frame, "lr")
        b = specificity(marker_adata, laris_lr_frame, "all")
        assert sorted(a.index) == sorted(b.index)
        assert sorted(a.columns) == sorted(b.columns)


class TestInvariances:
    """The R16.3 table, as executable assertions."""

    def test_lr_reference_ignores_other_genes(self, marker_adata, laris_lr_frame):
        """Padding the object with unrelated genes must change nothing."""
        base = specificity(marker_adata, laris_lr_frame, "lr")
        rng = np.random.default_rng(9)
        padding = sp.csr_matrix(
            rng.poisson(3.0, (marker_adata.n_obs, 120)).astype(np.float32))
        padded = ad.AnnData(
            X=sp.hstack([marker_adata.X, padding]).tocsr(),
            obs=marker_adata.obs.copy(),
            var=pd.DataFrame(index=list(marker_adata.var_names)
                             + [f"PAD{i}" for i in range(120)]),
        )
        out = specificity(padded, laris_lr_frame, "lr").reindex(index=base.index)
        assert np.allclose(base.values.astype(float), out.values.astype(float),
                           equal_nan=True)

    def test_all_reference_does_not_ignore_other_genes(self, marker_adata,
                                                       laris_lr_frame):
        """The cost of the 'all' reference, pinned so it cannot regress silently."""
        base = specificity(marker_adata, laris_lr_frame, "all")
        rng = np.random.default_rng(9)
        padding = sp.csr_matrix(
            rng.poisson(3.0, (marker_adata.n_obs, 120)).astype(np.float32))
        padded = ad.AnnData(
            X=sp.hstack([marker_adata.X, padding]).tocsr(),
            obs=marker_adata.obs.copy(),
            var=pd.DataFrame(index=list(marker_adata.var_names)
                             + [f"PAD{i}" for i in range(120)]),
        )
        out = specificity(padded, laris_lr_frame, "all").reindex(index=base.index)
        assert not np.allclose(base.values.astype(float), out.values.astype(float),
                               equal_nan=True)

    def test_all_reference_ignores_the_database(self, marker_adata, laris_lr_frame):
        """Halving the LR database must not move the shared genes' scores."""
        base = specificity(marker_adata, laris_lr_frame, "all")
        half = laris_lr_frame.iloc[: len(laris_lr_frame) // 2].reset_index(drop=True)
        out = specificity(marker_adata, half, "all")
        shared = [g for g in out.index if g in base.index]
        assert len(shared) > 5
        assert np.allclose(base.reindex(index=shared).values.astype(float),
                           out.reindex(index=shared).values.astype(float),
                           equal_nan=True)

    def test_lr_reference_does_not_ignore_the_database(self, marker_adata,
                                                       laris_lr_frame):
        base = specificity(marker_adata, laris_lr_frame, "lr")
        half = laris_lr_frame.iloc[: len(laris_lr_frame) // 2].reset_index(drop=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = specificity(marker_adata, half, "lr")
        shared = [g for g in out.index if g in base.index]
        assert not np.allclose(base.reindex(index=shared).values.astype(float),
                               out.reindex(index=shared).values.astype(float),
                               equal_nan=True)


class TestSmallPanelWarning:
    def test_warns_below_the_threshold(self, marker_adata):
        """Targeted panels match few database genes; the LR spread is unstable."""
        tiny = pd.DataFrame({"ligand": ["G0", "G1"], "receptor": ["G2", "G3"]})
        with pytest.warns(UserWarning, match="too few for a stable estimate"):
            specificity(marker_adata, tiny, "lr")

    def test_all_reference_does_not_warn(self, marker_adata):
        tiny = pd.DataFrame({"ligand": ["G0", "G1"], "receptor": ["G2", "G3"]})
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            specificity(marker_adata, tiny, "all")

    def test_no_warning_for_a_normal_database(self, marker_adata, laris_lr_frame):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            specificity(marker_adata, laris_lr_frame, "lr")


class TestPlumbing:
    def test_default_is_lr(self):
        import inspect
        assert (inspect.signature(la.tl.runLARIS)
                .parameters["specificity_reference"].default == "lr")

    def test_invalid_value_rejected(self, marker_adata):
        lr_df = pd.DataFrame({"ligand": ["G0", "G20"], "receptor": ["G21", "G40"]})
        lr_adata = la.tl.prepareLRInteraction(marker_adata, lr_df,
                                              use_rep_spatial="X_spatial")
        with pytest.raises(ValueError, match="specificity_reference"):
            la.tl.runLARIS(lr_adata, data=marker_adata, use_rep="X_spatial",
                           specificity_reference="transcriptome")

    def test_reaches_the_results(self, marker_adata, laris_lr_frame):
        """The choice must change interaction scores, not be silently dropped."""
        lr_df = laris_lr_frame.drop_duplicates(
            subset=["ligand", "receptor"]).reset_index(drop=True)
        lr_adata = la.tl.prepareLRInteraction(marker_adata, lr_df,
                                              use_rep_spatial="X_spatial")
        kwargs = dict(use_rep="X_spatial", use_rep_spatial="X_spatial",
                      groupby="cell_type", n_cells_expressed_threshold=1,
                      n_permutations=30, random_seed=0)
        np.random.seed(0)
        _, res_lr = la.tl.runLARIS(lr_adata.copy(), data=marker_adata,
                                   specificity_reference="lr", **kwargs)
        np.random.seed(0)
        _, res_all = la.tl.runLARIS(lr_adata.copy(), data=marker_adata,
                                    specificity_reference="all", **kwargs)
        merged = res_lr.merge(res_all, on=["sender", "receiver", "interaction_name"],
                              suffixes=("_lr", "_all"))
        assert len(merged) == len(res_lr)
        assert not np.allclose(merged["interaction_score_lr"],
                               merged["interaction_score_all"])


class TestStreamingAgrees:
    """Both references must behave identically on the streaming path."""

    @pytest.mark.parametrize("reference", ["lr", "all"])
    def test_stream_matches_memory(self, tmp_path, marker_adata, laris_lr_frame,
                                   reference):
        cytome = pytest.importorskip("cytome", reason="cytome not installed")
        path = str(tmp_path / "e.cytome")
        ds = cytome.from_anndata(marker_adata, output=path)
        ds.close()
        lr_df = laris_lr_frame.drop_duplicates(
            subset=["ligand", "receptor"]).reset_index(drop=True)
        lr_adata = la.tl.prepareLRInteraction(marker_adata, lr_df,
                                              use_rep_spatial="X_spatial")
        kwargs = dict(use_rep="X_spatial", use_rep_spatial="X_spatial",
                      groupby="cell_type", n_cells_expressed_threshold=1,
                      n_permutations=30, random_seed=0,
                      specificity_reference=reference)
        np.random.seed(0)
        _, memory = la.tl.runLARIS(lr_adata.copy(), data=marker_adata,
                                   cosg_backend="memory", **kwargs)
        np.random.seed(0)
        _, streamed = la.tl.runLARIS(lr_adata.copy(), data=path,
                                     cosg_backend="stream", **kwargs)
        merged = memory.merge(streamed, on=["sender", "receiver", "interaction_name"],
                              suffixes=("_m", "_s"))
        assert len(merged) == len(memory)
        assert np.allclose(merged["interaction_score_m"],
                           merged["interaction_score_s"], atol=1e-6)
