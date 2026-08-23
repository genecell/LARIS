"""Issue #4: LARIS supports 3D spatial coordinates.

The core is built on kneighbors_graph over the full obsm matrix, so it is
dimension-agnostic; these tests pin that property so a 2D assumption can
never silently creep into the pipeline. (The image-overlay plot projects
to the first two axes by design.)
"""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

import laris as la


@pytest.fixture
def adata_3d(rng):
    n_cells, n_genes = 200, 300
    X = rng.random((n_cells, n_genes), dtype=np.float32)
    X[X < 0.7] = 0
    adata = ad.AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame(
            {"cell_type": pd.Categorical(rng.choice(["A", "B", "C"], n_cells))},
            index=[f"c{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"Gene{i}" for i in range(n_genes)]),
    )
    adata.obsm["X_spatial"] = rng.random((n_cells, 3)) * 100  # 3D
    return adata


def test_full_pipeline_runs_on_3d(adata_3d, lr_df):
    lr_adata = la.tl.prepareLRInteraction(
        adata_3d, lr_df, use_rep_spatial="X_spatial"
    )
    assert lr_adata.shape == (200, len(lr_df))
    _, res = la.tl.runLARIS(
        lr_adata, adata_3d,
        use_rep="X_spatial", use_rep_spatial="X_spatial",
        groupby="cell_type", n_cells_expressed_threshold=1,
        n_permutations=50,
    )
    assert len(res) > 0
    assert not res["interaction_score"].isna().any()


def test_third_dimension_is_not_ignored(adata_3d, lr_df):
    """Results must differ from the 2D projection: z is really used."""
    a3 = adata_3d
    a2 = adata_3d.copy()
    a2.obsm["X_spatial"] = a2.obsm["X_spatial"][:, :2].copy()
    r3 = la.tl.prepareLRInteraction(a3, lr_df, use_rep_spatial="X_spatial")
    r2 = la.tl.prepareLRInteraction(a2, lr_df, use_rep_spatial="X_spatial")
    assert not np.allclose(r3.X.toarray(), r2.X.toarray())
