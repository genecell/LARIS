"""Shared fixtures for LARIS test suite."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import pytest


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_adata(rng):
    """Synthetic AnnData with 200 cells, 300 genes, 3 cell types, spatial coords.

    Expression matrix is sparse (~30% non-zero).
    """
    n_cells = 200
    n_genes = 300

    X = rng.random((n_cells, n_genes), dtype=np.float32)
    X[X < 0.7] = 0
    X = sp.csr_matrix(X)

    gene_names = [f"Gene{i}" for i in range(n_genes)]
    cell_names = [f"Cell{i}" for i in range(n_cells)]
    spatial = rng.random((n_cells, 2)) * 100

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"cell_type": pd.Categorical(rng.choice(["A", "B", "C"], n_cells))},
            index=cell_names,
        ),
        var=pd.DataFrame(index=gene_names),
    )
    adata.obsm["spatial"] = spatial
    adata.obsm["X_spatial"] = spatial
    return adata


@pytest.fixture
def synthetic_adata_dense(synthetic_adata):
    """Same as synthetic_adata but with dense X (for issue #2 testing)."""
    adata = synthetic_adata.copy()
    adata.X = adata.X.toarray()
    return adata


@pytest.fixture
def lr_df():
    """100 non-overlapping LR pairs drawn from Gene0..Gene199."""
    return pd.DataFrame({
        "ligand": [f"Gene{i}" for i in range(0, 200, 2)],
        "receptor": [f"Gene{i}" for i in range(1, 201, 2)],
    })


@pytest.fixture
def lr_df_with_duplicates(lr_df):
    """LR database with intentional duplicate rows (for dedup testing)."""
    dups = lr_df.iloc[:5].copy()
    return pd.concat([lr_df, dups], ignore_index=True)


@pytest.fixture
def lr_adata(synthetic_adata, lr_df):
    """Pre-computed lr_adata from prepareLRInteraction."""
    import laris as la
    return la.tl.prepareLRInteraction(
        synthetic_adata, lr_df,
        number_nearest_neighbors=10,
        use_rep_spatial="spatial",
    )
