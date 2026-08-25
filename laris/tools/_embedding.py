"""Joint embedding across samples, for the matched comparison."""

from typing import Optional, Union

import numpy as np
import pandas as pd
import anndata as ad

from .._compat import _UNSET, resolve_data_arg


def buildJointEmbedding(
    data=_UNSET,
    batch_key: Optional[str] = None,
    method: str = 'harmony',
    groupby: Optional[str] = None,
    layer: Optional[str] = None,
    normalize: bool = True,
    n_top_genes: int = 2000,
    n_comps: int = 30,
    key_added: str = 'X_joint',
    random_seed: int = 0,
    verbosity: int = 1,
    adata=_UNSET,
    **method_kwargs,
) -> ad.AnnData:
    """
    Build a joint expression embedding across samples, in ``obsm[key_added]``.

    A convenience for :func:`compareLARIS`'s matched estimator, which
    needs one embedding computed over all samples together. Three
    methods, all optional dependencies except the last:

    - ``'harmony'`` (default): HVG -> scale -> PCA -> Harmony over
      `batch_key`. Needs ``harmonypy``.
    - ``'gdr'``: PIASO's marker-gene-guided reduction
      (``piaso.tl.runGDR``), integration-free. Needs ``piaso`` and a
      `groupby` cluster column. On the Satb2 validation, harmony and GDR
      gave near-identical comparison results (Jaccard 0.66 on the hit
      set, same leaders), so this is preference, not correctness.
    - ``'pca'``: HVG -> scale -> PCA, no integration. The uncorrected
      baseline; on Satb2 it too matched the harmony hit set, but prefer a
      corrected method when batch may separate the *conditions* in
      expression space, which is the one situation the matching is
      sensitive to.

    None of these packages is a hard dependency: an informative
    ImportError tells you what to install only when you ask for a method
    that needs it.

    Parameters
    ----------
    data : AnnData
        Expression object with all samples concatenated (cells x genes).
        Counts or normalised - see `normalize`.
    batch_key : str
        ``.obs`` column identifying the sample/batch. Required.
    method : {'harmony', 'gdr', 'pca'}, default='harmony'
    groupby : str, optional
        ``'gdr'`` only: ``.obs`` cluster/cell-type column that guides the
        reduction.
    layer : str, optional
        Expression layer to use; default ``adata.X``.
    normalize : bool, default=True
        Apply ``normalize_total`` + ``log1p`` to a working copy first.
        Set False when the matrix is already normalised. Ignored by
        ``'gdr'``, which applies PIASO's INFOG normalisation itself and
        expects raw counts.
    n_top_genes : int, default=2000
        Highly-variable genes (batch-aware), for 'harmony' and 'pca'.
    n_comps : int, default=30
        Embedding dimensionality, for 'harmony' and 'pca'.
    key_added : str, default='X_joint'
        ``.obsm`` key written on `data` (modified in place; also
        returned).
    random_seed : int, default=0
    **method_kwargs
        Passed to ``harmonypy.run_harmony`` or ``piaso.tl.runGDR``.

    Returns
    -------
    AnnData
        `data`, with ``obsm[key_added]`` added.

    Notes
    -----
    ``method='gdr'`` runs PIASO's parallel workers in a *spawn*
    multiprocessing context: a driver **script** calling this must guard
    its top level with ``if __name__ == "__main__":`` or the workers
    re-execute the whole script. Notebooks are unaffected.
    """
    import scanpy as sc

    data = resolve_data_arg(data, 'buildJointEmbedding', canonical='data',
                            adata=adata)
    if not isinstance(data, ad.AnnData):
        raise TypeError(
            f"data must be an AnnData with all samples concatenated; got "
            f"{type(data).__name__}."
        )
    if batch_key is None or batch_key not in data.obs:
        raise ValueError(
            f"batch_key must name a .obs column identifying the sample; "
            f"got {batch_key!r}. Available: {list(data.obs.columns)}"
        )
    if method not in ('harmony', 'gdr', 'pca'):
        raise ValueError(
            f"method must be 'harmony', 'gdr' or 'pca', got {method!r}"
        )

    if method == 'gdr':
        try:
            import piaso
        except ImportError as exc:
            raise ImportError(
                "method='gdr' uses PIASO's runGDR.\n"
                "    pip install piaso-tools\n"
                "or choose method='harmony' or method='pca'."
            ) from exc
        if groupby is None or groupby not in data.obs:
            raise ValueError(
                "method='gdr' needs groupby= naming a .obs cluster or "
                "cell-type column to guide the reduction."
            )
        work = data.copy()
        if not isinstance(work.obs[groupby].dtype, pd.CategoricalDtype):
            work.obs[groupby] = pd.Categorical(work.obs[groupby].astype(str))
        piaso.tl.infog(work)
        piaso.tl.runGDR(work, batch_key=batch_key, groupby=groupby,
                        layer='infog', verbosity=0, **method_kwargs)
        gdr_key = 'X_gdr' if 'X_gdr' in work.obsm else sorted(
            k for k in work.obsm if 'gdr' in k.lower())[0]
        data.obsm[key_added] = np.asarray(work.obsm[gdr_key])
        return data

    work = data.copy()
    if layer is not None:
        work.X = work.layers[layer].copy()
    if normalize:
        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)
    sc.pp.highly_variable_genes(work, n_top_genes=n_top_genes,
                                batch_key=batch_key)
    work = work[:, work.var['highly_variable']].copy()
    sc.pp.scale(work, max_value=10)
    sc.tl.pca(work, n_comps=n_comps, random_state=random_seed)

    if method == 'pca':
        data.obsm[key_added] = np.asarray(work.obsm['X_pca'])
        return data

    try:
        import harmonypy
    except ImportError as exc:
        raise ImportError(
            "method='harmony' needs the harmonypy package.\n"
            "    pip install harmonypy\n"
            "or choose method='gdr' (needs piaso) or method='pca' "
            "(no extra dependency)."
        ) from exc
    ho = harmonypy.run_harmony(work.obsm['X_pca'], work.obs, [batch_key],
                               **method_kwargs)
    embedding = np.asarray(ho.Z_corr)
    if embedding.shape[0] != work.n_obs:
        embedding = embedding.T
    data.obsm[key_added] = embedding
    return data
