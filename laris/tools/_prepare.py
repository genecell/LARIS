"""Ligand-receptor interaction score preparation (prepareLRInteraction)."""

import warnings
from pathlib import Path
from typing import Optional, Union, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize

from . import _utils
from .._compat import _UNSET, resolve_data_arg
from ..preprocessing._io import (
    _ensure_expression_anndata,
    _is_cytome_source,
    _LRCytomeWriter,
    _open_cytome,
)

def _default_lr_output(source) -> str:
    """Derive an LR-cytome path from an expression cytome source.

    ``sample.cytome`` -> ``sample.lr.cytome``, alongside the input.
    """
    if not isinstance(source, (str, Path)):
        path = getattr(source, 'path', None)
        if path is None:
            raise ValueError(
                "return_type='cytome' needs an output path: the input is not "
                "a file, so there is nowhere obvious to write. Pass "
                "output='mysample.lr.cytome'."
            )
        source = path
    text = str(source)
    for suffix in ('.cytome', '.db'):
        if text.endswith(suffix):
            return f"{text[:-len(suffix)]}.lr{suffix}"
    return f"{text}.lr.cytome"


def prepareLRInteraction(
    data=_UNSET,
    lr_df: Optional[pd.DataFrame] = None,
    number_nearest_neighbors: int = 20,
    use_rep_spatial: str = 'X_spatial',
    unmatched: str = 'drop',
    sigma: Union[float, str] = 'adaptive',
    return_type: str = 'auto',
    output: Optional[str] = None,
    overwrite: bool = False,
    block_size: Optional[int] = None,
    adata=_UNSET,
):
    """
    Calculate ligand-receptor interaction scores using spatial neighborhood information.

    This function computes diffused ligand-receptor interaction scores by considering
    the spatial context of each cell. It uses k-nearest neighbors to create a spatial
    neighborhood graph and calculates element-wise multiplication of diffused ligand
    and receptor expression levels.

    Parameters
    ----------
    data : AnnData, str, Path or cytome.CytomeDataset
        Gene expression with spatial coordinates in ``.obsm[use_rep_spatial]``:
        an AnnData, a path to a ``.cytome``/``.db`` file, or an open cytome
        dataset. Only the ligand/receptor genes are ever read.
    lr_df : pd.DataFrame
        DataFrame containing ligand-receptor pairs with columns 'ligand' and 'receptor'.
    number_nearest_neighbors : int, default=20
        Number of nearest neighbors to consider for spatial diffusion.
        Matches the tutorial and manuscript settings, and the two
        neighbourhood sizes in :func:`runLARIS`. Pass 10 for pre-v0.10.0
        behaviour.
    use_rep_spatial : str, default='X_spatial'
        Key in ``.obsm`` containing spatial coordinates.
    unmatched : {'drop', 'error'}, default='drop'
        How to handle ligand or receptor names in `lr_df` that are absent from
        the data's ``var_names``. With 'drop', unmatched pairs are removed and a
        UserWarning summarises how many were dropped and lists example missing
        names; this is convenient for targeted panels (e.g. Xenium, MERFISH)
        where most database genes are legitimately absent. With 'error', a
        ValueError is raised instead, which is safer when supplying a custom
        database where a missing name likely indicates a typo.
    sigma : float or 'adaptive', default='adaptive'
        Bandwidth of the exponential kernel converting spatial k-NN
        distances into diffusion weights (``exp(-d / sigma)``). 'adaptive'
        uses half the mean k-NN edge distance, making the result independent
        of coordinate units (pixels vs micrometres) and platform spot
        spacing. Pass a number for an absolute bandwidth in coordinate
        units.
    return_type : {'auto', 'anndata', 'cytome'}, default='auto'
        Format of the returned LR-score object. 'auto' follows the input:
        an AnnData in gives an AnnData back, a cytome in gives an LR cytome
        back. This keeps large analyses on disk - materialising the scores
        for a 500k-cell dataset in memory is exactly what a cytome input is
        trying to avoid - while small AnnData workflows are unchanged.
    output : str, optional
        Where to write the LR cytome. Defaults to the input path with a
        ``.lr`` infix (``sample.cytome`` -> ``sample.lr.cytome``). Required
        when writing a cytome from an in-memory AnnData.
    overwrite : bool, default=False
        Allow replacing an existing ``output`` file.
    block_size : int, optional
        Number of cells to diffuse and score at a time. Bounds peak memory
        to roughly ``block_size x n_lr_pairs`` instead of
        ``n_cells x n_lr_pairs``. Defaults to a single pass for AnnData
        output and 10,000-cell blocks when writing a cytome. Results are
        identical for any block size.
    adata : deprecated
        Former name of ``data``.

    Returns
    -------
    AnnData or str
        With ``return_type='anndata'``, a new AnnData object:

        - `.X`: Sparse matrix of LR interaction scores (cells x LR pairs)
        - `.var_names`: Ligand-receptor pair names in format "ligand::receptor"
        - `.obs`: Cell metadata copied from the input
        - `.obsm`: Spatial and other representations copied from the input

        With ``return_type='cytome'``, the path to the written LR cytome:
        a dataset whose features are LR pairs, carrying a single
        ``RNA_lrscore`` layer (and no counts, because there are none).
        Read it back with :func:`laris.pp.readLRCytome`, or pass it
        straight to :func:`laris.tl.runLARIS` as ``lr_data``.

    Examples
    --------
    >>> import laris as la
    >>> import pandas as pd
    >>>
    >>> # Define ligand-receptor pairs
    >>> lr_df = pd.DataFrame({
    ...     'ligand': ['Tgfb1', 'Vegfa'],
    ...     'receptor': ['Tgfbr1', 'Kdr']
    ... })
    >>>
    >>> # Calculate LR integration scores
    >>> lr_data = la.tl.prepareLRInteraction(adata, lr_df)
    >>> print(lr_data.shape)  # (n_cells, n_lr_pairs)
    >>>
    >>> # Straight from a cytome: scores stay on disk
    >>> path = la.tl.prepareLRInteraction("sample.cytome", lr_df)

    """
    data = resolve_data_arg(data, 'prepareLRInteraction', canonical='data',
                            adata=adata)
    if lr_df is None:
        raise TypeError(
            "prepareLRInteraction() missing required argument: 'lr_df'"
        )
    if return_type not in ('auto', 'anndata', 'cytome'):
        raise ValueError(
            f"return_type must be 'auto', 'anndata' or 'cytome', got "
            f"{return_type!r}"
        )

    source_is_cytome = _is_cytome_source(data)
    if return_type == 'auto':
        return_type = 'cytome' if source_is_cytome else 'anndata'
    if return_type == 'cytome':
        if output is None:
            output = _default_lr_output(data)
    elif output is not None:
        raise ValueError(
            "output= is only meaningful with return_type='cytome'."
        )

    if block_size is None:
        block_size = 10000 if return_type == 'cytome' else None
    elif block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    # Accept a cytome source: stream just the ligand/receptor gene subset
    # into an in-memory AnnData, then run the standard path unchanged.
    if not isinstance(data, ad.AnnData):
        lr_genes = list(pd.unique(pd.concat([
            lr_df['ligand'].astype(str), lr_df['receptor'].astype(str)
        ])))
        adata = _ensure_expression_anndata(data, genes=lr_genes)
    else:
        adata = data

    # Deduplicate lr_df to prevent duplicate var_names in the LR object, which
    # causes identical background sets and duplicate p-values downstream.
    n_before = len(lr_df)
    lr_df = lr_df.drop_duplicates(subset=['ligand', 'receptor']).reset_index(drop=True)
    n_dropped = n_before - len(lr_df)
    if n_dropped > 0:
        warnings.warn(
            f"prepareLRInteraction: removed {n_dropped} duplicate ligand-receptor "
            f"pair(s) from lr_df. Duplicate pairs cause identical p-values in "
            f"runLARIS. Ensure each (ligand, receptor) pair is unique.",
            UserWarning,
            stacklevel=2,
        )

    # Validate ligand/receptor names against var_names BEFORE indexing.
    # np.searchsorted below returns insertion positions, so an unmatched name
    # would otherwise silently resolve to a neighbouring gene with no error.
    if unmatched not in ('drop', 'error'):
        raise ValueError(
            f"unmatched must be 'drop' or 'error', got {unmatched!r}"
        )
    var_names_set = set(adata.var_names)
    ligand_present = lr_df['ligand'].isin(var_names_set)
    receptor_present = lr_df['receptor'].isin(var_names_set)
    matched_mask = ligand_present & receptor_present
    if not matched_mask.all():
        missing_names = sorted(
            set(lr_df.loc[~ligand_present, 'ligand'])
            | set(lr_df.loc[~receptor_present, 'receptor'])
        )
        n_unmatched_pairs = int((~matched_mask).sum())
        example_names = ', '.join(missing_names[:10])
        if len(missing_names) > 10:
            example_names += ', ...'
        message = (
            f"prepareLRInteraction: {n_unmatched_pairs} ligand-receptor pair(s) "
            f"reference {len(missing_names)} gene name(s) absent from "
            f"var_names (e.g. {example_names})."
        )
        if unmatched == 'error':
            raise ValueError(
                message + " Filter lr_df with .isin(adata.var_names) or pass "
                "unmatched='drop' to remove these pairs automatically."
            )
        warnings.warn(
            message + " These pairs were dropped. Pass unmatched='error' to "
            "raise instead (recommended when using a custom database, where a "
            "missing name may be a typo).",
            UserWarning,
            stacklevel=2,
        )
        lr_df = lr_df.loc[matched_mask].reset_index(drop=True)
    if len(lr_df) == 0:
        raise ValueError(
            "prepareLRInteraction: no ligand-receptor pairs remain after "
            "matching against var_names."
        )

    X_spatial = adata.obsm[use_rep_spatial].copy()

    # Ensure the expression matrix is sparse so downstream .multiply() /
    # .maximum() work. A dense array is converted to CSR.
    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)

    # Only the ligand/receptor genes take part in the diffusion, so subset
    # BEFORE diffusing. Diffusing the whole transcriptome and then keeping
    # ~4% of the rows is the single largest cost in this function
    # (measured on a 4k-cell Visium sample: 1,261 MB / 3.2 s for all 36,601
    # genes versus 410 MB / 0.6 s for the 1,306 LR genes, identical output).
    var_names = np.asarray(adata.var_names)
    sorter = np.argsort(var_names)
    lr_gene_names = pd.unique(np.concatenate([
        lr_df['ligand'].to_numpy(dtype=object),
        lr_df['receptor'].to_numpy(dtype=object),
    ]))
    gene_positions = sorter[
        np.searchsorted(var_names, lr_gene_names, sorter=sorter)
    ]
    X_lr = sp.csr_matrix(X[:, gene_positions])

    position_of = {gene: i for i, gene in enumerate(lr_gene_names)}
    ligand_idx = np.fromiter(
        (position_of[g] for g in lr_df['ligand']), dtype=int, count=len(lr_df))
    receptor_idx = np.fromiter(
        (position_of[g] for g in lr_df['receptor']), dtype=int, count=len(lr_df))

    # Spatial diffusion graph
    cellxcell = kneighbors_graph(
        X_spatial,
        n_neighbors=number_nearest_neighbors,
        mode='distance',
        include_self=True
    )
    cellxcell = _utils._apply_knn_kernel(cellxcell, sigma=sigma)
    cellxcell = sp.csr_matrix(cellxcell)

    # Keep the CSC orientation the un-subsetted implementation used: the
    # sparse product accumulates in storage order, so switching format
    # would perturb results in the last couple of ULPs for no reason.
    genexcell = X_lr.T
    n_obs = adata.n_obs
    lr_names = (lr_df['ligand'].astype(str) + '::' + lr_df['receptor'].astype(str))

    # A block of cells is diffused, scored and masked in one go. Column j of
    # ``order1`` only depends on row j of the k-NN graph, so
    # ``order1[:, block] == genexcell @ cellxcell[block].T`` exactly - the
    # block size changes peak memory, never the result.
    step = block_size or n_obs
    writer = None
    blocks = []
    spatial_uns = adata.uns['spatial'] if 'spatial' in getattr(adata, 'uns', {}) else None
    if return_type == 'cytome':
        writer = _LRCytomeWriter(
            output, lr_names=list(lr_names), obs=adata.obs,
            obs_names=adata.obs_names, obsm=dict(adata.obsm),
            spatial_uns=spatial_uns, overwrite=overwrite,
        )
    try:
        for start in range(0, n_obs, step):
            stop = min(start + step, n_obs)
            order1 = genexcell @ cellxcell[start:stop, :].T

            # Element-wise multiplication of diffused ligand and receptor
            lrxcell = order1[ligand_idx, :].multiply(order1[receptor_idx, :])
            block = sp.csr_matrix(lrxcell.T)

            # Keep only cells where the ligand or the receptor is expressed
            X_block = X_lr[start:stop]
            ligand_mask = X_block[:, ligand_idx] != 0
            receptor_mask = X_block[:, receptor_idx] != 0
            non_zero_mask = ligand_mask.maximum(receptor_mask)
            block = sp.csr_matrix(block.multiply(non_zero_mask))

            if writer is not None:
                writer.write_block(block, start)
            else:
                blocks.append(block)
    except Exception:
        if writer is not None:
            writer.ds.close()
        raise

    if writer is not None:
        return writer.close()

    lr_adata = sc.AnnData(sp.vstack(blocks).tocsr() if blocks else sp.csr_matrix((0, len(lr_names))))
    lr_adata.obs = adata.obs.copy()
    lr_adata.obsm = adata.obsm.copy()
    lr_adata.var_names = lr_names

    # Carry tissue images along so plotCCCSpatial can overlay from the LR
    # object alone (no data= argument needed). Cheap: the same array objects.
    if spatial_uns is not None:
        lr_adata.uns['spatial'] = spatial_uns

    return lr_adata
