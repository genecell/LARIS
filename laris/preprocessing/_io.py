"""
Input adapters for LARIS.

Lets the LARIS pipeline accept a cytome dataset (a ``.cytome`` path or an
open ``cytome.CytomeDataset``) anywhere an expression AnnData is expected.

LARIS only ever needs the expression of the ligand/receptor genes plus cell
metadata and spatial coordinates, so cytome sources are converted by
streaming just that gene subset into a small in-memory AnnData
(:func:`readCytome`), and the existing AnnData code paths run unchanged.
This makes cytome and AnnData inputs produce identical results by
construction.

cytome is an optional dependency: ``pip install laris[cytome]``.
"""

import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad


def _import_cytome():
    try:
        import cytome
    except ImportError as exc:
        raise ImportError(
            "Reading a .cytome dataset requires the optional cytome "
            "dependency.\n    pip install 'laris[cytome]'   (or: pip install cytome)\n"
            "AnnData input does not need it."
        ) from exc
    return cytome


def _looks_like_cytome_dataset(obj) -> bool:
    """Pure type test for a cytome Dataset instance (no liveness probe).

    Duck-typed on the class module and name (mirrors
    ``piaso.utils._cytome_compat``), so it accepts Dataset subclasses and
    never imports cytome — the check works when cytome is not installed.
    """
    if isinstance(obj, (str, Path)):
        return False
    cls = type(obj)
    module = cls.__module__ or ""
    return "cytome" in module.lower() and "dataset" in cls.__name__.lower()


def _assert_cytome_open(obj) -> None:
    """Raise an actionable error if ``obj`` is a closed cytome Dataset.

    Surfaces the closed state at the LARIS entry point instead of a
    confusing ``sqlite3.ProgrammingError`` deep inside a matrix read.
    Closed datasets are NOT auto-reopened: ``ds.close()`` is explicit user
    intent. Same behavior as PIASO's cytome compatibility layer.
    """
    if not _looks_like_cytome_dataset(obj):
        return
    check = getattr(obj, "_check_open", None)
    if callable(check):
        check()
        return
    is_closed = getattr(obj, "is_closed", None)
    if callable(is_closed):
        is_closed = is_closed()
    if is_closed is True:
        path = getattr(obj, "path", "<unknown>")
        raise RuntimeError(
            f"cytome Dataset is closed (path: {path}). "
            f"Re-open with cytome.open(...) and retry, or pass the path "
            f"string directly to the function."
        )


def _is_cytome_source(obj) -> bool:
    """True if ``obj`` is a cytome Dataset or a path to a .cytome/.db file.

    A *closed* Dataset raises ``RuntimeError`` (actionable message) instead
    of silently returning True — matching PIASO's `is_cytome_input`.
    """
    if isinstance(obj, (str, Path)):
        s = str(obj)
        return s.endswith('.cytome') or s.endswith('.db')
    if _looks_like_cytome_dataset(obj):
        _assert_cytome_open(obj)
        return True
    return False


def _open_cytome(source):
    """Return ``(ds, opened_here)`` for a path-or-Dataset cytome source.

    An already-open Dataset is returned as-is with ``opened_here=False``
    (the caller owns its lifecycle); a path is opened here and the caller
    must close it.
    """
    if isinstance(source, (str, Path)):
        cytome = _import_cytome()
        return cytome.open(source), True
    return source, False



def _strip_embedding_prefix(key: str, modality: str = 'RNA'):
    """Short obsm name for a cytome embedding key, or None if not ours.

    Cytome 0.2.6 renamed embeddings from ``{modality}_obsm_{key}`` to
    ``{modality}_{key}``; both generations remain readable, so both forms
    must resolve. Never hard-code either literal.
    """
    for prefix in (f"{modality}_obsm_", f"{modality}_"):
        if key.startswith(prefix):
            return key[len(prefix):]
    return None


def _spatial_uns_from(source) -> dict:
    """A scanpy-style ``uns['spatial']`` dict from any source, or ``{}``.

    Cytome first (the ``spatial_images`` accessor, cytome >= 0.2.6), then
    AnnData ``uns``. A file-backed image whose decoder is missing (PIL for
    PNG/JPEG, tifffile for TIFF, imagecodecs for JPEG-2000 OME-TIFF)
    degrades to no-image with a warning rather than failing a plot.

    Adapted from ``piaso.plotting._spatial_image`` (PIASO 1.2.2); kept as a
    copy so LARIS installs without piaso-tools. Keep the two in step if the
    overlay contract changes.
    """
    if isinstance(source, (str, Path)):
        if not _is_cytome_source(source):
            return {}
        try:
            ds, opened = _open_cytome(source)
        except Exception as exc:  # unreadable path: no image, not a crash
            warnings.warn(f"could not read spatial images from {source!r}: {exc}")
            return {}
        try:
            return _spatial_uns_from(ds)
        finally:
            if opened:
                ds.close()
    acc = getattr(source, 'spatial_images', None)
    if acc is not None:
        try:
            return acc.as_uns() or {}
        except ImportError as exc:      # optional decoder missing
            warnings.warn(f"spatial image present but not decodable: {exc}")
            return {}
        except Exception as exc:        # e.g. tifffile needs imagecodecs
            warnings.warn(f"spatial image could not be read: {exc}")
            return {}
    uns = getattr(source, 'uns', None)
    if uns is not None and 'spatial' in uns and isinstance(uns['spatial'], dict):
        return uns['spatial']
    return {}


def readCytome(
    source,
    genes: Optional[List[str]] = None,
    modality: str = 'RNA',
    layer: str = 'counts',
    gene_name_column: str = 'auto',
    obs_name_column: str = 'barcode',
) -> ad.AnnData:
    """
    Read a cytome dataset (optionally a gene subset) into an AnnData.

    Parameters
    ----------
    source : str, Path or cytome.CytomeDataset
        Path to a ``.cytome`` file, or an open dataset.
    genes : list of str, optional
        Gene names to load. When given, only these genes' expression is
        streamed from disk (one pass, bounded memory) — for LARIS this is
        the union of ligand and receptor names, typically a few thousand
        genes. Genes absent from the dataset are silently skipped here;
        `prepareLRInteraction` reports them through its `unmatched`
        handling. When None, the full matrix is materialized via
        ``cytome.to_anndata``.
    modality : str, default='RNA'
        Cytome modality to read.
    layer : str, default='counts'
        Matrix layer within the modality (``{modality}_{layer}``).
        'counts' is where ``cytome.from_anndata`` stores ``adata.X``.
    gene_name_column : str, default='auto'
        Column of the cytome genes table holding the gene names that match
        the LR database. 'auto' uses ``symbol`` where present and falls back
        to ``gene_id`` (``cytome.from_anndata`` stores AnnData var_names in
        ``gene_id`` and leaves ``symbol`` empty; GTF-annotated datasets
        carry readable names in ``symbol``).
    obs_name_column : str, default='barcode'
        Column of the cytome cells table to use as ``obs_names``.

    Returns
    -------
    AnnData
        With ``.X`` (CSR), ``.obs`` from the cells table, and every
        each embedding exposed under its short obsm key (``RNA_spatial``
        or the pre-0.2.6 ``RNA_obsm_X_spatial`` both give
        ``obsm['spatial']`` / ``obsm['X_spatial']``), and, when the
        dataset stores tissue images (cytome >= 0.2.6),
        ``uns['spatial']`` in the scanpy convention.
    """
    cytome = _import_cytome()
    ds, opened_here = _open_cytome(source)
    try:
        if genes is None:
            adata = ds.to_anndata(modality=modality)
            if not sp.issparse(adata.X):
                adata.X = sp.csr_matrix(adata.X)
            return adata

        genes_df = ds.genes.to_pandas()
        if gene_name_column == 'auto':
            symbols = genes_df.get('symbol')
            gene_ids = genes_df.get('gene_id')
            if symbols is None and gene_ids is None:
                raise KeyError(
                    f"Neither 'symbol' nor 'gene_id' found in the cytome "
                    f"genes table (available: {list(genes_df.columns)}). "
                    f"Pass gene_name_column= explicitly."
                )
            if symbols is None:
                symbols = gene_ids.astype(str)
            elif gene_ids is not None:
                symbols = symbols.where(symbols.notna(), gene_ids).astype(str)
            else:
                symbols = symbols.astype(str)
        else:
            if gene_name_column not in genes_df.columns:
                raise KeyError(
                    f"Column {gene_name_column!r} not found in the cytome genes "
                    f"table (available: {list(genes_df.columns)}). Pass "
                    f"gene_name_column= explicitly."
                )
            symbols = genes_df[gene_name_column].astype(str)
        requested = pd.Index(pd.unique(pd.Series(genes, dtype=str)))
        # Keep the dataset's first occurrence per symbol; preserve requested
        # gene order for a deterministic var_names layout.
        symbol_to_idx = {}
        for idx, symbol in zip(genes_df.index, symbols):
            if symbol not in symbol_to_idx:
                symbol_to_idx[symbol] = idx
        found = [g for g in requested if g in symbol_to_idx]
        if not found:
            raise ValueError(
                f"None of the {len(requested)} requested genes were found in "
                f"the cytome genes table column {gene_name_column!r}."
            )
        feat_indices = [symbol_to_idx[g] for g in found]

        arr = cytome.read_feature_columns(ds, modality, layer, feat_indices)
        X = sp.csr_matrix(arr)

        cells_df = ds.cells.to_pandas()
        obs = cells_df.copy()
        if obs_name_column in obs.columns:
            obs.index = obs[obs_name_column].astype(str)
        obs.index.name = None

        adata = ad.AnnData(
            X=X,
            obs=obs,
            var=pd.DataFrame(index=pd.Index(found, name=None)),
        )

        # Embedding names differ across cytome generations: files written
        # before 0.2.6 store obsm arrays as "{modality}_obsm_{key}", 0.2.6+
        # as "{modality}_{key}". Accept BOTH - a hard-coded prefix works on
        # one generation and KeyErrors on the other.
        for key in ds.embeddings.keys():
            short = _strip_embedding_prefix(key, modality)
            if short is not None:
                adata.obsm[short] = np.asarray(ds.embeddings[key])

        # Tissue images (cytome >= 0.2.6) travel with the data: expose them
        # in the scanpy convention so every downstream consumer - including
        # laris.pl.plotCCCSpatial - needs no cytome-specific branch.
        spatial_uns = _spatial_uns_from(ds)
        if spatial_uns:
            adata.uns['spatial'] = spatial_uns
        return adata
    finally:
        if opened_here:
            ds.close()


def _ensure_expression_anndata(
    source,
    genes: Optional[List[str]] = None,
    modality: str = 'RNA',
    layer: str = 'counts',
) -> ad.AnnData:
    """Pass AnnData through; convert cytome sources via :func:`readCytome`."""
    if isinstance(source, ad.AnnData):
        return source
    if _is_cytome_source(source):
        return readCytome(source, genes=genes, modality=modality, layer=layer)
    raise TypeError(
        f"Expected an AnnData, a .cytome/.db path, or an open cytome "
        f"Dataset; got {type(source).__name__}."
    )
