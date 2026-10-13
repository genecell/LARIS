# `laris.pp`: Reading data

<!-- Generated from the docstrings by tools/make_api_docs.py; do not edit by hand. -->

Cytome readers and multi-sample coordinate offsets.

| Function | Summary |
|---|---|
| [`readCytome`](#readcytome) | Read a cytome dataset (optionally a gene subset) into an AnnData. |
| [`readLRCytome`](#readlrcytome) | Read an LR-score cytome (written by `prepareLRInteraction`). |
| [`spatialOffsetMultisample`](#spatialoffsetmultisample) | Offset spatial coordinates so that kNN stays within each sample. |

---

<a id="readcytome"></a>

### `la.pp.readCytome`

```python
la.pp.readCytome(
    source,
    genes=None,
    modality='RNA',
    layer='auto',
    gene_name_column='auto',
    obs_name_column='barcode',
)
```

Read a cytome dataset (optionally a gene subset) into an AnnData.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `source` | str, Path or cytome.CytomeDataset | Path to a `.cytome` file, or an open dataset. |
| `genes` | list of str, optional | Gene names to load. When given, only these genes' expression is streamed from disk (one pass, sparse, in the stored dtype, so the values equal the same columns of a whole-file read) — for LARIS this is the union of ligand and receptor names, typically a few thousand genes. Genes absent from the dataset are silently skipped here; `prepareLRInteraction` reports them through its `unmatched` handling. When None, the full matrix is materialized via `cytome.to_anndata`. |
| `modality` | str, default='RNA' | Cytome modality to read. |
| `layer` | str, default='auto' | Matrix layer within the modality (`{modality}_{layer}`). 'auto' reads whichever matrix the file records as having been `adata.X` - `counts` on cytome < 0.3.0 and on genuine raw counts, `data` on 0.3.0+ files converted from normalised expression. Pass a name to override. |
| `gene_name_column` | str, default='auto' | Column of the cytome genes table holding the gene names that match the LR database. 'auto' uses `symbol` where present and falls back to `gene_id` (`cytome.from_anndata` stores AnnData var_names in `gene_id` and leaves `symbol` empty; GTF-annotated datasets carry readable names in `symbol`). |
| `obs_name_column` | str, default='barcode' | Column of the cytome cells table to use as `obs_names`. |

**Returns**

- **AnnData**

  With `.X` (CSR), `.obs` from the cells table, and every
  each embedding exposed under its short obsm key (`RNA_spatial`
  or the pre-0.2.6 `RNA_obsm_X_spatial` both give
  `obsm['spatial']` / `obsm['X_spatial']`), and, when the
  dataset stores tissue images (cytome >= 0.2.6),
  `uns['spatial']` in the scanpy convention.

---

<a id="readlrcytome"></a>

### `la.pp.readLRCytome`

```python
la.pp.readLRCytome(
    source,
    obs_name_column='barcode',
)
```

Read an LR-score cytome (written by `prepareLRInteraction`).

The matrix is streamed back in chunks, so peak memory is one chunk plus
the assembled result.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `source` | str, Path or cytome.CytomeDataset | Path to a `.cytome` file holding an `RNA_lrscore` matrix, or an open dataset. |
| `obs_name_column` | str, default='barcode' | Column of the cells table to use as `obs_names`. |

**Returns**

- **AnnData**

  `.X` cells x LR pairs (CSR), `.var_names` the pair names,
  `.obs` from the cells table, `.obsm` from the embeddings and,
  when present, `.uns['spatial']`.

**Notes**

String columns come back as `object` dtype: SQLite storage does not
preserve pandas categoricals. Cast with `.astype('category')` if a
downstream step needs it.

---

<a id="spatialoffsetmultisample"></a>

### `la.pp.spatialOffsetMultisample`

```python
la.pp.spatialOffsetMultisample(
    sampleKey,
    spatialKey='X_spatial',
    gridNCols=None,
    offsetFactor=2.0,
    copy=False,
)
```

Offset spatial coordinates so that kNN stays within each sample.

When multiple tissue sections are merged into a single AnnData, their
spatial coordinates may overlap. This function centers each sample's
coordinates to (0, 0) and then arranges them on a grid with large offsets
so that nearest-neighbor queries never cross sample boundaries.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `adata` | AnnData | Merged AnnData with spatial coordinates in `adata.obsm[spatialKey]`. |
| `sampleKey` | str | Column in `adata.obs` identifying samples (e.g., `'sample_id'`). |
| `spatialKey` | str, default `'X_spatial'` | Key in `adata.obsm` for spatial coordinates. Supports 2D and 3D. |
| `gridNCols` | int, optional | Number of columns in the layout grid. Defaults to `ceil(sqrt(n_samples))`. |
| `offsetFactor` | float, default 2.0 | Multiplier on the maximum sample diameter to determine grid spacing. Larger values give more separation. |
| `copy` | bool, default False | If True, return a modified copy instead of modifying in place. |

**Returns**

- **AnnData or None**

  If `copy=True`, returns a new AnnData. Otherwise modifies in place
  and returns None.

**Notes**

Metadata about the applied offsets is stored in
`adata.uns['spatial_offset_info']`.

**Examples**

```python
>>> import laris as la
>>> import anndata as ad
>>> adata = ad.concat([s1, s2, s3, s4], label='sample_id')
>>> la.pp.spatialOffsetMultisample(adata, sampleKey='sample_id')
>>> # Now run prepareLRInteraction — kNN will stay within-sample
>>> lr_adata = la.tl.prepareLRInteraction(adata, lr_df)
```
