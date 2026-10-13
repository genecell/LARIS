# `laris.tl`: Analysis

<!-- Generated from the docstrings by tools/make_api_docs.py; do not edit by hand. -->

LR scores, the matched-gene background and exact P values, cross-condition comparison, decoy controls.

| Function | Summary |
|---|---|
| [`prepareLRInteraction`](#preparelrinteraction) | Calculate ligand-receptor interaction scores using spatial neighborhood information. |
| [`prepareLRBackground`](#preparelrbackground) | Build the matched-gene background for factorized null p-values. |
| [`permuteLRPairs`](#permutelrpairs) | Build a decoy LR database: the calibration control for this test. |
| [`decoyFDR`](#decoyfdr) | Empirical pairing-FDR for each target row, by target-decoy estimation. |
| [`computeDecoyFDR`](#computedecoyfdr) | One-call target-decoy FDR: build the decoy, run it, return q_decoy. |
| [`decoyReport`](#decoyreport) | Dataset-level summary of what the database contributes over chance. |
| [`LRBackground`](#lrbackground) | Product of `prepareLRBackground`; consumed by `runLARIS`. |
| [`runLARIS`](#runlaris) | Identify spatially-specific ligand-receptor interactions using LARIS algorithm. |
| [`compareLARIS`](#comparelaris) | Compare LARIS cell-cell communication results across conditions. |
| [`compareLARISMatched`](#comparelarismatched) | Compare conditions on cell-level LR scores at matched cell states. |
| [`combineComparisons`](#combinecomparisons) | Combine two comparison tables into one p-value per interaction. |
| [`buildJointEmbedding`](#buildjointembedding) | Build a joint expression embedding across samples, in `obsm[key_added]`. |
| `readCytome` | Same function as [`la.pp.readCytome`](pp.md#readcytome). |

---

<a id="preparelrinteraction"></a>

### `la.tl.prepareLRInteraction`

```python
la.tl.prepareLRInteraction(
    data,
    lr_df=None,
    number_nearest_neighbors=20,
    use_rep_spatial='X_spatial',
    unmatched='drop',
    sigma='adaptive',
    return_type='auto',
    output=None,
    overwrite=False,
    block_size=None,
    section_key=None,
)
```

Calculate ligand-receptor interaction scores using spatial neighborhood information.

This function computes diffused ligand-receptor interaction scores by considering
the spatial context of each cell. It uses k-nearest neighbors to create a spatial
neighborhood graph and calculates element-wise multiplication of diffused ligand
and receptor expression levels.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `data` | AnnData, str, Path or cytome.CytomeDataset | Gene expression with spatial coordinates in `.obsm[use_rep_spatial]`: an AnnData, a path to a `.cytome`/`.db` file, or an open cytome dataset. Only the ligand/receptor genes are ever read. |
| `lr_df` | pd.DataFrame | DataFrame containing ligand-receptor pairs with columns 'ligand' and 'receptor'. |
| `number_nearest_neighbors` | int, default=20 | Number of nearest neighbors to consider for spatial diffusion. Matches the tutorial and manuscript settings, and the two neighbourhood sizes in `runLARIS`. Pass 10 for pre-v0.10.0 behaviour. |
| `use_rep_spatial` | str, default='X_spatial' | Key in `.obsm` containing spatial coordinates. |
| `unmatched` | {'drop', 'error'}, default='drop' | How to handle ligand or receptor names in `lr_df` that are absent from the data's `var_names`. With 'drop', unmatched pairs are removed and a UserWarning summarises how many were dropped and lists example missing names; this is convenient for targeted panels (e.g. Xenium, MERFISH) where most database genes are legitimately absent. With 'error', a ValueError is raised instead, which is safer when supplying a custom database where a missing name likely indicates a typo. |
| `sigma` | float or 'adaptive', default='adaptive' | Bandwidth of the exponential kernel converting spatial k-NN distances into diffusion weights (`exp(-d / sigma)`). 'adaptive' uses half the mean k-NN edge distance, making the result independent of coordinate units (pixels vs micrometres) and platform spot spacing. Pass a number for an absolute bandwidth in coordinate units. |
| `return_type` | {'auto', 'anndata', 'cytome'}, default='auto' | Format of the returned LR-score object. 'auto' follows the input: an AnnData in gives an AnnData back, a cytome in gives an LR cytome back. This keeps large analyses on disk - materialising the scores for a 500k-cell dataset in memory is exactly what a cytome input is trying to avoid - while small AnnData workflows are unchanged. |
| `output` | str, optional | Where to write the LR cytome. Defaults to the input path with a `.lr` infix (`sample.cytome` -> `sample.lr.cytome`). Required when writing a cytome from an in-memory AnnData. |
| `overwrite` | bool, default=False | Allow replacing an existing `output` file. |
| `block_size` | int, optional | Number of cells to diffuse and score at a time. Bounds peak memory to roughly `block_size x n_lr_pairs` instead of `n_cells x n_lr_pairs`. Defaults to a single pass for AnnData output and 10,000-cell blocks when writing a cytome. Results are identical for any block size. |
| `section_key` | str, optional | Column in `.obs` identifying which tissue section each cell belongs to. When several sections are tiled into one coordinate system, the spatial k-NN otherwise joins cells that are close on the slide but came from different sections. With `section_key` the neighbour graph is built independently within each section, so no edge crosses a boundary (GitHub issue #8). Leave as None for a single section. |
| `adata` | deprecated | Former name of `data`. |

**Returns**

- **AnnData or str**

  With `return_type='anndata'`, a new AnnData object:

  - `.X`: Sparse matrix of LR interaction scores (cells x LR pairs)
  - `.var_names`: Ligand-receptor pair names in format "ligand::receptor"
  - `.obs`: Cell metadata copied from the input
  - `.obsm`: Spatial and other representations copied from the input

  With `return_type='cytome'`, the path to the written LR cytome:
  a dataset whose features are LR pairs, carrying a single
  `RNA_lrscore` layer (and no counts, because there are none).
  Read it back with `laris.pp.readLRCytome`, or pass it
  straight to `laris.tl.runLARIS` as `lr_data`.

**Examples**

```python
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
```

---

<a id="preparelrbackground"></a>

### `la.tl.prepareLRBackground`

```python
la.tl.prepareLRBackground(
    data,
    lr_df=None,
    n_pool=4000,
    augment_pool=True,
    n_matched_genes='auto',
    matching='balanced',
    exclude_self=True,
    number_nearest_neighbors=20,
    use_rep_spatial='X_spatial',
    sigma='adaptive',
    use_rep_gsp=None,
    n_nearest_neighbors_gsp=20,
    sigma_gsp='adaptive',
    layer=None,
    section_key=None,
    block_size=512,
    verbosity=1,
)
```

Build the matched-gene background for factorized null p-values.

One background serves every `runLARIS` call on the same cells: the
Gram tables depend only on the spatial graphs and the gene pool, not on
the cell-type grouping or on `mu` / `spatial_weight`, so parameter
sweeps and different `groupby` choices reuse it unchanged.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `data` | AnnData, str or cytome dataset | The same expression object passed to `prepareLRInteraction` (`layer` selects a layer): an AnnData, or a cytome path or open dataset. |
| `lr_df` | DataFrame | The LR database subset used for the analysis ('ligand'/'receptor' columns). Needed to know which genes require matched sets and which gene combinations are real database pairs (excluded from nulls). |
| `n_pool` | int | Candidate pool size U. The Gram tables are U x U; the default keeps them ~130 MB each in float64. |
| `augment_pool` | bool, default=True | Repair the candidate pool so that every ligand and receptor has peers on both sides by expression. Two rules:<br><br>• *saturation*: genes whose matched set would lie entirely below them contribute their own nearest neighbours from the whole transcriptome (the quantile grid is rank-uniform and starves the extreme top of the abundance range; such a gene would beat its own null arithmetically);<br>• *side repair* (v0.15.0): every LR gene gets k/2 candidates below and k/2 above it by mean, taken from the genes nearest to it, and only genes detected in at least one cell are candidates. A gene the whole transcriptome cannot serve on one side gets a smaller set that still straddles it (see `matching`).<br><br>Set False to reproduce the v0.12.0 pool exactly. |
| `n_matched_genes` | int or 'auto', default='auto' | Matched genes per side (k). The factorized support per pair is at most k**2 and the exact p-value floor is 1/(k**2+1) before exclusions; a row's own support is reported in its `null_size` column. Within a sender-receiver pair BH can call an isolated interaction at the floor only if 1/(k**2+1) <= 0.05/m with m tested LR pairs, i.e. k**2 >= 20 m. 'auto' sizes k to the database: the smallest k with k**2 >= 20 x (number of LR pairs present in the data), clipped to [100, 200] (v0.15.0; earlier versions used 100, which cannot call an isolated hit among more than 500 tested pairs). Only the null assembly grows with k (as k**2); the Gram tables do not depend on it. |
| `matching` | {'balanced', 'knn'}, default='balanced' | How the k matched genes are chosen around each LR gene in (mean, variance) space. 'balanced' takes k/2 nearest pool genes below the gene's mean and k/2 above, so the matched set straddles it. A gene with fewer than k/2 candidates on one side, even after the side repair, gets t = min(k/2, below, above) on each side: a smaller set that still straddles it, reported as `edge_limited` (up to the v0.15.0 tag the short side was backfilled from the other side, and such pairs were left untested). 'knn' takes the k nearest regardless of side (the pre-v0.13 behaviour); it cannot straddle a gene at the edge of the pool and is **deprecated**. |
| `exclude_self` | bool, default=True | Never put an LR gene into its own matched set. |
| `number_nearest_neighbors`, `use_rep_spatial`, `sigma` | int, str, float or 'adaptive' | The diffusion graph: must match the `prepareLRInteraction` call (defaults 20, 'X_spatial', 'adaptive'). `sigma` is the diffusion kernel's bandwidth, 'adaptive' or a number in coordinate units; up to v0.15.0 the background always used 'adaptive'. `runLARIS` compares these with the settings `prepareLRInteraction` records on its output and refuses a mismatch before any work. |
| `use_rep_gsp`, `n_nearest_neighbors_gsp`, `sigma_gsp` | str, int, float or 'adaptive' | The spatial-specificity graph: must match `runLARIS`'s `use_rep` / `n_nearest_neighbors` / `sigma` (defaults: as `use_rep_spatial`, 20, 'adaptive'). `runLARIS` checks them before any work. |
| `layer` | str, optional | Layer holding the expression values (default: `.X`); must match what `prepareLRInteraction` read. For a cytome source it names the cytome layer; None reads whichever matrix the file records as `adata.X`. |
| `section_key` | str, optional | Column of `.obs` naming each cell's tissue section, as passed to `prepareLRInteraction`: the neighbour graphs are then built within sections. |
| `block_size` | int, default=512 | Cells per block when diffusing the table genes (bounds peak memory; results do not depend on it). |
| `verbosity` | int, default=1 | 1 prints progress; 0 is silent. |
| `adata` | deprecated | Former name of `data`. |

**Returns**

- **LRBackground**

  The candidate pool, every LR gene's matched sets, and the diffused
  profiles and Gram tables that score any pseudo-pair; pass it to
  `runLARIS(..., background=)`. `params` records the settings it
  was built with.

**Notes**

An AnnData and a cytome of the same data give the **same background,
bit for bit**: the pool and matched sets are chosen once, by one
function, from per-gene statistics over every gene, and the tables are
built by one builder. The cytome path only reads the data in bounded
gene blocks (v0.14.1; up to v0.14.0 the two could differ, see the
changelog).

---

<a id="permutelrpairs"></a>

### `la.tl.permuteLRPairs`

```python
la.tl.permuteLRPairs(
    lr_df,
    data=None,
    random_seed=0,
    preserve_genes=True,
    method='degree',
)
```

Build a decoy LR database: the calibration control for this test.

LARIS's p-value is a *competitive* test in the sense of Goeman &
Bühlmann (2007): it asks whether a real ligand-receptor pair beats
expression-matched genes. The null distribution of a competitive test
is generated by permuting **gene identity**, not sample labels.

Shuffling cell-type labels or spatial coordinates is the calibration
procedure for a *self-contained* test (the design CellPhoneDB and
CellChat use), and it is not this test's null. Under a label shuffle
every group converges to the tissue average, so the cell-type factors
become a constant that cancels between the observed score and the
null, leaving a comparison of spatial co-expression that real
database pairs win on merit. Such a run can therefore return *more*
calls than the real one, and that is not evidence of a defect. See
tutorial 07.

This function builds the control that does apply: a database of the
same size in which the pairings are random. Every returned pair is a
decoy, so a correctly calibrated run over it should yield p-values
that are close to uniform and essentially no FDR-significant calls,
at any dataset size.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `lr_df` | pandas.DataFrame | The real database, with 'ligand' and 'receptor' columns. Its size sets the size of the decoy database so that the multiple-testing burden matches. |
| `data` | AnnData, optional | If given, decoy genes are drawn from `data.var_names`. Without it they are drawn from the genes appearing in `lr_df`. |
| `random_seed` | int, default=0 | Seed for the pairing. |
| `preserve_genes` | bool, default=True | Draw ligands from the real ligand pool and receptors from the real receptor pool, re-pairing them at random. This keeps the marginal expression distribution of each side realistic and changes only the pairing, which is the thing being tested. Set False to draw both sides from all available genes. Ignored when `method='degree'`, which always reuses the real genes. |
| `method` | {'degree', 'uniform'}, default='degree' | How the decoys are drawn.<br><br>`'degree'` rewires the real database by repeated double-edge swaps: pick two pairs (L1,R1) and (L2,R2) and replace them with (L1,R2) and (L2,R1). Every gene therefore appears in **exactly as many decoy pairs as real ones**, and only the pairing changes.<br><br>This matters more than it sounds. Curated databases are strongly hub-structured - in CellChatDB's human table restricted to a tonsil, ITGB1 appears in 55 pairs and the median gene in 2, with the top 10% of genes carrying ~40% of the pairs. Drawing decoys uniformly flattens that: it under-samples the broadly expressed integrin and collagen hubs, which are individually unremarkable and dilute the real database's hit rate, and over-samples rare cell-type-restricted genes, which are not. A uniform decoy is therefore an *easier* database than the real one and will overstate the false-positive rate.<br><br>`'uniform'` is the naive version, kept because it is the obvious thing to try and the comparison between the two is informative. |

**Returns**

- **pandas.DataFrame**

  Columns 'ligand', 'receptor' and 'interaction_name', containing no
  pair present in `lr_df`.

**Examples**

```python
>>> decoy = la.tl.permuteLRPairs(lr_df, adata, random_seed=0)
>>> bg_d = la.tl.prepareLRBackground(adata, decoy,
...                                  use_rep_spatial="X_spatial")
>>> lr_d = la.tl.prepareLRInteraction(adata, decoy,
...                                   use_rep_spatial="X_spatial")
>>> _, res_d = la.tl.runLARIS(lr_d, adata, groupby="cell_type",
...                           background=bg_d)
>>> (res_d.p_value_fdr < 0.05).sum()      # should be ~0
```

---

<a id="decoyfdr"></a>

### `la.tl.decoyFDR`

```python
la.tl.decoyFDR(
    target_res,
    decoy_res,
    p_col='p_value',
    pseudocount=1.0,
)
```

Empirical pairing-FDR for each target row, by target-decoy estimation.

The factorized p-value certifies that a pair's expression is arranged
with respect to the tested cell types beyond expression-matched
chance. It does not certify the *pairing*: a degree-preserving
rewiring of the database reproduces part of the calls (tonsil,
v0.15.0: 50 against the real database's 95 at FDR < 0.05), because
for marker-structured pairs the pairing is unidentifiable from
expression and coordinates alone. See tutorial 07.

This function measures that error rate empirically, the way
proteomics has done for two decades (target-decoy search, Elias &
Gygi 2007): run the same pipeline on a decoy database whose pairings
are scrambled but whose genes and degrees are identical
(`permuteLRPairs` with `method='degree'`), and estimate, at
each threshold t,

```text
FDR(t) = (decoy rows with p <= t, +pseudocount) / n_decoy
         -------------------------------------------------
         (target rows with p <= t)               / n_target
```

monotonized into a q-value (minimum over all looser thresholds) and
clipped to 1. The +1 pseudocount keeps the estimate conservative
when decoy counts are small.

Because the degree decoy re-pairs real genes, a fraction of decoys
may be genuine unannotated interactions; on tonsil this contamination
measures 0.3% (pathway-sharing check), so the estimate is effectively
unbiased there, and in general it errs conservative.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `target_res`, `decoy_res` | pandas.DataFrame | Results from `runLARIS` on the real and decoy databases, run with the same settings and (ideally) the same background. Rows of `decoy_res` whose pair coincides with a real database pair should be excluded first (`computeDecoyFDR` does this). |
| `p_col` | str, default='p_value' | Column holding the raw p-values. Raw, not BH-adjusted: the two runs have different multiplicity structures, and the raw p is the comparable scale. |
| `pseudocount` | float, default=1.0 | Added to the decoy count at every threshold. |

**Returns**

- **pandas.Series**

  `q_decoy`, aligned to `target_res.index`; NaN where the
  target p is NaN.

---

<a id="computedecoyfdr"></a>

### `la.tl.computeDecoyFDR`

```python
la.tl.computeDecoyFDR(
    data,
    lr_df,
    target_res,
    background,
    random_seed=0,
    prepare_kwargs=None,
    run_kwargs=None,
    verbosity=1,
)
```

One-call target-decoy FDR: build the decoy, run it, return q_decoy.

Builds a degree-preserving decoy database, runs it through
`prepareLRInteraction` + `runLARIS` against the *same* background
(valid because the rewiring preserves the exact gene multiset, so
every per-gene matched set is identical), drops decoy rows whose
pair coincides with a real one, and returns
`(q_decoy, decoy_res)`.

`prepare_kwargs` and `run_kwargs` must repeat the settings of the
target run (`use_rep_spatial`, `groupby`, ...); they are passed
through verbatim.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `data` | AnnData, str or cytome dataset | The expression data of the target run. |
| `lr_df` | pandas.DataFrame | The real LR database of the target run ('ligand'/'receptor' columns); it is rewired into the decoy. |
| `target_res` | pandas.DataFrame | The target run's results (`runLARIS`'s cell-type table), whose raw `p_value` the q-values are computed for. |
| `background` | LRBackground | The target run's background. The decoy keeps every gene and degree, so the same matched sets and tables apply. |
| `random_seed` | int, default=0 | Seed of the rewiring. |
| `prepare_kwargs` | dict, optional | Keyword arguments for `prepareLRInteraction` on the decoy database; repeat the target run's settings. |
| `run_kwargs` | dict, optional | Keyword arguments for `runLARIS` on the decoy database; repeat the target run's settings. |
| `verbosity` | int, default=1 | 1 prints the decoy's size; 0 is silent. |

**Returns**

- **`q_decoy` : pandas.Series**

  The target-decoy q-value of every target row
  (`decoyFDR`), aligned to `target_res.index`.

- **`decoy_res` : pandas.DataFrame**

  The decoy run's results, without rows whose pair coincides with a
  real database pair.

**Examples**

```python
>>> q, dec = la.tl.computeDecoyFDR(
...     adata, lr_df, res, background=bg,
...     prepare_kwargs={"use_rep_spatial": "X_spatial"},
...     run_kwargs={"use_rep": "X_spatial",
...                 "use_rep_spatial": "X_spatial",
...                 "groupby": "cell_type"})
>>> res["q_decoy"] = q
```

---

<a id="decoyreport"></a>

### `la.tl.decoyReport`

```python
la.tl.decoyReport(
    target_res,
    decoy_res,
    thresholds=(0.05, 0.01),
    p_col='p_value',
    fdr_col='p_value_fdr',
    verbosity=1,
)
```

Dataset-level summary of what the database contributes over chance.

`p_value` answers: *is this pair's expression arranged specifically
with respect to these two cell types, beyond expression-matched
chance?* It does not answer *does this ligand bind this receptor
here?* - the pairing is asserted by the database, not measured from
the data. A degree-preserving rewiring of the database, scored
identically, quantifies the difference: whatever it recovers is what
chance pairing alone achieves on this dataset.

This returns (and by default prints) that comparison at the dataset
level, which is the scale at which it is informative - the per-row
q-value from `decoyFDR` is close to constant within a dataset,
so a single figure carries nearly all of the signal.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `target_res`, `decoy_res` | pandas.DataFrame | `runLARIS` results for the real and decoy databases, same settings, ideally the same background. See `computeDecoyFDR`, which produces `decoy_res`. |
| `thresholds` | tuple of float, default=(0.05, 0.01) | FDR thresholds to report. |
| `p_col` | str, default='p_value' | Column holding the raw p-values (for the q-values). |
| `fdr_col` | str, default='p_value_fdr' | Column holding each run's own FDR (for the call counts). |
| `verbosity` | int, default=1 | 1 prints the report; 0 returns it silently. |

**Returns**

- **dict**

  `per_threshold` (target calls, decoy calls, and the estimated
  pairing-FDR at each threshold), `q_min`, and `n_rows`.

**Examples**

```python
>>> q, dec = la.tl.computeDecoyFDR(adata, lr_df, res, background=bg,
...                                prepare_kwargs=..., run_kwargs=...)
>>> rep = la.tl.decoyReport(res, dec)
```

---

<a id="lrbackground"></a>

### `la.tl.LRBackground`

Product of `prepareLRBackground`; consumed by `runLARIS`.

`gene_index` lists the U genes covered by the tables (candidate pool
plus every LR gene, so real pairs resolve against the same tables).
`P` is their diffused matrix (**cells x U**, dense float32,
C-contiguous) using the same kernel graph as
`prepareLRInteraction`. The cells-major layout is deliberate: every
consumer gathers cells, and that is a contiguous copy here. The Gram tables are
over the *gsp* graph that `runLARIS` builds for spatial specificity.

`P_det` is `P` on the entries whose raw count is non-zero (sparse
CSR, cells x U, genes ascending): the detected part of every profile,
which is all the masked pseudo-pair profiles need. It replaces the dense
`Q = P - P_det` that versions up to 0.14 stored (a second cells x U
float32 array: 7.9 GB at 500,000 cells and 4,240 genes); `Q` is still
available as a property, rebuilt exactly on demand.

Built by `prepareLRBackground`; not meant to be constructed
directly.

**Attributes**

| Parameter | Type | Description |
|---|---|---|
| `gene_index` | pandas.Index | The U table genes: the candidate pool plus every LR gene. |
| `P` | numpy.ndarray | Diffused profiles, cells x U, float32, C-contiguous. |
| `P_det` | scipy.sparse.csr_matrix | `P` on the entries whose raw count is non-zero. |
| `G_W`, `G_W2` | numpy.ndarray | Edge Gram tables (U x U, float64) over the spatial-specificity graph W and over W'W. |
| `G_dot`, `G_sq` | numpy.ndarray | Moment Gram tables (U x U, float64): sums over cells of products of profiles and of squared profiles. |
| `R` | float | Analytic expectation of a random-graph row's squared weights, times the number of cells (the shuffled-graph baseline). |
| `n_cells` | int | Number of cells. |
| `matched_ligand`, `matched_receptor` | dict of str to numpy.ndarray | Every ligand's and receptor's matched genes, as positions in `gene_index`: k genes, or fewer for a gene whose set is truncated (possibly none). |
| `db_pairs` | frozenset of tuple | The database's (ligand, receptor) pairs; never used as pseudo-pairs. |
| `match_frac_below` | dict of str to float | Per LR gene, the fraction of its matched set with a lower mean (1.0: every match is weaker, the null is easy; NaN: empty set). |
| `edge_limited` | int | Number of LR genes whose matched set is truncated (fewer than k/2 candidates on one side even after the side repair). |
| `params` | dict | The settings the background was built with. |
| `edge_genes` | frozenset of str or None | The LR genes whose matched set is truncated (None for backgrounds built before v0.15.0). |

---

<a id="runlaris"></a>

### `la.tl.runLARIS`

```python
la.tl.runLARIS(
    lr_data,
    data,
    use_rep='X_spatial',
    n_nearest_neighbors=20,
    random_seed=27,
    n_repeats=None,
    mu=0.25,
    sigma='adaptive',
    remove_lowly_expressed=True,
    expressed_pct=0.1,
    n_cells_expressed_threshold=100,
    n_top_lr=4000,
    by_celltype=True,
    groupby='CellTypes',
    use_rep_spatial='X_spatial',
    number_nearest_neighbors=20,
    mu_celltype=100,
    sigma_celltype='adaptive',
    expressed_pct_celltype=0.1,
    remove_lowly_expressed_celltype=True,
    mask_threshold=1e-06,
    calculate_pvalues=True,
    layer_celltype=None,
    n_neighbors_permutation=30,
    n_permutations=1000,
    chunk_size=50000,
    prefilter_fdr=True,
    prefilter_threshold=0.0,
    score_threshold=1e-06,
    spatial_weight=3.0,
    use_conditional_pvalue=None,
    rescale=True,
    cosg_backend='auto',
    specificity_reference='lr',
    section_key=None,
    background=None,
    min_null_support='auto',
    sort_by='interaction_score',
)
```

Identify spatially-specific ligand-receptor interactions using LARIS algorithm.

LARIS (Ligand And Receptor Interaction in Spatial transcriptomics) identifies LR pairs
that show spatial specificity by comparing observed spatial correlation patterns
with randomized null distributions. When `by_celltype=True`, the function also
computes cell type-specific interaction scores with optional statistical testing.

This is the main analytical function of the LARIS package, providing:

1. **Spatial Specificity Analysis**: Identifies LR pairs that show non-random
   spatial co-localization patterns (higher scores = stronger spatial organization)

2. **Cell type-specific Scores**: Integrates spatial specificity with cell type
   expression specificity and spatial co-localization to identify which sender-
   receiver cell type pairs are communicating via which LR pairs

3. **Statistical Testing**: Optional permutation-based P values with FDR correction
   to identify statistically significant interactions

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `lr_data` | AnnData, str or cytome dataset | LR interaction scores from `prepareLRInteraction()`: an AnnData, or an LR-score cytome (path or open dataset).<br><br>Required contents:<br>• `.X`: Diffused LR scores (cells × LR pairs)<br>• `.var_names`: LR pair names ("ligand::receptor")<br>• `.obsm[use_rep]`: Spatial coordinates or other representation |
| `data` | AnnData, str or cytome dataset, optional | The expression data the LR scores were computed from, with spatial information (an AnnData, or a cytome path or open dataset). **Required when `by_celltype=True`**.<br><br>Required contents (when by_celltype=True):<br>• `.obs[groupby]`: Cell type annotations<br>• `.obsm[use_rep_spatial]`: Spatial coordinates<br>• `.X` or `.layers[layer_celltype]`: Gene expression |
| `use_rep` | str, default='X_spatial' | Key in `lr_adata.obsm` for coordinates to use in spatial specificity analysis. Typically spatial coordinates, but could be other representations. |
| `n_nearest_neighbors` | int, default=20 | Number of spatial neighbors for building the adjacency matrix in the spatial specificity analysis. Larger values capture broader spatial patterns. Note the physical area spanned by k neighbors differs between platforms (a k of 20 covers a much smaller area at 8 um Visium HD than at Stereo-seq bin50); rankings are robust across k = 3-50, but top-hit identity shifts at very small k. (Changed from 10 to 20 in v0.10.0 to match the published analyses; pass n_nearest_neighbors=10 to reproduce pre-0.10 results.) |
| `random_seed` | int, default=27 | Random seed for reproducibility of permutation tests. |
| `n_repeats` | int, default=3 | Number of random permutations to generate the null distribution for spatial specificity scoring. More repeats give more stable estimates. |
| `mu` | float, default=0.25 | Strength of the null subtraction in the spatial specificity score: `score = cos_observed - mu * cos_shuffled`. Larger values subtract the randomized-background signal more strongly and are more conservative; mu=1 corresponds to the plain difference Delta = cos_observed - cos_shuffled. The default 0.25 matches the value used in the LARIS manuscript analyses. Not to be confused with `mu_celltype`, which is the (unrelated) COSG regularization parameter for cell type specificity. (Changed from 1 to 0.25 in v0.10.0; pass mu=1 to reproduce pre-0.10 results.) |
| `sigma` | float or 'adaptive', default='adaptive' | Bandwidth of the exponential kernel converting spatial k-NN distances to edge weights in the spatial specificity step (`exp(-d / sigma)`). 'adaptive' uses half the mean k-NN edge distance, which makes results independent of coordinate units (chip pixels, image pixels, or micrometres) and platform spot spacing, and matches the kernel used by prepareLRInteraction and the cell-type co-localization step. A numeric value is an absolute distance in coordinate units. (Changed from a fixed 100 to 'adaptive' in v0.10.0; pass sigma=100 to reproduce pre-0.10 results, which assumed roughly-micrometre coordinates.) |
| `remove_lowly_expressed` | bool, default=True | Whether to filter out LR pairs with low expression before ranking. |
| `expressed_pct` | float, default=0.1 | Minimum fraction of cells expressing an LR pair (if remove_lowly_expressed=True). |
| `n_cells_expressed_threshold` | int or float, default=100 | Minimum number of cells expressing an LR pair for it to be ranked. Pairs below this threshold receive a penalty in ranking. A value >= 1 is an absolute cell count; a float in (0, 1) is interpreted as a fraction of the total number of cells (e.g. 0.001 = 0.1% of cells), which scales better across platforms with very different cell/bin counts (e.g. 8 um Visium HD bins). The penalty affects the ranking only. The cell-type interaction score always uses the pair's actual spatial specificity (v0.15.0; earlier versions also fed the penalised ranking value into the score, which zeroed every pair below the threshold). |
| `n_top_lr` | int, default=4000 | Number of top-ranked spatially-specific LR pairs to return. With `background=` and `calculate_pvalues=True` every LR pair is still *tested*: the cell-type analysis and the FDR run over all pairs, and only the returned tables are restricted to the top `n_top_lr`. Selecting pairs by their own spatial specificity before testing would bias the p-values (the selection would be on the outcome), so `n_top_lr` only limits what is reported. |
| `by_celltype` | bool, default=True | Whether to compute cell type-specific interaction scores. If False, only returns spatial specificity results (much faster). If True, `data` must be provided. The Other Parameters configure this cell-type analysis and its P values; they are used only when `by_celltype=True`. |

**Other Parameters**

| Parameter | Type | Description |
|---|---|---|
| `groupby` | str, default='CellTypes' | Column in `adata.obs` defining cell type groups. |
| `use_rep_spatial` | str, default='X_spatial' | Key in `adata.obsm` for spatial coordinates (for cell type analysis). |
| `number_nearest_neighbors` | int, default=20 | Number of spatial neighbors for cell type co-localization analysis. (Changed from 10 to 20 in v0.10.0 for consistency with n_nearest_neighbors; pass number_nearest_neighbors=10 to reproduce pre-0.10 results.) |
| `mu_celltype` | float, default=100 | Regularization parameter for COSG cell type specificity calculation. Higher values more strongly penalize broadly expressed genes. This is COSG's lambda-style regularizer and is unrelated to `mu` (the null-subtraction strength in the spatial specificity step). |
| `sigma_celltype` | float or 'adaptive', default='adaptive' | Kernel bandwidth for the cell-type co-localization spatial graph, with the same semantics as `sigma`. 'adaptive' reproduces the pre-0.10 behaviour of this step. |
| `expressed_pct_celltype` | float, default=0.1 | Minimum expression fraction for cell type analysis. |
| `remove_lowly_expressed_celltype` | bool, default=True | Whether to filter lowly expressed genes in cell type analysis. |
| `mask_threshold` | float, default=1e-6 | Numerical threshold for masking near-zero values. |
| `calculate_pvalues` | bool, default=True | Whether to perform permutation testing for statistical significance. Set to False for faster exploratory analysis. |
| `layer_celltype` | str, optional | Layer in `adata.layers` to use for expression data. If None, uses `adata.X`. |
| `n_neighbors_permutation` | int, default=30 | Number of similar interactions to use as background controls for permutation testing. These are selected based on similarity of diffused score profiles. |
| `n_permutations` | int, default=1000 | **Legacy path only, and ignored when `background=` is given.**<br><br>Without a background, p-values come from sampling `n_permutations` draws from a resampled-pair null, and the floor is `1/(n_permutations+1)`.<br><br>With a background the null is *enumerated exactly* - every one of the `n_matched_genes ** 2` pseudo-pairs is scored and the p-value is the exact tail `(exceed + 1) / (support + 1)`. There is no sampling, no seed and nothing for this parameter to do: the floor is `1/(n_matched_genes ** 2 + 1)` and `n_matched_genes` is the only knob that moves it. Passing this parameter together with `background=` raises a `FutureWarning`. |
| `chunk_size` | int, default=50000 | Number of interactions to process simultaneously during permutation. Larger values are faster but use more memory. |
| `prefilter_fdr` | bool, default=True | **Legacy (no-background) path only.** With `background=` the interactions that Benjamini-Hochberg covers are fixed by the null (see `min_null_support`) and this parameter is ignored.<br><br>In the legacy path, True restricts BH to rows whose score exceeds `prefilter_threshold`. Restricting the tested rows by the observed score is only valid with a p-value that conditions on the same event, so the legacy path then uses the conditional p-value (`use_conditional_pvalue`). |
| `prefilter_threshold` | float, default=0.0 | Minimum interaction score for FDR testing in the legacy path (if prefilter_fdr=True). Ignored with `background=`. |
| `score_threshold` | float, default=1e-6 | Scores below this are set to exactly 0.0. With `background=` the same rule is applied to every pseudo-pair, so a pseudo-pair counts as non-zero (`null_support`) under exactly the condition an observed score does. |
| `spatial_weight` | float, default=3.0 | Exponent applied to the (non-negative-clamped) spatial specificity score before it multiplies the cell-type interaction score: `interaction_score *= max(delta, 0) ** spatial_weight`. **Despite the name this is a power, not a multiplicative weight** - raising it changes the score non-linearly.<br>• 0: Ignore spatial specificity entirely<br>• 1: Linear influence<br>• >1: Stronger emphasis on spatially specific LR pairs (the default 3 is the generally recommended setting and matches the tutorial); increase it further to sharpen the ranking toward pairs with strong spatial structure, decrease toward 1 when spatial specificity estimates are noisy (e.g. very sparse data)<br>• <1: Weaker emphasis Negative spatial specificity scores are clamped to 0 before powering (v0.10.0): a pair whose spatial pattern is indistinguishable from or weaker than the random background contributes no interaction score, instead of producing NaN (fractional weights) or a sign flip (even weights). (Changed from 1.0 to 3.0 in v0.10.0; pass spatial_weight=1.0 to reproduce pre-0.10 results.) |
| `use_conditional_pvalue` | bool or None, default=None | **Legacy (no-background) path only.** When True:<br>• Interactions with score=0 get p-value=1.0<br>• Non-zero scores are compared only to non-zero background scores None (default) uses the conditional p-value exactly when `prefilter_fdr=True`, which is the combination that keeps the FDR valid: a test restricted to non-zero scores needs a p-value that conditions on the score being non-zero. Passing False together with `prefilter_fdr=True` warns and uses the conditional p-value. |
| `rescale` | bool, default=True | Rescale interaction scores so that the mean of the top-100 scores is 0.1. The applied factor is dataset-dependent and is recorded in `lr_adata.uns['laris_scale_factor']` and `celltype_results.attrs['laris_scale_factor']` (1.0 when rescale=False), so separate runs can be put back on a common scale by dividing scores by their respective factors. Because the factor varies between runs (and with subsampling), never compare or subtract rescaled `interaction_score` values across runs directly. |
| `cosg_backend` | {'auto', 'memory', 'stream'}, default='auto' | How to compute cell type specificity. 'stream' reads the expression in chunks from a cytome via `cosg.run_cosg_cytome`, so no cells x LR-genes AnnData is built; 'memory' materialises the ligand/receptor gene subset. 'auto' streams for cytome sources and uses memory otherwise, so AnnData workflows are unaffected. The two agree to float32 precision. |
| `specificity_reference` | {'lr', 'all'}, default='lr' | Which genes set the scale when cell type specificity is normalised. `cosg.iqrLogNormalize` divides each cell type's scores by that cell type's `q0.95 - q0.75` spread over the genes it is given, so the reference set matters. 'lr' uses the ligand/receptor genes, which makes results independent of whatever *other* genes are in the object - subsetting to highly variable genes changes nothing. 'all' uses every gene, which instead makes results independent of which LR database you look up. Gene rankings within a cell type are identical either way (the transform is monotone); what changes is the relative scale between cell types, and therefore the ranking of sender-receiver pairs. 'lr' is the default and matches the published results; prefer 'all' when very few database genes are present, as on targeted panels, where the spread over a handful of LR genes is unstable (a warning fires below 100). |
| `section_key` | str, optional | Column in `.obs` identifying which tissue section each cell belongs to. Several sections tiled into one coordinate system otherwise share a spatial k-NN, so cells near a tile edge acquire neighbours from the adjacent section. With `section_key` every neighbour graph in the pipeline - the spatial specificity step, its randomised background, and the cell-type co-localization step - is built independently within each section (GitHub issue #8). Pass the same column to `prepareLRInteraction`. |
| `background` | LRBackground, optional | From `prepareLRBackground`, built on the same cells and graph settings. With `calculate_pvalues=True` it gives every LR pair an exact P value against its expression-matched null (the factorized matched-gene test): `p_value`, `p_value_fdr` and the diagnostic columns described under Returns. Its diffusion and spatial-specificity graphs must be the ones the LR scores and this call use; a mismatch raises before any work. Without it, P values come from the legacy permutation path. |
| `min_null_support` | int or 'auto', default='auto' | With `background=`: within each sender-receiver pair, Benjamini-Hochberg covers every LR pair whose matched-gene null has at least `max(1, min_null_support)` non-zero pseudo-pairs (`null_support`). Rows below that get p-value 1.0 and are not tested. 'auto' (v0.15.0) is 2 x `n_matched_genes` of the background (400 at k = 200): below that, the non-zero part of the null can come from a single matched gene's row or column of the k x k cross, and a tiny score beats an otherwise all-zero null - the test would measure whether the pair clears the 10% detection gate, not its strength. Pass 0 to test every row with a non-zero null.<br><br>The p-value is the exact tail `(exceed + 1) / (n_valid + 1)` over all valid pseudo-pairs, zeros included. That is a valid p-value only if the set of tested rows is not chosen by the observed score, so that set is defined by the null alone: a row whose null is identically zero cannot reach significance and is left out, every other row is in, whatever its own score (v0.15.0; earlier versions tested only the rows with a positive score, which inflated the FDR). `min_null_support` is a property of the null, fixed in advance, so any value keeps the FDR valid. |
| `sort_by` | {'interaction_score', 'p_value'}, default='interaction_score' | Order of the cell-type results. 'interaction_score' ranks by the score, descending, as in earlier versions. 'p_value' ranks by the p-value, ascending, ties by the score, so the head of the table is the strongest evidence; it needs `calculate_pvalues=True`. The order changes no value. |
| `lr_adata`, `adata` | deprecated | Former names of `lr_data` and `data`. |

**Returns**

- **pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]**

  If `by_celltype=False`:
      Single DataFrame with spatial specificity results:

  ```text
  Columns:
  - 'ligand': Ligand gene name
  - 'receptor': Receptor gene name  
  - 'score': LARIS spatial specificity score
  - 'Rank': Rank (0 = highest scoring)

  Index: LR pair names ("ligand::receptor")
  Sorted by score (descending)
  ```

  If `by_celltype=True`:
      Tuple of (laris_lr, celltype_results) where:

  ```text
  - **laris_lr**: DataFrame as described above

  - **celltype_results**: DataFrame with cell type-specific scores:

    Columns:
    - 'sender': Cell type sending the ligand
    - 'receiver': Cell type receiving the signal
    - 'ligand': Ligand gene name
    - 'receptor': Receptor gene name
    - 'interaction_name': "ligand::receptor"
    - 'interaction_score': Integrated LARIS score
    - 'p_value': P value (if calculate_pvalues=True). With
      ``background=``, the exact tail of the row's score among
      its expression-matched pseudo-pairs in the same
      sender-receiver pair.
    - 'p_value_fdr': Benjamini-Hochberg FDR within each
      sender-receiver pair (if calculate_pvalues=True). With
      ``background=``, a row with ``p_value_fdr < 0.05`` is an
      interaction whose score exceeds that of expression-matched
      gene pairs in the same sender-receiver context, at an FDR
      of 5% within that context. Pairs of the two cell types'
      marker genes are part of that null, so a call certifies
      specificity beyond cell-type identity; it does not test
      whether the two proteins bind (the pairing comes from the
      database).
    - 'nlog10_p_value_fdr': -log10(FDR) for visualization

    With ``background=``, these diagnostic columns are added:

    - 'null_support': how many of the row's pseudo-pairs scored
      above zero. This is the null's *effective* resolution: a
      row backed by 100 non-zero pseudo-pairs cannot really
      resolve the 1e-4 its p-value may report. See
      ``min_null_support``.
    - 'null_size': how many pseudo-pairs the row's null has: k**2
      minus the database pairs inside it, fewer when a gene's
      matched set is truncated. The p-value floor is
      1/(null_size + 1).
    - 'edge_limited': True when the ligand or the receptor has a
      truncated matched set (one of the few most or least
      abundant genes, without k/2 peers on one side; see
      ``prepareLRBackground(matching=)``). The set still
      straddles the gene, so the row is tested as usual.
    - 'ligand_rank', 'receptor_rank': descriptive partner ranks.
      ``ligand_rank`` is the observed pair's rank among the
      ligand's matched genes, each paired with the observed
      receptor: (1 + number of those pairs scoring at least as
      high) / (1 + number of valid pairs). ``receptor_rank`` is
      the same with the receptor's matched genes and the observed
      ligand. 1/(k + 1) means the gene beats all of its peers
      given its partner; a call that leans on one partner (a
      specific ligand acting through a broadly expressed
      receptor) has a small rank on that side only. Resolution
      1/(k + 1), so not a test; 1.0 for a row whose score is 0.
    - 'null_matchability': the larger of the two genes'
      saturation values - the fraction of a gene's matched set
      whose mean lies below the gene itself. 0.5 is ideal; 1.0
      means every matched gene is weaker, so every pseudo-pair
      is weaker than the real pair and the p-value overstates.
      The pool augmentation in ``prepareLRBackground`` normally
      prevents this; values near 1 among called rows mean it was
      disabled or could not reach that gene.
    - 'pair_breadth': the fraction of tested sender-receiver
      combinations in which this pair is called at FDR < 0.05.
      Genuine cell-type-specific results are narrow (medians of
      1-2% measured across four datasets); a pair called across
      more than ~25% of the grid carries no cell-type
      information, however real its expression - typically a
      tissue-ubiquitous pair.

    Sorted by ``interaction_score`` (descending) by default; with
    ``sort_by='p_value'``, by ``p_value`` (ascending), ties by
    ``interaction_score`` (descending).
  ```

**Raises**

- **ValueError**

  If by_celltype=True but adata is not provided, or if required data
  is missing from adata or lr_adata.

- **ImportError**

  If required helper functions are not available.

**See also:** `prepareLRInteraction`

**Examples**

**Example 1: Quick spatial specificity analysis (no cell types)**

```python
>>> import laris as la
>>> 
>>> # Prepare LR scores
>>> lr_adata = la.tl.prepareLRInteraction(adata, lr_df)
>>> 
>>> # Identify spatially-specific LR pairs only
>>> laris_lr = la.tl.runLARIS(
...     lr_adata,
...     by_celltype=False,
...     n_top_lr=1000
... )
>>> 
>>> print(laris_lr.head())
```

**Example 2: Full analysis with cell type-specific scores**

```python
>>> # Full LARIS analysis with cell types
>>> laris_lr, celltype_results = la.tl.runLARIS(
...     lr_adata,
...     adata,
...     by_celltype=True,
...     groupby='cell_type',
...     calculate_pvalues=True,
...     n_permutations=5000
... )
>>> 
>>> # Filter for significant interactions
>>> sig_results = celltype_results[
...     celltype_results['p_value_fdr'] < 0.05
... ]
>>> 
>>> print(f"Found {len(sig_results)} significant interactions")
```

**Example 3: Fast exploratory analysis (no p-values)**

```python
>>> laris_lr, celltype_results = la.tl.runLARIS(
...     lr_adata,
...     adata,
...     by_celltype=True,
...     calculate_pvalues=False  # Much faster!
... )
```

**Example 4: Conservative testing for sparse data**

```python
>>> laris_lr, celltype_results = la.tl.runLARIS(
...     lr_adata,
...     adata,
...     by_celltype=True,
...     use_conditional_pvalue=True,  # Robust for sparse data
...     n_permutations=5000,
...     prefilter_fdr=True,
...     prefilter_threshold=0.01  # Only test score > 0.01
... )
```

**Example 5: Emphasize spatial specificity**

```python
>>> laris_lr, celltype_results = la.tl.runLARIS(
...     lr_adata,
...     adata,
...     by_celltype=True,
...     spatial_weight=2.0  # Square the spatial scores
... )
```

---

<a id="comparelaris"></a>

### `la.tl.compareLARIS`

```python
la.tl.compareLARIS(
    data,
    conditionMap=None,
    referenceCondition=None,
    sampleToSubject=None,
    scoreCol='interaction_score',
    minSubjectsObserved=2,
    minCellTypePairs=3,
    pseudocount=1e-06,
    fdrMethod='fdr_bh',
    level='both',
    logPseudocount='auto',
    universe=None,
    conditionKey=None,
    sampleKey=None,
    subjectKey=None,
    use_rep='X_pca',
    n_anchors=100,
    k_anchor=10,
    random_seed=0,
)
```

Compare LARIS cell-cell communication results across conditions.

One entry point, two estimators, **selected by the input type**:

- `dict` of per-sample results tables (from
  `runLARIS(by_celltype=True)`) -> the **aggregate estimator**.
- `AnnData` of cell-level LR scores (per-sample
  `prepareLRInteraction` outputs concatenated, with a joint
  embedding in `.obsm`) -> the **matched estimator**.

**Which estimator, when.** They answer different questions and
neither subsumes the other:

- *Input needed*: aggregate - results tables only; matched -
  cell-level scores and a joint embedding.
- *Estimand*: aggregate - change in the subject's interaction profile,
  composition included; matched - change in LR score at *matched cell
  states*.
- *Resolution*: aggregate - per (sender, receiver, LR) triple, so
  directional claims; matched - per LR pair.
- *Composition shifts*: aggregate - part of the effect; matched -
  controlled away.
- *Typical power*: aggregate - conservative at small n; matched -
  higher (cell-level information).

Use the **aggregate** estimator when only result tables are available
(e.g. shared by a collaborator), when you need sender-to-receiver
claims, or as a fast first pass. Use the **matched** estimator when
cell-level data exists, when cell-type composition plausibly differs
between conditions (usual in disease), or when n is small and power
matters. Their disagreements are informative rather than
contradictory: an aggregate hit that vanishes at matched states
changed *compositionally*. When both inputs exist, run both and
combine with `combineComparisons` - the intersection carries
the strongest claims.

Note the two estimators can test different numbers of LR pairs on the
same cohort: detection is assessed on the input each one sees (after
cell-type aggregation and masking for the aggregate path, on the raw
cell-level matrix for the matched path), and the FDR is computed over
the tests actually performed - reported in the output. To fix the
tested set explicitly, pass `universe`.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `data` | dict or AnnData | Either `{sample_name: celltype_results_df}` (aggregate) or a concatenated cell-level LR-score AnnData (matched). The keyword `results=` remains as a deprecated alias. |
| `conditionMap` | dict | Aggregate input only: `{sample_name: condition_label}`. |
| `referenceCondition` | str | The condition every other condition is compared against. Required for both estimators. |
| `sampleToSubject` | dict, optional | Aggregate input only: `{sample_name: subject_id}`. Required when samples include technical replicates (slices of one animal); slices of one subject are averaged before testing. If None, each sample is treated as its own subject (a warning is emitted). |
| `scoreCol` | str | Aggregate input only: score column in the results tables. |
| `minSubjectsObserved` | int, default=2 | Minimum subjects per condition for a testable contrast. |
| `minCellTypePairs` | int | Retired; accepted for backward compatibility and ignored. |
| `pseudocount` | float | Aggregate input only: floor for the descriptive log2FC columns. |
| `fdrMethod` | str, default='fdr_bh' | Multiple-testing correction, over the tests actually performed. |
| `level` | {'both', 'lr', 'triple'}, default='both' | Aggregate input only: which output tables to compute. 'lr' is the fast first pass on large cohorts. |
| `logPseudocount` | 'auto' or float, default='auto' | Floor added before the log used for centring and testing. 'auto' is a per-sample, scale-equivariant floor (half the 5th percentile of the sample's non-zero scores), which makes the centring cancel any multiplicative per-sample factor exactly at any sparsity. Pass a float (e.g. 1e-8) for the fixed-floor behaviour of earlier versions. Applies to both estimators. |
| `universe` | sequence of str, optional | Restrict testing (and hence the FDR burden) to these interaction names, for both estimators. This is the clean way to make two runs - or the two estimators - answer over the *same* set of hypotheses: pass the same universe to both. Interactions in the universe but absent from the data are simply not testable; interactions outside it are dropped before any test. |
| `conditionKey`, `sampleKey`, `subjectKey`, `use_rep`, `n_anchors`, `k_anchor`, `random_seed` | see compareLARISMatched | Matched input only; see `compareLARISMatched` for their meaning. `conditionKey` and `sampleKey` are required for AnnData input. |
| `results` | deprecated | Former name of `data`. |

**Returns**

- **tuple of pd.DataFrame**

  Aggregate input: `(lr_comparison, triple_comparison)`.
  Matched input: `(comparison, per_subject_profiles)`.
  In both, the first element carries `interaction_name`,
  `log_diff`, `pvalue`, `pvalue_fdr`, subject and detection
  counts, and `test_method`.

**Examples**

```python
>>> # aggregate: from per-sample runLARIS results
>>> lr_cmp, triple_cmp = la.tl.compareLARIS(
...     {"s1": res1, "s2": res2, ...},
...     conditionMap={"s1": "control", "s2": "disease", ...},
...     referenceCondition="control",
...     sampleToSubject={"s1": "mouse1", ...})
>>>
>>> # matched: from concatenated cell-level LR scores
>>> cmp_, profiles = la.tl.compareLARIS(
...     lr_all, conditionKey="genotype", referenceCondition="WT",
...     sampleKey="sample", subjectKey="mouse", use_rep="X_joint")
>>>
>>> # both available: combine into one calibrated p-value per pair
>>> combined = la.tl.combineComparisons(lr_cmp, cmp_)
```

---

<a id="comparelarismatched"></a>

### `la.tl.compareLARISMatched`

```python
la.tl.compareLARISMatched(
    lr_data,
    conditionKey,
    referenceCondition,
    sampleKey,
    subjectKey=None,
    use_rep='X_pca',
    n_anchors=100,
    k_anchor=10,
    anchor_balance=True,
    logPseudocount='auto',
    minSubjectsObserved=2,
    fdrMethod='fdr_bh',
    random_seed=0,
    universe=None,
)
```

Compare conditions on cell-level LR scores at matched cell states.

Complements `compareLARIS`, which tests subject aggregates of
the cell-type-level results: this function works one level lower, on
the cells x LR-pairs score matrix from
`prepareLRInteraction`, and controls for differences in
cell-state composition between samples. Each subject's scores are
per-sample centred (the same scale-equivariant transform as
compareLARIS), evaluated at a common set of expression-space anchors
using the subject's own cells only, and the resulting per-subject
profiles are compared across conditions with the moderated t.

Because every subject is read at the same anchors, a condition that
merely has *more* of some cell state does not shift the comparison -
the estimand is the LR score *at matched states*. And because each
profile uses only its own subject's cells, profiles are independent
across subjects, which is what makes the between-subject test valid
(see the module docstring for the two designs that violated this and
failed their negative controls).

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `lr_data` | AnnData | Cell-level LR scores for ALL samples concatenated: the `lr_data` objects from per-sample `prepareLRInteraction` runs, stacked with matching `var_names`. `.obs` must carry `conditionKey` and `sampleKey` (and `subjectKey` if given); `.obsm[use_rep]` must hold a joint embedding computed across all samples (e.g. Harmony-corrected PCA of the expression data). Raw per-sample score scales do not matter - the per-sample centring removes them exactly. |
| `conditionKey` | str | `.obs` column with the condition label. |
| `referenceCondition` | str | The condition every other condition is compared against. |
| `sampleKey` | str | `.obs` column identifying the sample (section/puck) - the unit of the centring transform. |
| `subjectKey` | str, optional | `.obs` column identifying the biological subject - the unit of inference. Defaults to `sampleKey` (one sample per subject). Slices of one subject are pooled into one profile. |
| `use_rep` | str, default='X_pca' | `.obsm` key of the joint embedding used for anchor matching. |
| `n_anchors` | int, default=100 | Number of k-means anchors spanning the embedding. |
| `k_anchor` | int, default=10 | Cells of each subject averaged per anchor. Small values follow local structure; large values smooth toward the subject mean (`k_anchor -> n_cells` degenerates to the unstandardised mean). |
| `anchor_balance` | bool, default=True | Fit the anchor k-means on an equal number of cells from each condition. Without it the anchors follow pooled cell density, so the larger condition's states dominate the matched effect - which is what this estimator is meant to condition away. See `_anchor_points`. |
| `logPseudocount` | 'auto' or float, default='auto' | Passed to the per-sample centring; see `compareLARIS`. |
| `minSubjectsObserved` | int, default=2 | Minimum subjects per condition for a testable contrast. |
| `fdrMethod` | str, default='fdr_bh' | Multiple-testing correction, over the tests actually performed. |
| `random_seed` | int, default=0 | Seed for the k-means anchors. |
| `universe` | sequence of str, optional | Restrict testing and the FDR burden to these interaction names; the centring and profiles are still computed on the full data, so effect sizes do not change. See `compareLARIS`. |

**Returns**

- **`(comparison, profiles)` : tuple of pd.DataFrame**

  `comparison` has one row per (comparison, LR pair) with
  `log_diff` (difference of mean anchor-standardised profiles),
  `pvalue` (moderated t, or Fisher's exact on subject detection
  counts when one condition never detects the pair),
  `pvalue_fdr`, subject and detection counts, and
  `test_method`. `profiles` holds the per-subject standardised
  profiles (subjects x LR pairs) for inspection and plotting.

**Notes**

A subject whose cells do not cover part of the embedding still
answers at every anchor (its nearest cells are used, however far);
with strongly disjoint state distributions the profile leans on
extrapolation. Inspect `profiles` and the embedding overlap before
trusting fine effects in that situation.

---

<a id="combinecomparisons"></a>

### `la.tl.combineComparisons`

```python
la.tl.combineComparisons(
    comparison_a,
    comparison_b,
    on='interaction_name',
    method='cauchy',
    fdrMethod='fdr_bh',
    suffixes=('_a', '_b'),
)
```

Combine two comparison tables into one p-value per interaction.

Intended for the two `compareLARIS` estimators run on the same
cohort: the aggregate and matched estimators use the same data, so
their p-values are arbitrarily dependent - which is exactly the case
the **Cauchy combination test** (ACAT; Liu & Xie 2020) is valid for.
Each p is transformed to `tan((0.5 - p) * pi)` (standard Cauchy
under the null), the transforms are averaged, and the average is
mapped back to a p-value; the result is a valid p under arbitrary
dependence and is dominated by the smaller input, so an interaction
strongly supported by either estimator survives.

The combined null hypothesis is the *union*: "no change under either
estimand". A rejection means the interaction changed in at least one
of the two senses (in profile including composition, or at matched
states); the per-estimator columns retained in the output say which.

The merge is an inner join, so the combined table defines a single
shared universe and the FDR is computed over exactly that set -
resolving the different effective multiplicities of the two
estimators for the combined call.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `comparison_a`, `comparison_b` | pd.DataFrame | Tables with `on` and `pvalue` columns (any first element returned by `compareLARIS`). |
| `on` | str or list of str, default='interaction_name' | Merge key(s). Use `['interaction_name', 'comparison']` for multi-condition tables. |
| `method` | {'cauchy'}, default='cauchy' | Combination rule. Cauchy is the only one offered because it is the one that stays valid when the inputs share data. |
| `fdrMethod` | str, default='fdr_bh' | Correction applied to the combined p-values. |
| `suffixes` | tuple, default=('_a', '_b') | Suffixes for the per-estimator columns. |

**Returns**

- **pd.DataFrame**

  One row per shared interaction: the merge keys, per-estimator
  `pvalue`/`log_diff` columns, `concordant` (same effect
  sign), `pvalue_combined` and `pvalue_fdr`.

---

<a id="buildjointembedding"></a>

### `la.tl.buildJointEmbedding`

```python
la.tl.buildJointEmbedding(
    data,
    batch_key=None,
    method='harmony',
    groupby=None,
    layer=None,
    normalize=True,
    n_top_genes=2000,
    n_comps=30,
    key_added='X_joint',
    random_seed=0,
    verbosity=1,
    **method_kwargs,
)
```

Build a joint expression embedding across samples, in `obsm[key_added]`.

A convenience for `compareLARIS`'s matched estimator, which
needs one embedding computed over all samples together. Three
methods, all optional dependencies except the last:

- `'harmony'` (default): HVG -> scale -> PCA -> Harmony over
  `batch_key`. Needs `harmonypy`.
- `'gdr'`: PIASO's marker-gene-guided reduction
  (`piaso.tl.runGDR`), integration-free. Needs `piaso` and a
  `groupby` cluster column. On the Satb2 validation, harmony and GDR
  gave near-identical comparison results (Jaccard 0.66 on the hit
  set, same leaders), so this is preference, not correctness.
- `'pca'`: HVG -> scale -> PCA, no integration. The uncorrected
  baseline; on Satb2 it too matched the harmony hit set, but prefer a
  corrected method when batch may separate the *conditions* in
  expression space, which is the one situation the matching is
  sensitive to.

None of these packages is a hard dependency: an informative
ImportError tells you what to install only when you ask for a method
that needs it.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `data` | AnnData | Expression object with all samples concatenated (cells x genes). Counts or normalised - see `normalize`. |
| `batch_key` | str | `.obs` column identifying the sample/batch. Required. |
| `method` | {'harmony', 'gdr', 'pca'}, default='harmony' |  |
| `groupby` | str, optional | `'gdr'` only: `.obs` cluster/cell-type column that guides the reduction. |
| `layer` | str, optional | Expression layer to use; default `adata.X`. |
| `normalize` | bool, default=True | Apply `normalize_total` + `log1p` to a working copy first. Set False when the matrix is already normalised. Ignored by `'gdr'`, which applies PIASO's INFOG normalisation itself and expects raw counts. |
| `n_top_genes` | int, default=2000 | Highly-variable genes (batch-aware), for 'harmony' and 'pca'. |
| `n_comps` | int, default=30 | Embedding dimensionality, for 'harmony' and 'pca'. |
| `key_added` | str, default='X_joint' | `.obsm` key written on `data` (modified in place; also returned). |
| `random_seed` | int, default=0 | Seed of the PCA solver (`method='pca'` and the PCA that harmony corrects). Pass harmony's or GDR's own seeds through `method_kwargs`. |
| `verbosity` | int, default=1 | 1 prints progress; 0 is silent. |
| `adata` | deprecated | Former name of `data`. |
| `**method_kwargs` |  | Passed to `harmonypy.run_harmony` or `piaso.tl.runGDR`. |

**Returns**

- **AnnData**

  `data`, with `obsm[key_added]` added.

**Notes**

`method='gdr'` runs PIASO's parallel workers in a *spawn*
multiprocessing context: a driver **script** calling this must guard
its top level with `if __name__ == "__main__":` or the workers
re-execute the whole script. Notebooks are unaffected.
