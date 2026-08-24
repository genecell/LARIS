# Changelog

## v0.10.0 (2026-08-22, unreleased)

Deliberate behavior-changing release: defaults now match the recommended /
published settings, so results differ from v0.9.x. Each change lists how to
reproduce the old behavior. The tonsil tutorial reference values will be
re-generated with these defaults.

### Changed defaults (results-changing)
- `runLARIS(mu=)` default 1 -> **0.25** (the manuscript value). The parameter
  scales the null subtraction `score = cos_observed - mu * cos_shuffled`.
  Old behavior: `mu=1`. The docstring now also distinguishes `mu` from the
  unrelated COSG regularizer `mu_celltype`.
- `runLARIS(sigma=)` default fixed 100 -> **'adaptive'** (half the mean k-NN
  edge distance). This makes the spatial-specificity kernel independent of
  coordinate units and platform spot spacing, and consistent with the kernel
  already used by `prepareLRInteraction` and the cell-type co-localization
  step. Old behavior: `sigma=100` (assumed roughly-micrometre coordinates).
- `runLARIS(n_nearest_neighbors=)`, `runLARIS(number_nearest_neighbors=)` and
  `prepareLRInteraction(number_nearest_neighbors=)` defaults 10 -> **20**,
  matching the tutorial/paper settings and unifying all three neighbourhood
  sizes. The diffusion step in `prepareLRInteraction` kept a k=10 default that
  matched neither the tutorial nor the published reference values, so anyone
  relying on defaults got a different diffusion from the documented one.
  Old behavior: pass 10.
- `runLARIS(spatial_weight=)` default 1.0 -> **3.0** (the generally
  recommended setting, used by the tutorial). Old behavior: pass 1.0.

### Fixed
- **Cytome embeddings lost their scanpy `X_` prefix.**
  `cytome.from_anndata` stores `obsm['X_spatial']` as `RNA_spatial` and
  its own reader restores the prefix, but `readCytome` stripped only the
  modality prefix and returned `obsm['spatial']` - so LARIS's own default
  `use_rep_spatial='X_spatial'` raised `KeyError` on any dataset with a
  single spatial key. Both spellings now resolve to the same array. (The
  test fixture set both `obsm['spatial']` and `obsm['X_spatial']`, which
  is why the suite did not catch it.)
- `runLARIS` no longer raises `AttributeError` on `.cat` when the cell
  type column holds plain strings rather than a pandas categorical -
  which is what a cytome round-trip returns, since SQLite has no
  categorical type.

### Fixed (results-changing for affected pairs)
- Negative spatial specificity scores are clamped to 0 before the
  `spatial_weight` power is applied. Previously, fractional weights produced
  NaN and even weights silently flipped the sign, ranking anti-correlated
  pairs as interacting. Affected pairs now contribute an interaction score
  of 0.

### Added
- **Native cytome support, end to end.** Every entry point accepts a
  `.cytome`/`.db` path or an open `cytome.CytomeDataset`, and
  `prepareLRInteraction` now returns whichever type it was given
  (`return_type='auto'`): a cytome input produces an **LR cytome** rather
  than an in-memory score matrix, which is the case a cytome input exists
  to avoid. The LR cytome reuses the registered `RNA` modality with the
  ligand-receptor pairs as its feature table - so every cytome reader
  works on it without a new modality registration - and holds a single
  `RNA_lrscore` layer and no counts, because there are none. Read it with
  the new `laris.pp.readLRCytome`, or pass it straight to `runLARIS` and
  the plots. `return_type='anndata'`/`'cytome'`, `output=` and
  `overwrite=` control it explicitly.
- **Streaming cell-type specificity.** The expression object is the only
  thing the cell-type step reads, and for a cytome source it is now
  streamed from disk by `cosg.run_cosg_cytome`; no cells x LR-genes
  AnnData is built. Results match the in-memory path to float32 precision
  (~3e-8, verified end to end on the tonsil sample across all 389,060
  sender-receiver-interaction rows). `cosg_backend='memory'` forces the
  previous behavior; `'auto'` streams only for cytome sources, so AnnData
  users are unaffected.
- **`specificity_reference=`** on `runLARIS` (`'lr'` default, `'all'`
  opt-in): which genes set the scale when cell type specificity is
  normalized. `cosg.iqrLogNormalize` divides each cell type's scores by
  that cell type's `q0.95 - q0.75` spread over the genes it is given, so
  the reference set changes the answer. `'lr'` is exactly invariant to
  which *other* genes are in the object; `'all'` is exactly invariant to
  which LR database is used. Rankings within a cell type are identical
  either way (the transform is monotone); the relative scale between cell
  types, and hence the ranking of sender-receiver pairs, is what moves.
  A warning fires when fewer than 100 database genes are present, as on
  targeted panels, where the LR-only spread is unstable.
- **`block_size=`** on `prepareLRInteraction`: diffuse and score cells in
  blocks, bounding peak memory. Column *j* of the diffused matrix depends
  only on row *j* of the k-NN graph, so this is an exact decomposition -
  the block size never changes the result.

### Changed (no effect on results)
- **`prepareLRInteraction` no longer diffuses genes it does not use.** It
  diffused the whole transcriptome and then kept only the
  ligand/receptor rows: on the 25,583-gene tonsil sample, 25,583 genes
  were diffused so that 1,028 could be used. Subsetting first cuts that
  run from 1.64 s / 811 MB to 0.35 s / 166 MB, or 72 MB with
  `block_size=2000`, with **bitwise identical** output.
- **Polymorphic arguments renamed to `data=` / `lr_data=`**, matching
  PIASO: `prepareLRInteraction(data=)`, `runLARIS(lr_data=, data=)`,
  `prepareDotPlotAdata(lr_data=, data=)`, `plotCCCSpatial(lr_data=,
  data=)`, `plotCCCNetwork(data=)`, `plotCCCNetworkCumulative(data=)`.
  They accept an AnnData, a cytome dataset or a path, so naming them for
  one of those types was misleading. The old `adata=` / `lr_adata=`
  keywords keep working with a `FutureWarning`; positional calls are
  unaffected.

### Added
- **Tissue images from cytome datasets** (cytome >= 0.2.6): a cytome's
  `spatial_images` store joins the image waterfall, so
  `plotCCCSpatial(..., color_by='score')` overlays H&E / ssDNA read
  straight out of the `.cytome` file. `readCytome` copies the images into
  `uns['spatial']` and `prepareLRInteraction` propagates them into
  `lr_adata`, so no extra argument is needed. Missing optional decoders
  (PIL / tifffile / imagecodecs) degrade to an imageless plot with a
  warning. New `library_key=` narrows library selection to the plotted
  cells; ambiguous multi-library requests raise instead of guessing.
- **cytome input support across the pipeline**: `prepareLRInteraction` and
  `runLARIS(adata=...)` accept a `.cytome` path or an open
  `cytome.CytomeDataset` anywhere an expression AnnData was expected; only
  the ligand/receptor gene subset is streamed from disk, and results are
  identical to the AnnData path by construction. New public helper
  `laris.tl.readCytome`. Install with `pip install laris[cytome]`.
- **Tissue-image overlay in `laris.pl.plotCCCSpatial`** (GitHub issue #1):
  `color_by='score'` renders the continuous per-spot interaction score, and
  the image parameters (`img=` + `scale_factor=`, or the scanpy
  `uns['spatial']` convention via `library_id`/`img_key`) put H&E or ssDNA
  under either color mode - including the existing cell-type highlighting.
  One function, no separate overlay entry point. A missing interaction now
  raises `ValueError` instead of printing and returning None.
- **`laris.tl.compareLARIS`**: cross-condition comparison of LARIS results.
  Method (validated by calibration simulation and on two real
  multi-condition datasets): per-sample median-centred log scores (exactly
  invariant to per-sample scale factors), subject-level aggregation over
  cell-type pairs and technical-replicate slices (the subject is the unit
  of inference), and an empirical-Bayes moderated t-test. Includes
  `level={'both','lr','triple'}` with a fast per-LR path.
- `sigma` parameter on `prepareLRInteraction` and `sigma_celltype` on
  `runLARIS` ('adaptive' default preserves those steps' previous behavior;
  numeric values give an absolute bandwidth).


### Fixed (cytome compatibility)
- Cytome 0.2.6 renamed obsm embeddings from `{modality}_obsm_{key}` to
  `{modality}_{key}`; `readCytome` accepted only the old form and silently
  dropped every embedding — including spatial coordinates — from
  0.2.6-written files. Both generations now resolve.
- The adaptive kernel no longer produces NaN weights when all spatial
  coordinates are identical (zero bandwidth).

## v0.9.4 (2026-08-22)

Bugfix release. Default results are unchanged: the tonsil tutorial reproduces
value-for-value against v0.9.3.

### Fixed
- `prepareLRInteraction` no longer silently maps ligand/receptor names absent
  from `adata.var_names` onto a neighbouring gene (an `np.searchsorted`
  insertion-position artifact). Unmatched pairs are now dropped with a
  `UserWarning` listing example missing names; pass `unmatched='error'` to
  raise instead (recommended for custom databases).
- `runLARIS` no longer crashes with `IndexError` when fewer than 100 LR pairs
  are analyzed (`sc.pp.calculate_qc_metrics` `percent_top` is now clamped to
  the number of features).
- `import laris` works with matplotlib >= 3.9 (removed module-level
  `cm.get_cmap` calls; a version-compatible lookup is used throughout).
- `networkx` is now declared as a dependency (it was imported but undeclared,
  breaking clean installs).
- Pinned `cosg>=1.0.3,!=1.1.0,!=1.1.1`: cosg 1.1.0/1.1.1 raise ImportError on
  plain AnnData input when the optional cytome dependency is missing.
- Replaced chained `fillna(..., inplace=True)` (silently broken under
  pandas 3.0) with direct assignment.
- `plotCCCDotPlot` raises `ValueError` on invalid sender/receiver input
  instead of printing and returning None (silent failure in batch scripts).
  The mismatch message points to `plotCCCDotPlotFacet` for the
  Cartesian-product case.
- `plotCCCNetwork` / `plotCCCNetworkCumulative`: `cell_type_color_key` now
  defaults to `f"{groupby}_colors"` (scanpy convention) with a generated
  palette fallback instead of `KeyError`; stored palettes are paired with
  categorical categories order (previously appearance order, which could
  mis-assign colors).
- Removed the spurious "No genes or groups specified" warning emitted by an
  internal call on every `runLARIS` run.

### Added
- The interaction-score scaling factor (top-100 mean pinned to 0.1) is now
  recorded in `lr_adata.uns['laris_scale_factor']` and
  `celltype_results.attrs['laris_scale_factor']`, and rescaling can be
  disabled with `runLARIS(..., rescale=False)`. This lets separate runs be
  put back on a common scale when merging into one object is impractical.
- `n_cells_expressed_threshold` accepts a float in (0, 1) as a fraction of
  the number of cells (values >= 1 remain absolute counts).
- `runLARIS` warns when sender-receiver groups have a minimum achievable FDR
  above 0.05 given `n_permutations` (the permutation p-value floor of
  1/(n_permutations+1) times the group size), with guidance on choosing
  `n_permutations`.
