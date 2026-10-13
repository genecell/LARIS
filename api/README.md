# LARIS API reference

<!-- Generated from the docstrings by tools/make_api_docs.py; do not edit by hand. -->

Every public function, generated from its docstring. For worked examples on real data see the [tutorials](../tutorials/README.md).

## [`laris.tl`](tl.md): Analysis

LR scores, the matched-gene background and exact P values, cross-condition comparison, decoy controls.

| Function | Summary |
|---|---|
| [`prepareLRInteraction`](tl.md#preparelrinteraction) | Calculate ligand-receptor interaction scores using spatial neighborhood information. |
| [`prepareLRBackground`](tl.md#preparelrbackground) | Build the matched-gene background for factorized null p-values. |
| [`permuteLRPairs`](tl.md#permutelrpairs) | Build a decoy LR database: the calibration control for this test. |
| [`decoyFDR`](tl.md#decoyfdr) | Empirical pairing-FDR for each target row, by target-decoy estimation. |
| [`computeDecoyFDR`](tl.md#computedecoyfdr) | One-call target-decoy FDR: build the decoy, run it, return q_decoy. |
| [`decoyReport`](tl.md#decoyreport) | Dataset-level summary of what the database contributes over chance. |
| [`LRBackground`](tl.md#lrbackground) | Product of `prepareLRBackground`; consumed by `runLARIS`. |
| [`runLARIS`](tl.md#runlaris) | Identify spatially-specific ligand-receptor interactions using LARIS algorithm. |
| [`compareLARIS`](tl.md#comparelaris) | Compare LARIS cell-cell communication results across conditions. |
| [`compareLARISMatched`](tl.md#comparelarismatched) | Compare conditions on cell-level LR scores at matched cell states. |
| [`combineComparisons`](tl.md#combinecomparisons) | Combine two comparison tables into one p-value per interaction. |
| [`buildJointEmbedding`](tl.md#buildjointembedding) | Build a joint expression embedding across samples, in `obsm[key_added]`. |
| `readCytome` | Same function as [`la.pp.readCytome`](pp.md#readcytome). |

## [`laris.pl`](pl.md): Plotting

Heatmaps, networks, dot plots, spatial maps and comparison volcanoes.

| Function | Summary |
|---|---|
| [`plotCCCHeatmap`](pl.md#plotcccheatmap) | Create a heatmap showing the number of cell-cell communication interactions. |
| [`plotCCCNetwork`](pl.md#plotcccnetwork) | Plot an interaction network for a specific cell type. |
| [`plotCCCNetworkCumulative`](pl.md#plotcccnetworkcumulative) | Plot a cumulative interaction network across all cell types. |
| [`plotCCCDotPlot`](pl.md#plotcccdotplot) | Create a bubble plot for selected cell type pairs and interactions. |
| [`plotCCCDotPlotFacet`](pl.md#plotcccdotplotfacet) | Create faceted bubble plots organized by sending or receiving cell type. |
| [`plotLRDotPlot`](pl.md#plotlrdotplot) | Create three side-by-side (or stacked) dot plots for LR pairs, ligands, and receptors. |
| [`plotCCCSpatial`](pl.md#plotcccspatial) | Plot spatial distribution of ligand-receptor interactions. |
| [`plotCompareLARIS`](pl.md#plotcomparelaris) | Volcano plot of a `laris.tl.compareLARIS` result. |
| [`prepareDotPlotAdata`](pl.md#preparedotplotadata) | Prepare combined AnnData for dot plot visualizations. |

## [`laris.pp`](pp.md): Reading data

Cytome readers and multi-sample coordinate offsets.

| Function | Summary |
|---|---|
| [`readCytome`](pp.md#readcytome) | Read a cytome dataset (optionally a gene subset) into an AnnData. |
| [`readLRCytome`](pp.md#readlrcytome) | Read an LR-score cytome (written by `prepareLRInteraction`). |
| [`spatialOffsetMultisample`](pp.md#spatialoffsetmultisample) | Offset spatial coordinates so that kNN stays within each sample. |

## [`laris.datasets`](datasets.md): Datasets

The bundled ligand-receptor database.

| Function | Summary |
|---|---|
| [`lrDatabase`](datasets.md#lrdatabase) | Load a bundled ligand-receptor interaction database. |
