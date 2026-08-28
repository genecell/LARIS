# LARIS tutorials

Markdown tutorials with executed code and embedded output figures — no
website build needed, readable on GitHub, in an editor, or by a coding
agent. Every output shown was produced by the code above it, on the named
public dataset, with the LARIS version stated in each file.

Written for **LARIS v0.12.0**. If you are reporting p-values, read
[tutorial 07](07_significance_and_background.md) — significance testing
changed in this release.

| Tutorial | Dataset | Covers |
|---|---|---|
| [01 — Core pipeline](01_laris_slidetags_tonsil.md) | Slide-tags human tonsil (Zenodo [10.5281/zenodo.19981287](https://doi.org/10.5281/zenodo.19981287)) | `prepareLRInteraction`, `runLARIS`, spatial / dot / network plots, defaults & reference values |
| [02 — Compare conditions (aggregate)](02_comparelaris_visium_mi.md) | Kuppe MI Visium (cellxgene) | `compareLARIS` from per-sample results tables, subjects vs slices, Fisher detection route, volcano |
| [03 — Compare at matched cell states](03_comparelaris_matched_merfish_gut.md) | MERFISH gut atlas (Dryad [10.5061/dryad.p5hqbzm0z](https://doi.org/10.5061/dryad.p5hqbzm0z)) | matched estimator, `buildJointEmbedding` (harmony / PIASO GDR / PCA), per-subject profiles, `combineComparisons` |
| [04 — Cytome guide](04_cytome_guide.md) | tonsil | the on-disk workflow end to end: LR cytome, streaming COSG, bit-identical guarantee |
| [05 — Tissue image overlay](05_tissue_image_overlay.md) | Xenium Prime 5K human skin (10x) | H&E overlay at single-cell resolution, registering a post-hoc image, 5K-panel LR analysis |
| [06 — Two-variable analysis](06_two_variable_crossed_labels.md) | MERFISH gut atlas | crossed cell-type x region labels (PIASO `getCrossCategories`), region-resolved comparison |
| [07 — Statistical significance](07_significance_and_background.md) | Slide-tags human tonsil | `prepareLRBackground`, what the p-value tests, choosing `n_matched_genes`, reusing one background across analyses |

For the steps upstream of LARIS on spatial data, QC, clustering,
annotation, RNA regulon analysis and visualization, see PIASO's
[spatial tutorials](https://piaso.org/tutorials/spatial-xenium/).

LARIS is part of the [PIASO](https://github.com/genecell/PIASO)
ecosystem — the tutorials use PIASO's figure styling and mention its
INFOG/GDR options where they fit — but **PIASO is not a required
dependency**: every LARIS step runs without it, and optional packages
raise an informative install pointer only when a feature that needs them
is requested.
