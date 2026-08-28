# Statistical significance: what the p-value tests, and how to get one

`runLARIS` ranks interactions by score whether or not you ask for
p-values. This tutorial is about the p-value: what question it answers,
how to compute one, and how to read the result honestly. All outputs
were produced by exactly this code with LARIS v0.12.0 on the Slide-tags
tonsil (Zenodo
[10.5281/zenodo.19981287](https://doi.org/10.5281/zenodo.19981287)).

## 1. The two-line version

```python
bg = la.tl.prepareLRBackground(adata, lr_df, use_rep_spatial="X_spatial")
laris_lr, res = la.tl.runLARIS(lr_data, adata,
                               use_rep="X_spatial",
                               use_rep_spatial="X_spatial",
                               groupby="cell_type",
                               background=bg)          # <- p-values
```

`prepareLRBackground` builds the reference the test compares against.
It is a separate step because it is the expensive part and it is
**reusable**: it depends only on the cells, the spatial graph and the
gene set, so one background serves every `groupby`, every parameter
sweep, and every re-run on the same object.

## 2. What the p-value actually tests

For each (sender, receiver, ligand::receptor) row the question is:

> Is this ligand-receptor pair more strongly and specifically
> co-organized between these two cell types than an
> **expression-matched pair of genes** would be?

The null is built by replacing the ligand with one of its
mean/variance-matched genes and the receptor with one of its own —
independently, so `n_matched_genes=100` per side gives **10,000
pseudo-pairs per real pair**. Each pseudo-pair is scored through the
*whole* pipeline (its own spatial specificity, its own detection
fractions, its own co-localization), and the p-value is the exact tail:

```
p = (# pseudo-pairs scoring at least as high + 1) / (# pseudo-pairs + 1)
```

Three consequences worth internalising:

- **It is exact and deterministic.** No sampling, no seed, no
  `n_permutations`. Re-running gives the same number.
- **The floor is 1/10,001 ≈ 1e-4** at defaults. A p-value equal to the
  floor means "beat every pseudo-pair", not "p = 0".
- **This is not a test of "is there any signal".** It is competitive:
  against expression-matched chance. A highly expressed, broadly
  co-expressed pair can score high and still be unremarkable *for its
  expression level* — that is the intended behaviour.

## 3. Running it, with real output

```python
import laris as la
import scanpy as sc

adata = sc.read_h5ad("adata_tonsil.h5ad")
lr_df = la.datasets.lrDatabase(species="human")
lr_df = lr_df[lr_df.ligand.isin(adata.var_names)
              & lr_df.receptor.isin(adata.var_names)]

lr_data = la.tl.prepareLRInteraction(adata, lr_df,
                                     use_rep_spatial="X_spatial")
bg = la.tl.prepareLRBackground(adata, lr_df, use_rep_spatial="X_spatial")
```

```text
  - candidate pool: 4,000 genes (quantile grid over mean/variance)
  - diffusing 4,718 genes through the 20-NN kernel graph...
  - building spatial-specificity graph and Gram tables...
  - done: 4,718 genes, support per pair 10,000 (floor 1.00e-04)
```

The background took about 5 minutes on this 5,695-cell dataset and is
the one-time cost; keep it in memory (or `pickle` it) and reuse.

![matched genes](images/tut07_matched_null.png)

This is what "expression-matched" means, for one pair. C3 and CR2
(stars) each receive 100 partners drawn from the candidate pool at the
same place in mean/variance space - the tight clouds around each star.
Every one of the 10,000 pseudo-pairs for `C3::CR2` is one blue gene
crossed with one orange gene, so the comparison controls for expression
level and variability instead of pitting the pair against arbitrary
genes.

Two things follow, and both matter when reading a result. Because the
matching is per gene, a *highly expressed* ligand is judged against
other highly expressed genes, so being abundant earns no significance
by itself. And because the two sides are drawn independently, the null
keeps each gene's own behaviour while destroying the pairing - which is
exactly the alternative hypothesis being tested.

```python
laris_lr, res = la.tl.runLARIS(lr_data, adata,
                               use_rep="X_spatial",
                               use_rep_spatial="X_spatial",
                               groupby="cell_type", background=bg)

res[res.p_value_fdr < 0.01].nsmallest(6, "p_value")[
    ["sender", "receiver", "interaction_name",
     "interaction_score", "p_value", "p_value_fdr"]]
```

```text
        sender             receiver interaction_name  interaction_score   p_value  p_value_fdr
      B_memory                  mDC   SEMA7A::PLXNC1             0.1373  0.000100       0.0053
  T_double_neg                  pDC    CD96::NECTIN1             0.0604  0.000100       0.0034
           MRC             FDC_LZDZ           C3::CR2             0.0371  0.000100       0.0084
      B_memory                T_CD8    COL4A3::ITGA1             0.0316  0.000100       0.0021
      FDC_LZDZ                  mDC       FN1::ITGA4             0.0297  0.000100       0.0028
    macrophage T_follicular_helper  PDCD1LG2::PDCD1             0.0231  0.000100       0.0056
```

`C3::CR2` from marginal reticular cells to follicular dendritic cells is
the germinal-centre antigen-trapping axis; `PDCD1LG2::PDCD1` from
macrophages to follicular helper T cells is the PD-1 checkpoint. Neither
was supplied as prior knowledge.

![p-values](images/tut07_pvalues.png)

**Left**: every p-value is an exact tail count, so the distribution
stops at the floor and nothing lies below it. **Right**: the same raw
p-value maps to different FDRs, because Benjamini-Hochberg runs *within*
each sender-receiver pair - a p-value of 1e-4 in a pair with few tested
interactions clears a stricter threshold than the same p-value in a
crowded one.

## 4. Choosing `n_matched_genes`

This is the only parameter most users need to think about, and it sets
the resolution of the test:

| `n_matched_genes` | pseudo-pairs per real pair | p-value floor |
|---|---|---|
| 30 | 900 | 1.1e-3 |
| **100 (default)** | **10,000** | **1.0e-4** |
| 200 | 40,000 | 2.5e-5 |

Raise it only if you need to *report* thresholds below the floor. The
cost is in the assembly, not the background build, and matching quality
degrades slowly in gene space (the 100th-nearest gene by mean/variance
is still a close match; the 100th-nearest *pair* would not be).

![n_matched_genes](images/tut07_n_matched.png)

Measured on the tonsil data, this is what the parameter does and does
not change:

| `n_matched_genes` | floor | FDR < 0.05 | FDR < 0.01 |
|---|---|---|---|
| 10 | 9.9e-3 | 886 | **0** |
| 30 | 1.1e-3 | 1,352 | 138 |
| 100 (default) | 1.0e-4 | 1,630 | 142 |

At `k = 10` the floor is 1/101, so **no interaction can reach FDR < 0.01
at all** - the zero in that row is arithmetic, not biology. By `k = 30`
the strict column is essentially at its final value (138 vs 142), and
the remaining gain from 30 to 100 is resolution for reporting rather
than new discoveries. The default is chosen to leave room below the
thresholds people actually quote.

`n_pool` (default 4,000) sets how many genes the matched sets are drawn
from. The Gram tables are quadratic in the pool, so this is the memory
and build-time dial; the default is a good balance and covers the
mean/variance space on a quantile grid.

## 5. Reading the results honestly

**Use the FDR column, not the raw p.** `p_value_fdr` is
Benjamini-Hochberg *within each sender-receiver pair*, because that is
the unit a result is reported in ("cell type A signals to cell type B
via L::R"). Different cell-type pairs have different specificity
profiles, so per-pair correction keeps each pair's threshold driven by
its own p-value distribution.

**The floor and the FDR interact.** In a group of `m` tested
interactions where `k` reach the p-value floor, the smallest attainable
FDR is `(m/k) x floor`. With the default floor of 1e-4 this is rarely
binding, but LARIS warns when a group's isolated-hit bound exceeds
0.05.

**`p_value = 1.0` is meaningful**, not missing: it is what the test
returns when the interaction's matched-gene support has no positive
mass, i.e. there is no evidence either way.

**Neither label nor coordinate permutation is this test's null.** If
you shuffle cell-type labels or cell positions and still get calls,
that is expected: real database LR pairs are genuinely co-expressed
genes, and the test asks about expression-matched chance, which
survives those permutations. The calibration check that *is* meaningful
is a synthetic dataset with no planted structure, which returns nothing.

## 6. Without a background

`runLARIS(..., calculate_pvalues=True)` without `background=` still
runs, using the older resampled-pair null. It is kept for one release
so existing scripts do not break, but its p-values have a coarse
support and should not be used for new work — see the v0.12.0 release
notes.

If you only need a ranking, `calculate_pvalues=False` skips the whole
question and is much faster.

## Notes

- **Reuse**: `bg` is independent of `groupby`, `mu` and
  `spatial_weight`. Building it once and passing it to several
  `runLARIS` calls (different cell-type annotations, parameter sweeps)
  is the intended pattern.
- **Cytome**: `prepareLRBackground` accepts a `.cytome` path or open
  dataset, and streams the gene statistics rather than loading the full
  matrix.
- **Sections**: pass the same `section_key` you pass to
  `prepareLRInteraction`, so the background's graphs respect the same
  section boundaries.
- **Speed**: wheels ship compiled Rust kernels used automatically. Set
  `LARIS_NO_RUST=1` to force the pure-NumPy path (identical results).
