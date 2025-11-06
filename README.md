# LARIS: Ligand And Receptor Interaction in Spatial transcriptomics data


LARIS is a Python package for analyzing ligand-receptor interactions in spatial transcriptomics data. It identifies spatially-specific cell-cell communication patterns by integrating gene expression, spatial information, and cell type annotations.

### Features

- **Spatial LR interaction strength**: Calculate ligand-receptor interaction scores using spatial adjacency information
- **Spatial specificity**: Identify LR pairs with significant spatial variable patterns
- **Inference at cell type level**: Compute sender-receiver cell type interaction scores
- **Spatial neighborhoods**: Analyze interactions in the context of spatial cell type neighborhoods

### 📦 Installation

For the development version in GitHub, you could install via:
```bash
pip install git+https://github.com/genecell/LARIS.git
```

You could simply install LARIS via `pip` in your conda environment (future):
```bash
pip install laris
```

### Quick Start

```python
import laris as la
import scanpy as sc
import pandas as pd

# Load your spatial transcriptomics data
adata = sc.read_h5ad('spatial_data.h5ad')

# Define ligand-receptor pairs
lr_df = pd.DataFrame({
    'ligand': ['Tgfb1', 'Vegfa', 'Cxcl12'],
    'receptor': ['Tgfbr1', 'Kdr', 'Cxcr4']
})

# Step 1: Calculate LR integration scores
lr_adata = la.tl.prepareLRInteraction(
    adata, 
    lr_df,
    number_nearest_neighbors=15,
    use_rep_spatial='X_spatial'
)

# Step 2: Identify spatially-specific LR interactions and infer the LR interaction at cell type level
laris_results, celltype_results = la.tl.runLARIS(
    lr_adata,
    adata,
    n_nearest_neighbors=15,
    n_repeats=100,
    n_top_lr=1000,
    by_celltype=True,
    groupby='cell_type'
)

# View top spatially-specific LR pairs
print(laris_results.head(10))

# View top cell type-specific interactions
print(celltype_results.head(10))

# Filter for specific cell type pairs
endothelial_to_tumor = celltype_results[
    (celltype_results['sending_celltype'] == 'Endothelial') &
    (celltype_results['receiving_celltype'] == 'Tumor')
]
print(endothelial_to_tumor.head(10))
```

