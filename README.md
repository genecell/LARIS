[![Stars](https://img.shields.io/github/stars/genecell/LARIS?logo=GitHub&color=yellow)](https://github.com/genecell/LARIS/stargazers)
[![PyPI](https://img.shields.io/pypi/v/laris?logo=PyPI)](https://pypi.org/project/laris)
[![Total downloads](https://static.pepy.tech/personalized-badge/laris?period=total&units=international_system&left_color=black&right_color=orange&left_text=downloads)](https://pepy.tech/project/laris)
[![Monthly downloads](https://static.pepy.tech/personalized-badge/laris?period=month&units=international_system&left_color=black&right_color=orange&left_text=downloads/month)](https://pepy.tech/project/laris)


# LARIS

##### Ligand And Receptor Interaction in Spatial transcriptomics data

LARIS is a Python package for analyzing ligand-receptor interactions in spatial transcriptomics data. It identifies spatially-specific cell-cell communication patterns by integrating gene expression, spatial information, and cell type annotations.

### Feature

- **Spatial LR interaction strength**: Calculate ligand-receptor interaction scores for each individual cell
- **Spatial specificity**: Identify LR pairs with significant spatial variable patterns
- **Inference at cell type level**: Compute sender-receiver cell type interaction scores
- **Spatial neighborhoods**: Analyze interactions in the context of spatial cell type neighborhoods

### Documentation

[LARIS documentation](https://genecell.github.io/LARIS/)

[Datasets for LARIS tutorial](https://drive.google.com/drive/folders/1cfj4IZxl5svnG4RO--Fcr2x-bmNokmQN)


### Installation

You could simply install LARIS via `pip` in your conda environment:
```bash
pip install laris
```

For the development version in GitHub, you could install via:
```bash
pip install git+https://github.com/genecell/LARIS.git
```

### Citation

If LARIS is useful for your research, please consider citing [M. Dai, T. Török, D. Sun, et al., LARIS enables accurate and efficient ligand and receptor interaction analysis in spatial transcriptomics, bioRxiv (2025)](https://www.biorxiv.org/content/10.1101/2025.11.26.690796v1). 


