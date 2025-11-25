"""
LARIS Tools Utilities Module (laris.tl._utils)

Internal utility functions for LARIS analysis tools.

This module contains helper functions used internally by the main tools module:
- Matrix similarity calculations
- Data selection and ranking utilities
- Adjacency matrix construction
- Random background generation
- Cell type specificity calculations

These functions are prefixed with underscore to indicate they are primarily
intended for internal use, though advanced users may access them if needed.
"""

import pandas as pd
import scanpy as sc
import numpy as np
import anndata as ad
import scipy.sparse as sp
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import cosg
from typing import Optional, Union, List, Tuple
from tqdm.auto import tqdm
from statsmodels.stats.multitest import multipletests
from sklearn.neighbors import KDTree


def _rowwise_cosine_similarity(
    A: sp.csr_matrix,
    B: sp.csr_matrix
) -> np.ndarray:
    """
    Compute the cosine similarity between corresponding rows of matrices A and B.
    
    This function efficiently calculates row-wise cosine similarity for both
    dense and sparse matrices. It handles zero-norm rows gracefully by returning
    zero similarity for those cases.

    Parameters
    ----------
    A : np.ndarray or scipy.sparse.csr_matrix
        A 2D array or CSR sparse matrix with shape (n, m).
    B : np.ndarray or scipy.sparse.csr_matrix
        A 2D array or CSR sparse matrix with shape (n, m).

    Returns
    -------
    np.ndarray
        A 1D array of shape (n,) containing the cosine similarity between
        corresponding rows of A and B.
        
    Raises
    ------
    ValueError
        If matrices A and B have different shapes.

    Examples
    --------
    >>> import numpy as np
    >>> import laris as la
    >>> 
    >>> # Dense matrices
    >>> A = np.array([[1, 2, 3], [4, 5, 6]])
    >>> B = np.array([[7, 8, 9], [1, 2, 3]])
    >>> similarities = la.tl._utils._rowwise_cosine_similarity(A, B)
    >>> print(similarities)
    
    >>> # Sparse matrices
    >>> from scipy.sparse import csr_matrix
    >>> A_sparse = csr_matrix(A)
    >>> B_sparse = csr_matrix(B)
    >>> similarities_sparse = la.tl._utils._rowwise_cosine_similarity(A_sparse, B_sparse)
    

    """
    if A.shape != B.shape:
        raise ValueError(f"Matrices A and B must have the same shape. Got A.shape = {A.shape}, B.shape = {B.shape}.")

    if issparse(A) and issparse(B):
        dot_products = A.multiply(B).sum(axis=1).A1  # A1 converts matrix to a flat array
        norm_A = np.sqrt(A.multiply(A).sum(axis=1)).A1
        norm_B = np.sqrt(B.multiply(B).sum(axis=1)).A1
    else:
        dot_products = np.einsum('ij,ij->i', A, B)  # Faster than (A * B).sum(axis=1)
        norm_A = np.linalg.norm(A, axis=1)
        norm_B = np.linalg.norm(B, axis=1)
    
    denominator = norm_A * norm_B
    cosine_similarities = np.divide(
        dot_products, 
        denominator, 
        out=np.zeros_like(dot_products), 
        where=denominator != 0
    )
    
    return cosine_similarities


def _select_top_n(
    scores: np.ndarray, 
    n_top: int
) -> np.ndarray:
    """
    Select indices of top n highest-scoring elements.
    
    This is an internal helper function that efficiently selects the top n
    elements from an array using partial sorting (argpartition), which is
    faster than full sorting for large arrays.

    Parameters
    ----------
    scores : np.ndarray
        1D array of scores.
    n_top : int
        Number of top elements to select.

    Returns
    -------
    np.ndarray
        Array of indices corresponding to the top n scores, sorted in
        descending order by score.

    Examples
    --------
    >>> import numpy as np
    >>> import laris as la
    >>> 
    >>> scores = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
    >>> top_indices = la.tl._utils._select_top_n(scores, 3)
    >>> print(top_indices)  # [1, 3, 4] (indices of 0.9, 0.7, 0.5)
    

    """
    reference_indices = np.arange(scores.shape[0], dtype=int)
    partition = np.argpartition(scores, -n_top)[-n_top:]
    partial_indices = np.argsort(scores[partition])[::-1]
    global_indices = reference_indices[partition][partial_indices]
    return global_indices


def _pairwise_row_multiply(
    sparse_matrix: sp.csr_matrix,
    cell_types: List[str],
    delimiter: str = "::"
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Calculate element-wise multiplication between all pairs of rows in a sparse matrix.
    
    This helper function computes pairwise products between all rows of a matrix,
    including self-interactions. This is useful for modeling sender-receiver
    cell type interactions in spatial transcriptomics.

    Parameters
    ----------
    sparse_matrix : scipy.sparse.csr_matrix
        Input sparse matrix of shape (N, M) where N is the number of cell types.
    cell_types : List[str]
        Array of cell type names, length N.
    delimiter : str, default="::"
        Delimiter to use when joining cell type pair names.

    Returns
    -------
    result : scipy.sparse.csr_matrix
        Sparse matrix of shape (N*N, M) containing all pairwise multiplications.
    row_names : np.ndarray
        Array of strings representing the cell type pairs for each row,
        in the format "cell_type_i::cell_type_j".

    Raises
    ------
    ValueError
        If length of cell_types doesn't match number of rows in sparse_matrix.
    TypeError
        If sparse_matrix is not a sparse matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> import laris as la
    >>> 
    >>> # Create a simple sparse matrix (3 cell types × 5 cells)
    >>> data = csr_matrix([[1, 0, 2, 0, 1],
    ...                    [0, 3, 0, 1, 0],
    ...                    [2, 0, 1, 0, 3]])
    >>> cell_types = ['TypeA', 'TypeB', 'TypeC']
    >>> 
    >>> # Compute pairwise products
    >>> result, names = la.tl._utils._pairwise_row_multiply(data, cell_types)
    >>> print(result.shape)  # (9, 5) for 3×3 pairs
    >>> print(names[:3])     # ['TypeA::TypeA', 'TypeA::TypeB', 'TypeA::TypeC']
    
    Notes
    -----
    This function is used internally to model sender-receiver cell type
    interactions. The pairwise products capture the joint signal patterns
    between different cell types in spatial neighborhoods.
    
    For N cell types, this generates N² combinations including self-interactions
    (TypeA::TypeA, TypeA::TypeB, etc.).
    """
    N, M = sparse_matrix.shape

    # Validate input
    if len(cell_types) != N:
        raise ValueError(f"Length of cell_types ({len(cell_types)}) must match number of rows in matrix ({N})")

    # Ensure the input is in CSR format for efficient row slicing
    if not sp.isspmatrix_csr(sparse_matrix):
        sparse_matrix = sparse_matrix.tocsr()

    # Preallocate lists to store all the row products and row names
    row_products = []
    row_names = []

    # Iterate through all pairs of rows (including self-pairs)
    for i in range(N):
        row_i = sparse_matrix[i]
        cell_type_i = cell_types[i]

        for j in range(N):  # Include all pairs, even when i == j
            row_j = sparse_matrix[j]
            cell_type_j = cell_types[j]

            # Element-wise multiplication
            product = row_i.multiply(row_j)
            row_products.append(product)

            # Create row name for this combination
            row_name = f"{cell_type_i}{delimiter}{cell_type_j}"
            row_names.append(row_name)

    # Vertically stack all multiplied pairs
    result = sp.vstack(row_products)

    # Convert row names to numpy array
    row_names = np.array(row_names)

    return result, row_names


def _compute_avg_expression(
    adata: ad.AnnData, 
    groupby: str = 'Leiden', 
    genes: Optional[List[str]] = None, 
    groups: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute the average expression levels of genes across cell groups.

    This function calculates the mean expression values of genes for each group 
    of cells defined by `groupby`. It allows selection of specific genes and/or 
    groups for computation.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing gene expression data.
    groupby : str, default='Leiden'
        The column name in `adata.obs` that defines the cell groups. 
        Typically, this is a clustering column (e.g., 'Leiden', 'louvain', or 'cell_type').
    genes : List[str], optional
        A list of gene names to compute the average expression for. 
        If None, all genes in `adata.var_names` are included.
    groups : List[str], optional
        A list of group labels to compute the average expression for. 
        If None, all groups present in `adata.obs[groupby]` are included.

    Returns
    -------
    pd.DataFrame
        A DataFrame where:
        - Rows correspond to groups (e.g., clusters or cell types).
        - Columns correspond to genes.
        - Values represent the mean expression of each gene in the respective group.

    Raises
    ------
    ValueError
        If `groupby` is not found in `adata.obs`.
        If any gene in `genes` is not found in `adata.var_names`.
        If any group in `groups` is not found in `adata.obs[groupby]`.

    Examples
    --------
    Compute the average expression of all genes across all groups:
    
    >>> avg_expr = la.tl._compute_avg_expression(adata, groupby='Leiden')

    Compute the average expression of specific genes across all groups:
    
    >>> avg_expr = la.tl._compute_avg_expression(adata, groupby='Leiden', genes=['Gad1', 'Gad2'])

    Compute the average expression of all genes in selected groups:
    
    >>> avg_expr = la.tl._compute_avg_expression(adata, groupby='Leiden', groups=['1', '2'])

    Compute the average expression of selected genes in selected groups:
    
    >>> avg_expr = la.tl._compute_avg_expression(adata, groupby='Leiden', genes=['Gad1', 'Gad2'], groups=['1', '2'])
    """
    # Error checks
    if groupby not in adata.obs.columns:
        raise ValueError(f"'{groupby}' column not found in adata.obs. Please provide a valid column name.")
        
    if genes and not all(gene in adata.var_names for gene in genes):
        missing_genes = [gene for gene in genes if gene not in adata.var_names]
        raise ValueError(f"The following genes are not found in adata.var_names: {', '.join(missing_genes)}")
    
    if groups and not all(group in adata.obs[groupby].cat.categories for group in groups):
        missing_groups = [group for group in groups if group not in adata.obs[groupby].cat.categories]
        raise ValueError(f"The following groups are not found in adata.obs['{groupby}']: {', '.join(missing_groups)}")

    # Warning checks
    if not genes and not groups:
        import warnings
        warnings.warn("No genes or groups specified; computing average expression for all genes and groups. "
                      "This may be memory-intensive for large datasets.")
    
    # Filter data based on provided genes and groups
    adata_copy = adata.copy()
    if genes:
        adata_copy = adata_copy[:, adata_copy.var_names.isin(genes)]
    if groups:
        adata_copy = adata_copy[adata_copy.obs[groupby].isin(groups), :]

    # Initialize result DataFrame with group names as index and genes as columns
    res = pd.DataFrame(columns=adata_copy.var_names, index=adata_copy.obs[groupby].cat.categories)

    # Compute mean expression per group
    for group in adata_copy.obs[groupby].cat.categories:
        res.loc[group] = adata_copy[adata_copy.obs[groupby].isin([group]), :].X.mean(0)

    return res


def _build_adjacency_matrix(
    adata: ad.AnnData,
    use_rep: str = 'X_pca',
    n_nearest_neighbors: int = 10,
    sigma: float = 100.0
) -> sp.csr_matrix:
    """
    Build an adjacency matrix from low-dimensional representations.
    
    This function constructs a weighted k-nearest neighbor graph from a specified
    representation (e.g., PCA, spatial coordinates). Edge weights are computed using
    distance-based exponential decay.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    use_rep : str, default='X_pca'
        Key in adata.obsm for the representation to use (e.g., 'X_pca', 'X_umap', 'X_spatial').
    n_nearest_neighbors : int, default=10
        Number of nearest neighbors to consider.
    sigma : float, default=100.0
        Bandwidth parameter for the exponential kernel. Larger values result in
        slower distance decay.
        
    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse adjacency matrix where entries represent weighted connections between cells.
        
    Examples
    --------
    >>> # Build adjacency from PCA representation
    >>> adj_pca = la.tl._build_adjacency_matrix(adata, use_rep='X_pca')
    >>> 
    >>> # Build adjacency from spatial coordinates
    >>> adj_spatial = la.tl._build_adjacency_matrix(adata, use_rep='X_spatial', sigma=50)
    

    """
    X_rep = adata.obsm[use_rep].copy()
    
    cellxcell = kneighbors_graph(
        X_rep,
        n_neighbors=n_nearest_neighbors,
        mode='distance', 
        include_self=False
    )
    
    cellxcell.data = 1 / np.exp(cellxcell.data / sigma)
    
    return cellxcell


def _build_random_adjacency_matrix(
    adata: ad.AnnData,
    cellxcell: sp.csr_matrix,
    n_nearest_neighbors: int = 10,
    random_seed: int = 0
) -> sp.csr_matrix:
    """
    Efficiently create a random adjacency matrix for null model generation.
    
    This function shuffles the edge weights of an existing adjacency matrix while
    maintaining the same sparsity pattern (number of neighbors per cell). This is
    useful for generating null distributions in statistical tests.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (used only for determining matrix dimensions).
    cellxcell : scipy.sparse.csr_matrix
        Original adjacency matrix whose weights will be shuffled.
    n_nearest_neighbors : int, default=10
        Number of nearest neighbors per cell.
    random_seed : int, default=0
        Random seed for reproducibility.
        
    Returns
    -------
    scipy.sparse.csr_matrix
        Randomized adjacency matrix with shuffled weights.
        
    Examples
    --------
    >>> # Build original adjacency
    >>> adj = la.tl._build_adjacency_matrix(adata)
    >>> 
    >>> # Generate random null model
    >>> adj_random = la.tl._build_random_adjacency_matrix(adata, adj, random_seed=42)
    

    """
    row_ind = np.repeat(np.arange(adata.n_obs), n_nearest_neighbors)
    np.random.seed(random_seed)
    col_ind = np.random.choice(np.arange(adata.n_obs), adata.n_obs * n_nearest_neighbors, replace=True)
    
    # Shuffle the weights
    connectivity = cellxcell.data.copy()
    np.random.shuffle(connectivity)

    cellxcell_shuffle = csr_matrix((connectivity, (row_ind, col_ind)), shape=cellxcell.shape)
    
    return cellxcell_shuffle


def _generate_random_background(
    adata: ad.AnnData,
    cellxcell: sp.csr_matrix,
    genexcell: Union[np.ndarray, sp.csr_matrix],
    n_nearest_neighbors: int = 10,
    n_repeats: int = 30,
    random_seed: int = 0
) -> List[np.ndarray]:
    """
    Generate random background distributions for statistical significance testing.
    
    This function creates multiple randomized versions of the spatial neighborhood
    graph and computes gene spatial specificity scores for each randomization. These
    scores form a null distribution for testing.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    cellxcell : scipy.sparse.csr_matrix
        Original cell-cell adjacency matrix.
    genexcell : np.ndarray or scipy.sparse.csr_matrix
        Gene expression matrix (genes × cells).
    n_nearest_neighbors : int, default=10
        Number of nearest neighbors per cell.
    n_repeats : int, default=30
        Number of random permutations to generate.
    random_seed : int, default=0
        Random seed for reproducibility.
        
    Returns
    -------
    List[np.ndarray]
        List of arrays, each containing gene spatial specificity scores
        for one random permutation.
        
    Examples
    --------
    >>> # Generate null distribution
    >>> adj = la.tl._build_adjacency_matrix(adata)
    >>> genexcell = adata.X.T
    >>> random_scores = la.tl._generate_random_background(
    ...     adata, adj, genexcell, n_repeats=100
    ... )
    

    """
    random_gsp_list = []
    np.random.seed(random_seed)
    
    random_seed_list = np.random.choice(1902772, size=n_repeats, replace=False)
    
    for i in np.arange(n_repeats):
        random_seed = random_seed_list[i]
    
        cellxcell_shuffle = _build_random_adjacency_matrix(
            adata,
            cellxcell,
            n_nearest_neighbors=n_nearest_neighbors,
            random_seed=random_seed
        )
        
        # Normalize the weights (L1 normalization for each row)
        cellxcell_shuffle = normalize(cellxcell_shuffle, axis=1, norm='l1')
        
        random_order1 = genexcell @ cellxcell_shuffle.T

        random_gsp_list.append(_rowwise_cosine_similarity(genexcell, random_order1))
    
    return random_gsp_list


def _calculate_ligand_receptor_specificity(
    adata: ad.AnnData,
    laris_lr: pd.DataFrame,
    groupby: str = 'CellTypes',
    mu: float = 100,
    expressed_pct: float = 0.1,
    remove_lowly_expressed: bool = True,
    mask_threshold: float = 1e-6
) -> pd.DataFrame:
    """
    Calculate cell type specificity of individual ligands and receptors.
    
    This function computes how specifically each ligand and receptor gene is expressed
    in different cell types using the COSG (COSine similarity-based Gene specificity)
    method. Results are normalized to create a gene × cell type matrix.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with gene expression data.
    laris_lr : pd.DataFrame
        DataFrame containing ligand-receptor pairs (from `runLARIS()`).
        Must have 'ligand' and 'receptor' columns.
    groupby : str, default='CellTypes'
        Column in adata.obs defining cell type groups.
    mu : float, default=100
        Regularization parameter for COSG algorithm.
    expressed_pct : float, default=0.1
        Minimum fraction of cells in a group that must express a gene.
    remove_lowly_expressed : bool, default=True
        Whether to filter lowly expressed genes.
    mask_threshold : float, default=1e-6
        Threshold below which values are set to 0.
        
    Returns
    -------
    pd.DataFrame
        DataFrame where:
        - Rows: Gene names (ligands and receptors)
        - Columns: Cell types
        - Values: Normalized specificity scores [0, 1]
        
    Examples
    --------
    >>> # Calculate specificity of L and R genes
    >>> lr_specificity = la.tl._calculate_ligand_receptor_specificity(
    ...     adata, laris_results, groupby='cell_type'
    ... )
    >>> 
    >>> # Check specificity of a specific ligand
    >>> print(lr_specificity.loc['Vegfa', :])
    

    """
    # Extract unique ligands and receptors
    lr_set = np.unique(np.hstack([laris_lr['ligand'].values, laris_lr['receptor'].values]))
    lr_gene_adata = adata[:, lr_set].copy()

    # Perform COSG analysis
    cosg.cosg(
        lr_gene_adata,
        key_added='cosg',
        mu=mu,
        expressed_pct=expressed_pct,
        remove_lowly_expressed=remove_lowly_expressed,
        n_genes_user=lr_gene_adata.n_vars,
        groupby=groupby
    )
    ### Normalize the COSG scores
    cosg_scores=cosg.indexByGene(
        lr_gene_adata.uns['cosg']['COSG'],
        # gene_key="names", score_key="scores",
        set_nan_to_zero=True,
        convert_negative_one_to_zero=True
    )
    gene_by_group_cosg=cosg.iqrLogNormalize(cosg_scores)

    # # Process COSG result and prepare to populate the DataFrame
    # names = pd.DataFrame(lr_gene_adata.uns['cosg']['names']).T
    # scores = pd.DataFrame(lr_gene_adata.uns['cosg']['scores']).T
    # cell_types = names.index.values

    # # Initialize DataFrame for results
    # flat_genes = np.unique(np.concatenate(names.values))
    # gene_by_group_cosg = pd.DataFrame(index=flat_genes, columns=cell_types)

    # # Fill the DataFrame with scores
    # for cell_type in cell_types:
    #     gene_names = names.loc[cell_type].values
    #     gene_scores = scores.loc[cell_type].values
    #     gene_by_group_cosg.loc[gene_names, cell_type] = gene_scores

    # # Mask and normalize
    # mask_tiny = gene_by_group_cosg < mask_threshold
    # gene_by_group_cosg[mask_tiny] = 0
    # gene_by_group_cosg = normalize(gene_by_group_cosg, norm="l1", axis=0)
    # gene_by_group_cosg = pd.DataFrame(gene_by_group_cosg, columns=cell_types, index=flat_genes)
    # gene_by_group_cosg[mask_tiny] = 0

    return gene_by_group_cosg


def _calculate_diffused_lr_specificity(
    lr_adata: ad.AnnData,
    groupby: str = 'CellTypes',
    mu: float = 100,
    expressed_pct: float = 0.05,
    remove_lowly_expressed: bool = True,
    mask_threshold: float = 1e-6
) -> pd.DataFrame:
    """
    Calculate cell type specificity of diffused ligand-receptor interaction scores.
    
    This function computes how specifically each LR interaction score (computed by
    `prepareLRInteraction()`) is enriched in different cell types.
    
    Parameters
    ----------
    lr_adata : AnnData
        AnnData object containing LR interaction scores.
    groupby : str, default='CellTypes'
        Column in lr_adata.obs defining cell type groups.
    mu : float, default=100
        Regularization parameter for COSG algorithm.
    expressed_pct : float, default=0.05
        Minimum fraction of cells in a group that must have non-zero LR score.
    remove_lowly_expressed : bool, default=True
        Whether to filter lowly expressed LR pairs.
    mask_threshold : float, default=1e-6
        Threshold below which values are set to 0.
        
    Returns
    -------
    pd.DataFrame
        DataFrame where:
        - Rows: LR pair names (format: "ligand::receptor")
        - Columns: Cell types
        - Values: Min-max normalized specificity scores [0, 1]
        
    Examples
    --------
    >>> # Calculate diffused LR specificity
    >>> lr_spec = la.tl._calculate_diffused_lr_specificity(
    ...     lr_adata, groupby='cell_type'
    ... )
    

    """
    # Perform COSG analysis
    cosg.cosg(
        lr_adata,
        key_added='cosg',
        mu=mu,
        expressed_pct=expressed_pct,
        remove_lowly_expressed=remove_lowly_expressed,
        n_genes_user=lr_adata.n_vars,
        groupby=groupby
    )


    ### Normalize the COSG scores
    cosg_scores=cosg.indexByGene(
        lr_adata.uns['cosg']['COSG'],
        # gene_key="names", score_key="scores",
        set_nan_to_zero=True,
        convert_negative_one_to_zero=True
    )
    lr_by_group_cosg=cosg.iqrLogNormalize(cosg_scores)

    # # Retrieve names and scores from COSG results
    # names = pd.DataFrame(lr_adata.uns['cosg']['names']).T
    # scores = pd.DataFrame(lr_adata.uns['cosg']['scores']).T
    # cell_types = names.index.values

    # # Initialize lists to store the ordered gene names and scores
    # ordered_genes = []
    # ordered_scores = []

    # # Collect genes and scores per cell type
    # for cell_type in cell_types:
    #     gene_names = names.loc[cell_type].values
    #     gene_scores = scores.loc[cell_type].values
    #     ordered_genes.append(gene_names)
    #     ordered_scores.append(gene_scores)

    # # Create a flat list of unique gene names
    # flat_genes = np.unique(np.concatenate(ordered_genes))

    # # Initialize a DataFrame to hold the final results
    # df = pd.DataFrame(index=flat_genes, columns=cell_types)

    # # Populate the DataFrame with scores
    # for i, cell_type in enumerate(cell_types):
    #     gene_names = ordered_genes[i]
    #     gene_scores = ordered_scores[i]
    #     df.loc[gene_names, cell_type] = gene_scores

    # # Normalize and handle tiny values
    # df.fillna(0, inplace=True)
    # df[df < mask_threshold] = 0
    # df = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))

    return lr_by_group_cosg


def _calculate_specificity_in_spatial_neighborhood(
    adata: ad.AnnData,
    lr_adata: ad.AnnData,
    groupby: str = 'CellTypes',
    use_rep_spatial: str = 'X_spatial',
    number_nearest_neighbors: int = 10,
    mu: float = 100,
    return_by_group: bool = True,
    key_added: str = 'cosg',
    column_delimiter: str = '@@'
) -> pd.DataFrame:
    """
    Calculate cell type pair specificity scores in spatial neighborhoods.
    
    This function computes how specifically each LR interaction is enriched in
    different combinations of spatially neighboring cell types. It considers
    both the cell types of sender and receiver cells and their spatial proximity.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with spatial coordinates and cell type annotations.
        Must have `.obsm[use_rep_spatial]` and `.obs[groupby]`.
    lr_adata : AnnData
        AnnData object containing LR interaction scores.
        Must have `.X` with scores and `.var_names` with LR pair identifiers.
    groupby : str, default='CellTypes'
        Key in adata.obs for cell type/group annotations.
    use_rep_spatial : str, default='X_spatial'
        Key in adata.obsm for spatial coordinates.
    number_nearest_neighbors : int, default=10
        Number of neighbors to consider in the spatial neighborhood.
    mu : float, default=100
        Regularization parameter for the COSG-style specificity calculation.
        Higher values increase regularization strength.
    return_by_group : bool, default=True
        Whether to return results organized by cell type pairs.
    key_added : str, default='cosg'
        Key under which to store results in lr_adata.uns.
    column_delimiter : str, default='@@'
        Delimiter used to separate attributes in multi-level column names.
    
    Returns
    -------
    pd.DataFrame
        DataFrame where:
        - Rows: LR pair names (format: "ligand::receptor")
        - Columns: Cell type pair names (format: "cell_type_i::cell_type_j")
        - Values: IQR log-normalized specificity scores
    
    Raises
    ------
    ValueError
        If required data is missing from AnnData objects or if parameters
        have invalid values.
    
    Examples
    --------
    >>> # Calculate spatial neighborhood specificity
    >>> spatial_spec = la.tl._calculate_specificity_in_spatial_neighborhood(
    ...     adata, lr_adata, groupby='cell_type'
    ... )
    >>> 
    >>> # Check specificity for a specific LR pair and cell type combination
    >>> print(spatial_spec.loc['Vegfa::Kdr', 'Endothelial::Tumor'])
    
    Notes
    -----
    This function:
    1. Builds a spatial neighborhood graph
    2. Computes cell type composition of each cell's neighborhood
    3. Creates all pairwise cell type combinations
    4. Calculates cosine similarity between LR scores and cell type pair patterns
    5. Applies regularization and normalization
    
    The results quantify how much each LR interaction is associated with
    specific sender-receiver cell type pairs in the spatial context.
    
    This function stores results in lr_adata.uns[key_added] and returns
    a DataFrame of normalized scores.
    """
    # Check if required data exists in AnnData objects
    if groupby not in adata.obs:
        raise ValueError(f"'{groupby}' not found in adata.obs")
    if use_rep_spatial not in adata.obsm:
        raise ValueError(f"'{use_rep_spatial}' not found in adata.obsm")

    # Get group information
    group_info = adata.obs[groupby].copy()
    n_cell = adata.n_obs
    
    # Determine group order
    if hasattr(group_info, "cat"):
        groups_order = list(group_info.cat.categories)
    else:
        unique_values = group_info.unique()
        
        # Helper function to check if all values are numeric
        def _is_all_numeric(groups):
            try:
                [float(x) for x in groups]
                return True
            except ValueError:
                return False

        if _is_all_numeric(unique_values):
            groups_order = sorted(unique_values, key=lambda x: float(x))
        else:
            groups_order = sorted(unique_values)
    
    n_cluster = len(groups_order)
    
    # Create sparse matrix for cluster membership
    group_to_row = {group: i for i, group in enumerate(groups_order)}
    row_indices = np.array([group_to_row[group] for group in group_info])
    col_indices = np.arange(n_cell)
    data = np.ones_like(col_indices, dtype=int)
    cluster_mat = csr_matrix((data, (row_indices, col_indices)), shape=(n_cluster, n_cell))
    
    # Create spatial neighborhood graph
    X_spatial = adata.obsm[use_rep_spatial].copy()
    cellxcell = kneighbors_graph(
        X_spatial, 
        n_neighbors=number_nearest_neighbors,
        mode='distance', 
        include_self=False
    )
    cellxcell.data = 1 / np.exp(cellxcell.data / (np.mean(cellxcell.data) / 2))
    
    # Calculate first-order neighborhood composition
    order1 = cluster_mat @ cellxcell.T
    
    # Get cell type pair interactions
    ctXct_cell, ctXct = _pairwise_row_multiply(order1, cell_types=groups_order)
    
    # Calculate cosine similarity between genes and cell type pairs
    cosine_sim = cosine_similarity(
        X=lr_adata.X.T,
        Y=ctXct_cell,
        dense_output=False
    )
    
    # Apply regularization to cosine similarity
    genexlambda = cosine_sim.copy()
    genexlambda.data = np.multiply(genexlambda.data, genexlambda.data)
    e_power2_sum = np.array(genexlambda.sum(axis=1)).flatten()
    
    if mu == 1:
        genexlambda.data = genexlambda.data / np.repeat(e_power2_sum, np.diff(genexlambda.indptr))
    else:
        genexlambda.data = genexlambda.data / (
            (1 - mu) * genexlambda.data + mu * np.repeat(e_power2_sum, np.diff(genexlambda.indptr))
        )
    
    genexlambda.data = np.multiply(genexlambda.data, cosine_sim.data)
    
    # Set up storage for results
    if key_added is None:
        key_added = 'cosg'
        
    lr_adata.uns[key_added] = {}
    lr_adata.uns[key_added]['params'] = dict(
        groupby=groupby,
        method='COSG',
        mu=mu,
    )

    ### No expression percentage filtering here, as it's already filtered in the ligand/receptor specificity calculation
    # # Get gene expression matrix
    # cellxgene = lr_adata.X.copy()
    
    # # Helper function for counting non-zeros
    # if sp.issparse(cellxgene):
    #     cellxgene.eliminate_zeros()
    #     get_nonzeros = lambda X: X.getnnz(axis=0)
    # else:
    #     get_nonzeros = lambda X: np.count_nonzero(X, axis=0)
    
    # Process each cell type
    rank_stats = None
    order_i = 0
    
    for group_i in ctXct:
        # # Create boolean mask for cells in this group
        # idx_i = group_info == group_i
        # idx_i = idx_i.values  # Convert to numpy array
        
        # Get specificity scores
        if sp.issparse(genexlambda):
            scores = genexlambda[:, order_i].toarray()[:, 0]
        else:
            scores = genexlambda[:, order_i]
        
        # # Filter lowly expressed genes if requested
        # if remove_lowly_expressed:
        #     n_cells_expressed = get_nonzeros(cellxgene[idx_i])
        #     n_cells_i = np.sum(idx_i)
        #     scores[n_cells_expressed < n_cells_i * expressed_pct] = -1
        
        # Select top genes
        global_indices = _select_top_n(scores, lr_adata.n_vars)
        
        # Initialize DataFrame if needed
        if rank_stats is None:
            rank_stats = pd.DataFrame()
        
        # Prepare data for new columns
        columns_data = {
            (group_i, 'names'): lr_adata.var_names.values[global_indices],
            (group_i, 'scores'): scores[global_indices]
        }
        
        # Create and concatenate new data
        new_data = pd.DataFrame(columns_data)
        rank_stats = pd.concat([rank_stats, new_data], axis=1)
        
        order_i = order_i + 1
    
    # Create by-group results if requested
    if return_by_group:
        # Swap levels to ensure attribute comes first, then cell group
        rank_stats_swapped = rank_stats.copy()
        rank_stats_swapped.columns = rank_stats_swapped.columns.swaplevel()
        
        # Create flattened column names
        flattened_columns = [column_delimiter.join(map(str, col)) for col in rank_stats_swapped.columns]
        cosg_results = rank_stats_swapped.copy()
        cosg_results.columns = flattened_columns
        
        lr_adata.uns[key_added]['COSG'] = cosg_results
    
    # Store results in lr_adata
    dtypes = {
        'names': 'O',
        'scores': 'float32',
        'logfoldchanges': 'float32',
    }
    
    rank_stats.columns = rank_stats.columns.swaplevel()
    for col in rank_stats.columns.levels[0]:
        lr_adata.uns[key_added][col] = rank_stats[col].to_records(
            index=False, column_dtypes=dtypes[col]
        )
    
    # Process and return normalized results
    cosg_lr_ctc = cosg.indexByGene(
        lr_adata.uns[key_added]['COSG'],
        column_delimiter=column_delimiter,
    )
    cosg_lr_ctc = cosg.iqrLogNormalize(cosg_lr_ctc)
    
    return cosg_lr_ctc

def _prepare_background_genes(
    adata: ad.AnnData,
    layer: str = None,
    n_nearest_neighbors: int = 30,
    leaf_size: int = 30
) -> pd.DataFrame:
    """
    Pre-calculates a nearest-neighbor graph for all genes based on their
    mean expression and variance.
    """
    print("Preparing background gene set by calculating mean and variance for all genes...")
    if layer is not None and layer in adata.layers:
        expression_matrix = adata.layers[layer]
    else:
        expression_matrix = adata.X

    if not sp.isspmatrix_csr(expression_matrix):
        expression_matrix = expression_matrix.tocsr()

    mean_expression = np.array(expression_matrix.mean(axis=0)).flatten()
    mean_squared_expression = np.array(expression_matrix.power(2).mean(axis=0)).flatten()
    variance_expression = mean_squared_expression - (mean_expression ** 2)
    mean_variance_data = np.array([mean_expression, variance_expression]).T

    print(f"Building KDTree to find {n_nearest_neighbors} nearest neighbors for each gene...")
    kdt = KDTree(mean_variance_data, leaf_size=leaf_size, metric='euclidean')
    _, indices = kdt.query(mean_variance_data, k=n_nearest_neighbors + 1)
    
    neighbor_indices = indices[:, 1:]
    knn_df = pd.DataFrame(neighbor_indices, index=adata.var_names)
    print("Finished preparing background gene set.")
    return knn_df


def _prepare_background_interactions(
    lr_adata: ad.AnnData,
    n_nearest_neighbors: int = 30,
    leaf_size: int = 30
) -> pd.DataFrame:
    """
    Build a background set of similar interactions for permutation testing.
    
    This function creates a reference set for each ligand-receptor (LR) pair by 
    identifying other LR pairs with similar diffused score profiles. Similarity 
    is determined by computing the mean and variance of diffused scores across 
    cells, then finding nearest neighbors in this 2D feature space using a KDTree.
    
    The rationale is that interactions with similar expression patterns across 
    space should serve as appropriate null controls for statistical testing, 
    rather than using completely random interactions.
    
    Parameters
    ----------
    lr_adata : AnnData
        AnnData object with interactions in `.var` and diffused scores in `.X`.
        Rows are cells, columns are LR pairs (e.g., "Tgfb1::Tgfbr1").
    n_nearest_neighbors : int, default=30
        Number of similar interactions to identify for each LR pair. These will
        be used to build the null distribution during permutation testing.
    leaf_size : int, default=30
        Leaf size parameter for the KDTree algorithm. Larger values can speed up
        construction but may slow down queries.
        
    Returns
    -------
    pd.DataFrame
        DataFrame where:
        - Index: interaction names (e.g., 'Tgfb1::Tgfbr1')
        - Columns: [0, 1, ..., n_nearest_neighbors-1]
        - Values: Integer indices corresponding to positions in `lr_adata.var_names`
        
        For each row (interaction), the values represent the indices of its
        n_nearest_neighbors most similar interactions based on diffused score profiles.
    
    Notes
    -----
    The background interactions are used during statistical testing to create a
    more biologically meaningful null distribution. Rather than comparing each
    LR pair to all possible interactions, we compare it to interactions with
    similar spatial expression patterns.
    
    The features used for similarity are:
    - Mean diffused score: Average interaction strength across all cells
    - Variance: Spatial heterogeneity of the interaction
    
    Examples
    --------
    >>> background = _prepare_background_interactions(lr_adata, n_nearest_neighbors=50)
    >>> # For the first interaction, get its control set
    >>> first_interaction = lr_adata.var_names[0]
    >>> control_indices = background.loc[first_interaction].values
    >>> control_names = lr_adata.var_names[control_indices]
    """
    print(f"  - Calculating summary statistics for {lr_adata.n_vars} interactions...")
    
    # Ensure the expression matrix is in CSR format for efficient row operations
    expression_matrix = lr_adata.X
    if not sp.isspmatrix_csr(expression_matrix):
        expression_matrix = expression_matrix.tocsr()
    
    # Compute mean and variance of diffused scores across cells
    # Mean: E[X]
    mean_scores = np.asarray(expression_matrix.mean(axis=0)).flatten()
    
    # Variance: E[X^2] - (E[X])^2
    mean_squared_scores = np.asarray(expression_matrix.power(2).mean(axis=0)).flatten()
    var_scores = mean_squared_scores - (mean_scores ** 2)
    
    # Create feature matrix: each interaction is represented by (mean, variance)
    features = np.column_stack([mean_scores, var_scores])
    
    print(f"  - Building KDTree to find {n_nearest_neighbors} nearest neighbors...")
    
    # Build KDTree for efficient nearest neighbor search
    # We query for n+1 neighbors because the first neighbor is always the point itself
    kdt = KDTree(features, leaf_size=leaf_size, metric='euclidean')
    distances, indices = kdt.query(features, k=n_nearest_neighbors + 1)
    
    # Remove self from neighbors (always at index 0, with distance=0)
    neighbor_indices = indices[:, 1:]
    
    # Create DataFrame with interaction names as index
    background_df = pd.DataFrame(
        neighbor_indices,
        index=lr_adata.var_names,
        columns=range(n_nearest_neighbors)
    )
    
    print(f"  - Background set prepared: {len(background_df)} interactions")
    
    return background_df



def _calculate_laris_score_by_celltype(
    adata: ad.AnnData,
    lr_adata: ad.AnnData,
    laris_lr: pd.DataFrame,
    groupby: str = 'CellTypes',
    use_rep_spatial: str = 'X_spatial',
    number_nearest_neighbors: int = 10,
    mu: float = 100,
    expressed_pct: float = 0.1,
    remove_lowly_expressed: bool = True,
    mask_threshold: float = 1e-6,
    calculate_pvalues: bool = True,
    layer: Optional[str] = None,
    n_nearest_neighbors: int = 30,
    n_permutations: int = 1000,
    chunk_size: int = 50000,
    prefilter_fdr: bool = True,
    prefilter_threshold: float = 0.0,
    score_threshold: float = 1e-6,
    spatial_weight: float = 1.0,
    use_conditional_pvalue: bool = False
) -> pd.DataFrame:
    """
    Calculate cell type-specific LARIS interaction scores with statistical testing.
    
    This is the core function for computing sender-receiver cell type pair interaction
    scores. It integrates multiple sources of information to identify biologically
    meaningful ligand-receptor interactions:
    
    1. **Spatial specificity**: From the runLARIS() output, indicating which LR pairs
       show spatial co-localization patterns
    2. **Cell type specificity**: Expression specificity of ligand and receptor genes
       in different cell types (computed using COSG algorithm)
    3. **Diffused LR scores**: Neighborhood-averaged interaction scores that capture
       local microenvironment effects
    4. **Spatial co-localization**: How frequently sender-receiver cell types are
       found in proximity
    5. **Statistical significance**: Permutation-based p-values and FDR correction
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with gene expression and spatial information.
        Must contain:
        - `.obs[groupby]`: Cell type annotations
        - `.obsm[use_rep_spatial]`: Spatial coordinates
        - `.X` or `.layers[layer]`: Gene expression data
        
    lr_adata : AnnData
        AnnData object containing LR interaction scores from prepareLRInteraction().
        Must have the same cells as `adata`.
        
    laris_lr : pd.DataFrame
        DataFrame of spatially-specific LR pairs from runLARIS(). Must contain:
        - 'ligand': Ligand gene names
        - 'receptor': Receptor gene names
        - 'score': LARIS spatial specificity scores
        
    groupby : str, default='CellTypes'
        Column name in `adata.obs` that defines cell type groups.
        
    use_rep_spatial : str, default='X_spatial'
        Key in `adata.obsm` containing spatial coordinates (n_cells × 2 or 3).
        
    number_nearest_neighbors : int, default=10
        Number of spatial neighbors to consider for neighborhood analysis.
        Larger values capture broader spatial contexts.
        
    mu : float, default=100
        Regularization parameter for COSG specificity calculation.
        Higher values increase penalty for broadly expressed genes.
        
    expressed_pct : float, default=0.1
        Minimum fraction of cells in a group that must express a gene for it
        to be considered "expressed" in that group.
        
    remove_lowly_expressed : bool, default=True
        Whether to filter out lowly expressed genes before analysis.
        
    mask_threshold : float, default=1e-6
        Numerical threshold below which values are masked as zero.
        
    calculate_pvalues : bool, default=True
        Whether to perform permutation testing for statistical significance.
        If False, only interaction scores are computed (much faster).
        
    layer : str, optional
        Layer in `adata.layers` to use for expression data. If None, uses `adata.X`.
        
    n_nearest_neighbors : int, default=30
        Number of similar interactions to use as background for permutation testing.
        These are selected based on similarity of diffused score profiles.
        
    n_permutations : int, default=1000
        Number of permutations for statistical testing. More permutations give
        more precise p-values but take longer. Common values: 1000-10000.
        
    chunk_size : int, default=50000
        Number of interactions to process simultaneously during permutation testing.
        Larger values are faster but use more memory.
        
    prefilter_fdr : bool, default=True
        If True, only interactions with scores > prefilter_threshold are tested.
        Others are assigned FDR p-value = 1.0. This reduces multiple testing burden
        and focuses power on high-scoring interactions.
        
    prefilter_threshold : float, default=0.0
        Minimum interaction score for FDR testing (if prefilter_fdr=True).
        
    score_threshold : float, default=1e-6
        Numerical precision threshold. Scores below this are set to exactly 0.0
        to avoid floating-point artifacts in p-value calculations.
        
    spatial_weight : float, default=1.0
        Exponent applied to spatial specificity scores. Controls how much spatial
        specificity influences final scores:
        - 0: Ignore spatial specificity (all weights = 1)
        - 1: Linear influence (default)
        - >1: Stronger emphasis on spatial specificity
        - <1: Weaker emphasis on spatial specificity
        
    use_conditional_pvalue : bool, default=False
        Use conditional p-value calculation for zero-inflated data. Recommended
        for sparse datasets. When True:
        - Interactions with score=0 get p-value=1.0
        - Non-zero scores are compared only to non-zero background scores
        This prevents artificially significant p-values from sparse null distributions.
        
    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with one row per sender-receiver-LR combination.
        Columns:
        - 'sender': Cell type sending the ligand
        - 'receiver': Cell type receiving the signal  
        - 'ligand': Ligand gene name
        - 'receptor': Receptor gene name
        - 'interaction_name': Format "ligand::receptor"
        - 'interaction_score': Integrated LARIS score (higher = stronger interaction)
        - 'p_value': Raw p-value from permutation test (if calculate_pvalues=True)
        - 'p_value_fdr': Benjamini-Hochberg FDR-corrected p-value per cell type pair
        - 'nlog10_p_value_fdr': -log10(p_value_fdr) for visualization
        
        Sorted by interaction_score in descending order.
    
    Examples
    --------
    >>> # Basic usage with default parameters
    >>> results = _calculate_laris_score_by_celltype(
    ...     adata=adata,
    ...     lr_adata=lr_adata,
    ...     laris_lr=laris_lr
    ... )
    
    >>> # Fast mode without p-values
    >>> results = _calculate_laris_score_by_celltype(
    ...     adata=adata,
    ...     lr_adata=lr_adata, 
    ...     laris_lr=laris_lr,
    ...     calculate_pvalues=False
    ... )
    
    >>> # Conservative testing for sparse data
    >>> results = _calculate_laris_score_by_celltype(
    ...     adata=adata,
    ...     lr_adata=lr_adata,
    ...     laris_lr=laris_lr,
    ...     use_conditional_pvalue=True,
    ...     n_permutations=5000
    ... )
    
    >>> # Emphasize spatial specificity
    >>> results = _calculate_laris_score_by_celltype(
    ...     adata=adata,
    ...     lr_adata=lr_adata,
    ...     laris_lr=laris_lr,
    ...     spatial_weight=2.0
    ... )
    
    See Also
    --------
    runLARIS : Identifies spatially-specific LR pairs (upstream of this function)
    prepareLRInteraction : Computes diffused LR scores (upstream of this function)
    """
    
    # =========================================================================
    # STEP 1: Calculate Cell Type Specificity Scores
    # =========================================================================
    print("\n" + "="*70)
    print("LARIS CELL TYPE ANALYSIS")
    print("="*70)
    print("\n--- Step 1: Calculating ligand and receptor cell type specificity ---")
    
    gene_by_group_cosg = _calculate_ligand_receptor_specificity(
        adata,
        laris_lr,
        groupby=groupby,
        mu=mu,
        expressed_pct=expressed_pct,
        remove_lowly_expressed=remove_lowly_expressed,
        mask_threshold=mask_threshold
    )
    
    print(f"  ✓ Computed specificity for {gene_by_group_cosg.shape[0]} genes "
          f"across {gene_by_group_cosg.shape[1]} cell types")
    
    # =========================================================================
    # STEP 2: Calculate Diffused LR-score Distribution
    # =========================================================================
    print("\n--- Step 2: Calculating diffused LR-score distribution ---")
    
    # Create binary version of lr_adata for presence/absence calculation
    lr_adata_percentage = lr_adata.copy()
    lr_adata_percentage.X = sp.csr_matrix(lr_adata_percentage.X)
    lr_adata_percentage.X.data = np.ones_like(lr_adata_percentage.X.data)
    
    # Compute average expression per cell type
    avg_gene_expr_lr = _compute_avg_expression(
        lr_adata_percentage, 
        groupby=groupby
    ).T
    del lr_adata_percentage
    
    # Ensure column order matches cell type order
    avg_gene_expr_lr = avg_gene_expr_lr.loc[:, gene_by_group_cosg.columns]
    
    print(f"  ✓ Computed diffused scores for {avg_gene_expr_lr.shape[0]} interactions")

    # =========================================================================
    # STEP 3: Compute Interaction Scores for All Sender-Receiver Pairs
    # =========================================================================
    print(f"\n--- Step 3: Computing interaction scores (spatial_weight={spatial_weight}) ---")
    
    results = []
    n_processed = 0
    n_skipped = 0
    
    # Iterate over each LR pair in laris_lr
    for idx, row in laris_lr.iterrows():
        ligand = row['ligand']
        receptor = row['receptor']
        spatial_score = row['score']
        interaction_name = f"{ligand}::{receptor}"

        # Skip if ligand or receptor not in cell type specificity data
        if ligand not in gene_by_group_cosg.index or receptor not in gene_by_group_cosg.index:
            n_skipped += 1
            continue

        # Get cell type specificity scores
        ligand_scores = gene_by_group_cosg.loc[ligand, :]
        receptor_scores = gene_by_group_cosg.loc[receptor, :]
        
        # Get diffused LR scores per cell type
        lr_by_group_cosg_i = avg_gene_expr_lr.loc[interaction_name]
        
        # Apply spatial weight transformation
        if spatial_weight == 0:
            spatial_factor = 1.0
        else:
            spatial_factor = np.power(spatial_score, spatial_weight)
        
        # Create interaction matrix: outer product of ligand and receptor specificity
        interaction_matrix = np.outer(ligand_scores, receptor_scores) * spatial_factor
        
        interaction_df = pd.DataFrame(
            interaction_matrix,
            index=gene_by_group_cosg.columns,
            columns=gene_by_group_cosg.columns
        )
        
        # Incorporate diffused LR scores (sender × receiver outer product)
        lr_interaction_df = pd.DataFrame(
            np.outer(lr_by_group_cosg_i, lr_by_group_cosg_i),
            index=gene_by_group_cosg.columns,
            columns=gene_by_group_cosg.columns
        )
        
        # Multiply the two components
        interaction_df = lr_interaction_df.multiply(interaction_df)
        
        # Convert to long format
        interaction_df = interaction_df.reset_index().melt(
            id_vars='index',
            var_name='receiver',
            value_name='interaction_score'
        )
        interaction_df.rename(columns={'index': 'sender'}, inplace=True)
        interaction_df['ligand'] = ligand
        interaction_df['receptor'] = receptor
        interaction_df['interaction_name'] = interaction_name
        
        results.append(interaction_df)
        n_processed += 1

    print(f"  ✓ Processed {n_processed} interactions")
    if n_skipped > 0:
        print(f"  ⚠ Skipped {n_skipped} interactions (genes not found)")

    # Concatenate all results
    if results:
        res_laris = pd.concat(results, ignore_index=True)
    else:
        print("  ✗ No valid interactions found!")
        return pd.DataFrame(
            columns=['sender', 'receiver', 'ligand', 'receptor', 
                    'interaction_name', 'interaction_score']
        )

    res_laris = res_laris.sort_values(by='interaction_score', ascending=False)
    res_laris.reset_index(drop=True, inplace=True)
    
    # =========================================================================
    # STEP 3.5: Rescale Scores
    # =========================================================================
    print("\n--- Step 3.5: Rescaling interaction scores ---")
    
    if not res_laris.empty:
        n_for_scaling = min(100, len(res_laris))
        top_scores_mean = res_laris['interaction_score'].head(n_for_scaling).mean()
        
        if top_scores_mean > 0:
            scale_factor = 0.1 / top_scores_mean
            print(f"  - Scaling factor: {scale_factor:.6f} (based on top {n_for_scaling} scores)")
            res_laris['interaction_score'] *= scale_factor
        else:
            print("  ⚠ Skipping rescaling: Mean of top scores is 0")
    else:
        print("  ⚠ Skipping rescaling: No interaction scores")
    
    # =========================================================================
    # STEP 4: Incorporate Spatial Cell Type Neighborhood Information
    # =========================================================================
    print("\n--- Step 4: Incorporating spatial cell type co-localization ---")
    
    cosg_lr_ctc = _calculate_specificity_in_spatial_neighborhood(
        adata,
        lr_adata,
        groupby=groupby,
        use_rep_spatial=use_rep_spatial,
        number_nearest_neighbors=number_nearest_neighbors,
        mu=mu,
        return_by_group=True,
        key_added='cosg_lr',
        column_delimiter='@@'
    )
    
    print(f"  ✓ Calculated co-localization for {cosg_lr_ctc.shape[0]} interactions "
          f"and {cosg_lr_ctc.shape[1]} sender-receiver pairs")

    if not res_laris.empty:
        ctc_cosg_scores = []
        for row in res_laris.itertuples():
            ctc_i = f"{row.sender}::{row.receiver}"
            lr_i = row.interaction_name
            ctc_cosg_score_i = cosg_lr_ctc.loc[lr_i, ctc_i]
            ctc_cosg_scores.append(ctc_cosg_score_i)

        res_laris['interaction_score'] *= ctc_cosg_scores
        res_laris = res_laris.sort_values('interaction_score', ascending=False, ignore_index=True)
        print(f"  ✓ Applied co-localization weights")
    
    # =========================================================================
    # STEP 4.5: Clean Up Numerical Precision Issues
    # =========================================================================
    print(f"\n--- Step 4.5: Cleaning up numerical precision (threshold={score_threshold}) ---")
    
    res_laris['interaction_score'] = res_laris['interaction_score'].astype(np.float64)
    
    n_below_threshold = (res_laris['interaction_score'] < score_threshold).sum()
    n_exact_zero = (res_laris['interaction_score'] == 0.0).sum()
    
    print(f"  - Before: {n_below_threshold:,} scores < threshold, "
          f"{n_exact_zero:,} exactly zero")
    
    # Set very small values to exactly 0.0
    res_laris.loc[res_laris['interaction_score'] < score_threshold, 'interaction_score'] = 0.0
    
    n_zero_after = (res_laris['interaction_score'] == 0.0).sum()
    print(f"  - After: {n_zero_after:,} scores set to exactly 0.0")
    
    # =========================================================================
    # STEP 5: Statistical Significance Testing (Optional)
    # =========================================================================
    if calculate_pvalues:
        print("\n" + "="*70)
        print("STATISTICAL SIGNIFICANCE TESTING")
        print("="*70)
        
        # Step 5.1: Prepare Background Interactions
        print("\n--- Step 5.1: Preparing background interaction sets ---")
        
        all_interaction_names = lr_adata.var_names.values
        
        background_interactions = _prepare_background_interactions(
            lr_adata, 
            n_nearest_neighbors=n_nearest_neighbors
        )
        
        # Step 5.2: Calculate P-values via Permutation Testing
        print(f"\n--- Step 5.2: Calculating p-values ({n_permutations:,} permutations) ---")
        print(f"  - Method: {'Conditional' if use_conditional_pvalue else 'Standard'}")
        print(f"  - Chunk size: {chunk_size:,}")
        
        p_values_list = []
        grouped_interactions = res_laris.groupby(['sender', 'receiver'])
        
        n_cell_type_pairs = len(grouped_interactions)
        print(f"  - Testing {n_cell_type_pairs} sender-receiver pairs")

        for (sender, receiver), group_df in tqdm(
            grouped_interactions, 
            total=n_cell_type_pairs,
            desc="  Processing pairs"
        ):
            # Create score lookup for this cell type pair
            score_lookup = {
                (row.ligand, row.receptor): row.interaction_score 
                for row in group_df.itertuples()
            }
            
            # Process in chunks to manage memory
            for i in range(0, len(group_df), chunk_size):
                chunk_df = group_df.iloc[i:i + chunk_size]
                n_in_chunk = len(chunk_df)

                # Get observed data
                interaction_names = chunk_df['interaction_name'].values
                observed_scores = chunk_df['interaction_score'].values
                
                # Sample background interactions
                control_interaction_indices = background_interactions.loc[interaction_names].values
                
                rand_indices = np.random.randint(
                    0, control_interaction_indices.shape[1],
                    size=(n_in_chunk, n_permutations)
                )
                
                row_idx = np.arange(n_in_chunk)[:, np.newaxis]
                control_indices = control_interaction_indices[row_idx, rand_indices]
                
                control_interaction_names = all_interaction_names[control_indices]
                
                # Look up scores for background interactions
                control_interaction_names_flat = control_interaction_names.flatten()
                null_scores_flat = []
                
                for name in control_interaction_names_flat:
                    try:
                        L, R = name.split('::')
                        null_scores_flat.append(score_lookup.get((L, R), 0.0))
                    except ValueError:
                        null_scores_flat.append(0.0)
                
                null_distribution = np.array(
                    null_scores_flat, 
                    dtype=np.float64
                ).reshape(n_in_chunk, n_permutations)
                
                # Calculate p-values
                if use_conditional_pvalue:
                    # Conditional p-value for zero-inflated data
                    p_values_chunk = np.ones(n_in_chunk, dtype=np.float64)
                    non_zero_obs_mask = observed_scores > 0.0
                    
                    if np.any(non_zero_obs_mask):
                        obs_to_test = observed_scores[non_zero_obs_mask]
                        null_to_test = null_distribution[non_zero_obs_mask, :]
                        
                        null_non_zero_mask = null_to_test > 0.0
                        n_null_nonzero = np.sum(null_non_zero_mask, axis=1)
                        
                        is_greater_all = null_to_test >= obs_to_test[:, np.newaxis]
                        is_greater_and_nonzero = is_greater_all & null_non_zero_mask
                        n_numerator = np.sum(is_greater_and_nonzero, axis=1)
                        
                        p_values_calculated = (n_numerator + 1) / (n_null_nonzero + 1)
                        p_values_chunk[non_zero_obs_mask] = p_values_calculated
                else:
                    # Standard p-value calculation
                    is_greater = null_distribution >= observed_scores[:, np.newaxis]
                    p_values_chunk = (np.sum(is_greater, axis=1) + 1) / (n_permutations + 1)
                
                p_values_list.append(pd.Series(p_values_chunk, index=chunk_df.index))

        if p_values_list:
            res_laris['p_value'] = pd.concat(p_values_list)
        else:
            res_laris['p_value'] = np.nan
        
        print(f"  ✓ Calculated p-values for {len(res_laris):,} interaction combinations")
        
        # Step 5.3: FDR Correction
        print("\n--- Step 5.3: Applying FDR correction (Benjamini-Hochberg) ---")
        print(f"  - Strategy: {'Pre-filtered' if prefilter_fdr else 'All interactions'}")
        if prefilter_fdr:
            print(f"  - Pre-filter threshold: {prefilter_threshold}")
        
        res_laris['p_value_fdr'] = np.nan
        
        n_pairs_corrected = 0
        n_interactions_tested = 0
        n_interactions_filtered = 0

        for (sender, receiver), group_df in tqdm(
            res_laris.groupby(['sender', 'receiver']),
            desc="  FDR correction"
        ):
            p_values = group_df['p_value']
            if p_values.empty or p_values.isnull().all():
                continue
            
            # Round for numerical stability
            p_values = p_values.round(6)

            if prefilter_fdr:
                filter_mask = group_df['interaction_score'] > prefilter_threshold
                p_values_to_test = p_values[filter_mask]
                
                n_interactions_tested += len(p_values_to_test)
                n_interactions_filtered += len(p_values) - len(p_values_to_test)

                if not p_values_to_test.empty:
                    _, p_values_corrected, _, _ = multipletests(
                        p_values_to_test, 
                        method='fdr_bh'
                    )
                    res_laris.loc[p_values_to_test.index, 'p_value_fdr'] = p_values_corrected
                
                # Filtered interactions get FDR = 1.0
                if not p_values.empty:
                    res_laris.loc[p_values[~filter_mask].index, 'p_value_fdr'] = 1.0
            else:
                _, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')
                res_laris.loc[group_df.index, 'p_value_fdr'] = p_values_corrected
                n_interactions_tested += len(p_values)
            
            n_pairs_corrected += 1

        # Clean up FDR values
        res_laris['p_value_fdr'] = res_laris['p_value_fdr'].clip(upper=1.0)
        res_laris['p_value_fdr'].fillna(1.0, inplace=True)

        # Calculate -log10(FDR) for visualization
        res_laris['nlog10_p_value_fdr'] = -np.log10(res_laris['p_value_fdr'] + 1e-10)
        res_laris['nlog10_p_value_fdr'] = res_laris['nlog10_p_value_fdr'].clip(lower=0)
        
        print(f"  ✓ Corrected {n_pairs_corrected} sender-receiver pairs")
        print(f"  - Interactions tested: {n_interactions_tested:,}")
        if prefilter_fdr:
            print(f"  - Interactions filtered: {n_interactions_filtered:,}")
        
        # Report significant interactions
        n_sig_05 = (res_laris['p_value_fdr'] < 0.05).sum()
        n_sig_01 = (res_laris['p_value_fdr'] < 0.01).sum()
        print(f"\n  Significant interactions:")
        print(f"  - FDR < 0.05: {n_sig_05:,} ({100*n_sig_05/len(res_laris):.2f}%)")
        print(f"  - FDR < 0.01: {n_sig_01:,} ({100*n_sig_01/len(res_laris):.2f}%)")
        
        print("\n" + "="*70)
        print("STATISTICAL TESTING COMPLETE")
        print("="*70)
    else:
        # No p-value calculation
        res_laris['p_value'] = np.nan
        res_laris['p_value_fdr'] = np.nan
        res_laris['nlog10_p_value_fdr'] = np.nan
    
    print("\n" + "="*70)
    print("LARIS CELL TYPE ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nFinal results: {len(res_laris):,} sender-receiver-LR combinations")
    print(f"Score range: [{res_laris['interaction_score'].min():.6f}, "
          f"{res_laris['interaction_score'].max():.6f}]")
    
    return res_laris

# Define public API for utility functions (advanced users)
__all__ = [
    '_rowwise_cosine_similarity',
    '_select_top_n',
    '_calculate_laris_score_by_celltype',
    '_calculate_specificity_in_spatial_neighborhood',
]
