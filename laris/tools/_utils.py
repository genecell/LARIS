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
    
    Notes
    -----
    For sparse matrices, this function uses element-wise multiplication and
    sum operations optimized for sparse data structures, making it efficient
    for large, sparse datasets common in single-cell genomics.
    
    The cosine similarity is computed as:
        similarity = (A · B) / (||A|| * ||B||)
    where · denotes dot product and ||·|| denotes L2 norm.
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
    
    Notes
    -----
    This function uses np.argpartition for O(n) time complexity rather than
    O(n log n) for full sorting, making it efficient for large arrays when
    only the top elements are needed.
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
    
    Notes
    -----
    The edge weights are computed as: weight = 1 / exp(distance / sigma)
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
    
    Notes
    -----
    The randomization preserves:
    - The number of neighbors per cell
    - The distribution of edge weights
    But randomizes:
    - Which cells are connected
    - The assignment of weights to edges
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
    
    Notes
    -----
    This function is typically used internally by `runLARIS()` to compute
    statistical significance of spatial specificity scores.
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
    
    Notes
    -----
    This function is typically used as an intermediate step in 
    `_calculate_laris_score_by_celltype()` to determine which cell types
    send and receive specific signals.
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

    # Process COSG result and prepare to populate the DataFrame
    names = pd.DataFrame(lr_gene_adata.uns['cosg']['names']).T
    scores = pd.DataFrame(lr_gene_adata.uns['cosg']['scores']).T
    cell_types = names.index.values

    # Initialize DataFrame for results
    flat_genes = np.unique(np.concatenate(names.values))
    gene_by_group_cosg = pd.DataFrame(index=flat_genes, columns=cell_types)

    # Fill the DataFrame with scores
    for cell_type in cell_types:
        gene_names = names.loc[cell_type].values
        gene_scores = scores.loc[cell_type].values
        gene_by_group_cosg.loc[gene_names, cell_type] = gene_scores

    # Mask and normalize
    mask_tiny = gene_by_group_cosg < mask_threshold
    gene_by_group_cosg[mask_tiny] = 0
    gene_by_group_cosg = normalize(gene_by_group_cosg, norm="l1", axis=0)
    gene_by_group_cosg = pd.DataFrame(gene_by_group_cosg, columns=cell_types, index=flat_genes)
    gene_by_group_cosg[mask_tiny] = 0

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
    
    Notes
    -----
    This differs from `_calculate_ligand_receptor_specificity()` by analyzing
    the diffused interaction scores rather than individual L/R gene expression.
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

    # Retrieve names and scores from COSG results
    names = pd.DataFrame(lr_adata.uns['cosg']['names']).T
    scores = pd.DataFrame(lr_adata.uns['cosg']['scores']).T
    cell_types = names.index.values

    # Initialize lists to store the ordered gene names and scores
    ordered_genes = []
    ordered_scores = []

    # Collect genes and scores per cell type
    for cell_type in cell_types:
        gene_names = names.loc[cell_type].values
        gene_scores = scores.loc[cell_type].values
        ordered_genes.append(gene_names)
        ordered_scores.append(gene_scores)

    # Create a flat list of unique gene names
    flat_genes = np.unique(np.concatenate(ordered_genes))

    # Initialize a DataFrame to hold the final results
    df = pd.DataFrame(index=flat_genes, columns=cell_types)

    # Populate the DataFrame with scores
    for i, cell_type in enumerate(cell_types):
        gene_names = ordered_genes[i]
        gene_scores = ordered_scores[i]
        df.loc[gene_names, cell_type] = gene_scores

    # Normalize and handle tiny values
    df.fillna(0, inplace=True)
    df[df < mask_threshold] = 0
    df = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))

    return df


def _calculate_specificity_in_spatial_neighborhood(
    adata: ad.AnnData,
    lr_adata: ad.AnnData,
    groupby: str = 'CellTypes',
    use_rep_spatial: str = 'X_spatial',
    number_nearest_neighbors: int = 10,
    mu: float = 100,
    remove_lowly_expressed: bool = True,
    expressed_pct: float = 0.1,
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
    remove_lowly_expressed : bool, default=True
        Whether to filter out lowly expressed LR pairs from specificity calculation.
    expressed_pct : float, default=0.1
        Minimum percentage of cells (0 to 1) in a group that must express
        an LR pair for it to be considered in the specificity calculation.
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
    
    # Get gene expression matrix
    cellxgene = lr_adata.X.copy()
    
    # Helper function for counting non-zeros
    if sp.issparse(cellxgene):
        cellxgene.eliminate_zeros()
        get_nonzeros = lambda X: X.getnnz(axis=0)
    else:
        get_nonzeros = lambda X: np.count_nonzero(X, axis=0)
    
    # Process each cell type
    rank_stats = None
    order_i = 0
    
    for group_i in ctXct:
        # Create boolean mask for cells in this group
        idx_i = group_info == group_i
        idx_i = idx_i.values  # Convert to numpy array
        
        # Get specificity scores
        if sp.issparse(cellxgene):
            scores = genexlambda[:, order_i].toarray()[:, 0]
        else:
            scores = genexlambda[:, order_i]
        
        # Filter lowly expressed genes if requested
        if remove_lowly_expressed:
            n_cells_expressed = get_nonzeros(cellxgene[idx_i])
            n_cells_i = np.sum(idx_i)
            scores[n_cells_expressed < n_cells_i * expressed_pct] = -1
        
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
    mask_threshold: float = 1e-6
) -> pd.DataFrame:
    """
    Calculate LARIS interaction scores for all sender-receiver cell type pairs.
    
    This is the main function for computing cell type-specific ligand-receptor
    interaction scores. It integrates information from:
    1. Spatial specificity of LR pairs (from `runLARIS()`)
    2. Cell type specificity of ligand and receptor genes
    3. Diffused LR interaction scores
    4. Spatial co-localization of cell type pairs
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with gene expression and spatial information.
    lr_adata : AnnData
        AnnData object containing LR interaction scores.
    laris_lr : pd.DataFrame
        DataFrame of spatially-specific LR pairs (from `runLARIS()`).
    groupby : str, default='CellTypes'
        Column in adata.obs defining cell type groups.
    use_rep_spatial : str, default='X_spatial'
        Key in adata.obsm for spatial coordinates.
    number_nearest_neighbors : int, default=10
        Number of neighbors for spatial graph construction.
    mu : float, default=100
        Regularization parameter for COSG.
    expressed_pct : float, default=0.1
        Minimum expression fraction threshold.
    remove_lowly_expressed : bool, default=True
        Whether to filter lowly expressed genes.
    mask_threshold : float, default=1e-6
        Threshold for masking low values.
        
    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        - 'sending_celltype': Cell type sending the ligand
        - 'receiving_celltype': Cell type receiving the signal
        - 'ligand': Ligand gene name
        - 'receptor': Receptor gene name
        - 'interaction_name': Format "ligand::receptor"
        - 'interaction_score': Integrated LARIS score
        Sorted by interaction_score (descending).
        
    Examples
    --------
    >>> # Complete LARIS analysis workflow
    >>> # Step 1: Calculate LR integration scores
    >>> lr_adata = la.tl.prepareLRInteraction(adata, lr_df)
    >>> 
    >>> # Step 2: Identify spatially-specific LR pairs
    >>> laris_lr = la.tl.runLARIS(lr_adata)
    >>> 
    >>> # Step 3: Calculate cell type-specific interaction scores
    >>> laris_celltype = la.tl._calculate_laris_score_by_celltype(
    ...     adata, lr_adata, laris_lr, groupby='cell_type'
    ... )
    >>> 
    >>> # View top interactions
    >>> print(laris_celltype.head(20))
    >>> 
    >>> # Filter for specific cell type pair
    >>> endothelial_to_tumor = laris_celltype[
    ...     (laris_celltype['sending_celltype'] == 'Endothelial') &
    ...     (laris_celltype['receiving_celltype'] == 'Tumor')
    ... ]
    
    Notes
    -----
    The interaction score integrates multiple signals:
    - L and R gene cell type specificity (outer product)
    - Spatial specificity score from `runLARIS()`
    - Diffused LR score cell type distribution
    - Spatial co-localization of sending and receiving cell types
    
    Higher scores indicate stronger, more specific interactions between
    the sender and receiver cell types in the spatial context.
    """
    # Calculate the specificity score for ligand and receptor
    gene_by_group_cosg = _calculate_ligand_receptor_specificity(
        adata,
        laris_lr,
        groupby=groupby,
        mu=mu,
        expressed_pct=expressed_pct,
        remove_lowly_expressed=remove_lowly_expressed,
        mask_threshold=mask_threshold
    )
    
    # Calculate the diffused LR-score percentage
    lr_adata_percentage = lr_adata.copy()
    lr_adata_percentage.X = sp.csr_matrix(lr_adata_percentage.X)
    lr_adata_percentage.X.data = np.repeat(1, len(lr_adata_percentage.X.data))
    avg_gene_expr_lr = _compute_avg_expression(lr_adata_percentage, groupby=groupby).T
    del lr_adata_percentage
    
    # Reorder the column names
    avg_gene_expr_lr = avg_gene_expr_lr.loc[:, gene_by_group_cosg.columns]

    # Prepare DataFrame for results
    results = []
    
    # Iterate over each row in laris_lr
    for idx, row in laris_lr.iterrows():
        ligand = row['ligand']
        receptor = row['receptor']
        interaction_score = row['score']
        interaction_name = f"{ligand}::{receptor}"

        # Get the ligand-receptor diffusion score
        lr_by_group_cosg_i = avg_gene_expr_lr.loc[interaction_name]

        # Calculate the interaction matrix if ligand and receptor are in the DataFrame
        if ligand in gene_by_group_cosg.index and receptor in gene_by_group_cosg.index:
            ligand_scores = gene_by_group_cosg.loc[ligand, :]
            receptor_scores = gene_by_group_cosg.loc[receptor, :]
            interaction_matrix = np.outer(ligand_scores, receptor_scores) * np.power(interaction_score, 4)

            interaction_df = pd.DataFrame(
                interaction_matrix, 
                index=gene_by_group_cosg.columns, 
                columns=gene_by_group_cosg.columns
            ) * 1
            
            # Add the information from the LR diffusion matrix
            lr_interaction_df = pd.DataFrame(
                np.outer(lr_by_group_cosg_i, lr_by_group_cosg_i),
                index=gene_by_group_cosg.columns, 
                columns=gene_by_group_cosg.columns
            )
            
            # Combine these two interaction matrices
            interaction_df = lr_interaction_df.multiply(interaction_df)

            # Melt into a long format
            interaction_df = interaction_df.reset_index().melt(
                id_vars='index', 
                var_name='receiving_celltype', 
                value_name='interaction_score'
            )
            interaction_df.rename(columns={'index': 'sending_celltype'}, inplace=True)
            interaction_df['ligand'] = ligand
            interaction_df['receptor'] = receptor
            interaction_df['interaction_name'] = interaction_name

            results.append(interaction_df)

    # Concatenate all results into a single DataFrame if there are any results
    if results:
        res_laris = pd.concat(results, ignore_index=True)
    else:
        res_laris = pd.DataFrame(
            columns=['sending_celltype', 'receiving_celltype', 'ligand', 
                    'receptor', 'interaction_name', 'interaction_score']
        )

    res_laris.reset_index(drop=True, inplace=True)
    res_laris = res_laris.sort_values(by='interaction_score', ascending=False)

    # Reset the index
    res_laris = res_laris.reset_index(drop=True)
    
    # Change the score scale
    res_laris['interaction_score'] = res_laris['interaction_score'] * (
        0.1 / np.mean(res_laris['interaction_score'][:100])
    )
    
    # Include the spatial cell type neighborhood information
    cosg_lr_ctc = _calculate_specificity_in_spatial_neighborhood(
        adata,
        lr_adata,
        groupby=groupby,
        use_rep_spatial=use_rep_spatial,
        number_nearest_neighbors=number_nearest_neighbors,
        mu=mu,
        remove_lowly_expressed=remove_lowly_expressed,
        expressed_pct=expressed_pct,
        return_by_group=True,
        key_added='cosg_lr',
        column_delimiter='@@'
    )
    
    ctc_cosg_scores = list()
    for row in res_laris.itertuples():
        ctc_i = row.sending_celltype + '::' + row.receiving_celltype
        lr_i = row.interaction_name
        ctc_cosg_score_i = cosg_lr_ctc.loc[lr_i, ctc_i]
        ctc_cosg_scores.append(ctc_cosg_score_i)

    res_laris['interaction_score'] = res_laris['interaction_score'] * ctc_cosg_scores

    res_laris = res_laris.sort_values('interaction_score', ascending=False, ignore_index=True)
    
    return res_laris


# Define public API for utility functions (advanced users)
__all__ = [
    '_rowwise_cosine_similarity',
    '_select_top_n',
    '_pairwise_row_multiply',
    '_compute_avg_expression',
    '_build_adjacency_matrix',
    '_build_random_adjacency_matrix',
    '_generate_random_background',
    '_calculate_ligand_receptor_specificity',
    '_calculate_diffused_lr_specificity',
    '_calculate_laris_score_by_celltype',
    '_calculate_specificity_in_spatial_neighborhood',
]
