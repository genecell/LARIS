import pandas as pd

import scanpy as sc
import pandas as pd
import os
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.neighbors import kneighbors_graph
import numpy as np
import scanpy as sc

### Calculate LR interaction score
def calculateLigandReceptorIntegrationScore(
    adata,
    lr_df,
    number_nearest_neighbors=10,
    use_rep_spatial='X_spatial'
):
    
    X_spatial = adata.obsm[use_rep_spatial].copy()
    # Create the diffused ligand/receptor matrix
    cellxcell = kneighbors_graph(X_spatial, n_neighbors=number_nearest_neighbors, mode='distance', include_self=False)
    cellxcell.data = 1 / np.exp(cellxcell.data / (np.mean(cellxcell.data) / 2))

    genexcell = adata.X.copy().T
    order1 = genexcell @ cellxcell.T

    # Estimate diffused ligand-receptor activity
    sorter = np.argsort(adata.var_names)
    ligand_idx = sorter[np.searchsorted(adata.var_names, lr_df['ligand'], sorter=sorter)]
    receptor_idx = sorter[np.searchsorted(adata.var_names, lr_df['receptor'], sorter=sorter)]
    ### Element-wise multiplication
    lrxcell = order1[ligand_idx, :].multiply(order1[receptor_idx, :])

    # Create an AnnData object
    lr_names = lr_df['ligand'].astype(str) + '::' + lr_df['receptor'].astype(str)
    lr_adata = sc.AnnData(lrxcell.T)
    lr_adata.obs = adata.obs.copy()
    lr_adata.obsm = adata.obsm.copy()
    lr_adata.var_names = lr_names

    return lr_adata



def _select_top_n(scores, n_top):
    reference_indices = np.arange(scores.shape[0], dtype=int)
    partition = np.argpartition(scores, -n_top)[-n_top:]
    partial_indices = np.argsort(scores[partition])[::-1]
    global_indices = reference_indices[partition][partial_indices]
    return global_indices

### Calculate average expression
def computeAvgExpression(adata, groupby='Leiden', genes=None, groups=None):
    """
    Computes the average expression levels of genes across cell groups.

    This function calculates the mean expression values of genes for each group 
    of cells defined by `groupby`. It allows selection of specific genes and/or 
    groups for computation.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing gene expression data.
    groupby : str, optional (default: 'Leiden')
        The column name in `adata.obs` that defines the cell groups. 
        Typically, this is a clustering column (e.g., 'Leiden', 'louvain', or 'cell_type').
    genes : list of str, optional (default: None)
        A list of gene names to compute the average expression for. 
        If None, all genes in `adata.var_names` are included.
    groups : list of str, optional (default: None)
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
    
    >>> avg_expr = computeAvgExpression(adata, groupby='Leiden')

    Compute the average expression of specific genes across all groups:
    
    >>> avg_expr = computeAvgExpression(adata, groupby='Leiden', genes=['Gad1', 'Gad2'])

    Compute the average expression of all genes in selected groups:
    
    >>> avg_expr = computeAvgExpression(adata, groupby='Leiden', groups=['1', '2'])

    Compute the average expression of selected genes in selected groups:
    
    >>> avg_expr = computeAvgExpression(adata, groupby='Leiden', genes=['Gad1', 'Gad2'], groups=['1', '2'])
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

### Compute adjacency martix given the low-dimensional representations
def buildADJ(adata,
            use_rep:str='X_pca',
             n_nearest_neighbors:int=10,
             sigma:float=100.0
            ):
    
    X_rep=adata.obsm[use_rep].copy()
    from sklearn.neighbors import kneighbors_graph
    cellxcell = kneighbors_graph(X_rep,
                         n_neighbors=n_nearest_neighbors,
                         mode='distance', include_self=False)
    
    cellxcell.data=1/np.exp(cellxcell.data/sigma)
    
    return cellxcell


from scipy.sparse import issparse
def rowwise_cosine_similarityV2(A, B):
    """
    Computes the cosine similarity between corresponding rows of matrices A and B.

    Parameters:
    - A (numpy.ndarray or scipy.sparse.csr_matrix): A 2D array or CSR sparse matrix representing matrix A with shape (n, m).
    - B (numpy.ndarray or scipy.sparse.csr_matrix): A 2D array or CSR sparse matrix representing matrix B with shape (n, m).

    Returns:
    - cosine_similarities (numpy.ndarray): A 1D array of shape (n,) containing the cosine similarity between corresponding rows of A and B.
    """

    if A.shape != B.shape:
        raise ValueError(f"Matrices A and B must have the same shape. Got A.shape = {A.shape}, B.shape = {B.shape}.")

    if issparse(A) and issparse(B):
        dot_products = A.multiply(B).sum(axis=1).A1  # A1 converts matrix to a flat array
        norm_A = np.sqrt(A.multiply(A).sum(axis=1)).A1
        norm_B = np.sqrt(B.multiply(B).sum(axis=1)).A1
    else:
        dot_products = np.einsum('ij,ij->i', A, B)  # It's faster than (A * B).sum(axis=1)
        norm_A = np.linalg.norm(A, axis=1)
        norm_B = np.linalg.norm(B, axis=1)
    
    denominator = norm_A * norm_B
    # cosine_similarities = np.where(denominator!=0, dot_products / denominator, 0)
    cosine_similarities = np.divide(dot_products, denominator, out=np.zeros_like(dot_products), where=denominator!=0)
    
    return cosine_similarities



from scipy.sparse import csr_matrix
### To efficiently create a random adjacency matrix
def buildRandomADJ(
    adata,
    cellxcell,
    n_nearest_neighbors:int=10,
    random_seed:int=0,
            ):
    
    row_ind=np.repeat(np.arange(adata.n_obs), n_nearest_neighbors)
    np.random.seed(random_seed)
    col_ind=np.random.choice(np.arange(adata.n_obs), adata.n_obs*n_nearest_neighbors, replace=True)
    
    ### Shuffle the weights
    connectivity=cellxcell.data.copy()
    np.random.shuffle(connectivity)

    cellxcell_shuffle=csr_matrix((connectivity,(row_ind, col_ind)),shape=cellxcell.shape)
    
    return cellxcell_shuffle

def generateRandomBackground(
    adata,
    cellxcell,
    genexcell,
    n_nearest_neighbors:int=10,
    n_repeats:int=30,
    random_seed:int=0,
    
):
    random_gsp_list=[]
    np.random.seed(random_seed)
    
    random_seed_list=np.random.choice(1902772, size=n_repeats, replace=False)
    
    for i in np.arange(n_repeats):
        random_seed=random_seed_list[i]
    
        cellxcell_shuffle=buildRandomADJ(adata,
                                         cellxcell,
                                         n_nearest_neighbors=n_nearest_neighbors,
                                         random_seed=random_seed)
        
        ### Need to normalize the weights, then the add up would be comparable, l1 normalization for each row
        cellxcell_shuffle=normalize(cellxcell_shuffle, axis=1, norm='l1')
        
    
        random_order1=genexcell @ cellxcell_shuffle.T


        random_gsp_list.append(rowwise_cosine_similarityV2(genexcell, random_order1))
    
    return(random_gsp_list)


### Identifying DR-specific genes
from sklearn.preprocessing import normalize
def runZoneTalk(
    lr_adata,
    use_rep:str='X_spatial',
    # layer:str='log1p',
    n_nearest_neighbors:int=10,
    random_seed:int=27,
    n_repeats:int=3,
    mu:float=1,
    sigma:float=100,
    remove_lowly_expressed=True,
    expressed_pct=0.1,
    n_cells_expressed_threshold:int=100,
    n_top_lr:int=4000
                
            ):

    cellxcell=buildADJ(lr_adata,
                   use_rep=use_rep,
                   n_nearest_neighbors=n_nearest_neighbors,
                       sigma=sigma,
                  )
    # if layer in adata.layers:
    #     genexcell=adata.layers[layer].T
    # else:
    #     print('Warning: the specified layer', layer, 'is not found, using adata.X instead.')
    #     genexcell=adata.X.T
    
    genexcell=lr_adata.X.T
    order1=genexcell @ cellxcell.T    
    gsp=rowwise_cosine_similarityV2(genexcell, order1)

    ### Build the random background by sampling
    random_gsp_list=generateRandomBackground(
        lr_adata, cellxcell, genexcell,
        n_nearest_neighbors=n_nearest_neighbors,
        n_repeats=n_repeats,
        random_seed=random_seed
    )
    
    
    random_gsp=np.mean(random_gsp_list,axis=0)
    

    lr_adata.var['LRSS_Target']=np.array(gsp).ravel()
    lr_adata.var['LRSS_Random']=np.array(random_gsp).ravel()
    
    gsp_score=gsp-mu*random_gsp
    gsp_score=np.array(gsp_score).ravel()

    if remove_lowly_expressed:
        ### Remaining to be written
        lr_adata.var['LR_SpatialSpecificity']=gsp_score
        # adata.var['MarkG'][~adata.var['knn_highly_expressed'].values]=min(gsp_score)-1
        
        
    else:
        lr_adata.var['LR_SpatialSpecificity']=gsp_score
    # print ('The MarkG score for all the genes are save in `MarkG` in `adata.var`')

    sc.pp.calculate_qc_metrics(lr_adata, inplace=True)
    lr_var=lr_adata.var.sort_values(by='LR_SpatialSpecificity', ascending=False).copy()
    n_cells_expressed=lr_var['n_cells_by_counts'].values
    # n_cells_expressed<n_cells_expressed_threshold
    # np.where(n_cells_expressed<threshold)[0]
    gsp_score_for_ranking=lr_var['LR_SpatialSpecificity'].values
    gsp_score_for_ranking[np.where(n_cells_expressed<n_cells_expressed_threshold)[0]] = np.min(gsp_score_for_ranking)-0.001

    spag_list=lr_var.index.values[_select_top_n(gsp_score_for_ranking,n_top_lr)]
    spag_list_ligand=[np.str_.split(i,'::')[0] for i in spag_list]
    spag_list_receptor=[np.str_.split(i,'::')[1] for i in spag_list]
    spag_list_score=gsp_score_for_ranking[_select_top_n(gsp_score_for_ranking, n_top_lr)]
    
    zonetalk_lr = pd.DataFrame({'ligand': spag_list_ligand, 'receptor': spag_list_receptor, 'score': spag_list_score})
    zonetalk_lr.index=zonetalk_lr['ligand']+'::'+zonetalk_lr['receptor']
    zonetalk_lr['Rank']=np.arange(zonetalk_lr.shape[0])
    
    return(zonetalk_lr)

    
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import cosg

### Calculate the specificity of ligand and receptor at gene level
def calculateLigandReceptorSpecificity(
    adata,
    zonetalk_lr,
    groupby='CellTypes',
    mu=100,
    expressed_pct=0.1,
    remove_lowly_expressed=True,
    mask_threhsold=1e-6
):
    # Extract unique ligands and receptors
    lr_set = np.unique(np.hstack([zonetalk_lr['ligand'].values, zonetalk_lr['receptor'].values]))
    lr_gene_adata = adata[:, lr_set].copy()

    # Perform COSG analysis
    cosg.cosg(lr_gene_adata,
              key_added='cosg',
              mu=mu,
              expressed_pct=expressed_pct,
              remove_lowly_expressed=remove_lowly_expressed,
              n_genes_user=lr_gene_adata.n_vars,
              groupby=groupby)

    
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
    mask_tiny = gene_by_group_cosg < mask_threhsold
    gene_by_group_cosg[mask_tiny] = 0
    gene_by_group_cosg = normalize(gene_by_group_cosg, norm="l1", axis=0)
    gene_by_group_cosg = pd.DataFrame(gene_by_group_cosg, columns=cell_types, index=flat_genes)
    gene_by_group_cosg[mask_tiny] = 0

    return gene_by_group_cosg



import numpy as np
import pandas as pd
import cosg 
### Calculate the specificity of diffused ligand and receptor interaction scores
def calculateDiffusedLigandReceptorSpecificity(
    lr_adata,
    groupby='CellTypes',
    mu=100,
    expressed_pct=0.05,
    remove_lowly_expressed=True,
    mask_threshold=1e-6
):
    
    # Perform COSG analysis
    cosg.cosg(lr_adata,
              key_added='cosg',
              mu=mu,
              expressed_pct=expressed_pct,
              remove_lowly_expressed=remove_lowly_expressed,
              n_genes_user=lr_adata.n_vars,
              groupby=groupby)

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

### Example
# lr_by_group_cosg=zonetalk.calculateDiffusedLigandReceptorSpecificity(
#     lr_adata,
#     groupby='CellTypes',
#     mu=100,
#     expressed_pct=0.05,
#     remove_lowly_expressed=True,
#     mask_threshold=1e-6
# )


import numpy as np
import pandas as pd
from scipy import sparse

def calculateZoneTalkScoreByCellType(
    adata,
    lr_adata,
    zonetalk_lr,
    groupby:str='CellTypes',
    use_rep_spatial: str = 'X_spatial', 
    number_nearest_neighbors: int = 10, 
    mu:float=100,
    expressed_pct:float=0.1,
    remove_lowly_expressed:bool=True,
    mask_threhsold:float=1e-6

):
    
    
    ### If using the percentage of gene expression for L and R, run the following codes
    # # Extract unique ligands and receptors
    # lr_set = np.unique(np.hstack([zonetalk_lr['ligand'].values, zonetalk_lr['receptor'].values]))
    # lr_gene_adata = adata[:, lr_set].copy()
    # # Calculate the expression percentage for genes
    # lr_gene_adata_percentage = lr_gene_adata.copy()
    # lr_gene_adata_percentage.X.data = np.repeat(1, len(lr_gene_adata_percentage.X.data))
    # avg_gene_expr = computeAvgExpression(lr_gene_adata, groupby=groupby).T
    # del lr_gene_adata_percentage
    
    
    ### Calculate the specificity score for ligand and receptor
    gene_by_group_cosg=calculateLigandReceptorSpecificity(
        adata,
        zonetalk_lr,
        groupby=groupby,
        mu=mu,
        expressed_pct=expressed_pct,
        remove_lowly_expressed=remove_lowly_expressed,
        mask_threhsold=mask_threhsold
    )
    
    # Calculate the diffused LR-score percentage
    lr_adata_percentage = lr_adata.copy()
    lr_adata_percentage.X = sparse.csr_matrix(lr_adata_percentage.X)
    lr_adata_percentage.X.data = np.repeat(1, len(lr_adata_percentage.X.data))
    avg_gene_expr_lr = computeAvgExpression(lr_adata_percentage, groupby=groupby).T
    del lr_adata_percentage
    ### Reorder the column names
    # print(avg_gene_expr_lr.columns)
    # print(gene_by_group_cosg.columns)
    avg_gene_expr_lr=avg_gene_expr_lr.loc[:,gene_by_group_cosg.columns]
    # print(avg_gene_expr_lr.columns)

    # Prepare DataFrame for results
    results = []
    # Iterate over each row in zonetalk_lr
    for idx, row in zonetalk_lr.iterrows():
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

            interaction_df = pd.DataFrame(interaction_matrix, index=gene_by_group_cosg.columns, columns=gene_by_group_cosg.columns) * 1
            
            # Add the information from the LR diffusion matrix
            # print(lr_by_group_cosg_i.shape)
            lr_interaction_df = pd.DataFrame(np.outer(lr_by_group_cosg_i, lr_by_group_cosg_i),
                                             index=gene_by_group_cosg.columns, columns=gene_by_group_cosg.columns)
            # Combine these two interaction matrices
            interaction_df = lr_interaction_df.multiply(interaction_df)

            # Melt into a long format
            interaction_df = interaction_df.reset_index().melt(id_vars='index', var_name='receiving_celltype', value_name='interaction_score')
            interaction_df.rename(columns={'index': 'sending_celltype'}, inplace=True)
            interaction_df['ligand'] = ligand
            interaction_df['receptor'] = receptor
            interaction_df['interaction_name'] = interaction_name

            results.append(interaction_df)

    # Concatenate all results into a single DataFrame if there are any results
    if results:
        res_zonetalk = pd.concat(results, ignore_index=True)
    else:
        res_zonetalk = pd.DataFrame(columns=['sending_celltype', 'receiving_celltype', 'ligand', 'receptor', 'interaction_name', 'interaction_score'])

    res_zonetalk.reset_index(drop=True, inplace=True)
    res_zonetalk = res_zonetalk.sort_values(by='interaction_score', ascending=False)

    ## Reset the index
    res_zonetalk = res_zonetalk.reset_index(drop=True)
    
    ## Change the score scale
    res_zonetalk['interaction_score']=res_zonetalk['interaction_score']*(0.1/np.mean(res_zonetalk['interaction_score'][:100]))
    
    
    ### Include the spatial cell type neighborhood information
    cosg_lr_ctc=calculateSpecificityInSpatialCellTypeNeighborhood(
        adata,
        lr_adata,
        groupby = groupby, 
        use_rep_spatial = use_rep_spatial, 
        number_nearest_neighbors = number_nearest_neighbors, 
        mu = mu, 
        remove_lowly_expressed = remove_lowly_expressed, 
        expressed_pct = expressed_pct, 
        return_by_group= True, 
        key_added = 'cosg_lr', 
        column_delimiter = '@@'
    )
    
    ctc_cosg_scores=list()
    for row in res_zonetalk.itertuples():
        ctc_i=row.sending_celltype+'::'+row.receiving_celltype
        lr_i=row.interaction_name
        ctc_cosg_score_i=cosg_lr_ctc.loc[lr_i, ctc_i]
        ctc_cosg_scores.append(ctc_cosg_score_i)

    res_zonetalk['interaction_score']=res_zonetalk['interaction_score']*ctc_cosg_scores

    res_zonetalk=res_zonetalk.sort_values('interaction_score', ascending=False, ignore_index=True)
    
    return res_zonetalk



### Include the cell type spatial distribution information
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity
import cosg
from typing import Optional, Union, List, Tuple, Dict, Any

    
# Helper function for pairwise row multiplication
def _pairwise_row_multiply_matrix(
    sparse_matrix: sp.csr_matrix, 
    cell_types: List[str], 
    delimiter: str = "::"
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Calculate element-wise multiplication between all pairs of rows in a sparse CSR matrix,
    including self-interactions, and track the cell type combinations.

    Parameters:
    -----------
    sparse_matrix : scipy.sparse.csr_matrix
        Input sparse matrix of shape (N, M) where N is the number of cell types
    cell_types : array-like
        Array of cell type names, length N
    delimiter : str, default="::"
        Delimiter to use when joining cell type pair names

    Returns:
    --------
    result : scipy.sparse.csr_matrix
        Sparse matrix of shape (N*N, M) containing all pairwise multiplications
    row_names : numpy.ndarray
        Array of strings representing the cell type pairs for each row,
        in the format "cell_type_i::cell_type_j"

    Raises:
    -------
    ValueError
        If length of cell_types doesn't match number of rows in sparse_matrix
    TypeError
        If sparse_matrix is not a sparse matrix
    """
    N, M = sparse_matrix.shape

    # Validate input
    if len(cell_types) != N:
        raise ValueError(f"Length of cell_types ({len(cell_types)}) must match number of rows in matrix ({N})")

    # Ensure the input is in CSR format for efficient row slicing
    if not sp.isspmatrix_csr(sparse_matrix):
        sparse_matrix = sparse_matrix.tocsr()

    # Preallocate a list to store all the row products and row names
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
### Include the cell type spatial distribution information
def calculateSpecificityInSpatialCellTypeNeighborhood(
    adata,
    lr_adata,
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
    Calculate cell type specificity scores for ligand-receptor interaction in spatial neighborhoods.
    
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing spatial information and cell type annotations.
        Must have `.obsm[use_rep_spatial]` containing spatial coordinates and `.obs[groupby]` with cell type annotations.
    lr_adata : AnnData
        Annotated data matrix containing ligand-receptor interaction scores at single-cell resolution.
        Must have `.X` containing ligand-receptor interaction scores and `.var_names` with ligand-receptor pair identifiers.
    groupby : str, default='CellTypes'
        Key in adata.obs for cell type/group annotations.
    use_rep_spatial : str, default='X_spatial'
        Key in adata.obsm for spatial coordinates.
    number_nearest_neighbors : int, default=10
        Number of neighbors to consider in the spatial neighborhood. Must be positive.
    mu : float, default=100
        Regularization parameter for cosine similarity calculation. Must be non-negative.
    remove_lowly_expressed : bool, default=True
        Whether to filter out lowly expressed genes.
    expressed_pct : float, default=0.1
        Minimum percentage of cells in a group that must express a gene. Must be between 0 and 1.
    return_by_group : bool, default=True
        Whether to return results organized by group.
    key_added : str, default='cosg'
        Key to add to adata.uns for storing results.
    column_delimiter : str, default='@@'
        Delimiter for column names in the returned DataFrame.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing gene-wise specificity scores normalized across all cell types.
    
    Raises
    ------
    ValueError
        If parameters have invalid values or required data is missing from AnnData objects.
    TypeError
        If parameters have incorrect types.
    
    Notes
    -----
    The function adds results to lr_adata.uns[key_added] and returns a DataFrame of normalized scores.
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
    cellxcell = kneighbors_graph(X_spatial, n_neighbors=number_nearest_neighbors, 
                                mode='distance', include_self=False)
    cellxcell.data = 1 / np.exp(cellxcell.data / (np.mean(cellxcell.data) / 2))
    
    # Calculate first-order neighborhood composition
    order1 = cluster_mat @ cellxcell.T
    
    
    
    # Get cell type pair interactions
    ctXct_cell, ctXct = _pairwise_row_multiply_matrix(order1, cell_types=groups_order)
    
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
        genexlambda.data = genexlambda.data / ((1 - mu) * genexlambda.data + mu * np.repeat(e_power2_sum, np.diff(genexlambda.indptr)))
    
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
