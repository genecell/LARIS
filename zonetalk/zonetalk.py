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
    Computes average expression levels for the given data.

    Parameters:
    - adata: AnnData object containing gene expression data.
    - groupby (str, optional): The column name in adata.obs to group by. Default is 'Leiden'.
    - genes (list, optional): List of genes to compute average expression for.
    - groups (list, optional): List of groups to compute average expression for.

    Returns:
    - DataFrame with average expression levels.
    
    
    ### For all the genes and all the clusters
    avg_gene_expr=computeAvgExpression(adata,
                          groupby='Leiden',

                          )

    ### For selected genes
    avg_gene_expr=computeAvgExpression(adata,
                          groupby='Leiden',
                          genes=['Gad1', 'Gad2']
                          )

    ### For selected groups
    avg_gene_expr=computeAvgExpression(adata,
                          groupby='Leiden',
                                         groups=['1','2']
                          )

    ### For selected genes and groups
    avg_gene_expr=computeAvgExpression(adata,
                          groupby='Leiden',
                          genes=['Gad1', 'Gad2'],
                                         groups=['1','2']
                          )
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
        warnings.warn("No genes or groups specified; computing average expression for all genes and groups. This may be memory-intensive for large datasets.")
    
    # If only reading data and not modifying, no need for a copy
    # If you modify data later, ensure to work on a copy to prevent unintended side-effects
    adata_copy = adata  
    
    # Filter data based on provided genes and groups
    if genes:
        adata_copy = adata_copy[:, adata_copy.var_names.isin(genes)]
    if groups:
        adata_copy = adata_copy[adata_copy.obs[groupby].isin(groups), :]

    res = pd.DataFrame(columns=adata_copy.var_names, index=adata_copy.obs[groupby].cat.categories)

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
    groupby='CellTypes',
    mu=100,
    expressed_pct=0.1,
    remove_lowly_expressed=True,
    mask_threhsold=1e-6

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
    
    return res_zonetalk
