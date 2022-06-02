### load required libraries

import scipy
import sklearn as sk  # used for L2 normalization

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA #for creating PCAs
from sklearn.preprocessing import StandardScaler #for creating PCAs
from scipy.spatial import cKDTree #for calculating nearest neighbours (part of imputation)


### functions

def pearson_residuals(counts, theta=100):
    """
    Computes analytical residuals for NB model with a fixed theta,
    clipping outlier residuals to sqrt(N) as proposed in
    Lause et al. 2021 https://doi.org/10.1186/s13059-021-02451-7

    Parameters
    ----------
    counts: `matrix`
        Matrix (dense) with cells in rows and genes in columns
    theta: `int` (default: 100)
        Gene-shared overdispersion parameter
    """

    counts_sum0 = np.sum(counts, axis=0)
    counts_sum1 = np.sum(counts, axis=1)
    counts_sum = np.sum(counts)

    # get residuals
    mu = counts_sum1 @ counts_sum0 / counts_sum
    z = (counts - mu) / np.sqrt(mu + (np.square(mu) / theta))

    # clip to sqrt(n)
    n = counts.shape[0]
    z[z > np.sqrt(n)] = np.sqrt(n)
    z[z < -np.sqrt(n)] = -np.sqrt(n)

    return z


def get_hvgs(adata, no_of_hvgs=2000, theta=100, layer='spliced'):
    '''
    Function to select the top x highly variable genes (HVGs)
    from an anndata object.

    Parameters
    ----------
    adata
        Annotated data matrix
    no_of_hvgs: `int` (default: 2000)
        Number of HVGs to return
    theta: `int` (default: 100)
        Gene-shared overdispersion parameter used in pearson_residuals
    layer: `str` (default: 'spliced')
        Name of layer that is used to find the HVGs.
    '''

    ### get pearson residuals
    if scipy.sparse.issparse(adata.layers[layer]):
        residuals = pearson_residuals(adata.layers[layer].todense(), theta)
    else:
        residuals = pearson_residuals(adata.layers[layer], theta)

    ### get variance of residuals
    residuals_variance = np.var(residuals, axis=0)
    variances = pd.DataFrame({"variances": pd.Series(np.array(residuals_variance).flatten()),
                              "genes": pd.Series(np.array(adata.var_names))})

    ### get top x genes with highest variance
    hvgs = variances.sort_values(by="variances", ascending=False)[0:no_of_hvgs]["genes"].values

    return hvgs

def normalise_layers(adata, mode='combined', norm='L1', unspliced_layer='unspliced', spliced_layer='spliced', total_counts=None):
    
    """
    Normalise layers of choice in Anndata object. You can choose between an L1 and L2 normalisation. 
    Additionally, there is the option to normalise the layers combined (rather than both separately).
    
    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    mode: `str` (default: 'combined')
        Whether to normalise the layers combined ('combined') (total counts of each cell will be
        calculated using both layers) or seperate ('separate') (total counts of each cell will be
        calculated per layer).
    norm: `str` (default: 'L1')
        Whether to apply L1 normalisation ('L1') or L2 normalisation ('L2').
    unspliced_layer: `str` (default: 'unspliced')
        Name of layer that contains the unspliced counts.
    spliced_layer: `str` (default: 'spliced')
        Name of layer that contains the spliced counts.
    total_counts: `list of int` (default: None)
        XXXXX
    
    """
    
    # test if layers are not sparse but dense
    for layer in [unspliced_layer, spliced_layer]:
        if scipy.sparse.issparse(adata.layers[layer]): adata.layers[layer] = adata.layers[layer].todense()
    
    # get total counts and normalize
    if total_counts is not None:
        mean_counts = int(np.mean(total_counts))
        adata.layers[unspliced_layer] = np.asarray(adata.layers[unspliced_layer]/total_counts*mean_counts)
        adata.layers[spliced_layer] = np.asarray(adata.layers[spliced_layer]/total_counts*mean_counts)
    
    # normalize if total counts are given
    else:
        if mode=='combined':
            us_combined = np.concatenate((adata.layers[unspliced_layer], adata.layers[spliced_layer]), axis=1)
            if norm=='L1': total_counts = get_total_counts(us_combined, squared=False)
            if norm=='L2': total_counts = get_total_counts(us_combined, squared=True)
            mean_counts = int(np.mean(total_counts))
            adata.layers[unspliced_layer] = np.asarray(adata.layers[unspliced_layer].T/total_counts.flatten()*mean_counts).T
            adata.layers[spliced_layer] = np.asarray(adata.layers[spliced_layer].T/total_counts.flatten()*mean_counts).T
          
        if mode=='separate':
            for layer in [unspliced_layer, spliced_layer]:
                if norm=='L1': total_counts = get_total_counts(adata.layers[layer], squared=False)
                if norm=='L2': total_counts = get_total_counts(adata.layers[layer], squared=True)
                mean_counts = int(np.mean(total_counts))
                adata.layers[layer] = np.asarray(adata.layers[layer].T/total_counts*mean_counts).T


def get_total_counts(X, squared=False):
    
    """
    Get total counts in each row (cells).
    
    Parameters
    ----------
    X: 'np.ndarray'
        n_obs (cells) * n_vars (genes) matrix
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    squared: 'bool' (default: False)
        Whether to calculate the sum of squared counts (needed for L2 normalisation).
    
    Returns
    -------
    total_counts: `list of int` (default: None)
        List of total counts per cell.     
    """
    
    if squared == False:
        #total_counts = np.squeeze(np.asarray(X.sum(axis=1)))
        total_counts = np.asarray(X.sum(axis=1))
    if squared == True:
        #total_counts = np.squeeze(np.asarray(np.square(X).sum(axis=1)))
        total_counts = np.asarray(np.square(X).sum(axis=1))
    
    return total_counts

def get_high_us_genes(adata, minlim_u=3, minlim_s=3, unspliced_layer='unspliced', spliced_layer='spliced'):
    '''
    Function to select genes that have spliced and unspliced counts above a certain threshold. Genes of 
    which the maximum u and s count is above a set threshold are selected. Threshold varies per dataset 
    and influences the numbers of genes that are selected.
    
    Parameters
    ----------
    adata
        Annotated data matrix
    minlim_u: `int` (default: 3)
        Threshold above which the maximum unspliced counts of a gene should fall to be included in the 
        list of high US genes.
    minlim_s: `int` (default: 3)
        Threshold above which the maximum spliced counts of a gene should fall to be included in the 
        list of high US genes.
    unspliced_layer: `str` (default: 'unspliced')
        Name of layer that contains the unspliced counts.
    spliced_layer: `str` (default: 'spliced')
        Name of layer that contains the spliced counts.
    '''
    
    # test if layers are not sparse but dense
    for layer in [unspliced_layer, spliced_layer]:
        if scipy.sparse.issparse(adata.layers[layer]): adata.layers[layer] = adata.layers[layer].todense()
    
    # get high US genes
    u_genes = np.max(adata.layers[unspliced_layer], axis=0) > minlim_u
    s_genes = np.max(adata.layers[spliced_layer], axis=0) > minlim_s
    us_genes = adata.var_names[np.array(u_genes & s_genes).flatten()].values
    
    return us_genes

def impute_counts(adata, n_neighbours = 30, n_pcs = 15, layer_NN = 'spliced', unspliced_layer='unspliced', spliced_layer='spliced'):
    '''
    Function to impute the counts in the unspliced and spliced layer of an adata object. First the 
    function reduces the dimensions of the inputed layer (layer_NN) using PCA to the desired number 
    of dimensions (n_pcs). In this lower dimensional space, a selected number of neighbours (n_neighbours)
    is found for every cell. For every gene, we then impute the counts by taking the average counts 
    of all neighbours. 
    
    Parameters
    ----------
    adata
        Annotated data matrix
    n_neighbours: `int` (default: 30)
        Number of neighbours to use for imputation per cell.
    n_pcs: `int` (default: 3)
        Number of principal components (PCs) to use.
    layer_NN: `str` (default: 'spliced')
        Name of layer that is used to find the neighbours of each cell (after reducing dimension 
        using PCA).
    '''
    
    # scale layer 
    scal = StandardScaler()
    spliced_scaled = scal.fit_transform(adata.layers[layer_NN])
    
    # run PCA
    pca = PCA(n_components=n_pcs)
    pca.fit(spliced_scaled)
    pca_embedding = pca.transform(spliced_scaled)
    
    # find nearest neighbours
    NN = cKDTree(pca_embedding).query(x=pca_embedding, k=n_neighbours, n_jobs=1)[1]
    
    # impute counts using nearest neighbours (NN)
    Mu = np.nanmean(np.array(adata.layers[unspliced_layer])[NN], axis=1)
    Ms = np.nanmean(np.array(adata.layers[spliced_layer])[NN], axis=1)

    # add imputed counts to adata
    adata.layers["Ms"] = Ms
    adata.layers["Mu"] = Mu


### OLD function

def L1_normalise(adata):
    # merge spliced and unspliced per cell
    us_combined = np.concatenate((adata.layers['spliced'], adata.layers['unspliced']), axis=1)
    # L1 normalization
    us_combined_L2 = sk.preprocessing.normalize(us_combined, norm='l1')
    # replace X, U and S in adata object
    adata.layers['spliced'] = us_combined_L2[:, 0:len(adata.var_names)]
    adata.layers['unspliced'] = us_combined_L2[:, len(adata.var_names):us_combined_L2.shape[1]]
    adata.X = us_combined_L2[:, 0:len(adata.var_names)]

    return adata
