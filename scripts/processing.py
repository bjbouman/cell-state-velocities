import scipy
import sklearn as sk  # used for L2 normalization

import numpy as np
import pandas as pd


def pearson_residuals(counts, theta=100):
    '''
    Computes analytical residuals for NB model with a fixed theta,
    clipping outlier residuals to sqrt(N) as proposed in
    Lause et al. 2021 https://doi.org/10.1186/s13059-021-02451-7

    Parameters
    ----------
    counts: `matrix`
        Matrix (dense) with cells in rows and genes in columns
    theta: `int` (default: 100)
        Gene-shared overdispersion parameter
    '''

    counts_sum0 = np.sum(counts, axis=0)
    counts_sum1 = np.sum(counts, axis=1)
    counts_sum = np.sum(counts)

    ### get residuals
    mu = counts_sum1 @ counts_sum0 / counts_sum
    z = (counts - mu) / np.sqrt(mu + (np.square(mu) / theta))

    ### clip to sqrt(n)
    n = counts.shape[0]
    z[z > np.sqrt(n)] = np.sqrt(n)
    z[z < -np.sqrt(n)] = -np.sqrt(n)

    return z


def get_hvgs(adata, no_of_hvgs=2000, theta=100):
    '''
    Function to select the top x highly variable genes (HVGs)
    from an anndata object.

    Parameters
    ----------
    adata
        Annotated data matrix
    no_of_hvgs: `int` (default: 2000)
        Number of hig
    theta: `int` (default: 100)
        Gene-shared overdispersion parameter used in pearson_residuals
    '''

    ### get pearson residuals
    if scipy.sparse.issparse(adata.X):
        residuals = pearson_residuals(adata.X.todense(), theta)
    else:
        residuals = pearson_residuals(adata.X.todense(), theta)

    ### get variance of residuals
    residuals_variance = np.var(residuals, axis=0)
    variances = pd.DataFrame({"variances": pd.Series(np.array(residuals_variance).flatten()),
                              "genes": pd.Series(np.array(adata.var_names))})

    ### get top x genes with highest variance
    hvgs = variances.sort_values(by="variances", ascending=False)[0:no_of_hvgs]["genes"].values

    return hvgs


def L1_normalise(adata):
    # merge spliced and unspliced per cell
    us_combined = np.concatenate((adata.layers['spliced'], adata.layers['unspliced']), axis=1)
    # L1 normasization
    us_combined_L2 = sk.preprocessing.normalize(us_combined, norm='l1')
    # replace X, U and S in adata object
    adata.layers['spliced'] = us_combined_L2[:, 0:len(adata.var_names)]
    adata.layers['unspliced'] = us_combined_L2[:, len(adata.var_names):us_combined_L2.shape[1]]
    adata.X = us_combined_L2[:, 0:len(adata.var_names)]

    return adata