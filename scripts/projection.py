import numpy as np
import scanpy as sc
from sklearn.metrics import pairwise_distances
import scipy


def get_trans_p(d2, sigma, density_norm=None):
    # get transition matrix
    ed2 = np.exp(-d2 / sigma ** 2)  # S1)
    z_x = np.sum(ed2, axis=0)
    z_x_z_y = None
    if density_norm is None:
        t_p = 1 / z_x * ed2
    else:
        if density_norm is True:  # densitiy normalized transition matrix:
            z_x_z_y = z_x[:, np.newaxis].dot(z_x[:, np.newaxis].T)
        else:  # densitiy normalized transition matrix given z_x_z_y
            z_x_z_y = density_norm
        z_tilde = ed2 / z_x_z_y
        np.fill_diagonal(z_tilde, 0)
        z_tilde_sum = np.sum(z_tilde, axis=0)
        t_p = 1 / z_tilde_sum * ed2 / z_x_z_y
    np.fill_diagonal(t_p, 0)
    return t_p, z_x_z_y


def change(v_t, v_0, metric="direction"):
    if metric == "direction":
        return np.array(
            [np.dot(v_0[i], v_t[i].T) / (np.linalg.norm(v_0[i]) * np.linalg.norm(v_t[i])) for i in range(v_t.shape[0])])
    if metric == "length":
        return np.array([np.linalg.norm(v_0[i]) - np.linalg.norm(v_t[i]) for i in range(v_t.shape[0])])
    else:
        print("False metric choosen. Choose direction or length.")


def get_velocity(adata, use_raw=True, key="fit"):
    S, U = adata.layers["spliced" if use_raw else "Ms"], adata.layers["unspliced" if use_raw else "Mu"]
    alpha, beta, gamma = np.array(adata.var[key + "_alpha"]), np.array(adata.var[key + "_beta"]), np.array(
        adata.var[key + "_gamma"])
    V = (beta * U) - (gamma * S)
    u_steady, s_steady = alpha / beta, alpha / gamma
    V[(U > u_steady) & (S > s_steady)] = 0
    V /= (np.nanstd(V.flatten()) * 10)
    adata.layers["velocity"] = V


#################
#      PCA      #
#################

#################
# diffusion map #
#################

def get_transition_prob_asymmetric(X0, X1, sigma, density_norm=True):
    d2 = pairwise_distances(X0, X1, metric="euclidean")
    t_p, dens_norm = get_trans_p(d2, sigma, density_norm=density_norm)
    return t_p


def diffmap_eigen(adata, sigma=50):
    if "PCs" not in adata.varm:
        print("Calculating PCA")
        sc.pp.pca(adata)

    pca_transform = adata.varm["PCs"]
    X_ = np.dot(adata.X, pca_transform)
    d2 = pairwise_distances(X_, metric="euclidean")
    # get transition matrix
    t_p, dens_norm = get_trans_p(d2, sigma, density_norm=True)
    # get eigen decomp of transition matrix
    evals, evecs = scipy.sparse.linalg.eigsh(t_p, k=10)
    return evals[::-1], evecs[:, ::-1], dens_norm


def add_proj(X_old, X_new, evecs, evals, sigma, dens_norm, pca_transform):
    t_p = get_transition_prob_asymmetric(np.dot(X_old, pca_transform), np.dot(X_new, pca_transform),
                                         sigma, density_norm=dens_norm)
    return (evecs.T.dot(t_p)).T / evals


#################
#     t-sne     #
#################

def nystrom(trans_p, proj):
    """

    Parameters
    ----------
    trans_p: 'np.ndarray'
        n*n transition probability matrix
    proj: 'np.ndarray'
        n*d projected observations

    Returns
    -------
    W: 'np.ndarray'
        n*d inferred transformation matrix
    """
    # given the relation
    # trans_p*W = proj
    # we want to recover W=trans_p^-1*Y
    p_inv = np.linalg.inv(trans_p)
    W = np.dot(p_inv, proj)
    return W


EPSILON_DBL = 1e-8


def Hbeta_NN(Di, beta, i=-1):
    Pi = np.zeros(Di.shape)
    n_neighbors = Di.shape[0]
    sum_Pi = 0.0
    for j in range(n_neighbors):
        if j != i:
            Pi[j] = np.exp(-Di[j] * beta)
            sum_Pi += Pi[j]

    if sum_Pi == 0.0:
        sum_Pi = EPSILON_DBL
    sum_disti_Pi = 0.0
    # print()
    # print(Pi)
    for j in range(n_neighbors):
        Pi[j] /= sum_Pi
        sum_disti_Pi += Di[j] * Pi[j]
    # print(Pi)
    entropy = np.log(sum_Pi) + beta * sum_disti_Pi
    return Pi, entropy, sum_Pi


def tsne_d2p_NN(D, NN, u=30, tol=1e-4, row_norm=True):
    """
    d2p Identifies appropriate sigma's to get k NNs up to some tolerance,
    and turns a distance matrix d to a transition probability matrix P

    Identifies the required precision (= 1 / variance^2) to obtain a Gaussian
    kernel with a certain uncertainty for every datapoint. The desired
    uncertainty can be specified through the perplexity u (default = 30). The
    desired perplexity is obtained up to some tolerance that can be specified
    by tol (default = 1e-4).
    The function returns the final Gaussian kernel in P, as well as the
    employed precisions per instance in beta.


    (C) Laurens van der Maaten, 2008
    Maastricht University

    Parameters
    ----------
    D: 'np.ndarray'
        n*k distance matrix to k nearest neighbors
    u: 'int' (default:30)
        perplexity
    tol: 'float' (default:1e-4)
        tolerance
    self: 'bool' (default: False)
        whether the distances are to themselves
    Returns
    -------

    """
    logU = np.log(u)  # note / todo change to log2 if shannon entropy changed back to log2
    beta = np.ones(D.shape[0])  # corresponds to 1/2\rho^2
    n_loop = 50
    sumP = np.zeros(D.shape[0], dtype=np.float64)

    # subset distances to nearest neighbors
    D = np.array([D[i, NN[i]] for i in range(D.shape[0])])
    P = np.zeros(D.shape, dtype=np.float64)

    for i in range(D.shape[0]):  # for each row, == for every datapoint
        betamin, betamax = -np.infty, np.infty
        for _ in range(n_loop):
            thisP, HPi, sumPi = Hbeta_NN(D[i, :], beta[i])
            # range of beta
            # test whether perplexity is whithin tolerance
            Hdiff = HPi - logU

            if np.abs(Hdiff) <= tol:
                break

            # if not, increase or decrease precision
            if Hdiff > 0:  # increase precision
                betamin = beta[i]
                beta[i] = beta[i] * 2 if betamax == np.infty else (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                beta[i] = beta[i] / 2 if betamin == -np.infty else (beta[i] + betamin) / 2
        P[i, :] = thisP  # set row to final value
        sumP[i] = sumPi

    if row_norm:
        # P /= np.sum(P, axis=0)
        P = (P.T / np.sum(P, axis=1)).T

    # transform back to n*n transition p matrix
    # todo find better way to do this, this must be very slow
    P_ = np.zeros((D.shape[0], D.shape[0]), dtype=np.float64)
    for i in range(D.shape[0]):
        P_[i, NN[i]] = P[i]

    return P_  # , beta, sumP


def get_duplicate_row(a):
    unq, count = np.unique(a, axis=0, return_counts=True)
    repeated_groups = unq[count > 1]
    drop = []

    for repeated_group in repeated_groups:
        repeated_idx = np.argwhere(np.all(a == repeated_group, axis=1))
        drop.extend(repeated_idx.tolist()[0])
    return drop


def tsne_decomp(P, no_dims=2, max_iter=500):
    # Initialize variables
    n = P.shape[0]

    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01

    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.  # early exaggeration
    P = np.maximum(P, 1e-12)

    # check early break if no more
    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y


########################################################################################################################
#                                                      deprecated                                                      #
########################################################################################################################

def Hbeta_deprecated(Di, betai, NN=False):
    """

    Parameters
    ----------
    Di: 'np.array'
        row of distance matrix
    betai: 'np.array'
        = 1/(2*sigma^2)
    Returns
    -------
    Pi: 'np.array'
        row of transition probability matrix for that observed variable
    HPi: 'float'
        shannon entropy value for that variable
    """
    eDi = np.exp(-Di * betai)
    # eDi[i]=0 # diagonal is zero
    sum_eDi = np.sum(eDi) - 1  # if not NN else np.sum(eDi)  # -1 bc diagonal should be zero
    # HPi is shannon entropy of Pi
    # note / todo the following simplification is only true if we take natural log in shannon entropy instead of base2
    # in the 2008 paper they take base 2 log
    HPi = np.log(sum_eDi) + betai * np.sum(Di * eDi) / sum_eDi
    Pi = eDi / sum_eDi
    return Pi, HPi


def tsne_d2p_deprecated(D, u=30, tol=1e-4, row_norm=False):
    """
    d2p Identifies appropriate sigma's to get k NNs up to some tolerance,
    and turns a distance matrix d to a transition probability matrix P

    Identifies the required precision (= 1 / variance^2) to obtain a Gaussian
    kernel with a certain uncertainty for every datapoint. The desired
    uncertainty can be specified through the perplexity u (default = 30). The
    desired perplexity is obtained up to some tolerance that can be specified
    by tol (default = 1e-4).
    The function returns the final Gaussian kernel in P, as well as the
    employed precisions per instance in beta.


    (C) Laurens van der Maaten, 2008
    Maastricht University

    Parameters
    ----------
    D: 'np.ndarray'
        n*n distance matrix
    u: 'int' (default:30)
        perplexity
    tol: 'float' (default:1e-4)
        tolerance
    row_norm: 'bool' (default: False)
        whether to normalise by rows
        this is necessary if we want to use the transition matrix for diffusion maps
    Returns
    -------

    """
    P = np.zeros(D.shape)
    logU = np.log(u)  # note / todo change to log2 if shannon entropy changed back to log2
    beta = np.ones(D.shape[0])  # corresponds to 1/2\rho^2

    for i in range(D.shape[0]):  # for each row, == for every datapoint
        thisP, HPi = Hbeta(D[i, :], beta[i])

        # range of beta
        betamin, betamax = -np.infty, np.infty
        # test whether perplexity is whithin tolerance
        Hdiff = HPi - logU
        loop = 0
        while (np.abs(Hdiff) > tol) & (loop < 50):
            # if not, increase or decrease precision
            if Hdiff > 0:  # increase precision
                betamin = beta[i]
                beta[i] = beta[i] ** 2 if betamax == np.infty else (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                beta[i] = beta[i] / 2 if betamin == -np.infty else (beta[i] + betamin) / 2
            thisP, HPi = Hbeta(D[i, :], beta[i])
            Hdiff = HPi - logU
            loop += 1
        P[i, :] = thisP  # set row to final value

    np.fill_diagonal(P, 0)

    if row_norm:
        P /= np.sum(P, axis=0)

    return P, beta
