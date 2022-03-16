import numpy as np
import scvelo as scv
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy
import pandas as pd
import sklearn as sk #used for L2 normalization
import sklearn.preprocessing

from scvelo.tools.dynamical_model_utils import mRNA, vectorize, get_vars


def f_u(alpha, beta, u, u0):
    u_adj = np.log((u - (alpha / beta)) / (u0 - (alpha / beta)))
    return -1 / beta * u_adj


def f_s(alpha, beta, gamma, u, s, u0, s0):
    num = s + ((alpha - (beta * u)) / (gamma - beta)) - (alpha / gamma)
    denum = s0 + ((alpha - (beta * u0)) / (gamma - beta)) - (alpha / gamma)
    adj = np.log(num / denum)
    return -1 / gamma * adj


def u_t(alpha, beta, t, u0):
    return u0 * np.exp(-beta * t) + alpha / beta * (1 - np.exp(-beta * t))


def s_t(alpha, beta, gamma, t, u0, s0):
    return s0 * np.exp(-gamma * t) + (alpha / gamma) * (1 - np.exp(-gamma * t)) + \
           (alpha - beta * u0) / (gamma - beta) * (np.exp(-gamma * t) - np.exp(-beta * t))


def get_pars(adata, basis, use_raw=False):
    idx = adata.var_names.get_loc(basis) if isinstance(basis, str) else basis
    alpha, beta, gamma, scaling, t_ = get_vars(adata[:, basis], key="fit")
    if "fit_u0" in adata.var.keys():
        u0_offset, s0_offset = adata.var["fit_u0"][idx], adata.var["fit_s0"][idx]
    else:
        u0_offset, s0_offset = 0, 0
    t = adata.layers["fit_t"][:, idx]
    tau, alpha_, u0, s0 = vectorize(t, t_, alpha, beta, gamma)
    u_switch, s_switch = mRNA(t_, u0_offset, s0_offset, alpha, beta, gamma)
    ut, st = mRNA(tau, u0, s0, alpha_, beta, gamma)
    u_switch *= scaling
    ut, st = ut * scaling + u0_offset, st + s0_offset
    ignore = .1  # we need to remove those bc the density approximation becomes tricky towards the exp function limit
    upper = (st > ignore * np.max(st)) | (ut > ignore * np.max(ut))
    lower = (st < (1 - ignore) * np.max(st)) | (ut < (1 - ignore) * np.max(ut))
    down_reg = np.array((t > t_) & upper & lower)
    up_reg = np.array((t < t_) & upper & lower)
    beta /= scaling
    # u, s=adata.layers["unspliced" if use_raw else "Ms"][:, idx], adata.layers["spliced" if use_raw else "Ms"][:, idx]
    return alpha, beta, gamma, ut, st, u0_offset, s0_offset, u_switch, s_switch, up_reg, down_reg


def sample(n, ut, reg, up_reg_idx):
    i = np.random.choice(up_reg_idx, (2, n))
    where = ut[i[0]] > ut[i[1]] if reg == "up" else ut[i[0]] < ut[i[1]]
    i[:, where] = i[:, where][[1, 0]]  # adapt order of sampled cells
    return i


kwargs = dict(bounds=[(0.1, None), (0.1, None), (0.1, None), (0.01, None)], x0=np.array([.1, 1, 1, .1]),
              options={"maxiter": 2000, 'disp': True}, tol=1e-8, method="COBYLA")


def get_f_and_delta_t(ut, st, alpha, beta, gamma, r_, reg, mode="u"):
    ordr = np.argsort(ut[r_]) if reg == "up" else np.argsort(-ut[r_])
    reg_idx = np.where(r_)[0][ordr]
    loc_idx = np.array([np.where(reg_idx == i)[0][0] if i in reg_idx else np.nan for i in
                        range(np.max(reg_idx) + 1)])  # get index of cell on that side of the almond
    i = sample(5000, ut, reg, reg_idx)  # sampling of 10k random pairs
    t_dist = np.abs(loc_idx[i[1]] - loc_idx[i[0]])  # number of cells between i[0] and i[1]
    (u0, u1) = ut[i[0]], ut[i[1]]  # u values for i[0] and i[1]
    if True:  # mode == "u":
        f = f_u(alpha if reg == "up" else 0, beta, u1, u0)
    else:
        (s0, s1) = (st[i[0]], st[i[1]])
        f = f_s(alpha if reg == "up" else 0, beta, gamma, u1, s1, u0, s0)
    return t_dist, f


# density kappa estimation
def get_intervals(adata, basis, mode="s", reg="both", use_raw=False):
    """
    Parameters
    ----------
    use_raw
    adata:: :class:`~anndata.AnnData`
        Annotated data matrix on which to compute the kappa estimates
    basis: `str` or `int
        gene name (str) or index (int) for which the kappa estimates should be computed (todo: vectorize if basis=[])
    n: `int`
        width of bin (in cells)
    mode: `str` in ['s', 'u', 'both']
        compute the kappa estimates on spliced values (s), unspliced values (u) or on both
    reg: `str` in ['up', 'down', 'both']
        compute the kappa estimates for up- or down- regulation only or for both
    handle_steady: `str` in ['hard', 'var']
    thres_up, thres_down: `float` in [0, .5[
        threshold for hard steady-state cutoff

    Returns
    -------
    array of kappa estimates
    """
    kappa_u_up, kappa_s_up = [], []
    kappa_u_down, kappa_s_down = [], []

    # display error if input incorrect
    # mode can be u, s or both to return all u_kappa, all s_kappa or both (simple concatenation)
    if mode not in ["s", "u", "both"]:
        print("error: mode should be \"u\", \"s\" or \"both\"")
        return
    if reg not in ["up", "down", "both"]:
        print("error: reg should be \"up\", \"down\" or \"both\" depending on whether we should calculate the kappa "
              "estimates for up- or down- regulation only or for both.")
        return
    reg_ = [reg] if reg != "both" else ["up", "down"]

    # get parameters for each gene
    alpha, beta, gamma, ut, st, _, _, _, _, up_reg, down_reg = get_pars(adata, basis, use_raw)
    # todo filter : check if gene can be recovered

    for reg in reg_:
        r_ = up_reg if reg == "up" else down_reg

        # at least 30% of the cells need to be in the considered transient state
        if np.sum(r_) > 0.40 * (np.sum(up_reg) + np.sum(down_reg)):
            t_dist, f = get_f_and_delta_t(ut, st, alpha, beta, gamma, r_, reg, mode)
            k = get_slope(t_dist, f)

            if mode != "s":
                if reg == "up":
                    kappa_u_up = [k]
                else:
                    kappa_u_down = [k]
            if mode != "u":
                if reg == "up":
                    kappa_s_up = [k]
                else:
                    kappa_s_down = [k]

    kappa_u_ = np.concatenate((kappa_u_up, kappa_u_down))
    kappa_s_ = np.concatenate((kappa_s_up, kappa_s_down))
    # return depending on the mode
    if mode == "u":
        return kappa_u_
    if mode == "s":
        return kappa_s_
    if mode == "both":
        return np.concatenate((kappa_u_, kappa_s_))


def get_slope(x, y):
    mn = op.minimize(cost_parallelogram, args=(x / np.max(x), y / np.max(y)), **kwargs)
    a, b, c, d = mn.x
    return (b * np.max(y)) / (a * np.max(x))


def dist_pt_line(a, b, x1, y1):
    # line given by a*x+b, point (x1, y1)
    # perpendicular to ax+b:
    #   is -x/a+h
    #   passes through pt x1, y1
    #   -x1/a+h=y1 <-> h = y1+x1/a
    h = y1 + (x1 / a)
    # we want point where perpendicular -x/a+h crosses a*x+b
    #   -x/a+h = a*x+b <-> x*(a+1/a)=h-b
    #                  <-> x=(h-b)/(a+(1/a))
    #                  <-> y=(h-b)/(a+(1/a))*a+b
    x_, y_ = (h - b) / (a + (1 / a)), (h - b) / (a + (1 / a)) * a + b
    return np.sqrt((x_ - x1) ** 2 + (y_ - y1) ** 2)


def cost_parallelogram(params, *args):
    a, b, c, d = params
    x, y = args[0], args[1]
    x, y = x[x > .1], y[x > .1]

    s1, s2 = d / c, b / a  # slope of lines of parallelogram
    h1, h2 = b - a * s1, d - s2 * c  # offset of lines defining parallelogram (two go through origin)

    # get surface of parallelogram
    d1 = np.sqrt(c ** 2 + d ** 2)  # length of base
    d2 = dist_pt_line(s1, 0, a, b)  # distance from pt (a, b) to line y=d/c*x+0, == length from base to corner
    surface = d1 * d2

    top, bottom = y > x * s1 + h1, y < x * s1
    left, right = y > x * s2, y < x * s2 + h2

    # min dist of pts to corners
    dist = (top & left) * np.sqrt(((x - a) ** 2 + (y - b) ** 2))
    dist += (top & right) * np.sqrt((x - (a + c)) ** 2 + (y - (b + d)) ** 2)
    dist += (bottom & right) * np.sqrt((x - c) ** 2 + (y - d) ** 2)
    # min dist of pts to lower line
    dist += (bottom & ~right & ~left) * dist_pt_line(s1, 0, x, y)
    # min dist of pts to upper line
    dist += (top & ~right & ~left) * dist_pt_line(s1, h1, x, y)  # ((x-x_)**2+(y-y_)**2)
    # min dist of pts to left line
    dist += (left & ~bottom & ~top) * dist_pt_line(s2, 0, x, y)
    # min dist of pts to right line
    dist += (right & ~bottom & ~top) * dist_pt_line(s2, h2, x, y)

    return surface * 60 + np.sum(dist)  # note weight found by testing max correlation true kappa vs recovered


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



#########
# plotting
##########

def scatter_kappas(adata, genes, show_cluster=True, cluster_name="clusters", use_raw=False, reg_=["up", "down"]):
    fig, ax = plt.subplots(len(genes), 2, figsize=(14, 4 * len(genes)))

    for idx, j in enumerate(genes):
        alpha, beta, gamma, ut, st, _, _, _, _, up_reg, down_reg = get_pars(adata, j, use_raw=use_raw)
        scv.pl.scatter(adata, j, ax=ax[idx, 0], show=False, frameon=False, use_raw=use_raw, )

        for reg in reg_:
            r_ = up_reg if reg == "up" else down_reg
            # ordr = np.argsort(ut[r_]) if reg == "up" else np.argsort(-ut[r_])
            # reg_idx = np.where(r_)[0][ordr]

            if np.sum(r_) > 0.3 * (np.sum(up_reg) + np.sum(down_reg)):
                t_dist, f = get_f_and_delta_t(ut, st,
                                              alpha, beta, gamma, r_, reg, mode="s")
                # print(np.sum(np.isnan(f)))
                mn = op.minimize(cost_parallelogram, args=(t_dist / np.max(t_dist), f / np.max(f)), **kwargs)
                a, b, c, d = mn.x

                ax[idx, 1].scatter(t_dist, f, s=3, color="grey" if reg == "up" else "lightblue", alpha=.5)
                ax[idx, 1].set_xlabel("Delta t"), ax[idx, 1].set_ylabel("f(u) or f(s)")
                ax[idx, 1].spines['right'].set_visible(False)
                ax[idx, 1].spines['top'].set_visible(False)

                ax[idx, 1].plot(np.array([0, a, (a + c), c, 0]) * np.max(t_dist),
                                np.array([0, b, (b + d), d, 0]) * np.max(f), color="red" if reg == "up" else "orange")
        ax[idx, 1].set_xlim(0), ax[idx, 1].set_ylim(0)
    plt.show()
