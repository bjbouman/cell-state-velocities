import numpy as np
import scipy.optimize as op


def kappa_velo(adata, mode="u", inplace=True, key="fit"):
    """
    Scales the anndata object.
    Parameters
    ----------
    adata: :class:'~anndata.AnnData'
        Annotated data matrix.
    mode: 'str' (default: u)
        Whether to caculate kapas from unspliced counts ("u") or from unspliced and spiced counts ("s")
    inplace: 'bool' (default: True)
        Whether to scale the adata object directly or a copy and return the copy.
    key: 'str' (default: "fit")
        key under which the fitted parameters are saved in the anndata object.
        For example per default "alpha" parameter is searched under adata.var["fit_alpha"].
    Returns
    -------
        adata: scaled anndata object if inplace == False

    """
    # get kappa
    # we get kappa first for down-reg (if there are sufficient cells in that state) then for up-reg, where down-reg
    # did not work. This order is chosen bc alpha=0 in down-reg, meaning we depend on one less fitted parameter.
    kappas = np.array([get_kappa(adata, i, mode=mode, reg="down", key=key) for i in adata.var_names])
    idx = np.where(np.isnan(kappas))[0]
    kappas[idx] = np.array([get_kappa(adata, i, mode=mode, reg="up", key=key) for i in idx])
    # check if any could still not be recovered
    # scale parameters in anndata object
    if not inplace:
        adata = adata.copy()
    adata.var[key + "_beta"] *= kappas
    adata.var[key + "_alpha"] *= kappas
    adata.var[key + "_gamma"] *= kappas
    adata.var[key + "_t_"] /= kappas
    adata.layers[key + "_t"] /= kappas
    adata.var[key + "_kappa"] = kappas
    if not inplace:
        return adata


def get_pars(adata, gene, key="fit"):
    """
    Gets fitted parameters from scVelo.
    Parameters
    ----------
    adata: :class:'~anndata.AnnData'
        Annotated data matrix.
    gene: 'str' or 'int'
        Gene name or index for which the parameters are fetched.
    key: 'str' (default: "fit")
        key under which the fitted parameters are saved in the anndata object.
        For example per default "alpha" parameter is searched under adata.var["fit_alpha"].
    Returns
    -------

    """
    # todo check if key in adata.var.keys()
    idx = adata.var_names.get_loc(gene) if isinstance(gene, str) else gene

    alpha = adata[:, idx].var[key + "_alpha"].values
    beta = adata[:, idx].var[key + "_beta"].values
    gamma = adata[:, idx].var[key + "_gamma"].values
    scaling = (adata[:, idx].var[key + "_scaling"].values if f"{key}_scaling" in adata.var.keys() else 1)
    t_ = adata[:, idx].var[key + "_t_"].values
    beta *= scaling
    if key + "_u0" in adata.var.keys():
        u0_offset, s0_offset = adata.var[key + "_u0"][idx], adata.var[key + "_s0"][idx]
    else:
        u0_offset, s0_offset = 0, 0
    t = adata.layers[key + "_t"][:, idx]
    tau, alpha_, u0, s0 = vectorize(t, t_, alpha, beta, gamma)
    ut, st = u_t(alpha_, beta, tau, u0), s_t(alpha_, beta, gamma, tau, u0, s0)
    ut, st = ut * scaling + u0_offset, st + s0_offset
    ignore = .1  # we need to remove those bc the density approximation becomes tricky towards the exp function limit
    upper = (st > ignore * np.max(st)) | (ut > ignore * np.max(ut))
    lower = (st < (1 - ignore) * np.max(st)) | (ut < (1 - ignore) * np.max(ut))
    down_reg = np.array((t > t_) & upper & lower)
    up_reg = np.array((t < t_) & upper & lower)
    beta /= scaling
    # u, s=adata.layers["unspliced" if use_raw else "Ms"][:, idx], adata.layers["spliced" if use_raw else "Ms"][:, idx]
    return alpha, beta, gamma, ut, st, up_reg, down_reg


def f_u(alpha, beta, u, u0):
    """
    Calculates right side of eq (6) of our RNA velocity paper.
    """
    u_adj = np.log((u - (alpha / beta)) / (u0 - (alpha / beta)))
    return -1 / beta * u_adj


def f_s(alpha, beta, gamma, u, s, u0, s0):
    """
    Same as f_u() but when starting from s(t) (see supplementary note S1).
    """
    num = s + ((alpha - (beta * u)) / (gamma - beta)) - (alpha / gamma)
    denum = s0 + ((alpha - (beta * u0)) / (gamma - beta)) - (alpha / gamma)
    s_adj = np.log(num / denum)
    return -1 / gamma * s_adj


def get_f_and_delta_t(u, s, alpha, beta, gamma, r_, reg, mode="u", n=5000):
    """
    Helper function to get delta-t and f_s or f_u for n randomly sampled pairs of cells.
    Parameters
    ----------
    u: 'np.array' of 'int' of length n_obs
        Array of unspliced counts for every observation (= for every cell).
    s: 'np.array' of 'int' of length n_obs
        Array of spliced counts for every observation.
    alpha: 'int'
        Fitted alpha for given gene. Assumed to be 0 if reg!="up"
    beta: 'int'
        Fitted beta for given gene.
    gamma: 'int'
        Fitted gamma for given gene.
    r_: 'np.array' of 'bool' of length n_obs
        Boolean array defining whether cells are in transcriptional state (up- or down-reg) of interest.
    reg: 'str'
        Defines the current transcriptional state of interest ("up" or "down" for up- or down-reg).
    mode: 'str'
        Whether to compute f_u or f_s.
    n: 'int'
        Number of pairs of cells to sample.

    Returns
    -------
        delta_t: 'np.array' of length n containing the distance (in # of cells between sampled pairs)
        f: 'np.array' of length n containing f_u or f_s (depending on mode) for the same sampled pairs
    """
    ordr = np.argsort(u[r_]) if reg == "up" else np.argsort(-u[r_])
    reg_idx = np.where(r_)[0][ordr]
    loc_idx = np.array([np.where(reg_idx == i)[0][0] if i in reg_idx else np.nan for i in
                        range(np.max(reg_idx) + 1)])  # get index of cell on that side of the almond
    i = sample(n, u, reg, reg_idx)  # sampling of random pairs
    delta_t = np.abs(loc_idx[i[1]] - loc_idx[i[0]])  # number of cells between i[0] and i[1]
    (u0, u1) = u[i[0]], u[i[1]]  # u values for i[0] and i[1]
    if mode == "u":
        f = f_u(alpha if reg == "up" else 0, beta, u1, u0)
    else:
        (s0, s1) = (s[i[0]], s[i[1]])
        f = f_s(alpha if reg == "up" else 0, beta, gamma, u1, s1, u0, s0)
    return delta_t, f


def get_kappa(adata, gene, mode="u", reg="up", key="fit"):
    """
    Parameters
    ----------
    adata:: :class: '~anndata.AnnData'
        Annotated data matrix on which to compute the kappa estimates
    gene: 'str' or 'int'
        gene name (str) or index (int) for which the kappa estimates should be computed.
    mode: 'str' in ['s', 'u']
        compute the kappa estimates on spliced values (s), unspliced values (u) or on both
    reg: 'str' in ['up', 'down']
        compute the kappa estimates for up- or down- regulation only or for both
    key: 'str' (default: "fit")

    Returns
    -------
    kappa estimate for gene
    """

    # display error if input incorrect
    # mode can be u, s or both to return all u_kappa, all s_kappa or both (simple concatenation)
    if mode not in ["s", "u"]:
        print("error: mode should be \"u\", \"s\"")
        return
    if reg not in ["up", "down", "both"]:
        print("error: reg should be \"up\", \"down\" depending on whether we should calculate the kappa "
              "estimates for up- or down- regulation.")
        return

    # get parameters for each gene
    alpha, beta, gamma, ut, st, up_reg, down_reg = get_pars(adata, gene, key)

    r_ = up_reg if reg == "up" else down_reg

    # at least 30% of the cells need to be in the considered transient state
    if np.sum(r_) > 0.40 * (np.sum(up_reg) + np.sum(down_reg)):
        t_dist, f = get_f_and_delta_t(ut, st, alpha, beta, gamma, r_, reg, mode)
        k = get_slope(t_dist, f)
    else:
        k = np.nan

    return k


def get_slope(x, y):
    """
    Fits parallelogram to set of points defined by (x, y) and returns the slope of the left hand side of
    the fitted parallelogram.
    """
    kwargs = dict(bounds=[(0.1, None), (0.1, None), (0.1, None), (0.01, None)], x0=np.array([.1, 1, 1, .1]),
                  options={"maxiter": 2000, 'disp': True}, tol=1e-8, method="COBYLA")
    mn = op.minimize(cost_parallelogram, args=(x / np.max(x), y / np.max(y)), **kwargs)
    a, b, c, d = mn.x
    return (b * np.max(y)) / (a * np.max(x))


def dist_pt_line(a, b, x1, y1):
    """
    Calculates the minimal distance between a line given by a*x+b and the point (x1, y1).
    """
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
    """
    Cost function of the parallelogram.
    Parameters
    ----------
    params: set of 4 parameters a, b, c, d ('int')
        Define the parallelogram through 4 points: (0, 0), (a, b), (a+c, b+d), (c, d).
        These are the point we can change to try to minimize this cost function
    args: set of 2 parameters x, y ('np.array' of int)
        Define the set of points that are to be fitted by the parallelogram.
    Returns
    -------
    Cost of parallelogram ('float') defined as the surface + the distance of all points outside of the parallelogram.
    """
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


#############
#   utils   #
#############


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))

    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def sample(n, u, reg, reg_idx):
    """
    Helper function to randomly sample pairs of cells in the current transcriptional state of interest (up- or down-reg
    defined by reg_idx). The cells are ordered by u depending on the state.
    """
    i = np.random.choice(reg_idx, (2, n))
    where = u[i[0]] > u[i[1]] if reg == "up" else u[i[0]] < u[i[1]]
    i[:, where] = i[:, where][[1, 0]]  # adapt order of sampled cells
    return i


def vectorize(t, t_, alpha, beta, gamma=None, alpha_=0, u0=0, s0=0):
    o = np.array(t < t_, dtype=int)
    tau = t * o + (t - t_) * (1 - o)

    u0_ = u_t(alpha, beta, t_, u0)
    s0_ = s_t(alpha, beta, gamma if gamma is not None else beta / 2, t_, s0, u0)

    u0 = u0 * o + u0_ * (1 - o)
    s0 = s0 * o + s0_ * (1 - o)
    alpha = alpha * o + alpha_ * (1 - o)
    return tau, alpha, u0, s0


def inv(x):
    """
    Calculates the inverse of x where x!=0.
    """
    x_inv = 1 / x * (x != 0)
    return x_inv


def u_t(alpha, beta, t, u0):
    """
    Calculates u(t) with t the time for one or multiple observations.
    """
    expu = np.exp(-beta * t)
    return u0 * expu + alpha / beta * (1 - expu)


def s_t(alpha, beta, gamma, t, u0, s0):
    """
    Calculates s(t) with t the time for one or multiple observations.
    """
    expu, exps = np.exp(-beta * t), np.exp(-gamma * t)
    return s0 * exps + (alpha / gamma) * (1 - exps) + (alpha - beta * u0) / inv(gamma - beta) * (exps - expu)


############
# eco-velo #
############


def find_mutual_nn(orig, future, k=20, n_jobs=1):
    from scipy.spatial import cKDTree
    # we want the k NN of the current states (orig) in the future states (future)
    # the returned indices need to be the indices of the future states
    orig_NN = cKDTree(future).query(x=orig, k=k, n_jobs=n_jobs)[1]
    # we want to check that the NN of the future states (future) are also in NN fo the current states (orig)
    # the returned indices need to be the indices of the current states
    future_NN = cKDTree(orig).query(x=future, k=k, n_jobs=n_jobs)[1]
    mnn = np.zeros(orig.shape[0]).astype(int) - 1
    for cell in range(orig.shape[0]):
        cell_nn = orig_NN[cell]

        for i in cell_nn:
            if cell in future_NN[i]:
                mnn[cell] = i
                break
    return mnn
