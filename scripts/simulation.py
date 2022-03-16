from anndata import AnnData
import numpy as np

exp = np.exp


def vectorize(t, t_, alpha, beta, gamma=None, alpha_=0, u0=0, s0=0, sorted=False):
    o = np.array(t < t_, dtype=int)
    tau = t * o + (t - t_) * (1 - o)

    u0_ = unspliced(t_, u0, alpha, beta)
    s0_ = spliced(t_, s0, u0, alpha, beta, gamma if gamma is not None else beta / 2)

    # vectorize u0, s0 and alpha
    u0 = u0 * o + u0_ * (1 - o)
    s0 = s0 * o + s0_ * (1 - o)
    alpha = alpha * o + alpha_ * (1 - o)

    if sorted:
        idx = np.argsort(t)
        tau, alpha, u0, s0 = tau[idx], alpha[idx], u0[idx], s0[idx]
    return tau, alpha, u0, s0


def inv(x):
    x_inv = 1 / x * (x != 0)
    return x_inv


def unspliced(tau, u0, alpha, beta):
    expu = exp(-beta * tau)
    return u0 * expu + alpha / beta * (1 - expu)


def spliced(tau, s0, u0, alpha, beta, gamma):
    c = (alpha - u0 * beta) * inv(gamma - beta)
    expu, exps = exp(-beta * tau), exp(-gamma * tau)
    return s0 * exps + alpha / gamma * (1 - exps) + c * (exps - expu)


def mRNA(tau, u0, s0, alpha, beta, gamma):
    expu, exps = exp(-beta * tau), exp(-gamma * tau)
    expus = (alpha - u0 * beta) * inv(gamma - beta) * (exps - expu)
    u = u0 * expu + alpha / beta * (1 - expu)
    s = s0 * exps + alpha / gamma * (1 - exps) + expus
    return u, s


def cycle(array, n_vars=None):
    if isinstance(array, (np.ndarray, list, tuple)):
        return array if n_vars is None else array * int(np.ceil(n_vars / len(array)))
    else:
        return [array] if n_vars is None else [array] * n_vars


def simulation(
        n_obs=300,
        n_vars=None,
        alpha=None,
        beta=None,
        gamma=None,
        t_max=30,
        noise_level=1,
        switches=None,
        start_t=None,
        random_seed=0,
):
    """Simulation of mRNA splicing kinetics.


    Simulated mRNA metabolism with transcription, splicing and degradation.
    The parameters for each reaction are randomly sampled from a log-normal distribution
    and time events follow the Poisson law. The total time spent in a transcriptional
    state is varied between two and ten hours.

    .. image:: https://user-images.githubusercontent.com/31883718/79432471-16c0a000-7fcc-11ea-8d62-6971bcf4181a.png
       :width: 600px

    Returns
    -------
    Returns `adata` object
    """

    np.random.seed(random_seed)

    def true_time(n, t_max):
        from random import uniform, seed
        seed(random_seed)
        t = np.arange(0, (n - 1) / 10, .1)  # np.cumsum([uniform(0, 1) for _ in range(n - 1)])
        t = np.insert(t, 0, 0)
        t *= t_max / np.max(t)
        return t  # prepend t0=0

    def simulate_dynamics(tau, alpha, beta, gamma, u0, s0, noise_level):
        ut, st = mRNA(tau, u0, s0, alpha, beta, gamma)
        ut += np.random.normal(scale=noise_level * np.percentile(ut, 99) / 10, size=len(ut))
        st += np.random.normal(scale=noise_level * np.percentile(st, 99) / 10, size=len(st))
        ut, st = np.clip(ut, 0, None), np.clip(st, 0, None)
        return ut, st

    alpha = 5 if alpha is None else alpha
    beta = 0.5 if beta is None else beta
    gamma = 0.3 if gamma is None else gamma

    t = true_time(n_obs, t_max)

    # switching time point corresponds to % of steady-state value reached before switching
    switches = (
        cycle([0.7, 0.8, 0.999, 0.9], n_vars)
        if switches is None
        else cycle(switches, n_vars)
    )
    # genes should start so that they end after running through two full switching cycles
    # time spend in transient state is given by:
    # u = sw * (alpha/beta)  # alpha / beta is steady_u ratio at limit, sw=1-switch
    # alpha/beta * (1-e**(-beta*t)) = sw * (alpha/beta)
    # (...) solve for t
    # t = -np.log(sw)/\beta
    transient_t = -np.log(1 - switches) / beta
    start_t = t_max - (transient_t * 2)  # np.zeros(n_vars) if start_t is None else np.array(cycle(start_t, n_vars))
    start_t[start_t < 0] = 0

    t_ = np.array([np.max(t[t < t_i * t_max]) for t_i in switches])
    #t_ = transient_t#*2

    noise_level = cycle(noise_level, n_vars)

    U = np.zeros(shape=(len(t), n_vars))
    S = np.zeros(shape=(len(t), n_vars))

    for i in range(n_vars):
        alpha_i = alpha[i] if isinstance(alpha, (tuple, list, np.ndarray)) else alpha
        beta_i = beta[i] if isinstance(beta, (tuple, list, np.ndarray)) else beta
        gamma_i = gamma[i] if isinstance(gamma, (tuple, list, np.ndarray)) else gamma
        t_start = np.copy(t) - start_t[i]
        t_start[t_start < 0] = 0
        tau, alpha_vec, u0_vec, s0_vec = vectorize(t_start, t_[i], alpha_i, beta_i, gamma_i)

        U[:, i], S[:, i] = simulate_dynamics(
            tau,
            alpha_vec,
            beta_i,
            gamma_i,
            u0_vec,
            s0_vec,
            noise_level[i],
        )

    obs = {"true_t": t.round(2)}
    var = {"true_t_": t_[:n_vars],
           "true_alpha": np.ones(n_vars) * alpha,
           "true_beta": np.ones(n_vars) * beta,
           "true_gamma": np.ones(n_vars) * gamma,
           "true_scaling": np.ones(n_vars)}
    layers = {"unspliced": U, "spliced": S}

    return AnnData(S, obs, var, layers=layers)
