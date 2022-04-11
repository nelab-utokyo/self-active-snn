
import numpy as np

def _logL_alpha_prime_overN(alpha, s_min, s_max, b):
    ss = np.arange(s_min, s_max+1)
    log_ss = np.log(ss)
    power_ss = np.power(ss, alpha)

    r = -np.dot(log_ss, power_ss)/np.sum(power_ss)+b
    return r

def _logL_lambda_prime_overN(lam, s_min, s_max, b):
    ss = np.arange(s_min, s_max+1)
    exp_minus_lambda_ss = np.exp(-lam*ss)

    r = -np.dot(ss, exp_minus_lambda_ss)/np.sum(exp_minus_lambda_ss)+b
    return r

def get_avalanche_sizes(spike_t, unit=1, delta_t=None, sort=True):
    diff = np.diff(spike_t/unit)
    if delta_t is None: delta_t = np.mean(diff)
    pos_neg = np.insert(np.where(diff>delta_t)[0], 0, -1)
    avalanche_sizes = np.diff(pos_neg)
    if sort: avalanche_sizes = np.sort(avalanche_sizes)
    return avalanche_sizes

def get_counts(avalanche_sizes):
    sizes, counts = np.unique(avalanche_sizes, return_counts=True)
    return sizes, counts

def fit_powerlaw(avalanche_sizes, s_min, s_max, max_error=1e-9):
    indices = np.where((avalanche_sizes>=s_min)&(avalanche_sizes<=s_max))[0]
    p_min, p_max = None, -1.
    alpha = -3.
    delta = -1.
    b = np.mean(np.log(avalanche_sizes[indices]))
    N = len(indices)
    is_converged = False
    c, LL = None, None

    while(1):
        error = N*_logL_alpha_prime_overN(alpha, s_min, s_max, b)
        if error < 0:
            p_max = alpha
            alpha += delta
        else:
            p_min = alpha
            break
    for _ in range(1000):
        alpha = (p_min+p_max)/2.
        error = N*_logL_alpha_prime_overN(alpha, s_min, s_max, b)
        if np.abs(error) <= max_error:
            is_converged = True
            break
        else:
            if error < 0:
                p_max = alpha
            else:
                p_min = alpha

    if is_converged:
        c = 1./np.sum(np.power(np.arange(s_min, s_max+1), alpha))
        LL = N*np.log(c)+alpha*np.sum(np.log(avalanche_sizes[indices]))
    else:
        alpha = None
    return alpha, c, LL

def fit_expon(avalanche_sizes, s_min, s_max, max_error=1e-9):
    indices = np.where((avalanche_sizes>=s_min)&(avalanche_sizes<=s_max))[0]
    p_min, p_max = None, None
    lam = 2.
    delta = -0.5
    b = np.mean(avalanche_sizes[indices])
    N = len(indices)
    is_converged = False
    c, LL = None, None

    while(1):
        error = N*_logL_lambda_prime_overN(lam, s_min, s_max, b)
        if error < 0:
            p_max = lam
            lam += delta
        else:
            p_min = lam
            break
    if p_max is None:
        lam = 0.
        while(1):
            error = N*_logL_lambda_prime_overN(lam, s_min, s_max, b)
            if error < 0:
                p_max = lam
                break
            else:
                p_min = lam
                lam += delta
    for _ in range(1000):
        lam = (p_min+p_max)/2.
        error = N*_logL_lambda_prime_overN(lam, s_min, s_max, b)
        if np.abs(error) <= max_error:
            is_converged = True
            break
        else:
            if error < 0:
                p_max = lam
            else:
                p_min = lam

    if is_converged:
        c = 1./np.sum(np.exp(-lam*np.arange(s_min, s_max+1)))
        LL = N*np.log(c)-lam*np.sum(avalanche_sizes[indices])
    else:
        lam = None
    return lam, c, LL
