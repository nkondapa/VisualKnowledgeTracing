import numpy as np


def get_confidence_intervals_bayesian(error_count, total_count):
    '''Start with a uniform prior and update it with every observation.'''
    BAYESIAN_STEP = 0.001  # step for sampling P
    CONF = 0.025  # the confidence interval is [CONF, 1-CONF]
    P = error_count / (total_count + np.finfo(float).eps)

    Pp_1 = np.arange(BAYESIAN_STEP / 2, 1 + BAYESIAN_STEP / 2, BAYESIAN_STEP)  # P(z=1|p)
    Pp_0 = np.arange(1 - BAYESIAN_STEP / 2, 0 - BAYESIAN_STEP / 2, -BAYESIAN_STEP)
    p_val = Pp_1  # value of p

    N_SAMPLES = len(Pp_1)
    Pp = np.ones(N_SAMPLES) / N_SAMPLES  # initialize to the uniform

    # compute posterior - using log to avoid over/underflow when N is very large
    Pp_log = np.log(Pp) + error_count * np.log(Pp_1) + (total_count - error_count) * np.log(Pp_0)
    Pp = np.exp(Pp_log)
    Pp = Pp / np.sum(Pp)  # normalize - THE NUMERICS COULD BE SUCH THAT THIS FAILS

    Cp = np.cumsum(Pp)  # estimate of the cumulative density
    bay_lo = p_val[np.min(np.where(Cp > CONF))] if Cp[-1] >= CONF else p_val[-1]
    bay_hi = p_val[np.max(np.where(Cp < (1 - CONF)))] if Cp[0] < (1 - CONF) else p_val[0]
    bay_mn = p_val[np.min(np.where(Cp > 0.5))]

    return bay_lo, bay_mn, bay_hi


def propci_wilson_cc(count, nobs, alpha=0.05):
    '''get confidence limits for proportion using Wilson score method w/ cont correction
    i.e. Method 4 in Newcombe [1]; verified via Table 1.
    see wikipedia article: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval'''
    EPS = 1.0
    from scipy import stats
    n = np.maximum(nobs, EPS)
    p = count / n
    q = 1. - p
    z = stats.norm.isf(alpha / 2.)
    z2 = z ** 2
    denom = 2 * (n + z2)
    num = 2. * n * p + z2 - 1. - z * np.sqrt(z2 - 2 - 1. / n + 4 * p * (n * q + 1))
    ci_l = num / denom
    num = 2. * n * p + z2 + 1. + z * np.sqrt(z2 + 2 - 1. / n + 4 * p * (n * q - 1))
    ci_u = num / denom

    # fix the pathological cases
    if np.isscalar(count):
        if nobs == 0:
            ci_l = 0
            ci_u = 1
        if count == 0:
            ci_l = 0
        if p == 1:
            ci_u = 1
    else:
        n0_idx = np.where(nobs == 0.)  # no observations
        p0_idx = np.where(count == 0.)  # no positive events
        p1_idx = np.where(p == 1.)  # all positive events

        ci_l[p0_idx] = 0.
        ci_u[p1_idx] = 1.
        ci_l[n0_idx] = 0.
        ci_u[n0_idx] = 1.

    return ci_l, p, ci_u


# pass a list of number of negatives and list of totals,
# compute confidence interval over list
def generate_ci_over_list(ci_method, count, total):
    low, mid, high = [], [], []
    for c in range(len(total)):
        l, m, h = ci_method(count[c], total[c])
        low.append(l)
        mid.append(m)
        high.append(h)

    low = np.array(low)
    mid = np.array(mid)
    high = np.array(high)
    return low, mid, high