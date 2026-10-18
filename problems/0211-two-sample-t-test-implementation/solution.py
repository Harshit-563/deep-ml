import numpy as np
import math

# scipy is not available in this environment. The two helpers below compute
# the regularized incomplete beta function via the Numerical Recipes
# continued-fraction algorithm — you'll need them to compute the p-value
# from the t-distribution.
def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function."""
    max_iter, eps = 200, 3e-12
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps: break
    return h

def _betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b




def two_sample_t_test(sample1: list[float], sample2: list[float],
                      alpha: float = 0.05) -> dict:
    """
    Perform a two-sample independent t-test (Welch's t-test).
    """
    x1, x2 = np.asarray(sample1, dtype=float), np.asarray(sample2, dtype=float)
    n1, n2 = len(x1), len(x2)
    mean1, mean2 = x1.mean(), x2.mean()
    var1, var2 = x1.var(ddof=1), x2.var(ddof=1)

    # Welch's t-statistic
    t_stat = (mean1 - mean2) / math.sqrt(var1/n1 + var2/n2)

    # Welch–Satterthwaite degrees of freedom
    df_num = (var1/n1 + var2/n2)**2
    df_den = (var1**2 / (n1**2 * (n1-1))) + (var2**2 / (n2**2 * (n2-1)))
    df = df_num / df_den

    # Convert t-statistic to p-value using incomplete beta
    x = df / (df + t_stat**2)
    p_one_tail = 0.5 * _betainc(df/2.0, 0.5, x)
    p_value = 2 * min(p_one_tail, 1 - p_one_tail)  # two-tailed

    # Cohen's d (pooled SD)
    pooled_sd = math.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    cohens_d = (mean1 - mean2) / pooled_sd if pooled_sd > 0 else 0.0

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "degrees_of_freedom": df,
        "reject_null": p_value < alpha,
        "cohens_d": cohens_d
    }