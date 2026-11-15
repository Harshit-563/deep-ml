from math import sqrt

def pooled_stats(m1, s1, n1, m2, s2, n2):
    n = n1 + n2

    mean = (n1 * m1 + n2 * m2) / n

    ss = (
        (n1 - 1) * s1**2 +
        (n2 - 1) * s2**2 +
        n1 * (m1 - mean)**2 +
        n2 * (m2 - mean)**2
    )

    sd = sqrt(ss / (n - 1))

    return mean, sd