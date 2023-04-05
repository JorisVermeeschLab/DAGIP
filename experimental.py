import math
from typing import Tuple

import scipy.stats


def find_equiprobable_point(mu1, mu2, s1, s2, nu1, nu2) -> float:
    if s1 == s2:
        return 0.5 * (mu1 + mu2)
    a = nu2 * s2 ** 2 - nu1 * s1 ** 2
    b = 2 * (nu1 * mu2 * s1 ** 2 - nu2 * mu1 * s2 ** 2)
    c = nu2 * (mu1 * s2) ** 2 - nu1 * (mu2 * s1) ** 2
    sqrt_delta = math.sqrt(b ** 2 - 4 * a * c)
    x1 = (-b + sqrt_delta) / (2 * a)
    x2 = (-b - sqrt_delta) / (2 * a)
    if (mu1 <= x1 <= mu2) or (mu2 <= x1 <= mu1):
        return x1
    else:
        return x2


mu1, s1, nu1 = 4.3, 1.0, 2.1
mu2, s2, nu2 = 2.3, 0.78, 2.1

# x = (nu2 * mu1 - nu1 * mu2) / math.sqrt(nu2 + (nu1 * (s1 ** 2 / s2 ** 2)))
x = find_equiprobable_point(mu1, mu2, s1, s2, nu1, nu2)
print(x)

print((x - mu1) ** 2 / s1 ** 2, (x - mu2) ** 2 / s2 ** 2)
print(scipy.stats.t.logpdf(x, nu1, loc=mu1, scale=s1), scipy.stats.t.logpdf(x, nu2, loc=mu2, scale=s2))
