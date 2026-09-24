"""Closed-form terms for the two-state expansion used on the example pages.

For a two-state chain switching at rate lam each way, eps = 1/lam, gbar = (g1+g2)/2, gt = (g1-g2)/2 and gt(0) = 0:
    log m(t) = int gbar + (eps/2) int gt^2 - (eps^2/8) gt(t)^2 + O(eps^3)
    omega(t) = (eps/2) gt(t) - (eps^2/4) gt'(t) + O(eps^3),       a_{1,2} = m (1 +/- omega).
(The initial layer adds (eps^2/4) gt'(0) exp(-2 lam t) to omega, and first enters log m at eps^4.)
For constant g the two-state system is solved exactly:
    a_{1,2}(t) = exp((gbar - lam) t) [cosh(s t) + (lam +/- gt) sinh(s t) / s],   s = sqrt(lam^2 + gt^2).
"""
import cmath, math
import numpy as np


# ---------------------------------------------------------------- Vasicek B = (1 - e^{-kappa t}) / kappa
def vasicek_moment(k, m, T, kappa):
    """int_0^T e^{-k kappa t} B(t)^m dt = kappa^{-m} sum_l C(m, l) (-1)^l (1 - e^{-(k+l) kappa T}) / ((k+l) kappa).
    The alternating sum cancels when kappa T is small; there the Taylor series in kappa T (exact integer
    coefficients) is used, and in between, where both lose digits, Gauss-Legendre quadrature of the positive
    integrand."""
    x = kappa * T
    if x == 0:
        return T ** (m + 1) / (m + 1)
    terms = [math.comb(m, l) * (-1) ** l * (T if k + l == 0 else -math.expm1(-(k + l) * x) / ((k + l) * kappa))
             for l in range(m + 1)]
    tot = math.fsum(terms)
    if tot > 0 and math.fsum(abs(v) for v in terms) < 100 * tot:
        return tot / kappa ** m
    # e^{-ku} (1 - e^{-u})^m = sum_n c_n u^n / n!,  c_n = sum_l C(m, l) (-1)^l (-(k+l))^n,  c_n = 0 for n < m
    if x * (k + m + 1) <= 10:
        tot, big = 0.0, 0.0
        for n in range(m, m + 200):
            c = sum(math.comb(m, l) * (-1) ** l * (-(k + l)) ** n for l in range(m + 1))
            term = c / math.factorial(n) * x ** (n - m) / (n + 1)
            tot += term
            big = max(big, abs(term))
            if n > m + 2 and abs(term) < 1e-17 * abs(tot):
                break
        if tot > 0 and big < 100 * tot:
            return tot * T ** (m + 1)
    panels = max(1, math.ceil(x * (k + m)))
    z, w = np.polynomial.legendre.leggauss(20)
    edges = np.linspace(0, T, panels + 1)
    t = ((z[None, :] + 1) / 2 * np.diff(edges)[:, None] + edges[:-1, None]).ravel()
    wt = (w[None, :] / 2 * np.diff(edges)[:, None]).ravel()
    return float(np.sum(wt * np.exp(-k * kappa * t) * (-np.expm1(-kappa * t) / kappa) ** m))


def I_k(k, t, kappa):
    """int_0^t B^k = kappa^{-k} (t + sum_j C(k, j) (-1)^j (1 - e^{-j kappa t}) / (j kappa))."""
    return vasicek_moment(0, k, t, kappa)


def J_1(t, kappa, m):
    """int_0^t 1/(1 + m B)."""
    c, q = 1 + m / kappa, m / kappa
    return t / c + math.log((c - q * math.exp(-kappa * t)) / (c - q)) / (c * kappa)


def J_2(t, kappa, m):
    """int_0^t 1/(1 + m B)^2."""
    c, q = 1 + m / kappa, m / kappa
    return J_1(t, kappa, m) / c - (1 / (c - q * math.exp(-kappa * t)) - 1 / (c - q)) / (c * kappa)


def vasicek_jump_integrals(t, kappa, a, b, c, m):
    """int_0^t f^2 for f = a B + b B^2 + c (r - 1), r = 1/(1 + m B)  (c = 0 or m = 0: no jump part)."""
    I1, I2, I3, I4 = (I_k(k, t, kappa) for k in (1, 2, 3, 4))
    out = a * a * I2 + 2 * a * b * I3 + b * b * I4
    if c:
        J1, J2 = J_1(t, kappa, m), J_2(t, kappa, m)
        int_B_rm1 = (t - J1) / m - I1                       # int B (r - 1)
        int_B2_rm1 = (I1 - (t - J1) / m) / m - I2           # int B^2 (r - 1)
        int_rm1_sq = J2 - 2 * J1 + t                        # int (r - 1)^2
        out += c * c * int_rm1_sq + 2 * a * c * int_B_rm1 + 2 * b * c * int_B2_rm1
    return out


def vasicek_jump_mean(t, kappa, a, b, c, m):
    """int_0^t (a B + b B^2 + c (r - 1))."""
    out = a * I_k(1, t, kappa) + b * I_k(2, t, kappa)
    if c:
        out += c * (J_1(t, kappa, m) - t)
    return out


# ---------------------------------------------------------------- CIR B = 2(e^{ht}-1) / ((h+kappa)(e^{ht}-1) + 2h)
def cir_B(t, kappa, sigma):
    """B = 2(1-q) / ((h+kappa)(1-q) + 2h q), q = e^{-ht}: the form above divided by e^{ht}."""
    h = math.sqrt(kappa ** 2 + 2 * sigma ** 2)
    p = -math.expm1(-h * t)
    return 2 * p / ((h + kappa) * p + 2 * h * (1 - p))


def cir_int_B(t, kappa, sigma):
    """int_0^t B = -(2/sigma^2) log(2h e^{(kappa+h)t/2} / ((h+kappa)(e^{ht}-1) + 2h))
                 = 2t/(h+kappa) - 2(1-q) log(1+y) / (y h (h+kappa)),   q = e^{-ht},  y = -sigma^2 (1-q) / (h (h+kappa)),
    which stays accurate as sigma -> 0 (Vasicek: t/kappa - (1-q)/kappa^2). For h t < 1, where the two terms cancel,
    20-point Gauss-Legendre quadrature of B."""
    h = math.sqrt(kappa ** 2 + 2 * sigma ** 2)
    if h * t < 1:
        z, w = np.polynomial.legendre.leggauss(20)
        return sum(wi * t / 2 * cir_B(t * (zi + 1) / 2, kappa, sigma) for zi, wi in zip(z, w))
    p = -math.expm1(-h * t)
    y = -sigma ** 2 * p / (h * (h + kappa))
    ly = math.log1p(y) / y if y else 1.0
    return 2 * t / (h + kappa) - 2 * p * ly / (h * (h + kappa))


def cir_int_B2(t, kappa, sigma):
    """int_0^t B^2 by E = e^{hs} and partial fractions: B = 2(E-1)/(cE + e0), c = h+kappa, e0 = h-kappa.
    With q = e^{-ht}, y = -e0 (1-q) / (2h) and L(y) = (log(1+y) - y) / y^2,
        int_0^t B^2 = (4/h) [ (h t + log(1+y)) / c^2 - (1-q)^2 L(y) / (4h^2) - (1-q)(c(2-q) + e0) / (2h c (c + e0 q)) ].
    The poles of the partial fractions merge as sigma -> 0; this grouping keeps every term finite, and at sigma = 0
    it is Vasicek's int B^2. For h t < 1 the terms cancel to O((ht)^3), and 20-point Gauss-Legendre quadrature of the
    analytic integrand is used instead."""
    h = math.sqrt(kappa ** 2 + 2 * sigma ** 2)
    if h * t < 1:
        z, w = np.polynomial.legendre.leggauss(20)
        return sum(wi * t / 2 * cir_B(t * (zi + 1) / 2, kappa, sigma) ** 2 for zi, wi in zip(z, w))
    c, e0 = h + kappa, 2 * sigma ** 2 / (h + kappa)       # e0 = h - kappa without cancellation
    q, p = math.exp(-h * t), -math.expm1(-h * t)          # p = 1 - q
    y = -e0 * p / (2 * h)
    if abs(y) < 1e-3:
        L = sum((-1) ** (j + 1) * y ** j / (j + 2) for j in range(8))
    else:
        L = (math.log1p(y) - y) / (y * y)
    return 4 / h * ((h * t + math.log1p(y)) / c ** 2 - p * p * L / (4 * h * h)
                    - p * (c * (2 - q) + e0) / (2 * h * c * (c + e0 * q)))


# ---------------------------------------------------------------- assembling
def two_state_second_order(int_gbar, int_gt2, gt_T, gtp_T, eps, sign=+1):
    """a_{1 or 2}(T) through eps^2 from the closed-form pieces (gt(0) = 0)."""
    logm = int_gbar + eps / 2 * int_gt2 - eps ** 2 / 8 * gt_T ** 2
    om = eps / 2 * gt_T - eps ** 2 / 4 * gtp_T
    return cmath.exp(logm) * (1 + sign * om), logm, om


def two_state_constant_exact(gbar, gt, lam, t, sign=+1):
    """exp((gbar - lam) t) [cosh(s t) + (lam +/- gt) sinh(s t) / s], evaluated without overflow: for |s t| >= 1 as
    ((1+p)/2) e^{(gbar-lam+s) t} + ((1-p)/2) e^{(gbar-lam-s) t}, p = (lam +/- gt) / s, Re s >= 0."""
    s = cmath.sqrt(lam * lam + gt * gt)
    b = lam + sign * gt
    if abs(s * t) < 1:
        z = s * t
        shc = cmath.sinh(z) / z if z else 1.0                # sinh(s t) / (s t)
        return cmath.exp((gbar - lam) * t) * (cmath.cosh(z) + b * t * shc)
    p = b / s
    return 0.5 * (1 + p) * cmath.exp((gbar - lam + s) * t) + 0.5 * (1 - p) * cmath.exp((gbar - lam - s) * t)
