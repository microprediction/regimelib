"""First-order closed form for a call on a zero-coupon bond when the Vasicek mean level and volatility switch.

Model: dx = kappa (theta_y - x) dt + sigma_y dW, two regimes switching at rate lam each way, start in regime i0.
Call expiring at T on the bond maturing at S, strike K. With eps = 1/lam:

  E[e^{-int x - c x_T} 1{y_T = j}] = 1/2 * exp(alpha - mu c + v c^2/2) * (1 + eps Pi_j(c)) + O(eps^2),
  Pi_j(c) = (1/2) int_0^T gt_c^2 +/- (1/2) gt_c(T) + s_j (1/2) gt_c(0),   gt_c = -kappa tht Bc + (1/2) st Bc^2,
  Bc(t) = c e^{-kappa t} + B(t),

so under the averaged T-forward measure x_T ~ N(mu, v), and a factor c^k acts as the k-th derivative of that
Gaussian density. The bond at expiry in regime j is A_j e^{-b x_T}, with
  A_{1,2} = exp(int_0^tau gbar + (eps/2) int_0^tau gt^2) (1 +/- (eps/2) gt(tau)),  tau = S - T.
The call price is
  C = P(0,T) sum_j (1/2) [ Black(A_j) + eps sum_k pi_{jk} (-1)^k R_k(A_j) ],
with R_k(A) = int (A e^{-bx} - K)^+ f^{(k)}(x) dx / (-1)^k in closed form through Hermite polynomials.
"""
import math
import numpy as np
from math import comb
from scipy.stats import norm
from .explicit import vasicek_moment


def Phi_n(n, T, kappa):
    return T if n == 0 else (1 - math.exp(-n * kappa * T)) / (n * kappa)


def M(k, m, T, kappa):
    """int_0^T e^{-k kappa t} B(t)^m dt = kappa^{-m} sum_l C(m, l) (-1)^l Phi_{k+l}(T), B = (1 - e^{-kappa t})/kappa.
    Evaluated by explicit.vasicek_moment, which avoids the cancellation of the sum when kappa T is small."""
    return vasicek_moment(k, m, T, kappa)


def int_Bc_pow(n, T, kappa):
    """int_0^T Bc(t)^n as a polynomial in c: coefficients [c^0, ..., c^n]."""
    return [comb(n, k) * M(k, n - k, T, kappa) for k in range(n + 1)]


def polymul(a, b):
    out = [0.0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def polyadd(*ps):
    n = max(len(p) for p in ps)
    return [sum(p[i] for p in ps if i < len(p)) for i in range(n)]


def scale(p, s):
    return [s * x for x in p]


def pieces(kappa, th, sig, x0, T, S, lam):
    thb, tht = np.mean(th), (th[0] - th[1]) / 2
    s2 = [s * s for s in sig]
    sb, st = np.mean(s2), (s2[0] - s2[1]) / 2
    eps = 1 / lam
    E = math.exp(-kappa * T)
    B = (1 - E) / kappa
    I1, I2 = M(0, 1, T, kappa), M(0, 2, T, kappa)
    EB = M(1, 1, T, kappa)
    alpha = -B * x0 - kappa * thb * I1 + 0.5 * sb * I2          # log of the averaged P(0, T)
    mu = E * x0 + kappa * thb * B - sb * EB                      # forward-measure mean of x_T
    v = sb * M(2, 0, T, kappa)                                   # forward-measure variance of x_T
    # Pi_j(c) as polynomials: int gt_c^2 = k^2 tht^2 int Bc^2 - k tht st int Bc^3 + st^2/4 int Bc^4
    intg2 = polyadd(scale(int_Bc_pow(2, T, kappa), (kappa * tht) ** 2),
                    scale(int_Bc_pow(3, T, kappa), -kappa * tht * st),
                    scale(int_Bc_pow(4, T, kappa), st * st / 4))
    BcT = [B, E]                                                 # Bc(T) = B + E c
    gtT = polyadd(scale(BcT, -kappa * tht), scale(polymul(BcT, BcT), 0.5 * st))
    gt0 = [0.0, -kappa * tht, 0.5 * st]                          # Bc(0) = c
    # bond factors at expiry, tau = S - T, c = 0
    tau = S - T
    b = (1 - math.exp(-kappa * tau)) / kappa
    Itau = {k: M(0, k, tau, kappa) for k in (1, 2, 3, 4)}
    gbar_int = -kappa * thb * Itau[1] + 0.5 * sb * Itau[2]
    gt2_int = (kappa * tht) ** 2 * Itau[2] - kappa * tht * st * Itau[3] + st * st / 4 * Itau[4]
    gt_tau = -kappa * tht * b + 0.5 * st * b * b
    return dict(eps=eps, alpha=alpha, mu=mu, v=v, intg2=intg2, gtT=gtT, gt0=gt0, b=b,
                Abar=math.exp(gbar_int), gbar_int=gbar_int, gt2_int=gt2_int, gt_tau=gt_tau)


def black(A, K, b, mu, v):
    """int (A e^{-bx} - K)^+ N(mu, v)(dx)."""
    sd = math.sqrt(v)
    xs = math.log(A / K) / b
    zs = (xs - mu) / sd
    return A * math.exp(-b * mu + 0.5 * b * b * v) * norm.cdf(zs + b * sd) - K * norm.cdf(zs)


def hermite(n, z):
    h0, h1 = 1.0, z
    if n == 0:
        return h0
    for k in range(1, n):
        h0, h1 = h1, z * h1 - k * h0
    return h1


def R(k, A, K, b, mu, v):
    """(-1)^k int (A e^{-bx} - K)^+ f^{(k)}(x) dx for f the N(mu, v) density, via f^{(k)} = (-1)^k v^{-k/2} He_k(z) f."""
    if k == 0:
        return black(A, K, b, mu, v)
    sd = math.sqrt(v)
    zs = (math.log(A / K) / b - mu) / sd
    beta = b * sd
    # int_{-inf}^{zs} He_k(z) phi(z) dz = -He_{k-1}(zs) phi(zs)
    tailK = -hermite(k - 1, zs) * norm.pdf(zs)
    # int_{-inf}^{zs} e^{-beta z} He_k(z) phi(z) dz = e^{beta^2/2} int_{-inf}^{zs+beta} He_k(w - beta) phi(w) dw
    ws = zs + beta
    tailA = 0.0
    for i in range(k + 1):
        c = comb(k, i) * (-beta) ** (k - i)
        tailA += c * (norm.cdf(ws) if i == 0 else -hermite(i - 1, ws) * norm.pdf(ws))
    tailA *= math.exp(beta * beta / 2)
    val = A * math.exp(-b * mu) * tailA - K * tailK
    return val / sd ** k                                 # (-1)^k (-1)^k cancels


def call(kappa, th, sig, x0, T, S, K, lam, start=0, order=1):
    p = pieces(kappa, th, sig, x0, T, S, lam)
    eps = p['eps'] if order else 0.0
    sign = 1 if start == 0 else -1
    total = 0.0
    for j, sj in ((0, 1), (1, -1)):
        Aj = math.exp(p['gbar_int'] + eps / 2 * p['gt2_int']) * (1 + sj * eps / 2 * p['gt_tau'])
        Pi = polyadd(scale(p['intg2'], 0.5), scale(p['gtT'], 0.5 * sign), scale(p['gt0'], 0.5 * sj))
        corr = sum(Pi[k] * (-1) ** k * R(k, Aj, K, p['b'], p['mu'], p['v']) for k in range(len(Pi)))
        total += 0.5 * (R(0, Aj, K, p['b'], p['mu'], p['v']) + eps * corr)
    return math.exp(p['alpha']) * total


if __name__ == '__main__':
    from options import zcb_call
    kappa, th, sig, x0, T, S = 0.5, [0.05, 0.03], [0.015, 0.010], 0.04, 1.0, 4.0
    for lam in (10.0, 20.0, 40.0):
        Q = lam * np.array([[-1.0, 1.0], [1.0, -1.0]])
        for K in (0.86, 0.88, 0.90):
            num = zcb_call(T, S, K, x0, 0, kappa, th, sig, Q)
            c0, c1 = call(kappa, th, sig, x0, T, S, K, lam, order=0), call(kappa, th, sig, x0, T, S, K, lam, order=1)
            print(f"lam {lam:4.0f} K {K}: numerical {num:.8f}  Jamshidian-averaged err {abs(c0-num):.2e}  first-order err {abs(c1-num):.2e}")
