"""Models that reduce to a' = (Q + diag g(t)) a, and so fall to the fast-switching engine.

Each builder returns (g, gfuncs, prefactor): g for the engine (ExpSum or Cheb), gfuncs as plain callables for an
independent numerical solution, and the x-dependent factor that multiplies a_i(t).
"""
import math
import cmath
import numpy as np
from .fastswitch import ExpSum, Cheb


# ---------------------------------------------------------------- Gaussian factors (sums of exponentials)
def gaussian_factors(kappas, thetas, sigmas, rhos, weights):
    """Factors dx_j = kappa_j (theta_j[y] - x_j) dt + sigma_j[y] dW_j, corr(dW_j, dW_l) = rhos[y][j][l].
    Quantity E[exp(-int sum_j c_j x_j)] with c = weights: prefactor exp(-sum_j c_j B_j x_j),
    B_j = c_j (1 - exp(-kappa_j t)) / kappa_j, and
    g_i = -sum_j kappa_j theta_j[i] B_j + (1/2) sum_{j,l} rho_i[j][l] sigma_j[i] sigma_l[i] B_j B_l."""
    J, n = len(kappas), len(thetas[0])
    Bs = [ExpSum({0: weights[j] / kappas[j], kappas[j]: -weights[j] / kappas[j]}) for j in range(J)]
    g = []
    for i in range(n):
        gi = ExpSum()
        for j in range(J):
            gi = gi + Bs[j].scale(-kappas[j] * thetas[j][i])
            for l in range(J):
                gi = gi + (Bs[j] * Bs[l]).scale(0.5 * rhos[i][j][l] * sigmas[j][i] * sigmas[l][i])
        g.append(gi)
    gfuncs = [(lambda gi: (lambda t: gi.value(t)))(gi) for gi in g]

    def prefactor(t, xs):
        e = -sum(Bs[j].value(t) * xs[j] for j in range(J))
        return cmath.exp(e) if isinstance(e, complex) else math.exp(e)
    return g, gfuncs, prefactor


# ---------------------------------------------------------------- CIR with a switching mean level (Chebyshev)
def cir_switching_mean(kappa, thetas, sigma, T):
    """dx = kappa (theta[y] - x) dt + sigma sqrt(x) dW. B solves B' = 1 - kappa B - sigma^2 B^2 / 2, B(0) = 0,
    independent of the regime, so u_i = exp(-B x) a_i with g_i = -kappa theta_i B."""
    h = math.sqrt(kappa ** 2 + 2 * sigma ** 2)

    def B(t):
        e = math.exp(h * t) - 1
        return 2 * e / ((h + kappa) * e + 2 * h)
    gfuncs = [(lambda th: (lambda t: -kappa * th * B(t)))(th) for th in thetas]
    Bc = Cheb.fit(B, T, 80)
    g = [Bc.scale(-kappa * th) for th in thetas]
    return g, gfuncs, (lambda t, x: math.exp(-B(t) * x)), B


# ---------------------------------------------------------------- Vasicek with jumps at a switching intensity
def vasicek_jumps(kappa, thetas, sigmas, intensities, jump_mean, T):
    """dx = kappa (theta[y] - x) dt + sigma[y] dW + dJ, J compound Poisson at rate intensities[y] with
    exponential jumps of mean m. u_i = exp(-B x) a_i, g_i = -kappa theta_i B + sigma_i^2 B^2 / 2
    + l_i (1 / (1 + m B) - 1), B = (1 - exp(-kappa t)) / kappa."""
    def B(t):
        return (1 - math.exp(-kappa * t)) / kappa

    def gf(th, s, l):
        return lambda t: -kappa * th * B(t) + 0.5 * s * s * B(t) ** 2 + l * (1 / (1 + jump_mean * B(t)) - 1)
    gfuncs = [gf(thetas[i], sigmas[i], intensities[i]) for i in range(len(thetas))]
    g = [Cheb.fit(f, T, 80) for f in gfuncs]
    return g, gfuncs, (lambda t, x: math.exp(-B(t) * x))


# ---------------------------------------------------------------- Markov-modulated Poisson counts
def mmpp(rates, z):
    """N counts at rate rates[y]. a_i(t) = E[z^N_t | y_0 = i] solves a' = (Q + (z - 1) diag rates) a."""
    g = [ExpSum({0: (z - 1) * r}) for r in rates]
    gfuncs = [(lambda c: (lambda t: c))((z - 1) * r) for r in rates]
    return g, gfuncs


# ---------------------------------------------------------------- Heston with a switching long-run variance
def heston_switching_theta(u, kappa, thetas, xi, rho, T):
    """Log-price X with dX = -v/2 dt + sqrt(v) dW, dv = kappa (theta[y] - v) dt + xi sqrt(v) dZ, corr rho.
    E[exp(i u X_T)] = exp(i u X_0 + D(T) v_0) a_i(T), with D the Heston Riccati solution (regime free) and
    g_i = kappa theta_i D."""
    d = cmath.sqrt((rho * xi * 1j * u - kappa) ** 2 + xi ** 2 * (1j * u + u * u))
    gm = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)

    def D(t):
        e = cmath.exp(-d * t)
        return (kappa - rho * xi * 1j * u - d) / xi ** 2 * (1 - e) / (1 - gm * e)
    gfuncs = [(lambda th: (lambda t: kappa * th * D(t)))(th) for th in thetas]
    Dc = Cheb.fit(D, T, 80)
    g = [Dc.scale(kappa * th) for th in thetas]
    return g, gfuncs, D


# ---------------------------------------------------------------- a fast mean-reverting factor (Hermite form)
def fast_factor(kappa, theta0, theta1, sig0, sig1, rho, order):
    """dx = kappa (theta0 + theta1 y - x) dt + (sig0 + sig1 y) dW, fast factor dY = -Y/eps dt + sqrt(2/eps) dZ,
    corr(dW, dZ) = rho. With u = exp(-B x) a(t, y), B = (1 - exp(-kappa t)) / kappa, and delta = sqrt(eps):
        a' = ( L / delta^2 + G_{-1}(t) / delta + G_0(t) ) a,
        G_0 = g(t, y) = -kappa theta(y) B + sigma(y)^2 B^2 / 2,   G_{-1} = -sqrt(2) rho B sigma(y) d/dy,
    in the probabilists' Hermite basis, truncated well above the modes the expansion reaches.
    Returns (L, pi, one, Gs) for FastSwitchGen with q = 2."""
    from fastswitch_op import Op, hermite_ops
    M = 2 * order + 10
    L, Y, pi, one = hermite_ops(M)
    D = np.zeros((M, M))
    for n in range(1, M):
        D[n - 1, n] = float(n)
    B = ExpSum({0: 1 / kappa, kappa: -1 / kappa})
    BB = B * B
    g0 = B.scale(-kappa * theta0) + BB.scale(0.5 * sig0 ** 2)
    g1 = B.scale(-kappa * theta1) + BB.scale(sig0 * sig1)
    g2 = BB.scale(0.5 * sig1 ** 2)
    Gs = {0: Op([(g0, np.eye(M)), (g1, Y), (g2, Y @ Y)])}
    if rho:
        Gs[-1] = Op([(B.scale(-math.sqrt(2) * rho * sig0), D), (B.scale(-math.sqrt(2) * rho * sig1), Y @ D)])
    return L, pi, one, Gs


def fast_factor_exact(t, y, eps, kappa, theta0, theta1, sig0, sig1, rho):
    """a(t, y) = exp(A + C1 y + C2 y^2) exactly, from three Riccati equations (scipy Radau)."""
    from scipy.integrate import solve_ivp
    se = math.sqrt(eps)

    def f(r, z):
        A, C1, C2 = z
        b = (1 - math.exp(-kappa * r)) / kappa
        c = -math.sqrt(2) * rho * b / se
        return [(C1 * C1 + 2 * C2) / eps - kappa * theta0 * b + 0.5 * sig0 ** 2 * b * b + c * sig0 * C1,
                (-C1 + 4 * C1 * C2) / eps - kappa * theta1 * b + sig0 * sig1 * b * b + c * (2 * sig0 * C2 + sig1 * C1),
                (-2 * C2 + 4 * C2 * C2) / eps + 0.5 * sig1 ** 2 * b * b + 2 * c * sig1 * C2]
    z = solve_ivp(f, (0, t), [0, 0, 0], method='Radau', rtol=3e-14, atol=1e-18).y[:, -1]
    return math.exp(z[0] + z[1] * y + z[2] * y * y)


# ---------------------------------------------------------------- Black-Scholes with a switching volatility
def bs_switching(u, r, sigmas):
    """dX = (r - sigma_y^2 / 2) dt + sigma_y dW for X = log S. E[exp(i u (X_T - X_0)) | y_0 = i] = a_i(T) with
    constant g_i = i u (r - sigma_i^2 / 2) - u^2 sigma_i^2 / 2."""
    cs = [1j * u * (r - s * s / 2) - u * u * s * s / 2 for s in sigmas]
    return [ExpSum({0: c}) for c in cs], [(lambda c: (lambda t: c))(c) for c in cs]


# ---------------------------------------------------------------- Vasicek with a terminal exponential payoff
def vasicek_terminal(kappa, thetas, sigmas, c):
    """E[exp(-int_0^t x - c x_t) 1{y_t = j} | x_0, y_0 = i] = exp(-Bc(t) x_0) a_i(t) with a(0) = e_j,
    Bc(t) = c exp(-kappa t) + (1 - exp(-kappa t)) / kappa, g_i = -kappa theta_i Bc + sigma_i^2 Bc^2 / 2."""
    Bc = ExpSum({0: 1 / kappa, kappa: c - 1 / kappa})
    g = [Bc.scale(-kappa * th) + (Bc * Bc).scale(0.5 * s * s) for th, s in zip(thetas, sigmas)]
    gfuncs = [(lambda gi: (lambda t: gi.value(t)))(gi) for gi in g]
    return g, gfuncs, Bc
