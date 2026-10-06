"""Regime-switching versions of QuantLib models the engine covers exactly: the switched parameters enter only the
regime's forcing g_i, never the Riccati equation for the state coefficient.

Characteristic functions of the return X_T - X_0 (log price, zero rates): E[exp(i u (X_T - X_0))] = a_i(T) or
exp(D v0) a_i(T), with a' = (Q + diag g) a, a(0) = 1.
"""
import cmath
from .fastswitch import ExpSum, Cheb
from .models import heston_switching_theta


def _expm1_less_x(x):
    """e^x - 1 - x: its series when x is small, where forming e^x - 1 and subtracting x would cancel."""
    if abs(x) < 1e-2:
        term, total = x * x / 2, 0.0
        for k in range(3, 14):
            total += term; term *= x / k
        return total + term
    return cmath.exp(x) - 1 - x


def compensated_jump(u, mu_j, sig_j):
    """phi_J(u) - 1 - i u kbar for lognormal jumps, kbar = e^{mu + sig^2/2} - 1. With a = i u mu - u^2 sig^2 / 2 and
    b = mu + sig^2 / 2 the first-order parts combine exactly, a - i u b = -(u^2 + i u) sig^2 / 2, so a small jump
    keeps its second-order size instead of losing it in the difference of two numbers of size mu."""
    a, b = 1j * u * mu_j - 0.5 * u * u * sig_j ** 2, mu_j + 0.5 * sig_j ** 2
    if b > 700.0:
        raise ValueError(f"the mean jump factor exp(logJumpMean + logJumpVol^2 / 2) = exp({b:g}) overflows")
    return _expm1_less_x(a) - 1j * u * _expm1_less_x(b) - 0.5 * (u * u + 1j * u) * sig_j ** 2


def merton76(u, sigmas, lams, mu_j, sig_j):
    """Merton jump diffusion (QuantLib Merton76Process, JumpDiffusionEngine) with the diffusion volatility and the jump
    intensity switched. Lognormal jumps with log-mean mu_j and log-sd sig_j; drift compensates each regime's jumps."""
    cJ = compensated_jump(u, mu_j, sig_j)
    cs = [-0.5 * s * s * (1j * u + u * u) + lam * cJ for s, lam in zip(sigmas, lams)]
    return [ExpSum({0: c}) for c in cs], [(lambda c: (lambda t: c))(c) for c in cs]


def variance_gamma(u, sigmas, nus, thetas):
    """Variance gamma (QuantLib VarianceGammaProcess), all three parameters switched, martingale-corrected drift."""
    cs = []
    for s, nu, th in zip(sigmas, nus, thetas):
        psi = lambda z: -cmath.log(1 - 1j * th * nu * z + 0.5 * s * s * nu * z * z) / nu
        omega = -psi(-1j)                     # makes E[exp(X)] = 1
        cs.append(psi(u) + 1j * u * omega)
    return [ExpSum({0: c}) for c in cs], [(lambda c: (lambda t: c))(c) for c in cs]


def bates(u, kappa, thetas, xi, rho, lams, mu_j, sig_j, T):
    """Bates (QuantLib BatesModel, BatesEngine) with the variance level and the jump intensity switched. Returns g, gfuncs
    and the regime-free Heston coefficient D, so that phi_i = exp(D(T) v0) a_i(T)."""
    g, gf, D = heston_switching_theta(u, kappa, thetas, xi, rho, T)
    cJ = compensated_jump(u, mu_j, sig_j)
    jc = [lam * cJ for lam in lams]
    g2 = [gi + Cheb.fit(lambda t, c=c: c, T, 4) for gi, c in zip(g, jc)]
    gf2 = [(lambda f, c: (lambda t: f(t) + c))(f, c) for f, c in zip(gf, jc)]
    return g2, gf2, D
