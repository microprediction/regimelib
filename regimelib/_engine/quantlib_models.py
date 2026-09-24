"""Regime-switching versions of QuantLib models the engine covers exactly: the switched parameters enter only the
regime's forcing g_i, never the Riccati equation for the state coefficient.

Characteristic functions of the return X_T - X_0 (log price, zero rates): E[exp(i u (X_T - X_0))] = a_i(T) or
exp(D v0) a_i(T), with a' = (Q + diag g) a, a(0) = 1.
"""
import cmath
from .fastswitch import ExpSum, Cheb
from .models import heston_switching_theta


def merton76(u, sigmas, lams, mu_j, sig_j):
    """Merton jump diffusion (QuantLib Merton76Process, JumpDiffusionEngine) with the diffusion volatility and the jump
    intensity switched. Lognormal jumps with log-mean mu_j and log-sd sig_j; drift compensates each regime's jumps."""
    phiJ = cmath.exp(1j * u * mu_j - 0.5 * u * u * sig_j ** 2)
    kbar = cmath.exp(mu_j + 0.5 * sig_j ** 2) - 1
    cs = [-0.5 * s * s * (1j * u + u * u) + lam * (phiJ - 1 - 1j * u * kbar) for s, lam in zip(sigmas, lams)]
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
    phiJ = cmath.exp(1j * u * mu_j - 0.5 * u * u * sig_j ** 2)
    kbar = cmath.exp(mu_j + 0.5 * sig_j ** 2) - 1
    jc = [lam * (phiJ - 1 - 1j * u * kbar) for lam in lams]
    g2 = [gi + Cheb.fit(lambda t, c=c: c, T, 4) for gi, c in zip(g, jc)]
    gf2 = [(lambda f, c: (lambda t: f(t) + c))(f, c) for f, c in zip(gf, jc)]
    return g2, gf2, D
