"""Regime-switching versions of QuantLib models. Constructor arguments follow the QuantLib class named in each
docstring; a parameter that switches is given as a list with one entry per regime (a scalar means no switching).

Every model exposes forcing(...): the per-regime functions g_i(t) of the reduced system a' = (Q + diag g) a, plus
the regime-free prefactor, which is what the engines consume."""
import math
import cmath
import numpy as np
from ._engine import models as _m
from ._engine import quantlib_models as _q
from ._engine.fastswitch import ExpSum, Cheb


def _per_regime(x, n):
    return [float(x)] * n if np.isscalar(x) else [float(v) for v in x]


class SwitchingModel:
    def __init__(self, chain):
        self.chain = chain
        self.n = chain.numberOfRegimes()


# ---------------------------------------------------------------- short-rate models: zero-coupon bonds
class SwitchingVasicek(SwitchingModel):
    """QuantLib Vasicek(r0, a, b, sigma): dr = a (b - r) dt + sigma dW. b and sigma may switch."""
    def __init__(self, chain, r0, a, b, sigma):
        super().__init__(chain)
        self.r0, self.a = float(r0), float(a)
        self.b, self.sigma = _per_regime(b, self.n), _per_regime(sigma, self.n)

    def bondForcing(self, T):
        rhos = [[[1.0]]] * self.n
        g, gfuncs, pre = _m.gaussian_factors([self.a], [self.b], [self.sigma], rhos, [1.0])
        return g, gfuncs, (lambda t: pre(t, [self.r0]))


class SwitchingCoxIngersollRoss(SwitchingModel):
    """QuantLib CoxIngersollRoss(r0, theta, k, sigma): dr = k (theta - r) dt + sigma sqrt(r) dW. theta may switch."""
    def __init__(self, chain, r0, theta, k, sigma):
        super().__init__(chain)
        self.r0, self.k, self.sigma = float(r0), float(k), float(sigma)
        self.theta = _per_regime(theta, self.n)

    def bondForcing(self, T):
        g, gfuncs, pre, B = _m.cir_switching_mean(self.k, self.theta, self.sigma, T)
        return g, gfuncs, (lambda t: pre(t, self.r0))


# ---------------------------------------------------------------- equity models: characteristic functions
class SwitchingBlackScholesProcess(SwitchingModel):
    """QuantLib BlackScholesMertonProcess with a constant rate r, dividend yield q and volatility sigma; sigma may switch."""
    def __init__(self, chain, S0, r, q, sigma):
        super().__init__(chain)
        self.S0, self.r, self.q = float(S0), float(r), float(q)
        self.sigma = _per_regime(sigma, self.n)

    def forward(self, T):
        return self.S0 * math.exp((self.r - self.q) * T)

    def returnForcing(self, u, T):
        """g_i for the characteristic function of the martingale log return X, S_T = F exp(X)."""
        g, gfuncs = _m.bs_switching(u, 0.0, self.sigma)
        return g, gfuncs, (lambda: 1.0)


class SwitchingHestonModel(SwitchingModel):
    """QuantLib HestonModel / HestonProcess(r, q, S0, v0, kappa, theta, sigma, rho); the long-run variance theta may switch."""
    def __init__(self, chain, S0, r, q, v0, kappa, theta, sigma, rho):
        super().__init__(chain)
        self.S0, self.r, self.q, self.v0 = float(S0), float(r), float(q), float(v0)
        self.kappa, self.sigma, self.rho = float(kappa), float(sigma), float(rho)
        self.theta = _per_regime(theta, self.n)

    def forward(self, T):
        return self.S0 * math.exp((self.r - self.q) * T)

    def returnForcing(self, u, T):
        g, gfuncs, D = _m.heston_switching_theta(u, self.kappa, self.theta, self.sigma, self.rho, T)
        return g, gfuncs, (lambda: cmath.exp(D(T) * self.v0))


class SwitchingMerton76Process(SwitchingModel):
    """QuantLib Merton76Process(S0, q, r, sigma, jumpIntensity, jumpMean (log), jumpVol); sigma and the intensity may switch."""
    def __init__(self, chain, S0, r, q, sigma, jumpIntensity, logJumpMean, logJumpVol):
        super().__init__(chain)
        self.S0, self.r, self.q = float(S0), float(r), float(q)
        self.sigma = _per_regime(sigma, self.n)
        self.jumpIntensity = _per_regime(jumpIntensity, self.n)
        self.logJumpMean, self.logJumpVol = float(logJumpMean), float(logJumpVol)

    def forward(self, T):
        return self.S0 * math.exp((self.r - self.q) * T)

    def returnForcing(self, u, T):
        g, gfuncs = _q.merton76(u, self.sigma, self.jumpIntensity, self.logJumpMean, self.logJumpVol)
        return g, gfuncs, (lambda: 1.0)


class SwitchingBatesModel(SwitchingModel):
    """QuantLib BatesModel / BatesProcess(r, q, S0, v0, kappa, theta, sigma, rho, lambda, nu, delta): Heston with
    lognormal jumps of log-mean nu and log-sd delta at intensity lambda. theta and lambda may switch."""
    def __init__(self, chain, S0, r, q, v0, kappa, theta, sigma, rho, jumpIntensity, logJumpMean, logJumpVol):
        super().__init__(chain)
        self.S0, self.r, self.q, self.v0 = float(S0), float(r), float(q), float(v0)
        self.kappa, self.sigma, self.rho = float(kappa), float(sigma), float(rho)
        self.theta = _per_regime(theta, self.n)
        self.jumpIntensity = _per_regime(jumpIntensity, self.n)
        self.logJumpMean, self.logJumpVol = float(logJumpMean), float(logJumpVol)

    def forward(self, T):
        return self.S0 * math.exp((self.r - self.q) * T)

    def returnForcing(self, u, T):
        g, gfuncs, D = _q.bates(u, self.kappa, self.theta, self.sigma, self.rho, self.jumpIntensity,
                                self.logJumpMean, self.logJumpVol, T)
        return g, gfuncs, (lambda: cmath.exp(D(T) * self.v0))


class SwitchingVarianceGammaProcess(SwitchingModel):
    """QuantLib VarianceGammaProcess(S0, q, r, sigma, nu, theta); all three parameters may switch."""
    def __init__(self, chain, S0, r, q, sigma, nu, theta):
        super().__init__(chain)
        self.S0, self.r, self.q = float(S0), float(r), float(q)
        self.sigma, self.nu, self.theta = _per_regime(sigma, self.n), _per_regime(nu, self.n), _per_regime(theta, self.n)

    def forward(self, T):
        return self.S0 * math.exp((self.r - self.q) * T)

    def returnForcing(self, u, T):
        g, gfuncs = _q.variance_gamma(u, self.sigma, self.nu, self.theta)
        return g, gfuncs, (lambda: 1.0)
