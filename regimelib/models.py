"""Regime-switching versions of QuantLib models. Constructor arguments follow the QuantLib class named in each
docstring; a parameter that switches is given as a list with one entry per regime (a scalar means no switching).

Every model exposes forcing(...): the per-regime functions g_i(t) of the reduced system a' = (Q + diag g) a, plus
the regime-free prefactor, which is what the engines consume."""
import math
import cmath
import numpy as np
import scipy.sparse as sp
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


class SwitchingHullWhite(SwitchingModel):
    """QuantLib HullWhite(termStructure, a, sigma): dr = (theta(t) - a r) dt + sigma dW with theta(t) fitted to the
    initial curve. sigma may switch. theta(t) is fitted with the stationary-average variance, so with every regime
    equal the model is QuantLib's and reproduces the curve exactly; with switching, the averaged model reproduces the
    curve and the expansion adds the switching corrections. termStructure: a flat rate, a callable t -> discount factor,
    or a QuantLib YieldTermStructureHandle (times in years from its reference date)."""
    def __init__(self, chain, termStructure, a, sigma):
        super().__init__(chain)
        self.a = float(a); self.sigma = _per_regime(sigma, self.n)
        if np.isscalar(termStructure):
            r = float(termStructure); self.discount = lambda t: math.exp(-r * t)
        elif hasattr(termStructure, "discount"):
            ts = termStructure; self.discount = lambda t: float(ts.discount(float(t)))
        else:
            self.discount = termStructure
        if hasattr(termStructure, "forwardRate"):               # QuantLib's own instantaneous forward at 0
            import QuantLib as ql
            self.r0 = float(termStructure.forwardRate(0.0, 0.0, ql.Continuous).rate())
        else:
            self.r0 = -math.log(self.discount(1e-6)) / 1e-6      # instantaneous forward at 0

    def bondForcing(self, T):
        a = self.a; pi = self.chain.stationaryDistribution()
        s2bar = float(pi @ np.asarray(self.sigma) ** 2)
        # r = x + phi(t), dx = -a x dt + sigma dW, x0 = 0; phi = f(0,t) + s2bar (1 - e^{-a t})^2 / (2 a^2)
        # P(0,T) = exp(-int phi) a_i(T) with g_i = sigma_i^2 B^2 / 2, B = (1 - e^{-a t}) / a
        B = ExpSum({0: 1 / a, a: -1 / a})
        g = [(B * B).scale(0.5 * s * s) for s in self.sigma]
        gfuncs = [(lambda gi: (lambda t: gi.value(t)))(gi) for gi in g]
        int_shift = s2bar / (2 * a * a) * (T - 2 * (1 - math.exp(-a * T)) / a + (1 - math.exp(-2 * a * T)) / (2 * a))
        pre = lambda t: self.discount(t) * math.exp(-int_shift if t == T else -(s2bar / (2 * a * a)) * (t - 2 * (1 - math.exp(-a * t)) / a + (1 - math.exp(-2 * a * t)) / (2 * a)))
        return g, gfuncs, pre


class SwitchingG2(SwitchingModel):
    """QuantLib G2(termStructure, a, sigma, b, eta, rho): r = x + y + phi(t), dx = -a x dt + sigma dW1,
    dy = -b y dt + eta dW2, corr rho, phi(t) fitted to the initial curve. sigma, eta and rho may switch; phi is fitted
    with the stationary-average covariance, so with every regime equal the model is QuantLib's."""
    def __init__(self, chain, termStructure, a, sigma, b, eta, rho):
        super().__init__(chain)
        self.a, self.b = float(a), float(b)
        self.sigma, self.eta, self.rho = _per_regime(sigma, self.n), _per_regime(eta, self.n), _per_regime(rho, self.n)
        if np.isscalar(termStructure):
            r = float(termStructure); self.discount = lambda t: math.exp(-r * t)
        elif hasattr(termStructure, "discount"):
            ts = termStructure; self.discount = lambda t: float(ts.discount(float(t)))
        else:
            self.discount = termStructure

    def bondForcing(self, T):
        a, b = self.a, self.b; pi = self.chain.stationaryDistribution()
        Bx, By = ExpSum({0: 1 / a, a: -1 / a}), ExpSum({0: 1 / b, b: -1 / b})
        g, cov_bar = [], ExpSum()
        for i in range(self.n):
            s, e, r = self.sigma[i], self.eta[i], self.rho[i]
            gi = (Bx * Bx).scale(0.5 * s * s) + (By * By).scale(0.5 * e * e) + (Bx * By).scale(r * s * e)
            g.append(gi); cov_bar = cov_bar + gi.scale(pi[i])
        gfuncs = [(lambda gi: (lambda t: gi.value(t)))(gi) for gi in g]
        # phi absorbs the averaged variance term so that the averaged model reproduces the curve: P = D(t) e^{-int cov_bar} a
        pre = lambda t: self.discount(t) * math.exp(-cov_bar.integral(t))
        return g, gfuncs, pre


# ---------------------------------------------------------------- operator form for the first-order tier
def _bs_operators(self, instrument, n, width):
    """Log-price grid; L_bar = (r - q - s2bar/2) d_x + s2bar/2 d_xx - r; the switched sigma^2 multiplies A = (d_xx - d_x)/2."""
    from .firstorder import Grid1D
    T, K = instrument.maturity, instrument.strike; pi = self.chain.stationaryDistribution()
    s2 = np.asarray(self.sigma) ** 2; s2bar = float(pi @ s2)
    L = width or 8 * math.sqrt(max(s2) * T) + 2 * abs(self.r - self.q) * T
    x0 = math.log(self.S0); grid = Grid1D(x0 - L, x0 + L, n)
    D1, D2 = grid.d1(), grid.d2(); I = sp.identity(n, format="csr")
    A = 0.5 * (D2 - D1)
    Lbar = (self.r - self.q) * D1 + s2bar * A - self.r * I
    S = np.exp(grid.x); u0 = np.maximum(S - K, 0.0) if instrument.isCall else np.maximum(K - S, 0.0)
    return Lbar, [A], [s2], grid, u0, x0


SwitchingBlackScholesProcess.operators = _bs_operators


class SwitchingCEVProcess(SwitchingModel):
    """dS = (r - q) S dt + sigma_y S^beta dW (QuantLib CEV process parameters S0, r, q, sigma, beta); sigma switches.
    Non-affine: the switched sigma^2 multiplies A = S^{2 beta} d_SS / 2, which does not commute with the drift."""
    def __init__(self, chain, S0, r, q, sigma, beta):
        super().__init__(chain)
        self.S0, self.r, self.q, self.beta = float(S0), float(r), float(q), float(beta)
        self.sigma = _per_regime(sigma, self.n)

    def operators(self, instrument, n, width):
        from .firstorder import Grid1D
        T, K = instrument.maturity, instrument.strike; pi = self.chain.stationaryDistribution()
        s2 = np.asarray(self.sigma) ** 2; s2bar = float(pi @ s2)
        vol_eff = math.sqrt(max(s2)) * self.S0 ** (self.beta - 1)
        L = width or 6 * vol_eff * math.sqrt(T) * self.S0 + 2 * abs(self.r - self.q) * T * self.S0
        grid = Grid1D(max(self.S0 - L, 1e-8), self.S0 + L, n)
        D1, D2 = grid.d1(), grid.d2(); I = sp.identity(n, format="csr"); S = grid.x
        A = 0.5 * sp.diags(S ** (2 * self.beta)) @ D2
        Lbar = (self.r - self.q) * sp.diags(S) @ D1 + s2bar * A - self.r * I
        u0 = np.maximum(S - K, 0.0) if instrument.isCall else np.maximum(K - S, 0.0)
        return Lbar, [A], [s2], grid, u0, self.S0
