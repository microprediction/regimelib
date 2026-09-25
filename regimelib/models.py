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

    def operators(self, instrument, n, width):
        """Short-rate grid; L_i = a (b_i - r) d_r + sigma_i^2 / 2 d_rr - r, split as L_bar plus the switched parts
        (b_i - b_bar) a d_r and (sigma_i^2 - s2_bar) d_rr / 2. No terminal payoff: rate instruments build their own."""
        from .firstorder import Grid1D
        T = instrument.maturity; pi = self.chain.stationaryDistribution()
        b, s2 = np.asarray(self.b, float), np.asarray(self.sigma, float) ** 2
        bbar, s2bar = float(pi @ b), float(pi @ s2)
        if isinstance(width, tuple):
            grid = Grid1D(width[0], width[1], n)
        else:
            sd = math.sqrt(max(s2) / (2 * self.a)); L = width or 8 * sd + abs(b.max() - b.min()) + abs(self.r0 - bbar)
            grid = Grid1D(self.r0 - L, self.r0 + L, n)
        D1, D2 = grid.d1(), grid.d2(); r = grid.x
        Adrift, Adiff = self.a * D1, 0.5 * D2
        Lbar = sp.diags(bbar - r) @ Adrift + s2bar * Adiff - sp.diags(r)
        # the drift forcing (b_i - b_bar) a d_r is constant in r; the diffusion forcing (s2_i - s2_bar) d_rr / 2
        return Lbar, [Adrift, Adiff], [b, s2], grid, None, self.r0

    def bondOnGrid(self, grid, t, S):
        """Zero-coupon bond at time t for maturity S on the rate grid, one row per regime: a_j(S - t) exp(-B(S - t) r)."""
        from .engines import _numericalAVector
        from ._engine.models import vasicek_terminal
        r = grid.x; tau = S - t
        if tau <= 0:
            return np.ones((self.n, len(r)))
        g, gfuncs, Bc = vasicek_terminal(self.a, self.b, self.sigma, 0.0)
        avec = _numericalAVector(self.chain.generator, g, gfuncs, tau).real
        return avec[:, None] * np.exp(-Bc.value(tau) * np.asarray(r, float)[None, :])

    def deterministicDiscount(self, t1, t2):
        """The part of the discounting the grid does not carry (none for Vasicek)."""
        return 1.0


class SwitchingVasicekJumps(SwitchingModel):
    """Vasicek with compound-Poisson jumps: dr = a (b_y - r) dt + sigma_y dW + dJ, J jumping at intensity
    jumpIntensity_y with exponential jump sizes of mean jumpMean. b, sigma and the intensity may switch; the
    reduction is exact (the forcing gains l_i (1 / (1 + m B) - 1)). QuantLib has no jump short-rate model; the frozen
    limit is checked against the affine closed form."""
    def __init__(self, chain, r0, a, b, sigma, jumpIntensity, jumpMean):
        super().__init__(chain)
        self.r0, self.a, self.jumpMean = float(r0), float(a), float(jumpMean)
        self.b, self.sigma = _per_regime(b, self.n), _per_regime(sigma, self.n)
        self.jumpIntensity = _per_regime(jumpIntensity, self.n)

    def bondForcing(self, T):
        g, gfuncs, pre = _m.vasicek_jumps(self.a, self.b, self.sigma, self.jumpIntensity, self.jumpMean, T)
        return g, gfuncs, (lambda t: pre(t, self.r0))


class SwitchingCoxIngersollRoss(SwitchingModel):
    """QuantLib CoxIngersollRoss(r0, theta, k, sigma): dr = k (theta - r) dt + sigma sqrt(r) dW. theta may switch."""
    def __init__(self, chain, r0, theta, k, sigma):
        super().__init__(chain)
        self.r0, self.k, self.sigma = float(r0), float(k), float(sigma)
        self.theta = _per_regime(theta, self.n)

    def bondForcing(self, T):
        g, gfuncs, pre, B = _m.cir_switching_mean(self.k, self.theta, self.sigma, T)
        return g, gfuncs, (lambda t: pre(t, self.r0))

    def operators(self, instrument, n, width):
        """Short-rate grid on [0, r_max]; L_i = k (theta_i - r) d_r + sigma^2 r / 2 d_rr - r, the switched part being
        the drift k (theta_i - theta_bar) d_r. The origin is an outflow boundary when the Feller condition holds."""
        from .firstorder import Grid1D
        pi = self.chain.stationaryDistribution(); th = np.asarray(self.theta, float); thbar = float(pi @ th)
        if isinstance(width, tuple):
            grid = Grid1D(width[0], width[1], n)
        else:
            top = width or max(self.r0, th.max()) + 10 * self.sigma * math.sqrt(max(self.r0, th.max()) / (2 * self.k))
            grid = Grid1D(0.0, top, n)
        D1, D2 = grid.d1(), grid.d2(); r = grid.x
        Adrift = self.k * D1
        Lbar = sp.diags(thbar - r) @ Adrift + 0.5 * self.sigma ** 2 * sp.diags(r) @ D2 - sp.diags(r)
        return Lbar, [Adrift], [th], grid, None, self.r0

    def bondOnGrid(self, grid, t, S):
        """Bond at time t for maturity S per regime: a_j(S - t) exp(-B(S - t) r), B the CIR Riccati solution."""
        from .engines import _numericalAVector
        r = grid.x; tau = S - t
        if tau <= 0:
            return np.ones((self.n, len(r)))
        g, gfuncs, pre, B = _m.cir_switching_mean(self.k, self.theta, self.sigma, tau)
        avec = _numericalAVector(self.chain.generator, g, gfuncs, tau).real
        return avec[:, None] * np.exp(-B(tau) * np.asarray(r, float)[None, :])

    def deterministicDiscount(self, t1, t2):
        return 1.0


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

    def averageForcing(self, u, T):
        """g_i(tau) for E exp(i u Y), Y = (1/T) int_0^T log(S_t / S_0) dt = int_0^T (1 - t/T) d log S_t: the weight
        (1 - t/T) multiplies the drift and squares against the variance. The reduced system runs in time to
        maturity tau = T - t, where the weight is tau / T."""
        mu = self.r - self.q
        def make(sig):
            return lambda tau: 1j * u * (mu - sig * sig / 2) * (tau / T) - u * u * sig * sig * (tau / T) ** 2 / 2
        fs = [make(float(sig)) for sig in np.atleast_1d(self.sigma)]
        return [Cheb.fit(f, T, deg=8) for f in fs], fs


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
        pre = lambda t: self.discount(t) * math.exp(-self._intShift(t))
        return g, gfuncs, pre

    def _intShift(self, t):
        """int_0^t of the averaged variance term of phi: s2bar (1 - e^{-a s})^2 / (2 a^2) integrated."""
        a = self.a; pi = self.chain.stationaryDistribution(); s2bar = float(pi @ np.asarray(self.sigma) ** 2)
        return s2bar / (2 * a * a) * (t - 2 * (1 - math.exp(-a * t)) / a + (1 - math.exp(-2 * a * t)) / (2 * a))

    def deterministicDiscount(self, t1, t2):
        """exp(-int_{t1}^{t2} phi), the discounting carried by the fitted drift rather than by the factor x."""
        return self.discount(t2) / self.discount(t1) * math.exp(-(self._intShift(t2) - self._intShift(t1)))

    def operators(self, instrument, n, width):
        """Grid in the zero-mean factor x (r = x + phi(t)); L_i = -a x d_x + sigma_i^2 / 2 d_xx - x, the phi part of the
        discounting being deterministic (deterministicDiscount). Started from x0 = 0."""
        from .firstorder import Grid1D
        pi = self.chain.stationaryDistribution(); s2 = np.asarray(self.sigma, float) ** 2; s2bar = float(pi @ s2)
        if isinstance(width, tuple):
            grid = Grid1D(width[0], width[1], n)
        else:
            L = width or 8 * math.sqrt(max(s2) / (2 * self.a))
            grid = Grid1D(-L, L, n)
        D1, D2 = grid.d1(), grid.d2(); x = grid.x
        Adiff = 0.5 * D2
        Lbar = -self.a * sp.diags(x) @ D1 + s2bar * Adiff - sp.diags(x)
        return Lbar, [Adiff], [s2], grid, None, 0.0

    def bondOnGrid(self, grid, t, S):
        """Bond at time t for maturity S on the factor grid, per regime: exp(-int_t^S phi) a_j(S - t) exp(-B(S - t) x)."""
        from .engines import _numericalAVector
        from ._engine.models import vasicek_terminal
        x = grid.x; tau = S - t
        if tau <= 0:
            return np.ones((self.n, len(x)))
        g, gfuncs, Bc = vasicek_terminal(self.a, [0.0] * self.n, self.sigma, 0.0)
        avec = _numericalAVector(self.chain.generator, g, gfuncs, tau).real
        return self.deterministicDiscount(t, S) * avec[:, None] * np.exp(-Bc.value(tau) * np.asarray(x, float)[None, :])


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

    def deterministicDiscount(self, t1, t2):
        """exp(-int_{t1}^{t2} phi): the curve factor and the averaged covariance term."""
        _, _, pre = self.bondForcing(t2)
        return pre(t2) / pre(t1)

    def operators(self, instrument, n, width):
        """Grid in the two zero-mean factors (x, y), r = x + y + phi(t): L_i = -a x d_x - b y d_y + sigma_i^2 d_xx / 2
        + eta_i^2 d_yy / 2 + rho_i sigma_i eta_i d_xy - (x + y); the phi part of the discounting is deterministic.
        n is (nx, ny) or one count for both; width the half-widths (Lx, Ly) or one number."""
        from .firstorder import Grid2D
        pi = self.chain.stationaryDistribution()
        sig, eta, rho = (np.asarray(v, float) for v in (self.sigma, self.eta, self.rho))
        f = [sig ** 2, eta ** 2, rho * sig * eta]; fbar = [float(pi @ v) for v in f]
        nx, ny = (n, n) if np.isscalar(n) else n
        if width is None:                                                  # six standard deviations at the last exercise
            Tend = max(getattr(instrument, "exerciseTimes", None) or [instrument.maturity])
            sdx = math.sqrt(max(sig ** 2) * (1 - math.exp(-2 * self.a * Tend)) / (2 * self.a))
            sdy = math.sqrt(max(eta ** 2) * (1 - math.exp(-2 * self.b * Tend)) / (2 * self.b))
            Lx, Ly = 6 * sdx, 6 * sdy
        else:
            Lx, Ly = (width, width) if np.isscalar(width) else width
        grid = Grid2D(-Lx, Lx, nx, -Ly, Ly, ny)
        A = [0.5 * grid.d2x, 0.5 * grid.d2v, grid.d1xv]
        Lbar = (-self.a * sp.diags(grid.X) @ grid.d1x - self.b * sp.diags(grid.V) @ grid.d1v
                + fbar[0] * A[0] + fbar[1] * A[1] + fbar[2] * A[2] - sp.diags(grid.X + grid.V))
        return Lbar, A, f, grid, None, (0.0, 0.0)

    def bondOnGrid(self, grid, t, S):
        """Bond at time t for maturity S on the factor grid per regime: exp(-int_t^S phi) a_j(S - t) exp(-B_a x - B_b y)."""
        from .engines import _numericalAVector
        from .g2options import _g2_forcing
        tau = S - t
        if tau <= 0:
            return np.ones((self.n, grid.n))
        g, gf = _g2_forcing(self.a, self.b, self.sigma, self.eta, self.rho, 0.0, 0.0)
        avec = _numericalAVector(self.chain.generator, g, gf, tau).real
        Ba, Bb = (1 - math.exp(-self.a * tau)) / self.a, (1 - math.exp(-self.b * tau)) / self.b
        return self.deterministicDiscount(t, S) * avec[:, None] * np.exp(-Ba * grid.X - Bb * grid.V)[None, :]


class SwitchingHestonVolOfVol(SwitchingModel):
    """Heston with the volatility of variance xi switching (QuantLib HestonProcess(r, q, S0, v0, kappa, theta, sigma,
    rho) with sigma = xi per regime). The switched operators v d_vv / 2 and rho v d_xv do not reduce exactly, so this
    model lives in the first-order tier: FirstOrderFDEngine on the (log S, v) grid, SwitchingFDReferee for the
    switching solution."""
    def __init__(self, chain, S0, r, q, v0, kappa, theta, xi, rho):
        super().__init__(chain)
        self.S0, self.r, self.q, self.v0 = float(S0), float(r), float(q), float(v0)
        self.kappa, self.theta, self.rho = float(kappa), float(theta), float(rho)
        self.xi = _per_regime(xi, self.n)

    def forward(self, T):
        return self.S0 * math.exp((self.r - self.q) * T)

    def operators(self, instrument, n, width):
        """n is (nx, nv) or a single count used for both; width the half-width in log price (and v_max as a multiple
        of max(v0, theta) is fixed at 5). L = (r - q - v/2) d_x + v/2 d_xx + kappa (theta - v) d_v
        + xi_bar^2 v/2 d_vv + rho xi_bar v d_xv - r; the switched forcings are xi^2 on v d_vv / 2 and xi on rho v d_xv."""
        from .firstorder import Grid2D
        T, K = instrument.maturity, instrument.strike; pi = self.chain.stationaryDistribution()
        xi = np.asarray(self.xi, float); xibar, xi2bar = float(pi @ xi), float(pi @ xi ** 2)
        nx, nv = (n, n) if np.isscalar(n) else n
        vtop = max(self.v0, self.theta); L = width or 6 * math.sqrt(vtop * T) + 2 * abs(self.r - self.q) * T
        x0 = math.log(self.S0); grid = Grid2D(x0 - L, x0 + L, nx, 0.0, 5 * vtop, nv)
        V = sp.diags(grid.V); I = sp.identity(grid.n, format="csr")
        A1 = 0.5 * V @ grid.d2v                                        # multiplies xi^2
        A2 = self.rho * V @ grid.d1xv                                  # multiplies xi
        Lbar = (sp.diags(self.r - self.q - 0.5 * grid.V) @ grid.d1x + 0.5 * V @ grid.d2x
                + sp.diags(self.kappa * (self.theta - grid.V)) @ grid.d1v + xi2bar * A1 + xibar * A2 - self.r * I)
        S = np.exp(grid.X)
        u0 = np.maximum(S - K, 0.0) if instrument.isCall else np.maximum(K - S, 0.0)
        return Lbar, [A1, A2], [xi ** 2, xi], grid, u0, (x0, self.v0)


# ---------------------------------------------------------------- several intensities on one regime chain
class SwitchingIntensityBasket(SwitchingModel):
    """Default intensities lambda^1, ..., lambda^K (SwitchingVasicek or SwitchingCoxIngersollRoss instances) driven
    by the same regime chain, with independent diffusions. The joint survival E exp(-int sum_k lambda^k) is the
    bond of the summed forcing with the product of the prefactors, so ZeroCouponBond(t) is the probability that no
    name has defaulted by t and CreditDefaultSwap is a first-to-default swap. The common regime is the only source
    of dependence: defaultCorrelation(t) measures it."""
    def __init__(self, models):
        chain = models[0].chain
        if any(m.chain is not chain for m in models):
            raise ValueError("every intensity must be driven by the same RegimeChain instance")
        super().__init__(chain)
        self.models = list(models)

    def bondForcing(self, T):
        parts = [m.bondForcing(T) for m in self.models]
        if len({type(p[0][0]) for p in parts}) > 1:                    # exponential sums and Chebyshev series: fit all
            parts = [([Cheb.fit(gf, T, 80) for gf in p[1]], p[1], p[2]) for p in parts]
        g = [sum((p[0][i] for p in parts[1:]), parts[0][0][i]) for i in range(self.n)]
        gfuncs = [(lambda gi: (lambda t: gi.value(t)))(gi) for gi in g]
        pres = [p[2] for p in parts]
        return g, gfuncs, (lambda t: math.prod(pre(t) for pre in pres))

    def defaultCorrelation(self, t, regime=0):
        """Correlation of the default indicators of the first two names by time t, from the joint and marginal survivals."""
        from .engines import NumericalSwitchingEngine
        from .instruments import ZeroCouponBond
        def surv(model):
            b = ZeroCouponBond(t); b.setPricingEngine(NumericalSwitchingEngine(model, regime=regime)); return b.NPV()
        q1, q2, q12 = surv(self.models[0]), surv(self.models[1]), surv(SwitchingIntensityBasket(self.models[:2]))
        return (q12 - q1 * q2) / math.sqrt(q1 * (1 - q1) * q2 * (1 - q2))


# ---------------------------------------------------------------- operator form for the first-order tier
def _bs_operators(self, instrument, n, width):
    """Log-price grid; L_bar = (r - q - s2bar/2) d_x + s2bar/2 d_xx - r; the switched sigma^2 multiplies A = (d_xx - d_x)/2."""
    from .firstorder import Grid1D
    T, K = instrument.maturity, instrument.strike; pi = self.chain.stationaryDistribution()
    s2 = np.asarray(self.sigma) ** 2; s2bar = float(pi @ s2)
    x0 = math.log(self.S0)
    if isinstance(width, tuple):                                   # explicit log-price ends (barrier grids)
        grid = Grid1D(width[0], width[1], n)
    else:
        L = width or 8 * math.sqrt(max(s2) * T) + 2 * abs(self.r - self.q) * T
        grid = Grid1D(x0 - L, x0 + L, n)
    D1, D2 = grid.d1(), grid.d2(); I = sp.identity(n, format="csr")
    A = 0.5 * (D2 - D1)
    Lbar = (self.r - self.q) * D1 + s2bar * A - self.r * I
    S = np.exp(grid.x); u0 = instrument.payoffOnGrid(S) if hasattr(instrument, "payoffOnGrid") else (np.maximum(S - K, 0.0) if instrument.isCall else np.maximum(K - S, 0.0))
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
        grid = Grid1D(width[0], width[1], n) if isinstance(width, tuple) else Grid1D(max(self.S0 - L, 1e-8), self.S0 + L, n)
        D1, D2 = grid.d1(), grid.d2(); I = sp.identity(n, format="csr"); S = grid.x
        A = 0.5 * sp.diags(S ** (2 * self.beta)) @ D2
        Lbar = (self.r - self.q) * sp.diags(S) @ D1 + s2bar * A - self.r * I
        u0 = np.maximum(S - K, 0.0) if instrument.isCall else np.maximum(K - S, 0.0)
        return Lbar, [A], [s2], grid, u0, self.S0
