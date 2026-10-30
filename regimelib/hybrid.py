"""Equity with stochastic rates on one regime chain. The equity is a SwitchingBlackScholesProcess or
SwitchingHestonModel, the rates a SwitchingVasicek or SwitchingHullWhite, with independent Brownian motions; the
regime path is common, so the discount and the return are dependent through it. The discounted characteristic
function Psi(w) = E[exp(-int_0^T r) S_T^{i w}] splits, given the regime path, into the rates part
E[exp((i w - 1) int r)] (the Gaussian factor formula with weight c = 1 - i w) and the martingale return's
characteristic function; the reduced system takes the sum of the two forcings. Lewis's formula in discounted form:

    C = S0 e^{-qT} - (sqrt(K) / pi) int_0^inf Re[K^{-iu} Psi(u - i/2)] / (u^2 + 1/4) du."""
import math
import cmath
import numpy as np
import scipy.sparse as sp
from ._engine import models as _m
from ._engine.fastswitch import Cheb
from ._engine.fastswitch import ExpSum
from .models import SwitchingModel, SwitchingVasicek, SwitchingHullWhite, SwitchingBlackScholesProcess


class SwitchingEquityRates(SwitchingModel):
    defaultGrid = (161, 61)                            # (log price, rate) nodes when an engine is given none
    """`rho` is the equity-rate Brownian correlation, allowed for a Black-Scholes equity (the exponent stays Gaussian
    given the regime path, and the cross term -c i w rho sigma_S sigma_r B(tau) joins the forcing); Heston needs rho = 0."""
    def __init__(self, equity, rates, rho=0.0):
        if equity.chain is not rates.chain:
            raise ValueError("the equity and the rates must share one RegimeChain instance")
        if not (callable(getattr(equity, "returnForcing", None)) and hasattr(equity, "S0") and hasattr(equity, "q")):
            raise TypeError("the equity must be a model with a characteristic function (returnForcing, S0 and q); "
                            f"{type(equity).__name__} has none")
        rho = float(rho)
        if not math.isfinite(rho) or abs(rho) > 1.0:
            raise ValueError("rho must be finite and between -1 and 1")
        if not isinstance(rates, (SwitchingVasicek, SwitchingHullWhite)):
            raise TypeError("rates must be SwitchingVasicek or SwitchingHullWhite")
        if rho and not isinstance(equity, SwitchingBlackScholesProcess):
            raise ValueError("an equity-rate correlation is supported for the Black-Scholes equity only")
        super().__init__(equity.chain)
        self.equity, self.rates, self.rho = equity, rates, float(rho)
        self.S0, self.q, self.r = equity.S0, equity.q, None                  # r is stochastic

    def discountedForcing(self, w, T):
        """Forcing and prefactor for Psi(w): returns (g, gfuncs, prefactor value)."""
        c = 1 - 1j * w
        R = self.rates
        if isinstance(R, SwitchingVasicek):
            gr, gfr, pre_r = _m.gaussian_factors([R.a], [R.b], [R.sigma], [[[1.0]]] * self.n, [c], T)
            pre = pre_r(T, [R.r0])
        else:
            gr, gfr, pre_r = _m.gaussian_factors([R.a], [[0.0] * self.n], [R.sigma], [[[1.0]]] * self.n, [c], T)
            pre = R.deterministicDiscount(0.0, T) ** c                        # exp((i w - 1) int phi)
        ge, gfe, pre_e = self.equity.returnForcing(w, T)
        if self.rho:
            B, Bf = _m.loading(R.a, 0.0, T)                                   # a Chebyshev series near zero reversion
            sS = np.atleast_1d(np.asarray(self.equity.sigma, float)); sR = np.atleast_1d(np.asarray(R.sigma, float))
            cross = [-c * 1j * w * self.rho * float(sS[i]) * float(sR[i]) for i in range(self.n)]
            if type(B) is not type(ge[0]):                                    # one representation before adding
                ge = [Cheb.fit(f, T, 80) for f in gfe]; B = Cheb.fit(Bf, T, 24) if not isinstance(B, Cheb) else B
            ge = [ge[i] + B.scale(cross[i]) for i in range(self.n)]
            gfe = [(lambda f, x: (lambda t: f(t) + x * Bf(t)))(gfe[i], cross[i]) for i in range(self.n)]
        if type(gr[0]) is not type(ge[0]):
            gr = [Cheb.fit(f, T, 80) for f in gfr]; ge = [Cheb.fit(f, T, 80) for f in gfe]
        g = [gr[i] + ge[i] for i in range(self.n)]
        gfuncs = [(lambda fr, fe: (lambda t: fr(t) + fe(t)))(gfr[i], gfe[i]) for i in range(self.n)]
        factor = pre * pre_e() * self.S0 ** (1j * w) * cmath.exp(-1j * w * self.q * T)
        return g, gfuncs, factor

    def operators(self, instrument, n, width, stretch=None):
        """Grid in (y, r), y = log S - c r, for a Black-Scholes equity with Vasicek rates (the grid engine; Hull-White's
        fitted drift is time dependent and is not gridded). In (log S, r) the operator is L = (r - q - s2/2) d_x
        + s2/2 d_xx + a (b - r) d_r + sr2/2 d_rr + rho sigma sigma_r d_xr - r, and a strong correlation leaves the
        terminal log price as the small remainder of large diffusions that a mixed-derivative stencil resolves badly.
        c = (rho sigma sigma_r)_bar / (sigma_r^2)_bar removes the averaged cross term: with d_x = d_y and
        d_r -> d_r - c d_y, the switched forcings are sigma^2 on (d_yy - d_y)/2, b on a (d_r - c d_y), sigma_r^2 on
        (d_rr - 2c d_yr + c^2 d_yy)/2 and rho sigma sigma_r on d_yr - c d_yy, and the d_yr parts cancel in L_bar. n is
        (ny, nr) or one count; width the half-widths (Ly, Lr). The derivatives in y at fixed r are those in log S."""
        from .firstorder import Grid2D
        E, R = self.equity, self.rates
        if not isinstance(E, SwitchingBlackScholesProcess) or not isinstance(R, SwitchingVasicek):
            raise TypeError("the hybrid grid engine takes a Black-Scholes equity with Vasicek rates")
        T = instrument.maturity; pi = self.chain.stationaryDistribution()
        sS = np.atleast_1d(np.asarray(E.sigma, float)); sR = np.atleast_1d(np.asarray(R.sigma, float)); b = np.atleast_1d(np.asarray(R.b, float))
        sS, sR = np.broadcast_to(sS, (self.n,)), np.broadcast_to(sR, (self.n,))
        f = [sS ** 2, b, sR ** 2, self.rho * sS * sR]; fbar = [float(pi @ v) for v in f]
        c = fbar[3] / fbar[2] if fbar[2] > 0 else 0.0
        nx, nr = (n, n) if np.isscalar(n) else n
        if width is None:
            # y_T - y_0 = int sigma dW_S + int (k B(T - s) - c) sigma_r dW_r plus a mean, k = 1 + c a: in each regime,
            # frozen, its variance is sigma^2 T + sigma_r^2 (c^2 T - 2 c k I1 + k^2 I2) + 2 rho sigma sigma_r (k I1 - c T)
            # with I1, I2 the integrals of B and B^2 over [0, T]. The rate's mean moves a fraction 1 - e^{-a t} of the
            # way to a level by t, whatever the regime path, so by T it has carried log S by at most
            # |r0| T + max |b - r0| (T - B(T)), and r by (1 - e^{-a T}) max |b - r0|
            gap = abs(b - R.r0).max(); a = R.a; k = 1 + c * a
            I1 = (T - _m.stable_B(a, T)) / a if a * T >= _m.SMALL_SPEED else T * T / 2 - a * T ** 3 / 6
            I2 = _m.int_B2(a, T)
            var = sS ** 2 * T + sR ** 2 * (c * c * T - 2 * c * k * I1 + k * k * I2) + 2 * self.rho * sS * sR * (k * I1 - c * T)
            sd = math.sqrt(max(float(var.max()), 0.0))
            moveR = -math.expm1(-a * T) * gap
            Lx = 8 * sd + 2 * (abs(R.r0) * T + gap * (T - _m.stable_B(a, T)) + abs(E.q) * T + abs(c) * moveR)
            Lr = 6 * math.sqrt(max(sR ** 2) * _m.ou_variance(a, T)) + moveR
        else:
            Lx, Lr = (width, width) if np.isscalar(width) else width
        y0 = math.log(E.S0) - c * R.r0
        grid = Grid2D(y0 - Lx, y0 + Lx, nx, R.r0 - Lr, R.r0 + Lr, nr,
                      centers=(math.log(instrument.strike) - c * R.r0, R.r0), stretch=stretch)
        Rd = sp.diags(grid.V)
        # one cross stencil in both forcings, so that their d_yr parts cancel exactly in L_bar
        X = grid.d1xv if c else grid.cross(np.sign(self.rho))
        A = [0.5 * (grid.d2x - grid.d1x), R.a * (grid.d1v - c * grid.d1x), 0.5 * (grid.d2v - 2 * c * X + c * c * grid.d2x),
             X - c * grid.d2x]
        Lbar = (sp.diags(grid.V - E.q) @ grid.d1x + fbar[0] * A[0] + sp.diags(fbar[1] - grid.V) @ A[1]
                + fbar[2] * A[2] + fbar[3] * A[3] - Rd)
        S = np.exp(grid.X + c * grid.V)
        u0 = instrument.payoffOnGrid(S)
        return Lbar, A, f, grid, u0, (y0, R.r0)
