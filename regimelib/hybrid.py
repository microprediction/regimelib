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
from ._engine import models as _m
from ._engine.fastswitch import Cheb
from ._engine.fastswitch import ExpSum
from .models import SwitchingModel, SwitchingVasicek, SwitchingHullWhite, SwitchingBlackScholesProcess


class SwitchingEquityRates(SwitchingModel):
    """`rho` is the equity-rate Brownian correlation, allowed for a Black-Scholes equity (the exponent stays Gaussian
    given the regime path, and the cross term -c i w rho sigma_S sigma_r B(tau) joins the forcing); Heston needs rho = 0."""
    def __init__(self, equity, rates, rho=0.0):
        if equity.chain is not rates.chain:
            raise ValueError("the equity and the rates must share one RegimeChain instance")
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
            gr, gfr, pre_r = _m.gaussian_factors([R.a], [R.b], [R.sigma], [[[1.0]]] * self.n, [c])
            pre = pre_r(T, [R.r0])
        else:
            gr, gfr, pre_r = _m.gaussian_factors([R.a], [[0.0] * self.n], [R.sigma], [[[1.0]]] * self.n, [c])
            pre = R.deterministicDiscount(0.0, T) ** c                        # exp((i w - 1) int phi)
        ge, gfe, pre_e = self.equity.returnForcing(w, T)
        if self.rho:
            B = ExpSum({0: 1 / R.a, R.a: -1 / R.a})
            sS = np.atleast_1d(np.asarray(self.equity.sigma, float)); sR = np.atleast_1d(np.asarray(R.sigma, float))
            ge = [ge[i] + B.scale(-c * 1j * w * self.rho * float(sS[i]) * float(sR[i])) for i in range(self.n)]
            gfe = [(lambda gi: (lambda t: gi.value(t)))(gi) for gi in ge]
        if type(gr[0]) is not type(ge[0]):
            gr = [Cheb.fit(f, T, 80) for f in gfr]; ge = [Cheb.fit(f, T, 80) for f in gfe]
        g = [gr[i] + ge[i] for i in range(self.n)]
        gfuncs = [(lambda gi: (lambda t: gi.value(t)))(gi) for gi in g]
        factor = pre * pre_e() * self.S0 ** (1j * w) * cmath.exp(-1j * w * self.q * T)
        return g, gfuncs, factor
