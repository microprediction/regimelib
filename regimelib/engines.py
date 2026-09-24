"""Pricing engines. FastSwitchingEngine expands in the mean holding time to the given order; NumericalSwitchingEngine
solves the reduced system a' = (Q + diag g) a numerically and is the referee. Both take the starting regime."""
import math
import cmath
import numpy as np
from ._engine.fastswitch import FastSwitch, numerical_a_callable
from .instruments import ZeroCouponBond, VanillaOption, ZeroCouponBondOption
from ._engine.options import zcb_call
from .models import SwitchingVasicek, SwitchingHullWhite


def _gauss(U, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1) * U / 2, w * U / 2


class SwitchingEngine:
    def __init__(self, model, regime=0, nodes=96):
        self.model, self.regime, self.nodes = model, regime, nodes

    # a(T) for the reduced system; subclasses choose the method
    def _a(self, g, gfuncs, T, a0=None):
        raise NotImplementedError

    def calculate(self, instrument):
        m, T = self.model, instrument.maturity
        if isinstance(instrument, ZeroCouponBond):
            g, gfuncs, pre = m.bondForcing(T)
            a0 = None
            if instrument.regimeAtMaturity is not None:
                a0 = np.zeros(m.n); a0[instrument.regimeAtMaturity] = 1.0
            return float(np.real(pre(T) * self._a(g, gfuncs, T, a0)))
        if isinstance(instrument, VanillaOption):
            return self._vanilla(instrument)
        if isinstance(instrument, ZeroCouponBondOption):
            return self._bondOption(instrument)
        raise TypeError("unsupported instrument")

    def _bondOption(self, opt):
        m = self.model; T, S, K = opt.maturity, opt.bondMaturity, opt.strike
        if isinstance(m, SwitchingVasicek):
            call = zcb_call(T, S, K, m.r0, self.regime, m.a, m.b, m.sigma, m.chain.generator, order=self._order())
        elif isinstance(m, SwitchingHullWhite):
            # r = x + phi(t): P(T, S) = c P_x(T, S) with c = exp(-int_T^S phi), and the discount to T carries
            # exp(-int_0^T phi), so the call is exp(-int_0^T phi) c Call_x(strike K / c) under the zero-mean factor.
            a = m.a; pi = m.chain.stationaryDistribution(); s2 = float(pi @ np.asarray(m.sigma) ** 2)
            shift = lambda t: s2 / (2 * a * a) * (t - 2 * (1 - math.exp(-a * t)) / a + (1 - math.exp(-2 * a * t)) / (2 * a))
            e0T = m.discount(T) * math.exp(-shift(T)); c = m.discount(S) / m.discount(T) * math.exp(-(shift(S) - shift(T)))
            call = e0T * c * zcb_call(T, S, K / c, 0.0, self.regime, a, [0.0] * m.n, m.sigma, m.chain.generator, order=self._order())
        else:
            raise TypeError("bond options are priced under SwitchingVasicek or SwitchingHullWhite")
        if opt.isCall:
            return call
        bond = lambda t: self.calculate(ZeroCouponBond(t))          # put-call parity: C - P = P(0,S) - K P(0,T)
        return call - bond(S) + K * bond(T)

    def _order(self):
        return None

    def _vanilla(self, opt):
        """Lewis (2001): C = e^{-rT} [F - sqrt(F K) / pi int_0^inf Re(e^{i u k} phi(u - i/2)) / (u^2 + 1/4) du]."""
        m, T, K = self.model, opt.maturity, opt.strike
        F = m.forward(T); k = math.log(F / K)
        U = self._frequencyLimit(T, k)
        us, ws = _gauss(U, self._nodeCount(U, k))
        tot = 0.0
        for u, w in zip(us, ws):
            g, gfuncs, pre = m.returnForcing(u - 0.5j, T)
            phi = pre() * self._a(g, gfuncs, T)
            tot += w * (cmath.exp(1j * u * k) * phi).real / (u * u + 0.25)
        call = math.exp(-m.r * T) * (F - math.sqrt(F * K) / math.pi * tot)
        return call if opt.isCall else call - math.exp(-m.r * T) * (F - K)

    def _frequencyLimit(self, T, k=0.0):
        """Frequency beyond which the Lewis integrand is below 1e-16 under the averaged model, found by doubling.
        The averaged characteristic function is the prefactor times exp of the integral of the stationary-weighted
        forcing, which every model exposes in closed form."""
        m = self.model; pi = m.chain.stationaryDistribution()
        def size(u):
            g, gfuncs, pre = m.returnForcing(u - 0.5j, T)
            gbar = sum((g[i].scale(pi[i]) for i in range(1, len(g))), g[0].scale(pi[0]))
            return abs(pre() * cmath.exp(gbar.integral(T))) / (u * u + 0.25)
        U = 8.0
        while size(U) > 1e-16 and U < 1e4:
            U *= 2
        return U

    def _nodeCount(self, U, k):
        return int(min(4000, max(self.nodes, 2 * U * (1 + abs(k)))))

class FastSwitchingEngine(SwitchingEngine):
    """order: an integer, or None to add terms until successive orders agree to `tol` (relative) or the next term
    stops shrinking, the best truncation of an asymptotic series. `maxOrder` bounds the search. After calculate(),
    `orderUsed` and `lastIncrement` (relative size of the last term kept) are set."""
    def __init__(self, model, order=4, regime=0, nodes=96, tol=1e-10, maxOrder=12):
        super().__init__(model, regime, nodes)
        self.order, self.tol, self.maxOrder = order, tol, maxOrder
        self.orderUsed = self.lastIncrement = None

    def _a(self, g, gfuncs, T, a0=None):
        fs = FastSwitch(self.model.chain.generator, g, order=self._order(), a0=a0)
        return fs.a(T, self._order())[self.regime]

    def _order(self):
        return self.order if self.order is not None else self.maxOrder

    def calculate(self, instrument):
        if self.order is not None:
            self.orderUsed = self.order; return super().calculate(instrument)
        prev, prev_inc = None, None
        for n in range(0, self.maxOrder + 1):
            self.order = n
            try:
                v = super().calculate(instrument)
            finally:
                self.order = None
            if prev is not None:
                inc = abs(v - prev) / max(abs(v), 1e-300)
                if inc <= self.tol or (prev_inc is not None and inc > prev_inc):
                    # converged, or the terms have started to grow: keep the best truncation
                    self.orderUsed, self.lastIncrement = (n if inc <= self.tol else n - 1), min(inc, prev_inc or inc)
                    return v if inc <= self.tol else prev
                prev_inc = inc
            prev = v
        self.orderUsed, self.lastIncrement = self.maxOrder, prev_inc
        return prev


class NumericalSwitchingEngine(SwitchingEngine):
    def __init__(self, model, regime=0, nodes=96, rtol=1e-12):
        super().__init__(model, regime, nodes)
        self.rtol = rtol

    def _a(self, g, gfuncs, T, a0=None):
        return numerical_a_callable(T, self.model.chain.generator, gfuncs, rtol=self.rtol, a0=a0)[self.regime]
