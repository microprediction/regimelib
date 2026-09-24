"""Pricing engines. FastSwitchingEngine expands in the mean holding time to the given order; NumericalSwitchingEngine
solves the reduced system a' = (Q + diag g) a numerically and is the referee. Both take the starting regime."""
import math
import cmath
import numpy as np
from ._engine.fastswitch import FastSwitch, numerical_a_callable
from .instruments import ZeroCouponBond, VanillaOption


def _gauss(U, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1) * U / 2, w * U / 2


class SwitchingEngine:
    def __init__(self, model, regime=0, nodes=96):
        self.model, self.regime, self.nodes = model, regime, nodes

    # a(T) for the reduced system; subclasses choose the method
    def _a(self, g, gfuncs, T):
        raise NotImplementedError

    def calculate(self, instrument):
        m, T = self.model, instrument.maturity
        if isinstance(instrument, ZeroCouponBond):
            g, gfuncs, pre = m.bondForcing(T)
            return float(np.real(pre(T) * self._a(g, gfuncs, T)))
        if isinstance(instrument, VanillaOption):
            return self._vanilla(instrument)
        raise TypeError("unsupported instrument")

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
    def __init__(self, model, order=4, regime=0, nodes=96):
        super().__init__(model, regime, nodes)
        self.order = order

    def _a(self, g, gfuncs, T):
        return FastSwitch(self.model.chain.generator, g, order=self.order).a(T, self.order)[self.regime]


class NumericalSwitchingEngine(SwitchingEngine):
    def __init__(self, model, regime=0, nodes=96, rtol=1e-12):
        super().__init__(model, regime, nodes)
        self.rtol = rtol

    def _a(self, g, gfuncs, T):
        return numerical_a_callable(T, self.model.chain.generator, gfuncs, rtol=self.rtol)[self.regime]
