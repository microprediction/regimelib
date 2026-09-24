"""Pricing engines. FastSwitchingEngine expands in the mean holding time to the given order; NumericalSwitchingEngine
solves the reduced system a' = (Q + diag g) a numerically and is the referee. Both take the starting regime."""
import math
import cmath
import warnings
import numpy as np
from ._engine.fastswitch import FastSwitch, numerical_a_callable, Cheb
Cheb.MAXDEG = 400      # products of fitted forcings (Heston, CIR) at higher orders and with several regimes need room
from .instruments import ZeroCouponBond, VanillaOption, ZeroCouponBondOption, CouponBond
from ._engine.options import zcb_call
from .models import SwitchingVasicek, SwitchingHullWhite


def _gauss(U, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1) * U / 2, w * U / 2


class ExpansionWarning(UserWarning):
    """The fast-switching expansion may not have converged for this instrument and chain."""


class SwitchingEngine:
    supportsResults = True

    def __init__(self, model, regime=0, nodes=96):
        self.model, self.regime, self.nodes = model, regime, nodes

    # a(T) for the reduced system; subclasses choose the method
    def _a(self, g, gfuncs, T, a0=None):
        raise NotImplementedError

    def calculate(self, instrument, results=False):
        m, T = self.model, instrument.maturity
        if isinstance(instrument, ZeroCouponBond):
            g, gfuncs, pre = m.bondForcing(T)
            a0 = None
            if instrument.regimeAtMaturity is not None:
                a0 = np.zeros(m.n); a0[instrument.regimeAtMaturity] = 1.0
            P = float(np.real(pre(T) * self._a(g, gfuncs, T, a0)))
            out = {"value": P}
            B = self._bondB(T)
            if B is not None:                                     # sensitivities to the state r0
                out.update(delta=-B * P, gamma=B * B * P)
        elif isinstance(instrument, VanillaOption):
            out = self._digital(instrument) if instrument.payoffType != "vanilla" else self._vanillaAll(instrument)
        elif isinstance(instrument, CouponBond):
            out = {"value": 0.0, "delta": 0.0, "gamma": 0.0}
            for t, c in instrument.cashflows:
                r = self.calculate(ZeroCouponBond(t), results=True)
                for key in out:
                    out[key] += c * r.get(key, math.nan)
        elif isinstance(instrument, ZeroCouponBondOption):
            out = {"value": self._bondOption(instrument)}
        else:
            raise TypeError("unsupported instrument")
        return out if results else out["value"]

    def _bondB(self, T):
        m = self.model
        if isinstance(m, SwitchingVasicek):
            return (1 - math.exp(-m.a * T)) / m.a
        if hasattr(m, "k") and hasattr(m, "theta") and not hasattr(m, "S0"):      # CIR
            h = math.sqrt(m.k ** 2 + 2 * m.sigma ** 2); ex = math.exp(h * T) - 1
            return 2 * ex / ((h + m.k) * ex + 2 * h)
        return None

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
        return self._vanillaAll(opt)["value"]

    def _vanillaAll(self, opt):
        """Lewis (2001): C = e^{-rT} [F - sqrt(F K) / pi int_0^inf Re(e^{i u k} phi(u - i/2)) / (u^2 + 1/4) du],
        k = ln(F/K). Differentiating in F puts (1/2 + i u) under the integral for delta and rho and -(u^2 + 1/4)
        for gamma; in v0 it puts the Riccati coefficient D(T); in T it puts d phi / dT, which the reduced system gives
        as (Q + diag g(T)) a plus the derivative of the prefactor."""
        m, T, K = self.model, opt.maturity, opt.strike
        F = m.forward(T); k = math.log(F / K); dF = F / m.S0; mu = m.r - m.q
        U = self._frequencyLimit(T, k)
        us, ws = _gauss(U, self._nodeCount(U, k))
        I0 = I1 = I2 = Iv = It = 0.0
        stoch = hasattr(m, "v0")
        for u, w in zip(us, ws):
            z = u - 0.5j
            g, gfuncs, pre = m.returnForcing(z, T)
            avec, dvec = self._aVector(g, gfuncs, T)
            a = avec[self.regime]; phi = pre() * a
            e = cmath.exp(1j * u * k) * phi
            I0 += w * e.real / (u * u + 0.25)
            I1 += w * ((0.5 + 1j * u) * e).real / (u * u + 0.25)
            I2 += w * e.real
            if stoch:
                D, dD = self._riccatiD(m, z, T)
                Iv += w * (e * D).real / (u * u + 0.25)
                dphi = pre() * (dD * m.v0 * a + dvec[self.regime])
            else:
                dphi = pre() * dvec[self.regime]
            It += w * (cmath.exp(1j * u * k) * ((0.5 + 1j * u) * mu * phi + dphi)).real / (u * u + 0.25)
        disc = math.exp(-m.r * T); root = math.sqrt(F * K)
        call = disc * (F - root / math.pi * I0)
        delta = disc * dF * (1 - root / (math.pi * F) * I1)
        gamma = disc * dF * dF * root / (math.pi * F * F) * I2
        dC_dT = -m.r * call + disc * (mu * F - root / math.pi * It)
        rho = -T * call + disc * T * (F - root / math.pi * I1)
        out = dict(value=call, delta=delta, gamma=gamma, theta=-dC_dT, rho=rho)
        if stoch:
            out["vega"] = -disc * root / math.pi * Iv                 # in v0
        if not opt.isCall:                                    # put-call parity: P = C - e^{-rT} (F - K)
            out["value"] = call - disc * (F - K); out["delta"] = delta - disc * dF
            out["theta"] = -(dC_dT - (-m.r * disc * (F - K) + disc * mu * F))
            out["rho"] = rho - T * K * disc
        return out

    def _aVector(self, g, gfuncs, T, a0=None):
        """a(T) over all regimes and its time derivative (Q + diag g(T)) a(T)."""
        raise NotImplementedError

    def _digital(self, opt):
        """Gil-Pelaez: P(S_T > K) = 1/2 + (1/pi) int_0^inf Re(e^{-i u k} phi(u) / (i u)) du with k = ln(K/F) and phi the
        characteristic function of the log return; asset-or-nothing uses the share measure, phi(u - i) / phi(-i)."""
        m, T, K = self.model, opt.maturity, opt.strike
        F = m.forward(T); k = math.log(K / F); disc = math.exp(-m.r * T)
        U = self._frequencyLimit(T, k); us, ws = _gauss(U, self._nodeCount(U, k))
        shift = -1j if opt.payoffType == "asset" else 0.0
        I = 0.0
        for u, w in zip(us, ws):
            g, gfuncs, pre = m.returnForcing(u + shift, T)
            phi = pre() * self._a(g, gfuncs, T)
            I += w * (cmath.exp(-1j * u * k) * phi / (1j * u)).real
        prob = 0.5 + I / math.pi                              # P(S_T > K) under the relevant measure
        if opt.payoffType == "cash":
            value = disc * opt.cash * (prob if opt.isCall else 1 - prob)
        else:
            value = disc * F * (prob if opt.isCall else 1 - prob)
        return {"value": value}

    @staticmethod
    def _riccatiD(m, u, T):
        """The regime-free coefficient of v0 in the log characteristic function (Heston, Bates) and its T-derivative
        from the Riccati equation D' = xi^2 D^2 / 2 + (rho xi i u - kappa) D - (u^2 + i u) / 2."""
        d = cmath.sqrt((m.rho * m.sigma * 1j * u - m.kappa) ** 2 + m.sigma ** 2 * (1j * u + u * u))
        gm = (m.kappa - m.rho * m.sigma * 1j * u - d) / (m.kappa - m.rho * m.sigma * 1j * u + d)
        e = cmath.exp(-d * T)
        D = (m.kappa - m.rho * m.sigma * 1j * u - d) / m.sigma ** 2 * (1 - e) / (1 - gm * e)
        dD = 0.5 * m.sigma ** 2 * D * D + (m.rho * m.sigma * 1j * u - m.kappa) * D - 0.5 * (u * u + 1j * u)
        return D, dD

    def greeks(self, instrument):
        """All results the engine provides for the instrument, as a dict."""
        out = self.calculate(instrument, results=True); out.pop("value"); return out

    def _frequencyLimit(self, T, k=0.0):
        """Frequency beyond which the averaged characteristic function is below 1e-14, found by doubling; this bounds
        the price, delta, gamma and vega integrands alike.
        The averaged characteristic function is the prefactor times exp of the integral of the stationary-weighted
        forcing, which every model exposes in closed form."""
        m = self.model; pi = m.chain.stationaryDistribution()
        def size(u):
            g, gfuncs, pre = m.returnForcing(u - 0.5j, T)
            gbar = sum((g[i].scale(pi[i]) for i in range(1, len(g))), g[0].scale(pi[0]))
            return abs(pre() * cmath.exp(gbar.integral(T)))          # no 1/(u^2 + 1/4): the gamma integrand has none
        U = 8.0
        while size(U) > 1e-14 and U < 1e4:
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

    def _aVector(self, g, gfuncs, T, a0=None):
        """a(T) over all regimes through the engine's order, with the size of the last two terms recorded for the
        convergence diagnostics; where the expansion factor over the averaged value is not moderate (large forcing at
        high frequency) the averaged value is used at that node and `tailFallbacks` is incremented."""
        N = self._order()
        fs = FastSwitch(self.model.chain.generator, g, order=N, a0=a0)
        base = np.asarray(fs.a(T, 0), complex)
        with np.errstate(all="ignore"):
            try:
                avec = np.asarray(fs.a(T, N), complex)
                prev = np.asarray(fs.a(T, N - 1), complex) if N >= 1 else base
                prev2 = np.asarray(fs.a(T, N - 2), complex) if N >= 2 else prev
            except (OverflowError, FloatingPointError, ValueError):
                avec = np.full(self.model.n, np.nan, complex); prev = prev2 = base
        ok = np.all(np.isfinite(avec)) and not np.any(np.abs(avec) > 1e3 * np.maximum(np.abs(base), 1e-300))
        if not ok:
            self._diag["tailFallbacks"] += 1
            self._diag["tailWeight"] = max(self._diag["tailWeight"], float(np.max(np.abs(base))))
            avec = base
        else:
            i = self.regime; scale = max(abs(avec[i]), 1e-300)
            last, before = abs(avec[i] - prev[i]) / scale, abs(prev[i] - prev2[i]) / scale
            if abs(base[i]) > 1e-8 * self._diag.get("phiScale", 1.0):   # only nodes that matter
                self._diag["lastTerm"] = max(self._diag["lastTerm"], last)
                if N >= 2 and last >= before and last > 1e-14:
                    self._diag["notDecreasing"] = True
        gT = np.array([gi.value(T) for gi in g], complex)
        return avec, self.model.chain.generator @ avec + gT * avec

    def _a(self, g, gfuncs, T, a0=None):
        return self._aVector(g, gfuncs, T, a0)[0][self.regime]

    def _order(self):
        return self.order if self.order is not None else self.maxOrder

    def calculate(self, instrument, results=False):
        self._diag = dict(lastTerm=0.0, notDecreasing=False, tailFallbacks=0, tailWeight=0.0, phiScale=1.0)
        out = self._calculateOrders(instrument)
        eps = self.model.chain.meanHoldingTime()
        d = dict(epsilon=eps, orderUsed=self.orderUsed, lastTermRelative=self._diag["lastTerm"],
                 tailFallbacks=self._diag["tailFallbacks"])
        out["diagnostics"] = d
        if self._diag["notDecreasing"]:
            warnings.warn(f"the fast-switching expansion is not converging at order {self.orderUsed}: the last term "
                          f"({d['lastTermRelative']:.1e} of the value) is not smaller than the one before it; the holding "
                          f"time is {eps:.3g}. Use NumericalSwitchingEngine, or order=None to stop at the best truncation.",
                          ExpansionWarning, stacklevel=3)
        elif d["lastTermRelative"] > 1e-3:
            warnings.warn(f"the fast-switching expansion at order {self.orderUsed} is rough: the last term is "
                          f"{d['lastTermRelative']:.1e} of the value (holding time {eps:.3g}). Raise the order or use "
                          "NumericalSwitchingEngine.", ExpansionWarning, stacklevel=3)
        if self._diag["tailFallbacks"] and self._diag["tailWeight"] > 1e-8:
            warnings.warn(f"the expansion was replaced by the averaged value at {self._diag['tailFallbacks']} Fourier "
                          f"nodes where it diverged, the largest with characteristic function {self._diag['tailWeight']:.1e}; "
                          "the price may be affected. Use NumericalSwitchingEngine to check.", ExpansionWarning, stacklevel=3)
        return out if results else out["value"]

    def _calculateOrders(self, instrument):
        if self.order is not None:
            self.orderUsed = self.order; return super().calculate(instrument, results=True)
        prev, prev_inc, prev_out = None, None, None
        for n in range(0, self.maxOrder + 1):
            self.order = n
            try:
                out = super().calculate(instrument, results=True)
            finally:
                self.order = None
            v = out["value"]
            if prev is not None:
                inc = abs(v - prev) / max(abs(v), 1e-300)
                if inc <= self.tol or (prev_inc is not None and inc > prev_inc):
                    keep = out if inc <= self.tol else prev_out
                    self.orderUsed, self.lastIncrement = (n if inc <= self.tol else n - 1), min(inc, prev_inc or inc)
                    if inc > self.tol:
                        self._diag["notDecreasing"] = True
                    return keep
                prev_inc = inc
            prev, prev_out = v, out
        self.orderUsed, self.lastIncrement = self.maxOrder, prev_inc
        return prev_out


class NumericalSwitchingEngine(SwitchingEngine):
    def __init__(self, model, regime=0, nodes=96, rtol=1e-12):
        super().__init__(model, regime, nodes)
        self.rtol = rtol

    def _a(self, g, gfuncs, T, a0=None):
        return numerical_a_callable(T, self.model.chain.generator, gfuncs, rtol=self.rtol, a0=a0)[self.regime]

    def _aVector(self, g, gfuncs, T, a0=None):
        avec = np.asarray(numerical_a_callable(T, self.model.chain.generator, gfuncs, rtol=self.rtol, a0=a0), complex)
        gT = np.array([f(T) for f in gfuncs], complex)
        return avec, self.model.chain.generator @ avec + gT * avec
