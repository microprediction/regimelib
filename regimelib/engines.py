"""Pricing engines. FastSwitchingEngine expands in the mean holding time to the given order; NumericalSwitchingEngine
solves the reduced system a' = (Q + diag g) a numerically and is the referee. Both take the starting regime."""
import math
import cmath
import warnings
import numpy as np
from ._engine.fastswitch import FastSwitch, numerical_a_callable, Cheb
Cheb.MAXDEG = 400      # products of fitted forcings (Heston, CIR) at higher orders and with several regimes need room
from .instruments import ZeroCouponBond, VanillaOption, ZeroCouponBondOption, CouponBond, CouponBondOption, Swaption, CapFloor, ContinuousGeometricAsianOption, CreditDefaultSwap, rejectFeatures
from .bondoptions import coupon_bond_call
from ._engine.options import zcb_call
from ._engine.models import stable_B, cir_B
from .models import SwitchingVasicek, SwitchingHullWhite, SwitchingG2
from .chain import stateIndex
from .information import startingBelief, checkInformation, regimeIsKnown, byBelief, notRevealed
from .g2options import g2_zcb_call
from .hybrid import SwitchingEquityRates


def _gauss(U, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1) * U / 2, w * U / 2


def _noDiffusion(m):
    """None when the model diffuses or jumps. Otherwise "deterministic" if the path of the state is then known, or
    "switching level" if a level still switches, in which case the state at a future date has atoms."""
    parts = [m.equity, m.rates] if isinstance(m, SwitchingEquityRates) else [m]
    answer = "deterministic"
    for p in parts:
        zero = lambda name: not hasattr(p, name) or np.all(np.asarray(getattr(p, name), float) == 0.0)
        if hasattr(p, "v0") or hasattr(p, "nu") or hasattr(p, "beta") or hasattr(p, "xi"):
            return None                                                  # stochastic variance, pure jumps, CEV: not covered here
        if not (zero("sigma") and zero("eta") and zero("jumpIntensity")):
            return None
        if isinstance(p, SwitchingVasicek) and len(set(p.b)) > 1:
            answer = "switching level"
    return answer


def _atoms(what):
    return NotImplementedError(
        f"{what} with zero volatility and a switching level: the rate at expiry takes values with positive probability, "
        "which the Fourier inversion does not resolve. Give the rate a volatility, or freeze the level.")


def _expansionOrder(value, name, allowNone=False):
    """An expansion order is a nonnegative integer (or None for the adaptive order)."""
    import numbers
    if value is None and allowNone:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{name} must be {'None or ' if allowNone else ''}a non-negative integer, got {value!r}")
    return int(value)


class IntensityWarning(UserWarning):
    """A default intensity model has produced a survival probability above one."""


class ExpansionWarning(UserWarning):
    """The fast-switching expansion may not have converged for this instrument and chain."""


def _constantValue(gi):
    """The value of a forcing term that is constant in time (an ExpSum with only the zero rate, or a Chebyshev series
    of degree zero), else None."""
    if hasattr(gi, "t"):
        if all(a == 0.0 for a in gi.t):
            return complex(gi.t.get(0.0, 0.0))
        return None
    if hasattr(gi, "s") and len(gi.s.coef) == 1:
        return complex(gi.s.coef[0])
    return None


class SwitchingEngine:
    supportsResults = True

    def __init__(self, model, regime=0, nodes=96, information="inferred"):
        self.model, self.nodes, self.information = model, nodes, checkInformation(information)
        self.regime, self.belief = startingBelief(regime, model.n)      # a regime index, or a belief over the regimes
        self._memo = {}                                       # (fingerprint, T, u) -> terminal data, shared across strikes

    def _fingerprint(self):
        """The model's numerical parameters, so a memoised terminal vector is never reused after they change."""
        items = []
        for k, v in sorted(vars(self.model).items()):
            if hasattr(v, "generator"):
                items.append((k, np.asarray(v.generator, float).tobytes()))
            elif isinstance(v, np.ndarray):
                items.append((k, v.tobytes()))
            elif isinstance(v, (int, float, complex)):
                items.append((k, v))
            elif isinstance(v, (list, tuple)):
                items.append((k, tuple(v)))
            else:
                items.append((k, id(v)))
        return hash(tuple(items))

    def _terminal(self, T, u, stoch):
        """Memoised a(T), its derivative, the prefactor and (for stochastic volatility) the Riccati D, D' at z = u - i/2."""
        key = (self._fingerprint(), T, u)
        hit = self._memo.get(key)
        if hit is None:
            m = self.model; z = u - 0.5j
            g, gfuncs, pre = m.returnForcing(z, T)
            avec, dvec = self._aVector(g, gfuncs, T)
            hit = (avec, dvec, pre(), self._riccatiD(m, z, T) if stoch else None)
            if len(self._memo) > 20000:
                self._memo.clear()
            self._memo[key] = hit
        return hit

    # a(T) for the reduced system; subclasses choose the method
    def _a(self, g, gfuncs, T, a0=None):
        raise NotImplementedError

    @byBelief()
    def calculate(self, instrument, results=False):
        m, T = self.model, instrument.maturity
        if (isinstance(instrument, (ZeroCouponBondOption, CouponBondOption, Swaption, CapFloor))
                and self.information == "inferred" and not regimeIsKnown(m)):
            raise notRevealed("the price of an option on a bond or a swap")   # one exercise boundary per regime below
        if isinstance(instrument, VanillaOption):                       # the transforms below are of the terminal value
            rejectFeatures(instrument, "the characteristic-function engine", ("American exercise", "a barrier"),
                           "SwitchingFDEngine")
        if isinstance(instrument, Swaption) and instrument.exerciseTimes is not None and (
                len(instrument.exerciseTimes) > 1 or abs(instrument.exerciseTimes[0] - T) > 1e-12):
            raise TypeError("the characteristic-function engine prices European swaptions; "
                            "use SwitchingFDEngine for Bermudan exercise")
        exact = self._exactAtTheEnds(instrument)                        # zero maturity, nonpositive strike
        if exact is not None:
            return exact if results else exact["value"]
        if isinstance(instrument, ZeroCouponBond):
            g, gfuncs, pre = m.bondForcing(T)
            a0 = None
            if instrument.regimeAtMaturity is not None:
                a0 = np.zeros(m.n); a0[stateIndex(instrument.regimeAtMaturity, m.n, "regimeAtMaturity")] = 1.0
            P = float(np.real(pre(T) * self._a(g, gfuncs, T, a0)))
            out = {"value": P}
            B = self._bondB(T)
            if B is not None:                                     # sensitivities to the state r0
                out.update(delta=-B * P, gamma=B * B * P)
        elif isinstance(instrument, ContinuousGeometricAsianOption):
            out = self._geometricAsian(instrument)
        elif isinstance(instrument, VanillaOption) and isinstance(m, SwitchingEquityRates):
            out = self._hybridVanilla(instrument)
        elif isinstance(instrument, CreditDefaultSwap):
            out = self._cds(instrument)
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
        elif isinstance(instrument, CouponBondOption):
            out = {"value": self._couponBondOption(instrument.isCall, instrument.strike, instrument.maturity, instrument.cashflows)}
        elif isinstance(instrument, Swaption):
            out = {"value": self._couponBondOption(not instrument.isPayer, instrument.notional, instrument.maturity, instrument.cashflows)}
        elif isinstance(instrument, CapFloor):
            v = 0.0
            for T0, T1 in zip(instrument.times[:-1], instrument.times[1:]):
                tau = T1 - T0; A = 1.0 + tau * instrument.strike
                if A > 0:                                                # caplet = A puts on the bond struck at 1 / A
                    o = ZeroCouponBondOption("put" if instrument.isCap else "call", 1.0 / A, T0, T1)
                    v += instrument.notional * A * self.calculate(o)
                elif instrument.isCap:                                   # 1 - A P(T0, T1) > 0 always: no optionality
                    v += instrument.notional * (self.calculate(ZeroCouponBond(T0)) - A * self.calculate(ZeroCouponBond(T1)))
            out = {"value": v}
        else:
            raise TypeError("unsupported instrument")
        return out if results else out["value"]

    def _exactAtTheEnds(self, instrument):
        """Values that need no transform: at zero maturity a claim is worth its payoff, and a strike at or below
        zero removes the optionality of a claim on a positive price. Returning them here keeps the logarithms and
        variances of the inversion formulas away from the ends of their domain."""
        m, T = self.model, instrument.maturity
        if not math.isfinite(T) or T < 0:
            raise ValueError(f"the maturity must be finite and nonnegative, got {T}")
        bond = lambda t: 1.0 if t == 0 else self.calculate(ZeroCouponBond(t))
        still = _noDiffusion(m)                                          # None, "deterministic" or "switching level"
        if isinstance(instrument, ZeroCouponBond):
            if T > 0:
                return None
            j = instrument.regimeAtMaturity                              # no time has passed: the regime is the starting one
            paid = j is None or stateIndex(j, m.n, "regimeAtMaturity") == self.regime
            return {"value": 1.0 if paid else 0.0, "delta": 0.0, "gamma": 0.0}
        if isinstance(instrument, ZeroCouponBondOption):
            K, S = instrument.strike, instrument.bondMaturity
            if T == 0:
                return {"value": max(bond(S) - K, 0.0) if instrument.isCall else max(K - bond(S), 0.0)}
            if K <= 0:
                return {"value": bond(S) - K * bond(T) if instrument.isCall else 0.0}
            if still == "deterministic":                                 # P(T, S) = P(0, S) / P(0, T) is known today
                value = bond(S) - K * bond(T)
                return {"value": max(value, 0.0) if instrument.isCall else max(-value, 0.0)}
            if still:
                raise _atoms("an option on a bond")
            return None
        if isinstance(instrument, (CouponBondOption, Swaption)):
            isCall, K = ((instrument.isCall, instrument.strike) if isinstance(instrument, CouponBondOption)
                         else (not instrument.isPayer, instrument.notional))
            if T == 0 or (K <= 0 and all(c >= 0 for _, c in instrument.cashflows)):
                value = sum(c * bond(S) for S, c in instrument.cashflows) - K * bond(T)
                if T == 0:
                    return {"value": max(value, 0.0) if isCall else max(-value, 0.0)}
                return {"value": value if isCall else 0.0}
            if still == "deterministic":
                value = sum(c * bond(S) for S, c in instrument.cashflows) - K * bond(T)
                return {"value": max(value, 0.0) if isCall else max(-value, 0.0)}
            if still:
                raise _atoms("an option on a coupon bond or a swap")
            return None
        if not isinstance(instrument, VanillaOption):
            return None
        K, S0, call = instrument.strike, m.S0, instrument.isCall
        if isinstance(instrument, ContinuousGeometricAsianOption):
            if T == 0:                                                   # the average of a single point
                return {"value": float(instrument.payoffOnGrid([S0])[0])}
            if still == "deterministic" and K > 0:                       # log S is linear in t: its average is at T / 2
                G = S0 * math.exp((m.r - m.q) * T / 2)
                return {"value": math.exp(-m.r * T) * float(instrument.payoffOnGrid([G])[0])}
            return None                                                  # a nonpositive strike is handled with the forward of the average
        if T == 0:
            value = float(instrument.payoffOnGrid([S0])[0])
            if instrument.payoffType != "vanilla":
                return {"value": value}
            side = 1.0 if call else -1.0                                 # delta is one-sided; at the strike it is the midpoint
            inside = 1.0 if side * (S0 - K) > 0 else (0.5 if S0 == K else 0.0)
            theta = side * inside * (getattr(m, "q", 0.0) * S0 - (m.r if m.r is not None else 0.0) * K)
            return {"value": value, "delta": side * inside, "gamma": 0.0, "theta": theta, "rho": 0.0}
        if K <= 0:                                                       # S_T > 0 >= K: the call is a forward, the put is void
            if isinstance(m, SwitchingEquityRates):
                return {"value": S0 * math.exp(-m.q * T) - K * self._hybridBond(T) if call else 0.0}
            dq, dr = math.exp(-m.q * T), math.exp(-m.r * T)
            if instrument.payoffType == "cash":
                return {"value": dr * instrument.cash if call else 0.0}
            if instrument.payoffType == "asset":
                return {"value": S0 * dq if call else 0.0}
            if not call:
                return {"value": 0.0, "delta": 0.0, "gamma": 0.0, "theta": 0.0, "rho": 0.0}
            return {"value": S0 * dq - K * dr, "delta": dq, "gamma": 0.0, "theta": m.q * S0 * dq - m.r * K * dr,
                    "rho": T * K * dr}
        if still == "deterministic":                                     # the terminal price is known: discounted payoff
            if isinstance(m, SwitchingEquityRates):                      # S_T = S0 e^{-qT} / P(0, T)
                P = self._hybridBond(T); value = S0 * math.exp(-m.q * T) - K * P
                return {"value": max(value, 0.0) if call else max(-value, 0.0)}
            dq, dr = math.exp(-m.q * T), math.exp(-m.r * T); F = m.forward(T)
            if instrument.payoffType != "vanilla":
                return {"value": dr * float(instrument.payoffOnGrid([F])[0])}
            side = 1.0 if call else -1.0
            inside = 1.0 if side * (F - K) > 0 else (0.5 if F == K else 0.0)
            return {"value": max(side * (S0 * dq - K * dr), 0.0), "delta": side * inside * dq, "gamma": 0.0,
                    "theta": side * inside * (m.q * S0 * dq - m.r * K * dr), "rho": side * inside * T * K * dr}
        return None

    def _hybridBond(self, T):
        """E exp(-int_0^T r) from the starting regime, under the equity-with-rates model."""
        g, gfuncs, factor = self.model.discountedForcing(0.0, T)
        return float(np.real(factor * self._aVector(g, gfuncs, T)[0][self.regime]))

    def _bondB(self, T):
        m = self.model
        if isinstance(m, SwitchingVasicek) or hasattr(m, "jumpMean"):
            return stable_B(m.a, T)
        if hasattr(m, "k") and hasattr(m, "theta") and not hasattr(m, "S0"):      # CIR
            return cir_B(m.k, m.sigma, T)
        return None

    def _bondOption(self, opt):
        m = self.model; T, S, K = opt.maturity, opt.bondMaturity, opt.strike
        if isinstance(m, SwitchingVasicek):
            call = zcb_call(T, S, K, m.r0, self.regime, m.a, m.b, m.sigma, m.chain.generator, order=self._order())
        elif isinstance(m, SwitchingHullWhite):
            # r = x + phi(t): P(T, S) = c P_x(T, S) with c = exp(-int_T^S phi), and the discount to T carries
            # exp(-int_0^T phi), so the call is exp(-int_0^T phi) c Call_x(strike K / c) under the zero-mean factor.
            a = m.a; shift = m._intShift
            e0T = m.discount(T) * math.exp(-shift(T)); c = m.discount(S) / m.discount(T) * math.exp(-(shift(S) - shift(T)))
            call = e0T * c * zcb_call(T, S, K / c, 0.0, self.regime, a, [0.0] * m.n, m.sigma, m.chain.generator, order=self._order())
        elif isinstance(m, SwitchingG2):
            e0T, c = m.deterministicDiscount(0.0, T), m.deterministicDiscount(T, S)
            call = e0T * c * g2_zcb_call(T, S, K / c, self.regime, m.a, m.b, m.sigma, m.eta, m.rho, m.chain.generator, order=self._order())
        else:
            raise TypeError("bond options are priced under SwitchingVasicek, SwitchingHullWhite or SwitchingG2")
        if opt.isCall:
            return call
        bond = lambda t: self.calculate(ZeroCouponBond(t))          # put-call parity: C - P = P(0,S) - K P(0,T)
        return call - bond(S) + K * bond(T)

    def _couponBondOption(self, isCall, K, T, cashflows):
        """Under Hull-White the deterministic curve factor scales each cash flow: c_k -> c_k D(S_k)/D(T) e^{-(shift(S_k) - shift(T))}
        in the zero-mean factor model, and the discount to expiry carries exp(-int_0^T phi)."""
        m = self.model
        if isinstance(m, SwitchingVasicek):
            call = coupon_bond_call(T, cashflows, K, m.r0, self.regime, m.a, m.b, m.sigma, m.chain.generator, order=self._order())
        elif isinstance(m, SwitchingHullWhite):
            a = m.a; shift = m._intShift
            e0T = m.discount(T) * math.exp(-shift(T))
            scaled = [(S, c * m.discount(S) / m.discount(T) * math.exp(-(shift(S) - shift(T)))) for S, c in cashflows]
            call = e0T * coupon_bond_call(T, scaled, K, 0.0, self.regime, a, [0.0] * m.n, m.sigma, m.chain.generator, order=self._order())
        else:
            raise TypeError("coupon-bond options, swaptions and caps are priced under SwitchingVasicek or SwitchingHullWhite")
        if isCall:
            return call
        bond = sum(c * self.calculate(ZeroCouponBond(S)) for S, c in cashflows)       # parity: C - P = bond - K P(0,T)
        return call - bond + K * self.calculate(ZeroCouponBond(T))

    def _cds(self, cds):
        """Survival Q(t) is the model's bond price; premium leg = s sum tau_i D(t_i) Q(t_i) (+ accrual to the mid-point
        on default), protection = (1 - R) sum D(t_mid) (Q(t_{i-1}) - Q(t_i))."""
        Q = lambda t: 1.0 if t <= 0 else self._survival(t)
        D = cds.discount; annuity = prot = 0.0; t0 = 0.0            # the annuity is the premium leg per unit spread
        for t1 in cds.times:
            tau = t1 - t0; tm = 0.5 * (t0 + t1); q0, q1 = Q(t0), Q(t1)
            annuity += tau * D(t1) * q1
            if cds.accrualOnDefault:
                annuity += 0.5 * tau * D(tm) * (q0 - q1)
            prot += (1.0 - cds.recovery) * D(tm) * (q0 - q1)
            t0 = t1
        if annuity == 0.0:
            raise ValueError("the premium annuity is zero, so the fair spread is undefined")
        prem = cds.spread * annuity
        sign = 1.0 if cds.isBuyer else -1.0
        return {"value": sign * (prot - prem), "couponLegNPV": -sign * prem, "defaultLegNPV": sign * prot,
                "fairSpread": prot / annuity, "protection": prot, "annuity": annuity}

    def _survival(self, t):
        """The model's bond read as a survival probability. A Gaussian intensity (Vasicek) can be negative, so the
        value can exceed one at high volatility or long horizons; that is reported, not hidden."""
        q = self.calculate(ZeroCouponBond(t))
        if q > 1.0 + 1e-12:
            warnings.warn(f"the survival probability to t = {t:g} is {q:.6g}, above one: a Gaussian (Vasicek) intensity is "
                          "negative with positive probability, and at these parameters that matters. Use a "
                          "Cox-Ingersoll-Ross intensity or a lower volatility.", IntensityWarning, stacklevel=4)
        return q

    def _hybridVanilla(self, opt):
        """Lewis's formula with the discounted characteristic function (regimelib.hybrid); the put by parity with the
        switching bond from the same starting regime."""
        m, T, K = self.model, opt.maturity, opt.strike
        if opt.payoffType != "vanilla":
            raise TypeError("digitals are not priced under the hybrid model")
        def psi(w):
            g, gfuncs, factor = m.discountedForcing(w, T)
            return factor * self._aVector(g, gfuncs, T)[0][self.regime]
        U = 8.0
        while abs(psi(U - 0.5j)) > 1e-14 * K ** 0.5 and U < 1e4:
            U *= 2
        k = math.log(K); us, ws = _gauss(U, self._nodeCount(U, math.log(m.S0 / K)))
        I0 = sum(w * (cmath.exp(-1j * u * k) * psi(u - 0.5j)).real / (u * u + 0.25) for u, w in zip(us, ws))
        call = m.S0 * math.exp(-m.q * T) - math.sqrt(K) / math.pi * I0
        if opt.isCall:
            return {"value": call}
        bond = psi(0.0).real                                                    # E exp(-int r) from the starting regime
        return {"value": call - m.S0 * math.exp(-m.q * T) + K * bond}

    def _geometricAsian(self, opt):
        """Lewis's formula on the geometric average G = S0 exp(Y): the forward is F_G = S0 phi_Y(-i) and the
        martingale characteristic function is phi_Y(z) exp(-i z log(F_G / S0))."""
        m, T, K = self.model, opt.maturity, opt.strike
        if not hasattr(m, "averageForcing"):
            raise TypeError("the geometric Asian option needs a model with averageForcing (Black-Scholes)")
        def phiY(z):
            g, gfuncs = m.averageForcing(z, T)
            return self._aVector(g, gfuncs, T)[0][self.regime]
        FG = m.S0 * phiY(-1j).real
        if K <= 0:                                                       # the average is positive: a forward, or nothing
            return {"value": math.exp(-m.r * T) * (FG - K) if opt.isCall else 0.0}
        k = math.log(FG / K); lf = math.log(FG / m.S0)
        U = self._frequencyLimit(T, k); us, ws = _gauss(U, self._nodeCount(U, k))
        I0 = 0.0
        for u, w in zip(us, ws):
            z = u - 0.5j
            e = cmath.exp(1j * u * k) * phiY(z) * cmath.exp(-1j * z * lf)
            I0 += w * e.real / (u * u + 0.25)
        disc = math.exp(-m.r * T)
        call = disc * (FG - math.sqrt(FG * K) / math.pi * I0)
        return {"value": call if opt.isCall else call - disc * (FG - K)}

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
            avec, dvec, prev, ricc = self._terminal(T, u, stoch)
            a = avec[self.regime]; phi = prev * a
            e = cmath.exp(1j * u * k) * phi
            I0 += w * e.real / (u * u + 0.25)
            I1 += w * ((0.5 + 1j * u) * e).real / (u * u + 0.25)
            I2 += w * e.real
            if stoch:
                D, dD = ricc
                Iv += w * (e * D).real / (u * u + 0.25)
                dphi = prev * (dD * m.v0 * a + dvec[self.regime])
            else:
                dphi = prev * dvec[self.regime]
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
        if m.sigma == 0.0:                                               # the variance is deterministic: D' = -kappa D - (u^2 + iu)/2
            D = -0.5 * (u * u + 1j * u) * (1 - cmath.exp(-m.kappa * T)) / m.kappa
            return D, -m.kappa * D - 0.5 * (u * u + 1j * u)
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
        """Nodes grow with the oscillation e^{iuk}; the factor is rounded up to a power of two so that strikes of
        similar moneyness share one node set and the memoised terminal vectors."""
        factor = 2 ** math.ceil(math.log2(1 + abs(k))) if k else 1
        return int(min(4000, max(self.nodes, 2 * U * factor)))

class FastSwitchingEngine(SwitchingEngine):
    """order: an integer, or None to add terms until successive orders agree to `tol` (relative) or the next term
    stops shrinking, the best truncation of an asymptotic series. `maxOrder` bounds the search. After calculate(),
    `orderUsed` and `lastIncrement` (relative size of the last term kept) are set."""
    def __init__(self, model, order=4, regime=0, nodes=96, tol=1e-10, maxOrder=12, rtol=1e-12, information="inferred"):
        super().__init__(model, regime, nodes, information)
        order, maxOrder = (_expansionOrder(order, "order", allowNone=True), _expansionOrder(maxOrder, "maxOrder"))
        self.order, self.tol, self.maxOrder, self.rtol = order, tol, maxOrder, rtol
        self.orderUsed = self.lastIncrement = None

    def _aVector(self, g, gfuncs, T, a0=None):
        """a(T) over all regimes through the engine's order, with the size of the last two terms recorded for the
        convergence diagnostics. The series is asymptotic in the holding time times the forcing, so at Fourier nodes
        where the forcing is large it diverges (non-finite, far from the averaged value, or a last term no smaller than
        the one before): there the reduced system is solved numerically instead and `numericalNodes` counts them."""
        N = self._order()
        Q = self.model.chain.generator
        if self.model.n == 1:                                            # nothing to expand in: a = a0 exp(int g)
            avec = _numericalAVector(Q, g, gfuncs, T, self.rtol, a0)
            return avec, Q @ avec + np.array([gi.value(T) for gi in g], complex) * avec
        fs = FastSwitch(Q, g, order=N, a0=a0)
        base = np.asarray(fs.a(T, 0), complex)
        with np.errstate(all="ignore"):
            try:
                avec = np.asarray(fs.a(T, N), complex)
                prev = np.asarray(fs.a(T, N - 1), complex) if N >= 1 else base
                prev2 = np.asarray(fs.a(T, N - 2), complex) if N >= 2 else prev
            except (OverflowError, FloatingPointError, ValueError):
                avec = np.full(self.model.n, np.nan, complex); prev = prev2 = base
        i = self.regime
        finite = np.all(np.isfinite(avec)) and not np.any(np.abs(avec) > 1e3 * np.maximum(np.abs(base), 1e-300))
        matters = abs(base[i]) > 1e-8 * self._diag.get("phiScale", 1.0)
        if finite:
            scale = max(abs(avec[i]), 1e-300)
            last, before = abs(avec[i] - prev[i]) / scale, abs(prev[i] - prev2[i]) / scale
            diverging = N >= 2 and last >= before and last > 1e-12
        else:
            last, diverging = math.inf, True
        if diverging:
            self._diag["numericalNodes"] += 1
            if matters:
                self._diag["numericalWeight"] = max(self._diag["numericalWeight"], float(abs(base[i])))
            avec = _numericalAVector(Q, g, gfuncs, T, self.rtol, a0)
        elif matters:
            self._diag["lastTerm"] = max(self._diag["lastTerm"], last)
        gT = np.array([gi.value(T) for gi in g], complex)
        return avec, Q @ avec + gT * avec

    def _a(self, g, gfuncs, T, a0=None):
        return self._aVector(g, gfuncs, T, a0)[0][self.regime]

    def _order(self):
        if self.model.n == 1:
            return None                                                  # the inverters solve numerically: there is no series
        return self.order if self.order is not None else self.maxOrder

    @byBelief(worst=("orderUsed", "lastIncrement"))
    def calculate(self, instrument, results=False):
        self._diag = dict(lastTerm=0.0, notDecreasing=False, numericalNodes=0, numericalWeight=0.0, phiScale=1.0)
        out = self._calculateOrders(instrument)
        eps = self.model.chain.meanHoldingTime()
        d = dict(epsilon=eps, orderUsed=self.orderUsed, lastTermRelative=self._diag["lastTerm"],
                 numericalNodes=self._diag["numericalNodes"])
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
        if self._diag["numericalNodes"] and self._diag["numericalWeight"] > 1e-8:
            warnings.warn(f"the expansion diverged at {self._diag['numericalNodes']} Fourier nodes (large forcing at high "
                          f"frequency, the largest with characteristic function {self._diag['numericalWeight']:.1e}), where "
                          "the reduced system was solved numerically instead.", ExpansionWarning, stacklevel=3)
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


def _numericalAVector(Q, g, gfuncs, T, rtol=1e-12, a0=None):
    """a(T) from the reduced system without expansion: the matrix exponential for constant forcing, else the ODE."""
    const = [_constantValue(gi) for gi in g]
    if all(c is not None for c in const):
        from scipy.linalg import expm
        gT = np.array(const, complex)
        a0 = np.ones(len(g), complex) if a0 is None else np.asarray(a0, complex)
        return expm((Q + np.diag(gT)) * T) @ a0
    return np.asarray(numerical_a_callable(T, Q, gfuncs, rtol=rtol, a0=a0), complex)


class NumericalSwitchingEngine(SwitchingEngine):
    def __init__(self, model, regime=0, nodes=96, rtol=1e-12, information="inferred"):
        super().__init__(model, regime, nodes, information)
        self.rtol = rtol

    def _a(self, g, gfuncs, T, a0=None):
        return numerical_a_callable(T, self.model.chain.generator, gfuncs, rtol=self.rtol, a0=a0)[self.regime]

    def _aVector(self, g, gfuncs, T, a0=None):
        Q = self.model.chain.generator
        avec = _numericalAVector(Q, g, gfuncs, T, self.rtol, a0)
        gT = np.array([f(T) for f in gfuncs], complex)
        return avec, Q @ avec + gT * avec
        avec = np.asarray(numerical_a_callable(T, Q, gfuncs, rtol=self.rtol, a0=a0), complex)
        gT = np.array([f(T) for f in gfuncs], complex)
        return avec, Q @ avec + gT * avec
