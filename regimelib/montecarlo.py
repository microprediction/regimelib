"""A Monte Carlo referee that samples regime paths exactly (exponential holding times) and prices each path in
closed form, so it has no time grid: Vasicek bonds (the integrated rate is Gaussian given the path) and
Black-Scholes options (the integrated variance is known given the path). Returns the estimate; the standard error
is on the engine after calculate()."""
import math
import numpy as np
from .instruments import ZeroCouponBond, VanillaOption, rejectFeatures
from .models import SwitchingVasicek, SwitchingBlackScholesProcess
from .information import startingBelief, byBelief
from ._engine.models import stable_B, SMALL_SPEED


_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(8)


def _segment(a, lo, hi):
    """(int_lo^hi B(t) dt, int_lo^hi B(t)^2 dt) for the loading B(t) = (1 - e^{-a t}) / a: closed forms away from
    a = 0, Gauss-Legendre near it, where the closed forms cancel."""
    if a * hi >= SMALL_SPEED:
        I1 = lambda t: (t + math.expm1(-a * t) / a) / a
        I2 = lambda t: (t + 2.0 * math.expm1(-a * t) / a - math.expm1(-2.0 * a * t) / (2.0 * a)) / a ** 2
        return I1(hi) - I1(lo), I2(hi) - I2(lo)
    if a == 0.0:                                                    # B(t) = t
        d = hi - lo
        return d * (hi + lo) / 2, d * (hi * hi + hi * lo + lo * lo) / 3
    x, w = _NODES, _WEIGHTS
    ts = lo + (x + 1) * (hi - lo) / 2
    b = np.array([stable_B(a, t) for t in ts])
    return float(w @ b) * (hi - lo) / 2, float(w @ (b * b)) * (hi - lo) / 2


def _N(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


class MonteCarloSwitchingEngine:
    def __init__(self, model, regime=0, paths=100000, seed=0):
        import numbers
        if isinstance(paths, bool) or not isinstance(paths, numbers.Integral) or paths < 2:
            raise ValueError(f"paths must be an integer of at least 2, so that a standard error exists; got {paths!r}")
        self.model, self.paths, self.seed = model, int(paths), seed
        self.regime, self.belief = startingBelief(regime, model.n)
        self.standardError = None

    def _paths(self, T):
        """Yield (states, durations) for each sampled regime path on [0, T]."""
        # under a belief each starting regime gets its own stream, so that the conditional estimates are independent
        # and their standard errors combine as a root sum of squares
        stream = getattr(self, "_beliefRegime", None)
        seed = self.seed if stream is None or self.seed is None else [int(self.seed), int(stream)]
        rng = np.random.default_rng(seed); Q = self.model.chain.generator; n = Q.shape[0]
        for _ in range(self.paths):
            t, y, states, durs = 0.0, self.regime, [], []
            while t < T:
                rate = -Q[y, y]
                dwell = rng.exponential(1 / rate) if rate > 0 else math.inf
                last = dwell >= T - t                                   # decided before adding: t + d can stop an ulp short of T
                d = min(dwell, T - t); states.append(y); durs.append(d); t = T if last else t + d
                if not last:
                    p = Q[y].copy(); p[y] = 0; p /= p.sum(); y = rng.choice(n, p=p)
            yield np.array(states), np.array(durs)

    @byBelief(rss=("standardError",))
    def calculate(self, instrument):
        m, T = self.model, instrument.maturity
        if not (math.isfinite(T) and T >= 0.0):
            raise ValueError(f"the maturity must be finite and nonnegative, not {T!r}")
        vals = []
        if isinstance(instrument, ZeroCouponBond) and isinstance(m, SwitchingVasicek):
            a = m.a
            for states, durs in self._paths(T):
                # r follows dr = a (b_y - r) dt + sigma_y dW; int_0^T r ds is Gaussian given the path
                ends = np.cumsum(durs); starts = ends - durs
                # int_0^T r = r0 B(T) + sum over dwells of [a b_y int B(T - u) du] + noise of variance
                # sigma_y^2 int B(T - u)^2 du, with B the loading, evaluated stably for any a >= 0
                mean = m.r0 * stable_B(a, T); var = 0.0
                for y, s0, s1 in zip(states, starts, ends):
                    i1, i2 = _segment(a, T - s1, T - s0)
                    mean += a * m.b[y] * i1
                    var += m.sigma[y] ** 2 * i2
                final = states[-1] if len(states) else self.regime      # at T = 0 no dwell is recorded
                paid = instrument.regimeAtMaturity is None or final == instrument.regimeAtMaturity
                vals.append(math.exp(-mean + 0.5 * var) if paid else 0.0)
        elif isinstance(instrument, VanillaOption) and isinstance(m, SwitchingBlackScholesProcess):
            rejectFeatures(instrument, "the Monte Carlo referee", ("American exercise", "a barrier", "geometric averaging"),
                           "SwitchingFDEngine, or NumericalSwitchingEngine for the geometric Asian option")
            K, F, disc = instrument.strike, m.forward(T), math.exp(-m.r * T)
            if K <= 0:                                                          # S_T > 0 >= K on every path: no sampling
                self.standardError = 0.0
                if not instrument.isCall:
                    return 0.0
                return {"cash": disc * (instrument.cash or 0.0), "asset": disc * F}.get(instrument.payoffType, disc * (F - K))
            sign = 1.0 if instrument.isCall else -1.0
            for states, durs in self._paths(T):
                vols = np.asarray(m.sigma, float)[states]; top = float(vols.max()) if len(vols) else 0.0
                if top == 0.0:                                                  # the path never left a zero-volatility regime
                    vals.append(disc * float(instrument.payoffOnGrid([F])[0])); continue
                # the deviation given the path, scaled by its largest volatility so that a small one does not square to zero
                sv = top * math.sqrt(float(np.sum((vols / top) ** 2 * durs))); v = sv * sv
                if sv == 0.0:
                    vals.append(disc * float(instrument.payoffOnGrid([F])[0])); continue
                d1 = (math.log(F / K) + 0.5 * v) / sv; d2 = d1 - sv
                if instrument.payoffType == "cash":                             # pays the cash amount beyond the strike
                    vals.append(disc * instrument.cash * _N(sign * d2))
                elif instrument.payoffType == "asset":                          # pays the asset beyond the strike
                    vals.append(disc * F * _N(sign * d1))
                else:
                    call = disc * (F * _N(d1) - K * _N(d2))
                    vals.append(call if instrument.isCall else call - disc * (F - K))
        else:
            raise TypeError("Monte Carlo referee covers Vasicek bonds and Black-Scholes options")
        vals = np.asarray(vals)
        self.standardError = float(vals.std(ddof=1) / math.sqrt(len(vals)))
        return float(vals.mean())
