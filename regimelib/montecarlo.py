"""A Monte Carlo referee that samples regime paths exactly (exponential holding times) and prices each path in
closed form, so it has no time grid: Vasicek bonds (the integrated rate is Gaussian given the path) and
Black-Scholes options (the integrated variance is known given the path). Returns the estimate; the standard error
is on the engine after calculate()."""
import math
import numpy as np
from .instruments import ZeroCouponBond, VanillaOption, rejectFeatures
from .models import SwitchingVasicek, SwitchingBlackScholesProcess
from .information import startingBelief, byBelief


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
        rng = np.random.default_rng(self.seed); Q = self.model.chain.generator; n = Q.shape[0]
        for _ in range(self.paths):
            t, y, states, durs = 0.0, self.regime, [], []
            while t < T:
                rate = -Q[y, y]
                dwell = rng.exponential(1 / rate) if rate > 0 else math.inf
                d = min(dwell, T - t); states.append(y); durs.append(d); t += d
                if t < T:
                    p = Q[y].copy(); p[y] = 0; p /= p.sum(); y = rng.choice(n, p=p)
            yield np.array(states), np.array(durs)

    @byBelief(rss=("standardError",))
    def calculate(self, instrument):
        m, T = self.model, instrument.maturity
        vals = []
        if isinstance(instrument, ZeroCouponBond) and isinstance(m, SwitchingVasicek):
            a = m.a
            for states, durs in self._paths(T):
                # r follows dr = a (b_y - r) dt + sigma_y dW; int_0^T r ds is Gaussian given the path
                ends = np.cumsum(durs); starts = ends - durs
                if a == 0.0:                                                    # no reversion: r = r0 + int sigma dW
                    var = sum(m.sigma[y] ** 2 * ((T - s0) ** 3 - (T - s1) ** 3) / 3 for y, s0, s1 in zip(states, starts, ends))
                    final = states[-1] if len(states) else self.regime
                    paid = instrument.regimeAtMaturity is None or final == instrument.regimeAtMaturity
                    vals.append(math.exp(-m.r0 * T + 0.5 * var) if paid else 0.0); continue
                mean = m.r0 * (1 - math.exp(-a * T)) / a; var = 0.0
                for y, s0, s1 in zip(states, starts, ends):
                    # contribution of the drift toward b_y over [s0, s1] to int_0^T r, and of the noise there
                    mean += m.b[y] * ((s1 - s0) - (math.exp(-a * (T - s1)) - math.exp(-a * (T - s0))) / a)
                    # int over [s0,s1] of ((1 - e^{-a (T - u)}) / a)^2 sigma_y^2 du
                    f = lambda u: (u - 2 * math.exp(-a * (T - u)) / a + math.exp(-2 * a * (T - u)) / (2 * a)) / (a * a)
                    var += m.sigma[y] ** 2 * (f(s1) - f(s0))
                final = states[-1] if len(states) else self.regime      # at T = 0 no dwell is recorded
                paid = instrument.regimeAtMaturity is None or final == instrument.regimeAtMaturity
                vals.append(math.exp(-mean + 0.5 * var) if paid else 0.0)
        elif isinstance(instrument, VanillaOption) and isinstance(m, SwitchingBlackScholesProcess):
            rejectFeatures(instrument, "the Monte Carlo referee", ("American exercise", "a barrier", "geometric averaging"),
                           "SwitchingFDEngine, or NumericalSwitchingEngine for the geometric Asian option")
            K, F, disc = instrument.strike, m.forward(T), math.exp(-m.r * T)
            sign = 1.0 if instrument.isCall else -1.0
            for states, durs in self._paths(T):
                v = float(np.sum(np.asarray(m.sigma)[states] ** 2 * durs))      # integrated variance given the path
                if v == 0.0:                                                    # the path never left a zero-volatility regime
                    vals.append(disc * float(instrument.payoffOnGrid([F])[0])); continue
                sv = math.sqrt(v); d1 = (math.log(F / K) + 0.5 * v) / sv; d2 = d1 - sv
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
