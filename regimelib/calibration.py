"""Calibration in QuantLib's mold: helpers carry a market quote and compute the model value with an engine on the
model; `calibrate` fits chosen model parameters by least squares on the helpers' calibration errors
(QuantLib: HestonModelHelper, CalibratedModel.calibrate(helpers, method, endCriteria))."""
import math
import numpy as np
from scipy.optimize import least_squares, brentq
from .instruments import VanillaOption
from .engines import NumericalSwitchingEngine


class VolatilityHelper:
    """A European option quoted in Black volatility: `maturity` (years or QuantLib Date), `strike`, `volatility`,
    `kind` "call" or "put". Model prices come from the engine set by `setPricingEngine` (or by `calibrate`)."""
    def __init__(self, maturity, strike, volatility, kind="call", dayCounter=None):
        self.option = VanillaOption((kind, strike), maturity=maturity, dayCounter=dayCounter)
        self.volatility = float(volatility); self._engine = None

    def setPricingEngine(self, engine):
        self._engine = engine; self.option.setPricingEngine(engine)

    def _black(self, vol):
        m = self._engine.model; T, K = self.option.maturity, self.option.strike
        F, disc = m.forward(T), math.exp(-m.r * T); sv = vol * math.sqrt(T)
        N = lambda x: 0.5 * math.erfc(-x / math.sqrt(2))
        d1 = (math.log(F / K) + 0.5 * sv * sv) / sv; c = disc * (F * N(d1) - K * N(d1 - sv))
        return c if self.option.isCall else c - disc * (F - K)

    def marketValue(self):
        return self._black(self.volatility)

    def modelValue(self):
        return self.option.NPV()

    def impliedVolatility(self):
        return self.option.impliedVolatility()

    def calibrationError(self):
        """Relative price error, as QuantLib's RelativePriceError."""
        return self.modelValue() / self.marketValue() - 1.0

    def volatilityError(self):
        return self.impliedVolatility() - self.volatility


def _get(model, name):
    if name == "chain":
        Q = model.chain.generator; n = Q.shape[0]
        return np.array([Q[i, j] for i in range(n) for j in range(n) if i != j])
    return np.atleast_1d(np.asarray(getattr(model, name), float)).copy()


def _set(model, name, values):
    if name == "chain":
        from .chain import RegimeChain
        n = model.chain.generator.shape[0]; Q = np.zeros((n, n)); k = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    Q[i, j] = values[k]; k += 1
            Q[i, i] = -Q[i].sum()
        model.chain = RegimeChain(Q)
    else:
        setattr(model, name, values if len(values) > 1 else float(values[0]))


def calibrate(model, helpers, parameters, engine=None, bounds=None, useVolatilityError=False, **kwargs):
    """Fit `parameters` (model attribute names, per-regime arrays, and/or "chain" for the off-diagonal rates) to the
    helpers by least squares on their calibration errors. `engine` is a factory model -> engine (default: the
    numerical engine, whose terminal vectors are shared across strikes at each maturity). Returns the scipy result;
    the model is left at the fitted values."""
    engine = engine or (lambda m: NumericalSwitchingEngine(m))
    x0 = np.concatenate([_get(model, p) for p in parameters]); sizes = [len(_get(model, p)) for p in parameters]
    lo = np.full(len(x0), 1e-8); hi = np.full(len(x0), np.inf)
    if bounds is not None:
        lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)

    def apply(x):
        k = 0
        for p, s in zip(parameters, sizes):
            _set(model, p, x[k:k + s]); k += s
        eng = engine(model)
        for h in helpers:
            h.setPricingEngine(eng)

    def residuals(x):
        apply(x)
        return np.array([h.volatilityError() if useVolatilityError else h.calibrationError() for h in helpers])
    res = least_squares(residuals, x0, bounds=(lo, hi), **kwargs)
    apply(res.x)
    return res
