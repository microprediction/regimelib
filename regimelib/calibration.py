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
        if not math.isfinite(self.volatility) or self.volatility < 0:
            raise ValueError(f"volatility must be finite and nonnegative, got {self.volatility}")
        T, K = self.option.maturity, self.option.strike               # elsewhere the Black price has no vega to fit
        if not (math.isfinite(T) and T > 0):
            raise ValueError(f"a Black-volatility helper requires a positive finite maturity, got {T}")
        if not (math.isfinite(K) and K > 0):
            raise ValueError(f"a Black-volatility helper requires a positive finite strike, got {K}")

    def setPricingEngine(self, engine):
        self._engine = engine; self.option.setPricingEngine(engine)

    def _black(self, vol):
        m = self._engine.model; T, K = self.option.maturity, self.option.strike
        F, disc = m.forward(T), math.exp(-m.r * T); sv = vol * math.sqrt(T)
        if sv == 0.0:                                                    # no time or no volatility: intrinsic on the forward
            return disc * max(F - K, 0.0) if self.option.isCall else disc * max(K - F, 0.0)
        N = lambda x: 0.5 * math.erfc(-x / math.sqrt(2))
        d1 = (math.log(F / K) + 0.5 * sv * sv) / sv; d2 = d1 - sv
        if self.option.isCall:
            return disc * (F * N(d1) - K * N(d2))
        return disc * (K * N(-d2) - F * N(-d1))                          # directly: parity would cancel a small put

    def marketValue(self):
        return self._black(self.volatility)

    def modelValue(self):
        return self.option.NPV()

    def impliedVolatility(self):
        return self.option.impliedVolatility(maxVol=None)        # no ceiling: a quote of any size can be matched

    def calibrationError(self):
        """Relative price error, as QuantLib's RelativePriceError; the absolute error when the market value is zero."""
        market = self.marketValue()
        return self.modelValue() - market if market == 0.0 else self.modelValue() / market - 1.0

    def volatilityError(self):
        return self.impliedVolatility() - self.volatility


# The domain of a parameter by its attribute name. A name that is not listed is unbounded: rates, dividend yields, mean
# levels of Gaussian rates and log jump means are any real number, and a wrong guess here would silently constrain a fit.
_NONNEGATIVE = ("sigma", "eta", "xi", "v0", "jumpIntensity", "logJumpVol", "jumpMean")
_SPEEDS = ("kappa", "a", "k")                                    # reversion speeds: zero is a supported limit
_POSITIVE = ("nu", "S0")                                         # the gamma variance rate, the spot
_TINY = 1e-8
_WALL = 1e6                                                      # the residual at parameters that are not a model


def defaultBounds(model, name):
    """(lower, upper) for one calibrated parameter of `model`. Two names mean different things in different models:
    `theta` is a variance or square-root level (nonnegative) except in variance gamma, where it is a signed drift,
    and `b` is a signed mean level except in G2++, where it is a reversion speed."""
    from .models import SwitchingG2, SwitchingVarianceGammaProcess    # by class, so that a subclass keeps its domains
    if name == "chain":
        return 0.0, np.inf                                   # switching rates; zero closes a transition
    if name == "rho":
        return -1.0, 1.0
    if name == "theta":
        return (-np.inf, np.inf) if isinstance(model, SwitchingVarianceGammaProcess) else (0.0, np.inf)
    if name == "b":
        return (0.0, np.inf) if isinstance(model, SwitchingG2) else (-np.inf, np.inf)
    if name in _NONNEGATIVE or name in _SPEEDS:
        return 0.0, np.inf
    if name in _POSITIVE:
        return _TINY, np.inf
    return -np.inf, np.inf


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
    else:                                                    # a scalar stays a scalar, a per-regime list stays a list (a copy)
        scalar = np.isscalar(getattr(model, name))
        setattr(model, name, float(values[0]) if scalar else [float(v) for v in values])


def calibrate(model, helpers, parameters, engine=None, bounds=None, useVolatilityError=False, **kwargs):
    """Fit `parameters` (model attribute names, per-regime arrays, and/or "chain" for the off-diagonal rates) to the
    helpers by least squares on their calibration errors. `engine` is a factory model -> engine (default: the
    numerical engine, whose terminal vectors are shared across strikes at each maturity). Returns the scipy result;
    the model is left at the fitted values. Parameters outside the model's own domain (beyond the bounds, which are a
    box) are given a large residual instead of a price."""
    if kwargs.get("workers") is not None:
        # every residual writes the candidate into this one model and its helpers, so evaluations cannot overlap
        raise ValueError("calibrate evaluates its residuals on the one model it is given, in turn: parallel finite "
                         "differences (workers) would price one candidate through another's parameters. Omit workers.")
    engine = engine or (lambda m: NumericalSwitchingEngine(m))
    helpers, parameters = tuple(helpers), tuple(parameters)    # read once: either may be a generator
    if not helpers:
        raise ValueError("calibration requires at least one helper")
    if not parameters:
        raise ValueError("calibration requires at least one parameter")
    repeated = sorted({p for p in parameters if parameters.count(p) > 1})
    if repeated:
        raise ValueError(f"each parameter may be named once; repeated: {', '.join(repeated)}")
    x0 = np.concatenate([_get(model, p) for p in parameters]); sizes = [len(_get(model, p)) for p in parameters]
    if bounds is not None:
        lo, hi = (np.broadcast_to(np.asarray(b, float), x0.shape).copy() for b in bounds)
    else:                                                   # each parameter's own domain, not one bound for all
        domains = [defaultBounds(model, p) for p in parameters]
        lo = np.concatenate([np.full(s, d[0]) for s, d in zip(sizes, domains)])
        hi = np.concatenate([np.full(s, d[1]) for s, d in zip(sizes, domains)])
    k = 0
    for p, s in zip(parameters, sizes):
        outside = [float(v) for v, l, h in zip(x0[k:k + s], lo[k:k + s], hi[k:k + s]) if not l <= v <= h]
        if outside:
            raise ValueError(f"the starting value of {p} is outside its bounds [{lo[k]}, {hi[k]}]: {outside}")
        k += s

    def write(x):
        k = 0
        for p, s in zip(parameters, sizes):
            _set(model, p, x[k:k + s]); k += s

    def apply(x):
        write(x)
        eng = engine(model)
        for h in helpers:
            h.setPricingEngine(eng)

    def admissible():
        """Bounds are a box; a model may also tie its parameters together (variance gamma's finite E[S])."""
        try:
            getattr(model, "checkParameters", lambda: None)()
        except ValueError:
            return False
        return True

    def residuals(x):
        write(x)
        if not admissible():                                # not a model: a wall, so the fit stays where prices exist
            return np.full(len(helpers), _WALL)
        apply(x)
        return np.array([h.volatilityError() if useVolatilityError else h.calibrationError() for h in helpers])
    res = least_squares(residuals, x0, bounds=(lo, hi), **kwargs)
    apply(res.x)
    return res
