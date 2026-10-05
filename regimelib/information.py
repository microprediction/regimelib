"""What is known about the regime.

An engine starts either from a regime, `regime=0`, or from a belief about it, `regime=[0.3, 0.7]`. For a payoff that
depends only on what is observed (a bond, an equity option, a barrier, an average, a default) the price from a belief
is the average over regimes of the price from each: pricing is linear when nobody has to act on the regime.

It stops being linear when someone acts on it: the holder of an early-exercise right, or the market that sets the
price of the bond an option is written on. Then it matters whether the regime is `observed` or `inferred` from the
path. The two agree when the path reveals the regime, which it does at once, by its quadratic variation, when a
diffusion coefficient differs between every pair of regimes. They differ when only a drift level switches."""
import functools
import numbers
import numpy as np
from .chain import stateIndex


def startingBelief(regime, n):
    """(index, None) for a known starting regime, (None, probabilities) for a belief over the n regimes."""
    if isinstance(regime, numbers.Integral) and not isinstance(regime, bool):
        return stateIndex(regime, n), None
    if np.ndim(regime) != 1:
        raise ValueError(f"regime must be a regime index or a probability for each of the {n} regimes, got {regime!r}")
    p = np.asarray(regime, float)
    if len(p) != n:
        raise ValueError(f"a belief needs one probability per regime ({n}); got {len(p)}")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or abs(p.sum() - 1.0) > 1e-9:
        raise ValueError(f"a belief must be nonnegative and sum to one; got {list(p)}")
    return None, p / p.sum()


def checkInformation(information):
    if information not in ("observed", "inferred"):
        raise ValueError(f'information must be "observed" or "inferred", got {information!r}')
    return information


def _parts(model):
    """The models a signature is read from: the model itself, or the equity and the rates of a hybrid."""
    return [model.equity, model.rates] if hasattr(model, "equity") and hasattr(model, "rates") else [model]


def _signature(model, which):
    """Per regime, the tuple of the model's parameters named by `which` (None when the model does not declare them)."""
    rows = [[] for _ in range(model.n)]
    for part in _parts(model):
        names = getattr(part, which, None)
        if names is None:
            return None
        for name in names:
            values = np.broadcast_to(np.asarray(getattr(part, name), float), (model.n,))
            for i in range(model.n):
                rows[i].append(float(values[i]))
    return [tuple(r) for r in rows]


def identical(model):
    """Every regime has the same parameters, so there is nothing to know."""
    sig = _signature(model, "_switching")
    return sig is not None and all(s == sig[0] for s in sig)


def revealed(model):
    """The observed path identifies the regime at once: the diffusion coefficients differ between every two regimes."""
    sig = _signature(model, "_diffusion")
    return sig is not None and len(set(sig)) == model.n and all(len(s) > 0 for s in sig)


def regimeIsKnown(model):
    return model.n == 1 or identical(model) or revealed(model)


def byBelief(mean=(), worst=(), rss=()):
    """Decorate an engine's calculate so that it prices from a belief: the weighted average over starting regimes of
    the price from each. `mean`, `worst` and `rss` name engine attributes set by a calculation that are combined the
    same way, by the largest value, or as a root sum of squares (a standard error)."""
    def decorate(calculate):
        @functools.wraps(calculate)
        def wrapper(self, instrument, *args, **kwargs):
            belief = getattr(self, "belief", None)
            if belief is None:
                return calculate(self, instrument, *args, **kwargs)
            parts, attrs, saved = [], [], self.regime
            self.belief = None
            try:
                for i, w in enumerate(belief):
                    if w > 0.0:
                        self.regime = self._beliefRegime = i
                        parts.append((w, calculate(self, instrument, *args, **kwargs)))
                        attrs.append({name: getattr(self, name, None) for name in tuple(mean) + tuple(worst) + tuple(rss)})
            finally:
                self.regime, self.belief, self._beliefRegime = saved, belief, None
            for name in mean:
                if all(a[name] is not None for a in attrs):
                    setattr(self, name, sum(w * a[name] for (w, _), a in zip(parts, attrs)))
            for name in worst:
                if all(a[name] is not None for a in attrs):
                    setattr(self, name, max(a[name] for a in attrs))
            for name in rss:
                if all(a[name] is not None for a in attrs):
                    setattr(self, name, float(np.sqrt(sum((w * a[name]) ** 2 for (w, _), a in zip(parts, attrs)))))
            return _average(parts)
        return wrapper
    return decorate


def _average(parts):
    first = parts[0][1]
    if not isinstance(first, dict):
        return sum(w * v for w, v in parts)
    out = {}
    for key, value in first.items():
        if isinstance(value, numbers.Number) and not isinstance(value, bool):
            out[key] = sum(w * r[key] for w, r in parts)
        elif isinstance(value, dict):                                  # diagnostics: the worst case over the regimes
            out[key] = {k: (sum if k == "numericalNodes" else max)(r[key][k] for _, r in parts) for k in value}
        else:
            out[key] = value
    if "protection" in out and "annuity" in out:                       # a ratio of two averages, not an average of ratios
        out["fairSpread"] = out["protection"] / out["annuity"]
    return out


def notRevealed(what):
    return NotImplementedError(
        f"{what} depends on what is known about the regime, and with these parameters the path does not reveal it "
        "(no diffusion coefficient differs between every two regimes). For the price when the regime is inferred, use "
        'SwitchingFDEngine with a two-regime SwitchingVasicek or SwitchingCoxIngersollRoss model; information="observed" '
        "gives the price when the regime is observed, which is an upper bound.")
