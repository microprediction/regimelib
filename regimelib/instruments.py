"""Instruments in QuantLib's mold: build, setPricingEngine, NPV. Times are in years, or QuantLib Dates measured from
the QuantLib evaluation date with Actual/365 (or a day counter passed as `dayCounter`)."""
import math


def _years(t, dayCounter=None):
    if hasattr(t, "serialNumber"):                      # a QuantLib Date
        import QuantLib as ql
        dc = dayCounter or ql.Actual365Fixed()
        return float(dc.yearFraction(ql.Settings.instance().evaluationDate, t))
    return float(t)


class Instrument:
    """QuantLib's mold: setPricingEngine, then NPV() and the greeks the engine provides (delta(), gamma(), theta(),
    vega(), rho()); a greek the engine does not compute raises, as QuantLib's "not provided" does."""
    def __init__(self):
        self._engine = None; self._results = None

    def setPricingEngine(self, engine):
        self._engine = engine; self._results = None

    def _calculate(self):
        if self._engine is None:
            raise RuntimeError("no pricing engine set")
        self._results = self._engine.calculate(self, results=True) if hasattr(self._engine, "supportsResults") \
            else {"value": self._engine.calculate(self)}
        return self._results

    def NPV(self):
        return self._calculate()["value"]

    def _result(self, name):
        r = self._results if self._results is not None else self._calculate()
        if name not in r:
            raise RuntimeError(f"{name} not provided by the engine")
        return r[name]

    def impliedVolatility(self, price=None, accuracy=1e-10, maxEvaluations=200, minVol=1e-4, maxVol=4.0):
        """Black volatility that reproduces the price (the instrument's NPV unless a price is given), for a vanilla
        payoff, using the model's forward and discount; as QuantLib's VanillaOption.impliedVolatility."""
        import math
        from scipy.optimize import brentq
        if getattr(self, "payoffType", "vanilla") != "vanilla":
            raise RuntimeError("implied volatility is defined for plain vanilla payoffs")
        m = self._engine.model; T, K = self.maturity, self.strike
        F, disc = m.forward(T), math.exp(-m.r * T)
        target = self.NPV() if price is None else price
        N = lambda x: 0.5 * math.erfc(-x / math.sqrt(2))
        def black(v):
            sv = v * math.sqrt(T); d1 = (math.log(F / K) + 0.5 * sv * sv) / sv; d2 = d1 - sv
            c = disc * (F * N(d1) - K * N(d2))
            return c if self.isCall else c - disc * (F - K)
        return brentq(lambda v: black(v) - target, minVol, maxVol, xtol=accuracy, maxiter=maxEvaluations)

    def delta(self): return self._result("delta")
    def gamma(self): return self._result("gamma")
    def theta(self): return self._result("theta")
    def vega(self): return self._result("vega")
    def rho(self): return self._result("rho")


class ZeroCouponBond(Instrument):
    """Unit face value paid at maturity (years or a QuantLib Date). With regimeAtMaturity = j the face value is paid
    only if the regime at maturity is j, which prices the memory of the regime; the sum over j is the plain bond."""
    def __init__(self, maturity, dayCounter=None, regimeAtMaturity=None):
        super().__init__()
        self.maturity = _years(maturity, dayCounter)
        self.regimeAtMaturity = regimeAtMaturity


class VanillaOption(Instrument):
    """European option. Accepts QuantLib payoffs (PlainVanillaPayoff, CashOrNothingPayoff, AssetOrNothingPayoff) and
    EuropeanExercise objects, or a tuple ("call"|"put", strike) with the maturity in years or as a QuantLib Date.
    Digitals: ("cash", "call"|"put", strike, cash) or ("asset", "call"|"put", strike)."""
    def __init__(self, payoff, exercise=None, maturity=None, dayCounter=None):
        super().__init__()
        if maturity is None and exercise is not None and hasattr(exercise, "lastDate"):
            maturity = exercise.lastDate()
        self.payoffType, self.cash = "vanilla", None
        if hasattr(payoff, "strike"):                         # QuantLib payoff
            self.strike = float(payoff.strike())
            ot = payoff.optionType()
            self.isCall = ot == 1 or str(ot).lower().endswith("call")
            name = type(payoff).__name__
            if "CashOrNothing" in name:                       # QuantLib-Python exposes only the callable payoff
                self.payoffType = "cash"
                self.cash = float(payoff(2.0 * self.strike + 1.0) if self.isCall else payoff(0.0))
            elif "AssetOrNothing" in name:
                self.payoffType = "asset"
        elif payoff[0] in ("cash", "asset"):
            self.payoffType = payoff[0]; kind, self.strike = payoff[1], float(payoff[2])
            self.cash = float(payoff[3]) if payoff[0] == "cash" else None
            self.isCall = str(kind).lower() == "call"
        else:
            kind, self.strike = payoff
            self.isCall = str(kind).lower() == "call"
        if maturity is None:
            raise ValueError("give the maturity in years or as a QuantLib Date")
        self.maturity = _years(maturity, dayCounter)


class CouponBond(Instrument):
    """Fixed cash flows: a list of (time, amount), or QuantLib-style (faceAmount, couponRate, times) with the last time
    carrying the face. Priced as the sum of zero-coupon bonds; delta and gamma in r0 add up the same way."""
    def __init__(self, cashflows=None, faceAmount=None, couponRate=None, times=None, dayCounter=None):
        super().__init__()
        if cashflows is None:
            ts = [_years(t, dayCounter) for t in times]
            cashflows = [(t, faceAmount * couponRate * (t - (ts[i - 1] if i else 0.0))) for i, t in enumerate(ts)]
            cashflows[-1] = (ts[-1], cashflows[-1][1] + faceAmount)
        self.cashflows = [(_years(t, dayCounter), float(c)) for t, c in cashflows]
        self.maturity = max(t for t, _ in self.cashflows)


class ZeroCouponBondOption(Instrument):
    """European call or put, expiring at `maturity`, on the unit zero-coupon bond maturing at `bondMaturity`
    (QuantLib: Vasicek.discountBondOption(type, strike, maturity, bondMaturity))."""
    def __init__(self, kind, strike, maturity, bondMaturity):
        super().__init__()
        self.isCall = str(kind).lower() == "call"
        self.strike, self.maturity, self.bondMaturity = float(strike), _years(maturity), _years(bondMaturity)
        if self.bondMaturity <= self.maturity:
            raise ValueError("the bond must mature after the option")
