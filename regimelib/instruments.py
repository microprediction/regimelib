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
    def __init__(self):
        self._engine = None

    def setPricingEngine(self, engine):
        self._engine = engine

    def NPV(self):
        if self._engine is None:
            raise RuntimeError("no pricing engine set")
        return self._engine.calculate(self)


class ZeroCouponBond(Instrument):
    """Unit face value paid at maturity (years)."""
    def __init__(self, maturity, dayCounter=None):
        super().__init__()
        self.maturity = _years(maturity, dayCounter)


class VanillaOption(Instrument):
    """European call or put. Accepts QuantLib PlainVanillaPayoff / EuropeanExercise objects or (type, strike, maturity)."""
    def __init__(self, payoff, exercise=None, maturity=None, dayCounter=None):
        super().__init__()
        if maturity is None and exercise is not None and hasattr(exercise, "lastDate"):
            maturity = exercise.lastDate()
        if hasattr(payoff, "strike"):                         # QuantLib payoff
            self.strike = float(payoff.strike())
            ot = payoff.optionType()
            self.isCall = ot == 1 or str(ot).lower().endswith("call")
        else:
            kind, self.strike = payoff
            self.isCall = str(kind).lower() == "call"
        if maturity is None:
            raise ValueError("give the maturity in years or as a QuantLib Date")
        self.maturity = _years(maturity, dayCounter)


class ZeroCouponBondOption(Instrument):
    """European call or put, expiring at `maturity`, on the unit zero-coupon bond maturing at `bondMaturity`
    (QuantLib: Vasicek.discountBondOption(type, strike, maturity, bondMaturity))."""
    def __init__(self, kind, strike, maturity, bondMaturity):
        super().__init__()
        self.isCall = str(kind).lower() == "call"
        self.strike, self.maturity, self.bondMaturity = float(strike), _years(maturity), _years(bondMaturity)
        if self.bondMaturity <= self.maturity:
            raise ValueError("the bond must mature after the option")
