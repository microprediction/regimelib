"""Instruments in QuantLib's mold: build, setPricingEngine, NPV. Times are in years."""
import math


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
    def __init__(self, maturity):
        super().__init__()
        self.maturity = float(maturity)


class VanillaOption(Instrument):
    """European call or put. Accepts QuantLib PlainVanillaPayoff / EuropeanExercise objects or (type, strike, maturity)."""
    def __init__(self, payoff, exercise=None, maturity=None):
        super().__init__()
        if hasattr(payoff, "strike"):                         # QuantLib payoff
            self.strike = float(payoff.strike())
            ot = payoff.optionType()
            self.isCall = ot == 1 or str(ot).lower().endswith("call")
        else:
            kind, self.strike = payoff
            self.isCall = str(kind).lower() == "call"
        if maturity is None:
            raise ValueError("give the maturity in years")
        self.maturity = float(maturity)


class ZeroCouponBondOption(Instrument):
    """European call or put, expiring at `maturity`, on the unit zero-coupon bond maturing at `bondMaturity`
    (QuantLib: Vasicek.discountBondOption(type, strike, maturity, bondMaturity))."""
    def __init__(self, kind, strike, maturity, bondMaturity):
        super().__init__()
        self.isCall = str(kind).lower() == "call"
        self.strike, self.maturity, self.bondMaturity = float(strike), float(maturity), float(bondMaturity)
        if self.bondMaturity <= self.maturity:
            raise ValueError("the bond must mature after the option")
