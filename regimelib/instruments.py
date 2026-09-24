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
