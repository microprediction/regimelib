"""regimelib: QuantLib's models with a hidden Markov regime, priced by the fast-switching expansion."""
from .chain import RegimeChain
from .models import (SwitchingVasicek, SwitchingVasicekJumps, SwitchingCoxIngersollRoss, SwitchingBlackScholesProcess,
                     SwitchingHestonModel, SwitchingMerton76Process, SwitchingBatesModel, SwitchingVarianceGammaProcess, SwitchingHullWhite, SwitchingG2, SwitchingCEVProcess, SwitchingIntensityBasket, SwitchingHestonVolOfVol)
from .instruments import ZeroCouponBond, VanillaOption, ZeroCouponBondOption, CouponBond, CouponBondOption, Swaption, CapFloor, BarrierOption, ContinuousGeometricAsianOption, CreditDefaultSwap
FirstToDefaultSwap = CreditDefaultSwap                 # on a SwitchingIntensityBasket
from .engines import FastSwitchingEngine, NumericalSwitchingEngine, ExpansionWarning
from .montecarlo import MonteCarloSwitchingEngine
from .firstorder import FirstOrderFDEngine, SwitchingFDReferee
from .fd import SwitchingFDEngine
from .hybrid import SwitchingEquityRates
from .calibration import VolatilityHelper, calibrate
from . import symbolic
__all__ = ["RegimeChain", "SwitchingVasicek", "SwitchingVasicekJumps", "SwitchingCoxIngersollRoss", "SwitchingBlackScholesProcess",
           "SwitchingHestonModel", "SwitchingMerton76Process", "SwitchingBatesModel", "SwitchingVarianceGammaProcess", "SwitchingHullWhite", "SwitchingG2", "SwitchingCEVProcess", "SwitchingIntensityBasket", "SwitchingHestonVolOfVol", "ZeroCouponBond", "VanillaOption", "ZeroCouponBondOption", "CouponBond", "CouponBondOption", "Swaption", "CapFloor", "BarrierOption", "ContinuousGeometricAsianOption", "CreditDefaultSwap", "FirstToDefaultSwap",
           "FastSwitchingEngine", "NumericalSwitchingEngine", "ExpansionWarning", "MonteCarloSwitchingEngine", "FirstOrderFDEngine", "SwitchingFDReferee", "SwitchingFDEngine", "VolatilityHelper", "calibrate", "SwitchingEquityRates"]
__version__ = "0.0.1"
