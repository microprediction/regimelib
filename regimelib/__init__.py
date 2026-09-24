"""regimelib: QuantLib's models with a hidden Markov regime, priced by the fast-switching expansion."""
from .chain import RegimeChain
from .models import (SwitchingVasicek, SwitchingCoxIngersollRoss, SwitchingBlackScholesProcess,
                     SwitchingHestonModel, SwitchingMerton76Process, SwitchingBatesModel, SwitchingVarianceGammaProcess, SwitchingHullWhite, SwitchingG2, SwitchingCEVProcess)
from .instruments import ZeroCouponBond, VanillaOption, ZeroCouponBondOption, CouponBond, CouponBondOption, Swaption, CapFloor, BarrierOption
from .engines import FastSwitchingEngine, NumericalSwitchingEngine, ExpansionWarning
from .montecarlo import MonteCarloSwitchingEngine
from .firstorder import FirstOrderFDEngine, SwitchingFDReferee
from .fd import SwitchingFDEngine
from . import symbolic
__all__ = ["RegimeChain", "SwitchingVasicek", "SwitchingCoxIngersollRoss", "SwitchingBlackScholesProcess",
           "SwitchingHestonModel", "SwitchingMerton76Process", "SwitchingBatesModel", "SwitchingVarianceGammaProcess", "SwitchingHullWhite", "SwitchingG2", "SwitchingCEVProcess", "ZeroCouponBond", "VanillaOption", "ZeroCouponBondOption", "CouponBond", "CouponBondOption", "Swaption", "CapFloor", "BarrierOption",
           "FastSwitchingEngine", "NumericalSwitchingEngine", "ExpansionWarning", "MonteCarloSwitchingEngine", "FirstOrderFDEngine", "SwitchingFDReferee", "SwitchingFDEngine"]
__version__ = "0.0.1"
