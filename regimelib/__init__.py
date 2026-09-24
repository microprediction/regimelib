"""regimelib: QuantLib's models with a hidden Markov regime, priced by the fast-switching expansion."""
from .chain import RegimeChain
from .models import (SwitchingVasicek, SwitchingCoxIngersollRoss, SwitchingBlackScholesProcess,
                     SwitchingHestonModel, SwitchingMerton76Process, SwitchingBatesModel, SwitchingVarianceGammaProcess, SwitchingHullWhite, SwitchingG2, SwitchingCEVProcess)
from .instruments import ZeroCouponBond, VanillaOption, ZeroCouponBondOption, CouponBond, CouponBondOption, Swaption, CapFloor
from .engines import FastSwitchingEngine, NumericalSwitchingEngine, ExpansionWarning
from .montecarlo import MonteCarloSwitchingEngine
from .firstorder import FirstOrderFDEngine, SwitchingFDReferee
from . import symbolic
__all__ = ["RegimeChain", "SwitchingVasicek", "SwitchingCoxIngersollRoss", "SwitchingBlackScholesProcess",
           "SwitchingHestonModel", "SwitchingMerton76Process", "SwitchingBatesModel", "SwitchingVarianceGammaProcess", "SwitchingHullWhite", "SwitchingG2", "SwitchingCEVProcess", "ZeroCouponBond", "VanillaOption", "ZeroCouponBondOption", "CouponBond", "CouponBondOption", "Swaption", "CapFloor",
           "FastSwitchingEngine", "NumericalSwitchingEngine", "ExpansionWarning", "MonteCarloSwitchingEngine", "FirstOrderFDEngine", "SwitchingFDReferee"]
__version__ = "0.0.1"
