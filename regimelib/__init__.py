"""regimelib: QuantLib's models with a hidden Markov regime, priced by the fast-switching expansion."""
from .chain import RegimeChain
from .models import (SwitchingVasicek, SwitchingCoxIngersollRoss, SwitchingBlackScholesProcess,
                     SwitchingHestonModel, SwitchingMerton76Process, SwitchingBatesModel, SwitchingVarianceGammaProcess, SwitchingHullWhite)
from .instruments import ZeroCouponBond, VanillaOption, ZeroCouponBondOption
from .engines import FastSwitchingEngine, NumericalSwitchingEngine
from .montecarlo import MonteCarloSwitchingEngine
__all__ = ["RegimeChain", "SwitchingVasicek", "SwitchingCoxIngersollRoss", "SwitchingBlackScholesProcess",
           "SwitchingHestonModel", "SwitchingMerton76Process", "SwitchingBatesModel", "SwitchingVarianceGammaProcess", "SwitchingHullWhite", "ZeroCouponBond", "VanillaOption", "ZeroCouponBondOption",
           "FastSwitchingEngine", "NumericalSwitchingEngine", "MonteCarloSwitchingEngine"]
__version__ = "0.0.1"
