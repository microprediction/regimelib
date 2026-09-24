"""regimelib: QuantLib's models with a hidden Markov regime, priced by the fast-switching expansion."""
from .chain import RegimeChain
from .models import (SwitchingVasicek, SwitchingCoxIngersollRoss, SwitchingBlackScholesProcess,
                     SwitchingHestonModel, SwitchingMerton76Process, SwitchingBatesModel, SwitchingVarianceGammaProcess)
from .instruments import ZeroCouponBond, VanillaOption, ZeroCouponBondOption
from .engines import FastSwitchingEngine, NumericalSwitchingEngine
__all__ = ["RegimeChain", "SwitchingVasicek", "SwitchingCoxIngersollRoss", "SwitchingBlackScholesProcess",
           "SwitchingHestonModel", "SwitchingMerton76Process", "SwitchingBatesModel", "SwitchingVarianceGammaProcess", "ZeroCouponBond", "VanillaOption", "ZeroCouponBondOption",
           "FastSwitchingEngine", "NumericalSwitchingEngine"]
__version__ = "0.0.1"
