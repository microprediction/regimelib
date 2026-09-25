"""The engine warns when the expansion may not have converged, and stays quiet when it has."""
import warnings
import pytest
import regimelib as rl


def test_slow_chain_warns_and_fast_chain_does_not():
    model = lambda lam: rl.SwitchingVasicek(rl.RegimeChain.twoState(lam, 1.5 * lam), 0.03, 0.5, [0.10, 0.00], [0.02, 0.005])
    bond = rl.ZeroCouponBond(8.0)
    bond.setPricingEngine(rl.FastSwitchingEngine(model(0.3), order=6))
    with pytest.warns(rl.ExpansionWarning):
        bond.NPV()
    bond.setPricingEngine(rl.FastSwitchingEngine(model(40.0), order=4))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v = bond.NPV()
    d = bond._result("diagnostics")
    assert d["lastTermRelative"] < 1e-6 and d["orderUsed"] == 4 and d["numericalNodes"] == 0


def test_rough_fixed_order_warns():
    model = rl.SwitchingVasicek(rl.RegimeChain.twoState(2.0, 3.0), 0.03, 0.5, [0.10, 0.00], [0.02, 0.005])
    bond = rl.ZeroCouponBond(8.0); bond.setPricingEngine(rl.FastSwitchingEngine(model, order=1))
    with pytest.warns(rl.ExpansionWarning, match="rough|not converging"):
        bond.NPV()
