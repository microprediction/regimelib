"""order=None adds terms until the tolerance is met or the series stops improving."""
import pytest
import regimelib as rl


def test_adaptive_order_reaches_tolerance():
    chain = rl.RegimeChain.twoState(20.0, 30.0)
    model = rl.SwitchingVasicek(chain, 0.03, 0.5, [0.07, 0.01], [0.012, 0.008])
    bond = rl.ZeroCouponBond(5.0)
    bond.setPricingEngine(rl.NumericalSwitchingEngine(model)); ref = bond.NPV()
    eng = rl.FastSwitchingEngine(model, order=None, tol=1e-9); bond.setPricingEngine(eng)
    v = bond.NPV()
    assert abs(v - ref) / ref < 1e-8 and 2 <= eng.orderUsed <= 12 and eng.lastIncrement <= 1e-9


def test_adaptive_order_stops_at_best_truncation():
    chain = rl.RegimeChain.twoState(1.0, 1.5)              # slow switching: the series is asymptotic, not convergent
    model = rl.SwitchingVasicek(chain, 0.03, 0.5, [0.10, 0.00], [0.02, 0.005])
    bond = rl.ZeroCouponBond(8.0)
    bond.setPricingEngine(rl.NumericalSwitchingEngine(model)); ref = bond.NPV()
    eng = rl.FastSwitchingEngine(model, order=None, tol=1e-14, maxOrder=12); bond.setPricingEngine(eng)
    v = bond.NPV()
    errs = []
    for n in range(0, 13):
        bond.setPricingEngine(rl.FastSwitchingEngine(model, order=n)); errs.append(abs(bond.NPV() - ref))
    assert eng.orderUsed <= 12
    # the engine's own uncertainty estimate (the last increment) bounds its error, and it lands near the best fixed order
    assert abs(v - ref) <= 3 * eng.lastIncrement * abs(v) + 3 * min(errs) + 1e-15
