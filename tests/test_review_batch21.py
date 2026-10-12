"""Issues #282 and #283: the Hull-White grid at finite horizons, and two-dimensional greeks between factor slices."""
import math
import pytest
import regimelib as rl


def npv(instrument, engine):
    instrument.setPricingEngine(engine)
    return instrument.NPV()


@pytest.mark.parametrize("a", [0.0, 0.001, 0.05, 0.5])
def test_hull_white_grid_holds_the_factor_by_the_last_decision(a):
    chain = rl.RegimeChain.twoState(3.0, 5.0)
    model = rl.SwitchingHullWhite(chain, 0.03, a, [0.01, 0.01])
    option = rl.ZeroCouponBondOption("call", math.exp(-0.03 * 4.9), 0.1, 5.0)          # ATM forward, 0.1 years
    exact = npv(option, rl.NumericalSwitchingEngine(model))
    assert npv(option, rl.SwitchingFDEngine(model, n=1001, steps=400)) == pytest.approx(exact, rel=5e-5)  # -19% at 0.001


def test_two_dimensional_greeks_move_smoothly_with_the_second_factor():
    one = rl.RegimeChain([[0.0]]); mid = 0.039995508576615046            # a midpoint between two variance nodes
    out = []
    for v0 in (mid - 1e-6, mid + 1e-6):
        model = rl.SwitchingHestonVolOfVol(one, S0=100.0, r=0.03, q=0.01, v0=v0, kappa=2.0, theta=0.04, xi=0.8, rho=-0.6)
        option = rl.VanillaOption(("put", 105.0), maturity=1.0, exercise="american")
        option.setPricingEngine(rl.SwitchingFDEngine(model, n=(151, 61), steps=200))
        out.append((option.delta(), option.gamma()))
    assert out[0][0] == pytest.approx(out[1][0], rel=1e-4)               # 1.3% apart before
    assert out[0][1] == pytest.approx(out[1][1], rel=1e-4)               # 6.4% apart before
