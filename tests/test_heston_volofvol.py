"""Heston with a switched volatility of variance (first-order tier, two-dimensional grid): the frozen grid against
QuantLib's analytic Heston price, and the first-order engine against the coupled switching PDE."""
import warnings
import pytest
import QuantLib as ql
import regimelib as rl

REF = ql.Date(1, 1, 2020)
S0, r, q, v0, kappa, theta, rho, K, T = 100.0, 0.03, 0.0, 0.04, 2.0, 0.04, -0.6, 100.0, 1.0


@pytest.mark.slow
def test_frozen_vol_of_vol_matches_analytic_heston():
    ql.Settings.instance().evaluationDate = REF; dc = ql.Actual365Fixed(); xi = 0.5
    proc = ql.HestonProcess(ql.YieldTermStructureHandle(ql.FlatForward(REF, r, dc)), ql.YieldTermStructureHandle(ql.FlatForward(REF, q, dc)),
                            ql.QuoteHandle(ql.SimpleQuote(S0)), v0, kappa, theta, xi, rho)
    o = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, K), ql.EuropeanExercise(REF + ql.Period(365, ql.Days)))
    o.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(proc)))
    model = rl.SwitchingHestonVolOfVol(rl.RegimeChain.twoState(3.0, 5.0), S0, r, q, v0, kappa, theta, xi, rho)
    opt = rl.VanillaOption(("call", K), maturity=T); opt.setPricingEngine(rl.FirstOrderFDEngine(model, n=(201, 81)))
    assert opt.NPV() == pytest.approx(o.NPV(), rel=1e-3)                  # grid error; converges with the grid


def test_switched_vol_of_vol_first_order_against_switching_pde():
    model = rl.SwitchingHestonVolOfVol(rl.RegimeChain.twoState(6.0, 6.0), S0, r, q, v0, kappa, theta, [0.8, 0.2], rho)
    opt = rl.VanillaOption(("call", K), maturity=T)
    e = rl.FirstOrderFDEngine(model, regime=0, n=(151, 61)); opt.setPricingEngine(e)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); first = opt.NPV()
    opt.setPricingEngine(rl.SwitchingFDReferee(model, regime=0, n=(151, 61))); ref = opt.NPV()
    assert abs(first - ref) / ref < 1e-3
    assert abs(e.averaged - ref) > 5 * abs(first - ref)                  # the correction and memory terms matter
    assert e.diagnostics["correctionRelative"] > 0 and e.diagnostics["memoryRelative"] > 0
