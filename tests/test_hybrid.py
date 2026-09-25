"""Equity with stochastic rates on one chain: frozen limits against QuantLib's analytic Black-Scholes-Hull-White and
Heston-Hull-White engines (zero equity-rate correlation), put-call parity with the switching bond, and the expansion
converging to the numerical solution with switching."""
import math
import pytest
import QuantLib as ql
import regimelib as rl

REF = ql.Date(1, 1, 2020)


def _curves(r, q):
    ql.Settings.instance().evaluationDate = REF; dc = ql.Actual365Fixed()
    return (ql.YieldTermStructureHandle(ql.FlatForward(REF, r, dc)), ql.YieldTermStructureHandle(ql.FlatForward(REF, q, dc)), dc)


def test_black_scholes_hull_white_frozen():
    S0, r, q, sigma, a, sr, K = 100.0, 0.03, 0.01, 0.25, 0.4, 0.015, 105.0
    rts, qts, dc = _curves(r, q)
    proc = ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(S0)), qts, rts, ql.BlackVolTermStructureHandle(ql.BlackConstantVol(REF, ql.NullCalendar(), sigma, dc)))
    hw = ql.HullWhite(rts, a, sr)
    chain = rl.RegimeChain.twoState(3.0, 5.0)
    model = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(chain, S0, r, q, sigma), rl.SwitchingHullWhite(chain, rts, a, sr))
    for rho in (0.0, 0.4, -0.7):
        model = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(chain, S0, r, q, sigma), rl.SwitchingHullWhite(chain, rts, a, sr), rho=rho)
        for kind, qk in (("call", ql.Option.Call), ("put", ql.Option.Put)):
            o = ql.VanillaOption(ql.PlainVanillaPayoff(qk, K), ql.EuropeanExercise(REF + ql.Period(730, ql.Days)))
            o.setPricingEngine(ql.AnalyticBSMHullWhiteEngine(rho, proc, hw))
            ours = rl.VanillaOption((kind, K), maturity=730 / 365)
            ours.setPricingEngine(rl.NumericalSwitchingEngine(model)); assert ours.NPV() == pytest.approx(o.NPV(), rel=1e-8)
            ours.setPricingEngine(rl.FastSwitchingEngine(model, order=2)); assert ours.NPV() == pytest.approx(o.NPV(), rel=1e-8)


def test_heston_hull_white_frozen():
    S0, r, q, v0, kappa, theta, xi, rho, a, sr, K = 100.0, 0.03, 0.0, 0.04, 1.5, 0.05, 0.4, -0.5, 0.3, 0.012, 100.0
    rts, qts, dc = _curves(r, q)
    hp = ql.HestonProcess(rts, qts, ql.QuoteHandle(ql.SimpleQuote(S0)), v0, kappa, theta, xi, rho)
    hw = ql.HullWhite(rts, a, sr)
    o = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, K), ql.EuropeanExercise(REF + ql.Period(365, ql.Days)))
    o.setPricingEngine(ql.AnalyticHestonHullWhiteEngine(ql.HestonModel(hp), hw, 192))
    chain = rl.RegimeChain.twoState(3.0, 5.0)
    model = rl.SwitchingEquityRates(rl.SwitchingHestonModel(chain, S0, r, q, v0, kappa, theta, xi, rho), rl.SwitchingHullWhite(chain, rts, a, sr))
    ours = rl.VanillaOption(("call", K), maturity=1.0); ours.setPricingEngine(rl.NumericalSwitchingEngine(model))
    assert ours.NPV() == pytest.approx(o.NPV(), rel=1e-6)


def test_switching_rates_and_vol_orders_converge_and_parity():
    chain = rl.RegimeChain.twoState(12.0, 8.0)
    model = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(chain, 100.0, 0.03, 0.01, [0.35, 0.15]),
                                    rl.SwitchingVasicek(chain, 0.03, 0.4, [0.06, 0.01], [0.02, 0.008]))
    call, put = rl.VanillaOption(("call", 100.0), maturity=2.0), rl.VanillaOption(("put", 100.0), maturity=2.0)
    call.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=1)); ref = call.NPV()
    put.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=1))
    bond = rl.ZeroCouponBond(2.0); bond.setPricingEngine(rl.NumericalSwitchingEngine(model.rates, regime=1))
    assert ref - put.NPV() == pytest.approx(100.0 * math.exp(-0.02) - 100.0 * bond.NPV(), abs=1e-10)
    errs = []
    for order in (0, 1, 2, 3):
        call.setPricingEngine(rl.FastSwitchingEngine(model, order=order, regime=1)); errs.append(abs(call.NPV() - ref))
    assert errs[0] > errs[1] > errs[2] > errs[3] and errs[3] < 1e-5 * ref


def test_hybrid_grid_european_matches_lewis_and_american_frozen_matches_quantlib():
    chain = rl.RegimeChain.twoState(6.0, 4.0)
    model = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(chain, 100.0, 0.03, 0.01, [0.35, 0.15]),
                                    rl.SwitchingVasicek(chain, 0.03, 0.4, [0.06, 0.01], [0.02, 0.008]), rho=0.3)
    put = rl.VanillaOption(("put", 105.0), maturity=1.0)
    put.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=1)); ref = put.NPV()
    put.setPricingEngine(rl.SwitchingFDEngine(model, regime=1, n=(401, 61), steps=300))
    assert put.NPV() == pytest.approx(ref, rel=2e-3)                        # grid error on the two-dimensional grid
    am = rl.VanillaOption(("put", 105.0), exercise="american", maturity=1.0)
    am.setPricingEngine(rl.SwitchingFDEngine(model, regime=1, n=(401, 61), steps=300))
    assert am.NPV() > put.NPV()
    # rates frozen at a constant (tiny sigma_r, mean level r0): the American put is QuantLib's
    S0, r, q, sigma, K = 100.0, 0.03, 0.01, 0.25, 105.0
    rts, qts, dc = _curves(r, q)
    proc = ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(S0)), qts, rts, ql.BlackVolTermStructureHandle(ql.BlackConstantVol(REF, ql.NullCalendar(), sigma, dc)))
    o = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Put, K), ql.AmericanExercise(REF, REF + ql.Period(365, ql.Days)))
    o.setPricingEngine(ql.FdBlackScholesVanillaEngine(proc, 2000, 2000))
    frozen = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(chain, S0, r, q, sigma), rl.SwitchingVasicek(chain, r, 0.4, r, 1e-6))
    am.setPricingEngine(rl.SwitchingFDEngine(frozen, n=(801, 5), steps=400, width=(2.0, 1e-5)))
    assert am.NPV() == pytest.approx(o.NPV(), rel=1e-3)
    assert am.delta() == pytest.approx(o.delta(), abs=3e-3)
