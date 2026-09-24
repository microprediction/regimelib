"""Implied volatility, digital options and coupon bonds in the frozen limit against QuantLib."""
import math
import pytest
import QuantLib as ql
import regimelib as rl

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)
REF = ql.Date(1, 1, 2020)


def _bs_process(S0, r, q, s):
    dc = ql.Actual365Fixed(); ql.Settings.instance().evaluationDate = REF
    return ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(S0)), ql.YieldTermStructureHandle(ql.FlatForward(REF, q, dc)),
                                        ql.YieldTermStructureHandle(ql.FlatForward(REF, r, dc)),
                                        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(REF, ql.NullCalendar(), s, dc)))


def test_implied_volatility_round_trip_and_switching():
    S0, r, q, s, K, T = 100.0, 0.02, 0.01, [0.3, 0.15], 105.0, 1.0
    opt = rl.VanillaOption(("call", K), maturity=T)
    opt.setPricingEngine(rl.FastSwitchingEngine(rl.SwitchingBlackScholesProcess(CHAIN, S0, r, q, 0.25), order=2))
    assert opt.impliedVolatility() == pytest.approx(0.25, abs=1e-9)                   # frozen: exactly the input
    opt.setPricingEngine(rl.NumericalSwitchingEngine(rl.SwitchingBlackScholesProcess(CHAIN, S0, r, q, s), regime=0))
    iv = opt.impliedVolatility()
    assert 0.15 < iv < 0.3                                                             # a switching mixture


def test_digitals_match_quantlib():
    S0, r, q, s, K, T = 100.0, 0.02, 0.01, 0.25, 105.0, 1.0
    proc = _bs_process(S0, r, q, s); ex = ql.EuropeanExercise(REF + ql.Period(365, ql.Days))
    model = rl.SwitchingBlackScholesProcess(CHAIN, S0, r, q, s)
    for payoff in (ql.CashOrNothingPayoff(ql.Option.Call, K, 10.0), ql.CashOrNothingPayoff(ql.Option.Put, K, 10.0),
                   ql.AssetOrNothingPayoff(ql.Option.Call, K), ql.AssetOrNothingPayoff(ql.Option.Put, K)):
        qopt = ql.VanillaOption(payoff, ex); qopt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
        ours = rl.VanillaOption(payoff, ex, maturity=T); ours.setPricingEngine(rl.FastSwitchingEngine(model, order=1))
        assert ours.NPV() == pytest.approx(qopt.NPV(), rel=1e-8)


def test_digital_with_switching_against_monte_carlo_like_identity():
    # cash-or-nothing call plus put equals the discounted cash; asset call plus put equals the discounted forward
    model = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.02, 0.01, [0.3, 0.15]); T = 1.0
    e = rl.NumericalSwitchingEngine(model)
    c = rl.VanillaOption(("cash", "call", 105.0, 10.0), maturity=T); p = rl.VanillaOption(("cash", "put", 105.0, 10.0), maturity=T)
    c.setPricingEngine(e); p.setPricingEngine(e)
    assert c.NPV() + p.NPV() == pytest.approx(10.0 * math.exp(-0.02 * T), rel=1e-9)


def test_coupon_bond_matches_quantlib_sum():
    r0, a, b, s = 0.03, 0.5, 0.05, 0.01
    model = rl.SwitchingVasicek(CHAIN, r0, a, b, s); v = ql.Vasicek(r0, a, b, s)
    bond = rl.CouponBond(faceAmount=100.0, couponRate=0.04, times=[1.0, 2.0, 3.0]); bond.setPricingEngine(rl.FastSwitchingEngine(model, order=2))
    expected = sum(4.0 * v.discountBond(0.0, t, r0) for t in (1.0, 2.0, 3.0)) + 100.0 * v.discountBond(0.0, 3.0, r0)
    assert bond.NPV() == pytest.approx(expected, rel=1e-12)
    assert bond.delta() < 0 < bond.gamma()
