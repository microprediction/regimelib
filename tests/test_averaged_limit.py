"""Order zero of the expansion, with the regimes switching, is the original QuantLib model at stationary-averaged
parameters whenever the switched parameter enters the forcing linearly: theta in Vasicek, CIR, Hull-White and Heston,
sigma^2 in Black-Scholes and Merton, the jump intensity, the G2 covariances. Variance gamma is nonlinear in its
parameters and has no such reduction, so it is checked only in the frozen limit."""
import math
import pytest
import QuantLib as ql
import regimelib as rl

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)
PI = CHAIN.stationaryDistribution()
REF = ql.Date(1, 1, 2020)


def _avg(x):
    return float(PI @ x)


def _curves(r, q):
    dc = ql.Actual365Fixed()
    return (ql.YieldTermStructureHandle(ql.FlatForward(REF, r, dc)), ql.YieldTermStructureHandle(ql.FlatForward(REF, q, dc)))


def test_vasicek_averaged():
    b, s = [0.08, 0.01], [0.015, 0.006]
    bond = rl.ZeroCouponBond(4.0); bond.setPricingEngine(rl.FastSwitchingEngine(rl.SwitchingVasicek(CHAIN, 0.03, 0.5, b, s), order=0))
    assert bond.NPV() == pytest.approx(ql.Vasicek(0.03, 0.5, _avg(b), math.sqrt(_avg([x * x for x in s]))).discountBond(0.0, 4.0, 0.03), rel=1e-12)


def test_cir_averaged():
    th = [0.09, 0.02]
    bond = rl.ZeroCouponBond(3.0); bond.setPricingEngine(rl.FastSwitchingEngine(rl.SwitchingCoxIngersollRoss(CHAIN, 0.04, th, 1.2, 0.15), order=0))
    assert bond.NPV() == pytest.approx(ql.CoxIngersollRoss(0.04, _avg(th), 1.2, 0.15).discountBond(0.0, 3.0, 0.04), rel=1e-9)


def test_hull_white_and_g2_averaged_reproduce_the_curve():
    ql.Settings.instance().evaluationDate = REF
    ts, _ = _curves(0.03, 0.0)
    for model in (rl.SwitchingHullWhite(CHAIN, ts, 0.5, [0.02, 0.005]),
                  rl.SwitchingG2(CHAIN, ts, 0.5, [0.015, 0.005], 0.1, [0.01, 0.004], [-0.5, 0.2])):
        bond = rl.ZeroCouponBond(4.0); bond.setPricingEngine(rl.FastSwitchingEngine(model, order=0))
        assert bond.NPV() == pytest.approx(ts.discount(4.0), rel=1e-12)


def test_black_scholes_averaged():
    ql.Settings.instance().evaluationDate = REF
    S0, r, q, s, K, T = 100.0, 0.02, 0.01, [0.35, 0.12], 105.0, 1.0
    rts, qts = _curves(r, q); sbar = math.sqrt(_avg([x * x for x in s]))
    proc = ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(S0)), qts, rts,
                                        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(REF, ql.NullCalendar(), sbar, ql.Actual365Fixed())))
    payoff, ex = ql.PlainVanillaPayoff(ql.Option.Call, K), ql.EuropeanExercise(REF + ql.Period(365, ql.Days))
    qopt = ql.VanillaOption(payoff, ex); qopt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
    ours = rl.VanillaOption(payoff, ex, maturity=T); ours.setPricingEngine(rl.FastSwitchingEngine(rl.SwitchingBlackScholesProcess(CHAIN, S0, r, q, s), order=0))
    assert ours.NPV() == pytest.approx(qopt.NPV(), rel=1e-9)


def test_heston_averaged():
    ql.Settings.instance().evaluationDate = REF
    S0, r, q, v0, kappa, th, xi, rho, K, T = 100.0, 0.02, 0.0, 0.04, 1.5, [0.09, 0.02], 0.4, -0.6, 95.0, 1.0
    rts, qts = _curves(r, q)
    hp = ql.HestonProcess(rts, qts, ql.QuoteHandle(ql.SimpleQuote(S0)), v0, kappa, _avg(th), xi, rho)
    payoff, ex = ql.PlainVanillaPayoff(ql.Option.Put, K), ql.EuropeanExercise(REF + ql.Period(365, ql.Days))
    qopt = ql.VanillaOption(payoff, ex); qopt.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(hp)))
    ours = rl.VanillaOption(payoff, ex, maturity=T); ours.setPricingEngine(rl.FastSwitchingEngine(rl.SwitchingHestonModel(CHAIN, S0, r, q, v0, kappa, th, xi, rho), order=0))
    assert ours.NPV() == pytest.approx(qopt.NPV(), rel=1e-7)


def test_bates_averaged():
    ql.Settings.instance().evaluationDate = REF
    S0, r, q, v0, kappa, th, xi, rho, lam, nu, delta, K, T = 100.0, 0.02, 0.01, 0.04, 1.5, [0.09, 0.02], 0.4, -0.6, [3.0, 1.0], -0.05, 0.1, 100.0, 1.0
    rts, qts = _curves(r, q)
    bp = ql.BatesProcess(rts, qts, ql.QuoteHandle(ql.SimpleQuote(S0)), v0, kappa, _avg(th), xi, rho, _avg(lam), nu, delta)
    payoff, ex = ql.PlainVanillaPayoff(ql.Option.Call, K), ql.EuropeanExercise(REF + ql.Period(365, ql.Days))
    qopt = ql.VanillaOption(payoff, ex); qopt.setPricingEngine(ql.BatesEngine(ql.BatesModel(bp)))
    ours = rl.VanillaOption(payoff, ex, maturity=T)
    ours.setPricingEngine(rl.FastSwitchingEngine(rl.SwitchingBatesModel(CHAIN, S0, r, q, v0, kappa, th, xi, rho, lam, nu, delta), order=0))
    assert ours.NPV() == pytest.approx(qopt.NPV(), rel=1e-7)


def test_bond_option_averaged():
    b, s, T, S, K = [0.08, 0.01], [0.015, 0.006], 1.0, 4.0, 0.9
    opt = rl.ZeroCouponBondOption("call", K, T, S); opt.setPricingEngine(rl.FastSwitchingEngine(rl.SwitchingVasicek(CHAIN, 0.03, 0.5, b, s), order=0))
    assert opt.NPV() == pytest.approx(ql.Vasicek(0.03, 0.5, _avg(b), math.sqrt(_avg([x * x for x in s]))).discountBondOption(ql.Option.Call, K, T, S), rel=1e-8)
