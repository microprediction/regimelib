"""With every regime equal, each regimelib engine must reproduce the QuantLib engine it mirrors."""
import math
import pytest
import QuantLib as ql
import regimelib as rl

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)


def _ql_curves(r, q, ref):
    dc = ql.Actual365Fixed()
    return (ql.YieldTermStructureHandle(ql.FlatForward(ref, r, dc)),
            ql.YieldTermStructureHandle(ql.FlatForward(ref, q, dc)))


def test_vasicek_bond_matches_quantlib():
    r0, a, b, s, T = 0.03, 0.5, 0.05, 0.01, 4.0
    ours = rl.ZeroCouponBond(T); ours.setPricingEngine(rl.FastSwitchingEngine(rl.SwitchingVasicek(CHAIN, r0, a, b, s), order=2))
    assert ours.NPV() == pytest.approx(ql.Vasicek(r0, a, b, s).discountBond(0.0, T, r0), rel=1e-12)


def test_cir_bond_matches_quantlib():
    r0, th, k, s, T = 0.04, 0.05, 1.2, 0.15, 3.0
    ours = rl.ZeroCouponBond(T); ours.setPricingEngine(rl.NumericalSwitchingEngine(rl.SwitchingCoxIngersollRoss(CHAIN, r0, th, k, s)))
    assert ours.NPV() == pytest.approx(ql.CoxIngersollRoss(r0, th, k, s).discountBond(0.0, T, r0), rel=1e-9)


def test_black_scholes_matches_quantlib():
    S0, r, q, s, K, T = 100.0, 0.02, 0.01, 0.25, 105.0, 1.0
    ref = ql.Date(1, 1, 2020); ql.Settings.instance().evaluationDate = ref
    rts, qts = _ql_curves(r, q, ref)
    proc = ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(S0)), qts, rts,
                                        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(ref, ql.NullCalendar(), s, ql.Actual365Fixed())))
    payoff, ex = ql.PlainVanillaPayoff(ql.Option.Call, K), ql.EuropeanExercise(ref + ql.Period(365, ql.Days))
    qopt = ql.VanillaOption(payoff, ex); qopt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
    ours = rl.VanillaOption(payoff, ex, maturity=T)
    ours.setPricingEngine(rl.FastSwitchingEngine(rl.SwitchingBlackScholesProcess(CHAIN, S0, r, q, s), order=2))
    assert ours.NPV() == pytest.approx(qopt.NPV(), rel=1e-9)


def test_heston_matches_quantlib():
    S0, r, q, v0, kappa, th, xi, rho, K, T = 100.0, 0.02, 0.0, 0.04, 1.5, 0.05, 0.4, -0.6, 95.0, 1.0
    ref = ql.Date(1, 1, 2020); ql.Settings.instance().evaluationDate = ref
    rts, qts = _ql_curves(r, q, ref)
    hp = ql.HestonProcess(rts, qts, ql.QuoteHandle(ql.SimpleQuote(S0)), v0, kappa, th, xi, rho)
    payoff, ex = ql.PlainVanillaPayoff(ql.Option.Put, K), ql.EuropeanExercise(ref + ql.Period(365, ql.Days))
    qopt = ql.VanillaOption(payoff, ex); qopt.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(hp)))
    ours = rl.VanillaOption(payoff, ex, maturity=T)
    ours.setPricingEngine(rl.NumericalSwitchingEngine(rl.SwitchingHestonModel(CHAIN, S0, r, q, v0, kappa, th, xi, rho)))
    assert ours.NPV() == pytest.approx(qopt.NPV(), rel=1e-7)


def test_merton_matches_series():
    S0, r, s, lam, mj, sj, K, T = 100.0, 0.0, 0.2, 3.0, -0.05, 0.1, 100.0, 1.0
    kbar = math.exp(mj + 0.5 * sj * sj) - 1
    lam2 = lam * (1 + kbar); total = 0.0
    N = lambda x: 0.5 * math.erfc(-x / math.sqrt(2))
    for n_ in range(80):                       # Merton (1976): Poisson mixture of Black-Scholes prices
        v = s * s + n_ * sj * sj / T; rn = r - lam * kbar + n_ * math.log(1 + kbar) / T
        d1 = (math.log(S0 / K) + (rn + 0.5 * v) * T) / math.sqrt(v * T)
        total += math.exp(-lam2 * T) * (lam2 * T) ** n_ / math.factorial(n_) * (S0 * N(d1) - K * math.exp(-rn * T) * N(d1 - math.sqrt(v * T)))
    ours = rl.VanillaOption(("call", K), maturity=T)
    ours.setPricingEngine(rl.FastSwitchingEngine(rl.SwitchingMerton76Process(CHAIN, S0, r, 0.0, s, lam, mj, sj), order=2))
    assert ours.NPV() == pytest.approx(total, rel=1e-8)


def test_bates_matches_quantlib():
    S0, r, q, v0, kappa, th, xi, rho, lam, nu, delta, K, T = 100.0, 0.02, 0.01, 0.04, 1.5, 0.05, 0.4, -0.6, 2.0, -0.05, 0.1, 100.0, 1.0
    ref = ql.Date(1, 1, 2020); ql.Settings.instance().evaluationDate = ref
    rts, qts = _ql_curves(r, q, ref)
    bp = ql.BatesProcess(rts, qts, ql.QuoteHandle(ql.SimpleQuote(S0)), v0, kappa, th, xi, rho, lam, nu, delta)
    payoff, ex = ql.PlainVanillaPayoff(ql.Option.Call, K), ql.EuropeanExercise(ref + ql.Period(365, ql.Days))
    qopt = ql.VanillaOption(payoff, ex); qopt.setPricingEngine(ql.BatesEngine(ql.BatesModel(bp)))
    ours = rl.VanillaOption(payoff, ex, maturity=T)
    ours.setPricingEngine(rl.NumericalSwitchingEngine(rl.SwitchingBatesModel(CHAIN, S0, r, q, v0, kappa, th, xi, rho, lam, nu, delta)))
    assert ours.NPV() == pytest.approx(qopt.NPV(), rel=1e-7)


def test_variance_gamma_matches_quantlib():
    S0, r, q, s, nu, th, K, T = 100.0, 0.02, 0.0, 0.2, 0.3, -0.1, 100.0, 1.0
    ref = ql.Date(1, 1, 2020); ql.Settings.instance().evaluationDate = ref
    rts, qts = _ql_curves(r, q, ref)
    vp = ql.VarianceGammaProcess(ql.QuoteHandle(ql.SimpleQuote(S0)), qts, rts, s, nu, th)
    payoff, ex = ql.PlainVanillaPayoff(ql.Option.Call, K), ql.EuropeanExercise(ref + ql.Period(365, ql.Days))
    qopt = ql.VanillaOption(payoff, ex); qopt.setPricingEngine(ql.VarianceGammaEngine(vp))
    ours = rl.VanillaOption(payoff, ex, maturity=T)
    ours.setPricingEngine(rl.FastSwitchingEngine(rl.SwitchingVarianceGammaProcess(CHAIN, S0, r, q, s, nu, th), order=1))
    assert ours.NPV() == pytest.approx(qopt.NPV(), rel=1e-6)


def test_vasicek_bond_option_matches_quantlib():
    r0, a, b, s, T, S, K = 0.03, 0.5, 0.05, 0.01, 1.0, 4.0, 0.9
    model = ql.Vasicek(r0, a, b, s)
    ours_model = rl.SwitchingVasicek(CHAIN, r0, a, b, s)
    for kind, qtype in (("call", ql.Option.Call), ("put", ql.Option.Put)):
        opt = rl.ZeroCouponBondOption(kind, K, T, S)
        opt.setPricingEngine(rl.NumericalSwitchingEngine(ours_model))
        assert opt.NPV() == pytest.approx(model.discountBondOption(qtype, K, T, S), rel=1e-8)
        opt.setPricingEngine(rl.FastSwitchingEngine(ours_model, order=2))
        assert opt.NPV() == pytest.approx(model.discountBondOption(qtype, K, T, S), rel=1e-8)


def test_hull_white_matches_quantlib_and_dates():
    ref = ql.Date(1, 1, 2020); ql.Settings.instance().evaluationDate = ref
    a, s = 0.5, 0.012
    ts = ql.YieldTermStructureHandle(ql.ZeroCurve([ref, ref + ql.Period(2, ql.Years), ref + ql.Period(10, ql.Years)],
                                                  [0.02, 0.03, 0.035], ql.Actual365Fixed()))
    hw = ql.HullWhite(ts, a, s)
    model = rl.SwitchingHullWhite(CHAIN, ts, a, s)
    T = ref + ql.Period(4, ql.Years)                       # a QuantLib Date as the maturity
    bond = rl.ZeroCouponBond(T); bond.setPricingEngine(rl.FastSwitchingEngine(model, order=2))
    t = ql.Actual365Fixed().yearFraction(ref, T)
    assert bond.NPV() == pytest.approx(ts.discount(t), rel=1e-10)
    assert bond.NPV() == pytest.approx(hw.discountBond(0.0, t, model.r0), rel=1e-8)
    ex = ql.EuropeanExercise(ref + ql.Period(1, ql.Years))
    opt = rl.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, 100.0), ex)   # maturity taken from the exercise
    assert opt.maturity == pytest.approx(ql.Actual365Fixed().yearFraction(ref, ex.lastDate()))


def test_hull_white_switching_converges():
    ref = ql.Date(1, 1, 2020); ql.Settings.instance().evaluationDate = ref
    chain = rl.RegimeChain.twoState(20.0, 30.0)
    model = rl.SwitchingHullWhite(chain, 0.03, 0.5, [0.02, 0.005])
    bond = rl.ZeroCouponBond(5.0)
    bond.setPricingEngine(rl.NumericalSwitchingEngine(model)); ref_npv = bond.NPV()
    errs = []
    for o in (0, 1, 2, 3):
        bond.setPricingEngine(rl.FastSwitchingEngine(model, order=o)); errs.append(abs(bond.NPV() - ref_npv))
    assert errs[0] > errs[1] > errs[2] > errs[3]
    assert errs[0] > 1e-6                                   # the switching correction is visible at order 0


def test_g2_matches_quantlib():
    ref = ql.Date(1, 1, 2020); ql.Settings.instance().evaluationDate = ref
    ts = ql.YieldTermStructureHandle(ql.FlatForward(ref, 0.03, ql.Actual365Fixed()))
    a, s, b, e, rho, T = 0.5, 0.01, 0.1, 0.008, -0.3, 4.0
    g2 = ql.G2(ts, a, s, b, e, rho)
    bond = rl.ZeroCouponBond(T); bond.setPricingEngine(rl.FastSwitchingEngine(rl.SwitchingG2(CHAIN, ts, a, s, b, e, rho), order=2))
    assert bond.NPV() == pytest.approx(ts.discount(T), rel=1e-10)
    assert bond.NPV() == pytest.approx(g2.discountBond(0.0, T, [0.0, 0.0]), rel=1e-9)


def test_hull_white_bond_option_matches_quantlib():
    ref = ql.Date(1, 1, 2020); ql.Settings.instance().evaluationDate = ref
    a, s, T, S, K = 0.5, 0.012, 1.0, 4.0, 0.9
    ts = ql.YieldTermStructureHandle(ql.ZeroCurve([ref, ref + ql.Period(2, ql.Years), ref + ql.Period(10, ql.Years)],
                                                  [0.02, 0.03, 0.035], ql.Actual365Fixed()))
    hw = ql.HullWhite(ts, a, s); model = rl.SwitchingHullWhite(CHAIN, ts, a, s)
    for kind, qtype in (("call", ql.Option.Call), ("put", ql.Option.Put)):
        opt = rl.ZeroCouponBondOption(kind, K, T, S)
        opt.setPricingEngine(rl.FastSwitchingEngine(model, order=2))
        assert opt.NPV() == pytest.approx(hw.discountBondOption(qtype, K, T, S), rel=1e-7)


def test_hull_white_bond_option_switching_converges():
    chain = rl.RegimeChain.twoState(20.0, 30.0)
    model = rl.SwitchingHullWhite(chain, 0.03, 0.5, [0.02, 0.005])
    opt = rl.ZeroCouponBondOption("call", 0.9, 1.0, 4.0)
    opt.setPricingEngine(rl.NumericalSwitchingEngine(model)); ref_npv = opt.NPV()
    errs = []
    for o in (0, 1, 2):
        opt.setPricingEngine(rl.FastSwitchingEngine(model, order=o)); errs.append(abs(opt.NPV() - ref_npv))
    assert errs[0] > errs[1] > errs[2]
