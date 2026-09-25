"""The first-order finite-difference engine: certified on Black-Scholes (where the exact engine gives the first order)
and applied to CEV, with the switching PDE solved without expansion as the referee."""
import math
import pytest
import regimelib as rl

CHAIN = rl.RegimeChain.twoState(15.0, 10.0)


def test_black_scholes_first_order_matches_exact_engine():
    model = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.02, 0.0, [0.3, 0.15])
    opt = rl.VanillaOption(("call", 100.0), maturity=1.0)
    opt.setPricingEngine(rl.FastSwitchingEngine(model, order=1, regime=0)); exact1 = opt.NPV()
    opt.setPricingEngine(rl.FastSwitchingEngine(model, order=0, regime=0)); exact0 = opt.NPV()
    fd = rl.FirstOrderFDEngine(model, regime=0, n=1601); opt.setPricingEngine(fd); v = opt.NPV()
    assert fd.averaged == pytest.approx(exact0, rel=2e-4)                   # spatial discretization only
    assert v - fd.averaged == pytest.approx(exact1 - exact0, rel=2e-2)     # the first-order term itself
    opt.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0)); ref = opt.NPV()
    opt.setPricingEngine(rl.SwitchingFDReferee(model, regime=0, n=1601))
    assert opt.NPV() == pytest.approx(ref, rel=2e-4)


@pytest.mark.slow
def test_cev_first_order_against_switching_pde():
    model = rl.SwitchingCEVProcess(CHAIN, 100.0, 0.02, 0.0, [2.5, 1.2], 0.6)      # sigma S^0.6: vol 0.16 and 0.08 at S = 100
    opt = rl.VanillaOption(("call", 100.0), maturity=1.0)
    opt.setPricingEngine(rl.SwitchingFDReferee(model, regime=0, n=1601)); ref = opt.NPV()
    fd = rl.FirstOrderFDEngine(model, regime=0, n=1601); opt.setPricingEngine(fd); v = opt.NPV()
    err0, err1 = abs(fd.averaged - ref), abs(v - ref)
    assert err1 < 0.3 * err0                                                # first order beats the averaged model
    fast = rl.RegimeChain.twoState(60.0, 40.0)
    model2 = rl.SwitchingCEVProcess(fast, 100.0, 0.02, 0.0, [2.5, 1.2], 0.6)
    opt.setPricingEngine(rl.SwitchingFDReferee(model2, regime=0, n=1601)); ref2 = opt.NPV()
    fd2 = rl.FirstOrderFDEngine(model2, regime=0, n=1601); opt.setPricingEngine(fd2); v2 = opt.NPV()
    assert abs(v2 - ref2) < 0.3 * abs(v - ref)                              # and the residual falls with the holding time


def test_cev_frozen_matches_quantlib():
    import QuantLib as ql
    S0, r, q, s, beta, K, T = 100.0, 0.02, 0.0, 1.8, 0.6, 100.0, 1.0
    ref_date = ql.Date(1, 1, 2020); ql.Settings.instance().evaluationDate = ref_date
    payoff, ex = ql.PlainVanillaPayoff(ql.Option.Call, K), ql.EuropeanExercise(ref_date + ql.Period(365, ql.Days))
    qopt = ql.VanillaOption(payoff, ex)
    F = S0 * math.exp((r - q) * T)                   # QuantLib's CEV engine is written on the driftless forward
    qopt.setPricingEngine(ql.AnalyticCEVEngine(F, s, beta, ql.YieldTermStructureHandle(ql.FlatForward(ref_date, r, ql.Actual365Fixed()))))
    frozen = rl.SwitchingCEVProcess(CHAIN, S0, r, q, s, beta)
    opt = rl.VanillaOption(payoff, ex, maturity=T)
    fd = rl.FirstOrderFDEngine(frozen, n=2401); opt.setPricingEngine(fd); v = opt.NPV()
    assert fd.correction == pytest.approx(0.0, abs=1e-12) and fd.memory == pytest.approx(0.0, abs=1e-12)
    # the spot process sigma S^beta drifts, the forward process does not: the volatility scales differ by e^{(beta - 1) r t}
    assert v == pytest.approx(qopt.NPV(), rel=5e-3)


def test_first_order_diagnostics_track_the_error():
    """The neglected term is about the square of the relative first-order correction; slower chains warn."""
    import warnings
    errs, ests = [], []
    for rates in ((10.0, 10.0), (2.0, 2.0)):
        m = rl.SwitchingCEVProcess(rl.RegimeChain.twoState(*rates), 100.0, 0.03, 0.0, [0.35, 0.15], 0.7)
        o = rl.VanillaOption(("call", 100.0), maturity=1.0); e = rl.FirstOrderFDEngine(m, regime=0, n=801); o.setPricingEngine(e)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always"); v = o.NPV()
        assert bool(w) == (rates == (2.0, 2.0))
        o.setPricingEngine(rl.SwitchingFDReferee(m, regime=0, n=801)); ref = o.NPV()
        errs.append(abs(v - ref) / ref); ests.append(e.diagnostics["estimatedError"])
    for err, est in zip(errs, ests):
        assert 0.3 * est < err < 3 * est
