"""American and barrier options: frozen limit against QuantLib's finite-difference and analytic barrier engines;
with switching on, the European price from the time-stepping engine agrees with the characteristic-function engine."""
import pytest
import QuantLib as ql
import regimelib as rl

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)
REF = ql.Date(1, 1, 2020)


def _ql_process(S0, r, q, sigma):
    ql.Settings.instance().evaluationDate = REF; dc = ql.Actual365Fixed()
    return ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(S0)),
                                        ql.YieldTermStructureHandle(ql.FlatForward(REF, q, dc)),
                                        ql.YieldTermStructureHandle(ql.FlatForward(REF, r, dc)),
                                        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(REF, ql.NullCalendar(), sigma, dc)))


def test_american_put_matches_quantlib_fd():
    S0, r, q, sigma, K, T = 100.0, 0.05, 0.02, 0.25, 105.0, 1.0
    proc = _ql_process(S0, r, q, sigma)
    expiry = REF + ql.Period(365, ql.Days)
    payoff, ex = ql.PlainVanillaPayoff(ql.Option.Put, K), ql.AmericanExercise(REF, expiry)
    o = ql.VanillaOption(payoff, ex); o.setPricingEngine(ql.FdBlackScholesVanillaEngine(proc, 2000, 2000))
    model = rl.SwitchingBlackScholesProcess(CHAIN, S0, r, q, sigma)
    ours = rl.VanillaOption(payoff, ex); ours.setPricingEngine(rl.SwitchingFDEngine(model, n=1601, steps=800))
    assert ours.NPV() == pytest.approx(o.NPV(), rel=2e-4)
    assert ours.delta() == pytest.approx(o.delta(), abs=2e-3)
    assert ours.gamma() == pytest.approx(o.gamma(), rel=2e-2)


@pytest.mark.parametrize("btype, barrier", [(ql.Barrier.DownOut, 85.0), (ql.Barrier.UpOut, 130.0),
                                            (ql.Barrier.DownIn, 85.0), (ql.Barrier.UpIn, 130.0)])
def test_barrier_matches_analytic_engine(btype, barrier):
    S0, r, q, sigma, K, T = 100.0, 0.05, 0.02, 0.25, 100.0, 1.0
    proc = _ql_process(S0, r, q, sigma); expiry = REF + ql.Period(365, ql.Days)
    payoff, ex = ql.PlainVanillaPayoff(ql.Option.Call, K), ql.EuropeanExercise(expiry)
    o = ql.BarrierOption(btype, barrier, 0.0, payoff, ex); o.setPricingEngine(ql.AnalyticBarrierEngine(proc))
    model = rl.SwitchingBlackScholesProcess(CHAIN, S0, r, q, sigma)
    ours = rl.BarrierOption(btype, barrier, 0.0, payoff, ex); ours.setPricingEngine(rl.SwitchingFDEngine(model, n=2001, steps=800))
    assert ours.NPV() == pytest.approx(o.NPV(), rel=5e-4)


def test_european_with_switching_matches_lewis():
    chain = rl.RegimeChain.twoState(6.0, 4.0)
    model = rl.SwitchingBlackScholesProcess(chain, 100.0, 0.03, 0.0, [0.35, 0.15])
    opt = rl.VanillaOption(("put", 100.0), maturity=1.0)
    opt.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=1)); ref = opt.NPV()
    opt.setPricingEngine(rl.SwitchingFDEngine(model, regime=1, n=1601, steps=600))
    assert opt.NPV() == pytest.approx(ref, rel=3e-4)
    am = rl.VanillaOption(("put", 100.0), exercise="american", maturity=1.0)
    am.setPricingEngine(rl.SwitchingFDEngine(model, regime=1, n=1601, steps=600))
    assert am.NPV() > ref
    # switching a knock-in plus knock-out reproduces the vanilla exactly, by construction of the knock-in
    ko = rl.BarrierOption("downout", 80.0, 0.0, ("put", 100.0), maturity=1.0); ki = rl.BarrierOption("downin", 80.0, 0.0, ("put", 100.0), maturity=1.0)
    for x in (ko, ki):
        x.setPricingEngine(rl.SwitchingFDEngine(model, regime=1, n=1601, steps=600))
    assert ko.NPV() + ki.NPV() == pytest.approx(opt.NPV(), rel=1e-6)


def test_switching_barrier_matches_monte_carlo():
    """Knock-out put with switching volatility against Monte Carlo with exact regime paths and a Brownian-bridge
    crossing probability between steps."""
    import math
    import numpy as np
    chain = rl.RegimeChain.twoState(6.0, 4.0); model = rl.SwitchingBlackScholesProcess(chain, 100.0, 0.03, 0.0, [0.35, 0.15])
    ko = rl.BarrierOption("downout", 80.0, 0.0, ("put", 100.0), maturity=1.0)
    ko.setPricingEngine(rl.SwitchingFDEngine(model, regime=1, n=1601, steps=600)); fd = ko.NPV()
    rng = np.random.default_rng(1); N, M = 100000, 1000; dt = 1.0 / M; sig = np.array([0.35, 0.15]); Q = chain.generator
    S = np.full(N, 100.0); reg = np.ones(N, int); alive = np.ones(N, bool); logB = math.log(80.0)
    for _ in range(M):
        s = sig[reg]; Snew = S * np.exp((0.03 - 0.5 * s * s) * dt + s * math.sqrt(dt) * rng.standard_normal(N))
        p = np.exp(-2 * (np.log(S) - logB) * (np.log(Snew) - logB) / (s * s * dt))
        alive &= ~((Snew <= 80.0) | (rng.random(N) < p)); S = Snew
        reg = np.where(rng.random(N) < -np.diag(Q)[reg] * dt, 1 - reg, reg)
    pay = np.where(alive, np.maximum(100.0 - S, 0.0), 0.0) * math.exp(-0.03)
    assert abs(fd - pay.mean()) < 3 * pay.std() / math.sqrt(N) + 2e-3
