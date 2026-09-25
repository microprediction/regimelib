"""Continuous geometric-average Asian options: frozen limit against QuantLib's Kemna-Vorst engine, the expansion
against the numerical solution with switching, and both against Monte Carlo."""
import math
import numpy as np
import pytest
import QuantLib as ql
import regimelib as rl

REF = ql.Date(1, 1, 2020)


def test_geometric_asian_matches_kemna_vorst():
    S0, r, q, sigma, K = 100.0, 0.05, 0.02, 0.25, 100.0
    ql.Settings.instance().evaluationDate = REF; dc = ql.Actual365Fixed()
    proc = ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(S0)), ql.YieldTermStructureHandle(ql.FlatForward(REF, q, dc)),
                                        ql.YieldTermStructureHandle(ql.FlatForward(REF, r, dc)),
                                        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(REF, ql.NullCalendar(), sigma, dc)))
    expiry = REF + ql.Period(365, ql.Days); ex = ql.EuropeanExercise(expiry)
    model = rl.SwitchingBlackScholesProcess(rl.RegimeChain.twoState(3.0, 5.0), S0, r, q, sigma)
    for kind in (ql.Option.Call, ql.Option.Put):
        payoff = ql.PlainVanillaPayoff(kind, K)
        o = ql.ContinuousAveragingAsianOption(ql.Average.Geometric, payoff, ex)
        o.setPricingEngine(ql.AnalyticContinuousGeometricAveragePriceAsianEngine(proc))
        ours = rl.ContinuousGeometricAsianOption(payoff, ex)
        ours.setPricingEngine(rl.NumericalSwitchingEngine(model)); assert ours.NPV() == pytest.approx(o.NPV(), rel=1e-8)
        ours.setPricingEngine(rl.FastSwitchingEngine(model, order=2)); assert ours.NPV() == pytest.approx(o.NPV(), rel=1e-8)


def test_geometric_asian_switching_expansion_and_monte_carlo():
    chain = rl.RegimeChain.twoState(12.0, 8.0)
    model = rl.SwitchingBlackScholesProcess(chain, 100.0, 0.03, 0.0, [0.35, 0.15])
    o = rl.ContinuousGeometricAsianOption(("call", 100.0), maturity=1.0)
    o.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0)); ref = o.NPV()
    errs = []
    for order in (0, 2, 4):
        o.setPricingEngine(rl.FastSwitchingEngine(model, order=order, regime=0)); errs.append(abs(o.NPV() - ref))
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-5 * ref
    rng = np.random.default_rng(3); N, M = 100000, 500; dt = 1.0 / M; sig = np.array([0.35, 0.15]); Q = chain.generator
    X = np.zeros(N); reg = np.zeros(N, int); Y = np.zeros(N)
    for _ in range(M):
        s = sig[reg]; X += (0.03 - 0.5 * s * s) * dt + s * math.sqrt(dt) * rng.standard_normal(N)
        Y += X * dt                                                  # right-endpoint sum; O(dt) bias is below the tolerance
        reg = np.where(rng.random(N) < -np.diag(Q)[reg] * dt, 1 - reg, reg)
    pay = np.maximum(100.0 * np.exp(Y) - 100.0, 0.0) * math.exp(-0.03)
    assert abs(ref - pay.mean()) < 3 * pay.std() / math.sqrt(N) + 0.02
