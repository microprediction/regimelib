"""Any number of regimes: a four-state cycle (complex eigenvalues) and a three-state chain with unequal rates."""
import numpy as np
import pytest
import QuantLib as ql
import regimelib as rl

CYCLE = rl.RegimeChain([[-12, 12, 0, 0], [0, -8, 8, 0], [0, 0, -15, 15], [10, 0, 0, -10]])
THREE = rl.RegimeChain([[-5, 3, 2], [4, -9, 5], [1, 6, -7]])


def test_four_regime_cycle_vasicek():
    model = rl.SwitchingVasicek(CYCLE, 0.03, 0.5, [0.08, 0.05, 0.02, 0.04], [0.015, 0.01, 0.006, 0.012])
    bond = rl.ZeroCouponBond(5.0)
    bond.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=2)); ref = bond.NPV()
    errs = []
    for o in (0, 1, 2, 4, 6):
        bond.setPricingEngine(rl.FastSwitchingEngine(model, order=o, regime=2)); errs.append(abs(bond.NPV() - ref))
    assert errs[0] > errs[1] > errs[2] > errs[3] > errs[4] and errs[4] < 1e-9
    mc = rl.MonteCarloSwitchingEngine(model, regime=2, paths=30000, seed=3); bond.setPricingEngine(mc)
    assert abs(bond.NPV() - ref) < 4 * mc.standardError


def test_three_regime_heston_frozen_and_switching():
    ref_date = ql.Date(1, 1, 2020); ql.Settings.instance().evaluationDate = ref_date
    S0, r, q, v0, kappa, xi, rho, K, T = 100.0, 0.02, 0.0, 0.04, 1.5, 0.4, -0.6, 100.0, 1.0
    dc = ql.Actual365Fixed()
    rts = ql.YieldTermStructureHandle(ql.FlatForward(ref_date, r, dc)); qts = ql.YieldTermStructureHandle(ql.FlatForward(ref_date, q, dc))
    payoff, ex = ql.PlainVanillaPayoff(ql.Option.Call, K), ql.EuropeanExercise(ref_date + ql.Period(365, ql.Days))
    qopt = ql.VanillaOption(payoff, ex)
    qopt.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(ql.HestonProcess(rts, qts, ql.QuoteHandle(ql.SimpleQuote(S0)), v0, kappa, 0.05, xi, rho))))
    frozen = rl.SwitchingHestonModel(THREE, S0, r, q, v0, kappa, 0.05, xi, rho)
    ours = rl.VanillaOption(payoff, ex, maturity=T); ours.setPricingEngine(rl.FastSwitchingEngine(frozen, order=2, regime=1))
    assert ours.NPV() == pytest.approx(qopt.NPV(), rel=1e-7)
    model = rl.SwitchingHestonModel(THREE, S0, r, q, v0, kappa, [0.09, 0.05, 0.02], xi, rho)
    ours.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=1)); ref = ours.NPV()
    errs = []
    for o in (0, 1, 2, 4):
        ours.setPricingEngine(rl.FastSwitchingEngine(model, order=o, regime=1)); errs.append(abs(ours.NPV() - ref))
    assert errs[0] > errs[1] > errs[2] > errs[3] and errs[3] < 1e-4 * ref
