"""Two names on one regime chain: the joint survival is exact against Monte Carlo, independent names factorise in
the frozen limit (first-to-default against QuantLib's mid-point engine on the product survival curve), and the sign
of the default correlation follows whether the names' intensities rise in the same regime."""
import math
import numpy as np
import pytest
import QuantLib as ql
import regimelib as rl

REF = ql.Date(1, 1, 2020)


def test_joint_survival_frozen_factorises_and_ftd_matches_quantlib():
    chain = rl.RegimeChain.twoState(3.0, 5.0)
    m1 = rl.SwitchingVasicek(chain, 0.02, 0.4, 0.03, 0.004); m2 = rl.SwitchingCoxIngersollRoss(chain, 0.03, 0.02, 0.6, 0.06)
    basket = rl.SwitchingIntensityBasket([m1, m2])
    def surv(model, t):
        b = rl.ZeroCouponBond(t); b.setPricingEngine(rl.NumericalSwitchingEngine(model)); return b.NPV()
    for t in (1.0, 3.0, 5.0):
        assert surv(basket, t) == pytest.approx(surv(m1, t) * surv(m2, t), rel=1e-10)
    assert abs(basket.defaultCorrelation(3.0)) < 1e-9
    ql.Settings.instance().evaluationDate = REF; dc, cal = ql.Actual365Fixed(), ql.NullCalendar()
    fine = [i / 48 for i in range(1, 289)]
    curve = ql.DefaultProbabilityTermStructureHandle(ql.SurvivalProbabilityCurve(
        [REF] + [REF + ql.Period(int(round(t * 365)), ql.Days) for t in fine], [1.0] + [surv(basket, t) for t in fine], dc, cal))
    disc = ql.YieldTermStructureHandle(ql.FlatForward(REF, 0.03, dc))
    sched = ql.Schedule(REF, REF + ql.Period(5, ql.Years), ql.Period(6, ql.Months), cal, ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False)
    q = ql.CreditDefaultSwap(ql.Protection.Buyer, 1.0, 0.02, sched, ql.Unadjusted, dc, True, True)
    q.setPricingEngine(ql.MidPointCdsEngine(curve, 0.4, disc))
    ftd = rl.FirstToDefaultSwap("buyer", 0.02, [0.5 * i for i in range(1, 11)], 0.4, discount=0.03)
    ftd.setPricingEngine(rl.NumericalSwitchingEngine(basket))
    assert ftd.fairSpread() == pytest.approx(q.fairSpread(), rel=2e-4)


def test_common_regime_default_correlation_sign_and_monte_carlo():
    chain = rl.RegimeChain.twoState(1.0, 1.0)
    same = rl.SwitchingIntensityBasket([rl.SwitchingVasicek(chain, 0.02, 0.5, [0.01, 0.06], 0.003),
                                        rl.SwitchingVasicek(chain, 0.02, 0.5, [0.01, 0.06], 0.003)])
    opposite = rl.SwitchingIntensityBasket([rl.SwitchingVasicek(chain, 0.02, 0.5, [0.01, 0.06], 0.003),
                                            rl.SwitchingVasicek(chain, 0.02, 0.5, [0.06, 0.01], 0.003)])
    assert same.defaultCorrelation(5.0) > 0.005 and opposite.defaultCorrelation(5.0) < -0.005     # slow chain, rare defaults
    # joint survival against Monte Carlo with exact regime paths and exact Vasicek integrals per regime segment
    b = rl.ZeroCouponBond(5.0); b.setPricingEngine(rl.NumericalSwitchingEngine(same, regime=0)); ref = b.NPV()
    rng = np.random.default_rng(11); N, M = 200000, 1000; dt = 5.0 / M; Q = chain.generator
    l1 = np.full(N, 0.02); l2 = np.full(N, 0.02); I = np.zeros(N); reg = np.zeros(N, int); levels = np.array([0.01, 0.06])
    for _ in range(M):
        th = levels[reg]
        I += 0.5 * (l1 + l2) * dt
        l1 += 0.5 * (th - l1) * dt + 0.003 * math.sqrt(dt) * rng.standard_normal(N)
        l2 += 0.5 * (th - l2) * dt + 0.003 * math.sqrt(dt) * rng.standard_normal(N)
        I += 0.5 * (l1 + l2) * dt
        reg = np.where(rng.random(N) < -np.diag(Q)[reg] * dt, 1 - reg, reg)
    pay = np.exp(-I)
    assert abs(ref - pay.mean()) < 3 * pay.std() / math.sqrt(N) + 2e-4
