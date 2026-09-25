"""Credit default swaps on a switching CIR intensity: frozen limit against QuantLib's MidPointCdsEngine with the
survival curve taken from the CIR bond formula; with switching, the fair spread sits between the regimes' own."""
import math
import pytest
import QuantLib as ql
import regimelib as rl

REF = ql.Date(1, 1, 2020)


def test_cds_matches_midpoint_engine():
    ql.Settings.instance().evaluationDate = REF
    dc, cal = ql.Actual365Fixed(), ql.NullCalendar()
    r, spread, recovery = 0.03, 0.02, 0.4
    model = rl.SwitchingCoxIngersollRoss(rl.RegimeChain.twoState(3.0, 5.0), 0.02, 0.03, 0.5, 0.08)
    engine = rl.NumericalSwitchingEngine(model)
    times = [0.5 * i for i in range(1, 11)]
    dates = [REF + ql.Period(int(round(t * 365)), ql.Days) for t in times]
    # QuantLib survival curve from the same (frozen) CIR survival probabilities, on a fine grid so that its
    # log-linear interpolation does not enter
    fine = [i / 48 for i in range(1, 241)]
    probs = [1.0]
    for t in fine:
        b = rl.ZeroCouponBond(t); b.setPricingEngine(engine); probs.append(b.NPV())
    fineDates = [REF + ql.Period(int(round(t * 365)), ql.Days) for t in fine]
    curve = ql.DefaultProbabilityTermStructureHandle(ql.SurvivalProbabilityCurve([REF] + fineDates, probs, dc, cal))
    disc = ql.YieldTermStructureHandle(ql.FlatForward(REF, r, dc))
    sched = ql.Schedule(REF, dates[-1], ql.Period(6, ql.Months), cal, ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False)
    q = ql.CreditDefaultSwap(ql.Protection.Buyer, 1.0, spread, sched, ql.Unadjusted, dc, True, True)
    q.setPricingEngine(ql.MidPointCdsEngine(curve, recovery, disc))
    ours = rl.CreditDefaultSwap("buyer", spread, times, recovery, discount=r); ours.setPricingEngine(engine)
    print("cds", ours.couponLegNPV(), q.couponLegNPV(), ours.defaultLegNPV(), q.defaultLegNPV(), ours.fairSpread(), q.fairSpread())
    assert ours.couponLegNPV() == pytest.approx(q.couponLegNPV(), rel=2e-4)
    assert ours.defaultLegNPV() == pytest.approx(q.defaultLegNPV(), rel=2e-4)
    assert ours.fairSpread() == pytest.approx(q.fairSpread(), rel=2e-4)


def test_switching_intensity_fair_spread_between_regimes():
    times = [0.5 * i for i in range(1, 11)]
    def fair(model, regime=0):
        c = rl.CreditDefaultSwap("buyer", 0.01, times, 0.4, discount=0.03)
        c.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=regime)); return c.fairSpread()
    lo = fair(rl.SwitchingVasicek(rl.RegimeChain.twoState(1.0, 1.0), 0.01, 0.5, 0.01, 0.002))
    hi = fair(rl.SwitchingVasicek(rl.RegimeChain.twoState(1.0, 1.0), 0.01, 0.5, 0.05, 0.002))
    sw = rl.SwitchingVasicek(rl.RegimeChain.twoState(1.0, 1.0), 0.01, 0.5, [0.01, 0.05], 0.002)
    s0, s1 = fair(sw, 0), fair(sw, 1)
    assert lo < s0 < s1 < hi
