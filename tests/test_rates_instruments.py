"""Swaptions, caps and coupon-bond options: frozen limit against QuantLib's Jamshidian and analytic cap engines,
and convergence of the expansion with switching on."""
import math
import pytest
import QuantLib as ql
import regimelib as rl

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)
REF = ql.Date(1, 1, 2020)


def _ql_swaption(model_ql, ts, kind, expiryYears, tenorYears, K):
    ql.Settings.instance().evaluationDate = REF
    cal, dc = ql.NullCalendar(), ql.Actual365Fixed()
    index = ql.IborIndex("idx", ql.Period(1, ql.Years), 0, ql.USDCurrency(), cal, ql.Unadjusted, False, dc, ts)
    start = REF + ql.Period(expiryYears, ql.Years); end = start + ql.Period(tenorYears, ql.Years)
    sched = ql.Schedule(start, end, ql.Period(1, ql.Years), cal, ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False)
    swap = ql.VanillaSwap(ql.VanillaSwap.Payer if kind == "payer" else ql.VanillaSwap.Receiver, 1.0, sched, K, dc, sched, index, 0.0, dc)
    swaption = ql.Swaption(swap, ql.EuropeanExercise(start))
    swaption.setPricingEngine(ql.JamshidianSwaptionEngine(model_ql, ts))
    fixed_times = [dc.yearFraction(REF, d) for d in list(sched)[1:]]
    return swaption.NPV(), dc.yearFraction(REF, start), fixed_times


def test_swaption_matches_jamshidian_hull_white():
    r, a, s, K = 0.03, 0.5, 0.012, 0.03
    ts = ql.YieldTermStructureHandle(ql.FlatForward(REF, r, ql.Actual365Fixed()))
    hw = ql.HullWhite(ts, a, s); model = rl.SwitchingHullWhite(CHAIN, ts, a, s)
    for kind in ("payer", "receiver"):
        expected, T, times = _ql_swaption(hw, ts, kind, 2, 5, K)
        sw = rl.Swaption(kind, T, times, K, notional=1.0); sw.setPricingEngine(rl.FastSwitchingEngine(model, order=2))
        assert sw.NPV() == pytest.approx(expected, rel=2e-6)


def test_cap_matches_analytic_cap_floor_engine():
    r, a, s, K = 0.03, 0.5, 0.012, 0.03
    ql.Settings.instance().evaluationDate = REF
    cal, dc = ql.NullCalendar(), ql.Actual365Fixed()
    ts = ql.YieldTermStructureHandle(ql.FlatForward(REF, r, dc)); hw = ql.HullWhite(ts, a, s)
    index = ql.IborIndex("idx", ql.Period(1, ql.Years), 0, ql.USDCurrency(), cal, ql.Unadjusted, False, dc, ts)
    sched = ql.Schedule(REF + ql.Period(1, ql.Years), REF + ql.Period(5, ql.Years), ql.Period(1, ql.Years), cal, ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False)
    leg = ql.IborLeg([1.0], sched, index, dc)
    for kind, qcls in (("cap", ql.Cap), ("floor", ql.Floor)):
        q = qcls(leg, [K]); q.setPricingEngine(ql.AnalyticCapFloorEngine(hw, ts))
        times = [dc.yearFraction(REF, d) for d in list(sched)]
        ours = rl.CapFloor(kind, times, K, notional=1.0); ours.setPricingEngine(rl.FastSwitchingEngine(rl.SwitchingHullWhite(CHAIN, ts, a, s), order=2))
        assert ours.NPV() == pytest.approx(q.NPV(), rel=2e-6)


def test_swaption_switching_converges():
    chain = rl.RegimeChain.twoState(20.0, 30.0)
    model = rl.SwitchingVasicek(chain, 0.03, 0.5, [0.06, 0.02], [0.015, 0.008])
    sw = rl.Swaption("payer", 2.0, [3.0, 4.0, 5.0, 6.0, 7.0], 0.035, notional=100.0)
    sw.setPricingEngine(rl.NumericalSwitchingEngine(model)); ref = sw.NPV()
    errs = []
    for o in (0, 1, 2):
        sw.setPricingEngine(rl.FastSwitchingEngine(model, order=o)); errs.append(abs(sw.NPV() - ref))
    assert errs[0] > errs[1] > errs[2] and ref > 0
