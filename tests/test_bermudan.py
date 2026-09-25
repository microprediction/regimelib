"""Bermudan swaptions on a short-rate grid: the European case against the Jamshidian formula (with switching), the
frozen Bermudan against QuantLib's tree engine on Vasicek, and Bermudan >= European with switching."""
import pytest
import QuantLib as ql
import regimelib as rl

REF = ql.Date(1, 1, 2020)


def test_fd_european_swaption_matches_jamshidian_with_switching():
    chain = rl.RegimeChain.twoState(6.0, 4.0)
    model = rl.SwitchingVasicek(chain, 0.03, 0.5, [0.06, 0.02], [0.015, 0.008])
    times = [3.0, 4.0, 5.0, 6.0, 7.0]
    for kind in ("payer", "receiver"):
        sw = rl.Swaption(kind, 2.0, times, 0.035, notional=100.0)
        sw.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=1)); ref = sw.NPV()
        sw.setPricingEngine(rl.SwitchingFDEngine(model, regime=1, n=1201, steps=400))
        assert sw.NPV() == pytest.approx(ref, rel=2e-4)


def test_bermudan_swaption_frozen_matches_tree():
    ql.Settings.instance().evaluationDate = REF
    r0, a, b, sigma, K = 0.03, 0.5, 0.04, 0.012, 0.035
    cal, dc = ql.NullCalendar(), ql.Actual365Fixed()
    vas = ql.Vasicek(r0, a, b, sigma)
    # the tree engine discounts on a curve: Vasicek's own bond prices, so that the tree and the model agree
    dates = [REF + ql.Period(i, ql.Months) for i in range(0, 121)]
    dfs = [1.0] + [vas.discountBond(0.0, dc.yearFraction(REF, d), r0) for d in dates[1:]]
    ts = ql.YieldTermStructureHandle(ql.DiscountCurve(dates, dfs, dc))
    index = ql.IborIndex("idx", ql.Period(1, ql.Years), 0, ql.USDCurrency(), cal, ql.Unadjusted, False, dc, ts)
    start = REF + ql.Period(1, ql.Years); end = start + ql.Period(5, ql.Years)
    sched = ql.Schedule(start, end, ql.Period(1, ql.Years), cal, ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False)
    swap = ql.VanillaSwap(ql.VanillaSwap.Payer, 1.0, sched, K, dc, sched, index, 0.0, dc)
    dates = list(sched)[:-1]
    swaption = ql.Swaption(swap, ql.BermudanExercise(dates))
    swaption.setPricingEngine(ql.TreeSwaptionEngine(vas, 2000, ts))          # converges to the grid value from above
    fixed_times = [dc.yearFraction(REF, d) for d in list(sched)[1:]]
    ex_times = [dc.yearFraction(REF, d) for d in dates]
    model = rl.SwitchingVasicek(rl.RegimeChain.twoState(3.0, 5.0), r0, a, b, sigma)
    ours = rl.Swaption("payer", ex_times[0], fixed_times, K, notional=1.0, exerciseTimes=ex_times)
    ours.setPricingEngine(rl.SwitchingFDEngine(model, n=1201, steps=600))
    assert ours.NPV() == pytest.approx(swaption.NPV(), rel=1e-3)
    # the QuantLib exercise object is accepted directly
    same = rl.Swaption("payer", ql.BermudanExercise(dates), fixed_times, K, notional=1.0)
    same.setPricingEngine(rl.SwitchingFDEngine(model, n=1201, steps=600))
    assert same.NPV() == pytest.approx(ours.NPV(), rel=1e-6)


def test_bermudan_dominates_european_with_switching():
    model = rl.SwitchingVasicek(rl.RegimeChain.twoState(6.0, 4.0), 0.03, 0.5, [0.06, 0.02], [0.015, 0.008])
    times = [2.0, 3.0, 4.0, 5.0, 6.0]
    eu = rl.Swaption("payer", 1.0, times, 0.035, notional=100.0)
    be = rl.Swaption("payer", 1.0, times, 0.035, notional=100.0, exerciseTimes=[1.0, 2.0, 3.0, 4.0])
    for x in (eu, be):
        x.setPricingEngine(rl.SwitchingFDEngine(model, regime=0, n=1201, steps=400))
    assert be.NPV() > eu.NPV() > 0


def test_hull_white_bermudan_frozen_matches_fd_and_european_matches_jamshidian():
    ql.Settings.instance().evaluationDate = REF
    r, a, sigma, K = 0.03, 0.5, 0.012, 0.035
    cal, dc = ql.NullCalendar(), ql.Actual365Fixed()
    ts = ql.YieldTermStructureHandle(ql.FlatForward(REF, r, dc)); hw = ql.HullWhite(ts, a, sigma)
    index = ql.IborIndex("idx", ql.Period(1, ql.Years), 0, ql.USDCurrency(), cal, ql.Unadjusted, False, dc, ts)
    start = REF + ql.Period(1, ql.Years); end = start + ql.Period(5, ql.Years)
    sched = ql.Schedule(start, end, ql.Period(1, ql.Years), cal, ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False)
    swap = ql.VanillaSwap(ql.VanillaSwap.Payer, 1.0, sched, K, dc, sched, index, 0.0, dc)
    be = ql.Swaption(swap, ql.BermudanExercise(list(sched)[:-1]))
    be.setPricingEngine(ql.FdHullWhiteSwaptionEngine(hw, 400, 400))       # the tree converges to this from above
    fixed_times = [dc.yearFraction(REF, d) for d in list(sched)[1:]]; ex_times = [dc.yearFraction(REF, d) for d in list(sched)[:-1]]
    chain = rl.RegimeChain.twoState(3.0, 5.0)
    frozen = rl.SwitchingHullWhite(chain, ts, a, sigma)
    ours = rl.Swaption("payer", ex_times[0], fixed_times, K, notional=1.0, exerciseTimes=ex_times)
    ours.setPricingEngine(rl.SwitchingFDEngine(frozen, n=1201, steps=600))
    assert ours.NPV() == pytest.approx(be.NPV(), rel=1e-4)
    # with switching: the European swaption on the grid equals the Jamshidian value, and Bermudan dominates
    switching = rl.SwitchingHullWhite(rl.RegimeChain.twoState(6.0, 4.0), ts, a, [0.02, 0.006])
    eu = rl.Swaption("payer", ex_times[0], fixed_times, K, notional=1.0)
    eu.setPricingEngine(rl.NumericalSwitchingEngine(switching, regime=1)); ref = eu.NPV()
    eu.setPricingEngine(rl.SwitchingFDEngine(switching, regime=1, n=1201, steps=400))
    assert eu.NPV() == pytest.approx(ref, rel=3e-4)
    ours.setPricingEngine(rl.SwitchingFDEngine(switching, regime=1, n=1201, steps=600))
    assert ours.NPV() > ref
