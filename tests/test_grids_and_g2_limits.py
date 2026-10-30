"""Limits that used to build a zero-width grid or cancel digits: G2++ at small and zero reversion speed, grid engines
on a model with no diffusion, G2 with one factor switched off, and the jump term when the jump mean is tiny."""
import math
import warnings
import pytest
import regimelib as rl
from regimelib.firstorder import Grid1D

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)
FAST = rl.RegimeChain.twoState(30.0, 50.0)


def price(instrument, engine):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        instrument.setPricingEngine(engine)
        return instrument.NPV()


def test_g2_is_continuous_in_its_reversion_speeds():
    def model(a, b, chain=FAST):
        return rl.SwitchingG2(chain, 0.03, a, [0.015, 0.006], b, [0.012, 0.008], [-0.4, -0.7])
    bond, option = rl.ZeroCouponBond(5.0), rl.ZeroCouponBondOption("call", 0.9, 1.0, 3.0)
    limit = {"bond": price(bond, rl.NumericalSwitchingEngine(model(0.0, 0.0))), "option": price(option, rl.NumericalSwitchingEngine(model(0.0, 0.0)))}
    near = model(1e-7, 1e-7)
    assert price(bond, rl.NumericalSwitchingEngine(near)) == pytest.approx(limit["bond"], rel=1e-6)
    assert price(option, rl.NumericalSwitchingEngine(near)) == pytest.approx(limit["option"], rel=1e-5)
    for a, b in ((0.0, 0.0), (0.0, 0.1), (1e-4, 0.1), (0.5, 1e-6)):
        numerical = price(bond, rl.NumericalSwitchingEngine(model(a, b)))
        assert price(bond, rl.FastSwitchingEngine(model(a, b), order=3)) == pytest.approx(numerical, rel=1e-8)
    frozen = rl.SwitchingG2(CHAIN, 0.03, 0.0, 0.01, 0.0, 0.01, 0.0)             # two independent random walks, curve fitted
    assert price(bond, rl.NumericalSwitchingEngine(frozen)) == pytest.approx(math.exp(-0.15), rel=1e-8)


def test_grid_engine_on_a_deterministic_equity():
    model = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.05, 0.0, 0.0)
    engine = rl.SwitchingFDEngine(model, n=201, steps=50)
    F, dr = 100.0 * math.exp(0.05), math.exp(-0.05)
    assert price(rl.VanillaOption(("put", 110.0), maturity=1.0), engine) == pytest.approx(dr * (110.0 - F))
    assert price(rl.VanillaOption(("call", 110.0), maturity=1.0), engine) == 0.0
    assert price(rl.VanillaOption(("put", 110.0), exercise="american", maturity=1.0), engine) == pytest.approx(10.0)   # exercise now
    hit = math.log(103.0 / 100.0) / 0.05                                          # the path reaches 103 then
    assert price(rl.BarrierOption("upout", 103.0, 1.0, ("call", 90.0), maturity=1.0), engine) == pytest.approx(math.exp(-0.05 * hit))
    assert price(rl.BarrierOption("upin", 103.0, 1.0, ("call", 90.0), maturity=1.0), engine) == pytest.approx(dr * (F - 90.0))
    assert price(rl.BarrierOption("upout", 120.0, 1.0, ("call", 90.0), maturity=1.0), engine) == pytest.approx(dr * (F - 90.0))
    assert price(rl.BarrierOption("upin", 120.0, 1.0, ("call", 90.0), maturity=1.0), engine) == pytest.approx(dr * 1.0)   # the rebate
    flat = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.03, 0.03, 0.0)        # zero carry as well: the price never moves
    assert price(rl.VanillaOption(("call", 100.0), maturity=1.0), rl.SwitchingFDEngine(flat, n=201, steps=50)) == 0.0


def test_first_order_engines_on_deterministic_models():
    option = rl.VanillaOption(("call", 90.0), maturity=1.0)
    cev = rl.SwitchingCEVProcess(CHAIN, 100.0, 0.02, 0.02, 0.0, 0.6)             # zero volatility, zero carry
    quiet = rl.SwitchingHestonVolOfVol(CHAIN, 100.0, 0.03, 0.0, 0.0, 2.0, 0.0, [0.8, 0.2], -0.6)    # variance stays at zero
    for model, value, n in ((cev, math.exp(-0.02) * 10.0, 101), (quiet, 100.0 - 90.0 * math.exp(-0.03), (41, 21))):
        first = rl.FirstOrderFDEngine(model, n=n)
        assert price(option, first) == pytest.approx(value, rel=1e-13)
        assert (first.correction, first.memory) == (0.0, 0.0)
        assert price(option, rl.SwitchingFDReferee(model, n=n)) == pytest.approx(value, rel=1e-13)


def test_grid_engine_on_deterministic_rates():
    for model in (rl.SwitchingVasicek(CHAIN, 0.03, 0.5, 0.05, 0.0), rl.SwitchingHullWhite(CHAIN, 0.03, 0.5, 0.0),
                  rl.SwitchingG2(CHAIN, 0.03, 0.5, 0.0, 0.1, 0.0, 0.0)):
        engine = rl.SwitchingFDEngine(model, n=101, steps=40)
        bond = lambda t: price(rl.ZeroCouponBond(t), rl.NumericalSwitchingEngine(model))
        fixed, rate = [2.0, 3.0, 4.0], 0.03
        dates = [1.0, 2.0, 3.0]

        def swap(t):                                                             # payer swap entered at t, valued today
            flows = [(S, rate + (1.0 if S == fixed[-1] else 0.0)) for S in fixed if S > t]
            return bond(t) - sum(c * bond(S) for S, c in flows)
        european = price(rl.Swaption("payer", 1.0, fixed, rate), engine)
        assert european == pytest.approx(max(swap(1.0), 0.0), abs=1e-12)
        bermudan = price(rl.Swaption("payer", 1.0, fixed, rate, exerciseTimes=dates), engine)
        assert bermudan == pytest.approx(max(max(swap(t), 0.0) for t in dates), abs=1e-12)
        assert bermudan >= european


def test_g2_with_one_factor_switched_off_is_hull_white():
    hw = rl.SwitchingHullWhite(CHAIN, 0.03, 0.5, [0.012, 0.008])
    swaption = rl.Swaption("payer", 1.0, [2.0, 3.0, 4.0], 0.03, exerciseTimes=[1.0, 2.0])
    reference = price(swaption, rl.SwitchingFDEngine(hw, n=201, steps=100))
    x_only = rl.SwitchingG2(CHAIN, 0.03, 0.5, [0.012, 0.008], 0.1, 0.0, 0.0)
    y_only = rl.SwitchingG2(CHAIN, 0.03, 0.1, 0.0, 0.5, [0.012, 0.008], 0.0)
    for model, n in ((x_only, (201, 21)), (y_only, (21, 201))):          # n is (nx, ny): the live factor's count
        assert price(swaption, rl.SwitchingFDEngine(model, n=n, steps=100)) == pytest.approx(reference, rel=1e-12)


def test_a_grid_needs_a_positive_width():
    for lo, hi in ((1.0, 1.0), (2.0, 1.0), (0.0, float("inf")), (float("nan"), 1.0)):
        with pytest.raises(ValueError, match="grid"):
            Grid1D(lo, hi, 11)


def test_jump_term_keeps_the_drift_when_the_jump_mean_is_tiny():
    """With l m = c held fixed and m -> 0 the jumps become a drift of c per unit time: Vasicek with level b + c / a."""
    r0, a, b, sigma, c, T = 0.03, 0.5, 0.04, 0.01, 0.02, 3.0
    shifted = rl.SwitchingVasicek(CHAIN, r0, a, b + c / a, sigma)
    target = price(rl.ZeroCouponBond(T), rl.NumericalSwitchingEngine(shifted))
    for m in (1e-6, 1e-12, 1e-17):                                                # 1e-17 B is below the spacing of 1 + x
        jumps = rl.SwitchingVasicekJumps(CHAIN, r0, a, b, sigma, c / m, m)
        assert price(rl.ZeroCouponBond(T), rl.NumericalSwitchingEngine(jumps)) == pytest.approx(target, rel=1e-5)
    none = rl.SwitchingVasicekJumps(CHAIN, r0, a, b, sigma, 3.0, 0.0)             # jumps of size zero do nothing
    plain = rl.SwitchingVasicek(CHAIN, r0, a, b, sigma)
    assert price(rl.ZeroCouponBond(T), rl.NumericalSwitchingEngine(none)) == pytest.approx(price(rl.ZeroCouponBond(T), rl.NumericalSwitchingEngine(plain)), rel=1e-8)
