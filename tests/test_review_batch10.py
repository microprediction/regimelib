"""The settled-exercise bound with discounting and scale, grids at zero reversion, inputs that are not numbers, and
endpoints: the review of 7 October."""
import math
import warnings

import numpy as np
import pytest

import regimelib as rl

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)


def test_settled_allows_for_discounting_and_for_the_size_of_the_terms():
    from regimelib._engine.options import settled
    assert settled([2.0], 0.0, 0.0, 0.01, 0.0) == [1.0]                     # twenty standard deviations above
    assert settled([2.0], 0.0, 0.0, 0.01, 0.0, drift=1.0) == [None]         # unless discounting can move the mean there
    assert settled([-2.0], 0.0, 0.0, 0.01, 0.0, drift=1.0) == [None]
    assert settled([2.0], 0.0, 0.0, 0.01, 0.0, scale=1e60) == [None]        # or the term is so large that its tail counts
    assert settled([-9.0], 0.0, 0.0, 0.01, 0.0, scale=1e60) == [0.0]


def test_long_dated_bond_option_with_a_large_discounting_covariance_is_not_settled_to_zero():
    # the exercise boundary is far from the undiscounted mean and near the mean under the measures that price
    model = rl.SwitchingVasicek(CHAIN, r0=0.03, a=0.05, b=[0.03, 0.03], sigma=[0.03, 0.03])
    T, S = 30.0, 30.5
    engine = rl.NumericalSwitchingEngine(model, information="observed")
    bond = lambda t: (lambda b: (b.setPricingEngine(engine), b.NPV())[1])(rl.ZeroCouponBond(t))
    option = rl.ZeroCouponBondOption("call", bond(S) / bond(T), T, S); option.setPricingEngine(engine)
    assert option.NPV() > 1e-4 * bond(T)                                    # at the money forward: time value, not zero


def test_grids_at_zero_reversion():
    model = rl.SwitchingVasicek(CHAIN, 0.03, 0.0, [0.05, 0.02], [0.01, 0.02])
    option = rl.ZeroCouponBondOption("call", 0.93, 1.0, 3.0)
    option.setPricingEngine(rl.NumericalSwitchingEngine(model, information="observed")); exact = option.NPV()
    option.setPricingEngine(rl.SwitchingFDEngine(model, information="observed"))
    assert option.NPV() == pytest.approx(exact, rel=2e-3)
    cir = rl.SwitchingCoxIngersollRoss(CHAIN, r0=0.03, theta=[0.05, 0.02], k=0.0, sigma=0.1)
    assert math.isfinite(cir.operators(rl.ZeroCouponBond(2.0), 101, None)[3].x[-1])
    hybrid = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.0, 0.0, [0.2, 0.3]), model)
    call = rl.VanillaOption(("call", 100.0), maturity=1.0)
    call.setPricingEngine(rl.NumericalSwitchingEngine(hybrid)); exact = call.NPV()
    call.setPricingEngine(rl.SwitchingFDEngine(hybrid, n=(161, 41), steps=100))
    assert call.NPV() == pytest.approx(exact, rel=5e-3)


def test_inputs_that_are_not_numbers_are_refused():
    for build in (rl.SwitchingMerton76Process, rl.SwitchingBatesModel):
        extra = dict(v0=0.04, kappa=1.0, theta=0.04, rho=-0.5) if build is rl.SwitchingBatesModel else {}
        with pytest.raises(ValueError, match="logJumpMean"):
            build(CHAIN, S0=100.0, r=0.0, q=0.0, sigma=0.2, jumpIntensity=1.0, logJumpMean=math.nan, logJumpVol=0.1, **extra)
    with pytest.raises(ValueError, match="finite"):
        rl.CouponBond(cashflows=[(1.0, 5.0), (2.0, math.inf)])
    with pytest.raises(ValueError, match="nonnegative"):
        rl.CouponBondOption("call", 0.9, 1.0, [(2.0, -0.5), (3.0, 1.5)])


def test_geometric_asian_does_not_depend_on_the_unit_of_price():
    def value(s):
        model = rl.SwitchingBlackScholesProcess(CHAIN, S0=100.0 * s, r=0.02, q=0.0, sigma=[0.2, 0.3])
        option = rl.ContinuousGeometricAsianOption(("call", 100.0 * s), maturity=1.0)
        option.setPricingEngine(rl.NumericalSwitchingEngine(model)); return option.NPV() / s
    assert value(1e200) == pytest.approx(value(1.0), rel=1e-9) and value(1e-200) == pytest.approx(value(1.0), rel=1e-9)


def test_hybrid_theta_at_zero_maturity_uses_the_current_rate():
    rates = rl.SwitchingVasicek(CHAIN, 0.04, 0.5, [0.06, 0.02], [0.015, 0.008])
    hybrid = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.0, 0.01, [0.3, 0.15]), rates)
    option = rl.VanillaOption(("call", 90.0), maturity=0.0); option.setPricingEngine(rl.NumericalSwitchingEngine(hybrid))
    assert option.theta() == pytest.approx(0.01 * 100.0 - 0.04 * 90.0)


def test_american_knock_out_respects_a_deferred_exercise_start():
    ql = pytest.importorskip("QuantLib")
    today = ql.Date(1, 1, 2026); ql.Settings.instance().evaluationDate = today
    model = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.05, 0.0, [0.2, 0.3])
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, 100.0)
    def value(start):
        exercise = ql.AmericanExercise(start, today + ql.Period(1, ql.Years))
        option = rl.BarrierOption(ql.Barrier.DownOut, 70.0, 0.0, payoff, exercise, dayCounter=ql.Actual365Fixed())
        option.setPricingEngine(rl.SwitchingFDEngine(model, n=201, steps=200)); return option.NPV()
    european = rl.BarrierOption("downout", 70.0, 0.0, ("put", 100.0), maturity=1.0)
    european.setPricingEngine(rl.SwitchingFDEngine(model, n=201, steps=200))
    immediate, deferred = value(today), value(today + ql.Period(9, ql.Months))
    assert european.NPV() < deferred < immediate


def test_monte_carlo_terminal_regime_is_the_one_held_at_maturity():
    model = rl.SwitchingVasicek(rl.RegimeChain.twoState(1e-12, 1e-12), 0.0, 0.5, [0.0, 0.0], [0.0, 0.0])
    engine = rl.MonteCarloSwitchingEngine(model, paths=200, seed=3)
    for states, durations in engine._paths(0.1 + 0.2):                    # a maturity that is not a sum of halves
        assert len(states) == 1 and durations.sum() == pytest.approx(0.1 + 0.2)


def test_fixed_order_does_not_re_enter_a_series_past_its_smallest_term():
    model = rl.SwitchingVasicek(rl.RegimeChain.twoState(0.3, 0.3), r0=0.0, a=0.25, b=[0.20, 0.25], sigma=[0.02, 0.25])
    bond = rl.ZeroCouponBond(2.0)
    bond.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=1)); exact = bond.NPV()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rl.ExpansionWarning)
        for order in (2, 3, 4, 5, 6):                                    # the second term grew: every order from there solves
            bond.setPricingEngine(rl.FastSwitchingEngine(model, order=order, regime=1))
            assert bond.NPV() == pytest.approx(exact, rel=1e-9)


def test_convection_dominated_grid_is_refused():
    model = rl.SwitchingBlackScholesProcess(rl.RegimeChain([[0.0]]), S0=100.0, r=0.05, q=0.0, sigma=[3e-4])
    option = rl.VanillaOption(("call", 100.0 * math.exp(0.05)), maturity=1.0)
    for engine in (rl.FirstOrderFDEngine(model), rl.SwitchingFDEngine(model)):
        option.setPricingEngine(engine)
        with pytest.raises(ValueError, match="Peclet"):
            option.NPV()


def test_g2_factors_that_nearly_cancel_are_refused_and_exact_cancellation_is_priced():
    T, S, r = 1.0, 3.0, 0.03
    option = rl.ZeroCouponBondOption("call", math.exp(-r * (S - T)), T, S)
    near = rl.SwitchingG2(CHAIN, r, 0.5, 0.02, 0.5, 0.02, -1.0 + 5e-13)
    option.setPricingEngine(rl.NumericalSwitchingEngine(near))
    with pytest.raises(NotImplementedError, match="nearly cancel"):
        option.NPV()
    option.setPricingEngine(rl.NumericalSwitchingEngine(rl.SwitchingG2(CHAIN, r, 0.5, 0.02, 0.5, 0.02, -1.0)))
    assert option.NPV() == pytest.approx(0.0, abs=1e-12)


def test_g2_regime_is_revealed_by_the_covariance_not_by_its_parameters():
    from regimelib.information import revealed
    same = rl.SwitchingG2(CHAIN, 0.03, 0.5, [0.0, 0.0], 0.1, [0.01, 0.01], [0.3, -0.4])     # rho is idle when sigma = 0
    different = rl.SwitchingG2(CHAIN, 0.03, 0.5, [0.01, 0.02], 0.1, [0.01, 0.01], [0.3, 0.3])
    assert not revealed(same) and revealed(different)


def test_replacing_a_callable_discount_curve_invalidates_the_results():
    model = rl.SwitchingVasicek(CHAIN, 0.02, 0.5, [0.03, 0.01], [0.005, 0.005])
    cds = rl.CreditDefaultSwap("buyer", 0.01, [1.0, 2.0], 0.4, discount=lambda t: math.exp(-0.01 * t))
    cds.setPricingEngine(rl.NumericalSwitchingEngine(model)); first = cds.fairSpread(), cds._result("annuity")
    cds.discount = lambda t: math.exp(-0.20 * t)
    assert cds._result("annuity") < first[1] - 1e-3


def test_bond_options_on_a_chain_with_several_closed_classes_are_refused_by_the_transform_engines():
    chain = rl.RegimeChain([[0.0, 0.0], [0.0, 0.0]])
    model = rl.SwitchingVasicek(chain, 0.03, 0.5, [0.05, 0.02], [0.01, 0.02])
    option = rl.ZeroCouponBondOption("call", 0.9, 1.0, 3.0)
    option.setPricingEngine(rl.NumericalSwitchingEngine(model, information="observed"))
    with pytest.raises(NotImplementedError, match="closed classes"):
        option.NPV()
