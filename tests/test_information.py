"""What is known about the regime: a starting belief, and observed against inferred information.

A payoff of the observed state is linear in the belief. Options on bonds and swaps, and early exercise, are not: with
an inferred regime the belief is a state variable, checked here on a claim whose value is known exactly (a zero-strike
call on a bond is the bond), in the limit of no switching, and against the observed-regime price, which bounds it."""
import warnings
import numpy as np
import pytest
import regimelib as rl
from regimelib.information import revealed, identical

CHAIN = rl.RegimeChain.twoState(2.0, 3.0)
BELIEF = [0.3, 0.7]
LEVEL = rl.SwitchingVasicek(CHAIN, 0.03, 0.5, [0.07, 0.02], 0.01)                  # only the level switches
BOTH = rl.SwitchingVasicek(CHAIN, 0.03, 0.5, [0.07, 0.02], [0.015, 0.008])         # the volatility switches too
CIR = rl.SwitchingCoxIngersollRoss(CHAIN, 0.03, [0.07, 0.02], 0.5, 0.08)
FIXED = [2.0, 3.0, 4.0, 5.0, 6.0]


def mixture(make, price):
    """sum_i p_i price(engine started in regime i)."""
    return sum(p * price(make(i)) for i, p in enumerate(BELIEF))


def test_which_models_reveal_their_regime():
    assert revealed(BOTH) and not revealed(LEVEL) and not revealed(CIR)
    assert identical(rl.SwitchingVasicek(CHAIN, 0.03, 0.5, 0.05, 0.01)) and not identical(LEVEL)
    three = rl.RegimeChain([[-2.0, 1.0, 1.0], [1.0, -2.0, 1.0], [1.0, 1.0, -2.0]])
    assert not revealed(rl.SwitchingBlackScholesProcess(three, 100.0, 0.02, 0.0, [0.2, 0.2, 0.4]))    # two regimes look alike
    assert revealed(rl.SwitchingBlackScholesProcess(three, 100.0, 0.02, 0.0, [0.2, 0.3, 0.4]))


@pytest.mark.parametrize("bad", [[0.5, 0.6], [1.2, -0.2], [1.0], [0.2, 0.3, 0.5], [float("nan"), 1.0], 0.5, "0"])
def test_a_belief_is_a_probability_per_regime(bad):
    with pytest.raises(ValueError):
        rl.NumericalSwitchingEngine(LEVEL, regime=bad)


def test_payoffs_of_the_observed_state_average_over_the_belief():
    def bond(engine):
        b = rl.ZeroCouponBond(4.0); b.setPricingEngine(engine); return b.NPV()
    for make in (lambda r: rl.NumericalSwitchingEngine(LEVEL, regime=r), lambda r: rl.FastSwitchingEngine(LEVEL, order=2, regime=r)):
        assert bond(make(BELIEF)) == pytest.approx(mixture(make, bond), rel=1e-13)
    bs = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.03, 0.0, [0.3, 0.15])

    def call(engine, greek=False):
        o = rl.VanillaOption(("call", 105.0), maturity=1.0); o.setPricingEngine(engine); return o.delta() if greek else o.NPV()
    for make in (lambda r: rl.NumericalSwitchingEngine(bs, regime=r), lambda r: rl.SwitchingFDEngine(bs, regime=r, n=201, steps=50),
                 lambda r: rl.SwitchingFDReferee(bs, regime=r, n=201)):
        assert call(make(BELIEF)) == pytest.approx(mixture(make, call), rel=1e-12)
    price, delta = call, (lambda e: call(e, greek=True))
    make = lambda r: rl.NumericalSwitchingEngine(bs, regime=r)
    assert delta(make(BELIEF)) == pytest.approx(mixture(make, delta), rel=1e-12)
    assert price(make([1.0, 0.0])) == pytest.approx(price(make(0)), rel=1e-14)      # a certain belief is the regime


def test_first_order_and_monte_carlo_engines_take_a_belief():
    cev = rl.SwitchingCEVProcess(CHAIN, 100.0, 0.02, 0.0, [2.5, 1.2], 0.6)
    option = rl.VanillaOption(("call", 100.0), maturity=1.0)
    parts = []
    for r in (0, 1, BELIEF):
        engine = rl.FirstOrderFDEngine(cev, regime=r, n=201)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); option.setPricingEngine(engine); parts.append((option.NPV(), engine.memory))
    assert parts[2][0] == pytest.approx(0.3 * parts[0][0] + 0.7 * parts[1][0], rel=1e-12)
    assert parts[2][1] == pytest.approx(0.3 * parts[0][1] + 0.7 * parts[1][1], rel=1e-12)
    bond = rl.ZeroCouponBond(2.0)
    mc = rl.MonteCarloSwitchingEngine(LEVEL, regime=BELIEF, paths=4000, seed=1); bond.setPricingEngine(mc); value = bond.NPV()
    bond.setPricingEngine(rl.NumericalSwitchingEngine(LEVEL, regime=BELIEF))
    assert abs(value - bond.NPV()) < 4 * mc.standardError and mc.standardError > 0


def test_cds_fair_spread_is_a_ratio_of_averaged_legs():
    times = [0.5 * k for k in range(1, 9)]

    def legs(engine):
        c = rl.CreditDefaultSwap("buyer", 0.01, times, 0.4, discount=0.03); c.setPricingEngine(engine)
        return c.defaultLegNPV(), -c.couponLegNPV() / 0.01, c.fairSpread()
    make = lambda r: rl.NumericalSwitchingEngine(CIR, regime=r)
    prot = mixture(make, lambda e: legs(e)[0]); annuity = mixture(make, lambda e: legs(e)[1])
    assert legs(make(BELIEF))[2] == pytest.approx(prot / annuity, rel=1e-12)
    assert legs(make(BELIEF))[2] != pytest.approx(mixture(make, lambda e: legs(e)[2]), rel=1e-6)


def test_characteristic_function_engines_price_rate_options_only_when_the_regime_is_known():
    swaption = rl.Swaption("payer", 1.0, FIXED, 0.045)
    cap = rl.CapFloor("cap", [0.5, 1.0, 1.5], 0.04)
    for instrument in (swaption, cap, rl.ZeroCouponBondOption("call", 0.9, 1.0, 3.0)):
        instrument.setPricingEngine(rl.NumericalSwitchingEngine(LEVEL))
        with pytest.raises(NotImplementedError, match="does not reveal"):
            instrument.NPV()
        instrument.setPricingEngine(rl.NumericalSwitchingEngine(LEVEL, information="observed"))
        assert instrument.NPV() > 0
        instrument.setPricingEngine(rl.NumericalSwitchingEngine(BOTH))             # the volatility reveals the regime
        assert instrument.NPV() > 0
    bond = rl.ZeroCouponBond(3.0); bond.setPricingEngine(rl.NumericalSwitchingEngine(LEVEL))
    assert bond.NPV() > 0                                                          # a bond is linear: no information needed
    with pytest.raises(ValueError, match="information"):
        rl.NumericalSwitchingEngine(LEVEL, information="hidden")


@pytest.mark.parametrize("model", [LEVEL, CIR])
def test_inferred_grid_reproduces_the_bond(model):
    """A zero-strike call on a bond is the bond, whose value from a belief is exact; this exercises the belief's
    dynamics with a switching level."""
    for belief in (0, 1, BELIEF):
        option = rl.CouponBondOption("call", 0.0, 1.0, [(3.0, 1.0)])
        option.setPricingEngine(rl.SwitchingFDEngine(model, regime=belief, n=(161, 31), steps=120))
        bond = rl.ZeroCouponBond(3.0); bond.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=belief))
        assert option.NPV() == pytest.approx(bond.NPV(), rel=5e-6)


def test_inferred_grid_without_switching_is_the_one_regime_model():
    still = rl.RegimeChain([[0.0, 0.0], [0.0, 0.0]])                               # the regime never changes
    model = rl.SwitchingVasicek(still, 0.04, 0.5, [0.07, 0.02], 0.01)
    swaption = rl.Swaption("payer", 1.0, FIXED, 0.045, notional=100.0)
    for regime, level in ((0, 0.07), (1, 0.02)):
        swaption.setPricingEngine(rl.SwitchingFDEngine(model, regime=regime, n=(201, 21), steps=200)); value = swaption.NPV()
        one = rl.SwitchingVasicek(rl.RegimeChain([[0.0]]), 0.04, 0.5, level, 0.01)
        swaption.setPricingEngine(rl.NumericalSwitchingEngine(one))
        assert value == pytest.approx(swaption.NPV(), rel=2e-4, abs=1e-6)


@pytest.mark.parametrize("model", [LEVEL, CIR])
def test_inferred_price_is_below_observed_and_bermudan_above_european(model):
    def price(information, exercise=None):
        s = rl.Swaption("payer", 1.0, FIXED, 0.045, notional=100.0, exerciseTimes=exercise)
        n = (161, 31) if information == "inferred" else 301
        s.setPricingEngine(rl.SwitchingFDEngine(model, regime=BELIEF, n=n, steps=120, information=information)); return s.NPV()
    european, bermudan = price("inferred"), price("inferred", [1.0, 2.0, 3.0, 4.0])
    assert 0 < european < 0.995 * price("observed")                                # acting on a belief is worth less
    assert bermudan > european and bermudan < price("observed", [1.0, 2.0, 3.0, 4.0])


def test_inferred_cap_and_bond_option_on_the_grid():
    for instrument in (rl.CapFloor("cap", [0.5, 1.0, 1.5], 0.04, notional=100.0), rl.ZeroCouponBondOption("put", 0.9, 1.0, 3.0)):
        instrument.setPricingEngine(rl.SwitchingFDEngine(LEVEL, regime=BELIEF, n=(161, 31), steps=120)); inferred = instrument.NPV()
        instrument.setPricingEngine(rl.SwitchingFDEngine(LEVEL, regime=BELIEF, n=301, steps=120, information="observed"))
        assert 0 < inferred < instrument.NPV()
        instrument.setPricingEngine(rl.NumericalSwitchingEngine(LEVEL, regime=BELIEF, information="observed"))
        observed = instrument.NPV()
        instrument.setPricingEngine(rl.SwitchingFDEngine(LEVEL, regime=BELIEF, n=301, steps=120, information="observed"))
        assert instrument.NPV() == pytest.approx(observed, rel=3e-2)                # the grid and the transform agree


def test_cases_the_inferred_grid_does_not_cover_are_refused():
    three = rl.RegimeChain([[-2.0, 1.0, 1.0], [1.0, -2.0, 1.0], [1.0, 1.0, -2.0]])
    cir3 = rl.SwitchingCoxIngersollRoss(three, 0.03, [0.07, 0.04, 0.02], 0.5, 0.08)
    swaption = rl.Swaption("payer", 1.0, FIXED, 0.045)
    swaption.setPricingEngine(rl.SwitchingFDEngine(cir3, n=101, steps=40))
    with pytest.raises(NotImplementedError):
        swaption.NPV()
    swaption.setPricingEngine(rl.SwitchingFDEngine(cir3, n=101, steps=40, information="observed"))
    assert swaption.NPV() > 0
    hybrid = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.0, 0.0, 0.2), LEVEL)
    american = rl.VanillaOption(("put", 100.0), exercise="american", maturity=1.0)
    american.setPricingEngine(rl.SwitchingFDEngine(hybrid, n=(41, 21), steps=20))
    with pytest.raises(NotImplementedError, match="early exercise"):
        american.NPV()
    revealing = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.05, 0.0, [0.3, 0.15])
    american.setPricingEngine(rl.SwitchingFDEngine(revealing, regime=BELIEF, n=201, steps=50))
    assert american.NPV() > 0                                                       # the volatility reveals the regime
