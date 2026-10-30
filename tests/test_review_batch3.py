"""The rest of the findings against the limits work: greeks next to a barrier and on stretched grids, grid sizes for
planar models, the explicit bond-option formula and the symbolic CIR bond at their ends, and inputs that are refused."""
import math
import warnings
import numpy as np
import pytest
import regimelib as rl

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)


def test_explicit_bond_option_formula_at_zero_reversion():
    from regimelib._engine.bond_option_explicit import call
    args = ([0.05, 0.03], [0.015, 0.010], 0.04, 1.0, 4.0, 0.88, 20.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        limit = call(0.0, *args)
        near = [call(k, *args) for k in (1e-3, 1e-5, 1e-8)]
    gaps = [abs(v - limit) for v in near]
    assert limit > 0 and gaps[0] > gaps[1] > gaps[2] and gaps[2] < 1e-8
    model = rl.SwitchingVasicek(rl.RegimeChain.twoState(20.0, 20.0), 0.04, 0.0, [0.05, 0.03], [0.015, 0.010])
    option = rl.ZeroCouponBondOption("call", 0.88, 1.0, 4.0); option.setPricingEngine(rl.NumericalSwitchingEngine(model))
    assert limit == pytest.approx(option.NPV(), rel=2e-3)                       # first order in a holding time of 1/20


def test_greeks_next_to_a_barrier_and_on_a_stretched_grid():
    def barrier(S0, level, n=801):
        model = rl.SwitchingBlackScholesProcess(CHAIN, S0, 0.05, 0.0, 0.25)
        option = rl.BarrierOption("downout", level, 0.0, ("call", 100.0), maturity=1.0)
        option.setPricingEngine(rl.SwitchingFDEngine(model, n=n, steps=200)); return option
    for level in (99.9, 99.0, 90.0):                                             # the first is within a cell of the spot
        option = barrier(100.0, level)
        value, delta, gamma = option.NPV(), option.delta(), option.gamma()
        bump = 0.02
        up, down = barrier(100.0 + bump, level).NPV(), barrier(100.0 - bump, level).NPV()
        assert delta == pytest.approx((up - down) / (2 * bump), rel=3e-2)
        assert math.isfinite(gamma) and math.isfinite(option.theta()) and value > 0
    model = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.05, 0.02, 0.25)      # a vanilla on a stretched grid: Black-Scholes
    d1 = (math.log(100.0 / 105.0) + (0.03 + 0.03125)) / 0.25
    N = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))
    delta = math.exp(-0.02) * N(d1)
    gamma = math.exp(-0.02) * math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi) / (100.0 * 0.25)
    for stretch in (None, 0.2):
        option = rl.VanillaOption(("call", 105.0), maturity=1.0)
        option.setPricingEngine(rl.SwitchingFDEngine(model, n=801, steps=200, stretch=stretch))
        assert option.delta() == pytest.approx(delta, rel=2e-3) and option.gamma() == pytest.approx(gamma, rel=1e-2)


def test_planar_models_get_a_planar_default_grid():
    heston = rl.SwitchingHestonVolOfVol(CHAIN, 100.0, 0.03, 0.0, 0.04, 2.0, 0.04, [0.8, 0.2], -0.6)
    line = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.03, 0.0, [0.3, 0.15])
    assert rl.FirstOrderFDEngine(heston).n == (151, 61) and rl.SwitchingFDReferee(heston).n == (151, 61)
    assert rl.FirstOrderFDEngine(line).n == 801 and rl.FirstOrderFDEngine(heston, n=(41, 21)).n == (41, 21)
    with pytest.raises(ValueError, match="nodes per regime"):
        rl.FirstOrderFDEngine(heston, n=801)
    assert rl.FirstOrderFDEngine(heston, n=101).n == 101                          # a modest scalar is still accepted


def test_symbolic_cir_bond_at_long_maturity():
    from regimelib.symbolic import CIRBondFirstOrder
    f = CIRBondFirstOrder()
    k, sigma, theta, r0, T = 2.0, 3.0, 0.05, 0.03, 400.0                          # e^{hT} overflows a double
    numbers = f.coefficients(CHAIN, k, [theta, theta])
    h = math.sqrt(k * k + 2 * sigma * sigma)
    exact = math.exp((2 * k * theta / sigma ** 2) * (math.log(2 * h / (h + k)) + (k - h) * T / 2) - 2 / (h + k) * r0)
    assert f.evaluate(f.price, r0=r0, k=k, sigma=sigma, T=T, **numbers) == pytest.approx(exact, rel=1e-9)
    ordinary = rl.SwitchingCoxIngersollRoss(CHAIN, r0, [0.06, 0.02], 0.5, 0.1)    # and it is unchanged where it worked
    numbers = f.coefficients(CHAIN, 0.5, [0.06, 0.02])
    bond = rl.ZeroCouponBond(4.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); bond.setPricingEngine(rl.FastSwitchingEngine(ordinary, order=1)); first = bond.NPV()
    assert f.evaluate(f.price, r0=r0, k=0.5, sigma=0.1, T=4.0, **numbers) == pytest.approx(first, rel=1e-6)


def test_constant_rates_under_the_hybrid_are_the_equity_model():
    equity = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.0, 0.0, [0.3, 0.15])
    flat = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.03, 0.0, [0.3, 0.15])
    for rates in (rl.SwitchingVasicek(CHAIN, 0.03, 0.5, 0.03, 0.0), rl.SwitchingVasicek(CHAIN, 0.03, 0.0, [0.06, 0.01], 0.0)):
        hybrid = rl.SwitchingEquityRates(equity, rates)
        for exercise in (None, "american"):
            put = rl.VanillaOption(("put", 100.0), exercise=exercise, maturity=1.0)
            put.setPricingEngine(rl.SwitchingFDEngine(hybrid, n=(201, 21), steps=100)); value = put.NPV()
            put.setPricingEngine(rl.SwitchingFDEngine(flat, n=201, steps=100))
            assert value == pytest.approx(put.NPV(), rel=1e-12)
    moving = rl.SwitchingEquityRates(equity, rl.SwitchingVasicek(CHAIN, 0.03, 0.5, 0.06, 0.0))    # deterministic, not constant
    put = rl.VanillaOption(("put", 100.0), maturity=1.0); put.setPricingEngine(rl.SwitchingFDEngine(moving, n=(41, 21), steps=20))
    with pytest.raises(NotImplementedError, match="deterministic rate"):
        put.NPV()


def test_inputs_that_are_refused():
    cev = rl.SwitchingCEVProcess(CHAIN, 1.0, 0.0, 0.0, [1.0, 0.5], 0.5)
    for payoff in (("cash", "call", 0.0, 1.0), ("call", 0.0), ("put", -0.1)):     # the price can be absorbed at zero
        option = rl.VanillaOption(payoff, maturity=1.0)
        for engine in (rl.FirstOrderFDEngine(cev, n=101), rl.SwitchingFDReferee(cev, n=101), rl.SwitchingFDEngine(cev, n=101, steps=20)):
            option.setPricingEngine(engine)
            with pytest.raises(ValueError, match="reach zero"):
                option.NPV()
    for T, K in ((0.0, 100.0), (-1.0, 100.0), (1.0, 0.0), (1.0, -5.0), (float("inf"), 100.0), (1.0, float("nan"))):
        with pytest.raises(ValueError, match="helper requires"):                  # no vega to fit
            rl.VolatilityHelper(T, K, 0.2)
    assert rl.VolatilityHelper(1.0, 100.0, 0.2).volatility == 0.2
