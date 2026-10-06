"""At the ends of a contract's domain the price needs no transform: zero maturity pays the payoff, a strike at or
below zero removes the optionality of a claim on a positive price, and a one-regime chain has nothing to expand."""
import math
import warnings
import numpy as np
import pytest
import regimelib as rl

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)
BS = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.02, 0.01, [0.3, 0.15])
FAST = rl.RegimeChain.twoState(30.0, 50.0)                                   # the expansion of a bond option needs a fast chain
VASICEK = rl.SwitchingVasicek(FAST, 0.03, 0.5, [0.06, 0.02], [0.015, 0.008])
HW = rl.SwitchingHullWhite(FAST, 0.03, 0.5, [0.02, 0.006])
G2 = rl.SwitchingG2(FAST, 0.03, 0.5, [0.015, 0.006], 0.1, [0.012, 0.008], [-0.4, -0.7])


def engines(model):
    return [rl.NumericalSwitchingEngine(model), rl.FastSwitchingEngine(model, order=2)]


def price(instrument, engine):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)                      # no divide-by-zero or overflow on the way
        instrument.setPricingEngine(engine)
        return instrument.NPV()


@pytest.mark.parametrize("K, call, put", [(90.0, 10.0, 0.0), (100.0, 0.0, 0.0), (110.0, 0.0, 10.0)])
def test_zero_maturity_vanilla_digital_and_asian_pay_the_payoff(K, call, put):
    models = [BS, rl.SwitchingHestonModel(CHAIN, 100.0, 0.02, 0.0, 0.04, 2.0, [0.09, 0.02], 0.4, -0.6),
              rl.SwitchingMerton76Process(CHAIN, 100.0, 0.02, 0.0, 0.2, [2.0, 0.2], -0.08, 0.1)]
    for model in models:
        for engine in engines(model):
            assert price(rl.VanillaOption(("call", K), maturity=0.0), engine) == call
            assert price(rl.VanillaOption(("put", K), maturity=0.0), engine) == put
            assert price(rl.VanillaOption(("cash", "call", K, 7.0), maturity=0.0), engine) == (7.0 if 100.0 > K else 0.0)
    for engine in engines(BS):
        assert price(rl.ContinuousGeometricAsianOption(("call", K), maturity=0.0), engine) == call
        assert price(rl.ContinuousGeometricAsianOption(("put", K), maturity=0.0), engine) == put


def test_zero_maturity_greeks_are_those_of_the_payoff():
    option = rl.VanillaOption(("call", 90.0), maturity=0.0); option.setPricingEngine(rl.NumericalSwitchingEngine(BS))
    assert (option.delta(), option.gamma(), option.rho()) == (1.0, 0.0, 0.0)
    assert option.theta() == pytest.approx(0.01 * 100.0 - 0.02 * 90.0)        # d/dt of S e^{-q tau} - K e^{-r tau}
    out = rl.VanillaOption(("call", 110.0), maturity=0.0); out.setPricingEngine(rl.NumericalSwitchingEngine(BS))
    assert (out.delta(), out.theta()) == (0.0, 0.0)
    near = rl.VanillaOption(("call", 90.0), maturity=1e-6); near.setPricingEngine(rl.NumericalSwitchingEngine(BS))
    assert near.NPV() == pytest.approx(10.0, abs=1e-4) and near.delta() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("K", [0.0, -0.1])
def test_nonpositive_strikes_on_equity(K):
    T, dq, dr = 1.0, math.exp(-0.01), math.exp(-0.02)
    for engine in engines(BS):
        call = rl.VanillaOption(("call", K), maturity=T)
        assert price(call, engine) == pytest.approx(100.0 * dq - K * dr, rel=1e-14)
        assert call.delta() == pytest.approx(dq) and call.gamma() == 0.0
        assert price(rl.VanillaOption(("put", K), maturity=T), engine) == 0.0
        assert price(rl.VanillaOption(("cash", "call", K, 10.0), maturity=T), engine) == pytest.approx(10.0 * dr)
        assert price(rl.VanillaOption(("cash", "put", K, 10.0), maturity=T), engine) == 0.0
        assert price(rl.VanillaOption(("asset", "call", K), maturity=T), engine) == pytest.approx(100.0 * dq)
        assert price(rl.VanillaOption(("asset", "put", K), maturity=T), engine) == 0.0
        asian = rl.ContinuousGeometricAsianOption(("call", K), maturity=T)
        ordinary = lambda kind: price(rl.ContinuousGeometricAsianOption((kind, 80.0), maturity=T), engine)
        forward = ordinary("call") - ordinary("put") + 80.0 * dr           # parity at a strike that is quick to price
        assert price(asian, engine) == pytest.approx(forward - K * dr, rel=1e-6)
        assert price(rl.ContinuousGeometricAsianOption(("put", K), maturity=T), engine) == 0.0
    rates = rl.SwitchingVasicek(CHAIN, 0.03, 0.5, [0.06, 0.02], [0.015, 0.008])
    hybrid = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.0, 0.01, [0.3, 0.15]), rates)
    for engine in engines(hybrid):
        bond = rl.ZeroCouponBond(T); bond.setPricingEngine(rl.NumericalSwitchingEngine(rates))
        assert price(rl.VanillaOption(("call", K), maturity=T), engine) == pytest.approx(100.0 * dq - K * bond.NPV(), rel=1e-6)
        assert price(rl.VanillaOption(("put", K), maturity=T), engine) == 0.0


@pytest.mark.parametrize("model", [VASICEK, HW, G2])
def test_bond_options_at_zero_expiry_and_nonpositive_strike(model):
    for engine in engines(model):
        bond = lambda t: price(rl.ZeroCouponBond(t), engine)
        P3, P1 = bond(3.0), bond(1.0)
        for K in (P3 - 0.05, P3 + 0.05):
            assert price(rl.ZeroCouponBondOption("call", K, 0.0, 3.0), engine) == pytest.approx(max(P3 - K, 0.0), abs=1e-14)
            assert price(rl.ZeroCouponBondOption("put", K, 0.0, 3.0), engine) == pytest.approx(max(K - P3, 0.0), abs=1e-14)
        for K in (0.0, -0.2):
            assert price(rl.ZeroCouponBondOption("call", K, 1.0, 3.0), engine) == pytest.approx(P3 - K * P1, rel=1e-13)
            assert price(rl.ZeroCouponBondOption("put", K, 1.0, 3.0), engine) == 0.0


@pytest.mark.parametrize("model", [VASICEK, HW])
def test_coupon_bond_options_swaptions_and_caps_at_the_ends(model):
    for engine in engines(model):
        bond = lambda t: price(rl.ZeroCouponBond(t), engine)
        flows = [(1.0, 0.04), (2.0, 1.04)]
        value = sum(c * bond(t) for t, c in flows)
        assert price(rl.CouponBondOption("call", 1.0, 0.0, flows), engine) == pytest.approx(max(value - 1.0, 0.0), abs=1e-14)
        assert price(rl.CouponBondOption("put", 1.0, 0.0, flows), engine) == pytest.approx(max(1.0 - value, 0.0), abs=1e-14)
        swap = rl.Swaption("receiver", 0.0, [1.0, 2.0], 0.04)
        assert price(swap, engine) == pytest.approx(max(value - 1.0, 0.0), abs=1e-14)
        assert price(rl.CouponBondOption("call", 0.0, 0.5, flows), engine) == pytest.approx(value, rel=1e-13)
        # a cap whose first period fixes today, and strikes at or below -1 / tau, where the caplet has no optionality
        spot = rl.CapFloor("cap", [0.0, 0.5, 1.0], 0.03)
        later = rl.CapFloor("cap", [0.5, 1.0], 0.03)
        first = max(1.0 - 1.015 * bond(0.5), 0.0)                              # (1 + tau K) put struck at 1 / (1 + tau K), now
        assert price(spot, engine) == pytest.approx(first + price(later, engine), rel=1e-12)
        for K in (-2.0, -3.0):                                                  # tau = 0.5: A = 1 + tau K = 0, -0.5
            A = 1.0 + 0.5 * K
            assert price(rl.CapFloor("cap", [0.5, 1.0], K), engine) == pytest.approx(bond(0.5) - A * bond(1.0), rel=1e-13)
            assert price(rl.CapFloor("floor", [0.5, 1.0], K), engine) == 0.0


def test_zero_maturity_bonds_for_every_rate_model():
    cir = rl.SwitchingCoxIngersollRoss(FAST, 0.03, [0.06, 0.02], 0.5, 0.1)
    jumps = rl.SwitchingVasicekJumps(FAST, 0.03, 0.5, [0.06, 0.02], 0.01, [3.0, 0.2], 0.02)
    for model in (VASICEK, cir, jumps, HW, G2):
        for engine in engines(model):
            assert price(rl.ZeroCouponBond(0.0), engine) == 1.0
            assert price(rl.ZeroCouponBond(0.0, regimeAtMaturity=0), engine) == 1.0
            assert price(rl.ZeroCouponBond(0.0, regimeAtMaturity=1), engine) == 0.0
    bond = rl.ZeroCouponBond(0.0); bond.setPricingEngine(rl.NumericalSwitchingEngine(cir))
    assert (bond.delta(), bond.gamma()) == (0.0, 0.0)
    tiny = rl.ZeroCouponBond(1e-9); tiny.setPricingEngine(rl.NumericalSwitchingEngine(cir))
    assert tiny.NPV() == pytest.approx(1.0, abs=1e-9)


def test_one_regime_chain_is_the_model_itself_at_every_order():
    one = rl.RegimeChain([[0.0]])
    r0, a, b, sigma, T = 0.03, 0.5, 0.05, 0.01, 4.0
    B = (1 - math.exp(-a * T)) / a
    exact = math.exp((b - sigma ** 2 / (2 * a * a)) * (B - T) - sigma ** 2 * B * B / (4 * a) - B * r0)
    model = rl.SwitchingVasicek(one, r0, a, b, sigma)
    for order in (0, 4, None):
        assert price(rl.ZeroCouponBond(T), rl.FastSwitchingEngine(model, order=order)) == pytest.approx(exact, rel=1e-10)
    assert price(rl.ZeroCouponBond(T), rl.NumericalSwitchingEngine(model)) == pytest.approx(exact, rel=1e-10)
    swaption = rl.Swaption("payer", 1.0, [2.0, 3.0], 0.05)
    assert price(swaption, rl.FastSwitchingEngine(model, order=3)) == pytest.approx(price(swaption, rl.NumericalSwitchingEngine(model)), rel=1e-9)
    bs = rl.SwitchingBlackScholesProcess(one, 100.0, 0.02, 0.0, 0.2)
    call = rl.VanillaOption(("call", 105.0), maturity=1.0)
    d1 = (math.log(100.0 / 105.0) + 0.02 + 0.02) / 0.2; d2 = d1 - 0.2
    N = lambda x: 0.5 * math.erfc(-x / math.sqrt(2))
    black = 100.0 * N(d1) - 105.0 * math.exp(-0.02) * N(d2)
    for order in (0, 2, None):
        assert price(call, rl.FastSwitchingEngine(bs, order=order)) == pytest.approx(black, rel=1e-8)


def test_negative_maturities_are_refused():
    option = rl.VanillaOption(("call", 100.0), maturity=-0.5); option.setPricingEngine(rl.NumericalSwitchingEngine(BS))
    with pytest.raises(ValueError, match="maturity"):
        option.NPV()
