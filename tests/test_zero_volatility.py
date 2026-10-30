"""The zero-volatility limits: a deterministic terminal price or rate pays its discounted payoff, Heston with no
volatility of variance is Black-Scholes at the integrated variance, and the Black formula at zero total volatility is
intrinsic value on the forward."""
import math
import warnings
import numpy as np
import pytest
import regimelib as rl

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)
FAST = rl.RegimeChain.twoState(30.0, 50.0)


def N(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def black(F, K, v, disc=1.0):
    d1 = (math.log(F / K) + 0.5 * v) / math.sqrt(v)
    return disc * (F * N(d1) - K * N(d1 - math.sqrt(v)))


def price(instrument, engine):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        instrument.setPricingEngine(engine)
        return instrument.NPV()


def engines(model):
    return [rl.NumericalSwitchingEngine(model), rl.FastSwitchingEngine(model, order=2)]


@pytest.mark.parametrize("K", [90.0, 100.0 * math.exp(0.01), 110.0])
def test_deterministic_equity_pays_the_discounted_payoff(K):
    S0, r, q, T = 100.0, 0.02, 0.01, 1.0
    model = rl.SwitchingBlackScholesProcess(CHAIN, S0, r, q, 0.0)
    F, dr, dq = S0 * math.exp((r - q) * T), math.exp(-r * T), math.exp(-q * T)
    for engine in engines(model):
        call, put = rl.VanillaOption(("call", K), maturity=T), rl.VanillaOption(("put", K), maturity=T)
        assert price(call, engine) == pytest.approx(dr * max(F - K, 0.0), abs=1e-12)
        assert price(put, engine) == pytest.approx(dr * max(K - F, 0.0), abs=1e-12)
        assert call.gamma() == 0.0 and call.delta() == pytest.approx(dq if F > K else (0.5 * dq if F == K else 0.0))
        assert price(rl.VanillaOption(("cash", "call", K, 5.0), maturity=T), engine) == pytest.approx(5.0 * dr * (F > K))
        assert price(rl.VanillaOption(("asset", "put", K), maturity=T), engine) == pytest.approx(dr * F * (F < K))
        G = S0 * math.exp((r - q) * T / 2)
        assert price(rl.ContinuousGeometricAsianOption(("call", K), maturity=T), engine) == pytest.approx(dr * max(G - K, 0.0), abs=1e-12)
        assert price(rl.ContinuousGeometricAsianOption(("put", K), maturity=T), engine) == pytest.approx(dr * max(K - G, 0.0), abs=1e-12)
    jumpless = rl.SwitchingMerton76Process(CHAIN, S0, r, q, 0.0, 0.0, -0.1, 0.1)
    assert price(rl.VanillaOption(("call", 90.0), maturity=T), rl.NumericalSwitchingEngine(jumpless)) == pytest.approx(dr * (F - 90.0))


def test_monte_carlo_zero_variance_paths():
    model = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.02, 0.0, 0.0)
    F, dr = 100.0 * math.exp(0.02), math.exp(-0.02)
    for payoff, value in ((("call", 90.0), dr * (F - 90.0)), (("put", 90.0), 0.0), (("call", 110.0), 0.0),
                          (("cash", "call", 90.0, 3.0), 3.0 * dr), (("cash", "call", F, 3.0), 0.0)):      # strict at the strike
        assert price(rl.VanillaOption(payoff, maturity=1.0), rl.MonteCarloSwitchingEngine(model, paths=20)) == pytest.approx(value)
    mixed = rl.SwitchingBlackScholesProcess(rl.RegimeChain.twoState(0.5, 0.5), 100.0, 0.02, 0.0, [0.0, 0.3])
    mc = rl.MonteCarloSwitchingEngine(mixed, regime=0, paths=4000, seed=2)
    value = price(rl.VanillaOption(("call", 100.0), maturity=1.0), mc)          # some paths never leave the quiet regime
    assert value > dr * (F - 100.0) and math.isfinite(mc.standardError)


def test_black_formula_at_zero_total_volatility():
    model = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.02, 0.0, 0.0)
    engine = rl.NumericalSwitchingEngine(model)
    F, dr = 100.0 * math.exp(0.02), math.exp(-0.02)
    for K in (90.0, 110.0):
        for kind in ("call", "put"):
            option = rl.VanillaOption((kind, K), maturity=1.0); option.setPricingEngine(engine)
            assert option.impliedVolatility() == 0.0
            helper = rl.VolatilityHelper(1.0, K, 0.0, kind); helper.setPricingEngine(engine)
            intrinsic = dr * (max(F - K, 0.0) if kind == "call" else max(K - F, 0.0))
            assert helper.marketValue() == pytest.approx(intrinsic, abs=1e-14)
            assert math.isfinite(helper.calibrationError())                     # a zero market value is not divided by
    quiet = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.02, 0.0, 5e-5)       # below the default lower search bound
    option = rl.VanillaOption(("call", F), maturity=1.0); option.setPricingEngine(rl.NumericalSwitchingEngine(quiet))
    assert option.impliedVolatility(price=black(F, F, 5e-5 ** 2, dr)) == pytest.approx(5e-5, rel=1e-3)
    with pytest.raises(ValueError, match="Black range"):
        option.impliedVolatility(price=-0.01)                                   # below intrinsic
    with pytest.raises(ValueError, match="Black range"):
        option.impliedVolatility(price=101.0)                                   # above the forward
    with pytest.raises(ValueError):
        rl.VolatilityHelper(1.0, 100.0, -0.2)


def test_heston_without_volatility_of_variance():
    S0, v0, kappa, theta, T = 100.0, 0.09, 2.0, 0.04, 1.0
    V = theta * T + (v0 - theta) * (1 - math.exp(-kappa * T)) / kappa           # the variance path is deterministic
    frozen = rl.SwitchingHestonModel(CHAIN, S0, 0.0, 0.0, v0, kappa, theta, 0.0, -0.5)
    for engine in engines(frozen):
        option = rl.VanillaOption(("call", 105.0), maturity=T)
        assert price(option, engine) == pytest.approx(black(S0, 105.0, V), rel=1e-8)
        assert all(math.isfinite(g) for g in (option.delta(), option.gamma(), option.theta(), option.vega()))
    switching = lambda xi: rl.SwitchingHestonModel(CHAIN, S0, 0.0, 0.0, v0, kappa, [0.09, 0.02], xi, -0.5)
    option = rl.VanillaOption(("call", 100.0), maturity=T)
    limit = price(option, rl.NumericalSwitchingEngine(switching(0.0)))
    near, nearer = (abs(price(option, rl.NumericalSwitchingEngine(switching(xi))) - limit) for xi in (0.02, 0.005))
    assert near < 2e-3 * limit and nearer < 0.3 * near                          # the price is continuous at xi = 0
    bates = rl.SwitchingBatesModel(CHAIN, S0, 0.0, 0.0, v0, kappa, [0.09, 0.02], 0.0, -0.5, [2.0, 0.2], -0.08, 0.1)
    assert price(option, rl.NumericalSwitchingEngine(bates)) > limit              # jumps add value


def test_deterministic_rates_give_intrinsic_rate_options():
    models = [rl.SwitchingVasicek(FAST, 0.03, 0.5, 0.05, 0.0), rl.SwitchingHullWhite(FAST, 0.03, 0.5, 0.0),
              rl.SwitchingG2(FAST, 0.03, 0.5, 0.0, 0.1, 0.0, [-0.4, -0.7]),
              rl.SwitchingG2(FAST, 0.03, 0.5, [0.01, 0.02], 0.5, [0.01, 0.02], -1.0)]     # x + z cancels exactly
    for model in models:
        for engine in engines(model):
            bond = lambda t: price(rl.ZeroCouponBond(t), engine)
            P1, P3 = bond(1.0), bond(3.0)
            for K in (P3 / P1 - 0.03, P3 / P1 + 0.03):
                assert price(rl.ZeroCouponBondOption("call", K, 1.0, 3.0), engine) == pytest.approx(max(P3 - K * P1, 0.0), abs=1e-10)
                assert price(rl.ZeroCouponBondOption("put", K, 1.0, 3.0), engine) == pytest.approx(max(K * P1 - P3, 0.0), abs=1e-10)
    for model in models[:2]:
        for engine in engines(model):
            bond = lambda t: price(rl.ZeroCouponBond(t), engine)
            for rate in (0.02, 0.08):
                swap = rate * (bond(2.0) + bond(3.0)) + bond(3.0) - bond(1.0)       # receiver swap value at expiry, discounted
                assert price(rl.Swaption("receiver", 1.0, [2.0, 3.0], rate), engine) == pytest.approx(max(swap, 0.0), abs=1e-12)
                assert price(rl.Swaption("payer", 1.0, [2.0, 3.0], rate), engine) == pytest.approx(max(-swap, 0.0), abs=1e-12)
            cap = rl.CapFloor("cap", [1.0, 2.0], 0.02)
            assert price(cap, engine) == pytest.approx(max(bond(1.0) - 1.02 * bond(2.0), 0.0), abs=1e-12)


def test_zero_volatility_with_a_switching_level_is_refused():
    model = rl.SwitchingVasicek(CHAIN, 0.03, 0.5, [0.06, 0.02], 0.0)
    engine = rl.NumericalSwitchingEngine(model, information="observed")
    for instrument in (rl.ZeroCouponBondOption("call", 0.9, 1.0, 3.0), rl.Swaption("payer", 1.0, [2.0, 3.0], 0.04)):
        instrument.setPricingEngine(engine)
        with pytest.raises(NotImplementedError, match="zero volatility"):
            instrument.NPV()
    assert price(rl.ZeroCouponBond(3.0), engine) > 0                              # a bond needs no inversion


def test_deep_out_of_the_money_put_quotes_are_not_rounded_to_zero():
    """A put far below the forward is priced directly, not as a call less the forward, which would cancel it."""
    model = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.02, 0.0, 0.2)
    engine = rl.NumericalSwitchingEngine(model)
    F, dr, v = 100.0 * math.exp(0.02), math.exp(-0.02), 0.04
    for K in (20.0, 21.0, 30.0):
        d2 = (math.log(F / K) - 0.5 * v) / math.sqrt(v)
        exact = dr * (K * N(-d2) - F * N(-d2 - math.sqrt(v)))
        helper = rl.VolatilityHelper(1.0, K, 0.2, "put"); helper.setPricingEngine(engine)
        assert exact > 0 and helper.marketValue() == pytest.approx(exact, rel=1e-10)
        option = rl.VanillaOption(("put", K), maturity=1.0); option.setPricingEngine(engine)
        assert option.impliedVolatility(price=exact) == pytest.approx(0.2, rel=1e-6)
