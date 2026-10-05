"""Findings against the limits and the belief work: classifications that were too generous, a cache that did not know
what it depended on, and limits that had been fixed in one place and not in its neighbours."""
import math
import warnings
import numpy as np
import pytest
import scipy.sparse as sp
import regimelib as rl
from regimelib.engines import _noDiffusion

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)
FAST = rl.RegimeChain.twoState(30.0, 50.0)


def N(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def price(instrument, engine):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rl.ExpansionWarning)
        instrument.setPricingEngine(engine)
        return instrument.NPV()


def test_volatilities_and_reversion_speeds_are_nonnegative():
    bad = [lambda: rl.SwitchingVasicek(CHAIN, 0.03, 0.5, 0.05, [0.01, -0.01]),
           lambda: rl.SwitchingVasicek(CHAIN, 0.03, -0.5, 0.05, 0.01),
           lambda: rl.SwitchingCoxIngersollRoss(CHAIN, 0.03, 0.05, -0.5, 0.1),
           lambda: rl.SwitchingCoxIngersollRoss(CHAIN, 0.03, 0.05, 0.5, -0.1),
           lambda: rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.02, 0.0, -0.2),
           lambda: rl.SwitchingHullWhite(CHAIN, 0.03, 0.5, [0.01, -0.02]),
           lambda: rl.SwitchingG2(CHAIN, 0.03, 0.5, 0.01, -0.1, 0.01, 0.0),
           lambda: rl.SwitchingG2(CHAIN, 0.03, 0.5, 0.01, 0.1, -0.01, 0.0),
           lambda: rl.SwitchingHestonModel(CHAIN, 100.0, 0.02, 0.0, 0.04, -2.0, 0.04, 0.3, -0.5),
           lambda: rl.SwitchingHestonVolOfVol(CHAIN, 100.0, 0.02, 0.0, 0.04, 2.0, 0.04, [0.5, -0.2], -0.5),
           lambda: rl.SwitchingMerton76Process(CHAIN, 100.0, 0.02, 0.0, 0.2, 1.0, -0.1, -0.1),
           lambda: rl.SwitchingCEVProcess(CHAIN, 100.0, 0.02, 0.0, [2.0, -1.0], 0.6)]
    for build in bad:
        with pytest.raises(ValueError, match="nonnegative"):
            build()
    assert rl.SwitchingVasicek(CHAIN, 0.03, 0.0, -0.02, 0.0).b == [-0.02, -0.02]     # a level may be negative, a speed zero


def test_zero_volatility_cir_with_a_switching_level_is_not_deterministic():
    moving = rl.SwitchingCoxIngersollRoss(CHAIN, 0.03, [0.06, 0.02], 0.5, 0.0)
    assert _noDiffusion(moving) == "switching level"
    assert _noDiffusion(rl.SwitchingCoxIngersollRoss(CHAIN, 0.03, 0.04, 0.5, 0.0)) == "deterministic"
    assert _noDiffusion(rl.SwitchingCoxIngersollRoss(CHAIN, 0.03, [0.06, 0.02], 0.0, 0.0)) == "deterministic"     # no pull
    assert _noDiffusion(rl.SwitchingVasicek(CHAIN, 0.03, 0.0, [0.06, 0.02], 0.0)) == "deterministic"
    swaption = rl.Swaption("payer", 1.0, [2.0, 3.0], 0.04)
    for engine in (rl.NumericalSwitchingEngine(moving, information="observed"),
                   rl.SwitchingFDEngine(moving, n=101, steps=20, information="observed"), rl.SwitchingFDEngine(moving, n=101, steps=20)):
        swaption.setPricingEngine(engine)
        with pytest.raises(NotImplementedError, match="zero volatility"):
            swaption.NPV()
    bond = rl.ZeroCouponBond(3.0); bond.setPricingEngine(rl.NumericalSwitchingEngine(moving))
    assert 0 < bond.NPV() < 1


def test_fast_engine_prices_do_not_depend_on_what_was_priced_first():
    model = rl.SwitchingBlackScholesProcess(rl.RegimeChain.twoState(4.0, 2.0), 100.0, 0.02, 0.0, [0.40, 0.15])
    option = lambda K: rl.VanillaOption(("call", K), maturity=1.0)
    fresh = {r: price(option(100.0), rl.FastSwitchingEngine(model, order=3, regime=r)) for r in (0, 1)}
    for first, second in ((0, 1), (1, 0)):
        shared = rl.FastSwitchingEngine(model, order=3, regime=first)
        assert price(option(100.0), shared) == fresh[first]
        shared.regime = second
        assert price(option(100.0), shared) == fresh[second]
    mixed = price(option(100.0), rl.FastSwitchingEngine(model, order=3, regime=[0.3, 0.7]))
    assert mixed == pytest.approx(0.3 * fresh[0] + 0.7 * fresh[1], rel=1e-13)


def test_cached_nodes_still_report_their_diagnostics():
    model = rl.SwitchingBlackScholesProcess(rl.RegimeChain.twoState(4.0, 2.0), 100.0, 0.02, 0.0, [0.40, 0.15])

    def diagnostics(engine, K):
        option = rl.VanillaOption(("call", K), maturity=1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always"); option.setPricingEngine(engine); value = option.NPV()
        d = option._result("diagnostics")
        return value, d["lastTermRelative"], d["numericalNodes"], sum(issubclass(w.category, rl.ExpansionWarning) for w in caught)
    shared = rl.FastSwitchingEngine(model, order=3)
    for K in (100.0, 100.0, 105.0, 100.0):                                       # repeats are served from the cache
        assert diagnostics(shared, K) == diagnostics(rl.FastSwitchingEngine(model, order=3), K)


def test_adaptive_order_does_not_reuse_lower_order_vectors():
    model = rl.SwitchingBlackScholesProcess(FAST, 100.0, 0.02, 0.0, [0.30, 0.15])
    adaptive = rl.FastSwitchingEngine(model, order=None)
    option = rl.VanillaOption(("call", 100.0), maturity=1.0)
    value = price(option, adaptive)
    fixed = price(option, rl.FastSwitchingEngine(model, order=adaptive.orderUsed))
    assert adaptive.orderUsed >= 1 and value == pytest.approx(fixed, rel=1e-13)
    assert value != pytest.approx(price(option, rl.FastSwitchingEngine(model, order=0)), rel=1e-6)


def test_a_volatile_regime_with_no_stationary_weight_is_refused():
    leaving = rl.RegimeChain([[-1.0, 1.0], [0.0, 0.0]])                          # regime 0 is left for good
    vasicek = rl.SwitchingVasicek(leaving, 0.03, 0.5, 0.05, [0.01, 0.0])
    g2 = rl.SwitchingG2(leaving, 0.03, 0.5, [0.01, 0.0], 0.1, [0.01, 0.0], 0.0)
    for model in (vasicek, g2):
        option = rl.ZeroCouponBondOption("call", 0.9, 1.0, 3.0)
        option.setPricingEngine(rl.NumericalSwitchingEngine(model, information="observed"))
        with pytest.raises(NotImplementedError, match="stationary probability"):
            option.NPV()
    swaption = rl.Swaption("payer", 1.0, [2.0, 3.0], 0.04)
    swaption.setPricingEngine(rl.NumericalSwitchingEngine(vasicek, information="observed"))
    with pytest.raises(NotImplementedError, match="stationary probability"):
        swaption.NPV()


def test_american_obstacle_is_applied_after_each_implicit_half_step():
    model = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.05, 0.0, 0.2)
    engine = rl.SwitchingFDEngine(model, n=11, steps=1)
    L = sp.diags([-0.5, -1.0, -2.0]).tocsr(); u0 = np.array([1.0, 1.0, 1.0]); floor = np.array([0.9, 0.9, 0.9])
    half = 1.0 / (1.0 + 0.5 * np.array([0.5, 1.0, 2.0]))                         # one implicit half step of dt = 1, exactly
    expected = np.maximum(np.maximum(u0 * half, floor) * half, floor)
    assert engine._march(L, u0.copy(), 1.0, project=floor) == pytest.approx(expected, rel=1e-14)
    put = rl.VanillaOption(("put", 110.0), exercise="american", maturity=1.0)
    for steps in (1, 2, 50):
        put.setPricingEngine(rl.SwitchingFDEngine(model, n=401, steps=steps))
        assert put.NPV() >= 10.0 - 1e-12                                          # never below immediate exercise


def test_heston_without_vol_of_vol_at_small_and_zero_reversion():
    S0, v0, theta, T, K = 100.0, 0.09, 0.04, 1.0, 105.0
    for kappa in (0.0, 1e-16, 1e-12, 1.0):
        loading = T if kappa == 0 else -math.expm1(-kappa * T) / kappa
        V = theta * T + (v0 - theta) * loading
        d1 = (math.log(S0 / K) + 0.5 * V) / math.sqrt(V)
        black = S0 * N(d1) - K * N(d1 - math.sqrt(V))
        for model in (rl.SwitchingHestonModel(CHAIN, S0, 0.0, 0.0, v0, kappa, theta, 0.0, -0.5),
                      rl.SwitchingBatesModel(CHAIN, S0, 0.0, 0.0, v0, kappa, theta, 0.0, -0.5, 0.0, -0.1, 0.1)):
            for engine in (rl.NumericalSwitchingEngine(model), rl.FastSwitchingEngine(model, order=2)):
                option = rl.VanillaOption(("call", K), maturity=T)
                assert price(option, engine) == pytest.approx(black, rel=1e-7)
                assert math.isfinite(option.theta())


def test_equity_with_rates_at_small_and_zero_reversion():
    def hybrid(a, rho):
        return rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(FAST, 100.0, 0.0, 0.0, [0.3, 0.15]),
                                       rl.SwitchingVasicek(FAST, 0.03, a, [0.05, 0.02], [0.015, 0.008]), rho=rho)
    option = rl.VanillaOption(("call", 100.0), maturity=1.0)
    for rho in (0.0, -0.3):
        limit = price(option, rl.NumericalSwitchingEngine(hybrid(0.0, rho)))
        assert price(option, rl.NumericalSwitchingEngine(hybrid(1e-7, rho))) == pytest.approx(limit, rel=1e-6)
        assert price(option, rl.FastSwitchingEngine(hybrid(0.0, rho), order=3)) == pytest.approx(limit, rel=1e-5)
    ho_lee = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(FAST, 100.0, 0.0, 0.0, [0.3, 0.15]),
                                     rl.SwitchingHullWhite(FAST, 0.03, 0.0, [0.015, 0.008]))
    assert price(option, rl.NumericalSwitchingEngine(ho_lee)) > 0


def test_monte_carlo_bonds_are_continuous_in_the_reversion_speed():
    def bond(a):
        model = rl.SwitchingVasicek(CHAIN, 0.03, a, 0.03, [0.015, 0.008])
        engine = rl.MonteCarloSwitchingEngine(model, paths=300, seed=5)
        b = rl.ZeroCouponBond(5.0); b.setPricingEngine(engine); return b.NPV()
    limit = bond(0.0)
    gaps = [abs(bond(a) - limit) for a in (1e-3, 1e-6, 1e-9, 1e-12)]
    assert gaps[0] > gaps[1] > gaps[2] and gaps[3] < 1e-10                       # the same paths, a smooth function of a
    crossing = [bond(a) for a in (0.9 * 0.01, 1.1 * 0.01)]                       # across the switch of formula, a T = 0.05
    assert abs(crossing[0] - crossing[1]) < 1e-4
