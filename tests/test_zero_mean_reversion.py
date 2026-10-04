"""The limits of small and zero reversion speed. The loading (1 - e^{-kappa t}) / kappa tends to t, and an
exponential sum with coefficients of size 1 / kappa cannot represent that; near the limit the forcing is a Chebyshev
series of a stably evaluated loading instead. Checked at kappa = 0 against closed forms and for continuity."""
import math
import warnings
import pytest
import regimelib as rl

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)
FAST = rl.RegimeChain.twoState(30.0, 50.0)
R0, SIGMA, T = 0.03, 0.01, 5.0


def N(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def price(instrument, engine):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        instrument.setPricingEngine(engine)
        return instrument.NPV()


def engines(model):
    return [rl.NumericalSwitchingEngine(model), rl.FastSwitchingEngine(model, order=2)]


def test_vasicek_without_reversion_is_the_gaussian_random_walk():
    exact = math.exp(-R0 * T + SIGMA ** 2 * T ** 3 / 6)                       # the mean level does not enter at a = 0
    model = rl.SwitchingVasicek(CHAIN, R0, 0.0, [0.06, 0.02], SIGMA)
    for engine in engines(model):
        bond = rl.ZeroCouponBond(T)
        assert price(bond, engine) == pytest.approx(exact, rel=1e-10)
        assert bond.delta() == pytest.approx(-T * exact, rel=1e-10)
    mc = rl.MonteCarloSwitchingEngine(model, paths=50)
    assert price(rl.ZeroCouponBond(T), mc) == pytest.approx(exact, rel=1e-10)   # sigma does not switch: every path agrees


def test_vasicek_is_continuous_in_the_reversion_speed():
    limit = math.exp(-R0 * T + SIGMA ** 2 * T ** 3 / 6)
    gaps = []
    for a in (1e-2, 1e-4, 1e-6, 1e-8):
        model = rl.SwitchingVasicek(CHAIN, R0, a, R0, SIGMA)                    # level at r0, so the drift vanishes at first order
        values = [price(rl.ZeroCouponBond(T), e) for e in engines(model)]
        assert values[0] == pytest.approx(values[1], rel=1e-11)
        gaps.append(abs(values[0] - limit))
    assert gaps[0] > gaps[1] > gaps[2] and gaps[3] < 1e-8                      # no loss of digits on the way down


def test_switching_parameters_at_small_reversion_speed():
    for a in (1e-3, 1e-7, 0.0):
        model = rl.SwitchingVasicek(FAST, R0, a, [0.06, 0.02], [0.015, 0.008])
        numerical = price(rl.ZeroCouponBond(T), rl.NumericalSwitchingEngine(model))
        assert price(rl.ZeroCouponBond(T), rl.FastSwitchingEngine(model, order=4)) == pytest.approx(numerical, rel=1e-8)


def test_vasicek_jumps_without_reversion():
    lam, m = 2.0, 0.02                                                         # g = sigma^2 t^2 / 2 + lam (1 / (1 + m t) - 1)
    exact = math.exp(-R0 * T + SIGMA ** 2 * T ** 3 / 6 + lam * (math.log(1 + m * T) / m - T))
    model = rl.SwitchingVasicekJumps(CHAIN, R0, 0.0, 0.05, SIGMA, lam, m)
    for engine in engines(model):
        assert price(rl.ZeroCouponBond(T), engine) == pytest.approx(exact, rel=1e-9)
    near = rl.SwitchingVasicekJumps(CHAIN, R0, 1e-7, R0, SIGMA, lam, m)
    assert price(rl.ZeroCouponBond(T), rl.NumericalSwitchingEngine(near)) == pytest.approx(exact, rel=1e-5)


def test_hull_white_in_the_ho_lee_limit():
    frozen = rl.SwitchingHullWhite(CHAIN, 0.03, 0.0, 0.01)
    for engine in engines(frozen):
        assert price(rl.ZeroCouponBond(T), engine) == pytest.approx(math.exp(-0.03 * T), rel=1e-10)   # the curve is fitted
        P1, P3, v = math.exp(-0.03), math.exp(-0.09), 0.01 * 2.0                # bond volatility sigma (S - T) sqrt(T)
        d1 = math.log(P3 / (0.9 * P1)) / v + v / 2
        call = rl.ZeroCouponBondOption("call", 0.9, 1.0, 3.0)
        assert price(call, engine) == pytest.approx(P3 * N(d1) - 0.9 * P1 * N(d1 - v), rel=1e-8)
    switching = lambda a: rl.SwitchingHullWhite(FAST, 0.03, a, [0.012, 0.008])
    swaption = rl.Swaption("payer", 1.0, [2.0, 3.0], 0.04)
    limit = price(swaption, rl.NumericalSwitchingEngine(switching(0.0)))
    assert price(swaption, rl.NumericalSwitchingEngine(switching(1e-6))) == pytest.approx(limit, rel=1e-4)
    assert price(swaption, rl.FastSwitchingEngine(switching(0.0), order=2)) == pytest.approx(limit, rel=1e-4)


def test_cir_loading_at_the_ends():
    still = rl.SwitchingCoxIngersollRoss(CHAIN, 0.03, [0.05, 0.02], 0.0, 0.0)   # k = sigma = 0: the rate is constant
    for engine in engines(still):
        bond = rl.ZeroCouponBond(T)
        assert price(bond, engine) == pytest.approx(math.exp(-0.03 * T), rel=1e-10)
        assert bond.delta() == pytest.approx(-T * math.exp(-0.03 * T), rel=1e-10)
        assert bond.gamma() == pytest.approx(T * T * math.exp(-0.03 * T), rel=1e-10)
    tiny = rl.SwitchingCoxIngersollRoss(CHAIN, 0.03, 0.03, 1e-9, 1e-9)
    bond = rl.ZeroCouponBond(T)
    assert price(bond, rl.NumericalSwitchingEngine(tiny)) == pytest.approx(math.exp(-0.03 * T), rel=1e-7)
    assert bond.delta() == pytest.approx(-T * math.exp(-0.03 * T), rel=1e-7)
    k, sigma, theta, long = 2.0, 3.0, 0.05, 400.0                              # e^{h T} overflows; the bond is finite
    h = math.sqrt(k * k + 2 * sigma * sigma)
    B = 2 / (h + k)                                                            # the long-maturity loading
    logA = (2 * k * theta / sigma ** 2) * (math.log(2 * h / (h + k)) + (k - h) * long / 2)
    frozen = rl.SwitchingCoxIngersollRoss(CHAIN, 0.03, theta, k, sigma)
    bond = rl.ZeroCouponBond(long)
    value = price(bond, rl.NumericalSwitchingEngine(frozen))
    assert value == pytest.approx(math.exp(logA - B * 0.03), rel=1e-6)
    assert bond.delta() == pytest.approx(-B * value, rel=1e-9)
