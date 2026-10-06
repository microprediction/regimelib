"""Stationary laws on stiff and reducible chains, exercise windows on the grid, ODE failures that are reported, the
compensated jump exponent, default grids for planar models, and a few refusals."""
import cmath
import math
import warnings
import numpy as np
import pytest
import regimelib as rl

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)


def test_stationary_law_is_nonnegative_and_exact_on_a_stiff_chain():
    stiff = rl.RegimeChain([[-1e-12, 1e-12, 0.0], [1e6, -1e6 - 1.0, 1.0], [0.0, 1e-9, -1e-9]])
    pi = stiff.stationaryDistribution()
    assert np.all(pi >= 0) and pi.sum() == pytest.approx(1.0, abs=1e-15)
    assert np.max(np.abs(pi @ stiff.generator)) < 1e-20
    assert rl.RegimeChain.twoState(3.0, 5.0).stationaryDistribution() == pytest.approx([0.625, 0.375], rel=1e-15)
    scaled = rl.RegimeChain(1e9 * np.array([[-3.0, 3.0], [5.0, -5.0]]))
    assert scaled.stationaryDistribution() == pytest.approx([0.625, 0.375], rel=1e-15)


def test_closed_classes_and_what_needs_one():
    absorbing = rl.RegimeChain([[-1.0, 1.0], [0.0, 0.0]])                        # one closed class: the second regime
    assert [list(c) for c in absorbing.closedClasses()] == [[1]]
    assert absorbing.stationaryDistribution() == pytest.approx([0.0, 1.0])
    split = rl.RegimeChain([[0.0, 0.0, 0.0], [1.0, -3.0, 2.0], [0.0, 0.0, 0.0]])  # two closed classes, one transient regime
    assert sorted(list(c) for c in split.closedClasses()) == [[0], [2]]
    assert split.stationaryDistribution() == pytest.approx([(1 + 1 / 3) / 3, 0.0, (1 + 2 / 3) / 3])   # from a uniform start
    model = rl.SwitchingVasicek(split, 0.03, 0.5, [0.06, 0.04, 0.02], [0.01, 0.012, 0.008])
    with pytest.raises(ValueError, match="closed classes"):
        rl.FastSwitchingEngine(model)
    bs = rl.SwitchingBlackScholesProcess(split, 100.0, 0.02, 0.0, [0.3, 0.25, 0.15])
    with pytest.raises(ValueError, match="closed classes"):
        rl.FirstOrderFDEngine(bs)
    bond = rl.ZeroCouponBond(3.0)
    values = []
    for regime in (0, 1, 2):                                                     # the numerical engine needs no stationary law
        bond.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=regime)); values.append(bond.NPV())
    frozen = lambda b, s: (lambda z: (z.setPricingEngine(rl.NumericalSwitchingEngine(rl.SwitchingVasicek(rl.RegimeChain([[0.0]]), 0.03, 0.5, b, s))), z.NPV())[1])(rl.ZeroCouponBond(3.0))
    assert values[0] == pytest.approx(frozen(0.06, 0.01), rel=1e-9) and values[2] == pytest.approx(frozen(0.02, 0.008), rel=1e-9)
    assert min(values[0], values[2]) < values[1] < max(values[0], values[2])
    fast = rl.FastSwitchingEngine(rl.SwitchingVasicek(absorbing, 0.03, 0.5, [0.06, 0.02], 0.01), order=1)    # one class: allowed
    assert fast.model.n == 2


def test_an_ode_solve_that_stops_early_is_an_error():
    from regimelib._engine.fastswitch import numerical_a_callable
    blowup = [lambda t: 1.0 / (1.0 - t) ** 2, lambda t: 0.0]                      # not integrable to t = 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ArithmeticError, match="could not be integrated"):
            numerical_a_callable(2.0, [[-1.0, 1.0], [1.0, -1.0]], blowup)
    ok = numerical_a_callable(1.0, [[-1.0, 1.0], [1.0, -1.0]], [lambda t: -0.1, lambda t: -0.1])
    assert ok[0] == pytest.approx(math.exp(-0.1), rel=1e-10)


def test_american_exercise_window_and_theta_when_exercised():
    import QuantLib as ql
    ref = ql.Date(1, 1, 2020); ql.Settings.instance().evaluationDate = ref
    model = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.05, 0.0, [0.25, 0.25])
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, 160.0)        # far enough in the money to exercise at once
    expiry = ref + ql.Period(365, ql.Days)
    engine = rl.SwitchingFDEngine(model, n=401, steps=200)
    now = rl.VanillaOption(payoff, ql.AmericanExercise(ref, expiry)); now.setPricingEngine(engine)
    later = rl.VanillaOption(payoff, ql.AmericanExercise(ref + ql.Period(292, ql.Days), expiry)); later.setPricingEngine(engine)
    european = rl.VanillaOption(payoff, ql.EuropeanExercise(expiry)); european.setPricingEngine(engine)
    assert now.earliestExercise == 0.0 and later.earliestExercise == pytest.approx(0.8)
    assert now.NPV() == pytest.approx(60.0, abs=1e-9) and now.theta() == 0.0     # exercised today: the payoff, no decay
    assert european.NPV() < later.NPV() < now.NPV()                              # a later window is worth less
    assert european.theta() != 0.0
    still = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.05, 0.0, 0.0)        # known path: best date in the window
    later.setPricingEngine(rl.SwitchingFDEngine(still, n=101, steps=20))
    assert later.NPV() == pytest.approx(math.exp(-0.05 * 0.8) * (160.0 - 100.0 * math.exp(0.05 * 0.8)), rel=1e-12)


def test_bermudan_exercise_today_is_not_skipped():
    model = rl.SwitchingHullWhite(CHAIN, 0.03, 0.5, [0.012, 0.012])
    fixed, rate = [1.0, 2.0, 3.0], 0.08                                          # a receiver deep in the money
    bermudan = rl.Swaption("receiver", 0.0, fixed, rate, exerciseTimes=[0.0, 1.0, 2.0])
    later = rl.Swaption("receiver", 1.0, [2.0, 3.0], rate, exerciseTimes=[1.0, 2.0])
    engine = rl.SwitchingFDEngine(model, n=401, steps=200)
    bermudan.setPricingEngine(engine); later.setPricingEngine(engine)
    intrinsic = sum(c * math.exp(-0.03 * t) for t, c in bermudan.cashflows) - 1.0
    assert intrinsic > 0 and bermudan.NPV() >= intrinsic - 1e-9                  # at least immediate exercise
    assert bermudan.NPV() >= later.NPV() - 1e-9
    today = rl.Swaption("receiver", 0.0, fixed, rate); today.setPricingEngine(engine)
    assert today.NPV() == pytest.approx(intrinsic, abs=1e-9)                     # a single exercise date, today


def test_compensated_jump_exponent_keeps_small_jumps():
    from regimelib._engine.quantlib_models import compensated_jump
    u = 1 - 0.5j
    direct = lambda mu, de: cmath.exp(1j * u * mu - 0.5 * u * u * de * de) - 1 - 1j * u * (cmath.exp(mu + 0.5 * de * de) - 1)
    assert compensated_jump(u, -0.08, 0.10) == pytest.approx(direct(-0.08, 0.10), rel=1e-13)
    assert compensated_jump(u, 0.0, 0.0) == 0
    mu = de = 1e-9                                                               # second order: -(u^2 + iu)(de^2 + mu^2) / 2
    assert compensated_jump(u, mu, de) == pytest.approx(-0.5 * (u * u + 1j * u) * (de * de + mu * mu), rel=1e-6)
    S0, sigma, c = 100.0, 0.2, 0.03                                              # intensity c / de^2 with jumps of size de:
    for de in (1e-3, 1e-5):                                                      # a diffusion of variance sigma^2 + c
        merton = rl.SwitchingMerton76Process(CHAIN, S0, 0.0, 0.0, sigma, c / de ** 2, 0.0, de)
        black = rl.SwitchingBlackScholesProcess(CHAIN, S0, 0.0, 0.0, math.sqrt(sigma ** 2 + c))
        option = rl.VanillaOption(("call", 100.0), maturity=1.0)
        option.setPricingEngine(rl.NumericalSwitchingEngine(merton)); value = option.NPV()
        option.setPricingEngine(rl.NumericalSwitchingEngine(black))
        assert value == pytest.approx(option.NPV(), rel=1e-4)


def test_default_grids_for_planar_models_on_the_grid_engine():
    g2 = rl.SwitchingG2(CHAIN, 0.03, 0.5, [0.015, 0.006], 0.1, [0.012, 0.008], [-0.4, -0.7])
    hybrid = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.0, 0.0, [0.3, 0.15]),
                                     rl.SwitchingVasicek(CHAIN, 0.03, 0.5, [0.05, 0.02], [0.015, 0.008]))
    line = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.03, 0.0, [0.3, 0.15])
    assert rl.SwitchingFDEngine(g2).n == (161, 81) and rl.SwitchingFDEngine(hybrid).n == (161, 61)
    assert rl.SwitchingFDEngine(line).n == 1001
    with pytest.raises(ValueError, match="nodes per regime"):
        rl.SwitchingFDEngine(g2, n=1001)


def test_smaller_refusals_and_additions():
    with pytest.raises(ValueError, match="nu must be"):
        rl.SwitchingVarianceGammaProcess(CHAIN, 100.0, 0.02, 0.0, 0.2, [0.5, 0.0], -0.1)
    cev = rl.SwitchingCEVProcess(CHAIN, 100.0, 0.02, 0.0, [2.5, 1.2], 0.6)
    option = rl.VanillaOption(("call", 100.0), maturity=1.0)
    option.setPricingEngine(rl.SwitchingFDReferee(cev, n=101))
    with pytest.raises(NotImplementedError, match="implied volatility needs"):
        option.impliedVolatility()
    hybrid = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.0, 0.0, [0.3, 0.15]),
                                     rl.SwitchingVasicek(CHAIN, 0.03, 0.5, [0.05, 0.02], [0.015, 0.008]))
    option.setPricingEngine(rl.NumericalSwitchingEngine(hybrid))
    with pytest.raises(NotImplementedError, match="implied volatility needs"):
        option.impliedVolatility()
    heston = rl.SwitchingHestonModel(CHAIN, 100.0, 0.02, 0.01, 0.04, 2.0, [0.09, 0.02], 0.4, -0.6)
    for payoff, T in ((("call", 90.0), 0.0), (("put", 110.0), 0.0), (("call", 0.0), 1.0), (("put", -1.0), 1.0)):
        for engine in (rl.NumericalSwitchingEngine(heston), rl.FastSwitchingEngine(heston, order=1, regime=1)):
            exact = rl.VanillaOption(payoff, maturity=T); exact.setPricingEngine(engine)
            assert exact.vega() == 0.0                                           # a payoff or a forward: no dependence on v0


def test_chain_from_and_to_a_transition_matrix():
    """A utility for whoever estimates a chain at a data frequency; regimelib does not estimate."""
    chain = rl.RegimeChain.twoState(0.25, 1.0)
    P = chain.transitionMatrix(1 / 12)
    assert P.sum(axis=1) == pytest.approx([1.0, 1.0]) and np.all(P >= 0)
    back = rl.RegimeChain.fromTransitionMatrix(P, 1 / 12)
    assert back.generator == pytest.approx(chain.generator, rel=1e-10)
    three = rl.RegimeChain([[-5.0, 3.0, 2.0], [4.0, -9.0, 5.0], [1.0, 6.0, -7.0]])
    back = rl.RegimeChain.fromTransitionMatrix(three.transitionMatrix(0.02), 0.02)
    assert back.generator == pytest.approx(three.generator, rel=1e-8, abs=1e-8)
    assert rl.RegimeChain.fromTransitionMatrix(np.eye(2), 1.0).generator == pytest.approx(np.zeros((2, 2)))
    assert chain.transitionMatrix(0.0) == pytest.approx(np.eye(2))
    for bad in ([[0.4, 0.6], [0.6, 0.4]], [[0.5, 0.5], [0.5, 0.5]], [[0.0, 1.0], [1.0, 0.0]]):     # p11 + p22 <= 1
        with pytest.raises(ValueError, match="no continuous-time chain"):
            rl.RegimeChain.fromTransitionMatrix(bad, 1.0)
    cyclic = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]                # a rotation: no generator
    with pytest.raises(ValueError):
        rl.RegimeChain.fromTransitionMatrix(cyclic, 1.0)
    for bad in ([[0.9, 0.2], [0.1, 0.9]], [[1.1, -0.1], [0.1, 0.9]], [[0.9, 0.1]]):
        with pytest.raises(ValueError):
            rl.RegimeChain.fromTransitionMatrix(bad, 1.0)
    with pytest.raises(ValueError, match="dt"):
        rl.RegimeChain.fromTransitionMatrix(P, 0.0)
