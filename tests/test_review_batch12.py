"""Ends of the parameter domain in calibration, validation by class rather than by name, precision at small and
large parameters, and contracts that need no grid or transform."""
import math
import warnings

import mpmath as mp
import numpy as np
import pytest

import regimelib as rl
from regimelib.calibration import defaultBounds

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)
ONE = rl.RegimeChain([[0.0]])


def test_calibration_starts_from_zero_reversion():
    model = rl.SwitchingHestonModel(CHAIN, 100.0, 0.02, 0.0, 0.04, 0.0, [0.04, 0.04], 0.3, -0.5)
    assert defaultBounds(model, "kappa") == (0.0, np.inf)
    assert rl.calibrate(model, [rl.VolatilityHelper(1.0, 100.0, 0.2)], ["kappa"], max_nfev=1).nfev == 1


def test_a_subclass_keeps_its_domains():
    class AliasVG(rl.SwitchingVarianceGammaProcess):
        pass

    class AliasG2(rl.SwitchingG2):
        pass
    vg = AliasVG(CHAIN, 100.0, 0.01, 0.0, sigma=[0.2, 0.25], nu=[0.4, 0.5], theta=[-0.15, -0.05])
    assert defaultBounds(vg, "theta") == (-np.inf, np.inf)
    g2 = AliasG2(CHAIN, 0.02, 0.1, 0.01, 0.2, 0.01, -0.5)
    assert defaultBounds(g2, "b") == (0.0, np.inf)
    with pytest.raises(ValueError, match="b must be finite and nonnegative"):
        AliasG2(ONE, 0.03, 0.10, 0.01, -0.20, 0.01, 0.0)


def test_quantlib_cash_digital_keeps_its_cash_at_any_strike():
    ql = pytest.importorskip("QuantLib")
    for kind, strike in ((ql.Option.Call, -2.0), (ql.Option.Put, -2.0), (ql.Option.Put, 50.0), (ql.Option.Call, 50.0)):
        assert rl.VanillaOption(ql.CashOrNothingPayoff(kind, strike, 10.0), maturity=1.0).cash == 10.0


def test_a_payment_of_nothing_changes_nothing():
    model = rl.SwitchingVasicek(CHAIN, r0=0.03, a=0.0, b=[0.03, 0.03], sigma=[0.20, 0.20])
    values = []
    for flows in ([(2.0, 1.0)], [(2.0, 1.0), (50.0, 0.0)]):
        option = rl.CouponBondOption("call", strike=1.01, maturity=1.0, cashflows=flows)
        option.setPricingEngine(rl.NumericalSwitchingEngine(model)); values.append(option.NPV())
    assert values[0] == values[1]


def test_volatility_error_has_no_ceiling():
    for sigma in (4.0, 5.0):
        model = rl.SwitchingBlackScholesProcess(ONE, S0=100.0, r=0.02, q=0.0, sigma=[sigma])
        helper = rl.VolatilityHelper(1.0, 100.0, sigma); helper.setPricingEngine(rl.NumericalSwitchingEngine(model))
        assert abs(helper.volatilityError()) < 1e-9
    model = rl.SwitchingBlackScholesProcess(ONE, S0=100.0, r=0.02, q=0.0, sigma=[30.0])      # the Black price has saturated
    helper = rl.VolatilityHelper(1.0, 100.0, 30.0); helper.setPricingEngine(rl.NumericalSwitchingEngine(model))
    with pytest.raises(ValueError, match="upper bound"):
        helper.volatilityError()


def test_cir_parameter_greeks_vanish_without_reversion():
    from regimelib.symbolic import CIRBondFirstOrder
    formula = CIRBondFirstOrder()
    for wrt in (("theta", 0), ("theta", 1), ("q", 0, 1), ("q", 1, 0)):
        assert formula.parameterGreek(CHAIN, wrt, regime=0, r0=0.03, k=0.0, sigma=0.08, T=4.0, thetas=[0.06, 0.02]) == 0.0


def test_heston_without_reversion_at_zero_frequency():
    equity = rl.SwitchingHestonModel(chain=CHAIN, S0=100.0, r=0.03, q=0.01, v0=0.04, kappa=0.0, theta=[0.04, 0.09],
                                     sigma=0.4, rho=-0.5)
    assert equity.returnForcing(0.0, 1.0)[1][0](0.5) == 0
    hybrid = rl.SwitchingEquityRates(equity, rl.SwitchingHullWhite(CHAIN, 0.03, a=0.3, sigma=0.0))
    values = []
    for model in (equity, hybrid):
        option = rl.VanillaOption(("call", 100.0), maturity=1.0)
        option.setPricingEngine(rl.NumericalSwitchingEngine(model)); values.append(option.NPV())
    assert values[1] == pytest.approx(values[0], abs=1e-8)


def test_two_state_formula_near_coalescing_eigenvalues():
    from regimelib.symbolic import TwoStateConstantForcing
    q = 1e8; g1, g2 = complex(math.nextafter(q, math.inf), q), complex(q, -q)
    formula = TwoStateConstantForcing()
    actual = formula.evaluate(formula.a(0), q12=q, q21=q, g1=g1, g2=g2, T=1.0)
    with mp.workdps(80):
        M = mp.matrix([[mp.mpc(g1) - q, q], [q, mp.mpc(g2) - q]])
        expected = complex((mp.expm(M) * mp.matrix([1, 1]))[0])
    assert abs(actual - expected) < 1e-12 * abs(expected)


def test_two_state_exact_keeps_the_slow_eigenvalue():
    from regimelib._engine.explicit import two_state_constant_exact
    for lam in (1e6, 1e8, 1e10):                             # the leading exponent is t gt^2 / (2 lam) = 1 / 2
        assert two_state_constant_exact(0.0, 1.0, lam, lam).real == pytest.approx(math.exp(0.5), rel=1e-5)


def test_stationary_law_refuses_rates_beyond_the_floating_range():
    with pytest.raises(ArithmeticError, match="floating-point range"), warnings.catch_warnings():
        warnings.simplefilter("error")
        rl.RegimeChain.twoState(1e154, 1e-155).stationaryDistribution()
    assert np.allclose(rl.RegimeChain.twoState(1e10, 1e-10).stationaryDistribution(), [1e-20, 1.0])


def test_monte_carlo_tiny_volatility_does_not_depend_on_the_chain():
    for rates in ((0.1, 0.1), (30.0, 50.0)):
        model = rl.SwitchingBlackScholesProcess(rl.RegimeChain.twoState(*rates), S0=100.0, r=0.0, q=0.0, sigma=2e-162)
        option = rl.VanillaOption(("cash", "call", 100.0, 7.0), maturity=1.0)
        engine = rl.MonteCarloSwitchingEngine(model, regime=0, paths=200, seed=1); option.setPricingEngine(engine)
        assert option.NPV() == pytest.approx(3.5) and engine.standardError == pytest.approx(0.0, abs=1e-12)


def test_variance_gamma_exponent_as_nu_falls():
    from regimelib._engine.quantlib_models import variance_gamma
    for nu in (1e-10, 1e-14, 1e-16):                         # tends to -sigma^2 (u^2 + i u) / 2
        c = variance_gamma(1.0, [0.2], [nu], [0.3])[0][0].value(0)
        assert abs(c - (-0.02 - 0.02j)) < 1e-9
    coarse = variance_gamma(1.3, [0.2], [0.5], [0.3])[0][0].value(0)         # away from the limit: the plain logarithms
    import cmath
    psi = lambda z: -cmath.log(1 - 1j * 0.3 * 0.5 * z + 0.5 * 0.04 * 0.5 * z * z) / 0.5
    assert coarse == pytest.approx(psi(1.3) - 1.3j * psi(-1j), rel=1e-13)


def test_finite_difference_engines_at_zero_maturity():
    models = (rl.SwitchingBlackScholesProcess(CHAIN, S0=100.0, r=0.03, q=0.01, sigma=[0.30, 0.15]),
              rl.SwitchingCEVProcess(CHAIN, S0=100.0, r=0.03, q=0.01, sigma=[2.5, 1.2], beta=0.6))
    for model in models:
        for Engine in (rl.FirstOrderFDEngine, rl.SwitchingFDReferee, rl.SwitchingFDEngine):
            option = rl.VanillaOption(("call", 90.0), maturity=0.0); option.setPricingEngine(Engine(model))
            assert option.NPV() == 10.0


def test_bond_option_ends_under_inferred_information():
    p, K = [0.3, 0.7], 0.89
    model = rl.SwitchingVasicek(rl.RegimeChain.twoState(2.0, 3.0), r0=0.03, a=0.5, b=[0.07, 0.02], sigma=0.01)

    def bond(regime):
        z = rl.ZeroCouponBond(3.0); z.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=regime)); return z.NPV()
    P0, P1 = bond(0), bond(1); P = p[0] * P0 + p[1] * P1
    free, now = rl.ZeroCouponBondOption("call", 0.0, 1.0, 3.0), rl.ZeroCouponBondOption("call", K, 0.0, 3.0)
    engine = rl.NumericalSwitchingEngine(model, regime=p)
    free.setPricingEngine(engine); now.setPricingEngine(engine)
    assert free.NPV() == pytest.approx(P, abs=1e-12)
    assert now.NPV() == pytest.approx(max(P - K, 0.0), abs=1e-12)        # the payoff of the belief-weighted bond
    now.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=p, information="observed"))
    assert now.NPV() == pytest.approx(p[0] * max(P0 - K, 0.0) + p[1] * max(P1 - K, 0.0), abs=1e-12)
    now.setPricingEngine(rl.FastSwitchingEngine(model, regime=p, order=2))
    assert now.NPV() == pytest.approx(max(P - K, 0.0), abs=5e-5)
    known = rl.ZeroCouponBondOption("call", P1 - 0.01, 0.0, 3.0)
    known.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=1))
    assert known.NPV() == pytest.approx(0.01, abs=1e-12)
    inside = rl.ZeroCouponBondOption("call", K, 1.0, 3.0); inside.setPricingEngine(engine)
    with pytest.raises(NotImplementedError):
        inside.NPV()


def test_closed_form_bond_option_needs_a_volatility():
    from regimelib._engine.bond_option_explicit import call
    with pytest.raises(ValueError, match="positive volatility"):
        call(kappa=0.5, th=[0.03, 0.03], sig=[0.0, 0.0], x0=0.03, T=1.0, S=3.0, K=0.9, lam=40.0)


def test_first_order_diagnostics_do_not_depend_on_labels():
    def run(Q, sigma, belief):
        model = rl.SwitchingCEVProcess(rl.RegimeChain(Q), S0=100.0, r=0.02, q=0.0, sigma=sigma, beta=0.6)
        option = rl.VanillaOption(("call", 100.0), maturity=1.0)
        engine = rl.FirstOrderFDEngine(model, regime=belief, n=201); option.setPricingEngine(engine)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            option.NPV()
        return engine.diagnostics
    one, other = run([[-2.0, 2.0], [3.0, -3.0]], [2.5, 1.2], [0.3, 0.7]), run([[-3.0, 3.0], [2.0, -2.0]], [1.2, 2.5], [0.7, 0.3])
    for key in one:
        assert one[key] == pytest.approx(other[key], rel=1e-8)


def test_g2_slow_factor_beside_a_fast_one():
    strike = math.exp(-0.03)
    moderate = rl.SwitchingG2(CHAIN, 0.03, 1e-8, 0.01, 20.0, 0.02, 0.0)
    bond = rl.ZeroCouponBond(2.0); bond.setPricingEngine(rl.NumericalSwitchingEngine(moderate))
    assert bond.NPV() == pytest.approx(math.exp(-0.06), abs=1e-10)       # identical regimes: the supplied curve
    extreme = rl.SwitchingG2(CHAIN, 0.03, 1e-8, 1e-8, 1000.0, 1000.0, 0.0)
    option = rl.ZeroCouponBondOption("call", strike, 1.0, 2.0); option.setPricingEngine(rl.NumericalSwitchingEngine(extreme))
    with pytest.raises(NotImplementedError, match="not resolved"):
        option.NPV()


def test_vol_of_vol_grid_reaches_the_transient_variance():
    model = rl.SwitchingHestonVolOfVol(chain=CHAIN, S0=100.0, r=0.03, q=0.0, v0=0.04, kappa=0.1, theta=0.0, xi=2.0, rho=-0.6)
    option = rl.VanillaOption(("call", 100.0), maturity=1.0)
    _, _, _, grid, _, _ = model.operators(option, (21, 21), None)
    assert grid.V.max() > 0.04 + 4 * math.sqrt(0.04 * 4.0 * 0.9)         # v0 + sd of v_T, not 5 v0


def test_transition_matrix_error_says_what_was_tested():
    from scipy.linalg import expm
    Q = np.array([[-5.0, 5.0, 0.0], [0.0, -9.0, 9.0], [10.0, 0.0, -10.0]])
    with pytest.raises(ValueError, match="another branch"):
        rl.RegimeChain.fromTransitionMatrix(expm(Q * 0.5), 0.5)


def test_bermudan_rate_grid_is_sized_for_the_last_exercise():
    lead = 1.0 / 365.0
    exercises, payments = [lead + i for i in range(11)], [lead + i for i in range(1, 12)]
    for model, wide in ((rl.SwitchingVasicek(CHAIN, r0=0.03, a=0.05, b=0.03, sigma=0.01), 2.10706),
                        (rl.SwitchingCoxIngersollRoss(CHAIN, r0=0.03, theta=0.03, k=0.05, sigma=0.10), 4.32565)):
        swaption = rl.Swaption("payer", lead, payments, fixedRate=0.04, notional=100.0, exerciseTimes=exercises)
        swaption.setPricingEngine(rl.SwitchingFDEngine(model, n=401, steps=400))
        assert swaption.NPV() == pytest.approx(wide, rel=5e-4)           # the default grid returned 0 and 1.196


def test_cir_bond_under_a_change_of_time_unit():
    def price(c):
        model = rl.SwitchingCoxIngersollRoss(ONE, r0=c * 0.03, theta=[c * 0.04], k=c * 0.5, sigma=c * 0.1)
        bond = rl.ZeroCouponBond(2.0 / c); bond.setPricingEngine(rl.NumericalSwitchingEngine(model)); return bond.NPV()
    for c in (1e-150, 4e154):
        assert price(c) == pytest.approx(0.93506311024783, rel=1e-10)
    with pytest.raises(ValueError, match="underflows"):
        price(1e-162)


def test_two_state_exact_keeps_a_rare_transition():
    from regimelib._engine.explicit import two_state_constant_exact
    for lam in (1e-12, 1e-16, 1e-18):                        # a(t) = lam e^{-lam t} + e^{-(1 + lam) t} + O(lam^2)
        assert two_state_constant_exact(-0.5, -0.5, lam, 100.0, sign=+1).real == pytest.approx(lam, rel=1e-9)
    from scipy.linalg import expm
    for gbar, gt, lam, t in ((0.3 - 0.2j, 0.7 + 0.1j, 2.0, 1.5), (-1.0, 0.01j, 50.0, 3.0)):
        M = np.array([[gbar + gt - lam, lam], [lam, gbar - gt - lam]]); reference = expm(M * t) @ np.ones(2)
        for sign, k in ((+1, 0), (-1, 1)):
            assert two_state_constant_exact(gbar, gt, lam, t, sign=sign) == pytest.approx(reference[k], rel=1e-11)


def test_g2_convexity_beyond_the_floating_range_is_refused():
    wild = rl.SwitchingG2(CHAIN, 0.03, 0.03, 0.50, 0.05, 0.50, 0.0)       # the fitted curve is e^{-1025} times e^{1025}
    for instrument in (rl.ZeroCouponBond(30.0), rl.ZeroCouponBondOption("call", 0.5, 5.0, 30.0)):
        instrument.setPricingEngine(rl.NumericalSwitchingEngine(wild))
        with pytest.raises(ArithmeticError, match="convexity"):
            instrument.NPV()
    plain = rl.SwitchingG2(CHAIN, 0.03, 0.03, 0.02, 0.05, 0.02, 0.0)
    bond = rl.ZeroCouponBond(30.0); bond.setPricingEngine(rl.NumericalSwitchingEngine(plain))
    assert bond.NPV() == pytest.approx(math.exp(-0.9), abs=1e-10)


def test_hybrid_digitals_at_the_ends_are_digitals():
    T = 1.0

    def value(option, engine):
        option.setPricingEngine(engine); return option.NPV()
    equity = rl.SwitchingBlackScholesProcess(CHAIN, S0=100.0, r=0.0, q=0.01, sigma=[0.30, 0.15])
    rates = rl.SwitchingVasicek(CHAIN, r0=0.03, a=0.5, b=[0.06, 0.02], sigma=[0.015, 0.008])
    engine = rl.NumericalSwitchingEngine(rl.SwitchingEquityRates(equity, rates))
    P = value(rl.ZeroCouponBond(T), rl.NumericalSwitchingEngine(rates))
    assert value(rl.VanillaOption(("cash", "call", 0.0, 10.0), maturity=T), engine) == pytest.approx(10.0 * P, rel=1e-10)
    assert value(rl.VanillaOption(("asset", "call", 0.0), maturity=T), engine) == pytest.approx(100.0 * math.exp(-0.01), rel=1e-12)
    assert value(rl.VanillaOption(("cash", "put", 0.0, 10.0), maturity=T), engine) == 0.0
    still = rl.NumericalSwitchingEngine(rl.SwitchingEquityRates(
        rl.SwitchingBlackScholesProcess(CHAIN, S0=100.0, r=0.0, q=0.01, sigma=0.0),
        rl.SwitchingVasicek(CHAIN, r0=0.03, a=0.5, b=0.03, sigma=0.0)))
    ST, disc = 100.0 * math.exp(0.02), math.exp(-0.03)                   # the terminal price is known
    for payoff, expected in ((("cash", "call", 100.0, 10.0), 10.0 * disc), (("asset", "call", 100.0), disc * ST),
                             (("cash", "put", 105.0, 10.0), 10.0 * disc), (("asset", "put", 105.0), disc * ST),
                             (("asset", "put", 101.0), 0.0), (("call", 100.0), disc * (ST - 100.0))):
        assert value(rl.VanillaOption(payoff, maturity=T), still) == pytest.approx(expected, abs=1e-9)


def test_a_curve_that_changes_behind_its_callable_is_not_cached():
    rate = [0.01]
    model = rl.SwitchingVasicek(ONE, r0=0.01, a=0.5, b=0.12, sigma=0.0)
    times = [0.25 * i for i in range(1, 21)]
    cds = rl.CreditDefaultSwap("buyer", spread=0.02, times=times, recovery=0.4, discount=lambda t: math.exp(-rate[0] * t))
    cds.setPricingEngine(rl.NumericalSwitchingEngine(model))
    cds.NPV(); before = cds.fairSpread()
    rate[0] = 0.05                                                       # the same function, another curve
    fresh = rl.CreditDefaultSwap("buyer", 0.02, times, 0.4, discount=0.05)
    fresh.setPricingEngine(rl.NumericalSwitchingEngine(model)); fresh.NPV()
    assert cds.fairSpread() == pytest.approx(fresh.fairSpread(), rel=1e-12) and cds.fairSpread() != before
    kept = fresh._results; fresh.fairSpread()                            # a curve given as a rate is compared by it
    assert fresh._results is kept
