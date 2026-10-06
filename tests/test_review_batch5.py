"""Formulas evaluated where their terms cancel, bond options on bonds maturing just after expiry, the fast engine's
bond options where the series blows up, and one set of warnings and diagnostics per instrument."""
import math
import warnings

import numpy as np
import pytest

import regimelib as rl

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)


def _first_order_bond(model, T):
    bond = rl.ZeroCouponBond(T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rl.ExpansionWarning)
        bond.setPricingEngine(rl.FastSwitchingEngine(model, order=1)); return bond.NPV()


@pytest.mark.parametrize("kappa", [0.5, 1e-3, 2.5e-5])
def test_symbolic_vasicek_at_small_reversion(kappa):
    from regimelib.symbolic import VasicekBondFirstOrder
    thetas, sigmas, T = [0.06, 0.02], [0.015, 0.008], 4.0
    formula = VasicekBondFirstOrder()
    value = formula.evaluate(formula.price, r0=0.03, kappa=kappa, T=T, **formula.coefficients(CHAIN, kappa, thetas, sigmas))
    assert value == pytest.approx(_first_order_bond(rl.SwitchingVasicek(CHAIN, 0.03, kappa, thetas, sigmas), T), rel=1e-8)


def test_symbolic_two_state_vasicek_at_small_reversion():
    from regimelib.symbolic import VasicekTwoStateBond
    formula = VasicekTwoStateBond()
    values = dict(r0=0.03, theta1=0.06, theta2=0.02, sigma1=0.015, sigma2=0.008, lam=4.0, T=4.0)
    near = formula.evaluate(formula.price, kappa=1e-4, **values)
    assert near == pytest.approx(formula.evaluate(formula.price, kappa=1.01e-4, **values), rel=1e-5)   # smooth, not noise
    with pytest.raises(ValueError, match="kappa T"):
        formula.evaluate(formula.price, kappa=1e-6, **values)


@pytest.mark.parametrize("sigma", [0.1, 1e-3, 2.5e-5])
def test_symbolic_cir_at_small_volatility(sigma):
    from regimelib.symbolic import CIRBondFirstOrder
    k, thetas, T = 0.6, [0.05, 0.03], 4.0
    formula = CIRBondFirstOrder()
    value = formula.evaluate(formula.price, r0=0.03, k=k, sigma=sigma, T=T, **formula.coefficients(CHAIN, k, thetas))
    model = rl.SwitchingCoxIngersollRoss(CHAIN, r0=0.03, theta=thetas, k=k, sigma=sigma)
    assert value == pytest.approx(_first_order_bond(model, T), rel=1e-8)


def test_symbolic_formulas_refuse_their_removable_limits():
    from regimelib.symbolic import CIRBondFirstOrder, VasicekBondFirstOrder
    cir = CIRBondFirstOrder()
    with pytest.raises(ValueError, match="sigma T = 0"):
        cir.evaluate(cir.price, r0=0.03, k=0.6, sigma=0.0, T=4.0, **cir.coefficients(CHAIN, 0.6, [0.04, 0.04]))
    vasicek = VasicekBondFirstOrder()
    with pytest.raises(ValueError, match="FastSwitchingEngine"):
        vasicek.evaluate(vasicek.price, r0=0.03, kappa=1e-6, T=4.0, **vasicek.coefficients(CHAIN, 1e-6, [0.06, 0.02], [0.015, 0.008]))


def test_symbolic_jump_vasicek_at_small_parameters():
    from regimelib.symbolic import VasicekJumpsBondFirstOrder
    thetas, sigmas, intensities, T = [0.04, 0.03], [0.012, 0.008], [2.0, 0.5], 3.0
    formula = VasicekJumpsBondFirstOrder()
    for kappa, mean in ((0.5, 0.004), (1e-4, 0.004), (0.5, 1e-4)):
        coefficients = formula.coefficients(CHAIN, kappa, thetas, sigmas, intensities)
        value = formula.evaluate(formula.price, r0=0.03, kappa=kappa, m=mean, T=T, **coefficients)
        model = rl.SwitchingVasicekJumps(CHAIN, 0.03, kappa, thetas, sigmas, intensities, mean)
        assert value == pytest.approx(_first_order_bond(model, T), rel=1e-7)
    with pytest.raises(ValueError, match="m T = 0"):
        formula.evaluate(formula.price, r0=0.03, kappa=0.5, m=0.0, T=T, **formula.coefficients(CHAIN, 0.5, thetas, sigmas, intensities))


VASICEK = rl.SwitchingVasicek(CHAIN, 0.03, 0.5, [0.06, 0.02], [0.015, 0.008])
G2 = rl.SwitchingG2(CHAIN, 0.03, 0.5, 0.012, 0.1, 0.009, -0.6)


@pytest.mark.parametrize("model", [VASICEK, G2])
@pytest.mark.parametrize("strike", [0.9, 1.1])
def test_bond_option_on_a_bond_maturing_just_after_expiry(model, strike):
    # the exercise boundary recedes as 1 / (S - T): exercise is settled, and the price is the forward intrinsic value
    T, S = 1.0, 1.000001
    for engine in (rl.NumericalSwitchingEngine(model, information="observed"),
                   rl.FastSwitchingEngine(model, order=2, information="observed")):
        bond = lambda t: (lambda b: (b.setPricingEngine(engine), b.NPV())[1])(rl.ZeroCouponBond(t))
        forward = bond(S) - strike * bond(T)
        for kind, sign in (("call", 1.0), ("put", -1.0)):
            option = rl.ZeroCouponBondOption(kind, strike, T, S); option.setPricingEngine(engine)
            assert option.NPV() == pytest.approx(max(sign * forward, 0.0), abs=1e-8)
            assert option.NPV() >= 0.0


def test_settled_exercise_agrees_with_the_integral_where_both_apply():
    from regimelib._engine.options import settled, zcb_call
    assert settled([10.0, -10.0, 0.5], 0.0, 1.0, 0.01, 2.0) == [1.0, 0.0, None]
    args = (1.0, 3.0, 0.5, 0.03, 0, 0.5, [0.06, 0.02], [0.015, 0.008], CHAIN.generator)      # far in the money
    bond = lambda t: (lambda b: (b.setPricingEngine(rl.NumericalSwitchingEngine(VASICEK)), b.NPV())[1])(rl.ZeroCouponBond(t))
    assert zcb_call(*args) == pytest.approx(bond(3.0) - 0.5 * bond(1.0), abs=1e-10)


def test_fast_cap_prices_where_the_series_blows_up():
    chain = rl.RegimeChain.twoState(12.0, 8.0)
    model = rl.SwitchingVasicek(chain, r0=0.031, a=0.4, b=[0.065, 0.018], sigma=[0.019, 0.007])
    cap = rl.CapFloor("cap", times=[0.5, 1.25], strike=0.027)
    cap.setPricingEngine(rl.NumericalSwitchingEngine(model, information="observed")); reference = cap.NPV()
    for order, rel in ((1, 2e-3), (2, 1e-3), (4, 1e-6), (None, 1e-3)):
        cap.setPricingEngine(rl.FastSwitchingEngine(model, order=order, maxOrder=8, information="observed"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", rl.ExpansionWarning)
            assert cap.NPV() == pytest.approx(reference, rel=rel)


def test_cds_warns_once_and_prices_each_date_once():
    calls = []

    class Counting(rl.FastSwitchingEngine):
        def _survival(self, t):
            calls.append(t); return super()._survival(t)

    model = rl.SwitchingVasicek(rl.RegimeChain.twoState(0.2, 0.3), 0.03, 0.5, [0.10, 0.00], [0.02, 0.005])
    times = [0.5 * i for i in range(1, 11)]
    cds = rl.CreditDefaultSwap("buyer", 0.01, times, 0.4, discount=0.03)
    cds.setPricingEngine(Counting(model, order=1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cds.NPV()
    assert calls == times
    assert len([w for w in caught if issubclass(w.category, rl.ExpansionWarning)]) == 1
    worst = cds._result("diagnostics")["lastTermRelative"]

    def lastTerm(t):                                                     # the diagnostics are of the whole instrument
        bond = rl.ZeroCouponBond(t); bond.setPricingEngine(rl.FastSwitchingEngine(model, order=1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", rl.ExpansionWarning)
            bond.NPV()
        return bond._result("diagnostics")["lastTermRelative"]
    assert worst == pytest.approx(max(lastTerm(t) for t in times))


def test_adaptive_order_reports_the_order_it_kept():
    model = rl.SwitchingVasicek(rl.RegimeChain.twoState(0.2, 0.3), 0.03, 0.5, [0.10, 0.00], [0.02, 0.005])
    bond = rl.ZeroCouponBond(5.0)
    adaptive = rl.FastSwitchingEngine(model, order=None); bond.setPricingEngine(adaptive)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rl.ExpansionWarning)
        value, kept = bond.NPV(), bond._result("diagnostics")
        fixed = rl.ZeroCouponBond(5.0); fixed.setPricingEngine(rl.FastSwitchingEngine(model, order=adaptive.orderUsed))
        assert fixed.NPV() == pytest.approx(value, rel=1e-12)
        assert fixed._result("diagnostics")["lastTermRelative"] == pytest.approx(kept["lastTermRelative"])


@pytest.mark.slow
def test_fast_heston_fallback_raises_no_numpy_warnings():
    chain = rl.RegimeChain([[-5, 3, 2], [4, -9, 5], [1, 6, -7]])
    model = rl.SwitchingHestonModel(chain, S0=100.0, r=0.02, q=0.0, v0=0.04, kappa=1.5, theta=[0.09, 0.05, 0.02], sigma=0.4, rho=-0.6)
    option = rl.VanillaOption(("call", 100.0), maturity=1.0)
    option.setPricingEngine(rl.FastSwitchingEngine(model, order=4, regime=1))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        warnings.simplefilter("ignore", rl.ExpansionWarning)
        assert math.isfinite(option.NPV()) and option._result("diagnostics")["numericalNodes"] > 0


def _bond(model, T, engine):
    bond = rl.ZeroCouponBond(T); bond.setPricingEngine(engine)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rl.ExpansionWarning)
        return bond.NPV(), (bond._result("diagnostics") if isinstance(engine, rl.FastSwitchingEngine) else None)


def test_adaptive_order_keeps_a_numerical_fallback():
    # at order one the series has a term nearly the size of the value: the reduced system is solved, and that is kept
    model = rl.SwitchingVasicek(rl.RegimeChain.twoState(0.0125, 0.005), r0=0.03, a=1.0, b=[0.03, 0.16], sigma=[0.005, 0.015])
    exact = _bond(model, 2.0, rl.NumericalSwitchingEngine(model))[0]
    value, diagnostics = _bond(model, 2.0, rl.FastSwitchingEngine(model, order=None, tol=1e-14, maxOrder=8))
    assert value == pytest.approx(exact, rel=1e-8) and diagnostics["numericalNodes"] == 1
    for order in (1, 2, 3):
        assert _bond(model, 2.0, rl.FastSwitchingEngine(model, order=order))[0] == pytest.approx(exact, rel=1e-8)


@pytest.mark.parametrize("order", range(1, 9))
def test_an_expansion_that_underflows_to_zero_is_not_agreement(order):
    model = rl.SwitchingVasicek(rl.RegimeChain.twoState(0.01, 0.015), r0=0.03, a=0.5, b=[0.10, 0.00], sigma=[0.02, 0.005])
    exact = _bond(model, 8.0, rl.NumericalSwitchingEngine(model))[0]
    value, diagnostics = _bond(model, 8.0, rl.FastSwitchingEngine(model, order=order))
    assert value == pytest.approx(exact, rel=1e-8) and diagnostics["numericalNodes"] == 1


def test_adaptive_order_is_not_stopped_by_a_correction_that_cancels():
    # the first-order correction vanishes at this maturity from regime 0, and the second does not
    model = rl.SwitchingVasicek(rl.RegimeChain.twoState(1.0, 1.5), r0=0.03, a=0.5, b=[0.10, 0.00], sigma=[0.02, 0.005])
    T = 19.91629846577584
    zero, one = (_bond(model, T, rl.FastSwitchingEngine(model, order=n))[0] for n in (0, 1))
    exact = _bond(model, T, rl.NumericalSwitchingEngine(model))[0]
    assert one == pytest.approx(zero, rel=1e-12) and abs(one - exact) > 1e-5
    assert _bond(model, T, rl.FastSwitchingEngine(model, order=None))[0] == pytest.approx(exact, rel=1e-6)
