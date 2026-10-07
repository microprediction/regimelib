"""Inputs refused with a reason, tolerances that scale with the contract, and code with no caller removed."""
import math
import warnings

import pytest

import regimelib as rl

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)


@pytest.mark.parametrize("maturity", [-1.0, math.nan, math.inf])
def test_monte_carlo_refuses_a_maturity_that_is_not_one(maturity):
    model = rl.SwitchingBlackScholesProcess(CHAIN, S0=100.0, r=0.05, q=0.0, sigma=[0.2, 0.2])
    option = rl.VanillaOption(("put", 100.0), maturity=maturity)
    option.setPricingEngine(rl.MonteCarloSwitchingEngine(model, paths=10, seed=1))
    with pytest.raises(ValueError, match="maturity"):
        option.NPV()


def test_two_state_formula_names_coalescing_eigenvalues():
    from regimelib.symbolic import TwoStateConstantForcing
    formula = TwoStateConstantForcing()
    with pytest.raises(ValueError, match="eigenvalues"):
        formula.evaluate(formula.blackScholes(0), q12=1.5, q21=6.0, u=-0.75, sigma1=4.0, sigma2=0.0, T=1.5)


def test_prototypes_without_callers_are_gone():
    import regimelib._engine.models as models
    import regimelib._engine.options as options
    assert not hasattr(models, "fast_factor") and not hasattr(options, "bs_call")


@pytest.mark.parametrize("scale", [1.0, 1e6, 1e10])
def test_coupon_bond_option_scales_with_its_notional(scale):
    model = rl.SwitchingVasicek(CHAIN, r0=0.03, a=0.5, b=[0.03, 0.03], sigma=[0.20, 0.20])
    def value(n):
        option = rl.CouponBondOption("call", 0.95 * n, 1.0, [(2.0, 0.05 * n), (3.0, 1.05 * n)])
        option.setPricingEngine(rl.NumericalSwitchingEngine(model, information="observed")); return option.NPV()
    assert value(scale) == pytest.approx(scale * value(1.0), rel=1e-8)


def test_barrier_grid_that_does_not_hold_the_spot_is_refused():
    model = rl.SwitchingBlackScholesProcess(CHAIN, S0=100.0, r=0.0, q=0.0, sigma=0.01)
    for kind, barrier in (("downout", 50.0), ("upout", 200.0), ("downin", 50.0)):
        option = rl.BarrierOption(kind, barrier=barrier, rebate=0.0, payoff=("call", 100.0), maturity=1.0)
        option.setPricingEngine(rl.SwitchingFDEngine(model, n=401, steps=200))
        with pytest.raises(ValueError, match="does not hold the spot"):
            option.NPV()


def test_default_correlation_is_refused_where_survival_is_not_resolved():
    m1 = rl.SwitchingVasicek(CHAIN, r0=1.0, a=0.5, b=[1.0, 1.0], sigma=[0.0, 0.0])
    m2 = rl.SwitchingVasicek(CHAIN, r0=1.5, a=0.7, b=[1.5, 1.5], sigma=[0.0, 0.0])
    basket = rl.SwitchingIntensityBasket([m1, m2])
    with pytest.raises(ValueError, match="no variance to correlate"):
        basket.defaultCorrelation(45.0)
    assert abs(basket.defaultCorrelation(1.0)) < 1e-8                     # deterministic intensities: independent defaults


def test_explicit_bond_option_formula_names_its_range():
    from regimelib._engine.bond_option_explicit import call
    args = (0.5, [0.08, 0.02], [0.03, 0.008], 0.04)
    with pytest.raises(ValueError, match="outer expansion"):
        call(*args, 0.001, 4.0, 0.85, 100.0, start=1)
    assert call(*args, 1.0, 4.0, 0.85, 100.0, start=1) > 0.0
