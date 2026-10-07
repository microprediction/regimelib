"""Laws the transform engines cannot invert, forcings a fitted series misses, and intermediates that depend on units."""
import math
import warnings

import pytest

import regimelib as rl

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)


def test_jumps_without_diffusion_are_refused_by_the_transform_engines():
    # with no diffusion the event of no jump is an atom, which a Fourier range that stops does not resolve
    model = rl.SwitchingMerton76Process(CHAIN, S0=100.0, r=0.02, q=0.01, sigma=0.0, jumpIntensity=0.7,
                                        logJumpMean=0.1, logJumpVol=0.2)
    for payoff in (("call", 100.0), ("cash", "call", 100.0, 10.0)):
        option = rl.VanillaOption(payoff, maturity=1.0)
        option.setPricingEngine(rl.NumericalSwitchingEngine(model))
        with pytest.raises(ArithmeticError, match="atom"):
            option.NPV()


def test_variance_gamma_too_peaked_to_invert_is_refused_and_an_ordinary_one_is_not():
    peaked = rl.SwitchingVarianceGammaProcess(CHAIN, 100.0, 0.02, 0.01, sigma=1e-4, nu=1.0, theta=0.0)
    option = rl.VanillaOption(("call", 100.0 * math.exp(0.01)), maturity=1.0)
    option.setPricingEngine(rl.NumericalSwitchingEngine(peaked))
    with pytest.raises(ArithmeticError, match="largest frequency"):
        option.NPV()
    ordinary = rl.SwitchingVarianceGammaProcess(CHAIN, 100.0, 0.02, 0.01, sigma=[0.2, 0.3], nu=0.2, theta=-0.1)
    option.setPricingEngine(rl.NumericalSwitchingEngine(ordinary))
    assert option.NPV() > 0.0 and math.isfinite(option.gamma())


def test_variance_gamma_gamma_is_not_reported_where_the_density_is_not_resolved():
    model = rl.SwitchingVarianceGammaProcess(CHAIN, 100.0, 0.02, 0.01, sigma=0.2, nu=1.0, theta=0.0)
    option = rl.VanillaOption(("call", 100.0), maturity=0.45)            # 2 T / nu < 1: the density is infinite at its mode
    option.setPricingEngine(rl.NumericalSwitchingEngine(model))
    assert option.NPV() > 0.0 and math.isfinite(option.delta())
    with pytest.raises(RuntimeError, match="gamma"):
        option.gamma()


@pytest.mark.parametrize("build", [
    lambda **k: rl.SwitchingHestonModel(**k),
    lambda **k: rl.SwitchingBatesModel(**k, jumpIntensity=[1.0, 2.0], logJumpMean=-0.05, logJumpVol=0.1)])
def test_heston_forcing_at_the_martingale_argument_with_rho_xi_above_kappa(build):
    model = build(chain=CHAIN, S0=100.0, r=0.02, q=0.01, v0=0.04, kappa=0.5, theta=[0.04, 0.08], sigma=1.0, rho=0.9)
    g, gfuncs, pre = model.returnForcing(-1j, 1.0)
    assert abs(pre() - 1.0) < 1e-12 and all(abs(f(0.7)) < 1e-12 for f in gfuncs)      # E exp(X_T) = 1


def test_jump_mean_that_overflows_is_named():
    model = rl.SwitchingMerton76Process(CHAIN, S0=100.0, r=0.0, q=0.0, sigma=0.2, jumpIntensity=1e-300,
                                        logJumpMean=710.0, logJumpVol=0.0)
    option = rl.VanillaOption(("call", 100.0), maturity=1.0); option.setPricingEngine(rl.NumericalSwitchingEngine(model))
    with pytest.raises(ValueError, match="overflows"):
        option.NPV()


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_vanilla_prices_and_greeks_do_not_depend_on_the_unit_of_price(scale):
    def results(s):
        model = rl.SwitchingBlackScholesProcess(CHAIN, S0=100.0 * s, r=0.0, q=0.0, sigma=[0.2, 0.25])
        option = rl.VanillaOption(("call", 105.0 * s), maturity=1.0)
        option.setPricingEngine(rl.NumericalSwitchingEngine(model))
        return option.NPV() / s, option.delta(), option.gamma() * s
    assert results(scale) == pytest.approx(results(1.0), rel=1e-10)


def test_fast_engine_solves_where_a_fitted_forcing_is_not_resolved():
    # the jump forcing 1 / (1 + m B(t)) - 1 has a layer of width 1 / m at zero that a degree-80 series misses
    model = rl.SwitchingVasicekJumps(CHAIN, r0=0.0, a=0.5, b=[0.0, 0.0], sigma=[0.0, 0.0], jumpIntensity=[1.0, 1.0],
                                     jumpMean=6950.021277048663)
    bond = rl.ZeroCouponBond(1.0)
    bond.setPricingEngine(rl.NumericalSwitchingEngine(model)); exact = bond.NPV()
    bond.setPricingEngine(rl.FastSwitchingEngine(model, order=2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rl.ExpansionWarning)
        assert bond.NPV() == pytest.approx(exact, rel=1e-9)
    assert bond._result("diagnostics")["numericalNodes"] == 1


def test_mixed_basket_integrates_the_names_own_forcings():
    zero = rl.SwitchingVasicek(CHAIN, r0=0.0, a=0.5, b=[0.0, 0.0], sigma=[0.0, 0.0])
    jumps = rl.SwitchingVasicekJumps(CHAIN, r0=0.0, a=0.5, b=[0.0, 0.0], sigma=[0.0, 0.0], jumpIntensity=[1.0, 1.0],
                                     jumpMean=6950.021277048663)
    alone, both = rl.ZeroCouponBond(1.0), rl.ZeroCouponBond(1.0)
    alone.setPricingEngine(rl.NumericalSwitchingEngine(jumps))
    both.setPricingEngine(rl.NumericalSwitchingEngine(rl.SwitchingIntensityBasket([zero, jumps])))
    assert both.NPV() == pytest.approx(alone.NPV(), rel=1e-10)


def test_vol_of_vol_grid_reaches_the_variance_it_must_hold():
    model = rl.SwitchingHestonVolOfVol(CHAIN, 100.0, 0.02, 0.0, v0=0.04, kappa=1.0, theta=0.04, xi=[1.5, 1.5], rho=-0.5)
    option = rl.VanillaOption(("call", 100.0), maturity=3.0)
    _, _, _, grid, _, _ = model.operators(option, (41, 21), None)
    assert grid.V.max() >= 0.04 + 5 * math.sqrt(0.04 * 1.5 ** 2 / 2.0) - 1e-12
