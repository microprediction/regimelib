"""Exact Monte Carlo option expiry endpoint (#334), without broadening feature support."""
import numpy as np
import pytest
import regimelib as rl


def model():
    return rl.SwitchingBlackScholesProcess(rl.RegimeChain.twoState(3., 5.),
                                          100., .03, .01, [.2, .4])


def payoff(kind, side, strike):
    if kind == "cash":
        return (kind, side, strike, 7.)
    if kind == "asset":
        return (kind, side, strike)
    return (side, strike)


@pytest.mark.parametrize("kind", ["vanilla", "cash", "asset"])
@pytest.mark.parametrize("side", ["call", "put"])
@pytest.mark.parametrize("strike", [-1., 0., 90., 100., 110.])
@pytest.mark.parametrize("regime", [0, 1, [.25, .75]])
def test_expiry_exact_payoff_and_no_sampling(kind, side, strike, regime):
    opt = rl.VanillaOption(payoff(kind, side, strike), maturity=0.)
    eng = rl.MonteCarloSwitchingEngine(model(), regime=regime, paths=2, seed=1)
    def no_paths(T):
        raise AssertionError("expiry must not sample regime paths")
    eng._paths = no_paths
    opt.setPricingEngine(eng)
    intrinsic = (100. - strike) if side == "call" else (strike - 100.)
    expected = max(intrinsic, 0.) if kind == "vanilla" else (7. if kind == "cash" else 100.) * (intrinsic > 0.)
    assert opt.NPV() == expected
    assert eng.standardError == 0.
    if isinstance(regime, list):
        assert eng.regime is None
        np.testing.assert_array_equal(eng.belief, regime)


@pytest.mark.parametrize("paths, seed", [(2, 1), (1000, 42), (2, None)])
def test_expiry_clears_previous_standard_error(paths, seed):
    eng = rl.MonteCarloSwitchingEngine(model(), paths=paths, seed=seed)
    eng.calculate(rl.VanillaOption(("call", 100.), maturity=1.))
    assert eng.standardError > 0.
    assert eng.calculate(rl.VanillaOption(("call", 90.), maturity=0.)) == 10.
    assert eng.standardError == 0.


@pytest.mark.parametrize("feature", ["american", "barrier", "asian"])
def test_expiry_does_not_bypass_unsupported_features(feature):
    if feature == "american":
        opt = rl.VanillaOption(("put", 110.), exercise="american", maturity=0.)
    elif feature == "barrier":
        opt = rl.BarrierOption("upout", 120., 3., ("call", 90.), maturity=0.)
    else:
        opt = rl.ContinuousGeometricAsianOption(("call", 90.), maturity=0.)
    with pytest.raises(TypeError, match="does not price"):
        rl.MonteCarloSwitchingEngine(model(), paths=2).calculate(opt)


@pytest.mark.parametrize("maturity", [-1., float("nan"), float("inf")])
def test_invalid_mutated_maturity_still_refused(maturity):
    opt = rl.VanillaOption(("call", 90.), maturity=1.)
    opt.maturity = maturity
    with pytest.raises(ValueError, match="finite and nonnegative"):
        rl.MonteCarloSwitchingEngine(model(), paths=2).calculate(opt)


def test_unsupported_model_still_refused_at_expiry():
    m = rl.SwitchingVasicek(rl.RegimeChain.twoState(3., 5.), .03, .5, [.03, .04], [.01, .02])
    with pytest.raises(TypeError, match="covers"):
        rl.MonteCarloSwitchingEngine(m, paths=2).calculate(rl.VanillaOption(("call", 90.), maturity=0.))
