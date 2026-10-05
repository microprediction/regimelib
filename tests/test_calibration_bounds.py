"""Default calibration bounds follow each parameter's domain. Reported by a contributor in #131 against #16: one lower
bound of 1e-8 for every parameter rejected correlations, rates and drifts that are legitimately zero or negative."""
import numpy as np
import pytest
import regimelib as rl
from regimelib.calibration import defaultBounds

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)


def helpers(truth, strikes=(90.0, 100.0, 110.0), maturities=(1.0,), regime=0):
    engine = rl.NumericalSwitchingEngine(truth, regime=regime)
    out = []
    for T in maturities:
        for K in strikes:
            o = rl.VanillaOption(("call", K), maturity=T); o.setPricingEngine(engine)
            out.append(rl.VolatilityHelper(T, K, o.impliedVolatility()))
    return out


def test_domains_by_parameter_and_model():
    heston = rl.SwitchingHestonModel(CHAIN, 100.0, 0.02, 0.0, 0.04, 2.0, 0.04, 0.3, -0.5)
    gamma = rl.SwitchingVarianceGammaProcess(CHAIN, 100.0, 0.02, 0.0, 0.2, 0.5, -0.1)
    vasicek = rl.SwitchingVasicek(CHAIN, 0.03, 0.5, 0.05, 0.01)
    g2 = rl.SwitchingG2(CHAIN, 0.03, 0.5, 0.01, 0.1, 0.01, -0.5)
    assert defaultBounds(heston, "rho") == (-1.0, 1.0)
    assert defaultBounds(heston, "theta") == (0.0, np.inf) and defaultBounds(gamma, "theta") == (-np.inf, np.inf)
    assert defaultBounds(vasicek, "b") == (-np.inf, np.inf) and defaultBounds(g2, "b")[0] > 0
    assert defaultBounds(heston, "sigma") == (0.0, np.inf) and defaultBounds(heston, "kappa")[0] > 0
    assert defaultBounds(heston, "r") == (-np.inf, np.inf) and defaultBounds(heston, "chain") == (0.0, np.inf)


def test_a_negative_correlation_is_an_admissible_start_and_stays_in_range():
    truth = rl.SwitchingHestonModel(CHAIN, 100.0, 0.02, 0.0, 0.04, 2.0, [0.06, 0.03], 0.4, -0.6)
    model = rl.SwitchingHestonModel(CHAIN, 100.0, 0.02, 0.0, 0.04, 2.0, [0.06, 0.03], 0.4, -0.2)
    rl.calibrate(model, helpers(truth, strikes=(100.0,)), ["rho"], max_nfev=1)   # raised before: -0.2 < 1e-8
    assert -1.0 <= model.rho <= 1.0


def test_a_negative_rate_and_a_negative_drift_are_admissible_starts():
    quotes = [rl.VolatilityHelper(1.0, 100.0, 0.2)]
    bs = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, -0.01, 0.0, [0.3, 0.15])
    rl.calibrate(bs, quotes, ["r"], max_nfev=1)
    gamma = rl.SwitchingVarianceGammaProcess(CHAIN, 100.0, 0.02, 0.0, 0.2, 0.5, -0.1)
    rl.calibrate(gamma, quotes, ["theta"], max_nfev=1)
    assert gamma.theta[0] < 0


def test_volatilities_and_switching_rates_cannot_go_negative_without_bounds():
    truth = rl.SwitchingBlackScholesProcess(rl.RegimeChain.twoState(4.0, 2.0), 100.0, 0.02, 0.0, [0.40, 0.15])
    quotes = helpers(truth, strikes=(80.0, 100.0, 120.0), maturities=(0.25, 1.0), regime=1)
    model = rl.SwitchingBlackScholesProcess(rl.RegimeChain.twoState(3.0, 3.0), 100.0, 0.02, 0.0, [0.30, 0.20])
    before = sum(h.calibrationError() ** 2 for h in _attach(quotes, model))
    rl.calibrate(model, quotes, ["sigma", "chain"], engine=lambda m: rl.NumericalSwitchingEngine(m, regime=1), max_nfev=8)
    after = sum(h.calibrationError() ** 2 for h in _attach(quotes, model))
    assert after < before                                                        # the fit moved towards the quotes
    assert min(model.sigma) >= 0 and model.chain.generator[0, 1] >= 0 and model.chain.generator[1, 0] >= 0
    still = rl.SwitchingBlackScholesProcess(rl.RegimeChain([[0.0, 0.0], [2.0, -2.0]]), 100.0, 0.02, 0.0, [0.3, 0.2])
    rl.calibrate(still, quotes, ["chain"], max_nfev=1)                           # a closed transition is a valid start


def _attach(quotes, model):
    engine = rl.NumericalSwitchingEngine(model, regime=1)
    for h in quotes:
        h.setPricingEngine(engine)
    return quotes


def test_explicit_bounds_override_and_an_infeasible_start_is_named():
    quotes = [rl.VolatilityHelper(1.0, 100.0, 0.2)]
    model = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.02, 0.0, [0.3, 0.15])
    rl.calibrate(model, quotes, ["sigma"], bounds=([0.1, 0.1], [0.5, 0.5]), max_nfev=3)
    assert all(0.1 <= s <= 0.5 for s in model.sigma)
    with pytest.raises(ValueError, match="sigma is outside its bounds"):
        rl.calibrate(rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.02, 0.0, [0.3, 0.15]), quotes, ["sigma"], bounds=([0.2, 0.2], [0.5, 0.5]))
