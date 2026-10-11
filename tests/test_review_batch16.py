"""Issues #262 to #269: a stale stationary law, parallel calibration, finite-horizon grids, the live G2 factor's
controls, the CEV Peclet test, Monte Carlo quadrature and the variance gamma domain in calibration."""
import math
import pytest
import regimelib as rl
from regimelib import montecarlo

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)
ONE = rl.RegimeChain([[0.0]])


def npv(instrument, engine):
    instrument.setPricingEngine(engine)
    return instrument.NPV()


def test_order_zero_theta_follows_a_replaced_chain():
    model = rl.SwitchingBlackScholesProcess(rl.RegimeChain.twoState(10.0, 10.0), S0=100.0, r=0.03, q=0.01, sigma=[0.5, 0.1])
    option = rl.VanillaOption(("call", 100.0), maturity=1.0)
    option.setPricingEngine(rl.FastSwitchingEngine(model, order=0, regime=0, nodes=128)); option.NPV()
    model.chain = rl.RegimeChain.twoState(1.0, 9.0)
    shared = option.theta()
    option.setPricingEngine(rl.FastSwitchingEngine(model, order=0, regime=0, nodes=128))
    assert shared == pytest.approx(option.theta(), rel=1e-12)


def test_calibration_refuses_parallel_workers():
    model = rl.SwitchingBlackScholesProcess(CHAIN, S0=100.0, r=0.02, q=0.0, sigma=[0.3, 0.2])
    with pytest.raises(ValueError, match="workers"):
        rl.calibrate(model, [rl.VolatilityHelper(0.5, 100.0, 0.25)], ["sigma"], workers=map)


def test_calibration_stays_inside_the_variance_gamma_domain():
    truth = rl.SwitchingVarianceGammaProcess(ONE, S0=100.0, r=0.02, q=0.0, sigma=0.2, nu=1.0, theta=0.85)
    helpers = []
    for K in (100.0, 110.0, 120.0):
        option = rl.VanillaOption(("call", K), maturity=1.0)
        option.setPricingEngine(rl.NumericalSwitchingEngine(truth))
        helpers.append(rl.VolatilityHelper(1.0, K, option.impliedVolatility()))
    model = rl.SwitchingVarianceGammaProcess(ONE, S0=100.0, r=0.02, q=0.0, sigma=0.2, nu=1.0, theta=0.5)
    rl.calibrate(model, helpers, ["theta"], max_nfev=30)               # overshot to theta = 1.01, past the domain, before
    model.checkParameters()
    assert model.theta[0] == pytest.approx(0.85, abs=1e-6)


def test_vasicek_grid_follows_the_finite_horizon_drift():
    r0, b, sigma = 0.03, 0.50, 0.001
    model = rl.SwitchingVasicek(ONE, r0, 0.02 / (b - r0), b, sigma)    # slow reversion to a distant level
    reference = rl.NumericalSwitchingEngine(model)
    K = npv(rl.ZeroCouponBond(3.0), reference) / npv(rl.ZeroCouponBond(1.0), reference)
    option = rl.ZeroCouponBondOption("call", K, 1.0, 3.0)
    assert npv(option, rl.SwitchingFDEngine(model)) == pytest.approx(npv(option, reference), rel=5e-3)  # 55% high before


@pytest.mark.parametrize("bermudan", [False, True])
def test_one_factor_g2_uses_the_live_factors_grid_controls(bermudan):
    vol = [0.02, 0.008]
    hw = rl.SwitchingHullWhite(CHAIN, 0.03, a=0.1, sigma=vol)
    x = rl.SwitchingG2(CHAIN, 0.03, a=0.1, sigma=vol, b=0.5, eta=0.0, rho=0.0)
    y = rl.SwitchingG2(CHAIN, 0.03, a=0.5, sigma=0.0, b=0.1, eta=vol, rho=0.0)
    swaption = lambda: rl.Swaption("payer", maturity=2.0, fixedTimes=[3.0, 4.0, 5.0, 6.0, 7.0], fixedRate=0.035,
                                   notional=100.0, exerciseTimes=[2.0, 3.0] if bermudan else None)
    fd = lambda m, n, w: rl.SwitchingFDEngine(m, n=n, width=w, steps=200, information="observed")
    expected = npv(swaption(), fd(hw, 401, 0.15))
    assert npv(swaption(), fd(x, (401, 11), (0.15, 0.01))) == pytest.approx(expected, rel=1e-12)
    assert npv(swaption(), fd(y, (11, 401), (0.01, 0.15))) == pytest.approx(expected, rel=1e-12)


def test_cev_grid_too_coarse_for_the_carry_is_refused():
    beta, S0 = 0.9, 100.0
    option = rl.VanillaOption(("call", S0 * math.exp(0.05)), maturity=1.0)
    quiet = rl.SwitchingCEVProcess(ONE, S0=S0, r=0.05, q=0.0, sigma=[1e-3 * S0 ** (1 - beta)], beta=beta)
    for engine in (rl.FirstOrderFDEngine(quiet), rl.SwitchingFDReferee(quiet), rl.SwitchingFDEngine(quiet)):
        with pytest.raises(ValueError, match="Peclet"):
            npv(option, engine)
    dormant = rl.SwitchingCEVProcess(CHAIN, S0=S0, r=0.05, q=0.0, sigma=[0.0, 0.2 * S0 ** (1 - beta)], beta=beta)
    with pytest.raises(ValueError, match="zero volatility"):
        npv(option, rl.SwitchingFDEngine(dormant))
    plain = rl.SwitchingCEVProcess(ONE, S0=S0, r=0.05, q=0.0, sigma=[0.2 * S0 ** (1 - beta)], beta=beta)
    assert npv(option, rl.SwitchingFDEngine(plain)) > 0.0              # an ordinary volatility is unaffected


def test_short_dated_vol_of_vol_grid_uses_the_finite_horizon_variance():
    model = rl.SwitchingHestonVolOfVol(CHAIN, 100.0, 0.03, 0.01, 0.016, 0.37, 0.044, [0.47, 0.47], -0.81)
    option = rl.VanillaOption(("call", 105.0), maturity=124 / 365)
    analytic = 0.5492474883838804                                        # QuantLib AnalyticHestonEngine, in the issue
    assert npv(option, rl.FirstOrderFDEngine(model)) == pytest.approx(analytic, rel=0.06)   # 13% high before


def test_monte_carlo_segments_without_rebuilding_quadrature():
    for lo, hi in ((0.0, 0.3), (0.7, 2.5)):                              # a = 0: elementary
        d = hi - lo
        assert montecarlo._segment(0.0, lo, hi) == pytest.approx((d * (hi + lo) / 2, (hi ** 3 - lo ** 3) / 3), rel=1e-14)
    small = montecarlo._segment(1e-6, 0.7, 2.5)                          # near zero: the quadrature, as before
    assert small == pytest.approx(montecarlo._segment(0.0, 0.7, 2.5), rel=1e-5)
