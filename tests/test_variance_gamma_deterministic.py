"""Exact zero-amplitude VG boundary (#343), not a small-noise approximation."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import regimelib as rl
from regimelib.engines import _noDiffusion


@pytest.mark.parametrize("Engine", [rl.NumericalSwitchingEngine, rl.FastSwitchingEngine])
@pytest.mark.parametrize("regime", [0, 1, [0.3, 0.7]])
@pytest.mark.parametrize("nu", [[0.2, 1.5], [0.7, 0.7]])
def test_zero_amplitudes_price_exact_payoffs_and_greeks(Engine, regime, nu):
    m = rl.SwitchingVarianceGammaProcess(rl.RegimeChain.twoState(3., 5.), 100., .02, .01, 0., nu, 0.)
    assert _noDiffusion(m) == "deterministic"
    engine = Engine(m, regime=regime)
    T = 1.3
    dq, dr = math.exp(-m.q*T), math.exp(-m.r*T)
    F = m.forward(T)
    for K in (90., F, 110.):
        for kind, side in (("call", 1.), ("put", -1.)):
            inside = 1. if side*(F-K) > 0 else (.5 if F == K else 0.)
            option = rl.VanillaOption((kind, K), maturity=T)
            option.setPricingEngine(engine)
            assert option.NPV() == pytest.approx(max(side*(m.S0*dq-K*dr), 0.), abs=1e-12)
            assert option.delta() == pytest.approx(side*inside*dq)
            assert option.gamma() == 0.
            assert option.theta() == pytest.approx(side*inside*(m.q*m.S0*dq-m.r*K*dr))
            assert option.rho() == pytest.approx(side*inside*T*K*dr)
            for payoff, amount in ((("cash", kind, K, 3.), 3.), (("asset", kind, K), F)):
                digital = rl.VanillaOption(payoff, maturity=T)
                digital.setPricingEngine(engine)
                assert digital.NPV() == pytest.approx(dr*amount*(side*(F-K) > 0))
        G = m.S0*math.exp((m.r-m.q)*T/2)
        asian = rl.ContinuousGeometricAsianOption(("call", K), maturity=T)
        asian.setPricingEngine(engine)
        assert asian.NPV() == pytest.approx(dr*max(G-K, 0.), abs=1e-12)
    for u in (0., 1., 100., 1-.5j):
        _, funcs, prefactor = m.returnForcing(u, T)
        assert all(f(.4) == 0j for f in funcs)
        assert prefactor() == 1.


@pytest.mark.parametrize("sigma,theta", [([1e-100, 0.], 0.), (0., [0., 1e-100]), (0., [-.1, 0.]), (.2, 0.)])
def test_nonzero_amplitudes_are_not_reclassified(sigma, theta):
    m = rl.SwitchingVarianceGammaProcess(rl.RegimeChain.twoState(3., 5.), 100., .02, .01, sigma, [.2, 1.5], theta)
    assert _noDiffusion(m) is None


def test_unrecognized_nu_model_is_still_conservative():
    assert _noDiffusion(SimpleNamespace(nu=[.2], sigma=[0.], theta=[0.])) is None


@pytest.mark.parametrize("Engine", [rl.NumericalSwitchingEngine, rl.FastSwitchingEngine])
def test_one_regime_zero_vol_helper_and_clock_calibration(Engine):
    m = rl.SwitchingVarianceGammaProcess(rl.RegimeChain([[0.]]), 100., .02, .01, 0., .2, 0.)
    helper = rl.VolatilityHelper(1., 90., 0.)
    helper.setPricingEngine(Engine(m))
    assert helper.calibrationError() == pytest.approx(0., abs=1e-12)
    # Clock parameters remain unidentifiable at this boundary, but a valid
    # initial point must still produce finite calibration residuals.
    rl.calibrate(m, [helper], ["nu"], engine=Engine, max_nfev=1)
    assert np.isfinite(m.nu).all()
