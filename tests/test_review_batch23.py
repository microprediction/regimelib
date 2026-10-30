"""Issue #287: calibration refuses complex-step Jacobians, which the real-valued pricers cannot carry."""
import pytest
import regimelib as rl


def test_complex_step_jacobian_is_refused_and_finite_differences_still_fit():
    def model():
        return rl.SwitchingBlackScholesProcess(rl.RegimeChain([[0.0]]), S0=100.0, r=0.02, q=0.0, sigma=[0.40])
    helper = rl.VolatilityHelper(maturity=1.0, strike=100.0, volatility=0.20)
    with pytest.raises(ValueError, match="cs"):
        rl.calibrate(model(), [helper], ["sigma"], jac="cs")             # stopped at 40% claiming success, before
    for jac in ("2-point", "3-point"):
        m = model()
        rl.calibrate(m, [helper], ["sigma"], jac=jac)
        assert m.sigma[0] == pytest.approx(0.20, abs=1e-8)
