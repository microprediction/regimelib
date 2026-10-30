"""Issue #337: independent one-jump oracle and unresolved hybrid-grid regressions.

These strict xfails expose current FD failures; this file does not repair them.
The oracle integrates a conditional Gaussian price, not a library CF or FD value.
"""
import math

import pytest
from scipy.integrate import quad
from scipy.special import ndtr

import regimelib as rl


def one_jump_case(sigma_r=(0.02, 0.005), rho=-0.7, regime=0):
    """One Exp(1) jump to an absorbing state, with a common Vasicek mean."""
    T, S0, r0, q, a = 2.0, 100.0, 0.03, 0.01, 0.5
    sigma_s = 0.20
    B = lambda t: -math.expm1(-a * t) / a
    I1 = lambda t: (t - B(t)) / a
    I2 = lambda t: (t - 2 * B(t) - math.expm1(-2 * a * t) / (2 * a)) / a**2

    def moments(j):
        h = T - j
        vi = sigma_r[0]**2 * (I2(T) - I2(h)) + sigma_r[1]**2 * I2(h)
        cov = rho * sigma_s * (sigma_r[0] * (I1(T) - I1(h)) + sigma_r[1] * I1(h))
        return math.exp(-r0 * T + vi / 2), vi + sigma_s**2 * T + 2 * cov

    def average(fn):
        if regime == 1:
            return fn(0.0)
        return math.exp(-T) * fn(T) + quad(lambda j: math.exp(-j) * fn(j), 0, T,
                                          epsabs=1e-11, epsrel=1e-11)[0]

    bond = average(lambda j: moments(j)[0])
    prepaid = S0 * math.exp(-q * T)
    K = prepaid / bond

    def call(j):
        P, variance = moments(j)
        forward = prepaid / P
        root = math.sqrt(variance)
        d1 = (math.log(forward / K) + variance / 2) / root
        return P * (forward * ndtr(d1) - K * ndtr(d1 - root))

    chain = rl.RegimeChain([[-1.0, 1.0], [0.0, 0.0]])
    model = rl.SwitchingEquityRates(
        rl.SwitchingBlackScholesProcess(chain, S0=S0, r=0.0, q=q, sigma=sigma_s),
        rl.SwitchingVasicek(chain, r0=r0, a=a, b=[r0, r0], sigma=sigma_r), rho=rho)
    return model, rl.VanillaOption(("call", K), maturity=T), average(call), prepaid


@pytest.mark.parametrize("sigma_r,rho", [((0.02, 0.005), -0.7), ((0.05, 0.001), -0.9),
                                         ((0.02, 0.005), 0.7), ((0.01, 0.01), -0.7), ((0.02, 0.005), 0.0)])
@pytest.mark.parametrize("regime", [0, 1])
def test_one_jump_oracle_matches_transform(sigma_r, rho, regime):
    model, option, exact, upper = one_jump_case(sigma_r, rho, regime)
    option.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=regime, nodes=256))
    assert 0 < exact < upper
    assert option.NPV() == pytest.approx(exact, abs=1e-7)


@pytest.mark.parametrize("engine", [rl.SwitchingFDReferee, rl.SwitchingFDEngine])
@pytest.mark.parametrize("regime", [0, 1])
def test_equal_rate_volatility_fd_control(engine, regime):
    model, option, exact, upper = one_jump_case((0.01, 0.01), -0.7, regime)
    option.setPricingEngine(engine(model, regime=regime))
    assert 0 <= option.NPV() <= upper
    assert option.NPV() == pytest.approx(exact, rel=0.01)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#337: stationary hybrid coordinate biases transient-regime FD pricing")
@pytest.mark.parametrize("engine", [rl.SwitchingFDReferee, rl.SwitchingFDEngine])
@pytest.mark.parametrize("rho", [-0.7, 0.7])
def test_transient_moderate_fd_accuracy(engine, rho):
    model, option, exact, _ = one_jump_case(rho=rho)
    option.setPricingEngine(engine(model, regime=0))
    assert option.NPV() == pytest.approx(exact, rel=0.01)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#337: transient hybrid grid violates the call's prepaid-spot upper bound")
@pytest.mark.parametrize("engine", [rl.SwitchingFDReferee, rl.SwitchingFDEngine])
@pytest.mark.parametrize("regime", [0, 1])
def test_transient_severe_fd_price_bound(engine, regime):
    model, option, _, upper = one_jump_case((0.05, 0.001), -0.9, regime)
    option.setPricingEngine(engine(model, regime=regime))
    value = option.NPV()
    assert math.isfinite(value) and 0 <= value <= upper


@pytest.mark.parametrize("engine", [rl.SwitchingFDReferee, rl.SwitchingFDEngine])
def test_closed_regime_only_grid_control(engine):
    # From regime 1, removing unreachable regime 0 leaves exactly the same law.
    _, option, exact, upper = one_jump_case((0.05, 0.001), -0.9, regime=1)
    chain = rl.RegimeChain([[0.0]])
    model = rl.SwitchingEquityRates(
        rl.SwitchingBlackScholesProcess(chain, S0=100.0, r=0.0, q=0.01, sigma=0.2),
        rl.SwitchingVasicek(chain, r0=0.03, a=0.5, b=0.03, sigma=0.001), rho=-0.9)
    option.setPricingEngine(engine(model))
    assert 0 <= option.NPV() <= upper
    assert option.NPV() == pytest.approx(exact, rel=0.001)
