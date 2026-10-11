"""Issues #271 to #274: transition-matrix roundoff, a finite-horizon hybrid grid, equal-speed G2 as Hull-White,
and American options expiring now under an inferred regime."""
import math
import numpy as np
import pytest
from scipy.special import ndtr
import regimelib as rl

ONE = rl.RegimeChain([[0.0]])


def npv(instrument, engine):
    instrument.setPricingEngine(engine)
    return instrument.NPV()


def test_transition_matrix_round_trips_through_structural_zeros():
    Q = np.array([[-0.4, 0.4, 0.0], [0.3, -0.3, 0.0], [0.9, 0.1, -1.0]])
    P = rl.RegimeChain(Q).transitionMatrix(2.0)                          # -1.5e-16 at a structural zero on some stacks
    P[0, 2] = -1.5e-16; P[0, 0] += 1.5e-16
    assert np.abs(rl.RegimeChain.fromTransitionMatrix(P, 2.0).generator - Q).max() < 1e-12
    with pytest.raises(ValueError, match="nonnegative"):
        rl.RegimeChain.fromTransitionMatrix([[1.0 + 1e-6, -1e-6], [0.2, 0.8]], 1.0)


def test_hybrid_grid_follows_the_finite_horizon_drift():
    S0, T, r0, b, sr, ss = 100.0, 1.0, 0.03, 0.50, 0.001, 0.02
    a = 0.02 / (b - r0)                                                  # slow reversion to a distant level
    B, B2 = -math.expm1(-a * T) / a, -math.expm1(-2 * a * T) / (2 * a)
    vI = sr ** 2 / a ** 2 * (T - 2 * B + B2)
    P = math.exp(-(b * T + (r0 - b) * B) + vI / 2); F = S0 / P
    exact = P * F * (2 * ndtr(math.sqrt(ss ** 2 * T + vI) / 2) - 1)
    model = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(ONE, S0=S0, r=0.0, q=0.0, sigma=ss),
                                    rl.SwitchingVasicek(ONE, r0=r0, a=a, b=b, sigma=sr))
    assert npv(rl.VanillaOption(("call", F), maturity=T), rl.SwitchingFDEngine(model)) == pytest.approx(exact, rel=5e-3)


def test_equal_speed_g2_is_hull_white_in_the_sum_of_its_factors():
    r, a, T, S = 0.03, 0.5, 1.0, 3.0
    K = math.exp(-r * (S - T))
    for rho in (-1.0, -0.95, 0.3):
        model = rl.SwitchingG2(ONE, r, a=a, sigma=0.02, b=a, eta=0.02, rho=rho)
        option = rl.ZeroCouponBondOption("call", K, T, S)
        exact = npv(option, rl.NumericalSwitchingEngine(model))
        assert npv(option, rl.SwitchingFDEngine(model)) == pytest.approx(exact, rel=2e-3, abs=1e-12)  # 12% high at -0.95
    chain = rl.RegimeChain.twoState(3.0, 5.0)
    model = rl.SwitchingG2(chain, r, a=a, sigma=[0.02, 0.01], b=a, eta=[0.01, 0.015], rho=[-0.7, 0.2])
    option = rl.ZeroCouponBondOption("call", K, T, S)
    exact = npv(option, rl.NumericalSwitchingEngine(model, information="observed"))
    assert npv(option, rl.SwitchingFDEngine(model, information="observed")) == pytest.approx(exact, rel=2e-3)


@pytest.mark.parametrize("information", ["inferred", "observed"])
def test_american_hybrid_option_expiring_now_is_its_payoff(information):
    chain = rl.RegimeChain.twoState(2.0, 3.0)
    model = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(chain, S0=100.0, r=0.0, q=0.0, sigma=0.20),
                                    rl.SwitchingVasicek(chain, r0=0.03, a=0.5, b=[0.07, 0.02], sigma=0.01))
    for payoff, expected in ((("call", 90.0), 10.0), (("put", 110.0), 10.0), (("put", 90.0), 0.0),
                             (("cash", "call", 90.0, 7.0), 7.0), (("asset", "put", 110.0), 100.0)):
        for regime in (0, [0.4, 0.6]):
            engine = rl.SwitchingFDEngine(model, regime=regime, n=(21, 11), steps=10, information=information)
            assert npv(rl.VanillaOption(payoff, exercise="american", maturity=0.0), engine) == pytest.approx(expected, abs=1e-12)
