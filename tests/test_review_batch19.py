"""Issues #261 and #264: strong correlation on two-dimensional grids (a sign-aware cross stencil; decorrelated
coordinates for the equity with rates)."""
import math
import numpy as np
import pytest
from scipy.special import ndtr
import regimelib as rl
from regimelib.firstorder import Grid2D


def npv(instrument, engine):
    instrument.setPricingEngine(engine)
    return instrument.NPV()


def test_cross_stencils_are_second_order_and_put_weight_on_the_matching_diagonal():
    def error(nx, nv, sign):
        g = Grid2D(-1.0, 1.0, nx, -0.5, 0.5, nv)
        u = np.sin(1.3 * g.X + 0.4) * np.cos(2.1 * g.V)                 # u_xv = -1.3 * 2.1 cos(1.3x + 0.4) sin(2.1v)
        exact = -1.3 * 2.1 * np.cos(1.3 * g.X + 0.4) * np.sin(2.1 * g.V)
        inside = (np.abs(g.X) < 0.9) & (np.abs(g.V) < 0.45)
        return np.abs((g.cross(sign) @ u - exact)[inside]).max()
    grid = Grid2D(-1.0, 1.0, 41, -0.5, 0.5, 21)
    for sign in (-1.0, 1.0):
        assert 3.5 < error(41, 21, sign) / error(81, 41, sign) < 4.5    # second order
        C = grid.cross(sign)
        row = C[20 * 21 + 10].toarray().ravel()                          # an interior node
        corners = [row[(20 + i) * 21 + 10 + j] for i, j in ((1, 1), (-1, -1), (1, -1), (-1, 1))]
        used, unused = (corners[:2], corners[2:]) if sign > 0 else (corners[2:], corners[:2])
        assert min(sign * w for w in used) > 0 and all(w == 0 for w in unused)   # times the coefficient: positive


def test_vol_of_vol_with_strong_negative_correlation():
    chain = rl.RegimeChain.twoState(3.0, 5.0)
    model = rl.SwitchingHestonVolOfVol(chain, 100.0, 0.03, 0.01, 0.016, 0.37, 0.044, [0.47, 0.47], -0.81)
    option = rl.VanillaOption(("call", 105.0), maturity=124 / 365)
    analytic = 0.5492474883838804                                        # QuantLib AnalyticHestonEngine, in #264
    assert npv(option, rl.FirstOrderFDEngine(model)) == pytest.approx(analytic, rel=0.02)   # 13% high in #264


@pytest.mark.parametrize("rho", [-0.95, -0.5, 0.0, 0.8])
def test_equity_with_rates_grid_across_correlations(rho):
    chain = rl.RegimeChain.twoState(2.0, 3.0)
    S0, T, r, q, ss, a, sr = 100.0, 5.0, 0.03, 0.01, 0.10, 0.10, 0.05
    B, B2 = -math.expm1(-a * T) / a, -math.expm1(-2 * a * T) / (2 * a)
    vI = sr ** 2 / a ** 2 * (T - 2 * B + B2)
    P = math.exp(-r * T + vI / 2); F = S0 * math.exp(-q * T) / P
    V = ss ** 2 * T + vI + 2 * rho * ss * sr / a * (T - B)
    exact = P * F * (2 * ndtr(math.sqrt(V) / 2) - 1)                    # Black at the forward, frozen regimes
    model = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(chain, S0=S0, r=r, q=q, sigma=ss),
                                    rl.SwitchingVasicek(chain, r0=r, a=a, b=r, sigma=sr), rho=rho)
    option = rl.VanillaOption(("call", F), maturity=T)
    assert npv(option, rl.SwitchingFDEngine(model, regime=0)) == pytest.approx(exact, rel=8e-3)   # +3.4% at -0.95 in #261


def test_switching_equity_with_rates_against_the_transform():
    chain = rl.RegimeChain.twoState(2.0, 3.0)
    model = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(chain, S0=100.0, r=0.03, q=0.01, sigma=[0.09, 0.11]),
                                    rl.SwitchingVasicek(chain, r0=0.03, a=0.1, b=[0.025, 0.035], sigma=[0.045, 0.055]), rho=-0.9)
    for payoff in (("put", 100.0), ("call", 120.0)):
        option = rl.VanillaOption(payoff, maturity=5.0)
        reference = npv(option, rl.NumericalSwitchingEngine(model, regime=0, nodes=256))
        assert npv(option, rl.SwitchingFDEngine(model, regime=0)) == pytest.approx(reference, rel=1e-2)
