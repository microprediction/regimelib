"""Unit Merton jump multipliers leave the exact deterministic law unchanged (#335)."""
import math
import pytest
import regimelib as rl
from regimelib.engines import _noDiffusion

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)


def model(intensity, sigma=0.0, mean=0.0, vol=0.0):
    return rl.SwitchingMerton76Process(CHAIN, 100.0, 0.02, 0.01, sigma, intensity, mean, vol)


@pytest.mark.parametrize("intensity", [2.0, [1.0, 2.0], [0.0, 2.0]])
@pytest.mark.parametrize("Engine", [rl.NumericalSwitchingEngine, rl.FastSwitchingEngine])
@pytest.mark.parametrize("regime", [0, 1, [0.3, 0.7]])
def test_exact_payoffs_and_greeks(intensity, Engine, regime):
    m = model(intensity)
    assert _noDiffusion(m) == "deterministic"
    engine = Engine(m, regime=regime)
    T = 1.0
    F, dr, dq = 100.0 * math.exp(0.01*T), math.exp(-0.02*T), math.exp(-0.01*T)
    for K in [90.0, F, 110.0]:
        for kind, side in [("call", 1.0), ("put", -1.0)]:
            inside = float(side * (F-K) > 0) if F != K else 0.5
            opt = rl.VanillaOption((kind, K), maturity=T)
            opt.setPricingEngine(engine)
            assert opt.NPV() == pytest.approx(dr*max(side*(F-K), 0), abs=1e-12)
            assert opt.delta() == pytest.approx(side*inside*dq)
            assert opt.gamma() == 0.0
            assert opt.theta() == pytest.approx(side*inside*(0.01*100*dq-0.02*K*dr))
            assert opt.rho() == pytest.approx(side*inside*T*K*dr)
            for payoff, expected in [
                (("cash", kind, K, 7.0), dr*7.0*(side*(F-K)>0)),
                (("asset", kind, K), dr*F*(side*(F-K)>0)),
            ]:
                opt = rl.VanillaOption(payoff, maturity=T)
                opt.setPricingEngine(engine)
                assert opt.NPV() == pytest.approx(expected, abs=1e-12)
            asian = rl.ContinuousGeometricAsianOption((kind, K), maturity=T)
            asian.setPricingEngine(engine)
            G = 100.0*math.exp(0.01*T/2)
            assert asian.NPV() == pytest.approx(dr*max(side*(G-K), 0), abs=1e-12)


@pytest.mark.parametrize("u", [0, 1, 13, -1j, 3-0.5j])
def test_unit_jump_forcing_is_exactly_zero(u):
    _, forcings, prefactor = model([1.0, 2.0]).returnForcing(u, 1.0)
    assert prefactor() == 1
    for f in forcings:
        assert f(0.0) == 0j
        assert f(0.7) == 0j


@pytest.mark.parametrize("sigma,mean,vol", [
    ([0.0, 0.1], 0.0, 0.0), (0.0, 1e-15, 0.0),
    (0.0, -1e-15, 0.0), (0.0, 0.0, 1e-15),
    (0.0, -0.1, 0.2),
])
def test_nonzero_risk_is_not_rounded_to_deterministic(sigma, mean, vol):
    assert _noDiffusion(model([1.0, 2.0], sigma, mean, vol)) is None


def test_zero_intensity_still_ignores_arbitrary_jump_sizes():
    assert _noDiffusion(model(0.0, mean=-0.1, vol=0.2)) == "deterministic"


@pytest.mark.parametrize("Engine", [rl.NumericalSwitchingEngine, rl.FastSwitchingEngine])
def test_genuine_jump_atom_is_not_silently_removed(Engine):
    opt = rl.VanillaOption(("call", 100.0), maturity=1.0)
    opt.setPricingEngine(Engine(model([1.0, 2.0], mean=0.1)))
    with pytest.raises(ArithmeticError, match="atom|decay|characteristic"):
        opt.NPV()
