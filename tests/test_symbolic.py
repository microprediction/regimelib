"""The symbolic second-order bond price equals the engine at order 2, and its symbolic greeks match finite
differences of the numerical engine to the order of the expansion."""
import math
import pytest
import sympy as sp
import regimelib as rl
from regimelib.symbolic import VasicekTwoStateBond

V = dict(r0=0.03, kappa=0.5, theta1=0.07, theta2=0.01, sigma1=0.012, sigma2=0.008, lam=25.0, T=5.0)


def _model(lam=V["lam"], **over):
    v = dict(V, **over)
    return rl.SwitchingVasicek(rl.RegimeChain.twoState(lam, lam), v["r0"], v["kappa"], [v["theta1"], v["theta2"]], [v["sigma1"], v["sigma2"]])


def _npv(engine_cls, **over):
    v = dict(V, **over); bond = rl.ZeroCouponBond(v["T"])
    bond.setPricingEngine(engine_cls(_model(**over), regime=0) if engine_cls is rl.NumericalSwitchingEngine else engine_cls(_model(**over), order=2, regime=0))
    return bond.NPV()


def test_formula_equals_engine_at_order_two():
    f = VasicekTwoStateBond(regime=0)
    assert f.evaluate(f.price, **V) == pytest.approx(_npv(rl.FastSwitchingEngine), rel=1e-12)
    assert f.evaluate(f.price, **V) == pytest.approx(_npv(rl.NumericalSwitchingEngine), rel=1e-6)


def test_symbolic_greeks_match_finite_differences():
    f = VasicekTwoStateBond(regime=0)
    fd = lambda name, h: (_npv(rl.NumericalSwitchingEngine, **{name: V[name] + h}) - _npv(rl.NumericalSwitchingEngine, **{name: V[name] - h})) / (2 * h)
    assert f.evaluate(f.delta(), **V) == pytest.approx(fd("r0", 1e-4), rel=1e-6)
    assert f.evaluate(f.greek("theta1"), **V) == pytest.approx(fd("theta1", 1e-4), rel=1e-5)
    assert f.evaluate(f.greek("sigma1"), **V) == pytest.approx(fd("sigma1", 1e-4), rel=1e-4)
    assert f.evaluate(f.greek("lam"), **V) == pytest.approx(fd("lam", 0.5), rel=2e-2)     # third-order remainder
    assert f.evaluate(f.theta(), **V) == pytest.approx(-fd("T", 1e-3), rel=1e-5)
    g2 = (_npv(rl.NumericalSwitchingEngine, r0=V["r0"] + 1e-3) - 2 * _npv(rl.NumericalSwitchingEngine) + _npv(rl.NumericalSwitchingEngine, r0=V["r0"] - 1e-3)) / 1e-6
    assert f.evaluate(f.gamma(), **V) == pytest.approx(g2, rel=1e-5)


def test_delta_formula_is_minus_B_times_price():
    f = VasicekTwoStateBond(regime=0)
    B = (1 - sp.exp(-f.symbols["kappa"] * f.symbols["T"])) / f.symbols["kappa"]
    assert sp.simplify(f.delta() / f.price + B) == 0


def test_first_order_formula_for_three_regimes():
    from regimelib.symbolic import VasicekBondFirstOrder
    chain = rl.RegimeChain([[-5, 3, 2], [4, -9, 5], [1, 6, -7]]); kappa_, thetas, sigmas, r0_, T_ = 0.5, [0.08, 0.05, 0.01], [0.015, 0.01, 0.006], 0.03, 4.0
    f = VasicekBondFirstOrder()
    for regime in (0, 1, 2):
        vals = dict(r0=r0_, kappa=kappa_, T=T_, **f.coefficients(chain, kappa_, thetas, sigmas, regime))
        bond = rl.ZeroCouponBond(T_); bond.setPricingEngine(rl.FastSwitchingEngine(rl.SwitchingVasicek(chain, r0_, kappa_, thetas, sigmas), order=1, regime=regime))
        assert f.evaluate(f.price, **vals) == pytest.approx(bond.NPV(), rel=1e-12)
        bond.setPricingEngine(rl.NumericalSwitchingEngine(rl.SwitchingVasicek(chain, r0_, kappa_, thetas, sigmas), regime=regime))
        assert f.evaluate(f.price, **vals) == pytest.approx(bond.NPV(), rel=1e-3)      # first order at eps = 3/21
    assert sp.simplify(f.greek("r0") / f.price + (1 - sp.exp(-f.symbols["kappa"] * f.symbols["T"])) / f.symbols["kappa"]) == 0


def test_two_state_constant_forcing_is_exact():
    """The closed-form two-state solution equals the numerical solution of the reduced system to round-off, and its
    Black-Scholes specialisation is the characteristic function the numerical engine integrates."""
    import numpy as np
    from regimelib.symbolic import TwoStateConstantForcing
    from regimelib.engines import _numericalAVector
    from regimelib._engine.models import bs_switching
    f = TwoStateConstantForcing()
    chain = rl.RegimeChain.twoState(3.0, 5.0); Q = chain.generator
    for u in (0.7, 2.5 - 0.5j):
        g, gfuncs = bs_switching(u, 0.0, [0.4, 0.15])
        num = _numericalAVector(Q, g, gfuncs, 1.5)
        for regime in (0, 1):
            sym = f.evaluate(f.a(regime), q12=3.0, q21=5.0, g1=complex(g[0].value(0)), g2=complex(g[1].value(0)), T=1.5)
            assert abs(sym - num[regime]) < 1e-12 * abs(num[regime])
            bs = f.evaluate(f.blackScholes(regime), q12=3.0, q21=5.0, u=u, sigma1=0.4, sigma2=0.15, T=1.5)
            assert abs(bs - num[regime]) < 1e-10 * abs(num[regime])


def test_cir_first_order_formula():
    from regimelib.symbolic import CIRBondFirstOrder
    chain = rl.RegimeChain([[-5, 3, 2], [4, -9, 5], [1, 6, -7]]); k_, thetas, sigma_, r0_, T_ = 0.6, [0.06, 0.03, 0.01], 0.08, 0.03, 4.0
    f = CIRBondFirstOrder()
    # I2 = int B^2 from the Riccati identity against quadrature
    from scipy.integrate import quad
    h = math.sqrt(k_ ** 2 + 2 * sigma_ ** 2); B = lambda t: 2 * (math.exp(h * t) - 1) / ((h + k_) * (math.exp(h * t) - 1) + 2 * h)
    vals0 = dict(r0=r0_, k=k_, sigma=sigma_, T=T_, thetabar=0.04, Kcc=1.0, mc=0.0)
    assert f.evaluate(f.terms["green_kubo"], **vals0) == pytest.approx(quad(lambda t: B(t) ** 2, 0, T_)[0], rel=1e-10)
    for regime in (0, 1, 2):
        vals = dict(r0=r0_, k=k_, sigma=sigma_, T=T_, **f.coefficients(chain, k_, thetas, regime))
        model = rl.SwitchingCoxIngersollRoss(chain, r0_, thetas, k_, sigma_)
        bond = rl.ZeroCouponBond(T_); bond.setPricingEngine(rl.FastSwitchingEngine(model, order=1, regime=regime))
        assert f.evaluate(f.price, **vals) == pytest.approx(bond.NPV(), rel=1e-9)
        bond.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=regime))
        assert f.evaluate(f.price, **vals) == pytest.approx(bond.NPV(), rel=1e-3)
        assert f.evaluate(f.greek("r0"), **vals) == pytest.approx(-f.evaluate(f.B * f.price, **vals), rel=1e-12)   # delta = -B P


@pytest.mark.slow
def test_vasicek_jumps_first_order_formula():
    from regimelib.symbolic import VasicekJumpsBondFirstOrder
    chain = rl.RegimeChain([[-5, 3, 2], [4, -9, 5], [1, 6, -7]])
    kappa_, thetas, sigmas, lams, m_, r0_, T_ = 0.5, [0.06, 0.03, 0.01], [0.015, 0.01, 0.006], [2.0, 0.5, 0.1], 0.01, 0.03, 3.0
    f = VasicekJumpsBondFirstOrder()
    for regime in (0, 1, 2):
        vals = dict(r0=r0_, kappa=kappa_, m=m_, T=T_, **f.coefficients(chain, kappa_, thetas, sigmas, lams, regime))
        model = rl.SwitchingVasicekJumps(chain, r0_, kappa_, thetas, sigmas, lams, m_)
        bond = rl.ZeroCouponBond(T_); bond.setPricingEngine(rl.FastSwitchingEngine(model, order=1, regime=regime))
        assert f.evaluate(f.price, **vals) == pytest.approx(bond.NPV(), rel=1e-9)
        bond.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=regime))
        assert f.evaluate(f.price, **vals) == pytest.approx(bond.NPV(), rel=1e-3)
        assert f.evaluate(f.greek("r0"), **vals) == pytest.approx(-f.evaluate(f.B * f.price, **vals), rel=1e-12)
