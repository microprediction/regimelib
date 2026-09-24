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
