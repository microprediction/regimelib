"""Analytic greeks against central finite differences of the same engine."""
import pytest
import regimelib as rl

CHAIN = rl.RegimeChain.twoState(12.0, 8.0)


def _fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h), (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)


def test_heston_delta_gamma_vega0():
    args = dict(r=0.02, q=0.01, v0=0.04, kappa=1.5, theta=[0.09, 0.02], sigma=0.4, rho=-0.6)
    opt = rl.VanillaOption(("put", 100.0), maturity=1.0)
    def price(S0, v0=0.04):
        a = dict(args, v0=v0); opt.setPricingEngine(rl.FastSwitchingEngine(rl.SwitchingHestonModel(CHAIN, S0, **a), order=2)); return opt.NPV()
    eng = rl.FastSwitchingEngine(rl.SwitchingHestonModel(CHAIN, 100.0, **args), order=2)
    g = eng.greeks(opt)
    d, _ = _fd(price, 100.0, 0.05); _, gm = _fd(price, 100.0, 1.0)
    assert g["delta"] == pytest.approx(d, rel=5e-6) and g["gamma"] == pytest.approx(gm, rel=1e-4)
    dv, _ = _fd(lambda v: price(100.0, v), 0.04, 0.002)
    assert g["vega0"] == pytest.approx(dv, rel=1e-4)


def test_black_scholes_delta_gamma_switching():
    opt = rl.VanillaOption(("call", 105.0), maturity=0.75)
    def price(S0):
        opt.setPricingEngine(rl.NumericalSwitchingEngine(rl.SwitchingBlackScholesProcess(CHAIN, S0, 0.03, 0.0, [0.3, 0.15]))); return opt.NPV()
    g = rl.NumericalSwitchingEngine(rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.03, 0.0, [0.3, 0.15])).greeks(opt)
    d, _ = _fd(price, 100.0, 0.05); _, gm = _fd(price, 100.0, 1.0)
    assert g["delta"] == pytest.approx(d, rel=5e-6) and g["gamma"] == pytest.approx(gm, rel=1e-4)


def test_bond_delta_gamma():
    bond = rl.ZeroCouponBond(4.0)
    def price(r0, cls=rl.SwitchingVasicek):
        bond.setPricingEngine(rl.FastSwitchingEngine(cls(CHAIN, r0, 0.5, [0.07, 0.02], [0.012, 0.006]) if cls is rl.SwitchingVasicek
                                                     else cls(CHAIN, r0, [0.07, 0.02], 1.2, 0.15), order=3)); return bond.NPV()
    for cls in (rl.SwitchingVasicek, rl.SwitchingCoxIngersollRoss):
        model = cls(CHAIN, 0.03, 0.5, [0.07, 0.02], [0.012, 0.006]) if cls is rl.SwitchingVasicek else cls(CHAIN, 0.03, [0.07, 0.02], 1.2, 0.15)
        g = rl.FastSwitchingEngine(model, order=3).greeks(bond)
        d, gm = _fd(lambda r: price(r, cls), 0.03, 1e-3)
        assert g["delta"] == pytest.approx(d, rel=5e-6) and g["gamma"] == pytest.approx(gm, rel=1e-4)
