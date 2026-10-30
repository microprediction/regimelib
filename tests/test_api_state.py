"""Results follow the state they were computed from, calibration leaves the model as it found its shapes, and a few
smaller contracts of the public interface."""
import math
import numpy as np
import pytest
import regimelib as rl

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)


def test_greeks_are_recomputed_after_the_model_engine_or_contract_changes():
    model = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.02, 0.0, [0.3, 0.15])
    engine = rl.NumericalSwitchingEngine(model)
    calls = []
    original = engine.calculate
    engine.calculate = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    option = rl.VanillaOption(("call", 100.0), maturity=1.0); option.setPricingEngine(engine)
    value, delta = option.NPV(), option.delta()
    assert len(calls) == 1 and option.gamma() > 0 and len(calls) == 1          # greeks reuse the calculation behind NPV()
    model.sigma[0] = 0.6                                                       # changed in place
    assert option.delta() != delta and len(calls) == 2
    assert option.NPV() > value
    engine.regime = 1
    quiet = option.delta(); assert len(calls) == 4 and quiet != delta          # NPV() above, then the new regime
    option.strike = 120.0
    assert option.delta() < quiet
    cir = rl.SwitchingCoxIngersollRoss(CHAIN, 0.02, 0.03, 0.5, 0.08)
    cds = rl.CreditDefaultSwap("buyer", 0.01, [0.5, 1.0, 1.5, 2.0], 0.4, discount=0.03)
    cds.setPricingEngine(rl.NumericalSwitchingEngine(cir)); spread = cds.fairSpread()
    cir.theta = [0.06, 0.06]
    assert cds.fairSpread() > spread


def test_high_precision_solver_restores_the_callers_precision():
    import mpmath as mp
    from regimelib._engine.fastswitch import numerical_a, ExpSum
    before = mp.mp.dps
    mp.mp.dps = 17
    try:
        numerical_a(1.0, [[-1.0, 1.0], [2.0, -2.0]], [ExpSum({0: -0.1}), ExpSum({0: -0.2})], dps=40)
        assert mp.mp.dps == 17
    finally:
        mp.mp.dps = before


def quotes(truth, strikes=(90.0, 100.0, 110.0)):
    engine = rl.NumericalSwitchingEngine(truth)
    out = []
    for K in strikes:
        o = rl.VanillaOption(("call", K), maturity=1.0); o.setPricingEngine(engine)
        out.append(rl.VolatilityHelper(1.0, K, o.impliedVolatility()))
    return out


def test_calibration_keeps_shapes_and_owns_its_values():
    one = rl.RegimeChain([[0.0]])
    truth = rl.SwitchingBlackScholesProcess(one, 100.0, 0.02, 0.0, [0.25])
    model = rl.SwitchingBlackScholesProcess(one, 100.0, 0.02, 0.0, [0.4])
    result = rl.calibrate(model, quotes(truth), ["sigma"])
    assert isinstance(model.sigma, list) and model.sigma == pytest.approx([0.25], rel=1e-6)     # still one entry per regime
    truth2 = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.02, 0.0, [0.4, 0.15])
    model2 = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.02, 0.0, [0.3, 0.2])
    result = rl.calibrate(model2, quotes(truth2, (80.0, 90.0, 100.0, 110.0, 120.0)), ["sigma"])
    fitted = list(model2.sigma)
    result.x[:] = 99.0                                                          # the caller's array, not the model's
    assert list(model2.sigma) == fitted and isinstance(model2.sigma, list)
    scalar = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.02, 0.0, [0.3, 0.2]); scalar.r = 0.02
    rl.calibrate(scalar, quotes(truth2), ["sigma"], max_nfev=2)
    assert isinstance(scalar.r, float)


def test_calibration_reads_its_inputs_once_and_refuses_repeats():
    truth = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.02, 0.0, [0.4, 0.15])
    start = lambda: rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.02, 0.0, [0.3, 0.2])
    listed, generated = start(), start()
    rl.calibrate(listed, quotes(truth), ["sigma"])
    rl.calibrate(generated, (h for h in quotes(truth)), (p for p in ["sigma"]))
    assert generated.sigma == pytest.approx(listed.sigma, rel=1e-9)
    untouched = start()
    with pytest.raises(ValueError, match="repeated"):
        rl.calibrate(untouched, quotes(truth), ["sigma", "sigma"])
    assert untouched.sigma == [0.3, 0.2]
    with pytest.raises(ValueError, match="helper"):
        rl.calibrate(start(), [], ["sigma"])
    with pytest.raises(ValueError, match="parameter"):
        rl.calibrate(start(), quotes(truth), [])


def test_coupon_bond_reports_only_the_greeks_its_parts_provide():
    flows = [(1.0, 0.04), (2.0, 1.04)]
    vasicek = rl.SwitchingVasicek(CHAIN, 0.03, 0.5, [0.06, 0.02], [0.015, 0.008])
    engine = rl.NumericalSwitchingEngine(vasicek)
    bond = rl.CouponBond(cashflows=flows); bond.setPricingEngine(engine)
    parts = []
    for t, c in flows:
        z = rl.ZeroCouponBond(t); z.setPricingEngine(engine); parts.append((c * z.NPV(), c * z.delta(), c * z.gamma()))
    assert (bond.NPV(), bond.delta(), bond.gamma()) == pytest.approx(tuple(sum(p[i] for p in parts) for i in range(3)), rel=1e-12)
    hull_white = rl.SwitchingHullWhite(CHAIN, 0.03, 0.5, [0.02, 0.006])
    bond.setPricingEngine(rl.NumericalSwitchingEngine(hull_white))
    assert bond.NPV() > 0
    with pytest.raises(RuntimeError, match="not provided"):
        bond.delta()


def test_coupon_bond_options_at_deep_strikes():
    model = rl.SwitchingVasicek(CHAIN, 0.03, 0.5, [0.06, 0.02], [0.015, 0.008])
    engine = rl.NumericalSwitchingEngine(model)
    flows = [(2.0, 0.04), (3.0, 1.04)]
    bond = lambda t: (lambda z: (z.setPricingEngine(engine), z.NPV())[1])(rl.ZeroCouponBond(t))
    underlying = sum(c * bond(t) for t, c in flows)
    for K in (1e-3, 0.05):                                                      # far in the money: a forward
        call = rl.CouponBondOption("call", K, 1.0, flows); call.setPricingEngine(engine)
        assert call.NPV() == pytest.approx(underlying - K * bond(1.0), rel=1e-8)
    for K in (5.0, 50.0):                                                       # far out of the money
        call = rl.CouponBondOption("call", K, 1.0, flows); call.setPricingEngine(engine)
        assert abs(call.NPV()) < 1e-9


def test_symbolic_black_scholes_forcing_is_the_martingale_one():
    import sympy as sp
    from regimelib.symbolic import TwoStateConstantForcing
    f = TwoStateConstantForcing()
    for regime in (0, 1):                                                       # E[S_T / F] = 1: the function is one at u = -i
        value = f.blackScholes(regime).subs({f.u: -sp.I, f.q12: 3.0, f.q21: 5.0, f.s1: 0.3, f.s2: 0.15, f.T: 1.0})
        assert abs(complex(sp.N(value)) - 1.0) < 1e-12


def test_explicit_bond_option_module_imports_its_referee():
    import importlib
    module = importlib.import_module("regimelib._engine.bond_option_explicit")
    from regimelib._engine.options import zcb_call
    assert callable(module.call) and callable(zcb_call)
