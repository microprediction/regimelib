"""Zero-coupon bond options and caps under switching G2++: frozen limit against QuantLib's G2 closed form, the
expansion converging to the numerical solution with switching, put-call parity, and a Monte Carlo check."""
import math
import numpy as np
import pytest
import QuantLib as ql
import regimelib as rl

REF = ql.Date(1, 1, 2020)


def test_g2_bond_option_frozen_matches_quantlib():
    ql.Settings.instance().evaluationDate = REF; dc = ql.Actual365Fixed()
    r, a, sigma, b, eta, rho = 0.03, 0.5, 0.012, 0.08, 0.009, -0.6
    ts = ql.YieldTermStructureHandle(ql.FlatForward(REF, r, dc)); g2 = ql.G2(ts, a, sigma, b, eta, rho)
    model = rl.SwitchingG2(rl.RegimeChain.twoState(3.0, 5.0), ts, a, sigma, b, eta, rho)
    T, S = 2.0, 5.0
    for kind, K in (("call", 0.90), ("put", 0.92), ("call", 0.95)):
        expected = g2.discountBondOption(ql.Option.Call if kind == "call" else ql.Option.Put, K, T, S)
        o = rl.ZeroCouponBondOption(kind, K, T, S)
        o.setPricingEngine(rl.NumericalSwitchingEngine(model)); assert o.NPV() == pytest.approx(expected, rel=1e-7)
        o.setPricingEngine(rl.FastSwitchingEngine(model, order=2)); assert o.NPV() == pytest.approx(expected, rel=1e-7)
    # a cap under G2: QuantLib has no analytic cap engine for two factors, so the caplets are its bond puts
    times, Kc = [1.0, 2.0, 3.0, 4.0, 5.0], 0.03
    expected = sum((1 + Kc) * g2.discountBondOption(ql.Option.Put, 1 / (1 + Kc), t0, t1) for t0, t1 in zip(times[:-1], times[1:]))
    ours = rl.CapFloor("cap", times, Kc); ours.setPricingEngine(rl.FastSwitchingEngine(model, order=2))
    assert ours.NPV() == pytest.approx(expected, rel=1e-7)


def test_g2_switching_orders_converge_and_parity():
    chain = rl.RegimeChain.twoState(12.0, 8.0)
    model = rl.SwitchingG2(chain, 0.03, 0.5, [0.02, 0.008], 0.08, [0.012, 0.005], [-0.7, -0.3])
    call, put = rl.ZeroCouponBondOption("call", 0.9, 2.0, 5.0), rl.ZeroCouponBondOption("put", 0.9, 2.0, 5.0)
    call.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0)); ref = call.NPV()
    errs = []
    for order in (0, 1, 2, 3):
        call.setPricingEngine(rl.FastSwitchingEngine(model, order=order, regime=0)); errs.append(abs(call.NPV() - ref))
    assert errs[0] > errs[1] > errs[2] > errs[3] and errs[3] < 1e-6 * ref
    put.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0))
    bond = lambda t: (lambda z: (z.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0)), z.NPV())[1])(rl.ZeroCouponBond(t))
    assert ref - put.NPV() == pytest.approx(bond(5.0) - 0.9 * bond(2.0), abs=1e-12)


def test_g2_switching_matches_monte_carlo():
    chain = rl.RegimeChain.twoState(4.0, 6.0)
    a, b, sig, eta, rho = 0.5, 0.08, np.array([0.025, 0.008]), np.array([0.012, 0.004]), np.array([-0.7, -0.2])
    model = rl.SwitchingG2(chain, 0.03, a, list(sig), b, list(eta), list(rho))
    T, S, K = 2.0, 5.0, 0.9
    call = rl.ZeroCouponBondOption("call", K, T, S); call.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0)); ref = call.NPV()
    rng = np.random.default_rng(7); N, M = 200000, 800; dt = T / M; Q = chain.generator
    x = np.zeros(N); y = np.zeros(N); I = np.zeros(N); reg = np.zeros(N, int)
    for _ in range(M):
        s, e, r = sig[reg], eta[reg], rho[reg]
        z1 = rng.standard_normal(N); z2 = r * z1 + np.sqrt(1 - r * r) * rng.standard_normal(N)
        I += 0.5 * (x + y) * dt
        x += -a * x * dt + s * math.sqrt(dt) * z1; y += -b * y * dt + e * math.sqrt(dt) * z2
        I += 0.5 * (x + y) * dt
        reg = np.where(rng.random(N) < -np.diag(Q)[reg] * dt, 1 - reg, reg)
    # the bond at T per regime from the model, the discounting factors from the curve fit
    Ba, Bb = (1 - math.exp(-a * (S - T))) / a, (1 - math.exp(-b * (S - T))) / b
    from regimelib.engines import _numericalAVector
    from regimelib.g2options import _g2_forcing
    g, gf = _g2_forcing(a, b, sig, eta, rho, 0.0, 0.0)
    A = _numericalAVector(Q, g, gf, S - T).real
    c = model.deterministicDiscount(T, S); e0T = model.deterministicDiscount(0.0, T)
    bondT = c * A[reg] * np.exp(-Ba * x - Bb * y)
    pay = e0T * np.exp(-I) * np.maximum(bondT - K, 0.0)
    assert abs(ref - pay.mean()) < 3 * pay.std() / math.sqrt(N) + 2e-5
