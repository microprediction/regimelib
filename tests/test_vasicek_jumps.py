"""Vasicek with jumps at a switching intensity: frozen limit against the affine closed form by quadrature, the
expansion converging to the numerical solution, and Monte Carlo with switching (exact regime paths, Euler in the rate)."""
import math
import numpy as np
import pytest
from scipy.integrate import quad
import regimelib as rl


def test_frozen_matches_affine_closed_form_and_orders_converge():
    r0, a, b, sigma, lam, m, T = 0.03, 0.5, 0.04, 0.012, 0.8, 0.01, 3.0
    B = lambda t: (1 - math.exp(-a * t)) / a
    g = lambda t: -a * b * B(t) + 0.5 * sigma ** 2 * B(t) ** 2 + lam * (1 / (1 + m * B(t)) - 1)
    expected = math.exp(-B(T) * r0 + quad(g, 0, T, epsabs=1e-14, epsrel=1e-13)[0])
    model = rl.SwitchingVasicekJumps(rl.RegimeChain.twoState(3.0, 5.0), r0, a, b, sigma, lam, m)
    bond = rl.ZeroCouponBond(T)
    for eng in (rl.NumericalSwitchingEngine(model), rl.FastSwitchingEngine(model, order=2)):
        bond.setPricingEngine(eng); assert bond.NPV() == pytest.approx(expected, rel=1e-10)
    sw = rl.SwitchingVasicekJumps(rl.RegimeChain.twoState(12.0, 8.0), r0, a, [0.06, 0.02], [0.015, 0.008], [2.0, 0.2], m)
    bond.setPricingEngine(rl.NumericalSwitchingEngine(sw, regime=0)); ref = bond.NPV()
    errs = []
    for order in (0, 1, 2, 3):
        bond.setPricingEngine(rl.FastSwitchingEngine(sw, order=order, regime=0)); errs.append(abs(bond.NPV() - ref))
    assert errs[0] > errs[1] > errs[2] > errs[3] and errs[3] < 1e-7
    assert bond.delta() == pytest.approx(-((1 - math.exp(-a * T)) / a) * bond.NPV())


def test_switching_intensity_matches_monte_carlo():
    chain = rl.RegimeChain.twoState(4.0, 6.0); r0, a, b, sigma, m, T = 0.03, 0.5, 0.04, 0.01, 0.01, 2.0
    lam = np.array([3.0, 0.3])
    model = rl.SwitchingVasicekJumps(chain, r0, a, b, sigma, list(lam), m)
    bond = rl.ZeroCouponBond(T); bond.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0)); ref = bond.NPV()
    rng = np.random.default_rng(5); N, M = 200000, 800; dt = T / M; Q = chain.generator
    r = np.full(N, r0); I = np.zeros(N); reg = np.zeros(N, int)
    for _ in range(M):
        I += 0.5 * r * dt
        jumps = rng.poisson(lam[reg] * dt)
        r += a * (b - r) * dt + sigma * math.sqrt(dt) * rng.standard_normal(N) + rng.gamma(np.maximum(jumps, 1e-12), m) * (jumps > 0)
        I += 0.5 * r * dt
        reg = np.where(rng.random(N) < -np.diag(Q)[reg] * dt, 1 - reg, reg)
    pay = np.exp(-I)
    assert abs(ref - pay.mean()) < 3 * pay.std() / math.sqrt(N) + 2e-4
