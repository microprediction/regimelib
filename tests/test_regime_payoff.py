"""A bond paid only if the regime at maturity is j: the parts sum to the plain bond, and with the regimes frozen the
part equals the bond times the transition probability."""
import numpy as np
from scipy.linalg import expm
import pytest
import regimelib as rl


def test_regime_conditional_bond():
    chain = rl.RegimeChain.twoState(8.0, 3.0); T = 2.0
    model = rl.SwitchingVasicek(chain, 0.03, 0.5, [0.07, 0.02], [0.012, 0.006])
    parts = []
    for j in (0, 1):
        b = rl.ZeroCouponBond(T, regimeAtMaturity=j); b.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0)); parts.append(b.NPV())
    whole = rl.ZeroCouponBond(T); whole.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0))
    assert sum(parts) == pytest.approx(whole.NPV(), rel=1e-10)
    # expansion with a terminal indicator (initial layer at order zero) converges to the numerical value
    b = rl.ZeroCouponBond(T, regimeAtMaturity=1)
    errs = []
    for o in (1, 2, 4):
        b.setPricingEngine(rl.FastSwitchingEngine(model, order=o, regime=0)); errs.append(abs(b.NPV() - parts[1]))
    assert errs[0] > errs[1] > errs[2]
    # frozen regimes: part = bond * P(regime j at T | start 0)
    frozen = rl.SwitchingVasicek(chain, 0.03, 0.5, 0.05, 0.01)
    b.setPricingEngine(rl.FastSwitchingEngine(frozen, order=3, regime=0))
    whole.setPricingEngine(rl.FastSwitchingEngine(frozen, order=3, regime=0))
    p = expm(chain.generator * T)[0, 1]
    assert b.NPV() == pytest.approx(whole.NPV() * p, rel=1e-9)
