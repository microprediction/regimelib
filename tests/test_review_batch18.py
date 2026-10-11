"""Issue #276: nearly equal decay rates in the fast engine's initial layer."""
import mpmath
import numpy as np
import pytest
import regimelib as rl


@pytest.mark.parametrize("spacing", [0.0, 1e-9, 1e-5, 1e-4, 1e-3, 0.04, 0.2])
def test_terminal_regime_probability_through_nearly_equal_rates(spacing):
    rates = 4.0 * (1.0 + spacing * np.arange(4))                       # four transitions in a row into an absorbing state
    Q = np.zeros((5, 5))
    for i, r in enumerate(rates):
        Q[i, i], Q[i, i + 1] = -r, r
    with mpmath.workdps(50):
        exact = float(mpmath.expm(mpmath.matrix(Q.tolist()) * 0.5)[0, 4])
    model = rl.SwitchingVasicek(rl.RegimeChain(Q), r0=0.0, a=0.5, b=0.0, sigma=0.0)   # the bond is P(Y_T = 4)
    for order in (0, 4):
        bond = rl.ZeroCouponBond(0.5, regimeAtMaturity=4)
        bond.setPricingEngine(rl.FastSwitchingEngine(model, order=order, regime=0))
        assert bond.NPV() == pytest.approx(exact, rel=1e-9)            # -8.7 against 0.143 at spacing 1e-5, before
