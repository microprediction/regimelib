"""Issue #278: hybrid Fourier pricing from a regime with no diffusion, where the terminal price has an atom."""
import math
import pytest
import regimelib as rl

CHAIN = rl.RegimeChain([[-0.1, 0.1], [0.0, 0.0]])


def hybrid(equity_sigma, rate_sigma, b=(0.10, -0.10), r0=0.10):
    return rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(CHAIN, S0=100.0, r=0.0, q=0.0, sigma=list(equity_sigma)),
                                   rl.SwitchingVasicek(CHAIN, r0=r0, a=2.0, b=list(b), sigma=list(rate_sigma)))


@pytest.mark.parametrize("model", [hybrid([0.0, 0.0], [0.0, 0.0]), hybrid([0.0, 0.2], [0.0, 0.01])])
def test_hybrid_transform_refuses_an_atom_at_once(model):
    option = rl.VanillaOption(("call", 100.0 * math.exp(0.1)), maturity=1.0)
    option.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0))   # ran for minutes before
    with pytest.raises(ArithmeticError, match="atom"):
        option.NPV()


def test_hybrid_transform_still_prices_from_a_regime_that_diffuses():
    option = rl.VanillaOption(("call", 100.0), maturity=1.0)
    option.setPricingEngine(rl.NumericalSwitchingEngine(hybrid([0.0, 0.2], [0.0, 0.01]), regime=1))
    assert option.NPV() > 0.0
