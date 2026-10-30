"""Fourier ranges sized from the transform being inverted, and values by parity kept within their bounds."""
import math
import warnings

import pytest

import regimelib as rl


def _black(F, K, v, disc, call=True):
    N = lambda x: 0.5 * math.erfc(-x / math.sqrt(2))
    d1 = (math.log(F / K) + v / 2) / math.sqrt(v); d2 = d1 - math.sqrt(v)
    return disc * (F * N(d1) - K * N(d2)) if call else disc * (K * N(-d2) - F * N(-d1))


@pytest.mark.parametrize("strike", [100.0, 120.0])
def test_call_from_a_quiet_regime_of_a_slow_chain(strike):
    # the chain rarely leaves regime 0 within the year, so the price is near Black-Scholes at 5%, far from the average
    lam = 0.01
    model = rl.SwitchingBlackScholesProcess(rl.RegimeChain.twoState(lam, lam), S0=100.0, r=0.0, q=0.0, sigma=[0.05, 0.9])
    option = rl.VanillaOption(("call", strike), maturity=1.0)
    option.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0)); value = option.NPV()
    quiet = _black(100.0, strike, 0.05 ** 2, 1.0)
    assert quiet * math.exp(-lam) <= value <= quiet + 100.0 * -math.expm1(-lam)   # no switch, or at most the spot if one
    option.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0, nodes=2048))       # and the rule has converged
    assert value == pytest.approx(option.NPV(), abs=1e-8)


def test_geometric_asian_range_is_sized_from_the_average():
    # one regime's volatility in both: the continuous geometric Asian is Black with variance sigma^2 T / 3
    s, r, q, T, S0, K = 0.03, 0.02, 0.0, 1.0, 100.0, 101.0
    model = rl.SwitchingBlackScholesProcess(rl.RegimeChain.twoState(3.0, 5.0), S0=S0, r=r, q=q, sigma=s)
    FG = S0 * math.exp((r - q - s * s / 2) * T / 2 + s * s * T / 6)
    option = rl.ContinuousGeometricAsianOption(("call", K), maturity=T)
    option.setPricingEngine(rl.NumericalSwitchingEngine(model))
    assert option.NPV() == pytest.approx(_black(FG, K, s * s * T / 3, math.exp(-r * T)), abs=5e-8)


@pytest.mark.parametrize("build", [
    lambda chain: rl.SwitchingVasicek(chain, r0=0.03, a=0.5, b=[0.03, 0.03], sigma=[0.001, 0.2]),
    lambda chain: rl.SwitchingG2(chain, 0.03, 0.5, [0.001, 0.2], 0.1, [0.001, 0.001], [0.0, 0.0])])
def test_bond_option_from_a_quiet_regime_of_a_slow_chain_is_refused_not_negative(build):
    model = build(rl.RegimeChain.twoState(0.01, 0.01))
    option = rl.ZeroCouponBondOption("call", 0.94, 0.5, 3.0)
    option.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0, information="observed"))
    with pytest.raises(ArithmeticError, match="SwitchingFDEngine"):
        option.NPV()
    option.setPricingEngine(rl.SwitchingFDEngine(model, regime=0, information="observed"))
    assert option.NPV() > 0.0


def test_values_by_parity_stay_within_their_bounds():
    chain = rl.RegimeChain.twoState(3.0, 5.0)
    model = rl.SwitchingBlackScholesProcess(chain, S0=100.0, r=0.02, q=0.0, sigma=0.20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rl.ExpansionWarning)
        for engine in (rl.NumericalSwitchingEngine(model), rl.FastSwitchingEngine(model, order=2)):
            for strike in (30.0, 25.0, 20.0):
                put = rl.VanillaOption(("put", strike), maturity=1.0); put.setPricingEngine(engine)
                exact = _black(100.0 * math.exp(0.02), strike, 0.04, math.exp(-0.02), call=False)
                assert put.NPV() >= 0.0 and put.NPV() == pytest.approx(exact, abs=1e-9)
            for kind, strike in (("call", 400.0), ("put", 20.0)):
                digital = rl.VanillaOption(("cash", kind, strike, 1.0), maturity=1.0); digital.setPricingEngine(engine)
                assert 0.0 <= digital.NPV() <= math.exp(-0.02)
        far = rl.BarrierOption("downin", barrier=20.0, rebate=0.0, payoff=("call", 100.0), maturity=1.0)
        far.setPricingEngine(rl.SwitchingFDEngine(model, n=201, steps=100))
        assert far.NPV() >= 0.0
