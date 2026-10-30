"""An engine must price the contract it is given or refuse it. These cover the cases where a feature of the contract
(early exercise, a barrier, an average, a digital payoff, a payment contingent on the final regime) was dropped and a
plausible European vanilla price returned instead."""
import math
import numpy as np
import pytest
import regimelib as rl

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)
BS = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.05, 0.0, [0.2, 0.2])


def N(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def fourier_engines(model):
    return [rl.FastSwitchingEngine(model, order=0), rl.NumericalSwitchingEngine(model)]


@pytest.mark.parametrize("payoff", [("put", 100.0), ("cash", "call", 100.0, 10.0), ("asset", "put", 100.0)])
def test_fourier_engines_refuse_american_exercise(payoff):
    option = rl.VanillaOption(payoff, exercise="american", maturity=1.0)
    for engine in fourier_engines(BS):
        option.setPricingEngine(engine)
        with pytest.raises(TypeError, match="American exercise"):
            option.NPV()


def test_fourier_engines_refuse_barriers():
    option = rl.BarrierOption("downout", 80.0, 0.0, ("put", 100.0), maturity=1.0)
    for engine in fourier_engines(BS):
        option.setPricingEngine(engine)
        with pytest.raises(TypeError, match="a barrier"):
            option.NPV()


def test_hybrid_fourier_engines_refuse_american_and_barrier():
    hybrid = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.0, 0.0, [0.2, 0.2]),
                                     rl.SwitchingVasicek(CHAIN, 0.03, 0.5, 0.04, 0.01))
    for option in (rl.VanillaOption(("put", 120.0), exercise="american", maturity=1.0),
                   rl.BarrierOption("upout", 130.0, 0.0, ("call", 100.0), maturity=1.0)):
        for engine in fourier_engines(hybrid):
            option.setPricingEngine(engine)
            with pytest.raises(TypeError):
                option.NPV()


def test_fourier_engines_refuse_bermudan_swaptions_and_accept_european():
    model = rl.SwitchingVasicek(CHAIN, 0.03, 0.5, [0.05, 0.05], [0.01, 0.01])
    fixed = [2.0, 3.0, 4.0]
    bermudan = rl.Swaption("payer", 1.0, fixed, 0.04, exerciseTimes=[1.0, 2.0])
    european = rl.Swaption("payer", 1.0, fixed, 0.04)
    single = rl.Swaption("payer", 1.0, fixed, 0.04, exerciseTimes=[1.0])
    for engine in fourier_engines(model):
        bermudan.setPricingEngine(engine)
        with pytest.raises(TypeError, match="Bermudan"):
            bermudan.NPV()
        european.setPricingEngine(engine); single.setPricingEngine(engine)
        assert single.NPV() == pytest.approx(european.NPV(), rel=1e-12)


def test_grid_engine_refuses_the_geometric_asian():
    option = rl.ContinuousGeometricAsianOption(("call", 100.0), maturity=1.0)
    option.setPricingEngine(rl.SwitchingFDEngine(BS, n=201, steps=50))
    with pytest.raises(TypeError, match="geometric averaging"):
        option.NPV()


def test_first_order_engines_refuse_path_features():
    cev = rl.SwitchingCEVProcess(CHAIN, 80.0, 0.0, 0.0, [0.2, 0.2], 1.0)
    options = [rl.VanillaOption(("put", 100.0), exercise="american", maturity=1.0),
               rl.BarrierOption("upout", 70.0, 0.0, ("call", 100.0), maturity=1.0),
               rl.ContinuousGeometricAsianOption(("call", 100.0), maturity=1.0)]
    for option in options:
        for engine in (rl.FirstOrderFDEngine(cev, n=201), rl.SwitchingFDReferee(cev, n=201)):
            option.setPricingEngine(engine)
            with pytest.raises(TypeError):
                option.NPV()


@pytest.mark.parametrize("kind", ["call", "put"])
def test_first_order_engines_price_digitals_as_digitals(kind):
    S0, r, sigma, K, T = 100.0, 0.03, 0.2, 105.0, 1.0
    cev = rl.SwitchingCEVProcess(CHAIN, S0, r, 0.0, [sigma, sigma], 1.0)          # beta = 1, equal regimes: Black-Scholes
    d1 = (math.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T)); d2 = d1 - sigma * math.sqrt(T)
    sign = 1.0 if kind == "call" else -1.0
    exact = {("cash", 1.0): math.exp(-r * T) * N(sign * d2), ("cash", 7.0): 7.0 * math.exp(-r * T) * N(sign * d2),
             ("asset", None): S0 * N(sign * d1)}
    for (ptype, cash), value in exact.items():
        payoff = ("cash", kind, K, cash) if ptype == "cash" else ("asset", kind, K)
        option = rl.VanillaOption(payoff, maturity=T)
        for engine in (rl.FirstOrderFDEngine(cev, n=801), rl.SwitchingFDReferee(cev, n=801)):
            option.setPricingEngine(engine)
            assert option.NPV() == pytest.approx(value, rel=2e-2)


def test_monte_carlo_refuses_path_features_and_prices_digitals():
    for option in (rl.VanillaOption(("put", 120.0), exercise="american", maturity=1.0),
                   rl.BarrierOption("upout", 130.0, 0.0, ("call", 100.0), maturity=1.0),
                   rl.ContinuousGeometricAsianOption(("call", 100.0), maturity=1.0)):
        option.setPricingEngine(rl.MonteCarloSwitchingEngine(BS, paths=10))
        with pytest.raises(TypeError):
            option.NPV()
    S0, r, sigma, K, T = 100.0, 0.05, 0.2, 110.0, 1.0
    d1 = (math.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T)); d2 = d1 - sigma * math.sqrt(T)
    for payoff, value in ((("cash", "call", K, 10.0), 10.0 * math.exp(-r * T) * N(d2)),
                          (("cash", "put", K, 10.0), 10.0 * math.exp(-r * T) * N(-d2)),
                          (("asset", "call", K), S0 * N(d1)), (("asset", "put", K), S0 * N(-d1))):
        option = rl.VanillaOption(payoff, maturity=T)
        option.setPricingEngine(rl.MonteCarloSwitchingEngine(BS, paths=50))       # equal regimes: every path agrees
        assert option.NPV() == pytest.approx(value, rel=1e-12)


def test_monte_carlo_pays_terminal_regime_bonds_only_in_that_regime():
    model = rl.SwitchingVasicek(CHAIN, 0.03, 0.5, [0.06, 0.02], [0.015, 0.008])
    T, parts = 2.0, []
    for j in (0, 1):
        claim = rl.ZeroCouponBond(T, regimeAtMaturity=j)
        mc = rl.MonteCarloSwitchingEngine(model, regime=0, paths=20000, seed=3)
        claim.setPricingEngine(mc); value = claim.NPV()
        claim.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0))
        assert abs(value - claim.NPV()) < 4 * mc.standardError
        parts.append(value)
    plain = rl.ZeroCouponBond(T); plain.setPricingEngine(rl.MonteCarloSwitchingEngine(model, regime=0, paths=20000, seed=3))
    assert sum(parts) == pytest.approx(plain.NPV(), rel=1e-12)                    # same paths, so the split is exact
    for j, expected in ((0, 1.0), (1, 0.0)):                                       # at T = 0 the regime is the starting one
        now = rl.ZeroCouponBond(0.0, regimeAtMaturity=j)
        now.setPricingEngine(rl.MonteCarloSwitchingEngine(model, regime=0, paths=5))
        assert now.NPV() == expected


def test_implied_volatility_refuses_contracts_the_black_formula_does_not_describe():
    options = [rl.VanillaOption(("put", 100.0), exercise="american", maturity=1.0),
               rl.BarrierOption("downout", 80.0, 0.0, ("put", 100.0), maturity=1.0)]
    for option in options:
        option.setPricingEngine(rl.SwitchingFDEngine(BS, n=201, steps=50))
        with pytest.raises(NotImplementedError):
            option.impliedVolatility()
    asian = rl.ContinuousGeometricAsianOption(("call", 100.0), maturity=1.0)
    asian.setPricingEngine(rl.NumericalSwitchingEngine(BS))
    with pytest.raises(NotImplementedError):
        asian.impliedVolatility()
    vanilla = rl.VanillaOption(("call", 100.0), maturity=1.0); vanilla.setPricingEngine(rl.NumericalSwitchingEngine(BS))
    assert vanilla.impliedVolatility() == pytest.approx(0.2, abs=1e-6)
