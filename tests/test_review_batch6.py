"""Chains the expansion is not for, payoffs in rare regimes, the fast engine's theta, and the hybrid model's
quadrature, grid width and barriers."""
import math
import warnings

import numpy as np
import pytest

import regimelib as rl


def test_rates_whose_sum_overflows_are_refused():
    with pytest.raises(ValueError, match="overflows"):
        rl.RegimeChain.twoState(1e308, 1e308)
    assert rl.RegimeChain.twoState(1e150, 1e150).meanHoldingTime() == pytest.approx(1e-150)


def test_fast_engine_refuses_a_chain_with_a_slow_mode_between_fast_groups():
    Q = np.array([[-1e7, 1e7, 0.0], [1e7, -1e7 - 1.0, 1.0], [0.0, 1.0, -1.0]])
    model = rl.SwitchingVasicek(rl.RegimeChain(Q), 0.03, 0.5, [0.06, 0.02, 0.04], 0.01)
    with pytest.raises(ValueError, match="NumericalSwitchingEngine"):
        rl.FastSwitchingEngine(model)
    rl.FastSwitchingEngine(rl.SwitchingVasicek(rl.RegimeChain.twoState(1e-8, 5.0), 0.03, 0.5, [0.06, 0.02], 0.01))   # one mode: fine


@pytest.mark.parametrize("rare", [1e-10, 1e-14, 2e-16])
def test_bond_contingent_on_a_rare_regime(rare):
    # zero rates: the bond paying in regime 1 is the transition probability, q01 (1 - e^{-(q01 + q10) T}) / (q01 + q10)
    chain = rl.RegimeChain.twoState(rare, 1.0)
    model = rl.SwitchingVasicek(chain, 0.0, 0.5, [0.0, 0.0], [0.0, 0.0])
    bond = rl.ZeroCouponBond(1.0, regimeAtMaturity=1)
    bond.setPricingEngine(rl.FastSwitchingEngine(model, order=2, information="observed"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rl.ExpansionWarning)
        assert bond.NPV() == pytest.approx(rare * -math.expm1(-(rare + 1.0)) / (rare + 1.0), rel=1e-9)


def test_fast_theta_is_of_the_order_of_the_price():
    chain = rl.RegimeChain.twoState(30.0, 20.0)
    model = rl.SwitchingBlackScholesProcess(chain, S0=100.0, r=0.03, q=0.01, sigma=[0.50, 0.10])

    def option(T, regime, engine):
        o = rl.VanillaOption(("call", 100.0), maturity=T); o.setPricingEngine(engine(regime)); return o
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rl.ExpansionWarning)
        for regime in (0, 1):
            zero = lambda i: rl.FastSwitchingEngine(model, regime=i, order=0, nodes=256)
            h = 1e-4
            slope = -(option(1.0 + h, regime, zero).NPV() - option(1.0 - h, regime, zero).NPV()) / (2 * h)
            assert option(1.0, regime, zero).theta() == pytest.approx(slope, rel=1e-6)       # at order zero, exactly its own
            exact = option(1.0, regime, lambda i: rl.NumericalSwitchingEngine(model, regime=i, nodes=256)).theta()
            errors = [abs(option(1.0, regime, lambda i: rl.FastSwitchingEngine(model, regime=i, order=n, nodes=256)).theta() - exact)
                      for n in (1, 2, 4)]
            assert errors[0] < 5e-3 and errors[1] < 5e-4 and errors[2] < 5e-5


@pytest.mark.parametrize("r, q, kind", [(0.03, 0.45, "call"), (0.40, 0.0, "put")])
def test_hybrid_quadrature_resolves_the_forward_moneyness(r, q, kind):
    # identical regimes and rho = 0: Black's formula with the Vasicek bond, the forward and the total variance
    chain = rl.RegimeChain.twoState(3.0, 5.0)
    a, sr, ss, T, S0, K = 0.4, 0.01, 0.05, 10.0, 100.0, 100.0
    model = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(chain, S0=S0, r=0.0, q=q, sigma=ss),
                                    rl.SwitchingVasicek(chain, r0=r, a=a, b=r, sigma=sr), rho=0.0)
    vR = sr ** 2 / a ** 2 * (T - 2 * -math.expm1(-a * T) / a + -math.expm1(-2 * a * T) / (2 * a))
    P = math.exp(-r * T + vR / 2); F = S0 * math.exp(-q * T) / P; V = ss ** 2 * T + vR
    N = lambda x: 0.5 * math.erfc(-x / math.sqrt(2))
    d1 = (math.log(F / K) + V / 2) / math.sqrt(V); d2 = d1 - math.sqrt(V)
    black = P * (F * N(d1) - K * N(d2)) if kind == "call" else P * (K * N(-d2) - F * N(-d1))
    option = rl.VanillaOption((kind, K), maturity=T)
    option.setPricingEngine(rl.FastSwitchingEngine(model, order=0))
    assert option.NPV() == pytest.approx(black, abs=1e-10)


def test_hybrid_grid_is_wide_enough_for_the_integrated_rate():
    chain = rl.RegimeChain.twoState(3.0, 5.0)
    model = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(chain, S0=100.0, r=0.03, q=0.0, sigma=0.02),
                                    rl.SwitchingVasicek(chain, r0=0.03, a=0.15, b=0.03, sigma=0.30), rho=0.0)
    option = rl.VanillaOption(("call", 100.0), maturity=3.0)
    option.setPricingEngine(rl.NumericalSwitchingEngine(model, nodes=256)); transform = option.NPV()
    option.setPricingEngine(rl.SwitchingFDEngine(model, n=(241, 61), steps=180))
    assert option.NPV() == pytest.approx(transform, rel=5e-3)


def test_hybrid_barriers_are_refused_with_stochastic_rates_and_priced_without():
    chain = rl.RegimeChain.twoState(3.0, 5.0)
    equity = rl.SwitchingBlackScholesProcess(chain, S0=100.0, r=0.03, q=0.01, sigma=0.20)
    barrier = rl.BarrierOption("downout", barrier=80.0, rebate=0.0, payoff=("call", 100.0), maturity=1.0)
    stochastic = rl.SwitchingEquityRates(equity, rl.SwitchingVasicek(chain, r0=0.03, a=0.4, b=0.03, sigma=0.01))
    barrier.setPricingEngine(rl.SwitchingFDEngine(stochastic, n=(101, 21), steps=50))
    with pytest.raises(NotImplementedError, match="stochastic rates"):
        barrier.NPV()
    frozen = rl.SwitchingEquityRates(equity, rl.SwitchingVasicek(chain, r0=0.03, a=0.4, b=0.03, sigma=0.0))
    barrier.setPricingEngine(rl.SwitchingFDEngine(frozen, n=(201, 21), steps=100)); hybrid = barrier.NPV()
    barrier.setPricingEngine(rl.SwitchingFDEngine(equity, n=201, steps=100))
    assert hybrid == pytest.approx(barrier.NPV(), rel=1e-10)


def test_two_state_symbolic_bond_is_one_at_zero_maturity_and_follows_the_engine_through_the_layer():
    from regimelib.symbolic import VasicekTwoStateBond
    values = dict(r0=0.03, kappa=0.5, theta1=0.07, theta2=0.01, sigma1=0.012, sigma2=0.008, lam=25.0)
    model = rl.SwitchingVasicek(rl.RegimeChain.twoState(25.0, 25.0), 0.03, 0.5, [0.07, 0.01], [0.012, 0.008])
    for regime in (0, 1):
        formula = VasicekTwoStateBond(regime=regime)
        assert formula.evaluate(formula.price, T=1e-3, **values) == pytest.approx(1.0, abs=1e-4)
        for T in (0.01, 0.1, 5.0):
            bond = rl.ZeroCouponBond(T); bond.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=regime))
            assert formula.evaluate(formula.price, T=T, **values) == pytest.approx(bond.NPV(), abs=2e-7)


def test_first_order_grid_engine_carries_the_initial_layer():
    # without the layer the memory term does not vanish as T -> 0: at T = 0.02 it gave 0.88 for 1.07 from regime 1
    chain = rl.RegimeChain.twoState(20.0, 30.0)
    model = rl.SwitchingBlackScholesProcess(chain, S0=100.0, r=0.03, q=0.01, sigma=[0.3, 0.15])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rl.ExpansionWarning)
        for T, rel in ((0.02, 6e-2), (0.1, 5e-3), (1.0, 2e-4)):
            for regime in (0, 1):
                option = rl.VanillaOption(("call", 100.0), maturity=T)
                option.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=regime)); exact = option.NPV()
                option.setPricingEngine(rl.FirstOrderFDEngine(model, n=801, regime=regime))
                assert option.NPV() == pytest.approx(exact, rel=rel)
