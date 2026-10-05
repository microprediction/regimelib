"""Follow-ups to the belief, ends-of-domain and contract-dispatch work."""
import math
import warnings
import numpy as np
import pytest
import regimelib as rl

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)


def test_belief_diagnostics_are_the_worst_case_and_do_not_depend_on_labels():
    def run(chain, sigma, belief):
        model = rl.SwitchingBlackScholesProcess(chain, 100.0, 0.02, 0.0, sigma)
        engine = rl.FastSwitchingEngine(model, order=None, regime=belief)
        option = rl.VanillaOption(("call", 100.0), maturity=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); option.setPricingEngine(engine); value = option.NPV()
        return value, engine.orderUsed, option._result("diagnostics")
    fast = rl.RegimeChain.twoState(40.0, 60.0); swapped = rl.RegimeChain.twoState(60.0, 40.0)
    value, used, diag = run(fast, [0.3, 0.15], [0.3, 0.7])
    value2, used2, diag2 = run(swapped, [0.15, 0.3], [0.7, 0.3])                 # the same model with the labels exchanged
    assert diag["orderUsed"] == used and diag2["orderUsed"] == used2
    assert value2 == pytest.approx(value, rel=1e-9) and diag2["orderUsed"] == diag["orderUsed"]
    assert diag2["lastTermRelative"] == pytest.approx(diag["lastTermRelative"], rel=1e-6, abs=1e-15)
    only, _, alone = run(fast, [0.3, 0.15], [1.0, 0.0])                          # a regime with no weight is not priced
    single, used0, diag0 = run(fast, [0.3, 0.15], 0)
    assert only == pytest.approx(single, rel=1e-14) and alone["orderUsed"] == diag0["orderUsed"]


def test_belief_monte_carlo_uses_independent_streams():
    """A symmetric chain with a symmetric belief: the two conditional estimates are independent, so the reported
    standard error is the root sum of squares, and it matches the spread of repeated runs."""
    chain = rl.RegimeChain.twoState(1.0, 1.0)
    model = rl.SwitchingVasicek(chain, 0.03, 0.5, [0.08, 0.01], [0.02, 0.005])
    estimates, errors = [], []
    for seed in range(24):
        engine = rl.MonteCarloSwitchingEngine(model, regime=[0.5, 0.5], paths=400, seed=seed)
        bond = rl.ZeroCouponBond(3.0); bond.setPricingEngine(engine)
        estimates.append(bond.NPV()); errors.append(engine.standardError)
    spread = float(np.std(estimates, ddof=1))
    assert 0.6 < spread / float(np.mean(errors)) < 1.5
    one = rl.MonteCarloSwitchingEngine(model, regime=0, paths=400, seed=3)       # without a belief the stream is the seed's
    bond = rl.ZeroCouponBond(3.0); bond.setPricingEngine(one); first = bond.NPV()
    again = rl.MonteCarloSwitchingEngine(model, regime=0, paths=400, seed=3); bond.setPricingEngine(again)
    assert bond.NPV() == first


@pytest.mark.parametrize("K", [0.0, -0.1])
def test_monte_carlo_at_a_nonpositive_strike(K):
    S0, r, q, T = 100.0, 0.02, 0.01, 1.0
    for sigma in (0.2, [0.3, 0.15]):
        model = rl.SwitchingBlackScholesProcess(CHAIN, S0, r, q, sigma)
        expected = {("call", K): S0 * math.exp(-q * T) - K * math.exp(-r * T), ("put", K): 0.0,
                    ("cash", "call", K, 7.0): 7.0 * math.exp(-r * T), ("cash", "put", K, 7.0): 0.0,
                    ("asset", "call", K): S0 * math.exp(-q * T), ("asset", "put", K): 0.0}
        for payoff, value in expected.items():
            engine = rl.MonteCarloSwitchingEngine(model, paths=10)
            option = rl.VanillaOption(payoff, maturity=T); option.setPricingEngine(engine)
            assert option.NPV() == pytest.approx(value, rel=1e-14) and engine.standardError == 0.0
            option.setPricingEngine(rl.NumericalSwitchingEngine(model))
            assert option.NPV() == pytest.approx(value, rel=1e-14)               # the transform engine agrees


def test_exercise_is_recognised_or_refused():
    import QuantLib as ql
    ref = ql.Date(1, 1, 2020); ql.Settings.instance().evaluationDate = ref
    expiry = ref + ql.Period(365, ql.Days)
    assert rl.VanillaOption(("put", 100.0), maturity=1.0).isAmerican is False
    assert rl.VanillaOption(("put", 100.0), exercise="American", maturity=1.0).isAmerican is True
    assert rl.VanillaOption(("put", 100.0), exercise="european", maturity=1.0).isAmerican is False
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, 100.0)
    assert rl.VanillaOption(payoff, ql.AmericanExercise(ref, expiry)).isAmerican is True
    assert rl.VanillaOption(payoff, ql.EuropeanExercise(expiry)).isAmerican is False
    for bad in ("americna", "", "bermudan"):
        with pytest.raises(ValueError, match="exercise"):
            rl.VanillaOption(("put", 100.0), exercise=bad, maturity=1.0)
    for bad in (1, 2.5, object(), ql.BermudanExercise([expiry])):
        with pytest.raises(TypeError, match="exercise"):
            rl.VanillaOption(("put", 100.0), exercise=bad, maturity=1.0)
