"""The grid-free Monte Carlo referee agrees with the numerical engine within a few standard errors, with switching on."""
import pytest
import regimelib as rl

CHAIN = rl.RegimeChain.twoState(6.0, 4.0)


def test_vasicek_bond_monte_carlo():
    model = rl.SwitchingVasicek(CHAIN, 0.03, 0.5, [0.08, 0.01], [0.015, 0.006])
    bond = rl.ZeroCouponBond(3.0)
    bond.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=1)); ref = bond.NPV()
    mc = rl.MonteCarloSwitchingEngine(model, regime=1, paths=40000, seed=1); bond.setPricingEngine(mc)
    est = bond.NPV()
    assert abs(est - ref) < 4 * mc.standardError
    bond.setPricingEngine(rl.FastSwitchingEngine(model, order=0, regime=1))        # the averaged model is off
    assert abs(bond.NPV() - ref) > 10 * mc.standardError


def test_black_scholes_option_monte_carlo():
    model = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.02, 0.0, [0.35, 0.12])
    opt = rl.VanillaOption(("put", 95.0), maturity=1.0)
    opt.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0)); ref = opt.NPV()
    mc = rl.MonteCarloSwitchingEngine(model, regime=0, paths=40000, seed=2); opt.setPricingEngine(mc)
    assert abs(opt.NPV() - ref) < 4 * mc.standardError
