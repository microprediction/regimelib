"""Constructors refuse inputs that do not describe a model or a contract, naming the argument, instead of pricing
something else or failing later inside an engine."""
import math
import numpy as np
import pytest
import regimelib as rl

CHAIN = rl.RegimeChain.twoState(3.0, 5.0)
NAN, INF = float("nan"), float("inf")


# ---------------------------------------------------------------- the chain
def test_generator_must_be_real_nonempty_and_consistent():
    with pytest.raises(ValueError, match="at least one regime"):
        rl.RegimeChain(np.empty((0, 0)))
    with pytest.raises(ValueError, match="real"):
        rl.RegimeChain([[-1 + 1j, 1 - 1j], [2, -2]])
    assert rl.RegimeChain(np.array([[-1, 1], [2, -2]], dtype=complex)).numberOfRegimes() == 2     # zero imaginary part
    assert rl.RegimeChain([[0.0]]).numberOfRegimes() == 1
    with pytest.raises(ValueError, match="sum to zero"):
        rl.RegimeChain([[-1.0, 1.0], [1.0, -0.9]])
    with pytest.raises(ValueError, match="sum to zero"):                    # the same 10% defect in a smaller time unit
        rl.RegimeChain(1e-8 * np.array([[-1.0, 1.0], [1.0, -0.9]]))
    assert rl.RegimeChain(1e-8 * np.array([[-1.0, 1.0], [1.0, -1.0]])).numberOfRegimes() == 2
    with pytest.raises(ValueError, match="finite"):
        rl.RegimeChain([[-INF, INF], [1.0, -1.0]])


def test_chain_owns_its_generator():
    Q = np.array([[-1.0, 1.0], [2.0, -2.0]])
    chain = rl.RegimeChain(Q)
    Q[0, 1] = 50.0                                                           # the caller's array, not the chain's
    assert chain.generator[0, 1] == 1.0
    with pytest.raises(ValueError):
        chain.generator[0, 1] = 50.0


# ---------------------------------------------------------------- per-regime parameters
def test_per_regime_lists_must_match_the_chain():
    assert rl.SwitchingVasicek(CHAIN, 0.03, 0.5, 0.05, 0.01).b == [0.05, 0.05]
    with pytest.raises(ValueError, match="b must be a scalar or have exactly 2 entries"):
        rl.SwitchingVasicek(CHAIN, 0.03, 0.5, [0.05, 0.04, 0.03], 0.01)
    with pytest.raises(ValueError, match="sigma"):
        rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.02, 0.0, [0.2])


@pytest.mark.parametrize("bad", [-0.1, [1.0, -0.5], NAN, INF])
def test_jump_intensities_are_nonnegative(bad):
    with pytest.raises(ValueError, match="jumpIntensity"):
        rl.SwitchingMerton76Process(CHAIN, 100.0, 0.02, 0.0, 0.2, bad, -0.1, 0.1)
    with pytest.raises(ValueError, match="jumpIntensity"):
        rl.SwitchingBatesModel(CHAIN, 100.0, 0.02, 0.0, 0.04, 2.0, 0.04, 0.3, -0.5, bad, -0.1, 0.1)
    with pytest.raises(ValueError, match="jumpIntensity"):
        rl.SwitchingVasicekJumps(CHAIN, 0.03, 0.5, 0.05, 0.01, bad, 0.02)


def test_zero_jump_intensity_and_zero_jump_mean_are_the_no_jump_limits():
    rl.SwitchingMerton76Process(CHAIN, 100.0, 0.02, 0.0, 0.2, 0.0, -0.1, 0.1)
    rl.SwitchingVasicekJumps(CHAIN, 0.03, 0.5, 0.05, 0.01, [0.0, 1.0], 0.02)
    for bad in (-0.01, NAN, INF):
        with pytest.raises(ValueError, match="jumpMean"):
            rl.SwitchingVasicekJumps(CHAIN, 0.03, 0.5, 0.05, 0.01, 1.0, bad)


@pytest.mark.parametrize("bad", [-1.0000001, 1.0000001, NAN, INF, -INF])
def test_correlations_lie_in_the_unit_interval(bad):
    with pytest.raises(ValueError, match="rho"):
        rl.SwitchingHestonModel(CHAIN, 100.0, 0.02, 0.0, 0.04, 2.0, 0.04, 0.3, bad)
    with pytest.raises(ValueError, match="rho"):
        rl.SwitchingBatesModel(CHAIN, 100.0, 0.02, 0.0, 0.04, 2.0, 0.04, 0.3, bad, 1.0, -0.1, 0.1)
    with pytest.raises(ValueError, match="rho"):
        rl.SwitchingHestonVolOfVol(CHAIN, 100.0, 0.02, 0.0, 0.04, 2.0, 0.04, [0.5, 0.2], bad)
    with pytest.raises(ValueError, match="rho"):
        rl.SwitchingG2(CHAIN, 0.03, 0.5, 0.01, 0.1, 0.01, [0.0, bad])
    equity = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.0, 0.0, 0.2)
    rates = rl.SwitchingVasicek(CHAIN, 0.03, 0.5, 0.04, 0.01)
    with pytest.raises(ValueError, match="rho"):
        rl.SwitchingEquityRates(equity, rates, rho=bad)


def test_correlation_endpoints_are_accepted():
    for rho in (-1.0, 1.0):
        rl.SwitchingHestonModel(CHAIN, 100.0, 0.02, 0.0, 0.04, 2.0, 0.04, 0.3, rho)
        rl.SwitchingG2(CHAIN, 0.03, 0.5, 0.01, 0.1, 0.01, rho)
        rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.0, 0.0, 0.2),
                                rl.SwitchingVasicek(CHAIN, 0.03, 0.5, 0.04, 0.01), rho=rho)


def test_variances_and_square_root_levels_are_nonnegative():
    with pytest.raises(ValueError, match="v0"):
        rl.SwitchingHestonModel(CHAIN, 100.0, 0.02, 0.0, -0.04, 2.0, 0.04, 0.3, -0.5)
    with pytest.raises(ValueError, match="theta"):
        rl.SwitchingHestonModel(CHAIN, 100.0, 0.02, 0.0, 0.04, 2.0, [0.04, -0.01], 0.3, -0.5)
    with pytest.raises(ValueError, match="theta"):
        rl.SwitchingBatesModel(CHAIN, 100.0, 0.02, 0.0, 0.04, 2.0, NAN, 0.3, -0.5, 1.0, -0.1, 0.1)
    with pytest.raises(ValueError, match="r0"):
        rl.SwitchingCoxIngersollRoss(CHAIN, -0.01, 0.05, 0.5, 0.1)
    with pytest.raises(ValueError, match="theta"):
        rl.SwitchingCoxIngersollRoss(CHAIN, 0.03, [0.05, -0.02], 0.5, 0.1)
    rl.SwitchingCoxIngersollRoss(CHAIN, 0.0, [0.05, 0.0], 0.5, 0.1)


def test_variance_gamma_needs_a_finite_martingale_moment():
    rl.SwitchingVarianceGammaProcess(CHAIN, 100.0, 0.02, 0.0, 0.2, 0.5, -0.1)
    with pytest.raises(ValueError, match="martingale"):                     # 1 - theta nu - sigma^2 nu / 2 = 1 - 1.5 - 0.02
        rl.SwitchingVarianceGammaProcess(CHAIN, 100.0, 0.02, 0.0, 0.2, [0.5, 1.0], [-0.1, 1.5])
    with pytest.raises(ValueError, match="nu"):
        rl.SwitchingVarianceGammaProcess(CHAIN, 100.0, 0.02, 0.0, 0.2, -0.5, -0.1)


def test_hybrid_needs_an_equity_with_a_characteristic_function():
    rates = rl.SwitchingVasicek(CHAIN, 0.03, 0.5, 0.04, 0.01)
    with pytest.raises(TypeError, match="characteristic function"):
        rl.SwitchingEquityRates(rl.SwitchingCEVProcess(CHAIN, 100.0, 0.0, 0.0, 2.0, 0.6), rates)


# ---------------------------------------------------------------- engines
def test_regime_indices_are_state_labels():
    model = rl.SwitchingVasicek(CHAIN, 0.03, 0.5, 0.05, 0.01)
    bs = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.02, 0.0, 0.2)
    for bad in (-1, 2, 0.0, True):
        for build in (lambda r: rl.NumericalSwitchingEngine(model, regime=r), lambda r: rl.FastSwitchingEngine(model, regime=r),
                      lambda r: rl.MonteCarloSwitchingEngine(model, regime=r), lambda r: rl.SwitchingFDEngine(bs, regime=r),
                      lambda r: rl.FirstOrderFDEngine(bs, regime=r), lambda r: rl.SwitchingFDReferee(bs, regime=r)):
            with pytest.raises(ValueError, match="regime"):
                build(bad)
    assert rl.NumericalSwitchingEngine(model, regime=np.int64(1)).regime == 1
    with pytest.raises(ValueError, match="regimeAtMaturity"):
        rl.ZeroCouponBond(1.0, regimeAtMaturity=-1)
    bond = rl.ZeroCouponBond(1.0, regimeAtMaturity=2); bond.setPricingEngine(rl.NumericalSwitchingEngine(model))
    with pytest.raises(ValueError, match="regimeAtMaturity"):
        bond.NPV()


def test_engine_controls_are_checked_at_construction():
    model = rl.SwitchingVasicek(CHAIN, 0.03, 0.5, 0.05, 0.01)
    bs = rl.SwitchingBlackScholesProcess(CHAIN, 100.0, 0.02, 0.0, 0.2)
    for bad in (-1, 1.5, True, "2"):
        with pytest.raises(ValueError, match="order"):
            rl.FastSwitchingEngine(model, order=bad)
    with pytest.raises(ValueError, match="maxOrder"):
        rl.FastSwitchingEngine(model, order=None, maxOrder=-3)
    assert rl.FastSwitchingEngine(model, order=None).order is None and rl.FastSwitchingEngine(model, order=np.int64(2)).order == 2
    for bad in (-5, 0, 2.5, True):
        with pytest.raises(ValueError, match="steps"):
            rl.SwitchingFDEngine(bs, steps=bad)
    for bad in (1, 0, -3, 1.9):
        with pytest.raises(ValueError, match="paths"):
            rl.MonteCarloSwitchingEngine(model, paths=bad)


# ---------------------------------------------------------------- instruments
def test_side_labels_are_not_guessed():
    with pytest.raises(ValueError):
        rl.VanillaOption(("cal", 100.0), maturity=1.0)
    with pytest.raises(ValueError):
        rl.VanillaOption(("cash", "putt", 100.0, 1.0), maturity=1.0)
    with pytest.raises(ValueError):
        rl.ZeroCouponBondOption("Put option", 0.9, 1.0, 2.0)
    with pytest.raises(ValueError):
        rl.CouponBondOption("c", 1.0, 1.0, [(2.0, 1.05)])
    with pytest.raises(ValueError):
        rl.Swaption("pay", 1.0, [2.0, 3.0], 0.03)
    with pytest.raises(ValueError):
        rl.CapFloor("collar", [0.5, 1.0], 0.03)
    with pytest.raises(ValueError):
        rl.CreditDefaultSwap("protection buyer", 0.01, [0.5, 1.0], 0.4)
    assert rl.Swaption("Receiver", 1.0, [2.0, 3.0], 0.03).isPayer is False
    assert rl.CreditDefaultSwap("SELLER", 0.01, [0.5, 1.0], 0.4).isBuyer is False


def test_schedules_are_ordered_finite_and_in_the_future():
    with pytest.raises(ValueError):
        rl.ZeroCouponBond(-1.0)
    with pytest.raises(ValueError):
        rl.ZeroCouponBond(NAN)
    assert rl.ZeroCouponBond(0.0).maturity == 0.0
    for times in ([], [1.0, 0.5], [0.5, 0.5], [0.0, 1.0], [-0.5, 1.0], [0.5, NAN]):
        with pytest.raises(ValueError):
            rl.CreditDefaultSwap("buyer", 0.01, times, 0.4)
        with pytest.raises(ValueError):
            rl.CouponBond(faceAmount=100.0, couponRate=0.05, times=times)
    for times in ([], [1.0], [1.0, 0.5], [0.5, 0.5], [0.5, INF]):
        with pytest.raises(ValueError):
            rl.CapFloor("cap", times, 0.03)
    for fixed in ([], [3.0, 2.0], [2.0, 2.0], [1.0, 2.0], [0.5, 2.0], [2.0, NAN]):
        with pytest.raises(ValueError):
            rl.Swaption("payer", 1.0, fixed, 0.03)


def test_bermudan_schedules_agree_with_the_swap():
    fixed = [2.0, 3.0, 4.0]
    rl.Swaption("payer", 1.0, fixed, 0.03, exerciseTimes=[1.0, 2.0, 3.0])
    for exercise in ([], [2.0, 1.0], [1.5, 2.0], [1.0, 4.0], [1.0, 5.0], [1.0, 1.0]):
        with pytest.raises(ValueError):
            rl.Swaption("payer", 1.0, fixed, 0.03, exerciseTimes=exercise)


@pytest.mark.parametrize("recovery", [-0.1, 1.1, NAN, INF])
def test_recovery_is_a_fraction(recovery):
    with pytest.raises(ValueError, match="recovery"):
        rl.CreditDefaultSwap("buyer", 0.01, [0.5, 1.0], recovery)


def test_recovery_endpoints_are_accepted():
    for recovery in (0.0, 1.0):
        rl.CreditDefaultSwap("buyer", 0.01, [0.5, 1.0], recovery)
