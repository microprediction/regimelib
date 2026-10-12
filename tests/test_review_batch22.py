"""Issue #285: the CIR grid sized by the level reached by the last decision."""
import pytest
import regimelib as rl


def test_cir_grid_follows_the_finite_horizon_level():
    model = rl.SwitchingCoxIngersollRoss(rl.RegimeChain([[0.0]]), 0.001, 0.04, 0.10, 0.01)   # far below its level
    option = rl.ZeroCouponBondOption("call", 0.9542351967574313, 0.02, 5.02)                 # one week, ATM forward
    option.setPricingEngine(rl.SwitchingFDEngine(model))
    assert option.NPV() == pytest.approx(0.00006818477536152390, rel=2e-3)   # exact CIR formula; 10% high before
