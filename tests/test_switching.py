"""With switching on, the expansion must converge to the numerical referee, faster at higher order."""
import pytest
import regimelib as rl


def _errors(model_fn, instrument, orders=(0, 1, 2, 4)):
    out = {}
    for lam in (10.0, 40.0):
        chain = rl.RegimeChain.twoState(lam, 1.5 * lam)
        model = model_fn(chain)
        instrument.setPricingEngine(rl.NumericalSwitchingEngine(model)); ref = instrument.NPV()
        errs = []
        for o in orders:
            instrument.setPricingEngine(rl.FastSwitchingEngine(model, order=o)); errs.append(abs(instrument.NPV() - ref))
        out[lam] = errs
    return out


def test_vasicek_orders_converge():
    e = _errors(lambda c: rl.SwitchingVasicek(c, 0.03, 0.5, [0.07, 0.01], [0.012, 0.008]), rl.ZeroCouponBond(5.0))
    for lam in e:
        assert e[lam][0] > e[lam][1] > e[lam][2] > e[lam][3]
    # error at order 4 falls by at least (10/40)^4 / 2 when the switching rate quadruples
    assert e[40.0][3] < 2 * e[10.0][3] * (10.0 / 40.0) ** 4 + 1e-13


def test_heston_orders_converge():
    e = _errors(lambda c: rl.SwitchingHestonModel(c, 100.0, 0.01, 0.0, 0.04, 1.5, [0.09, 0.02], 0.4, -0.5, ), rl.VanillaOption(("call", 100.0), maturity=1.0), orders=(0, 1, 2))
    for lam in e:
        assert e[lam][0] > e[lam][1] > e[lam][2]


def test_g2_orders_converge():
    e = _errors(lambda c: rl.SwitchingG2(c, 0.03, 0.5, [0.015, 0.005], 0.1, [0.01, 0.004], [-0.5, 0.2]), rl.ZeroCouponBond(5.0))
    for lam in e:
        assert e[lam][0] > e[lam][1] > e[lam][2] > e[lam][3]
