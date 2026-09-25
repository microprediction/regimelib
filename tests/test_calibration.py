"""Calibration recovers a two-regime Black-Scholes model from the smile it generates, in QuantLib's helper mold."""
import time
import numpy as np
import regimelib as rl


def test_calibrate_recovers_switching_smile():
    chain = rl.RegimeChain.twoState(4.0, 2.0)
    truth = rl.SwitchingBlackScholesProcess(chain, 100.0, 0.02, 0.0, [0.40, 0.15])
    eng = rl.NumericalSwitchingEngine(truth, regime=1)
    helpers = []
    for T in (0.25, 1.0):
        for K in (80.0, 90.0, 100.0, 110.0, 120.0):
            o = rl.VanillaOption(("call", K), maturity=T); o.setPricingEngine(eng)
            helpers.append(rl.VolatilityHelper(T, K, o.impliedVolatility()))
    vols = [h.volatility for h in helpers]
    assert max(vols) - min(vols) > 0.02                       # the switching model makes a real smile
    model = rl.SwitchingBlackScholesProcess(rl.RegimeChain.twoState(3.0, 3.0), 100.0, 0.02, 0.0, [0.30, 0.20])
    t = time.time()
    res = rl.calibrate(model, helpers, ["sigma", "chain"], engine=lambda m: rl.NumericalSwitchingEngine(m, regime=1), xtol=1e-8, ftol=1e-10)
    print(f"calibration {time.time() - t:.1f}s, cost {res.cost:.2e}, sigma {model.sigma}, rates {model.chain.generator[0,1]:.3f} {model.chain.generator[1,0]:.3f}")
    assert max(abs(h.calibrationError()) for h in helpers) < 1e-6
    assert np.allclose(model.sigma, [0.40, 0.15], atol=2e-3)
    assert abs(model.chain.generator[0, 1] - 4.0) < 0.1 and abs(model.chain.generator[1, 0] - 2.0) < 0.1
