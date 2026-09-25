Helpers and calibration
=======================

.. function:: rl.VolatilityHelper(maturity, strike, volatility, kind="call", dayCounter=None)

A European option quoted in Black volatility, as QuantLib's ``HestonModelHelper``. After ``setPricingEngine`` it
gives ``marketValue()``, ``modelValue()``, ``impliedVolatility()``, ``calibrationError()`` (relative price error)
and ``volatilityError()``.

.. function:: rl.calibrate(model, helpers, parameters, engine=None, bounds=None, useVolatilityError=False, **kwargs)

Least squares on the helpers' calibration errors over the named model attributes (per-regime lists) and ``"chain"``
for the off-diagonal switching rates; ``engine`` is a factory ``model -> engine`` (default the numerical engine).
Returns the scipy result and leaves the model at the fitted values.

.. code-block:: python

    helpers = [rl.VolatilityHelper(T, K, vol) for T, K, vol in quotes]
    model = rl.SwitchingBlackScholesProcess(rl.RegimeChain.twoState(3.0, 3.0), 100.0, 0.02, 0.0, [0.30, 0.20])
    result = rl.calibrate(model, helpers, ["sigma", "chain"], engine=lambda m: rl.NumericalSwitchingEngine(m, regime=1))
    model.sigma, model.chain.generator
    max(abs(h.calibrationError()) for h in helpers)
