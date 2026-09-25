Getting started
===============

Installation
------------

regimelib depends on numpy, scipy, mpmath and sympy. QuantLib is optional: instruments accept QuantLib payoff,
exercise and date objects when it is present, and the certificates compare with QuantLib's engines.

.. code-block:: bash

    pip install regimelib
    pip install QuantLib          # optional

Importing
---------

.. code-block:: python

    import regimelib as rl
    import QuantLib as ql         # optional

Everything documented here is exported at the top level: ``rl.RegimeChain``, ``rl.SwitchingVasicek``,
``rl.VanillaOption``, ``rl.FastSwitchingEngine`` and so on. The closed forms live in ``rl.symbolic``.

Three lines of pricing
----------------------

A model takes a chain and per-regime parameters (a scalar means the same value in every regime); an instrument
takes its contractual terms; an engine binds them.

.. code-block:: python

    chain = rl.RegimeChain.twoState(3.0, 5.0)
    model = rl.SwitchingBlackScholesProcess(chain, S0=100.0, r=0.03, q=0.0, sigma=[0.35, 0.15])
    option = rl.VanillaOption(("call", 100.0), maturity=1.0)
    option.setPricingEngine(rl.FastSwitchingEngine(model, order=4, regime=0))
    option.NPV(), option.delta(), option.gamma(), option.theta(), option.rho()

Change the engine to ``rl.NumericalSwitchingEngine(model, regime=0)`` to solve the reduced system without expanding,
or to ``rl.SwitchingFDEngine(model, regime=0)`` for an American exercise.
