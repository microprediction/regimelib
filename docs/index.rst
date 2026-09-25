regimelib
=========

QuantLib's models with a hidden Markov regime, priced by the fast-switching expansion. The Python reference
implementation of `regimeswitching.org <https://regimeswitching.org/>`_.

A regime is a finite-state Markov chain that the market does not observe directly. Any parameter of a QuantLib
model may take a different value in each regime: the Vasicek mean level, the Black–Scholes volatility, the Heston
long-run variance, a jump intensity, a default intensity. regimelib prices the usual instruments under such models
in QuantLib's mold — ``setPricingEngine``, ``NPV()``, greeks as methods on the instrument — with three tiers of
engine: the fast-switching expansion in the mean holding time of the chain, the numerical solution of the reduced
system it expands, and grids and Monte Carlo for early exercise and paths.

Every model has a frozen limit, all regimes equal, in which it is QuantLib's model; every certificate in the test
suite checks that limit against QuantLib's own engine and then checks the switching case against an independent
referee.

.. code-block:: python

    import regimelib as rl

    chain = rl.RegimeChain.twoState(3.0, 5.0)              # rates out of regime 0 and out of regime 1
    model = rl.SwitchingVasicek(chain, r0=0.03, a=0.5, b=[0.06, 0.02], sigma=[0.015, 0.008])
    bond = rl.ZeroCouponBond(5.0)
    bond.setPricingEngine(rl.FastSwitchingEngine(model, order=4, regime=0))
    print(bond.NPV(), bond.delta(), bond.gamma())

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getting_started

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/basics
   reference/instruments
   reference/engines
   reference/models
   reference/helpers
   reference/symbolic

.. toctree::
   :maxdepth: 1
   :caption: Background

   background
   certificates

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples
