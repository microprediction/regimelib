Basics
======

Regime chains
-------------

.. function:: rl.RegimeChain(generator)

The hidden regime is a continuous-time Markov chain given by its generator matrix ``Q`` (rows sum to zero, off-diagonal
entries are the switching rates). Any number of regimes is allowed.

.. code-block:: python

    chain = rl.RegimeChain([[-5.0, 3.0, 2.0],
                            [4.0, -9.0, 5.0],
                            [1.0, 6.0, -7.0]])
    chain.numberOfRegimes()          # 3
    chain.stationaryDistribution()   # the long-run occupation of each regime
    chain.meanHoldingTime()          # n / -trace Q, the expansion parameter

.. function:: rl.RegimeChain.twoState(rate12, rate21)

The two-regime chain with switching rate ``rate12`` out of regime 0 and ``rate21`` out of regime 1.

.. code-block:: python

    chain = rl.RegimeChain.twoState(3.0, 5.0)

Chains from data
----------------

regimelib prices; it does not estimate, and it takes no position on how a chain is fitted or under which measure.
What an estimator produces is usually a transition matrix at the data frequency and a probability for each regime
at the last observation. Two functions connect those to the library.

.. function:: rl.RegimeChain.fromTransitionMatrix(P, dt)

The chain whose transition matrix over a step ``dt`` is ``P`` (rows are the regime now). Not every transition
matrix comes from a continuous-time chain: for two regimes it does exactly when ``p11 + p22 > 1``, and in general
when its principal logarithm is a generator. A matrix that does not is refused, since how to move it to one that
does is a modelling choice.

.. function:: chain.transitionMatrix(dt)

``exp(Q dt)``, the transition matrix of the chain over a step ``dt``.

.. code-block:: python

    P = [[0.98, 0.02], [0.08, 0.92]]                         # monthly, from any estimator
    chain = rl.RegimeChain.fromTransitionMatrix(P, 1 / 12)
    model = rl.SwitchingVasicek(chain, r0=0.03, a=0.5, b=[0.045, 0.015], sigma=[0.006, 0.018])
    bond = rl.ZeroCouponBond(5.0)
    bond.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=[0.3, 0.7]))    # the filtered probabilities
    bond.NPV()

The starting regime given as probabilities is the estimator's belief at the last observation
(:doc:`../models/information`). A chain estimated from monthly or quarterly data typically switches slowly, where
``NumericalSwitchingEngine`` is the engine to use; the expansion is for chains that switch fast against the life of
the instrument.

Per-regime parameters
---------------------

A model parameter that may switch is given as a list with one value per regime, or as a scalar for the same value
in every regime. The **frozen limit** — every regime equal — is the QuantLib model with those parameters, whatever
the chain does; the certificates check each model against QuantLib in that limit.

.. code-block:: python

    rl.SwitchingVasicek(chain, r0=0.03, a=0.5, b=[0.06, 0.02], sigma=0.012)   # b switches, sigma does not

Maturities and dates
--------------------

Times are years. Wherever a maturity is expected, a QuantLib ``Date`` is also accepted and is measured from
``ql.Settings.instance().evaluationDate`` with Actual/365 unless a ``dayCounter`` is given; a ``VanillaOption`` takes
its maturity from a QuantLib exercise object.

.. code-block:: python

    ql.Settings.instance().evaluationDate = ql.Date(1, 1, 2020)
    bond = rl.ZeroCouponBond(ql.Date(1, 1, 2025))
    option = rl.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Put, 100.0),
                              ql.EuropeanExercise(ql.Date(1, 1, 2021)))

Instruments, engines and results
--------------------------------

.. function:: instrument.setPricingEngine(engine)
.. function:: instrument.NPV()
.. function:: instrument.delta(), instrument.gamma(), instrument.theta(), instrument.vega(), instrument.rho()

As in QuantLib, an instrument holds its terms, an engine holds the model and the method, and ``NPV()`` triggers the
calculation. The greeks are computed by differentiating the pricing formula, never by bumping; a greek the engine
does not provide raises ``RuntimeError("... not provided by the engine")``, as QuantLib's "not provided" does.

- Options under the characteristic-function engines: ``delta``, ``gamma``, ``theta``, ``rho`` in the spot, and
  ``vega`` in the initial variance for stochastic-volatility models.
- Bonds under Vasicek and CIR: ``delta`` and ``gamma`` in the short rate.
- Options on the finite-difference engine: ``delta``, ``gamma``, ``theta`` from the grid.

.. code-block:: python

    option.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=1))
    option.NPV()
    option.delta()
    option.impliedVolatility()            # the Black volatility reproducing the price

Regimes and the starting regime
-------------------------------

Every engine takes ``regime``, the regime in force today. The price of the same instrument differs by starting
regime; the difference is the *memory* of the regime, which the expansion carries in its initial-layer terms.

.. code-block:: python

    for regime in range(chain.numberOfRegimes()):
        option.setPricingEngine(rl.FastSwitchingEngine(model, order=4, regime=regime))
        print(regime, option.NPV())

Warnings and diagnostics
------------------------

.. class:: rl.ExpansionWarning

The expansion is asymptotic in the mean holding time times the forcing, so it can fail to converge: for a slow chain,
a large difference between regimes, or at high Fourier frequencies. ``FastSwitchingEngine`` raises an
``ExpansionWarning`` when the last term kept is not smaller than the one before it, when it is more than a
thousandth of the value, or when nodes of the Fourier integral had to be solved numerically because the series
diverged there. ``instrument._result("diagnostics")`` returns the holding time, the order used, the relative size
of the last term and the count of numerical nodes.

.. code-block:: python

    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        price = option.NPV()
    for w in caught:
        print(w.message)
    option._result("diagnostics")
