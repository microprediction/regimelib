Examples
========

A smile from a regime
---------------------

Two volatility regimes make a smile without any other ingredient; calibration recovers them from the quotes.

.. code-block:: python

    import regimelib as rl

    chain = rl.RegimeChain.twoState(4.0, 2.0)
    truth = rl.SwitchingBlackScholesProcess(chain, 100.0, 0.02, 0.0, [0.40, 0.15])
    engine = rl.NumericalSwitchingEngine(truth, regime=1)
    quotes = []
    for T in (0.25, 1.0):
        for K in (80.0, 90.0, 100.0, 110.0, 120.0):
            o = rl.VanillaOption(("call", K), maturity=T); o.setPricingEngine(engine)
            quotes.append(rl.VolatilityHelper(T, K, o.impliedVolatility()))

    model = rl.SwitchingBlackScholesProcess(rl.RegimeChain.twoState(3.0, 3.0), 100.0, 0.02, 0.0, [0.30, 0.20])
    rl.calibrate(model, quotes, ["sigma", "chain"], engine=lambda m: rl.NumericalSwitchingEngine(m, regime=1))
    print(model.sigma, model.chain.generator)        # [0.40, 0.15], rates 4 and 2

Which order to trust
--------------------

The expansion at successive orders against the numerical solution, and the warning when the chain is too slow.

.. code-block:: python

    chain = rl.RegimeChain.twoState(12.0, 8.0)
    model = rl.SwitchingVasicek(chain, 0.03, 0.5, [0.06, 0.02], [0.015, 0.008])
    swaption = rl.Swaption("payer", 2.0, [3.0, 4.0, 5.0, 6.0, 7.0], 0.035, notional=100.0)
    swaption.setPricingEngine(rl.NumericalSwitchingEngine(model)); reference = swaption.NPV()
    for order in (0, 1, 2, 3, 4):
        swaption.setPricingEngine(rl.FastSwitchingEngine(model, order=order))
        print(order, swaption.NPV() - reference)

    slow = rl.SwitchingBlackScholesProcess(rl.RegimeChain.twoState(4.0, 2.0), 100.0, 0.02, 0.0, [0.40, 0.15])
    option = rl.VanillaOption(("call", 100.0), maturity=1.0)
    option.setPricingEngine(rl.FastSwitchingEngine(slow, order=None))
    option.NPV()                                      # raises ExpansionWarning: use NumericalSwitchingEngine

Default correlation from a common regime
----------------------------------------

Two names whose intensities rise in the same regime default together more often than independent names would.

.. code-block:: python

    chain = rl.RegimeChain.twoState(1.0, 1.0)
    name = lambda levels: rl.SwitchingVasicek(chain, 0.02, 0.5, levels, 0.003)
    same = rl.SwitchingIntensityBasket([name([0.01, 0.06]), name([0.01, 0.06])])
    opposite = rl.SwitchingIntensityBasket([name([0.01, 0.06]), name([0.06, 0.01])])
    same.defaultCorrelation(5.0), opposite.defaultCorrelation(5.0)      # positive, negative

    ftd = rl.FirstToDefaultSwap("buyer", 0.01, [0.5 * i for i in range(1, 11)], 0.4, discount=0.03)
    ftd.setPricingEngine(rl.NumericalSwitchingEngine(same, regime=0))
    ftd.fairSpread()

A greek as a formula
--------------------

.. code-block:: python

    from regimelib.symbolic import VasicekTwoStateBond
    f = VasicekTwoStateBond(regime=0)
    f.terms["green_kubo"]                # the first-order term of log P as a sympy expression
    f.greek("lam")                       # sensitivity of the price to the switching rate
    f.logGreekTerms("theta1")            # d log P / d theta1, term by term
