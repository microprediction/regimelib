Pricing models
==============

Each model is QuantLib's model with a ``chain`` in front of the parameters and a list where a parameter may switch.
The third column of the tables says which parameters may switch; the rest are common to all regimes.

Short-rate models
-----------------

.. function:: rl.SwitchingVasicek(chain, r0, a, b, sigma)

QuantLib ``Vasicek(r0, a, b, sigma)``: ``dr = a (b - r) dt + sigma dW``. ``b`` and ``sigma`` may switch. Bonds,
bond options, swaptions, caps and Bermudans; as an intensity, survival probabilities and CDS.

.. code-block:: python

    rl.SwitchingVasicek(chain, r0=0.03, a=0.5, b=[0.06, 0.02], sigma=[0.015, 0.008])

.. function:: rl.SwitchingVasicekJumps(chain, r0, a, b, sigma, jumpIntensity, jumpMean)

Vasicek with compound-Poisson jumps of exponential size (mean ``jumpMean``) at intensity ``jumpIntensity``, which may
switch along with ``b`` and ``sigma``. The reduction is exact. QuantLib has no jump short-rate model; the frozen
limit is checked against the affine closed form.

.. function:: rl.SwitchingCoxIngersollRoss(chain, r0, theta, k, sigma)

QuantLib ``CoxIngersollRoss(r0, theta, k, sigma)``: ``dr = k (theta - r) dt + sigma sqrt(r) dW``. ``theta`` may
switch. Bonds and Bermudan swaptions on the rate grid; as an intensity, survival probabilities and CDS.

.. function:: rl.SwitchingHullWhite(chain, termStructure, a, sigma)

QuantLib ``HullWhite(termStructure, a, sigma)`` with ``sigma`` switching. ``termStructure`` is a flat rate, a
callable ``t -> discount factor`` or a QuantLib ``YieldTermStructureHandle``. The fitted drift uses the
stationary-average variance, so the averaged model reproduces the curve exactly and the expansion adds the
switching corrections.

.. code-block:: python

    ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.03, ql.Actual365Fixed()))
    hw = rl.SwitchingHullWhite(chain, ts, a=0.5, sigma=[0.02, 0.006])

.. function:: rl.SwitchingG2(chain, termStructure, a, sigma, b, eta, rho)

QuantLib ``G2(termStructure, a, sigma, b, eta, rho)``; ``sigma``, ``eta`` and ``rho`` may switch. Bonds, bond
options and caps by the characteristic-function engines; swaptions and Bermudans on the two-factor grid.

.. function:: rl.SwitchingIntensityBasket(models)

Several intensities (Vasicek or CIR) driven by one chain, with independent diffusions. ``ZeroCouponBond`` is the
joint survival probability and ``CreditDefaultSwap`` a first-to-default swap; ``defaultCorrelation(t)`` measures
the dependence the common regime creates. The frozen limit factorises into the marginal survivals.

.. code-block:: python

    basket = rl.SwitchingIntensityBasket([rl.SwitchingVasicek(chain, 0.02, 0.5, [0.01, 0.06], 0.003),
                                          rl.SwitchingVasicek(chain, 0.02, 0.5, [0.01, 0.06], 0.003)])
    basket.defaultCorrelation(5.0)

Equity models
-------------

.. function:: rl.SwitchingBlackScholesProcess(chain, S0, r, q, sigma)

QuantLib ``BlackScholesMertonProcess`` with ``sigma`` switching. Vanilla, digital, Asian, barrier and American
options; the two-regime characteristic function is available in closed form (:doc:`symbolic`).

.. function:: rl.SwitchingHestonModel(chain, S0, r, q, v0, kappa, theta, sigma, rho)

QuantLib ``HestonModel`` with the long-run variance ``theta`` switching. The reduction is exact; ``vega`` is the
derivative in ``v0``.

.. function:: rl.SwitchingHestonVolOfVol(chain, S0, r, q, v0, kappa, theta, xi, rho)

Heston with the volatility of variance ``xi`` switching. The switched operators do not reduce exactly, so this model
is priced by :func:`rl.FirstOrderFDEngine` on a ``(log S, v)`` grid, ``n = (nx, nv)``.

.. function:: rl.SwitchingMerton76Process(chain, S0, r, q, sigma, jumpIntensity, logJumpMean, logJumpVol)

QuantLib ``Merton76Process``; ``sigma`` and ``jumpIntensity`` may switch.

.. function:: rl.SwitchingBatesModel(chain, S0, r, q, v0, kappa, theta, sigma, rho, jumpIntensity, logJumpMean, logJumpVol)

QuantLib ``BatesModel``; ``theta`` and ``jumpIntensity`` may switch.

.. function:: rl.SwitchingVarianceGammaProcess(chain, S0, r, q, sigma, nu, theta)

QuantLib ``VarianceGammaProcess``; all three parameters may switch.

.. function:: rl.SwitchingCEVProcess(chain, S0, r, q, sigma, beta)

QuantLib ``CEVProcess`` with ``sigma`` switching; first-order tier (the switched diffusion does not commute with the
drift). Frozen limit: ``AnalyticCEVEngine``.

Hybrid models
-------------

.. function:: rl.SwitchingEquityRates(equity, rates, rho=0.0)

An equity (``SwitchingBlackScholesProcess`` or ``SwitchingHestonModel``) with stochastic rates (``SwitchingVasicek``
or ``SwitchingHullWhite``) on one chain, so that the discount and the return are dependent through the regime path.
``rho`` is the equity–rate Brownian correlation, allowed for the Black–Scholes equity. Vanilla options by Lewis's
formula with the discounted characteristic function; American options by :func:`rl.SwitchingFDEngine` on a
``(log S, r)`` grid, ``n = (nx, nr)``, for the Black–Scholes equity with Vasicek rates. Frozen limit: ``AnalyticBSMHullWhiteEngine``,
``AnalyticHestonHullWhiteEngine``.

.. code-block:: python

    hybrid = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(chain, 100.0, 0.03, 0.01, [0.35, 0.15]),
                                     rl.SwitchingVasicek(chain, 0.03, 0.4, [0.06, 0.01], [0.02, 0.008]), rho=0.3)
    option.setPricingEngine(rl.NumericalSwitchingEngine(hybrid, regime=1))
