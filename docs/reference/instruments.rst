Instruments
===========

Instruments accept QuantLib payoff and exercise objects or plain tuples. Each entry names the QuantLib engine its
frozen limit is checked against.

Bonds
-----

.. function:: rl.ZeroCouponBond(maturity, dayCounter=None, regimeAtMaturity=None)

Unit face value paid at ``maturity``. With ``regimeAtMaturity = j`` the face value is paid only if the regime at
maturity is ``j``, which prices the memory of the regime; the sum over ``j`` is the plain bond. Under an intensity
model the bond price is the survival probability. Frozen limit: ``ql.Vasicek.discountBond``,
``ql.CoxIngersollRoss.discountBond``, the Hull–White and G2 curves.

.. code-block:: python

    bond = rl.ZeroCouponBond(5.0)
    bond.setPricingEngine(rl.FastSwitchingEngine(model, order=4))
    bond.NPV(), bond.delta(), bond.gamma()          # delta and gamma in r0

.. function:: rl.CouponBond(cashflows=None, faceAmount=None, couponRate=None, times=None, dayCounter=None)

Fixed cash flows as a list of ``(time, amount)``, or QuantLib-style ``(faceAmount, couponRate, times)`` with the
last time carrying the face. Priced as the sum of zero-coupon bonds.

.. code-block:: python

    rl.CouponBond(faceAmount=100.0, couponRate=0.05, times=[1.0, 2.0, 3.0, 4.0, 5.0])
    rl.CouponBond(cashflows=[(1.0, 5.0), (2.0, 105.0)])

Options
-------

.. function:: rl.VanillaOption(payoff, exercise=None, maturity=None, dayCounter=None)

European or American option. ``payoff`` is a QuantLib ``PlainVanillaPayoff``, ``CashOrNothingPayoff`` or
``AssetOrNothingPayoff``, or a tuple: ``("call", K)``, ``("put", K)``, ``("cash", "call", K, cash)``,
``("asset", "put", K)``. ``exercise`` is a QuantLib ``EuropeanExercise`` or ``AmericanExercise``, or the string
``"american"``; otherwise give ``maturity``. American exercise is priced by ``SwitchingFDEngine``. Frozen limit:
``AnalyticEuropeanEngine``, ``AnalyticHestonEngine``, ``JumpDiffusionEngine``, ``BatesEngine``,
``VarianceGammaEngine``, ``AnalyticDigitalAmericanEngine`` for the digitals, ``FdBlackScholesVanillaEngine`` for
the American.

.. code-block:: python

    european = rl.VanillaOption(("call", 100.0), maturity=1.0)
    american = rl.VanillaOption(("put", 105.0), exercise="american", maturity=1.0)
    digital = rl.VanillaOption(("cash", "call", 100.0, 10.0), maturity=0.5)
    ql_style = rl.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Put, 105.0), ql.AmericanExercise(today, expiry))

.. function:: VanillaOption.impliedVolatility(price=None, accuracy=1e-10, maxEvaluations=200, minVol=1e-4, maxVol=4.0)

The Black volatility reproducing the price (the instrument's own ``NPV()`` unless ``price`` is given), for a plain
vanilla payoff, as QuantLib's ``VanillaOption.impliedVolatility``.

.. function:: rl.BarrierOption(barrierType, barrier, rebate, payoff, exercise=None, maturity=None, dayCounter=None)

Continuously monitored single barrier. ``barrierType`` is QuantLib's ``Barrier.DownIn``, ``UpIn``, ``DownOut``,
``UpOut`` or one of ``"downin"``, ``"upin"``, ``"downout"``, ``"upout"``; the rebate is paid at the hit for
knock-out and at expiry for knock-in. Priced by ``SwitchingFDEngine`` with the grid truncated at the barrier;
knock-in is the vanilla less the knock-out. Frozen limit: ``AnalyticBarrierEngine``.

.. code-block:: python

    ko = rl.BarrierOption("downout", 80.0, 0.0, ("put", 100.0), maturity=1.0)
    ko.setPricingEngine(rl.SwitchingFDEngine(model, regime=0, n=1601, steps=600))

.. function:: rl.ContinuousGeometricAsianOption(payoff, exercise=None, maturity=None, dayCounter=None)

Fixed-strike option on the continuous geometric average of the price from now to expiry. The time average of the
log price gives a forcing quadratic in time to maturity, so the characteristic-function engines price it exactly.
Frozen limit: ``AnalyticContinuousGeometricAveragePriceAsianEngine``.

.. code-block:: python

    asian = rl.ContinuousGeometricAsianOption(("call", 100.0), maturity=1.0)
    asian.setPricingEngine(rl.NumericalSwitchingEngine(model))

Interest-rate options
---------------------

.. function:: rl.ZeroCouponBondOption(kind, strike, maturity, bondMaturity)

European call or put expiring at ``maturity`` on the unit bond maturing at ``bondMaturity``, under
``SwitchingVasicek``, ``SwitchingHullWhite`` or ``SwitchingG2``, by Gil–Pelaez integrals conditioned on the regime
at expiry. Frozen limit: ``Vasicek.discountBondOption``, ``HullWhite.discountBondOption``, ``G2.discountBondOption``.

.. code-block:: python

    rl.ZeroCouponBondOption("call", 0.9, 2.0, 5.0)

.. function:: rl.CouponBondOption(kind, strike, maturity, cashflows, dayCounter=None)

European option on a bond with fixed cash flows ``[(time, amount), ...]`` after expiry, by Jamshidian's
decomposition conditioned on the regime at expiry (one crossing per regime), under ``SwitchingVasicek`` and
``SwitchingHullWhite``; on the short-rate grid under CIR and G2 as well.

.. function:: rl.Swaption(kind, maturity, fixedTimes, fixedRate, notional=1.0, dayCounter=None, exerciseTimes=None)

European or Bermudan swaption on a fixed-for-floating swap: ``kind`` ``"payer"`` or ``"receiver"``, expiry, the
fixed-leg payment times (the first accrual starts at expiry), fixed rate and notional. A receiver swaption is a call
on the coupon bond struck at par, a payer swaption the put. With ``exerciseTimes`` (or a QuantLib
``BermudanExercise`` in place of ``maturity``) the swaption is Bermudan and priced by ``SwitchingFDEngine`` on the
short-rate grid. Frozen limit: ``JamshidianSwaptionEngine``, ``G2SwaptionEngine``, ``FdHullWhiteSwaptionEngine``,
``FdG2SwaptionEngine``.

.. code-block:: python

    european = rl.Swaption("payer", 2.0, [3.0, 4.0, 5.0, 6.0, 7.0], 0.035, notional=100.0)
    bermudan = rl.Swaption("payer", 1.0, [2.0, 3.0, 4.0, 5.0, 6.0], 0.035, exerciseTimes=[1.0, 2.0, 3.0, 4.0, 5.0])
    bermudan.setPricingEngine(rl.SwitchingFDEngine(hullWhiteModel, n=1201, steps=600))

.. function:: rl.CapFloor(kind, times, strike, notional=1.0, dayCounter=None)

Cap or floor on the simple forward rate over the consecutive periods ``times = [T0, ..., Tn]``. Each caplet is
``(1 + tau K)`` puts on the zero-coupon bond maturing at the period end, expiring at its start, struck at
``1 / (1 + tau K)``; floorlets are the calls. Frozen limit: ``AnalyticCapFloorEngine`` (G2: the sum of its bond puts).

.. code-block:: python

    cap = rl.CapFloor("cap", [1.0, 2.0, 3.0, 4.0, 5.0], 0.03)

Credit
------

.. function:: rl.CreditDefaultSwap(side, spread, times, recovery, discount=0.0, accrualOnDefault=True, dayCounter=None)

Protection on a unit notional with the premium ``spread`` paid at ``times``, recovery ``recovery``, on a model whose
bond price is the survival probability (``SwitchingVasicek``, ``SwitchingCoxIngersollRoss``, ``SwitchingVasicekJumps``
used as intensities). ``discount`` is a flat risk-free rate or a callable ``t -> discount factor``. Protection is
valued at the mid-point of each accrual period, as QuantLib's ``MidPointCdsEngine``. ``fairSpread()``,
``couponLegNPV()`` and ``defaultLegNPV()`` are available after ``NPV()``.

.. code-block:: python

    intensity = rl.SwitchingCoxIngersollRoss(chain, 0.02, theta=[0.05, 0.01], k=0.5, sigma=0.08)
    cds = rl.CreditDefaultSwap("buyer", 0.02, [0.5 * i for i in range(1, 11)], 0.4, discount=0.03)
    cds.setPricingEngine(rl.NumericalSwitchingEngine(intensity, regime=0))
    cds.fairSpread()

.. function:: rl.FirstToDefaultSwap(...)

The same class on a ``SwitchingIntensityBasket``: the premium runs until the first default among the names, the
protection pays at the first default.
