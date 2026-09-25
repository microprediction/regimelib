Certificates
============

The test suite is the library's claim. Each certificate checks a frozen limit against QuantLib and the switching
case against an independent referee; the tolerances below are what the tests assert.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Model / instrument
     - Frozen limit (QuantLib)
     - Switching referee
   * - Vasicek, CIR bonds
     - ``discountBond`` 1e-10
     - numerical solution; Monte Carlo with exact regime paths
   * - Hull–White, G2 bonds and bond options
     - ``discountBondOption`` 1e-7
     - expansion converges to the numerical solution; Monte Carlo (G2)
   * - Black–Scholes, Heston, Merton, Bates, variance gamma
     - analytic engines 1e-8
     - numerical solution; averaged-parameter limit
   * - Digitals, implied volatility, geometric Asian
     - analytic engines 1e-8
     - numerical solution; Monte Carlo (Asian)
   * - American, barrier
     - ``FdBlackScholesVanillaEngine`` 5e-5, ``AnalyticBarrierEngine`` 3e-5
     - Brownian-bridge Monte Carlo
   * - Swaptions, caps, coupon-bond options
     - ``JamshidianSwaptionEngine``, ``AnalyticCapFloorEngine`` 2e-6
     - grid European equals Jamshidian with switching
   * - Bermudan swaptions
     - ``FdHullWhiteSwaptionEngine`` 1e-4, ``FdG2SwaptionEngine``
     - Bermudan dominates every European; grid convergence
   * - CDS, first-to-default
     - ``MidPointCdsEngine`` 2e-4
     - frozen basket factorises; Monte Carlo joint survival
   * - Hybrid equity–rates
     - ``AnalyticBSMHullWhiteEngine`` (with correlation), ``AnalyticHestonHullWhiteEngine`` 1e-6
     - expansion converges; parity with the switching bond
   * - CEV, Heston vol-of-vol (first-order tier)
     - ``AnalyticCEVEngine``, ``AnalyticHestonEngine`` (grid error)
     - coupled switching PDE; error tracks the estimate
   * - Symbolic formulas
     - —
     - engine at the same order to 1e-12; finite differences of the numerical solution
   * - Calibration
     - —
     - recovers a two-regime smile's parameters exactly

Run ``pytest`` for everything (about a quarter of an hour) or ``pytest -m 'not slow'`` for all but the four slowest.
