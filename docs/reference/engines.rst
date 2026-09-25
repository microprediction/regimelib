Pricing engines
===============

Every engine takes the model and the starting ``regime``. The characteristic-function engines price bonds, European
options, digitals, geometric Asians, bond options, swaptions, caps and credit default swaps; the grid engines price
early exercise and barriers; Monte Carlo is a referee.

The expansion
-------------

.. function:: rl.FastSwitchingEngine(model, order=4, regime=0, nodes=96, tol=1e-10, maxOrder=12, rtol=1e-12)

The fast-switching expansion of the reduced system ``a' = (Q + diag g) a`` in the mean holding time of the chain,
``n / -trace Q``, through ``order`` terms with the initial layer that carries the starting regime. ``order=None``
adds terms until successive orders agree to ``tol`` or the series stops improving — the best truncation of an
asymptotic series — and records ``orderUsed`` and ``lastIncrement`` after ``NPV()``. At Fourier nodes where the
series diverges (large forcing at high frequency) the reduced system is solved numerically instead and the count is
reported; see :class:`rl.ExpansionWarning`.

.. code-block:: python

    engine = rl.FastSwitchingEngine(model, order=None, regime=0)
    option.setPricingEngine(engine)
    option.NPV()
    engine.orderUsed, engine.lastIncrement
    option._result("diagnostics")

The numerical solution
----------------------

.. function:: rl.NumericalSwitchingEngine(model, regime=0, nodes=96, rtol=1e-12)

The reduced system solved without expansion: the matrix exponential when the forcing is constant in time
(Black–Scholes, Merton, variance gamma), an ODE solver (DOP853) otherwise. The terminal vectors are memoised across
strikes at each maturity, which is what makes calibration fast. This is the referee for the expansion with
switching on, and it is exact up to quadrature for every model in the all-orders tier.

.. code-block:: python

    option.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=1))

Monte Carlo
-----------

.. function:: rl.MonteCarloSwitchingEngine(model, regime=0, paths=100000, seed=0)

A grid-free referee: regime paths are sampled exactly and each path is priced in closed form (Vasicek bonds,
Black–Scholes options). ``standardError`` is set after ``NPV()``.

.. code-block:: python

    mc = rl.MonteCarloSwitchingEngine(model, regime=0, paths=200000, seed=1)
    bond.setPricingEngine(mc)
    bond.NPV(), mc.standardError

Finite differences
------------------

.. function:: rl.SwitchingFDEngine(model, regime=0, n=1001, steps=400, width=None, stretch=None)

Time-stepping on the coupled system ``u_i' = L_i u_i + sum_j Q_ij u_j``, Rannacher's implicit half-steps followed by
Crank–Nicolson. American exercise projects onto the payoff after every step; a knock-out barrier truncates the grid
at the barrier node; Bermudan swaptions and coupon-bond options compare with the exercise value, per regime, at
each exercise date on a short-rate grid (Vasicek, CIR, Hull–White in its zero-mean factor, G2 on a two-factor grid
with ``n = (nx, ny)``); American options under the hybrid model on a ``(log S, r)`` grid. ``stretch`` concentrates the
nodes of a two-dimensional grid around the strike and the starting factor by a sinh map (the width of the dense
region as a fraction of the interval; 0.15 halves the Heston vol-of-vol grid error at a given node count); uniform
when ``None``, which suits swaptions, whose exercise region is broad. ``delta``, ``gamma`` and ``theta`` come from the grid for equity options.

.. code-block:: python

    american = rl.VanillaOption(("put", 105.0), exercise="american", maturity=1.0)
    american.setPricingEngine(rl.SwitchingFDEngine(model, regime=0, n=1601, steps=800))
    american.NPV(), american.delta(), american.gamma()

The first-order tier
--------------------

.. function:: rl.FirstOrderFDEngine(model, regime=0, n=801, width=None, warnAbove=0.03, stretch=None)

For models whose reduction is not exact (``SwitchingCEVProcess``, ``SwitchingHestonVolOfVol``): the averaged
operator, the Green–Kubo correction ``sum K_jk A_j A_k`` carried by an augmented linear system, and the memory term
of the starting regime, all at first order in the holding time. After ``NPV()`` the engine exposes ``averaged``,
``correction`` and ``memory`` and ``diagnostics`` with the relative size of the correction and ``estimatedError``,
about its square; it warns above ``warnAbove``.

.. code-block:: python

    cev = rl.SwitchingCEVProcess(chain, 100.0, 0.03, 0.0, sigma=[0.35, 0.15], beta=0.7)
    engine = rl.FirstOrderFDEngine(cev, regime=0, n=801)
    option.setPricingEngine(engine)
    option.NPV()
    engine.averaged, engine.correction, engine.memory, engine.diagnostics

.. function:: rl.SwitchingFDReferee(model, regime=0, n=801, width=None)

The switching model solved without expansion on the same grid, the referee for the first-order tier.
