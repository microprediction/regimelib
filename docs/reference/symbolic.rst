Symbolic
========

Formulas rather than procedures: sympy expressions for prices and greeks, checked against the engines to round-off
where the engine computes the same truncation and to the expected order otherwise.

.. class:: rl.symbolic.VasicekTwoStateBond(regime=0)

The two-state Vasicek bond to second order in the holding time. ``.terms`` holds the state, averaged, Green–Kubo,
second-order and memory terms of ``log P``; ``.price`` is the formula; ``.greek("r0")``, ``.greek("theta1")``,
``.greek("lam")`` differentiate it; ``.delta()``, ``.gamma()``, ``.theta()`` are QuantLib's names;
``.evaluate(expr, **values)`` evaluates.

.. code-block:: python

    from regimelib.symbolic import VasicekTwoStateBond
    f = VasicekTwoStateBond(regime=0)
    f.greek("r0")
    f.evaluate(f.delta(), r0=0.03, kappa=0.5, theta1=0.06, theta2=0.02, sigma1=0.015, sigma2=0.008, lam=8.0, T=5.0)

.. class:: rl.symbolic.VasicekBondFirstOrder()
.. class:: rl.symbolic.CIRBondFirstOrder()
.. class:: rl.symbolic.VasicekJumpsBondFirstOrder()

The bond under any finite chain at first order in the holding time. The chain enters only through numbers —
the stationary averages, the Green–Kubo integrals and the memory coefficients of the starting regime — which
``coefficients(chain, ...)`` computes from the generator. The CIR integrals are closed by the Riccati identity,
the jump integrals by the substitution ``u = exp(-kappa t)``.

.. code-block:: python

    from regimelib.symbolic import VasicekBondFirstOrder
    f = VasicekBondFirstOrder()
    numbers = f.coefficients(chain, kappa_=0.5, thetas=[0.08, 0.05, 0.01], sigmas=[0.015, 0.01, 0.006], regime=0)
    f.evaluate(f.price, r0=0.03, kappa=0.5, T=4.0, **numbers)

.. function:: parameterGreek(chain, wrt, regime=0, **params)

Greeks in the model's own parameters: ``wrt = ("theta", i)``, ``("sigma", i)``, ``("intensity", i)`` for a
per-regime parameter or ``("q", a, b)`` for the switching rate out of regime ``a`` into ``b``. The chain rule runs
through the coefficients, whose derivatives are exact — the Green–Kubo matrix is bilinear in the forcing, and the
group inverse and the stationary distribution vary with the generator as ``dQ# = -Q# dQ Q# + 1 pi dQ (Q#)^2 +
(Q#)^2 dQ 1 pi`` and ``dpi = -pi dQ Q#``. Checked against finite differences of the engine at order 1.

.. code-block:: python

    params = dict(r0=0.03, kappa=0.5, T=4.0, thetas=[0.08, 0.05, 0.01], sigmas=[0.015, 0.01, 0.006])
    f.parameterGreek(chain, ("theta", 2), regime=1, **params)      # dP / d theta_2
    f.parameterGreek(chain, ("q", 0, 1), regime=1, **params)       # dP / d Q_01

.. class:: rl.symbolic.TwoStateConstantForcing()

The exact, all-orders solution of the two-regime reduced system with constant forcing — a 2 × 2 matrix exponential
written with its eigenvalues — which is the closed-form characteristic function of every two-regime Black–Scholes,
Merton or variance-gamma model. ``.a(regime)`` in ``q12, q21, g1, g2, T``; ``.blackScholes(regime)`` in
``u, sigma1, sigma2``.
