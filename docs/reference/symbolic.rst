Symbolic
========

Formulas rather than procedures: sympy expressions for prices and greeks, checked against the engines to round-off
where the engine computes the same truncation and to the expected order otherwise. Every class has ``.terms`` (the
terms of ``log P``, kept apart), ``.price``, ``.greek(...)`` and ``.evaluate(expr, **values)``.

.. list-table::
   :header-rows: 1
   :widths: 30 22 22 26

   * - Class
     - Chain
     - Order
     - Model page
   * - ``VasicekTwoStateBond``
     - two regimes, equal rates
     - second
     - :doc:`../models/vasicek`
   * - ``VasicekBondFirstOrder``
     - any finite chain
     - first
     - :doc:`../models/vasicek`
   * - ``CIRBondFirstOrder``
     - any finite chain
     - first
     - :doc:`../models/cir`
   * - ``VasicekJumpsBondFirstOrder``
     - any finite chain
     - first
     - :doc:`../models/vasicek_jumps`
   * - ``TwoStateConstantForcing``
     - two regimes, any rates
     - exact
     - :doc:`../models/black_scholes`, :doc:`../models/merton`, :doc:`../models/variance_gamma`

The other models have their closed forms written out, and evaluated in numpy, on their pages under
:doc:`../models/index`; they are not yet sympy classes.

Two regimes, second order
-------------------------

.. class:: rl.symbolic.VasicekTwoStateBond(regime=0)

The two-state Vasicek bond to second order in :math:`\varepsilon = 1/\lambda`:

.. math::

    \log P_{1,2}(T) = -B\,r_0 + \int_0^T\bar g + \frac\varepsilon2\int_0^T\tilde g^{\,2} - \frac{\varepsilon^2}{8}\tilde g(T)^2
      + \log\Big(1 \pm \frac\varepsilon2\tilde g(T) \mp \frac{\varepsilon^2}{4}\tilde g'(T)\Big),
    \qquad g_i = -\kappa\theta_iB + \tfrac12\sigma_i^2B^2 .

``.terms`` holds the five terms as ``state``, ``averaged``, ``green_kubo``, ``second_order`` and ``memory``;
``.greek("r0")``, ``.greek("theta1")``, ``.greek("lam")`` differentiate the price; ``.delta()``, ``.gamma()``,
``.theta()`` are QuantLib's names.

.. code-block:: python

    from regimelib.symbolic import VasicekTwoStateBond
    f = VasicekTwoStateBond(regime=0)
    f.greek("r0")
    f.evaluate(f.delta(), r0=0.03, kappa=0.5, theta1=0.06, theta2=0.02, sigma1=0.015, sigma2=0.008, lam=8.0, T=5.0)

Any chain, first order
----------------------

For a chain with any number of regimes the first-order bond depends on the chain only through numbers: the
stationary averages, the Green–Kubo integrals :math:`K` of the switched coefficients, and the memory coefficients
:math:`m` of the starting regime. ``coefficients(chain, ...)`` computes them from the generator, and the formula is
symbolic in them.

.. class:: rl.symbolic.VasicekBondFirstOrder()

Vasicek, with :math:`g_i = c_iB + d_iB^2`, :math:`c_i = -\kappa\theta_i`, :math:`d_i = \tfrac12\sigma_i^2` and
:math:`I_n = \int_0^TB^n`:

.. math::

    \log P_i = -B\,r_0 - \kappa\bar\theta\,I_1 + \tfrac12\overline{\sigma^2}\,I_2 + K_{cc}I_2 + 2K_{cd}I_3 + K_{dd}I_4
      + \log\big(1 + m_cB + m_dB^2\big).

.. code-block:: python

    from regimelib.symbolic import VasicekBondFirstOrder
    f = VasicekBondFirstOrder()
    numbers = f.coefficients(chain, kappa_=0.5, thetas=[0.08, 0.05, 0.01], sigmas=[0.015, 0.01, 0.006], regime=0)
    f.evaluate(f.price, r0=0.03, kappa=0.5, T=4.0, **numbers)

.. class:: rl.symbolic.CIRBondFirstOrder()

Cox–Ingersoll–Ross with a switching mean level, :math:`g_i = c_iB` with :math:`c_i = -k\theta_i` and :math:`B` the
CIR Riccati solution:

.. math::

    \log P_i = -B\,r_0 - k\bar\theta\,I_1 + K_{cc}I_2 + \log\big(1 + m_cB\big), \qquad
    I_2 = \frac{2}{\sigma^2}\big(T - kI_1 - B\big).

:math:`I_1` is the logarithm in the classical CIR bond, and the Riccati equation closes :math:`I_2` with no further
integration. ``coefficients(chain, k, thetas, regime=0)``.

.. class:: rl.symbolic.VasicekJumpsBondFirstOrder()

Vasicek with exponential jumps of mean :math:`m` at a switching intensity,
:math:`g_i = c_iB + d_iB^2 + \ell_iJ` with :math:`J(t) = 1/(1 + mB) - 1`:

.. math::

    \log P_i = -B\,r_0 + \bar c\,I_1 + \bar d\,I_2 + \bar\ell\,I_J
      + K_{cc}I_2 + 2K_{cd}I_3 + K_{dd}I_4 + 2K_{c\ell}I_{JB} + 2K_{d\ell}I_{JB^2} + K_{\ell\ell}I_{JJ}
      + \log\big(1 + m_cB + m_dB^2 + m_\ell J\big).

The jump integrals are closed by the substitution :math:`u = e^{-\kappa t}`. Building the formula takes about half
a minute of sympy. ``coefficients(chain, kappa_, thetas, sigmas, intensities, regime=0)``.

Greeks in the model's parameters
--------------------------------

.. function:: parameterGreek(chain, wrt, regime=0, **params)

A method of the three first-order classes. ``wrt = ("theta", i)``, ``("sigma", i)``, ``("intensity", i)`` for a
per-regime parameter or ``("q", a, b)`` for the switching rate out of regime ``a`` into ``b``. The chain rule runs
through the coefficients, whose derivatives are exact: the Green–Kubo matrix is bilinear in the forcing, and the
group inverse and the stationary distribution vary with the generator as

.. math::

    dQ^\# = -Q^\#\,dQ\,Q^\# + \mathbf 1\pi\,dQ\,(Q^\#)^2 + (Q^\#)^2\,dQ\,\mathbf 1\pi, \qquad d\pi = -\pi\,dQ\,Q^\#.

Checked against finite differences of the engine at order 1.

.. code-block:: python

    params = dict(r0=0.03, kappa=0.5, T=4.0, thetas=[0.08, 0.05, 0.01], sigmas=[0.015, 0.01, 0.006])
    f.parameterGreek(chain, ("theta", 2), regime=1, **params)      # dP / d theta_2
    f.parameterGreek(chain, ("q", 0, 1), regime=1, **params)       # dP / d Q_01

Two regimes, constant forcing
-----------------------------

.. class:: rl.symbolic.TwoStateConstantForcing()

The exact solution of the two-regime reduced system with constant forcing, a 2 × 2 matrix exponential written with
its eigenvalues :math:`\mu_\pm`:

.. math::

    a(T) = \frac{e^{\mu_+T}(M - \mu_-I) - e^{\mu_-T}(M - \mu_+I)}{\mu_+ - \mu_-}\,\mathbf 1, \qquad M = Q + \operatorname{diag} g .

This is the closed-form characteristic function of every two-regime Black–Scholes, Merton or variance-gamma model.
``.a(regime)`` in ``q12, q21, g1, g2, T``; ``.blackScholes(regime)`` in ``u, sigma1, sigma2``.
