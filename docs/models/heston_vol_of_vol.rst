Heston with a switching volatility of variance
==============================================

``rl.SwitchingHestonVolOfVol(chain, S0, r, q, v0, kappa, theta, xi, rho)``: Heston with the volatility of variance
``xi`` switching. This is the one model on these pages with no reduced system.

The model
---------

.. math::

    dX_t = (r - q - \tfrac12v_t)\,dt + \sqrt{v_t}\,dW_t, \qquad dv_t = \kappa\,(\theta - v_t)\,dt + \xi_{y_t}\sqrt{v_t}\,dZ_t,
    \qquad d\langle W, Z\rangle = \rho\,dt .

Why the reduction fails
-----------------------

The volatility of variance enters the Riccati equation for :math:`D`, so each regime would need its own
:math:`D` and the form :math:`e^{D(t)v}a_i(t)` cannot hold. In operator terms the generator is

.. math::

    L_i = \bar L + \big(\xi_i^2 - \overline{\xi^2}\big)\,A_1 + \big(\xi_i - \bar\xi\big)\,A_2, \qquad
    A_1 = \tfrac12v\,\partial_{vv}, \quad A_2 = \rho\,v\,\partial_{xv},

and :math:`A_1`, :math:`A_2` do not commute with the mean reversion in :math:`\bar L`.

The first-order rule
--------------------

The expansion still applies to the operator-valued system. With :math:`f_1 = \xi^2`, :math:`f_2 = \xi` the switched
coefficients, the Green–Kubo matrix and the memory vector of the chain are

.. math::

    K_{jk} = \int_0^\infty\operatorname{Cov}\big(f_j(y_0), f_k(y_t)\big)\,dt = -\pi\cdot\big(\tilde f_j\odot Q^\#\tilde f_k\big),
    \qquad m_j(i) = \big(Q^\#\tilde f_j\big)_i,

with :math:`Q^\#` the group inverse of the generator, and the price to first order is

.. math::

    u_i \approx e^{T\bar L}u_0 + \int_0^Te^{(T - s)\bar L}\Big(\sum_{jk}K_{jk}A_jA_k\Big)e^{s\bar L}u_0\,ds
      - \sum_jm_j(i)\,A_j\,e^{T\bar L}u_0 .

The three terms are the averaged model, the Green–Kubo correction applied through Duhamel's formula, and the
memory of the starting regime. For a symmetric two-state chain :math:`K_{jk} = \tilde f_j\tilde f_k/(2\lambda)` and
:math:`m_j = \mp\tilde f_j/(2\lambda)`. The library evaluates the three terms on a grid in :math:`(\log S, v)` and
estimates the neglected second-order term as the square of the relative correction.

Python
------

.. code-block:: python

    S0, r, q, v0, kappa, theta, xi, rho, K, lam, T = 100.0, 0.03, 0.0, 0.04, 2.0, 0.04, [0.8, 0.2], -0.6, 100.0, 6.0, 1.0

    model = rl.SwitchingHestonVolOfVol(rl.RegimeChain.twoState(lam, lam), S0, r, q, v0, kappa, theta, xi, rho)
    option = rl.VanillaOption(("call", K), maturity=T)
    engine = rl.FirstOrderFDEngine(model, regime=0, n=(151, 61))
    option.setPricingEngine(engine); first = option.NPV()
    option.setPricingEngine(rl.SwitchingFDReferee(model, regime=0, n=(151, 61))); referee = option.NPV()
    print(f"averaged model     {engine.averaged:.5f}")
    print(f"Green-Kubo term    {engine.correction:+.5f}")
    print(f"memory term        {engine.memory:+.5f}")
    print(f"first order        {first:.5f}")
    print(f"switching price    {referee:.5f}   (coupled equations on the same grid)")


.. code-block:: text

    averaged model     8.80699
    Green-Kubo term    +0.01893
    memory term        -0.06718
    first order        8.75875
    switching price    8.75772   (coupled equations on the same grid)
