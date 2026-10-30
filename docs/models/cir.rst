Cox–Ingersoll–Ross
==================

``rl.SwitchingCoxIngersollRoss(chain, r0, theta, k, sigma)`` is QuantLib's ``CoxIngersollRoss(r0, theta, k, sigma)``
with the mean level ``theta`` allowed to switch.

The model
---------

.. math::

    dr_t = k\,(\theta_{y_t} - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t .

Reduction
---------

With :math:`u_i = e^{-B(t)r}a_i(t)` the terms proportional to :math:`r` cancel when :math:`B` solves the Riccati
equation

.. math::

    B' = 1 - kB - \tfrac12\sigma^2B^2, \qquad
    B(t) = \frac{2(e^{ht} - 1)}{(h + k)(e^{ht} - 1) + 2h}, \quad h = \sqrt{k^2 + 2\sigma^2}.

The equation contains :math:`k` and :math:`\sigma` and not :math:`\theta`, which is why only the level may switch
exactly. What remains is

.. math::

    P_i(0, T) = e^{-B(T)\,r_0}\,a_i(T), \qquad g_i(t) = -k\,\theta_i\,B(t).

Closed form
-----------

The forcing is a multiple of :math:`B`, so only :math:`I_1 = \int_0^TB` and :math:`I_2 = \int_0^TB^2` are needed.
The first is the logarithm in the classical CIR bond, and the Riccati equation gives the second with no further
integration:

.. math::

    I_1 = -\frac{2}{\sigma^2}\log\frac{2h\,e^{(h + k)T/2}}{(h + k)(e^{hT} - 1) + 2h}, \qquad
    I_2 = \frac{2}{\sigma^2}\big(T - k\,I_1 - B(T)\big).

With :math:`\bar\theta, \tilde\theta = \tfrac12(\theta_1 \pm \theta_2)`,

.. math::

    \int_0^T\bar g = -k\bar\theta\,I_1, \qquad \int_0^T\tilde g^{\,2} = k^2\tilde\theta^2 I_2, \qquad
    \tilde g(T) = -k\tilde\theta\,B, \qquad \tilde g(0) = 0, \qquad
    \tilde g'(T) = -k\tilde\theta\,\big(1 - kB - \tfrac12\sigma^2B^2\big).

Python
------

.. code-block:: python

    r0, k, theta, sigma, lam, T = 0.03, 0.5, [0.06, 0.02], 0.10, 8.0, 5.0

    h = np.sqrt(k * k + 2 * sigma ** 2)
    eh = np.exp(h * T)
    B = 2 * (eh - 1) / ((h + k) * (eh - 1) + 2 * h)
    I1 = -2 / sigma ** 2 * np.log(2 * h * np.exp((h + k) * T / 2) / ((h + k) * (eh - 1) + 2 * h))
    I2 = 2 / sigma ** 2 * (T - k * I1 - B)
    thbar, tht = half(theta)
    closed = np.exp(-B * r0) * second_order(
        int_gbar=-k * thbar * I1,
        int_gt2=k * k * tht * tht * I2,
        gt_T=-k * tht * B,
        gt_0=0.0,
        dgt_T=-k * tht * (1 - k * B - sigma ** 2 * B * B / 2),
        lam=lam)

    model = rl.SwitchingCoxIngersollRoss(rl.RegimeChain.twoState(lam, lam), r0, theta, k, sigma)
    print(f"closed form, second order  {closed:.10f}")
    print(f"library, numerical         {library_bond(model, T):.10f}")


.. code-block:: text

    closed form, second order  0.8343376435
    library, numerical         0.8343377124

``regimelib.symbolic.CIRBondFirstOrder`` gives the first-order formula for any finite chain as a sympy expression
(:doc:`../reference/symbolic`). A switching volatility or reversion speed enters the Riccati equation and falls
outside the exact reduction.
