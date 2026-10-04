Hull–White
==========

``rl.SwitchingHullWhite(chain, termStructure, a, sigma)`` is QuantLib's ``HullWhite(termStructure, a, sigma)`` with
the volatility ``sigma`` allowed to switch.

The model
---------

.. math::

    r_t = x_t + \varphi(t), \qquad dx_t = -a\,x_t\,dt + \sigma_{y_t}\,dW_t, \qquad x_0 = 0,

with :math:`\varphi` a deterministic shift fitted to the market discount curve :math:`P^M(0, T)` and its forward
rate :math:`f^M(0, t)`.

Reduction
---------

The averaged model is Hull–White with variance :math:`\bar\sigma^2 = \tfrac12(\sigma_1^2 + \sigma_2^2)`, and it
reproduces the curve when

.. math::

    \varphi(t) = f^M(0, t) + \tfrac12\bar\sigma^2\,B(t)^2, \qquad B(t) = \frac{1 - e^{-at}}{a}.

This is the shift the library uses, so that with equal regimes the model is QuantLib's. The factor :math:`x` is a
Vasicek rate with zero mean level, so :math:`\mathbb{E}[e^{-\int_0^T x}] = a_i(T)` with

.. math::

    g_i(t) = \tfrac12\sigma_i^2\,B(t)^2, \qquad
    P_i(0, T) = e^{-\int_0^T\varphi}\,a_i(T) = P^M(0, T)\;e^{-\int_0^T\bar g}\;a_i(T).

The shift cancels the averaged part of :math:`a_i`. What is left is the effect of the switching.

Closed form
-----------

With :math:`\tilde s = \tfrac12(\sigma_1^2 - \sigma_2^2)`: :math:`\int\tilde g^{\,2} = \tfrac14\tilde s^2I_4`,
:math:`\tilde g(T) = \tfrac12\tilde sB^2`, :math:`\tilde g(0) = 0` and :math:`\tilde g'(T) = \tilde s\,B\,e^{-aT}`, so

.. math::

    \frac{P_{1,2}(0, T)}{P^M(0, T)} = \exp\Big(\frac{\varepsilon}{8}\tilde s^2I_4 - \frac{\varepsilon^2}{32}\tilde s^2B^4\Big)
      \Big(1 \pm \frac{\varepsilon}{4}\tilde s\,B^2 \mp \frac{\varepsilon^2}{4}\tilde s\,B\,e^{-aT}\Big) + O(\varepsilon^3).

At order zero every regime sees the market curve. At first order the two regimes move in opposite directions: a
start in the volatile regime means more variance in :math:`\int r`, more convexity, and a higher bond price.

Python
------

.. code-block:: python

    a, sigma, flat, lam, T = 0.3, [0.02, 0.008], 0.03, 5.0, 5.0

    E, B, I1, I2, I3, I4 = powers_of_B(a, T)
    sbar, st = half([s * s for s in sigma])
    ratio = second_order(int_gbar=0.0, int_gt2=st * st * I4 / 4, gt_T=st * B * B / 2, gt_0=0.0,
                         dgt_T=st * B * E, lam=lam)

    model = rl.SwitchingHullWhite(rl.RegimeChain.twoState(lam, lam), flat, a, sigma)
    market = np.exp(-flat * T)
    print(f"closed form, second order  {ratio:.10f}")
    print(f"library, numerical         {library_bond(model, T) / market:.10f}")


.. code-block:: text

    closed form, second order  1.0000554074
    library, numerical         1.0000553865

If the starting regime is known the fit can be made exact for it, since :math:`a_i` is known:
:math:`\varphi_i(t) = f^M(0, t) + \tfrac{d}{dt}\log a_i(t)`.

Instruments
-----------

Bonds, options on bonds, swaptions, caps and Bermudan swaptions. The deterministic shift scales each cash flow; the
rest is the :doc:`vasicek` construction on the factor :math:`x`.
