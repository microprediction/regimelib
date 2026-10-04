Black–Scholes
=============

``rl.SwitchingBlackScholesProcess(chain, S0, r, q, sigma)`` is QuantLib's ``BlackScholesMertonProcess`` with the
volatility ``sigma`` allowed to switch.

The model
---------

.. math::

    \frac{dS_t}{S_t} = (r - q)\,dt + \sigma_{y_t}\,dW_t .

Write :math:`S_T = F\,e^{X_T}` with :math:`F = S_0e^{(r - q)T}` the forward, so that :math:`e^{X}` is a martingale.

Reduction
---------

Over an interval :math:`dt` spent in regime :math:`i`, :math:`X` moves by a normal amount with mean
:math:`-\tfrac12\sigma_i^2\,dt` and variance :math:`\sigma_i^2\,dt`, which multiplies :math:`e^{iuX}` on average by
:math:`e^{g_i\,dt}`. Meanwhile the regime jumps at the rates in :math:`Q`. There is no state to factor out:

.. math::

    \phi_i(u) = \mathbb{E}\big[e^{iuX_T} \mid y_0 = i\big] = a_i(T), \qquad g_i = -\tfrac12\sigma_i^2\,(u^2 + iu).

Closed form
-----------

The forcing is constant, so the two-regime characteristic function is exact:

.. math::

    \phi_{1,2}(u) = e^{(\bar g - \lambda)T}\Big(\cosh sT + \frac{\lambda \pm \tilde g}{s}\,\sinh sT\Big), \qquad
    s = \sqrt{\lambda^2 + \tilde g^2},

.. math::

    \bar g = -\tfrac12\bar s\,(u^2 + iu), \qquad \tilde g = -\tfrac12\tilde s\,(u^2 + iu), \qquad
    \bar s, \tilde s = \tfrac12(\sigma_1^2 \pm \sigma_2^2).

Expanding in :math:`\varepsilon = 1/\lambda` gives Black–Scholes at the averaged variance, times the Green–Kubo
factor :math:`e^{\varepsilon\tilde g^2T/2}` and the memory of the starting regime. A call is Lewis's formula,

.. math::

    C = e^{-rT}\Big[F - \frac{\sqrt{FK}}{\pi}\int_0^\infty\operatorname{Re}\big(e^{iu\log(F/K)}\,\phi(u - \tfrac i2)\big)\frac{du}{u^2 + 1/4}\Big].

Python
------

.. code-block:: python

    S0, r, q, sigma, lam, T = 100.0, 0.03, 0.0, [0.30, 0.15], 25.0, 1.0

    def phi(u):
        g = [-0.5 * s * s * (u * u + 1j * u) for s in sigma]
        return exact_two_state(g[0], g[1], lam, T)

    model = rl.SwitchingBlackScholesProcess(rl.RegimeChain.twoState(lam, lam), S0, r, q, sigma)
    for K in (90.0, 110.0):
        print(f"K = {K:5.0f}   closed form, exact  {lewis_call(phi, S0, K, r, q, T):.6f}"
              f"   library  {library_call(model, K, T):.6f}")


.. code-block:: text

    K =    90   closed form, exact  16.600838   library  16.600838
    K =   110   closed form, exact  6.790125   library  6.790125

``regimelib.symbolic.TwoStateConstantForcing`` holds the same characteristic function as a sympy expression, for
unequal switching rates as well (:doc:`../reference/symbolic`).

Instruments
-----------

Vanilla and digital options by the characteristic function; the continuous geometric Asian option, whose forcing in
time to maturity :math:`\tau` is :math:`iu(r - q - \tfrac12\sigma_i^2)\tau/T - \tfrac12u^2\sigma_i^2\tau^2/T^2`;
barrier and American options on the coupled grid.
