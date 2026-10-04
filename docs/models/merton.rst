Merton jump diffusion
=====================

``rl.SwitchingMerton76Process(chain, S0, r, q, sigma, jumpIntensity, logJumpMean, logJumpVol)`` is QuantLib's
``Merton76Process`` with the volatility and the jump intensity allowed to switch.

The model
---------

.. math::

    \frac{dS_t}{S_{t-}} = \big(r - q - \ell_{y_t}\bar k\big)\,dt + \sigma_{y_t}\,dW_t + \big(e^{J} - 1\big)\,dN_t,

where, while :math:`y_t = i`, :math:`N` jumps at rate :math:`\ell_i`, each jump multiplies the price by
:math:`e^J` with :math:`J` normal of mean :math:`\mu_J` and standard deviation :math:`\delta`, and
:math:`\bar k = e^{\mu_J + \delta^2/2} - 1` is the mean relative jump.

Reduction
---------

Over :math:`dt` in regime :math:`i` the diffusion contributes the :doc:`black_scholes` exponent, and with
probability :math:`\ell_i\,dt` a jump multiplies :math:`e^{iuX}` by :math:`e^{iuJ}`, whose mean is
:math:`\phi_J(u) = \exp(iu\mu_J - \tfrac12u^2\delta^2)`. So, for the martingale log return,

.. math::

    \phi_i(u) = a_i(T), \qquad g_i = -\tfrac12\sigma_i^2\,(u^2 + iu) + \ell_i\,c_J, \qquad
    c_J = \phi_J(u) - 1 - iu\bar k .

The forcing is the Lévy exponent of regime :math:`i`. A jump law that differs by regime would only replace
:math:`c_J` by a per-regime :math:`c_{J,i}`; the class switches the intensity.

Closed form
-----------

The forcing is constant, so the two-regime characteristic function is exact, as for Black–Scholes, with

.. math::

    \bar g = -\tfrac12\bar s\,(u^2 + iu) + \bar\ell\,c_J, \qquad \tilde g = -\tfrac12\tilde s\,(u^2 + iu) + \tilde\ell\,c_J,
    \qquad \bar\ell, \tilde\ell = \tfrac12(\ell_1 \pm \ell_2).

The Green–Kubo exponent :math:`\tfrac12\varepsilon T\tilde g^2` has three parts: the volatility's, the jump
count's, and the covariance of the two, present because the same regime raises both.

Python
------

.. code-block:: python

    S0, r, q, sigma, ell, muJ, delta, lam, T = 100.0, 0.03, 0.0, [0.25, 0.12], [2.0, 0.2], -0.08, 0.10, 25.0, 1.0
    kbar = np.exp(muJ + delta ** 2 / 2) - 1

    def phi(u):
        cJ = np.exp(1j * u * muJ - 0.5 * u * u * delta ** 2) - 1 - 1j * u * kbar
        g = [-0.5 * s * s * (u * u + 1j * u) + l * cJ for s, l in zip(sigma, ell)]
        return exact_two_state(g[0], g[1], lam, T)

    model = rl.SwitchingMerton76Process(rl.RegimeChain.twoState(lam, lam), S0, r, q, sigma, ell, muJ, delta)
    for K in (90.0, 110.0):
        print(f"K = {K:5.0f}   closed form, exact  {lewis_call(phi, S0, K, r, q, T):.6f}"
              f"   library  {library_call(model, K, T):.6f}")


.. code-block:: text

    K =    90   closed form, exact  16.626633   library  16.626633
    K =   110   closed form, exact  6.533332   library  6.533332
