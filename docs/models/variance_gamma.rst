Variance gamma
==============

``rl.SwitchingVarianceGammaProcess(chain, S0, r, q, sigma, nu, theta)`` is QuantLib's ``VarianceGammaProcess`` with
all three parameters allowed to switch.

The model
---------

While :math:`y_t = i` the log price is a Brownian motion with drift :math:`\theta_i` and volatility
:math:`\sigma_i` run on a gamma clock with variance rate :math:`\nu_i`:

.. math::

    d\log S_t = (r - q + \omega_i)\,dt + dX^{(i)}_t, \qquad X^{(i)}_t = \theta_i\,G^{(i)}_t + \sigma_i\,W_{G^{(i)}_t},
    \qquad \omega_i = \frac{1}{\nu_i}\log\big(1 - \theta_i\nu_i - \tfrac12\sigma_i^2\nu_i\big).

Reduction
---------

Given the clock increment :math:`dG`, the increment of :math:`X` is normal with mean :math:`\theta_i\,dG` and
variance :math:`\sigma_i^2\,dG`, and the gamma increment has
:math:`\mathbb{E}[e^{-z\,dG}] = (1 + \nu_iz)^{-dt/\nu_i}`. Putting :math:`z = -iu\theta_i + \tfrac12\sigma_i^2u^2`
gives the Lévy exponent, and for the martingale log return

.. math::

    \phi_i(u) = a_i(T), \qquad g_i = \psi_i(u) + iu\,\omega_i, \qquad
    \psi_i(u) = -\frac{1}{\nu_i}\log\big(1 - iu\,\theta_i\nu_i + \tfrac12\sigma_i^2\nu_i\,u^2\big).

All three parameters sit inside :math:`g_i` and there is no state, hence no Riccati equation for them to disturb.
That is why all three may switch.

Closed form
-----------

The forcing is constant, so the two-regime characteristic function is exact, as for :doc:`black_scholes`, with
:math:`\bar g, \tilde g = \tfrac12(g_1 \pm g_2)`. The averaged model, :math:`e^{\bar gT}`, is the sum of two
independent variance gamma processes each run at half speed; it is not itself variance gamma unless the regimes
agree. The characteristic function decays like a power of :math:`u`, so the Fourier integral is carried further
than for a diffusion.

Python
------

.. code-block:: python

    S0, r, q, sigma, nu, theta, lam, T = 100.0, 0.03, 0.0, [0.25, 0.12], [0.5, 0.2], [-0.25, -0.10], 25.0, 1.0

    def phi(u):
        g = []
        for s, n, th in zip(sigma, nu, theta):
            omega = np.log(1 - th * n - 0.5 * s * s * n) / n
            g.append(-np.log(1 - 1j * u * th * n + 0.5 * s * s * n * u * u) / n + 1j * u * omega)
        return exact_two_state(g[0], g[1], lam, T)

    model = rl.SwitchingVarianceGammaProcess(rl.RegimeChain.twoState(lam, lam), S0, r, q, sigma, nu, theta)
    for K in (90.0, 110.0):
        print(f"K = {K:5.0f}   closed form, exact  {lewis_call(phi, S0, K, r, q, T, U=600.0, n=6000):.6f}"
              f"   library  {library_call(model, K, T):.6f}")


.. code-block:: text

    K =    90   closed form, exact  16.402558   library  16.402558
    K =   110   closed form, exact  5.107950   library  5.107950
