Heston
======

``rl.SwitchingHestonModel(chain, S0, r, q, v0, kappa, theta, sigma, rho)`` is QuantLib's ``HestonModel`` with the
long-run variance ``theta`` allowed to switch.

The model
---------

With :math:`S_T = F e^{X_T}`,

.. math::

    dX_t = -\tfrac12v_t\,dt + \sqrt{v_t}\,dW_t, \qquad dv_t = \kappa\,(\theta_{y_t} - v_t)\,dt + \xi\sqrt{v_t}\,dZ_t,
    \qquad d\langle W, Z\rangle = \rho\,dt .

Reduction
---------

Try :math:`\phi_i = e^{D(t)v}a_i(t)` in the equation for the characteristic function. The terms in :math:`v`
cancel when

.. math::

    D' = -\tfrac12(u^2 + iu) - b\,D + \tfrac12\xi^2D^2, \qquad b = \kappa - \rho\xi iu, \qquad D(0) = 0,

which involves :math:`\kappa`, :math:`\xi` and :math:`\rho` and not :math:`\theta`, so one :math:`D` serves both
regimes:

.. math::

    D(t) = \frac{b - d}{\xi^2}\,\frac{1 - e^{-dt}}{1 - \gamma e^{-dt}}, \qquad d = \sqrt{b^2 + \xi^2(u^2 + iu)},
    \qquad \gamma = \frac{b - d}{b + d},

.. math::

    \phi_i(u) = e^{D(T)\,v_0}\,a_i(T), \qquad g_i(t) = \kappa\,\theta_i\,D(t).

Closed form
-----------

The forcing is a multiple of :math:`D`, so two integrals are needed. With
:math:`L = \log\big((1 - \gamma e^{-dT})/(1 - \gamma)\big)`,

.. math::

    I_1 = \int_0^TD = \frac{(b - d)T - 2L}{\xi^2}, \qquad
    I_2 = \int_0^TD^2 = \Big(\frac{b - d}{\xi^2}\Big)^2\Big[T + \frac{\alpha L}{\gamma d}
      - \frac{\beta}{\gamma d}\Big(\frac{1}{1 - \gamma e^{-dT}} - \frac{1}{1 - \gamma}\Big)\Big],

with :math:`\alpha = (\gamma^2 - 1)/\gamma` and :math:`\beta = 2\gamma - 2 - \alpha`. Then, with
:math:`\bar\theta, \tilde\theta = \tfrac12(\theta_1 \pm \theta_2)`,

.. math::

    \int_0^T\bar g = \kappa\bar\theta\,I_1, \qquad \int_0^T\tilde g^{\,2} = \kappa^2\tilde\theta^2I_2, \qquad
    \tilde g(T) = \kappa\tilde\theta\,D, \qquad \tilde g(0) = 0, \qquad \tilde g'(T) = \kappa\tilde\theta\,D'.

Python
------

.. code-block:: python

    S0, r, q, v0, kappa, theta, xi, rho, lam, T = 100.0, 0.0, 0.0, 0.04, 2.0, [0.09, 0.02], 0.4, -0.6, 10.0, 1.0

    def heston_D(u):
        """D(T), D'(T) and the integrals of D and D^2 over [0, T]."""
        b = kappa - rho * xi * 1j * u
        d = np.sqrt(b * b + xi * xi * (u * u + 1j * u))
        gam, E = (b - d) / (b + d), np.exp(-d * T)
        D = (b - d) / xi ** 2 * (1 - E) / (1 - gam * E)
        dD = -0.5 * (u * u + 1j * u) - b * D + 0.5 * xi * xi * D * D
        L = np.log((1 - gam * E) / (1 - gam))
        al = (gam * gam - 1) / gam
        be = 2 * gam - 2 - al
        I1 = ((b - d) * T - 2 * L) / xi ** 2
        I2 = ((b - d) / xi ** 2) ** 2 * (T + al * L / (gam * d) - be / (gam * d) * (1 / (1 - gam * E) - 1 / (1 - gam)))
        return D, dD, I1, I2

    thbar, tht = half(theta)

    def phi(u):
        D, dD, I1, I2 = heston_D(u)
        return np.exp(D * v0) * second_order(kappa * thbar * I1, kappa ** 2 * tht ** 2 * I2,
                                             kappa * tht * D, 0.0, kappa * tht * dD, lam)

    model = rl.SwitchingHestonModel(rl.RegimeChain.twoState(lam, lam), S0, r, q, v0, kappa, theta, xi, rho)
    for K in (80.0, 100.0, 120.0):
        print(f"K = {K:5.0f}   closed form, second order  {lewis_call(phi, S0, K, r, q, T, U=40.0):.5f}"
              f"   library  {library_call(model, K, T):.5f}")


.. code-block:: text

    K =    80   closed form, second order  22.13388   library  22.13369
    K =   100   closed form, second order  8.40289   library  8.40279
    K =   120   closed form, second order  1.83003   library  1.82993

The difference is the third-order term. ``rl.FastSwitchingEngine(model, order=4)`` carries the same expansion
further.
