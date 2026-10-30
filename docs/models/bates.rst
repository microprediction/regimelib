Bates
=====

``rl.SwitchingBatesModel(chain, S0, r, q, v0, kappa, theta, sigma, rho, jumpIntensity, logJumpMean, logJumpVol)`` is
QuantLib's ``BatesModel`` with the long-run variance and the jump intensity allowed to switch.

The model
---------

Heston's variance with Merton's jumps: with :math:`S_T = Fe^{X_T}`,

.. math::

    dX_t = \big(-\tfrac12v_t - \ell_{y_t}\bar k\big)\,dt + \sqrt{v_t}\,dW_t + J\,dN_t, \qquad
    dv_t = \kappa\,(\theta_{y_t} - v_t)\,dt + \xi\sqrt{v_t}\,dZ_t, \qquad d\langle W, Z\rangle = \rho\,dt,

where :math:`N` jumps at rate :math:`\ell_i` in regime :math:`i` by normal amounts :math:`J` of mean
:math:`\mu_J` and standard deviation :math:`\delta`, and :math:`\bar k = e^{\mu_J + \delta^2/2} - 1`.

Reduction
---------

A jump in the log price multiplies :math:`e^{iuX}` by :math:`e^{iuJ}` and leaves the variance alone, so the jump
term of the pricing equation is a multiple of :math:`\phi_i`. The Heston form :math:`\phi_i = e^{D(t)v}a_i(t)` works
with the same :math:`D` as on the :doc:`heston` page, which involves neither :math:`\theta` nor :math:`\ell`:

.. math::

    \phi_i(u) = e^{D(T)\,v_0}\,a_i(T), \qquad g_i(t) = \kappa\,\theta_i\,D(t) + \ell_i\,c_J, \qquad
    c_J = e^{iu\mu_J - u^2\delta^2/2} - 1 - iu\bar k .

The first term vanishes at :math:`t = 0` and the second does not.

Closed form
-----------

With :math:`I_1 = \int_0^TD` and :math:`I_2 = \int_0^TD^2` from the Heston page,

.. math::

    \int_0^T\bar g = \kappa\bar\theta\,I_1 + \bar\ell\,c_J\,T, \qquad
    \int_0^T\tilde g^{\,2} = \kappa^2\tilde\theta^2I_2 + 2\kappa\tilde\theta\,\tilde\ell\,c_J\,I_1 + \tilde\ell^{\,2}c_J^2\,T,

.. math::

    \tilde g(T) = \kappa\tilde\theta\,D + \tilde\ell\,c_J, \qquad \tilde g(0) = \tilde\ell\,c_J, \qquad
    \tilde g'(T) = \kappa\tilde\theta\,D'.

The middle term of :math:`\int\tilde g^{\,2}` is the covariance of the variance level and the jump count, present
because the turbulent regime raises both. It needs no new integral.

Python
------

.. code-block:: python

    S0, r, q, v0, kappa, theta, xi, rho, lam, T = 100.0, 0.0, 0.0, 0.04, 2.0, [0.09, 0.02], 0.4, -0.6, 10.0, 1.0
    ell, muJ, delta = [2.0, 0.2], -0.08, 0.10
    kbar = np.exp(muJ + delta ** 2 / 2) - 1

    def heston_D(u):
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
    lbar, lt = half(ell)

    def phi(u):
        D, dD, I1, I2 = heston_D(u)
        cJ = np.exp(1j * u * muJ - 0.5 * u * u * delta ** 2) - 1 - 1j * u * kbar
        return np.exp(D * v0) * second_order(
            int_gbar=kappa * thbar * I1 + lbar * cJ * T,
            int_gt2=kappa ** 2 * tht ** 2 * I2 + 2 * kappa * tht * lt * cJ * I1 + lt ** 2 * cJ ** 2 * T,
            gt_T=kappa * tht * D + lt * cJ,
            gt_0=lt * cJ,
            dgt_T=kappa * tht * dD,
            lam=lam)

    model = rl.SwitchingBatesModel(rl.RegimeChain.twoState(lam, lam), S0, r, q, v0, kappa, theta, xi, rho,
                                   ell, muJ, delta)
    for K in (80.0, 100.0, 120.0):
        print(f"K = {K:5.0f}   closed form, second order  {lewis_call(phi, S0, K, r, q, T, U=40.0):.5f}"
              f"   library  {library_call(model, K, T):.5f}")


.. code-block:: text

    K =    80   closed form, second order  22.93733   library  22.93715
    K =   100   closed form, second order  9.92245   library  9.92241
    K =   120   closed form, second order  2.99202   library  2.99189
