Vasicek with jumps
==================

``rl.SwitchingVasicekJumps(chain, r0, a, b, sigma, jumpIntensity, jumpMean)``: the mean level, the volatility and
the jump intensity may switch. QuantLib has no jump short-rate model; the frozen limit is checked against the affine
closed form.

The model
---------

.. math::

    dr_t = a\,(b_{y_t} - r_t)\,dt + \sigma_{y_t}\,dW_t + dJ_t,

where, while :math:`y_t = i`, :math:`J` jumps at rate :math:`\ell_i` by exponentially distributed amounts with mean
:math:`m`.

Reduction
---------

The pricing equation is the :doc:`vasicek` one with the extra term
:math:`\ell_i\,\mathbb{E}[u_i(t, r + J) - u_i(t, r)]`. With :math:`u_i = e^{-B(t)r}a_i(t)` a jump of size :math:`J`
multiplies :math:`e^{-Br}` by :math:`e^{-BJ}`, and for exponential jumps
:math:`\mathbb{E}[e^{-BJ}] = 1/(1 + mB)`. So the jumps change only the forcing:

.. math::

    P_i(0, T) = e^{-B(T)\,r_0}\,a_i(T), \qquad
    g_i(t) = -a\,b_i\,B + \tfrac12\sigma_i^2\,B^2 + \ell_i\Big(\frac{1}{1 + mB} - 1\Big).

Closed form
-----------

Two integrals beyond the :math:`I_n` of the Vasicek page are needed,

.. math::

    J_1 = \int_0^T\frac{dt}{1 + mB} = \frac{a}{a + m}\Big(T + \frac1a\log\Big(1 + \frac ma(1 - E)\Big)\Big), \qquad
    J_2 = \int_0^T\frac{dt}{(1 + mB)^2} = \frac{a}{a + m}\Big(J_1 - \frac1a\Big(\frac{1}{1 + \frac ma(1 - E)} - 1\Big)\Big),

with :math:`E = e^{-aT}`. Then, with :math:`\bar\ell, \tilde\ell = \tfrac12(\ell_1 \pm \ell_2)`,

.. math::

    \int_0^T\bar g = -a\bar b\,I_1 + \tfrac12\bar s\,I_2 + \bar\ell\,(J_1 - T),

.. math::

    \int_0^T\tilde g^{\,2} = a^2\tilde b^2 I_2 - a\tilde b\,\tilde s\,I_3 + \tfrac14\tilde s^2 I_4
      + \tilde\ell^{\,2}(J_2 - 2J_1 + T)
      - 2a\tilde b\,\tilde\ell\Big(\frac{T - J_1}{m} - I_1\Big)
      + \tilde s\,\tilde\ell\Big(\frac1m\Big(I_1 - \frac{T - J_1}{m}\Big) - I_2\Big),

.. math::

    \tilde g(T) = -a\tilde b\,B + \tfrac12\tilde s\,B^2 + \tilde\ell\Big(\frac{1}{1 + mB} - 1\Big), \qquad
    \tilde g(0) = 0, \qquad
    \tilde g'(T) = \Big(-a\tilde b + \tilde s\,B - \frac{\tilde\ell\,m}{(1 + mB)^2}\Big)E.

Python
------

.. code-block:: python

    r0, a, b, sigma, ell, m, lam, T = 0.0, 2.0, [0.05, 0.02], [0.02, 0.01], [3.0, 0.2], 0.03, 10.0, 3.0

    E, B, I1, I2, I3, I4 = powers_of_B(a, T)
    J1 = a / (a + m) * (T + np.log(1 + m / a * (1 - E)) / a)
    J2 = a / (a + m) * (J1 - (1 / (1 + m / a * (1 - E)) - 1) / a)
    bbar, bt = half(b)
    sbar, st = half([s * s for s in sigma])
    lbar, lt = half(ell)
    closed = np.exp(-B * r0) * second_order(
        int_gbar=-a * bbar * I1 + sbar * I2 / 2 + lbar * (J1 - T),
        int_gt2=(a * a * bt * bt * I2 - a * bt * st * I3 + st * st * I4 / 4 + lt * lt * (J2 - 2 * J1 + T)
                 - 2 * a * bt * lt * ((T - J1) / m - I1) + st * lt * ((I1 - (T - J1) / m) / m - I2)),
        gt_T=-a * bt * B + st * B * B / 2 + lt * (1 / (1 + m * B) - 1),
        gt_0=0.0,
        dgt_T=(-a * bt + st * B - lt * m / (1 + m * B) ** 2) * E,
        lam=lam)

    model = rl.SwitchingVasicekJumps(rl.RegimeChain.twoState(lam, lam), r0, a, b, sigma, ell, m)
    print(f"closed form, second order  {closed:.8f}")
    print(f"library, numerical         {library_bond(model, T):.8f}")


.. code-block:: text

    closed form, second order  0.86213677
    library, numerical         0.86213670

``regimelib.symbolic.VasicekJumpsBondFirstOrder`` gives the first-order formula for any finite chain as a sympy
expression (:doc:`../reference/symbolic`).
