Vasicek
=======

``rl.SwitchingVasicek(chain, r0, a, b, sigma)`` is QuantLib's ``Vasicek(r0, a, b, sigma)`` with the mean level
``b`` and the volatility ``sigma`` allowed to switch.

The model
---------

.. math::

    dr_t = a\,(b_{y_t} - r_t)\,dt + \sigma_{y_t}\,dW_t,

with :math:`y_t` the hidden regime. The same equation serves as a default intensity, in which case the bond below
is a survival probability.

Reduction
---------

The bond :math:`u_i(t, r) = \mathbb{E}[e^{-\int_0^t r_s ds} \mid r_0 = r,\ y_0 = i]` solves

.. math::

    \partial_t u_i = a(b_i - r)\,\partial_r u_i + \tfrac12\sigma_i^2\,\partial_{rr}u_i - r\,u_i + \sum_j Q_{ij}u_j,
    \qquad u_i(0, r) = 1.

Try :math:`u_i = e^{-B(t)r}a_i(t)`. The terms proportional to :math:`r` cancel when :math:`B' = 1 - aB`, so
:math:`B(t) = (1 - e^{-at})/a`. The reversion speed does not switch, so one :math:`B` serves every regime, and

.. math::

    P_i(0, T) = e^{-B(T)\,r_0}\,a_i(T), \qquad g_i(t) = -a\,b_i\,B(t) + \tfrac12\sigma_i^2\,B(t)^2 .

Closed form
-----------

With :math:`\bar b, \tilde b = \tfrac12(b_1 \pm b_2)`, :math:`\bar s, \tilde s = \tfrac12(\sigma_1^2 \pm \sigma_2^2)`
and :math:`I_n = \int_0^T B^n`,

.. math::

    \int_0^T\bar g = -a\bar b\,I_1 + \tfrac12\bar s\,I_2, \qquad
    \int_0^T\tilde g^{\,2} = a^2\tilde b^2 I_2 - a\tilde b\,\tilde s\,I_3 + \tfrac14\tilde s^2 I_4,

.. math::

    \tilde g(T) = -a\tilde b\,B + \tfrac12\tilde s\,B^2, \qquad \tilde g(0) = 0, \qquad
    \tilde g'(T) = \big(-a\tilde b + \tilde s\,B\big)\,e^{-aT}.

Python
------

.. code-block:: python

    r0, a, b, sigma, lam, T = 0.03, 0.5, [0.06, 0.02], [0.015, 0.008], 8.0, 5.0

    E, B, I1, I2, I3, I4 = powers_of_B(a, T)
    bbar, bt = half(b)
    sbar, st = half([s * s for s in sigma])
    closed = np.exp(-B * r0) * second_order(
        int_gbar=-a * bbar * I1 + sbar * I2 / 2,
        int_gt2=a * a * bt * bt * I2 - a * bt * st * I3 + st * st * I4 / 4,
        gt_T=-a * bt * B + st * B * B / 2,
        gt_0=0.0,
        dgt_T=(-a * bt + st * B) * E,
        lam=lam)

    model = rl.SwitchingVasicek(rl.RegimeChain.twoState(lam, lam), r0, a, b, sigma)
    print(f"closed form, second order  {closed:.10f}")
    print(f"library, numerical         {library_bond(model, T):.10f}")


.. code-block:: text

    closed form, second order  0.8335592514
    library, numerical         0.8335593214

The same formula is available as a sympy expression, with its derivatives:

.. code-block:: python

    from regimelib.symbolic import VasicekTwoStateBond
    f = VasicekTwoStateBond(regime=0)
    values = dict(r0=r0, kappa=a, theta1=b[0], theta2=b[1], sigma1=sigma[0], sigma2=sigma[1], lam=lam, T=T)
    print(f"symbolic price             {f.evaluate(f.price, **values):.10f}")
    print(f"dP/dr0                     {f.evaluate(f.delta(), **values):.10f}")
    print(f"dP/dlambda                 {f.evaluate(f.greek('lam'), **values):.3e}")


.. code-block:: text

    symbolic price             0.8335592514
    dP/dr0                     -1.5302730828
    dP/dlambda                 1.123e-04

Instruments
-----------

Bonds, coupon bonds, options on bonds, swaptions, caps and Bermudan swaptions; as an intensity, survival
probabilities and credit default swaps. Options on bonds use the same forcing with a terminal exponent :math:`c`,
which changes :math:`B` to :math:`B_c(t) = c\,e^{-at} + (1 - e^{-at})/a`, and a terminal vector that selects the
regime at expiry (:doc:`../reference/instruments`).
