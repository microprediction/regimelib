Options on bonds, swaptions and caps
====================================

``rl.ZeroCouponBondOption``, ``rl.CouponBondOption``, ``rl.Swaption`` and ``rl.CapFloor`` under
``SwitchingVasicek``, ``SwitchingHullWhite`` or ``SwitchingG2``.

The problem
-----------

Under the :doc:`vasicek` model, a call expiring at :math:`T` with strike :math:`K` on the zero-coupon bond maturing
at :math:`S` is

.. math::

    C = \mathbb{E}\Big[e^{-\int_0^Tr_s\,ds}\,\big(P(T, S) - K\big)^+\Big].

The bond at expiry depends on the regime then: on :math:`\{y_T = j\}`,
:math:`P(T, S) = A_j\,e^{-b\,r_T}` with :math:`b = B(S - T)` and :math:`A_j = a_j(S - T)` from the bond's reduced
system. So the option pays when :math:`r_T < r_j^* = \log(A_j/K)/b`: there is one exercise boundary per regime.

Reduction
---------

The price is a sum over the regime at expiry of terms of the form

.. math::

    \mathbb{E}\big[e^{-\int_0^Tr - c\,r_T}\,\mathbf 1\{r_T < r_j^*\}\,\mathbf 1\{y_T = j\}\big], \qquad c = b \text{ and } c = 0 .

Without the indicator on :math:`r_T` this is a bond-like quantity with two changes. The payoff contains
:math:`e^{-c\,r}`, so the coefficient of the rate starts at :math:`c` rather than zero,

.. math::

    B_c(t) = c\,e^{-at} + \frac{1 - e^{-at}}{a},

and it is paid only in regime :math:`j`, so the terminal vector is the unit vector :math:`e_j` rather than
:math:`\mathbf 1`:

.. math::

    \Psi_j(c) = \mathbb{E}\big[e^{-\int_0^Tr - c\,r_T}\,\mathbf 1\{y_T = j\} \mid y_0 = i\big] = e^{-B_c(T)\,r_0}\,a_i(T),
    \qquad a' = (Q + \operatorname{diag} g)\,a, \quad a(0) = e_j,

.. math::

    g_i(t) = -a\,b_i\,B_c + \tfrac12\sigma_i^2\,B_c^2 .

The indicator on :math:`r_T` is restored by Gil–Pelaez inversion in the complex exponent :math:`c - iu`:

.. math::

    \mathbb{E}\big[e^{-\int_0^Tr - c\,r_T}\,\mathbf 1\{r_T < r_j^*\}\,\mathbf 1\{y_T = j\}\big]
      = \tfrac12\Psi_j(c) - \frac1\pi\int_0^\infty\operatorname{Im}\big(e^{-iu\,r_j^*}\,\Psi_j(c - iu)\big)\frac{du}{u}.

The regime at expiry costs one solve per regime and nothing else.

Closed form, first order
------------------------

With the terminal vector :math:`e_j = \tfrac12\mathbf 1 + \tfrac12s_j(1, -1)^\top`, :math:`s_1 = 1`, :math:`s_2 = -1`,
the stationary half is carried as for a bond and the other half relaxes, leaving a first-order imprint through
:math:`\tilde g(0)`, which is not zero here because :math:`B_c(0) = c`:

.. math::

    \Psi_j(c) = \tfrac12\,e^{-B_c(T)\,r_0}\exp\Big(\int_0^T\bar g + \frac\varepsilon2\int_0^T\tilde g^{\,2}\Big)
      \Big(1 \pm \frac\varepsilon2\,\tilde g(T) + s_j\,\frac\varepsilon2\,\tilde g(0)\Big) + O(\varepsilon^2).

The term in :math:`\tilde g(T)` is the memory of the regime at the start; the term in :math:`\tilde g(0)` is its
mirror image, the memory of the regime at expiry. Because :math:`B_c = c\,e^{-at} + B`, the first-order factor is a
polynomial in :math:`c`, :math:`\Pi_j(c) = \sum_{k=0}^4\pi_{jk}c^k`, and a factor :math:`c^k` on the transform is the
:math:`k`-th derivative of the Gaussian density of :math:`r_T` under the averaged model's :math:`T`-forward measure,
:math:`r_T \sim N(\mu, v)`. The price is then

.. math::

    C = \bar P(0, T)\sum_{j=1}^2\tfrac12\Big(R_0(A_j) + \varepsilon\sum_{k=0}^4(-1)^k\pi_{jk}\,R_k(A_j)\Big) + O(\varepsilon^2),

.. math::

    R_0(A) = A\,e^{-b\mu + \frac12b^2v}\,\Phi(z^* + b\sqrt v) - K\,\Phi(z^*), \qquad z^* = \frac{\log(A/K)/b - \mu}{\sqrt v},

.. math::

    R_k(A) = v^{-k/2}\Big(A\,e^{-b\mu + \frac12b^2v}\sum_{i=0}^k\binom ki(-b\sqrt v)^{k-i}H_i(z^* + b\sqrt v) - K\,H_k(z^*)\Big),
    \qquad H_0 = \Phi, \quad H_i(w) = -\mathrm{He}_{i-1}(w)\,\varphi(w).

:math:`R_0` is Jamshidian's formula, so at :math:`\varepsilon = 0` this is the averaged model's option price. The
ingredients are

.. math::

    \mu = e^{-aT}r_0 + a\bar b\,B(T) - \bar s\,M_{1,1}, \qquad v = \bar s\,M_{2,0}, \qquad
    M_{k,m} = \int_0^Te^{-kat}B(t)^m\,dt,

.. math::

    \Pi_j(c) = \tfrac12\int_0^T\tilde g_c^{\,2} \pm \tfrac12\,\tilde g_c(T) + \tfrac12\,s_j\,\tilde g_c(0), \qquad
    \tilde g_c = -a\tilde b\,B_c + \tfrac12\tilde s\,B_c^2 .

Python
------

The closed form is ``regimelib._engine.bond_option_explicit.call``; the library's engines solve the Gil–Pelaez
integrals without expanding.

.. code-block:: python

    from regimelib._engine.bond_option_explicit import call as closed_form_call
    r0, a, b, sigma, T, S = 0.04, 0.5, [0.05, 0.03], [0.015, 0.010], 1.0, 4.0

    print(" lam     K   Jamshidian, averaged   closed form, first order   library, numerical")
    for lam in (10.0, 20.0, 40.0):
        model = rl.SwitchingVasicek(rl.RegimeChain.twoState(lam, lam), r0, a, b, sigma)
        for K in (0.86, 0.90):
            option = rl.ZeroCouponBondOption("call", K, T, S)
            option.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0))
            averaged = closed_form_call(a, b, sigma, r0, T, S, K, lam, start=0, order=0)
            first = closed_form_call(a, b, sigma, r0, T, S, K, lam, start=0, order=1)
            print(f"{lam:4.0f}  {K:.2f}   {averaged:.8f}             {first:.8f}                 {option.NPV():.8f}")


.. code-block:: text

     lam     K   Jamshidian, averaged   closed form, first order   library, numerical
      10  0.86   0.02638714             0.02622112                 0.02621545
      10  0.90   0.00135934             0.00134597                 0.00134605
      20  0.86   0.02638714             0.02630380                 0.02630239
      20  0.90   0.00135934             0.00135215                 0.00135217
      40  0.86   0.02638714             0.02634539                 0.02634504
      40  0.90   0.00135934             0.00135562                 0.00135562

Each doubling of the switching rate divides the first-order error by about four.

Coupon bonds, swaptions and caps
--------------------------------

A coupon bond :math:`\sum_kc_kA_{kj}e^{-b_kr}` is decreasing in :math:`r` in every regime, so Jamshidian's
decomposition holds regime by regime. In regime :math:`j` there is one :math:`r_j^*` at which the coupon bond equals
the strike, and

.. math::

    \Big(\sum_kc_kA_{kj}e^{-b_kr_T} - K\Big)^+ = \sum_kc_k\big(A_{kj}e^{-b_kr_T} - K_{kj}\big)^+ \quad\text{on } \{y_T = j\},
    \qquad K_{kj} = A_{kj}e^{-b_kr_j^*}.

The strikes differ by regime, which is the only change from the model without regimes. A receiver swaption is the
call on the coupon bond at par and a payer swaption the put. A caplet on :math:`[t_{k-1}, t_k]` with strike
:math:`K` is :math:`1 + \tau_kK` puts on the bond maturing at :math:`t_k`, struck at :math:`1/(1 + \tau_kK)` and
expiring at :math:`t_{k-1}`. Under :doc:`hull_white` the deterministic shift scales each cash flow; under
:doc:`g2` the single Gaussian variable :math:`B_ax_T + B_bz_T` takes the place of :math:`r_T`.

.. code-block:: python

    lam = 10.0
    model = rl.SwitchingVasicek(rl.RegimeChain.twoState(lam, lam), 0.03, 0.5, [0.06, 0.02], [0.015, 0.008])
    swaption = rl.Swaption("payer", 2.0, [3.0, 4.0, 5.0, 6.0, 7.0], 0.035, notional=100.0)
    swaption.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0))
    reference = swaption.NPV()
    print(f"2x5 payer swaption, numerical   {reference:.6f}")
    for order in (0, 1, 2, 4):
        swaption.setPricingEngine(rl.FastSwitchingEngine(model, order=order, regime=0))
        print(f"  expansion, order {order}            {swaption.NPV():.6f}   error {abs(swaption.NPV() - reference):.1e}")

    cap = rl.CapFloor("cap", [0.5 * k for k in range(1, 11)], 0.04, notional=100.0)
    cap.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0))
    print(f"semiannual cap at 4%, 0.5y to 5y  {cap.NPV():.6f}")


.. code-block:: text

    2x5 payer swaption, numerical   1.924124
      expansion, order 0            1.895914   error 2.8e-02
      expansion, order 1            1.923805   error 3.2e-04
      expansion, order 2            1.924123   error 1.9e-06
      expansion, order 4            1.924124   error 3.7e-08
    semiannual cap at 4%, 0.5y to 5y  1.171073

Bermudan swaptions depend on the path of the rate and are priced on the coupled equations on a grid
(:doc:`instruments`).
