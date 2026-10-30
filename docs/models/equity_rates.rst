Equity with stochastic rates
============================

``rl.SwitchingEquityRates(equity, rates, rho=0.0)``: an equity (``SwitchingBlackScholesProcess`` or
``SwitchingHestonModel``) discounted at a short rate (``SwitchingVasicek`` or ``SwitchingHullWhite``) on one chain.
The frozen limit is QuantLib's ``AnalyticBSMHullWhiteEngine`` and ``AnalyticHestonHullWhiteEngine``.

The model
---------

For the Black–Scholes equity with Vasicek rates,

.. math::

    \frac{dS_t}{S_t} = (r_t - q)\,dt + \sigma_{y_t}\,dW^S_t, \qquad dr_t = a\,(b_{y_t} - r_t)\,dt + \eta_{y_t}\,dW^r_t,
    \qquad d\langle W^S, W^r\rangle = \rho\,dt .

Even with :math:`\rho = 0` the stock and the rate are dependent, through the regime.

Reduction
---------

The discount factor cannot be taken outside the expectation, so the object is the discounted characteristic
function

.. math::

    \Psi_i(w) = \mathbb{E}\big[e^{-\int_0^Tr}\,S_T^{\,iw} \mid r_0,\ y_0 = i\big], \qquad
    C = S_0e^{-qT} - \frac{\sqrt K}{\pi}\int_0^\infty\operatorname{Re}\big[K^{-iu}\,\Psi(u - \tfrac i2)\big]\frac{du}{u^2 + 1/4}.

Since :math:`S_T = S_0\exp(\int_0^T(r - q) + X_T)`, the rate is integrated with the complex weight
:math:`c = 1 - iw`: :math:`e^{-\int r}S_T^{\,iw} = S_0^{\,iw}e^{-iwqT}\exp(-c\int r + iwX_T)`. Trying
:math:`e^{-cB(t)r}a_i(t)` cancels the terms in :math:`r` when :math:`B' = 1 - aB`, as for a bond, and leaves

.. math::

    \Psi_i(w) = S_0^{\,iw}\,e^{-iwqT}\,e^{-cB(T)\,r_0}\,a_i(T),

.. math::

    g_i(t) = \underbrace{-a\,b_i\,c\,B + \tfrac12\eta_i^2\,c^2B^2}_{\text{rates, weight } c}
      \;\underbrace{-\,\tfrac12\sigma_i^2\,(w^2 + iw)}_{\text{equity}}
      \;\underbrace{-\,iw\,c\,\rho\,\sigma_i\eta_i\,B}_{\text{correlation}} .

Closed form
-----------

Collect the forcing by power of :math:`B`:

.. math::

    g_i = \alpha_iB + \beta_iB^2 + \gamma_i, \qquad \alpha_i = -c\,(a\,b_i + iw\,\rho\,\sigma_i\eta_i), \quad
    \beta_i = \tfrac12c^2\eta_i^2, \quad \gamma_i = -\tfrac12\sigma_i^2\,(w^2 + iw).

With bars and tildes the half-sums and half-differences, and :math:`I_n = \int_0^TB^n`,

.. math::

    \int_0^T\bar g = \bar\alpha I_1 + \bar\beta I_2 + \bar\gamma T, \qquad
    \int_0^T\tilde g^{\,2} = \tilde\alpha^2I_2 + 2\tilde\alpha\tilde\beta I_3 + \tilde\beta^2I_4
      + 2\tilde\alpha\tilde\gamma I_1 + 2\tilde\beta\tilde\gamma I_2 + \tilde\gamma^2T,

.. math::

    \tilde g(T) = \tilde\alpha B + \tilde\beta B^2 + \tilde\gamma, \qquad \tilde g(0) = \tilde\gamma, \qquad
    \tilde g'(T) = (\tilde\alpha + 2\tilde\beta B)\,e^{-aT}.

At :math:`w = 0` this is the :doc:`vasicek` bond. At :math:`w = -i` the weight is zero and
:math:`\Psi = S_0e^{-qT}` exactly.

Python
------

.. code-block:: python

    S0, q, sigma, rho, lam, T = 100.0, 0.0, [0.30, 0.15], -0.3, 25.0, 1.0
    r0, a, b, eta = 0.03, 0.5, [0.05, 0.02], [0.015, 0.008]
    E, B, I1, I2, I3, I4 = powers_of_B(a, T)

    def Psi(w):
        c = 1 - 1j * w
        (alb, alt) = half([-c * (a * b[i] + 1j * w * rho * sigma[i] * eta[i]) for i in range(2)])
        (beb, bet) = half([0.5 * c * c * eta[i] ** 2 for i in range(2)])
        (gab, gat) = half([-0.5 * sigma[i] ** 2 * (w * w + 1j * w) for i in range(2)])
        return S0 ** (1j * w) * np.exp(-1j * w * q * T - c * B * r0) * second_order(
            int_gbar=alb * I1 + beb * I2 + gab * T,
            int_gt2=(alt ** 2 * I2 + 2 * alt * bet * I3 + bet ** 2 * I4
                     + 2 * alt * gat * I1 + 2 * bet * gat * I2 + gat ** 2 * T),
            gt_T=alt * B + bet * B * B + gat,
            gt_0=gat,
            dgt_T=(alt + 2 * bet * B) * E,
            lam=lam)

    def discounted_lewis_call(K, U=40.0, n=400):
        x, wts = np.polynomial.legendre.leggauss(n)
        u, wts = (x + 1) * U / 2, wts * U / 2
        integral = np.sum(wts * (K ** (-1j * u) * Psi(u - 0.5j)).real / (u * u + 0.25))
        return S0 * np.exp(-q * T) - np.sqrt(K) / np.pi * integral

    chain = rl.RegimeChain.twoState(lam, lam)
    model = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(chain, S0, 0.0, q, sigma),
                                    rl.SwitchingVasicek(chain, r0, a, b, eta), rho=rho)
    for K in (90.0, 110.0):
        print(f"K = {K:5.0f}   closed form, second order  {discounted_lewis_call(K):.6f}"
              f"   library  {library_call(model, K, T):.6f}")
    print(f"bond       closed form, second order  {Psi(0.0).real:.6f}"
          f"   library  {library_bond(rl.SwitchingVasicek(chain, r0, a, b, eta), T):.6f}")


.. code-block:: text

    K =    90   closed form, second order  16.621954   library  16.621952
    K =   110   closed form, second order  6.774836   library  6.774838
    bond       closed form, second order  0.969317   library  0.969317

Other pairs
-----------

With Hull–White rates the level term drops, :math:`\alpha_i = -c\,iw\,\rho\,\sigma_i\eta_i`, and the prefactor
:math:`e^{-cBr_0}` becomes :math:`\exp(-c\int_0^T\varphi)`. With a Heston equity and :math:`\rho = 0` the equity
part of the forcing is :math:`\kappa\theta_iD(t; w)` and the prefactor gains :math:`e^{D(T; w)v_0}`. A correlation
between a Heston equity and the rate adds a term :math:`\rho\,\eta_i\sqrt v\,\partial_x\partial_r` that no
exponential-affine form absorbs.
