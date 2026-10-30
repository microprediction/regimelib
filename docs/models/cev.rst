CEV
===

``rl.SwitchingCEVProcess(chain, S0, r, q, sigma, beta)`` is QuantLib's CEV process with the volatility scale
``sigma`` allowed to switch. The library prices it on a grid with ``FirstOrderFDEngine`` and ``SwitchingFDReferee``.

The model
---------

.. math::

    dS_t = (r - q)\,S_t\,dt + \sigma_{y_t}\,S_t^{\beta}\,dW_t, \qquad 0 < \beta < 1 .

Reduction
---------

The model is not affine and there is no exponential-affine solution. In the price variable the switched operator
:math:`\tfrac12S^{2\beta}\partial_{SS}` does not commute with the drift, but the commutator is a multiple of the
operator itself,

.. math::

    \big[\,S\,\partial_S,\ S^{2\beta}\partial_{SS}\,\big] = 2(\beta - 1)\,S^{2\beta}\partial_{SS},

and the forward removes it. With :math:`F_t = S_te^{(r - q)(T - t)}`,

.. math::

    dF_t = \sigma_{y_t}\,\sqrt{h(T - t)}\;F_t^{\beta}\,dW_t, \qquad h(\tau) = e^{\kappa_h\tau}, \quad \kappa_h = 2(1 - \beta)(r - q).

The regime now only scales one fixed operator, so given the regime path :math:`F_T` is the unit-scale CEV diffusion
run for the total variance :math:`V`, and the price is a mixture:

.. math::

    u_i = e^{-rT}\,\mathbb{E}\big[C(F_0, V) \mid y_0 = i\big], \qquad V = \int_0^T\sigma_{y_t}^2\,h(T - t)\,dt,

.. math::

    C(F, v) = F\,\big[1 - \chi^2(x;\ d + 2,\ y)\big] - K\,\chi^2(y;\ d,\ x), \qquad
    x = \frac{K^{2(1 - \beta)}}{(1 - \beta)^2v}, \quad y = \frac{F^{2(1 - \beta)}}{(1 - \beta)^2v}, \quad d = \frac{1}{1 - \beta},

with :math:`\chi^2(\cdot\,; d, \nu)` the noncentral chi-square distribution function. This is exact for any chain.
The law of :math:`V` comes from a reduced system of the usual kind: in time to maturity,
:math:`\mathbb{E}[e^{-zV}] = a_i(T)` with :math:`g_i(\tau) = -z\,\sigma_i^2\,h(\tau)`.

Closed form
-----------

Expanding the transform with the common formula and replacing each power of :math:`z` by a derivative of
:math:`C` in its variance argument gives, with :math:`\bar s, \tilde s = \tfrac12(\sigma_1^2 \pm \sigma_2^2)`,

.. math::

    u_{1,2} = e^{-rT}\Big[C + \frac{\varepsilon}{2}\tilde s^2H_2\,C_{vv} \pm \frac{\varepsilon}{2}\tilde s\,h_T\,C_v
      + \varepsilon^2\Big(\frac{\tilde s^4H_2^2}{8}C_{vvvv} \pm \frac{\tilde s^3H_2h_T}{4}C_{vvv}
      - \frac{\tilde s^2(h_T^2 + 1)}{8}C_{vv} \mp \frac{\tilde s\,\kappa_hh_T}{4}C_v\Big)\Big] + O(\varepsilon^3),

evaluated at :math:`F_0 = S_0e^{(r - q)T}` and :math:`\bar v = \bar s\,H_1`, where

.. math::

    H_1 = \frac{e^{\kappa_hT} - 1}{\kappa_h}, \qquad H_2 = \frac{e^{2\kappa_hT} - 1}{2\kappa_h}, \qquad h_T = e^{\kappa_hT}.

The first-order terms read as on every page: half the variance of :math:`V` times the convexity of the price in
variance, and the shift of the mean of :math:`V` towards the starting regime times the sensitivity to variance.

Python
------

.. code-block:: python

    from scipy.stats import ncx2
    S0, r, q, beta, sigma, K, lam, T = 100.0, 0.02, 0.0, 0.6, [2.5, 1.2], 100.0, 10.0, 1.0
    F = S0 * np.exp((r - q) * T)

    def C(v):
        scale = (1 - beta) ** 2 * v
        x, y, d = K ** (2 * (1 - beta)) / scale, F ** (2 * (1 - beta)) / scale, 1 / (1 - beta)
        return F * (1 - ncx2.cdf(x, d + 2, y)) - K * ncx2.cdf(y, d, x)

    kh = 2 * (1 - beta) * (r - q)
    H1, H2, hT = (np.exp(kh * T) - 1) / kh, (np.exp(2 * kh * T) - 1) / (2 * kh), np.exp(kh * T)
    sbar, st = half([s * s for s in sigma])
    vbar, dv = sbar * H1, 1e-3 * sbar * H1
    Cv = (C(vbar + dv) - C(vbar - dv)) / (2 * dv)
    Cvv = (C(vbar + dv) - 2 * C(vbar) + C(vbar - dv)) / dv ** 2
    eps, disc = 1 / lam, np.exp(-r * T)
    averaged = disc * C(vbar)
    green_kubo = disc * eps / 2 * st ** 2 * H2 * Cvv
    memory = disc * eps / 2 * st * hT * Cv

    model = rl.SwitchingCEVProcess(rl.RegimeChain.twoState(lam, lam), S0, r, q, sigma, beta)
    option = rl.VanillaOption(("call", K), maturity=T)
    engine = rl.FirstOrderFDEngine(model, regime=0, n=801)
    option.setPricingEngine(engine); first = option.NPV()
    option.setPricingEngine(rl.SwitchingFDReferee(model, regime=0, n=801)); referee = option.NPV()
    print(f"                 closed form   library grid")
    print(f"averaged         {averaged:11.6f}   {engine.averaged:11.6f}")
    print(f"Green-Kubo term  {green_kubo:11.6f}   {engine.correction:11.6f}")
    print(f"memory term      {memory:11.6f}   {engine.memory:11.6f}")
    print(f"first order      {averaged + green_kubo + memory:11.6f}   {first:11.6f}")
    print(f"switching price                {referee:11.6f}   (coupled equations on the grid)")


.. code-block:: text

                     closed form   library grid
    averaged           13.249063     13.248985
    Green-Kubo term    -0.060274     -0.060275
    memory term         0.191121      0.191122
    first order        13.379911     13.379832
    switching price                  13.383492   (coupled equations on the grid)

The grid engine applies the first-order rule in the price variable without the change of variable, and returns the
same two terms.
