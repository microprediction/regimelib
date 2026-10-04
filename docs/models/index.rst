Models: mathematics and Python
==============================

One page per model. Each page states the model, reduces its pricing problem to a linear system, gives the closed
form for a two-regime chain, and then evaluates that closed form in a few lines of Python next to the library's
price for the same inputs.

.. rubric:: Short rates and credit

.. list-table::
   :header-rows: 1
   :widths: 16 26 12 30 16

   * - Model
     - Class
     - Switches
     - Forcing :math:`g_i`
     - Closed form
   * - :doc:`vasicek`
     - ``SwitchingVasicek``
     - :math:`b,\ \sigma`
     - :math:`-a b_i B + \tfrac12\sigma_i^2 B^2`
     - second order
   * - :doc:`cir`
     - ``SwitchingCoxIngersollRoss``
     - :math:`\theta`
     - :math:`-k\theta_i B`
     - second order
   * - :doc:`hull_white`
     - ``SwitchingHullWhite``
     - :math:`\sigma`
     - :math:`\tfrac12\sigma_i^2 B^2`
     - second order
   * - :doc:`g2`
     - ``SwitchingG2``
     - :math:`\sigma,\ \eta,\ \rho`
     - :math:`\tfrac12\sigma_i^2B_a^2 + \tfrac12\eta_i^2B_b^2 + \rho_i\sigma_i\eta_iB_aB_b`
     - second order
   * - :doc:`intensity_basket`
     - ``SwitchingIntensityBasket``
     - each name's
     - sum of the names' forcings
     - first order, for the dependence

.. rubric:: Equity diffusions

.. list-table::
   :header-rows: 1
   :widths: 16 26 12 30 16

   * - Model
     - Class
     - Switches
     - Forcing :math:`g_i`
     - Closed form
   * - :doc:`black_scholes`
     - ``SwitchingBlackScholesProcess``
     - :math:`\sigma`
     - :math:`-\tfrac12\sigma_i^2(u^2 + iu)`
     - exact
   * - :doc:`heston`
     - ``SwitchingHestonModel``
     - :math:`\theta`
     - :math:`\kappa\theta_i D(t)`
     - second order
   * - :doc:`cev`
     - ``SwitchingCEVProcess``
     - :math:`\sigma`
     - :math:`-z\sigma_i^2h(\tau)`
     - exact mixture; second order
   * - :doc:`heston_vol_of_vol`
     - ``SwitchingHestonVolOfVol``
     - :math:`\xi`
     - none: the switched operators do not commute
     - first-order rule

.. rubric:: Jump models

.. list-table::
   :header-rows: 1
   :widths: 16 26 12 30 16

   * - Model
     - Class
     - Switches
     - Forcing :math:`g_i`
     - Closed form
   * - :doc:`vasicek_jumps`
     - ``SwitchingVasicekJumps``
     - :math:`b,\ \sigma,\ \ell`
     - :math:`-a b_i B + \tfrac12\sigma_i^2 B^2 + \ell_i\big((1 + mB)^{-1} - 1\big)`
     - second order
   * - :doc:`merton`
     - ``SwitchingMerton76Process``
     - :math:`\sigma,\ \ell`
     - :math:`-\tfrac12\sigma_i^2(u^2 + iu) + \ell_i(\phi_J - 1 - iu\bar k)`
     - exact
   * - :doc:`bates`
     - ``SwitchingBatesModel``
     - :math:`\theta,\ \ell`
     - :math:`\kappa\theta_i D(t) + \ell_i(\phi_J - 1 - iu\bar k)`
     - second order
   * - :doc:`variance_gamma`
     - ``SwitchingVarianceGammaProcess``
     - :math:`\sigma,\ \nu,\ \theta`
     - :math:`\psi_i(u) + iu\,\omega_i`
     - exact

.. rubric:: Hybrid

.. list-table::
   :header-rows: 1
   :widths: 16 26 12 30 16

   * - Model
     - Class
     - Switches
     - Forcing :math:`g_i`
     - Closed form
   * - :doc:`equity_rates`
     - ``SwitchingEquityRates``
     - both parts'
     - rates forcing at weight :math:`1 - iw` plus equity forcing at :math:`w`
     - second order

The common formula
------------------

A price under a switching model is one function per regime, :math:`u_i(t, x)`, and the pricing equations are coupled
through the generator :math:`Q` of the chain. For every model in the tables except Heston with a switching volatility of variance, the state factors out and
what remains is a linear system in time alone,

.. math::

    a'(t) = \big(Q + \operatorname{diag} g(t)\big)\,a(t), \qquad a(0) = \mathbf 1,

with one forcing :math:`g_i(t)` per regime. The model pages differ only in :math:`g_i` and in the prefactor that
multiplies :math:`a_i`.

The closed forms are written for two regimes switching at rate :math:`\lambda` in each direction,
``rl.RegimeChain.twoState(lam, lam)``, with

.. math::

    \varepsilon = \frac1\lambda, \qquad \bar g = \tfrac12(g_1 + g_2), \qquad \tilde g = \tfrac12(g_1 - g_2).

The average :math:`\bar g` is the forcing of the averaged model, and the half-difference :math:`\tilde g` measures
how much the regimes disagree. To second order in :math:`\varepsilon`, up to terms of size :math:`e^{-2\lambda T}`,

.. math::

    a_{1,2}(T) = \exp\Big(\int_0^T \bar g + \frac{\varepsilon}{2}\int_0^T \tilde g^{\,2}
      - \frac{\varepsilon^2}{8}\big(\tilde g(T)^2 + \tilde g(0)^2\big)\Big)
      \Big(1 \pm \frac{\varepsilon}{2}\,\tilde g(T) \mp \frac{\varepsilon^2}{4}\,\tilde g'(T)\Big) + O(\varepsilon^3),

with the upper sign for a start in the first regime (``regime=0``). The three parts have fixed meanings:

- :math:`e^{\int\bar g}` is the averaged model;
- :math:`\tfrac12\varepsilon\int\tilde g^{\,2}` is the Green–Kubo term, half the variance of the fluctuating integral
  of :math:`g`, the same from both regimes;
- the bracket is the memory of the starting regime.

When :math:`g` does not depend on time the system is solved exactly:

.. math::

    a_{1,2}(T) = e^{(\bar g - \lambda)T}\Big(\cosh sT + \frac{\lambda \pm \tilde g}{s}\,\sinh sT\Big),
    \qquad s = \sqrt{\lambda^2 + \tilde g^2}.

So each page needs five things from its model: :math:`\int\bar g`, :math:`\int\tilde g^{\,2}`,
:math:`\tilde g(T)`, :math:`\tilde g(0)` and :math:`\tilde g'(T)`.

The helpers
-----------

The Python on every page uses these functions. They are the two formulas above, the integrals of
:math:`B(t) = (1 - e^{-at})/a`, Lewis's formula, and two wrappers around the library.

.. literalinclude:: helpers.py
   :language: python
   :lines: 3-

For a chain with more regimes or unequal rates the library's engines apply as they stand:
``FastSwitchingEngine`` runs the same expansion to any order with the group inverse of :math:`Q` in place of the
division by :math:`2\lambda`, and ``NumericalSwitchingEngine`` solves the linear system.

Adding jumps
------------

There are two ways to add jumps and keep a closed form. The first is a jump in the state at an intensity
:math:`\ell_i` that depends on the regime. Its term in the pricing equation is
:math:`\ell_i\,\mathbb{E}[u_i(x + J) - u_i(x)]`, and on an exponential-affine solution :math:`e^{-B(t)x}a_i(t)` a jump
of size :math:`J` only multiplies by :math:`e^{-BJ}`. The term therefore becomes
:math:`\ell_i\,(\mathbb{E}[e^{-B(t)J}] - 1)`, a function of time added to the forcing :math:`g_i`. Any jump law with a
known transform will do, the intensity may switch, and so may the law itself, by giving each regime its own
transform. This covers :doc:`vasicek_jumps`, :doc:`merton` and :doc:`bates`. A pure-jump Lévy model is the same
statement with nothing else in it: :math:`g_i` is the regime's Lévy exponent, so every parameter may switch, as in
:doc:`variance_gamma`, and the same holds for any Lévy process whose exponent is known. When the state coefficient
does not depend on time, as for an equity characteristic function, the forcing is constant and the two-regime
solution is exact. The one restriction is that the jump must not put a switched parameter into the Riccati
equation: an intensity proportional to the state does, and then only its constant part may switch.

The second is a jump that happens at the moment the regime changes. If a move from regime :math:`i` to :math:`j`
shifts the state by a random amount :math:`J_{ij}`, the switching term :math:`Q_{ij}u_j` becomes
:math:`Q_{ij}\,\mathbb{E}[u_j(x + J_{ij})]`, and the same factorisation multiplies each off-diagonal rate by a
transform: :math:`a' = (Q\circ\Phi(t) + \operatorname{diag} g)\,a` with :math:`\Phi_{ij} = \mathbb{E}[e^{-B(t)J_{ij}}]`
and :math:`\Phi_{ii} = 1`. The system is still linear, and for two regimes with constant :math:`g` and
:math:`\Phi` it is still exact,

.. math::

    a_1(T) = e^{(\bar g - \lambda)T}\Big(\cosh sT + \frac{\tilde g + \lambda\Phi_{12}}{s}\,\sinh sT\Big), \qquad
    s = \sqrt{\tilde g^2 + \lambda^2\,\Phi_{12}\Phi_{21}},

with :math:`a_2` given by exchanging the regimes. What changes is the expansion. These jumps arrive at the switching
rate, so as the chain speeds up they pile up unless their size shrinks with the holding time, and
:math:`Q\circ\Phi` is not a generator, so the averaged model is not simply the model at the averaged parameters. The
library implements the first kind and not the second.

.. toctree::
   :maxdepth: 1
   :hidden:

   vasicek
   vasicek_jumps
   cir
   hull_white
   g2
   intensity_basket
   black_scholes
   merton
   variance_gamma
   heston
   bates
   equity_rates
   cev
   heston_vol_of_vol
