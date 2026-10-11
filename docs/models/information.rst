Observed and inferred regimes
=============================

Every engine takes ``regime``, a starting regime or a belief about it, and the engines that price a decision take
``information``, ``"observed"`` or ``"inferred"``. This page says when the two differ and what is computed in each
case.

A belief as the starting regime
-------------------------------

``regime=0`` starts in a known regime. ``regime=[0.3, 0.7]`` starts from a belief: probability 0.3 that the regime
is 0. For a payoff that depends only on what is observed, the price from a belief is the average over regimes of
the price from each,

.. math::

    u(p) = \sum_ip_i\,u_i,

because pricing is linear when nobody has to act on the regime. This covers bonds, European and digital options on
equity, barriers, Asian options and credit default swaps. Ratios are taken after averaging: the fair spread of a
swap is the averaged protection leg over the averaged premium annuity.

.. code-block:: python

    chain = rl.RegimeChain.twoState(2.0, 3.0)
    model = rl.SwitchingVasicek(chain, 0.03, 0.5, [0.07, 0.02], 0.01)
    belief = [0.3, 0.7]
    print(f"bond from regime 0        {library_bond(model, 4.0, regime=0):.8f}")
    print(f"bond from regime 1        {library_bond(model, 4.0, regime=1):.8f}")
    print(f"bond from the belief      {library_bond(model, 4.0, regime=belief):.8f}")
    print(f"0.3 x first + 0.7 x second {0.3 * library_bond(model, 4.0, 0) + 0.7 * library_bond(model, 4.0, 1):.8f}")


.. code-block:: text

    bond from regime 0        0.84506184
    bond from regime 1        0.85227814
    bond from the belief      0.85011325
    0.3 x first + 0.7 x second 0.85011325

Where information matters
-------------------------

The average stops being the price when someone acts on the regime. Two cases arise.

- **Options on bonds and swaps, including European ones.** The payoff depends on the bond price at expiry, and that
  price depends on what the market knows about the regime then. If the regime is observed, the bond in regime
  :math:`j` is :math:`P_j` and the option pays :math:`(P_j - K)^+`. If it is inferred, the bond is the
  belief-weighted :math:`\sum_jp_jP_j` and the option pays :math:`(\sum_jp_jP_j - K)^+`, which is smaller by
  Jensen's inequality.
- **Early exercise.** The holder of an American or Bermudan right exercises on what is known. With an observed
  regime there is one exercise boundary per regime; with an inferred one the boundary depends on the belief.

So the observed-regime price is an upper bound on the inferred-regime price.

When the path reveals the regime
--------------------------------

In continuous time the quadratic variation of an observed path gives its diffusion coefficient at once. If that
coefficient differs between every two regimes, the regime is known from the path and the two prices coincide. This
is the case whenever a volatility switches with distinct values: the Black–Scholes or CEV volatility, the Vasicek or
Hull–White volatility, the G2++ volatilities or correlation. The regime-by-regime constructions on
:doc:`bond_options` and in the grid engine are then the price under either setting, and a starting belief is
resolved at once into the average.

The two differ when only a drift level switches and the volatilities agree: the Vasicek mean level alone, or the
CIR mean level, which is the only CIR parameter that switches.

``regimelib.information.revealed(model)`` reports which case a model is in.

The belief as a state variable
------------------------------

For two regimes that differ only in the level the rate reverts to, write :math:`p_t` for the probability of regime
0 given the path of the rate. Nobody sees the regime; everybody sees the rate, and the belief moves only when the
rate surprises. With :math:`s(r) = \sigma` for Vasicek and :math:`\sigma\sqrt r` for CIR, and :math:`q_{01}`,
:math:`q_{10}` the switching rates,

.. math::

    dr_t = a\,\big(\bar b(p_t) - r_t\big)\,dt + s(r_t)\,d\nu_t, \qquad \bar b(p) = p\,b_0 + (1 - p)\,b_1,

.. math::

    dp_t = \big(q_{10}(1 - p_t) - q_{01}\,p_t\big)\,dt + p_t(1 - p_t)\,\frac{a\,(b_0 - b_1)}{s(r_t)}\,d\nu_t .

There is one Brownian motion :math:`\nu`, the innovation: the part of the rate's move that the belief did not
expect. The first term of :math:`dp` is the chain pulling the belief towards its stationary value, and the second
is learning, proportional to how different the two drifts are and inversely to the noise. The pair
:math:`(r, p)` is Markov, so a claim is one function :math:`V(t, r, p)` solving

.. math::

    \partial_tV + a(\bar b - r)\,V_r + \tfrac12s^2V_{rr} + \mu_pV_p + \tfrac12\gamma^2V_{pp} + s\gamma\,V_{rp} - rV = 0,
    \qquad \mu_p = q_{10}(1 - p) - q_{01}p, \quad \gamma = \frac{p(1 - p)\,a(b_0 - b_1)}{s(r)},

with no regime blocks. The bond at :math:`(r, p)` is :math:`p\,P_0(r) + (1 - p)\,P_1(r)`, exactly, so the exercise
value of an option on bonds or swaps is a known function of :math:`(r, p)`. ``SwitchingFDEngine`` solves this on a
grid in :math:`(r, p)`.

A zero-strike call on a bond is the bond, whose value from a belief is known exactly, so it checks the belief's
dynamics with a switching level:

.. code-block:: python

    cir = rl.SwitchingCoxIngersollRoss(chain, 0.03, [0.07, 0.02], 0.5, 0.08)
    for name, m in (("Vasicek", model), ("CIR", cir)):
        option = rl.CouponBondOption("call", 0.0, 1.0, [(3.0, 1.0)])
        option.setPricingEngine(rl.SwitchingFDEngine(m, regime=belief, n=(201, 41), steps=200))
        print(f"{name:8s} belief grid {option.NPV():.7f}   exact bond from the belief {library_bond(m, 3.0, regime=belief):.7f}")


.. code-block:: text

    Vasicek  belief grid 0.8901135   exact bond from the belief 0.8901135
    CIR      belief grid 0.8903052   exact bond from the belief 0.8903048

Observed against inferred
-------------------------

A payer swaption and its Bermudan version under a Vasicek rate whose level alone switches, on a slow chain, from an
even belief:

.. code-block:: python

    slow = rl.RegimeChain.twoState(0.5, 0.5)
    level = rl.SwitchingVasicek(slow, 0.04, 0.5, [0.07, 0.02], 0.01)
    fixed = [2.0, 3.0, 4.0, 5.0, 6.0]

    def price(information, exerciseTimes=None):
        swaption = rl.Swaption("payer", 1.0, fixed, 0.045, notional=100.0, exerciseTimes=exerciseTimes)
        n = (301, 61) if information == "inferred" else 401
        swaption.setPricingEngine(rl.SwitchingFDEngine(level, regime=[0.5, 0.5], n=n, steps=200, information=information))
        return swaption.NPV()

    print("                     observed   inferred")
    print(f"European swaption    {price('observed'):.4f}     {price('inferred'):.4f}")
    print(f"Bermudan swaption    {price('observed', [1.0, 2.0, 3.0, 4.0]):.4f}     {price('inferred', [1.0, 2.0, 3.0, 4.0]):.4f}")


.. code-block:: text

                         observed   inferred
    European swaption    1.3835     1.1690
    Bermudan swaption    2.0220     1.9079

When the volatility also switches the regime is revealed, and the characteristic-function engines price the same
swaption under either setting:

.. code-block:: python

    both = rl.SwitchingVasicek(slow, 0.04, 0.5, [0.07, 0.02], [0.012, 0.008])
    swaption = rl.Swaption("payer", 1.0, fixed, 0.045, notional=100.0)
    for information in ("observed", "inferred"):
        swaption.setPricingEngine(rl.NumericalSwitchingEngine(both, regime=[0.5, 0.5], information=information))
        print(f"{information:9s} {swaption.NPV():.6f}")
    swaption.setPricingEngine(rl.NumericalSwitchingEngine(level, regime=[0.5, 0.5]))
    try:
        swaption.NPV()
    except NotImplementedError as error:
        print("level only, inferred: refused by this engine; use SwitchingFDEngine")


.. code-block:: text

    observed  1.382383
    inferred  1.382383
    level only, inferred: refused by this engine; use SwitchingFDEngine

What each engine does
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Case
     - ``information="observed"``
     - ``information="inferred"`` (the default)
   * - Payoff of the observed state
     - average over the belief
     - average over the belief
   * - Rate option or early exercise, regime revealed by a volatility
     - regime by regime, averaged over the belief
     - the same
   * - Rate option, level only, two regimes, Vasicek or CIR
     - regime by regime
     - ``SwitchingFDEngine`` on the :math:`(r, p)` grid; the characteristic-function engines refuse
   * - Anything else that depends on information
     - regime by regime
     - refused, with the observed price named as the upper bound

With three or more regimes the belief has two or more dimensions, and with a switching level inside a two-factor
model the grid would need a third; neither is implemented.
