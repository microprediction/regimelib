Other instruments
=================

How each remaining instrument is priced from the reduced system :math:`a' = (Q + \operatorname{diag} g)\,a`: what is
solved, with which terminal vector, and what is then integrated.

Bonds and claims on the regime
------------------------------

A bond, or a survival probability when the state is a default intensity, is
:math:`P_i(0, T) = e^{-B(T)\cdot x_0}a_i(T)` with :math:`a(0) = \mathbf 1`. The terminal vector says what is paid in
each regime at :math:`T`. Starting from the unit vector :math:`e_j` gives the value of one unit paid only if the
regime at :math:`T` is :math:`j`, and these add up to the bond.

.. code-block:: python

    lam, T = 5.0, 5.0
    model = rl.SwitchingVasicek(rl.RegimeChain.twoState(lam, lam), 0.03, 0.5, [0.06, 0.02], [0.015, 0.008])
    engine = rl.NumericalSwitchingEngine(model, regime=0)
    parts = []
    for j in (0, 1):
        claim = rl.ZeroCouponBond(T, regimeAtMaturity=j)
        claim.setPricingEngine(engine)
        parts.append(claim.NPV())
        print(f"paid only if the regime at T is {j}   {parts[-1]:.8f}")
    print(f"sum                                 {sum(parts):.8f}")
    print(f"bond                                {library_bond(model, T):.8f}")


.. code-block:: text

    paid only if the regime at T is 0   0.41647096
    paid only if the regime at T is 1   0.41655024
    sum                                 0.83302120
    bond                                0.83302120

European and digital options on equity
--------------------------------------

With :math:`\phi_i(u)` the characteristic function of the martingale log return, a call is Lewis's formula (the
``lewis_call`` helper), and a digital call paying one if :math:`S_T > K` is the Gil–Pelaez inversion of the same
function,

.. math::

    e^{-rT}\,\mathbb{P}(S_T > K) = e^{-rT}\Big[\frac12 + \frac1\pi\int_0^\infty\operatorname{Re}\Big(\frac{e^{iu\log(F/K)}\,\phi(u)}{iu}\Big)du\Big].

Sensitivities to the spot follow by differentiating under the integral. The derivative in :math:`T` needs
:math:`\partial_T\phi`, which the reduced system supplies without a further solve:
:math:`\partial_Ta(T) = (Q + \operatorname{diag} g(T))\,a(T)`.

.. code-block:: python

    S0, r, q, sigma, lam, T, K = 100.0, 0.03, 0.0, [0.30, 0.15], 25.0, 1.0, 105.0
    F = S0 * np.exp((r - q) * T)

    def phi(u):
        g = [-0.5 * s * s * (u * u + 1j * u) for s in sigma]
        return exact_two_state(g[0], g[1], lam, T)

    x, w = np.polynomial.legendre.leggauss(400)
    u, w = (x + 1) * 30.0, w * 30.0                                   # nodes on (0, 60)
    closed = np.exp(-r * T) * (0.5 + np.sum(w * (np.exp(1j * u * np.log(F / K)) * phi(u) / (1j * u)).real) / np.pi)

    model = rl.SwitchingBlackScholesProcess(rl.RegimeChain.twoState(lam, lam), S0, r, q, sigma)
    digital = rl.VanillaOption(("cash", "call", K, 1.0), maturity=T)
    digital.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0))
    print(f"digital call, closed form  {closed:.6f}   library  {digital.NPV():.6f}")


.. code-block:: text

    digital call, closed form  0.408966   library  0.408966

Credit default swaps
--------------------

When the state is a default intensity, :math:`Q(t) = e^{-B(t)x_0}a_i(t)` is the survival probability. A swap paying
the spread :math:`s` on dates :math:`t_1 < \dots < t_n`, with accrued premium paid at default and recovery
:math:`R`, has

.. math::

    \text{premium} = s\sum_k\Big[\tau_kD(t_k)Q(t_k) + \tfrac12\tau_kD(t_k^m)\big(Q(t_{k-1}) - Q(t_k)\big)\Big], \qquad
    \text{protection} = (1 - R)\sum_kD(t_k^m)\big(Q(t_{k-1}) - Q(t_k)\big),

with :math:`\tau_k = t_k - t_{k-1}`, :math:`t_k^m` the midpoint of the period and :math:`D` the discount curve. The
fair spread is the ratio of the protection leg to the premium leg per unit spread. Only survival probabilities are
needed, so the closed form is the :doc:`vasicek` bond evaluated at each date.

.. code-block:: python

    x0, a, b, sigma, lam = 0.02, 0.5, [0.06, 0.01], [0.01, 0.004], 5.0
    recovery, rate, times = 0.4, 0.03, [0.25 * k for k in range(1, 21)]
    bbar, bt = half(b)
    sbar, st = half([s * s for s in sigma])

    def survival(t):
        if t == 0:
            return 1.0
        E, B, I1, I2, I3, I4 = powers_of_B(a, t)
        return np.exp(-B * x0) * second_order(
            -a * bbar * I1 + sbar * I2 / 2, a * a * bt * bt * I2 - a * bt * st * I3 + st * st * I4 / 4,
            -a * bt * B + st * B * B / 2, 0.0, (-a * bt + st * B) * E, lam)

    premium = protection = 0.0
    for t0, t1 in zip([0.0] + times[:-1], times):
        tau, mid, dQ = t1 - t0, (t0 + t1) / 2, survival(t0) - survival(t1)
        premium += tau * np.exp(-rate * t1) * survival(t1) + 0.5 * tau * np.exp(-rate * mid) * dQ
        protection += (1 - recovery) * np.exp(-rate * mid) * dQ

    model = rl.SwitchingVasicek(rl.RegimeChain.twoState(lam, lam), x0, a, b, sigma)
    cds = rl.CreditDefaultSwap("buyer", 0.01, times, recovery, discount=rate)
    cds.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0))
    print(f"fair spread, closed form second order  {1e4 * protection / premium:.4f} bp")
    print(f"fair spread, library                   {1e4 * cds.fairSpread():.4f} bp")


.. code-block:: text

    fair spread, closed form second order  178.1448 bp
    fair spread, library                   178.1437 bp

Several names on one chain have joint survival equal to the bond of the summed forcing, so a first-to-default swap
is a single-name swap on the sum (:doc:`intensity_basket`).

Geometric Asian options
-----------------------

Under :doc:`black_scholes` with a switching volatility the continuous geometric average is :math:`G = S_0e^Y` with

.. math::

    Y = \frac1T\int_0^T\log\frac{S_t}{S_0}\,dt = \int_0^T\Big(1 - \frac tT\Big)\,d\log S_t .

Each increment of :math:`\log S` enters :math:`Y` scaled by the fraction of the life that remains, which in time to
maturity :math:`\tau` is :math:`\tau/T`. So :math:`\mathbb{E}[e^{iuY} \mid y_0 = i] = a_i(T)` with

.. math::

    g_i(\tau) = iu\,\big(r - q - \tfrac12\sigma_i^2\big)\frac\tau T - \tfrac12u^2\sigma_i^2\frac{\tau^2}{T^2},

a polynomial in :math:`\tau`, so every integral is elementary:

.. math::

    \int_0^T\bar g = \tfrac12iu\,\big(r - q - \tfrac12\bar s\big)T - \tfrac16u^2\bar s\,T, \qquad
    \int_0^T\tilde g^{\,2} = \tfrac14\tilde s^2T\Big(-\frac{u^2}{3} + \frac{iu^3}{2} + \frac{u^4}{5}\Big),

.. math::

    \tilde g(T) = -\tfrac12\tilde s\,(iu + u^2), \qquad \tilde g(0) = 0, \qquad \tilde g'(T) = -\frac{\tilde s}{2T}(iu + 2u^2).

The first integral is the Kemna–Vorst result: the average of a lognormal path has half the drift and a third of
the variance. The option is Lewis's formula on :math:`G`, whose forward is :math:`F_G = S_0\,\mathbb{E}[e^Y]`.

.. code-block:: python

    S0, r, q, sigma, lam, T = 100.0, 0.03, 0.0, [0.30, 0.15], 50.0, 1.0
    sbar, st = half([s * s for s in sigma])

    def phi_Y(u):
        return second_order(
            int_gbar=0.5j * u * (r - q - sbar / 2) * T - u * u * sbar * T / 6,
            int_gt2=st * st * T / 4 * (-u * u / 3 + 1j * u ** 3 / 2 + u ** 4 / 5),
            gt_T=-st / 2 * (1j * u + u * u),
            gt_0=0.0,
            dgt_T=-st / (2 * T) * (1j * u + 2 * u * u),
            lam=lam)

    def asian_call(K, U=60.0, n=400):
        FG = S0 * phi_Y(-1j).real
        k, lf = np.log(FG / K), np.log(FG / S0)
        x, w = np.polynomial.legendre.leggauss(n)
        u, w = (x + 1) * U / 2, w * U / 2
        z = u - 0.5j
        integral = np.sum(w * (np.exp(1j * u * k) * phi_Y(z) * np.exp(-1j * z * lf)).real / (u * u + 0.25))
        return np.exp(-r * T) * (FG - np.sqrt(FG * K) / np.pi * integral)

    model = rl.SwitchingBlackScholesProcess(rl.RegimeChain.twoState(lam, lam), S0, r, q, sigma)
    for K in (95.0, 105.0):
        asian = rl.ContinuousGeometricAsianOption(("call", K), maturity=T)
        asian.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0))
        print(f"K = {K:5.0f}   closed form, second order  {asian_call(K):.6f}   library  {asian.NPV():.6f}")


.. code-block:: text

    K =    95   closed form, second order  8.672621   library  8.672626
    K =   105   closed form, second order  3.785976   library  3.785979

Barrier, American and Bermudan instruments
------------------------------------------

These depend on the path of the state and not only on its terminal value, so the state does not factor out and
there is no reduced system. They are priced on the coupled equations themselves, one unknown function per regime,

.. math::

    \partial_tu_i + L_iu_i + \sum_jQ_{ij}u_j = 0,

on a grid in the state with one block per regime. A knock-out barrier is the condition :math:`u_i = \text{rebate}`
at the barrier in every regime, and a knock-in is the vanilla less the knock-out. An American option is the
variational inequality

.. math::

    \max\Big(\partial_tu_i + L_iu_i + \sum_jQ_{ij}u_j,\ \Phi - u_i\Big) = 0,

with an exercise boundary that differs by regime. A Bermudan swaption projects onto the exercise value at each
date, and that value is the coupon bond in each regime from the reduced system.

.. code-block:: python

    lam = 5.0
    model = rl.SwitchingBlackScholesProcess(rl.RegimeChain.twoState(lam, lam), 100.0, 0.05, 0.0, [0.30, 0.15])
    grid = rl.SwitchingFDEngine(model, regime=0, n=801, steps=400)
    european = rl.VanillaOption(("put", 100.0), maturity=1.0)
    american = rl.VanillaOption(("put", 100.0), exercise="american", maturity=1.0)
    knock_out = rl.BarrierOption("downout", 80.0, 0.0, ("put", 100.0), maturity=1.0)
    knock_in = rl.BarrierOption("downin", 80.0, 0.0, ("put", 100.0), maturity=1.0)
    for name, instrument in (("European put", european), ("American put", american),
                             ("down-and-out put", knock_out), ("down-and-in put", knock_in)):
        instrument.setPricingEngine(grid)
        print(f"{name:18s} {instrument.NPV():.4f}")
    print(f"{'out + in':18s} {knock_out.NPV() + knock_in.NPV():.4f}")


.. code-block:: text

    European put       7.1739
    American put       7.7214
    down-and-out put   1.1979
    down-and-in put    5.9760
    out + in           7.1739

Sensitivities to the parameters
-------------------------------

The closed forms are explicit in the parameters, so their derivatives are formulas. To first order the log price is
:math:`\int\bar g + \tfrac12\varepsilon\int\tilde g^{\,2} + \log(1 \pm \tfrac12\varepsilon\tilde g(T))`. If a parameter
:math:`p_i` of regime :math:`i` enters the forcing linearly, :math:`g_i = p_i\,f(t) + \dots`, then

.. math::

    \frac{\partial\log a_{1,2}}{\partial p_1} = \frac12\int_0^Tf + \frac\varepsilon2\int_0^T\tilde g\,f \pm \frac\varepsilon4f(T), \qquad
    \frac{\partial\log a_{1,2}}{\partial p_2} = \frac12\int_0^Tf - \frac\varepsilon2\int_0^T\tilde g\,f \mp \frac\varepsilon4f(T),

.. math::

    \frac{\partial\log a_{1,2}}{\partial\lambda} = -\varepsilon^2\Big(\frac12\int_0^T\tilde g^{\,2} \pm \frac12\tilde g(T)\Big).

The first term is the sensitivity of the averaged model, shared equally between the regimes; the second moves it
towards the regime whose parameter matters more; the third is the memory of the start. For the Vasicek mean level
:math:`f = -aB`.

.. code-block:: python

    x0, a, b, sigma, lam, T = 0.02, 0.5, [0.06, 0.01], [0.01, 0.004], 5.0, 5.0
    E, B, I1, I2, I3, I4 = powers_of_B(a, T)
    _, bt = half(b)
    _, st = half([s * s for s in sigma])
    eps = 1 / lam
    cross = -a * bt * I2 + st * I3 / 2                                    # int g~ B
    formula = {"b1": -a / 2 * I1 - eps * a / 2 * cross - eps * a / 4 * B,
               "b2": -a / 2 * I1 + eps * a / 2 * cross + eps * a / 4 * B,
               "lambda": -eps ** 2 * (0.5 * (a * a * bt * bt * I2 - a * bt * st * I3 + st * st * I4 / 4)
                                      + 0.5 * (-a * bt * B + st * B * B / 2))}

    def log_bond(b_, lam_):
        return np.log(library_bond(rl.SwitchingVasicek(rl.RegimeChain.twoState(lam_, lam_), x0, a, b_, sigma), T))

    d = 1e-5
    numeric = {"b1": (log_bond([b[0] + d, b[1]], lam) - log_bond([b[0] - d, b[1]], lam)) / (2 * d),
               "b2": (log_bond([b[0], b[1] + d], lam) - log_bond([b[0], b[1] - d], lam)) / (2 * d),
               "lambda": (log_bond(b, lam + 1e-3) - log_bond(b, lam - 1e-3)) / 2e-3}
    for name in formula:
        print(f"d log P / d {name:7s} formula {formula[name]:+.6f}   finite difference of the library {numeric[name]:+.6f}")


.. code-block:: text

    d log P / d b1      formula -1.622192   finite difference of the library -1.622189
    d log P / d b2      formula -1.541978   finite difference of the library -1.541981
    d log P / d lambda  formula +0.000429   finite difference of the library +0.000426

For a general chain the first-order price depends on the chain through the stationary distribution :math:`\pi` and
the group inverse :math:`Q^\#`, whose derivatives are exact:

.. math::

    d\pi = -\pi\,dQ\,Q^\#, \qquad dQ^\# = -Q^\#\,dQ\,Q^\# + \mathbf 1\pi\,dQ\,(Q^\#)^2 + (Q^\#)^2\,dQ\,\mathbf 1\pi .

``parameterGreek`` on the first-order symbolic classes implements this (:doc:`../reference/symbolic`).
