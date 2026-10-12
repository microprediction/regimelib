Testing whether volatility fluctuates
=====================================

To first order in the holding time, a fast-switching chain enters an option price through one number per switched
parameter, the Green–Kubo constant

.. math::

    K = \int_0^\infty \operatorname{Cov}_\pi\big(f(y_0), f(y_t)\big)\,dt = -\pi\cdot\big(\tilde f\,Q^\#\tilde f\big),
    \qquad \tilde f = f - \pi\cdot f,

plus the memory term of the starting regime. Here :math:`f = \sigma^2`. A constant volatility is :math:`K = 0`, so
the hypothesis that volatility does not fluctuate is one restriction, :math:`H_0: K = 0`, whatever the number of
regimes. This page states the restriction's effect on prices, builds a test of it from returns, and checks the
test's size and power on simulated paths. It is an example, not part of the library: the library prices, and
estimating :math:`K` from data is left to the user.

What K does to a price
----------------------

For Black–Scholes with a switching volatility, the averaged model is Black–Scholes at :math:`\bar\sigma^2 = \pi\cdot
\sigma^2`, and the first-order correction is

.. math::

    C_1 = K\,T\,\frac{\partial^2 C}{\partial V^2}\bigg|_{V = \bar\sigma^2 T},

with :math:`V` the total variance. For two regimes with rates :math:`\lambda_{01}, \lambda_{10}` and
:math:`\lambda = \lambda_{01} + \lambda_{10}`, :math:`K = \pi_0\pi_1(\sigma_0^2 - \sigma_1^2)^2/\lambda`. Both
statements against the library:

.. code-block:: python

    import math
    from scipy.stats import norm
    from regimelib.firstorder import green_kubo

    chain = rl.RegimeChain.twoState(30.0, 20.0)
    sigma = np.array([0.30, 0.15]); f = sigma ** 2
    pi = chain.stationaryDistribution(); lam = 50.0
    S0, r, T, strike = 100.0, 0.02, 1.0, 100.0

    def black(V):
        F = S0 * math.exp(r * T); sd = math.sqrt(V)
        d1 = (math.log(F / strike) + V / 2) / sd
        return math.exp(-r * T) * (F * norm.cdf(d1) - strike * norm.cdf(d1 - sd))

    K = green_kubo(chain, [f])[0][0, 0]
    V, h = float(pi @ f) * T, 1e-4
    curvature = (black(V + h) - 2 * black(V) + black(V - h)) / h ** 2
    engine = rl.FirstOrderFDEngine(rl.SwitchingBlackScholesProcess(chain, S0, r, 0.0, list(sigma)), regime=0, n=1601)
    option = rl.VanillaOption(("call", strike), maturity=T); option.setPricingEngine(engine); option.NPV()
    print(f"K from green_kubo     {K:.6e}")
    print(f"pi0 pi1 (f0-f1)^2/lam {pi[0] * pi[1] * (f[0] - f[1]) ** 2 / lam:.6e}")
    print(f"correction (engine)   {engine.correction:.5f}")
    print(f"K T d2C/dV2           {K * T * curvature:.5f}")


.. code-block:: text

    K from green_kubo     2.187000e-05
    pi0 pi1 (f0-f1)^2/lam 2.187000e-05
    correction (engine)   -0.01949
    K T d2C/dV2           -0.01949

K from returns
--------------

The integrated variance over an interval of length :math:`h` has

.. math::

    \operatorname{Var}\int_0^h\sigma^2(y_t)\,dt = 2\int_0^h (h - s)\,C(s)\,ds = 2Kh - 2\int_0^\infty s\,C(s)\,ds
    + o(1), \qquad C(s) = \operatorname{Cov}_\pi\big(f(y_0), f(y_s)\big),

so :math:`K` is half the long-run variance :math:`\int_{-\infty}^\infty C(s)\,ds` of :math:`\sigma^2`. For two
regimes :math:`C(s) = \pi_0\pi_1(f_0 - f_1)^2e^{-\lambda s}` and the variance is exactly
:math:`2K\,[h - (1 - e^{-\lambda h})/\lambda]`.

Take returns :math:`r_j` at spacing :math:`\Delta`, in :math:`B` blocks of :math:`n`, so a block has length
:math:`h = n\Delta`, and let :math:`RV_b = \sum_{j\in b} r_j^2`. If the returns were independent, the variance of a
block's :math:`RV_b` would be :math:`n` times the variance of one :math:`r_j^2`. A fluctuating :math:`\sigma^2`
makes squared returns serially dependent and adds :math:`\operatorname{Var}\int_h\sigma^2`. The estimate and the
statistic are

.. math::

    \hat K = \frac{s^2(RV) - n\,\overline{s^2_b(r^2)}}{2h}, \qquad
    R = \frac{s^2(RV)}{n\,\overline{s^2_b(r^2)}},

with :math:`s^2(RV)` the sample variance of the block sums and :math:`\overline{s^2_b(r^2)}` the average
within-block sample variance of the squared returns. Under :math:`H_0` with independent, identically distributed
returns of any law, the returns are exchangeable, so permuting them gives the exact null law of :math:`R`. The test
then makes no assumption about tails: independent jumps and fat-tailed returns leave its size at the nominal level.
Using the Gaussian fourth moment instead (:math:`\tfrac23\sum r^4` for :math:`n` times the variance of
:math:`r^2`) gives a test that rejects most of the time under fat tails with no fluctuation at all.

With blocks of length :math:`h`, :math:`\hat K` estimates :math:`K[1 - (1 - e^{-\lambda h})/(\lambda h)]`, less
what fluctuates within a block, since the within-block term absorbs it. One path of ten years of daily returns,
monthly blocks, regimes of 30% and 15% volatility that last about a week, and one path with a constant volatility
and Student-t tails:

.. code-block:: python

    rng = np.random.default_rng(1)
    dt, days, n = 1 / 252, 2520, 21

    def integrated_variance(f, l01, l10, rng):
        """The integrated variance over each day for a two-state chain started from its stationary law, exactly."""
        rates = (l01, l10); state = int(rng.random() >= l10 / (l01 + l10)); t = 0.0
        jump = rng.exponential(1 / rates[state]); out = np.empty(days)
        for d in range(days):
            end, acc = (d + 1) * dt, 0.0
            while jump < end:
                acc += f[state] * (jump - t); t = jump; state = 1 - state
                jump = t + rng.exponential(1 / rates[state])
            out[d] = acc + f[state] * (end - t); t = end
        return out

    def statistic(r):
        s = r[: len(r) // n * n].reshape(-1, n) ** 2
        between, within = s.sum(1).var(ddof=1), n * s.var(1, ddof=1).mean()
        return between / within, (between - within) / (2 * n * dt)

    def pvalue(r, permutations=199):
        R = statistic(r)[0]
        return (1 + sum(statistic(rng.permutation(r))[0] >= R for _ in range(permutations))) / (1 + permutations)

    f, lam = (0.09, 0.0225), 50.0
    switching = np.sqrt(integrated_variance(f, lam / 2, lam / 2, rng)) * rng.standard_normal(days)
    constant = 0.2 * math.sqrt(dt / 2) * rng.standard_t(4, days)
    h = n * dt; K = 0.25 * (f[0] - f[1]) ** 2 / lam
    print(f"switching   Khat {statistic(switching)[1]:.2e}   K over a block {K * (1 - (1 - math.exp(-lam * h)) / (lam * h)):.2e}   p {pvalue(switching):.3f}")
    print(f"constant    Khat {statistic(constant)[1]:+.2e}                          p {pvalue(constant):.3f}")


.. code-block:: text

    switching   Khat 1.37e-05   K over a block 1.74e-05   p 0.005
    constant    Khat -2.71e-06                          p 0.555

Size and power
--------------

Repeating that 200 times for each case, with 99 permutations, and rejecting at 5% (the rates for a constant
volatility are within the sampling error of 5%, about 1.5 points at 200 paths):

.. code-block:: python

    def rejection(make, reps=200):
        return np.mean([pvalue(make(), 99) <= 0.05 for _ in range(reps)])

    cases = [("constant, normal", lambda: 0.2 * math.sqrt(dt) * rng.standard_normal(days)),
             ("constant, t4 tails", lambda: 0.2 * math.sqrt(dt / 2) * rng.standard_t(4, days)),
             ("constant, iid jumps", lambda: 0.15 * math.sqrt(dt) * rng.standard_normal(days)
                                             + rng.poisson(10 * dt, days) * rng.normal(0.0, 0.03, days))]
    for lam in (50.0, 250.0, 1000.0):
        cases.append((f"switching, lam {lam:g}",
                      lambda lam=lam: np.sqrt(integrated_variance(f, lam / 2, lam / 2, rng)) * rng.standard_normal(days)))
    for name, make in cases:
        print(f"{name:22s} rejects {100 * rejection(make):5.1f}%")


.. code-block:: text

    constant, normal       rejects   4.0%
    constant, t4 tails     rejects   5.5%
    constant, iid jumps    rejects   5.5%
    switching, lam 50      rejects 100.0%
    switching, lam 250     rejects  33.5%
    switching, lam 1000    rejects   7.5%

Regimes that last a week or more are found with certainty in ten years of daily data. At :math:`\lambda = 250`
(a regime lasts about two trading days) power is low, and the price correction on a one-year at-the-money call is
then about 0.04% of the price; at :math:`\lambda = 1000` both are negligible. The test loses power where the
fluctuation stops mattering for the price.

What the test does not do
-------------------------

- It tests a fluctuating volatility against a constant one, not regimes against a diffusive stochastic volatility:
  both enter prices to first order through a constant of the same kind, and both make squared returns serially
  dependent.
- It needs returns that are independent under the null. Independent jumps are allowed; a jump intensity that itself
  varies is a fluctuation the test detects.
- :math:`\hat K` is biased low when the regimes switch within a block. Shorter blocks reduce that bias and the
  power with it.
