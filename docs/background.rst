The expansion
=============

A price under a switching model is ``u_i(t, x)``, one function per regime ``i``, and the pricing equation couples
them through the generator: ``u_i' = L_i u_i + sum_j Q_ij u_j``. For affine models the state enters through a known
factor and what is left is the reduced system

.. math::

    a'(t) = (Q + \operatorname{diag} g(t))\, a(t), \qquad a(0) = 1,

with ``g_i(t)`` the forcing of regime ``i`` (for a Vasicek bond, ``-kappa theta_i B(t) + sigma_i^2 B(t)^2 / 2``).
Every model in the all-orders tier is a statement of its forcing; the engines never see the model otherwise.

The fast-switching expansion writes ``Q = Q_1 / epsilon`` with ``epsilon`` the mean holding time and expands ``a`` in
``epsilon``. Order zero is the averaged model: the forcing replaced by its stationary average. Order one adds the
Green–Kubo integral of the forcing's fluctuation, which is the extra convexity the switching creates, and an initial
layer that remembers the starting regime. Higher orders follow by a recursion the engine runs to any order. The
series is asymptotic, so the engine stops at the best truncation and says when even that is rough.

Two things the library holds itself to:

- **Exact reduction only** in the all-orders tier. A model whose switched operators do not commute with the rest
  (CEV, a switched vol-of-vol) goes to the first-order tier, which computes the Green–Kubo correction on a grid and
  estimates the neglected term.
- **A referee for every number.** The frozen limit against QuantLib; with switching on, the numerical solution of the
  reduced system, the coupled PDE on a grid, or Monte Carlo with exact regime paths.

The mathematics — the recursion, the initial layers, the Green–Kubo matrix for a non-reversible chain, the survival
and bond formulas it started from — is on `homogenization.microprediction.org <https://homogenization.microprediction.org>`_.
