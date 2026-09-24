# The first-order tier

Everything in `regimelib` so far uses the exact reduction: the switched parameters enter only the per-regime forcing
of `a' = (Q + diag g) a`, so the expansion is valid to every order and the state factors out. The QuantLib models the
homogenization catalogue marks "first order only" break that: a switched vol-of-vol in Heston, correlated
Heston-Hull-White with a switched rate level, CEV or local-volatility with a switched scale. There the switched
parameter multiplies an operator that does not commute with the rest, and the state no longer factors out.

## The rule

Write the pricing equation as `u_t = L_y u` with `L_y = L_bar + sum_j f~_j(y) A_j`, where `f~_j` are the
zero-mean fluctuating coefficients over the regimes and `A_j` are the operators they multiply. To first order in
the mean holding time `eps`,

    u(T) = e^{T L_bar} u0 + eps * int_0^T e^{(T-s) L_bar} ( sum_jk K_jk A_j A_k ) e^{s L_bar} u0 ds + (regime memory) + O(eps^2),

with `K_jk = -pi . (f_j Q# f~_k)` the Green-Kubo matrix of the chain (its antisymmetric part acts through the
commutator `[A_j, A_k]`), and the memory term `eps (Q# f~)_i . A e^{T L_bar} u0` for a start in regime `i`. This is
the first-order rule of the any-equation page; the Duhamel form is used, not an exponentiated corrected generator.

## Implementation

A finite-difference engine on the state grid, method of lines:

- each model supplies `L_bar` and the `A_j` as sparse matrices on a tensor grid it chooses (1-D for CEV and
  Black-Scholes, 2-D for Heston with switched vol-of-vol);
- `K` comes from `RegimeChain` through the group inverse already in `_engine/fastswitch.py`;
- the augmented linear system `(u0, u1)' = [[L_bar, 0], [sum K A A, L_bar]] (u0, u1)` is integrated once
  (`scipy.sparse.linalg.expm_multiply`, or an implicit scheme with Rannacher start for non-smooth payoffs);
  `u0` is the averaged price and `u1` the first-order correction, both on the grid;
- prices and grid greeks are read off by interpolation at the initial state.

## Referees

- Frozen limit: the averaged PDE against QuantLib's own finite-difference engine for the model
  (`FdBlackScholesVanillaEngine`, `FdHestonVanillaEngine`) at the averaged parameters.
- Switching on, numerical solution: the full coupled system `u_i' = L_i u_i + sum_j Q_ij u_j` on the same grid,
  which is the switching model solved without expansion.
- Switching on, closed form where the exact reduction also applies: Black-Scholes with a switched volatility is
  affine, so the first-order finite-difference engine must match `FastSwitchingEngine(order=1)` there. This is the
  certificate that the operator machinery is right before it is used on the non-affine models.
- Switching on, Monte Carlo with exact regime paths and an Euler step for the state between switches.

## First targets, in order

1. Black-Scholes with switched volatility on a log-price grid (check against the exact engine at first order).
2. CEV, `dS = sigma_y S^beta dW`, with switched `sigma` (non-affine, one dimension).
3. Heston with switched vol-of-vol `xi` (two operators: `(1/2) v d_vv` for `xi^2` and `rho v d_xv` for `xi`, in two
   dimensions), frozen limit against `FdHestonVanillaEngine`.
4. Correlated Heston-Hull-White with a switched rate level, three dimensions, which decides whether the finite-
   difference route scales or a Fourier-space variant is needed.

## What it does not do

It gives the first-order term only. Second order in the non-affine case needs the second Duhamel term with the
three-point time correlations of the chain, which the site has not derived for operators that do not commute; that
is a separate piece of mathematics before it is code.
