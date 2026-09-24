# regimelib

QuantLib's models with a hidden Markov regime, priced by the fast-switching expansion of
[homogenization.microprediction.org](https://homogenization.microprediction.org).

Each class mirrors a QuantLib class and takes the same parameters. The chain can have any number of regimes. A parameter that switches with the regime is
given as a list, one entry per regime. With every regime equal, each engine reproduces the QuantLib engine it
mirrors; that is a test (`tests/test_quantlib_limit.py`), checked to about 1e-9.

```python
import regimelib as rl
chain = rl.RegimeChain([[-20, 15, 5], [10, -30, 20], [8, 12, -20]])  # any generator; rates per year
model = rl.SwitchingHestonModel(chain, S0=100, r=0.02, q=0.0, v0=0.04, kappa=1.5,
                                theta=[0.09, 0.05, 0.02], sigma=0.4, rho=-0.6)   # long-run variance switches
opt = rl.VanillaOption(("call", 100.0), maturity=1.0)
opt.setPricingEngine(rl.FastSwitchingEngine(model, order=4, regime=0))    # expansion in the mean holding time
print(opt.NPV())
opt.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=0))         # the referee: numerical solution
print(opt.NPV())
```

| QuantLib | regimelib | switches |
|---|---|---|
| `Vasicek(r0, a, b, sigma)` | `SwitchingVasicek(chain, r0, a, b, sigma)` | `b`, `sigma` |
| `CoxIngersollRoss(r0, theta, k, sigma)` | `SwitchingCoxIngersollRoss(chain, r0, theta, k, sigma)` | `theta` |
| `BlackScholesMertonProcess` | `SwitchingBlackScholesProcess(chain, S0, r, q, sigma)` | `sigma` |
| `HestonModel` | `SwitchingHestonModel(chain, S0, r, q, v0, kappa, theta, sigma, rho)` | `theta` |
| `Merton76Process` | `SwitchingMerton76Process(chain, S0, r, q, sigma, jumpIntensity, logJumpMean, logJumpVol)` | `sigma`, `jumpIntensity` |
| `BatesModel` | `SwitchingBatesModel(chain, S0, r, q, v0, kappa, theta, sigma, rho, jumpIntensity, logJumpMean, logJumpVol)` | `theta`, `jumpIntensity` |
| `VarianceGammaProcess` | `SwitchingVarianceGammaProcess(chain, S0, r, q, sigma, nu, theta)` | all three |
| `ZeroCouponBond`, `VanillaOption` | same names, `setPricingEngine`, `NPV()` | starting `regime` on the engine |
| `HullWhite(termStructure, a, sigma)` | `SwitchingHullWhite(chain, termStructure, a, sigma)`, curve as a flat rate, a callable or a QuantLib handle | `sigma` |
| `G2(termStructure, a, sigma, b, eta, rho)` | `SwitchingG2(chain, termStructure, a, sigma, b, eta, rho)` | `sigma`, `eta`, `rho` |
| `Vasicek.discountBondOption(type, K, T, S)` | `ZeroCouponBondOption(type, K, T, S)` under `SwitchingVasicek` | `b`, `sigma` |

Instruments accept QuantLib payoff and exercise objects or plain `("call", strike)` tuples. Maturities are years, or QuantLib Dates measured from the evaluation date (Actual/365 unless a `dayCounter` is given); a `VanillaOption` takes its maturity from a QuantLib exercise.

Engines: `FastSwitchingEngine(model, order, regime)` expands in the mean holding time `n / -trace Q` to any order,
with the initial layer; `order=None` adds terms until successive orders agree to `tol` or the series stops improving
(`orderUsed` and `lastIncrement` are set after `NPV()`). The engine raises an `ExpansionWarning` when the last term is
not smaller than the one before it, when it is more than a thousandth of the value, or when the expansion had to be
replaced by the averaged value at a Fourier node that mattered; `instrument._result('diagnostics')` gives the holding
time, the order used, the relative size of the last term and the fallback count; `NumericalSwitchingEngine(model, regime)` solves `a' = (Q + diag g) a` numerically. Vanilla
options use Lewis's formula with a frequency cutoff found from the averaged model's characteristic function.

Engines also include `MonteCarloSwitchingEngine(model, regime, paths, seed)`, a grid-free referee that samples regime paths exactly and prices each path in closed form (Vasicek bonds, Black-Scholes options), with `standardError` set after `NPV()`.

`ZeroCouponBond(T, regimeAtMaturity=j)` pays only if the regime at maturity is j; the parts sum to the plain bond and the expansion carries the order-zero initial layer this needs.

## The first-order tier

Models whose switched parameter multiplies an operator, so the state does not factor out, are priced to first order
in the holding time by `FirstOrderFDEngine(model, regime, n)`: one matrix exponential of the augmented system
`[[L_bar, 0], [K A A, L_bar]]` on a finite-difference grid gives the averaged price and the Duhamel correction, plus
the regime-memory term; `averaged`, `correction` and `memory` are set after `NPV()`. `SwitchingFDReferee` solves
the switching model on the same grid without expansion. `DESIGN.md` gives the rule and the certificates.

| QuantLib | regimelib | switches |
|---|---|---|
| `AnalyticCEVEngine` / CEV process | `SwitchingCEVProcess(chain, S0, r, q, sigma, beta)` | `sigma` |

`SwitchingBlackScholesProcess` also has the operator form, and the finite-difference first-order term agrees with the
exact engine's first-order term there: that check is the certificate for the operator machinery.

Greeks follow QuantLib: `option.delta()`, `gamma()`, `theta()`, `rho()`, and `vega()` (in `v0`) for Heston and Bates;
`bond.delta()` and `gamma()` in `r0` under Vasicek and CIR. They are computed by differentiating the pricing formula,
not by bumping; a greek the engine does not provide raises, as in QuantLib.

Greeks as formulas: `regimelib.symbolic.VasicekTwoStateBond(regime)` builds the closed-form second-order bond price
as a sympy expression and `.greek('r0')`, `.greek('theta1')`, `.greek('lam')`, `.theta()` are its symbolic
derivatives; `.evaluate(expr, **values)` evaluates one. The formula equals the engine at order 2 to 1e-12, and the
symbolic greeks match finite differences of the numerical solution (`tests/test_symbolic.py`).

Roadmap: symbolic closed forms for CIR, jumps, Black-Scholes and n regimes at first order (the explicit pages); Heston with a switched vol-of-vol (two-dimensional grid, frozen limit against `FdHestonVanillaEngine`);
correlated Heston-Hull-White with a switched rate level; two-name credit; options under G2 and Hull-White; greeks in
model parameters from the closed forms.

The engine (`regimelib/_engine/`) is a copy of the certificates' code in
[microprediction/homogenization](https://github.com/microprediction/homogenization); the mathematics is on the site.
