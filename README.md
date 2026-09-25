# regimelib

QuantLib's models with a hidden Markov regime, priced by the fast-switching expansion of
[homogenization.microprediction.org](https://homogenization.microprediction.org).

Documentation: [regimelib.microprediction.org](https://regimelib.microprediction.org) (QuantLib-Python's layout: instruments, engines, models, helpers, examples).

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
| (none: no jump short-rate model in QuantLib) | `SwitchingVasicekJumps(chain, r0, a, b, sigma, jumpIntensity, jumpMean)`, exponential jumps; frozen limit against the affine closed form | `b`, `sigma`, `jumpIntensity` |
| `CoxIngersollRoss(r0, theta, k, sigma)` | `SwitchingCoxIngersollRoss(chain, r0, theta, k, sigma)` | `theta` |
| `BlackScholesMertonProcess` | `SwitchingBlackScholesProcess(chain, S0, r, q, sigma)` | `sigma` |
| `HestonModel` | `SwitchingHestonModel(chain, S0, r, q, v0, kappa, theta, sigma, rho)` | `theta` |
| `Merton76Process` | `SwitchingMerton76Process(chain, S0, r, q, sigma, jumpIntensity, logJumpMean, logJumpVol)` | `sigma`, `jumpIntensity` |
| `BatesModel` | `SwitchingBatesModel(chain, S0, r, q, v0, kappa, theta, sigma, rho, jumpIntensity, logJumpMean, logJumpVol)` | `theta`, `jumpIntensity` |
| `VarianceGammaProcess` | `SwitchingVarianceGammaProcess(chain, S0, r, q, sigma, nu, theta)` | all three |
| `HybridHestonHullWhiteProcess` + `AnalyticHestonHullWhiteEngine`, `AnalyticBSMHullWhiteEngine` | `SwitchingEquityRates(equity, rates)`: Black–Scholes or Heston with Vasicek or Hull–White on one chain, equity–rate correlation `rho` for the Black–Scholes equity; Lewis with the discounted characteristic function; American options on the (log S, r) grid | whatever the two parts switch |
| `ZeroCouponBond`, `VanillaOption` | same names, `setPricingEngine`, `NPV()` | starting `regime` on the engine |
| `HullWhite(termStructure, a, sigma)` | `SwitchingHullWhite(chain, termStructure, a, sigma)`, curve as a flat rate, a callable or a QuantLib handle | `sigma` |
| `G2(termStructure, a, sigma, b, eta, rho)` | `SwitchingG2(chain, termStructure, a, sigma, b, eta, rho)` | `sigma`, `eta`, `rho` |
| `Vasicek.discountBondOption(type, K, T, S)`, `G2.discountBondOption` | `ZeroCouponBondOption(type, K, T, S)` under `SwitchingVasicek`, `SwitchingHullWhite` or `SwitchingG2` (caps and floors likewise) | `b`, `sigma`; `sigma`, `eta`, `rho` |
| `CEVProcess` | `SwitchingCEVProcess(chain, S0, r, q, sigma, beta)` (first-order tier) | `sigma` |
| `HestonProcess` with `sigma` (vol-of-vol) switching | `SwitchingHestonVolOfVol(chain, S0, r, q, v0, kappa, theta, xi, rho)` (first-order tier, 2-D grid; frozen limit against `AnalyticHestonEngine`) | `xi` |

Instruments, with the QuantLib engine that the frozen limit is checked against:

| QuantLib | regimelib | engine |
|---|---|---|
| `ZeroCouponBond`, `Bond` | `ZeroCouponBond(T)`, `CouponBond(cashflows)` | `FastSwitchingEngine`, `NumericalSwitchingEngine` |
| `VanillaOption` + `AnalyticEuropeanEngine`, `AnalyticHestonEngine`, ... | `VanillaOption(payoff, exercise)` with plain, cash-or-nothing and asset-or-nothing payoffs, `impliedVolatility()` | `FastSwitchingEngine`, `NumericalSwitchingEngine` (Lewis) |
| `VanillaOption` + `FdBlackScholesVanillaEngine` (American) | `VanillaOption(payoff, AmericanExercise)` | `SwitchingFDEngine`: Rannacher then Crank–Nicolson on the coupled regime system, projection each step |
| `BarrierOption` + `AnalyticBarrierEngine` | `BarrierOption(type, barrier, rebate, payoff, exercise)` | `SwitchingFDEngine`: grid truncated at the barrier, knock-in = vanilla − knock-out |
| `ContinuousAveragingAsianOption(Geometric)` + `AnalyticContinuousGeometricAveragePriceAsianEngine` | `ContinuousGeometricAsianOption(payoff, exercise)` | the time average of the log price is a quadratic forcing in time to maturity; `FastSwitchingEngine`, `NumericalSwitchingEngine` |
| `Swaption` + `JamshidianSwaptionEngine` | `Swaption(kind, expiry, fixedTimes, fixedRate, notional)`, `CouponBondOption(kind, K, T, cashflows)` | Jamshidian's decomposition conditioned on the regime at expiry |
| `Swaption` + `BermudanExercise` + `TreeSwaptionEngine` | `Swaption(..., exerciseTimes=[...])` or a QuantLib `BermudanExercise` | `SwitchingFDEngine` on a short-rate grid (`SwitchingVasicek`, `SwitchingCoxIngersollRoss`, `SwitchingHullWhite` in its zero-mean factor, `SwitchingG2` on its two-factor grid against `G2SwaptionEngine` and `FdG2SwaptionEngine`): exercise value per regime from the switching bond formula; frozen limit against `FdHullWhiteSwaptionEngine` |
| `Cap`/`Floor` + `AnalyticCapFloorEngine` | `CapFloor(kind, times, strike, notional)` | caplet = (1 + τK) × put on the zero-coupon bond |
| `NthToDefault` (first) | `SwitchingIntensityBasket([lambda1, lambda2, ...])` on one chain; `FirstToDefaultSwap`, `defaultCorrelation(t)` | joint survival = the bond of the summed forcing; the common regime is the dependence |
| `CreditDefaultSwap` + `MidPointCdsEngine` | `CreditDefaultSwap(side, spread, times, recovery, discount)` on a switching CIR or Vasicek intensity; `fairSpread()`, `couponLegNPV()`, `defaultLegNPV()` | survival = the intensity model's bond price; mid-point protection |
| `HestonModelHelper`, `model.calibrate(helpers, ...)` | `VolatilityHelper(T, K, vol)`, `calibrate(model, helpers, ["sigma", "chain"])` | least squares on relative price (or implied vol) errors; terminal vectors shared across strikes |

Instruments accept QuantLib payoff and exercise objects or plain `("call", strike)` tuples. Maturities are years, or QuantLib Dates measured from the evaluation date (Actual/365 unless a `dayCounter` is given); a `VanillaOption` takes its maturity from a QuantLib exercise.

Engines: `FastSwitchingEngine(model, order, regime)` expands in the mean holding time `n / -trace Q` to any order,
with the initial layer; `order=None` adds terms until successive orders agree to `tol` or the series stops improving
(`orderUsed` and `lastIncrement` are set after `NPV()`). The engine raises an `ExpansionWarning` when the last term is
not smaller than the one before it or when it is more than a thousandth of the value. The series is asymptotic in the
holding time times the forcing, so at Fourier nodes of high frequency it diverges; there the engine solves the reduced
system numerically instead and says so in a warning when those nodes carry weight. `instrument._result('diagnostics')`
gives the holding time, the order used, the relative size of the last term and the count of numerical nodes.
`NumericalSwitchingEngine(model, regime)` solves `a' = (Q + diag g) a` without expansion (the matrix exponential for
constant forcing, an ODE solver otherwise), memoising the terminal vectors across strikes. Vanilla options use Lewis's
formula with a frequency cutoff found from the averaged model's characteristic function.

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
`regimelib.symbolic.VasicekBondFirstOrder`, `CIRBondFirstOrder` and `VasicekJumpsBondFirstOrder` give the first-order bond under any finite chain (Green–Kubo
and memory coefficients from `coefficients(chain, ...)`, the CIR integrals closed by the Riccati identity); `parameterGreek(chain, wrt, ...)` gives dP/dθᵢ, dP/dσᵢ, dP/dλᵢ and dP/dQ_ab exactly by the chain rule through the coefficients. `regimelib.symbolic.TwoStateConstantForcing` is the exact, all-orders solution of the two-regime reduced system with
constant forcing (a 2 × 2 matrix exponential written with its eigenvalues), which is the closed-form characteristic
function of every two-regime Black–Scholes, Merton or variance-gamma model: `.a(regime)` in `q12, q21, g1, g2, T`
and `.blackScholes(regime)` in `u, sigma1, sigma2`; the formula agrees with the numerical solution to 1e-12.

Roadmap: a non-uniform grid for the two-factor engines; parameter greeks for the option formulas.

Tests: `pytest` runs every certificate (about nine minutes); `pytest -m 'not slow'` skips the four slowest and runs in about two.

The engine (`regimelib/_engine/`) is a copy of the certificates' code in
[microprediction/homogenization](https://github.com/microprediction/homogenization); the mathematics is on the site.
