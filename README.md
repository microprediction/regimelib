# regimelib

QuantLib's models with a hidden Markov regime, priced by the fast-switching expansion of
[homogenization.microprediction.org](https://homogenization.microprediction.org).

Each class mirrors a QuantLib class and takes the same parameters. A parameter that switches with the regime is
given as a list, one entry per regime. With every regime equal, each engine reproduces the QuantLib engine it
mirrors; that is a test (`tests/test_quantlib_limit.py`), checked to about 1e-9.

```python
import regimelib as rl
chain = rl.RegimeChain.twoState(20.0, 30.0)          # switching rates, per year
model = rl.SwitchingHestonModel(chain, S0=100, r=0.02, q=0.0, v0=0.04, kappa=1.5,
                                theta=[0.09, 0.02], sigma=0.4, rho=-0.6)   # long-run variance switches
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
| `Vasicek.discountBondOption(type, K, T, S)` | `ZeroCouponBondOption(type, K, T, S)` under `SwitchingVasicek` | `b`, `sigma` |

Instruments accept QuantLib payoff and exercise objects or plain `("call", strike)` tuples; times are in years.

Engines: `FastSwitchingEngine(model, order, regime)` expands in the mean holding time `n / -trace Q` to any order,
with the initial layer; `NumericalSwitchingEngine(model, regime)` solves `a' = (Q + diag g) a` numerically. Vanilla
options use Lewis's formula with a frequency cutoff found from the averaged model's characteristic function.

Roadmap: two-factor
Gaussian and credit models; QuantLib dates and day counters; Monte Carlo referee with exact regime paths.

The engine (`regimelib/_engine/`) is a copy of the certificates' code in
[microprediction/homogenization](https://github.com/microprediction/homogenization); the mathematics is on the site.
