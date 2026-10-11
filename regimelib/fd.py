"""Time-stepping finite differences for the switching model without expansion: early exercise and barriers, which
the characteristic-function engines do not reach. The coupled system u_i' = L_i u_i + sum_j Q_ij u_j is stepped by
Crank-Nicolson after Rannacher's four implicit half-steps; American exercise projects onto the payoff after every
step; a knock-out barrier truncates the grid at the barrier node with the rebate as the Dirichlet value, and a
knock-in is the vanilla less the knock-out (a rebate at expiry for the knock-in is the discounted rebate times the
probability of never hitting, which the same solve gives with a unit payoff). Greeks come from the grid: delta and
gamma by differencing at the spot, theta from the generator applied to the solution."""
import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from .instruments import VanillaOption, BarrierOption, Swaption, CouponBondOption, rejectFeatures
from .instruments import ZeroCouponBondOption, CapFloor
from .information import startingBelief, checkInformation, regimeIsKnown, byBelief, notRevealed


class SwitchingFDEngine:
    supportsResults = True

    def __init__(self, model, regime=0, n=None, steps=400, width=None, stretch=None, information="inferred"):
        """`stretch` concentrates the nodes of a two-dimensional grid around the strike and the starting factor
        (the width of the dense region as a fraction of the interval, 0.1 to 0.3 is usual); None is uniform."""
        import numbers
        if isinstance(steps, bool) or not isinstance(steps, numbers.Integral) or steps < 1:
            raise ValueError(f"steps must be a positive integer number of time steps, got {steps!r}")
        from .firstorder import _gridSize
        self.model, self.n, self.steps, self.width, self.stretch = model, _gridSize(model, n, 1001), int(steps), width, stretch
        self.regime, self.belief = startingBelief(regime, model.n)      # a regime index, or a belief over the regimes
        self.information = checkInformation(information)

    # -- the block operator on a given grid ---------------------------------------------------------------------
    def _system(self, instrument, width):
        m = self.model
        Lbar, As, f, grid, u0, x0 = m.operators(instrument, self.n, width, stretch=self.stretch)
        f = np.atleast_2d(np.asarray(f, float)); pi = m.chain.stationaryDistribution(); nR = m.n
        Q = m.chain.generator; I = sp.identity(grid.n, format="csr")
        blocks = [[None] * nR for _ in range(nR)]
        for i in range(nR):
            Li = Lbar + sum((f[j, i] - f[j] @ pi) * As[j] for j in range(len(As)))
            for k in range(nR):
                blocks[i][k] = Li + Q[i, i] * I if i == k else Q[i, k] * I
        return sp.bmat(blocks, format="csr"), grid, u0, x0, nR

    def _march(self, Big, u, T, project=None, fixed=None, window=None):
        """Rannacher start then Crank-Nicolson; `project` is applied after each step, `fixed` is (indices, value).
        `window` is the longest time to maturity at which exercise is allowed (all of them when None)."""
        N = Big.shape[0]; I = sp.identity(N, format="csr"); dt = T / self.steps
        allowed = lambda tau: project is not None and (window is None or tau <= window + 1e-12)
        if fixed is not None:
            idx, val = fixed
            keep = np.ones(N); keep[idx] = 0.0
            Big = sp.diags(keep) @ Big                                  # Dirichlet rows: u stays at its value
            u = u.copy(); u[idx] = val
        half = full_l = splu((I - 0.5 * dt * Big).tocsc())               # the implicit half step and Crank-Nicolson share it
        full_r = I + 0.5 * dt * Big
        for k in range(self.steps):
            if k < 2:                                                    # two implicit Euler half steps, each projected
                u = half.solve(u)
                if allowed((k + 0.5) * dt):
                    u = np.maximum(u, project)
                if fixed is not None:
                    u[idx] = val
                u = half.solve(u)
            else:
                u = full_l.solve(full_r @ u)
            if allowed((k + 1) * dt):
                u = np.maximum(u, project)
            if fixed is not None:
                u[idx] = val
        return u

    def _greeks(self, grid, v, x0, Big, nR):
        """Spot greeks from the regime block; the grid is in log price for Black-Scholes, in price for CEV. On a
        two-dimensional grid (log S, r) the derivatives are taken along log S at the starting rate."""
        blk = slice(self.regime * grid.n, (self.regime + 1) * grid.n)
        S0 = self.model.S0
        if hasattr(grid, "gx"):                                          # Grid2D: (log S, second factor)
            u2 = v[blk].reshape(len(grid.x), len(grid.v)); j = int(np.argmin(abs(grid.v - x0[1])))
            ux, uxx = _localDerivatives(grid.x, u2[:, j], x0[0])
            theta = -grid.interp((Big @ v)[blk], x0)
            return {"value": grid.interp(v[blk], x0), "delta": ux / S0, "gamma": (uxx - ux) / (S0 * S0), "theta": theta}
        u = v[blk]
        ux, uxx = _localDerivatives(grid.x, u, x0)
        logGrid = abs(x0 - math.log(S0)) < abs(x0 - S0)                  # the state is log S or S
        if logGrid:
            delta, gamma = ux / S0, (uxx - ux) / (S0 * S0)
        else:
            delta, gamma = ux, uxx
        theta = -grid.interp((Big @ v)[blk], x0)                         # dV/dt = -L V in calendar time
        return {"value": grid.interp(u, x0), "delta": delta, "gamma": gamma, "theta": theta}

    def calculate(self, instrument, results=False):
        """Options on bonds and swaps, and early exercise, depend on what is known about the regime. When the path
        reveals it (or the regimes coincide) the regime-by-regime solve below is the price under either information
        setting. When it does not, `information="observed"` uses that solve and `information="inferred"` carries the
        belief as a state variable (two regimes, Vasicek or CIR rate options)."""
        rate = isinstance(instrument, (Swaption, CouponBondOption, ZeroCouponBondOption, CapFloor))
        acts = rate or getattr(instrument, "isAmerican", False)
        exact = self._withoutAGrid(instrument, rate)                     # deterministic models, and G2 with one live factor
        if exact is not None:
            return exact if results else exact["value"]
        if rate and not isinstance(instrument, CapFloor) and _exerciseTerms(instrument)[0][-1] <= 0:
            out = self._exercisedNow(instrument)
            return out if results else out["value"]
        if acts and self.information == "inferred" and not regimeIsKnown(self.model):
            if not rate:
                raise notRevealed("early exercise")
            out = self._inferredRateOption(instrument)
            return out if results else out["value"]
        return self._byRegime(instrument, results)

    def _exercisedNow(self, inst):
        """An option on a bond or a swap whose last exercise date is today: the payoff on today's bonds, with no grid
        (the automatic one has no width over no time). The characteristic-function engine holds that payoff for a
        known regime and for either kind of belief."""
        from .engines import NumericalSwitchingEngine
        start = self.regime if self.belief is None else list(self.belief)
        return {"value": NumericalSwitchingEngine(self.model, regime=start, information=self.information).calculate(inst)}

    def _withoutAGrid(self, instrument, rate):
        """A model with no diffusion has a known path, and a grid in a state that does not move has no width. Price
        those directly. A G2 model with one factor's volatility identically zero is the one-factor Hull-White model
        in the other factor, and is solved as such."""
        from .engines import NumericalSwitchingEngine, _noDiffusion, deterministicEquity
        from .models import SwitchingG2, SwitchingHullWhite, SwitchingBlackScholesProcess
        from .hybrid import SwitchingEquityRates
        from .instruments import ZeroCouponBond
        m = self.model
        start = self.regime if self.belief is None else list(self.belief)
        if rate and _noDiffusion(m) == "switching level":
            from .engines import _atoms
            raise _atoms("an option on a bond or a swap")
        if isinstance(m, SwitchingEquityRates) and isinstance(instrument, VanillaOption):
            R = m.rates
            if np.all(np.asarray(R.sigma, float) == 0.0):               # the rate has no noise: is its path constant?
                level = set(getattr(R, "b", [None]))
                constant = hasattr(R, "r0") and hasattr(R, "b") and (R.a == 0.0 or level == {R.r0})
                if not constant or not isinstance(m.equity, SwitchingBlackScholesProcess):
                    raise NotImplementedError("the equity-with-rates grid needs a rate that diffuses; with a "
                                              "deterministic rate that is not constant this case is not priced")
                flat = SwitchingBlackScholesProcess(m.chain, m.S0, R.r0, m.q, m.equity.sigma)
                n = self.n[0] if isinstance(self.n, tuple) else self.n
                engine = SwitchingFDEngine(flat, regime=start, n=n, steps=self.steps, information=self.information)
                return engine.calculate(instrument, results=True)       # Black-Scholes at the constant rate
        if _noDiffusion(m) == "deterministic":
            if rate:
                reduced = NumericalSwitchingEngine(m, regime=start, information="observed")
                bond = lambda t: 1.0 if t == 0 else reduced.calculate(ZeroCouponBond(t))
                if isinstance(instrument, Swaption) and instrument.exerciseTimes:      # exercise on the best of the known dates
                    call, K = not instrument.isPayer, instrument.notional
                    values = []
                    for t in instrument.exerciseTimes:
                        swap = sum(c * bond(S) for S, c in instrument.cashflows if S > t + 1e-12) - K * bond(t)
                        values.append(max(swap if call else -swap, 0.0))
                    return {"value": max(values)}
                return {"value": reduced.calculate(instrument)}
            if isinstance(instrument, VanillaOption) and not isinstance(m, SwitchingEquityRates):
                rejectFeatures(instrument, "the switching finite-difference engine", ("geometric averaging",),
                               "NumericalSwitchingEngine or FastSwitchingEngine")
                return {"value": deterministicEquity(m, instrument)}
        if rate and isinstance(m, SwitchingG2):
            for axis, (speed, vols, dead) in enumerate(((m.a, m.sigma, m.eta), (m.b, m.eta, m.sigma))):
                if all(v == 0.0 for v in dead) and any(v != 0.0 for v in vols):
                    one = SwitchingHullWhite(m.chain, m.discount, speed, vols)
                    n = self.n[axis] if isinstance(self.n, tuple) else self.n          # the live factor's own controls
                    width = self.width[axis] if isinstance(self.width, tuple) else self.width
                    engine = SwitchingFDEngine(one, regime=start, n=n, steps=self.steps, width=width, information=self.information)
                    return engine.calculate(instrument, results=True)
        return None

    @byBelief()
    def _byRegime(self, instrument, results=False):
        if isinstance(instrument, ZeroCouponBondOption):
            instrument = CouponBondOption("call" if instrument.isCall else "put", instrument.strike, instrument.maturity,
                                          [(instrument.bondMaturity, 1.0)])
        if isinstance(instrument, CapFloor):
            out = {"value": sum(w * self._rateOption(o)["value"] for w, o in _caplets(instrument))}
            return out if results else out["value"]
        if isinstance(instrument, (Swaption, CouponBondOption)):
            out = self._rateOption(instrument)
            return out if results else out["value"]
        if not isinstance(instrument, VanillaOption):
            raise TypeError("the switching finite-difference engine prices vanilla, American, barrier and Bermudan instruments")
        rejectFeatures(instrument, "the switching finite-difference engine", ("geometric averaging",),
                       "NumericalSwitchingEngine or FastSwitchingEngine")
        if isinstance(instrument, BarrierOption):
            out = self._barrier(instrument)
        elif instrument.maturity == 0:                                   # no grid: the automatic one has no width at T = 0
            from .engines import payoffNow
            out = payoffNow(self.model, instrument)
        else:
            Big, grid, u0, x0, nR = self._system(instrument, self.width)
            T = instrument.maturity
            v = self._march(Big, np.tile(u0, nR), T, project=np.tile(u0, nR) if instrument.isAmerican else None,
                            window=T - getattr(instrument, "earliestExercise", 0.0))
            out = self._greeks(grid, v, x0, Big, nR)
            if instrument.isAmerican and getattr(instrument, "earliestExercise", 0.0) <= 0.0:
                held = float(instrument.payoffOnGrid([self.model.S0])[0])
                if held > 0 and out["value"] <= held * (1 + 1e-9):       # exercised now: the value is the payoff,
                    out["theta"] = 0.0                                   # which does not change with the date
        return out if results else out["value"]

    def _barrier(self, opt):
        m, T = self.model, opt.maturity
        if hasattr(m, "equity") and hasattr(m, "rates"):
            # the knock-out grid ends at the barrier in log S for every rate, a boundary this routine does not build
            raise NotImplementedError("barriers are not priced on the (log S, r) grid of an equity with stochastic rates; "
                                      "with a deterministic rate (zero rate volatility, one level) they are")
        logGrid = hasattr(m, "sigma") and not hasattr(m, "beta")
        b = math.log(opt.barrier) if logGrid else opt.barrier
        x0 = math.log(m.S0) if logGrid else m.S0
        if (opt.isUp and x0 >= b) or (not opt.isUp and x0 <= b):
            raise ValueError("the spot is beyond the barrier")
        if opt.isAmerican and not opt.isKnockOut:
            raise NotImplementedError("an American knock-in is not priced: vanilla less knock-out is a European identity. "
                                      "Price the knock-out, or the European knock-in.")
        # the vanilla on the default grid, and the knock-out on a grid truncated at the barrier node
        Big, grid, u0, _, nR = self._system(opt, self.width)
        L = (grid.x[-1] - grid.x[0]) / 2
        ends = (b, b + 2 * L) if not opt.isUp else (b - 2 * L, b)
        if not ends[0] < x0 < ends[1]:
            raise ValueError("the barrier is farther from the spot than the grid is wide, so the grid that ends at the "
                             "barrier does not hold the spot. The barrier is then all but unreachable: price the vanilla "
                             "(a knock-out) or nothing (a knock-in), or pass a larger width.")
        BigB, gridB, u0B, _, _ = self._system(opt, ends)
        bnode = 0 if not opt.isUp else gridB.n - 1
        idx = np.array([r * gridB.n + bnode for r in range(nR)])
        # a knock-out pays its rebate at the hit; a knock-in is the vanilla less the knock-out with no rebate, plus its
        # own rebate, paid at expiry if the barrier was never touched
        atHit = opt.rebate if opt.isKnockOut else 0.0
        vOut = self._march(BigB, np.tile(u0B, nR), T, project=np.tile(u0B, nR) if opt.isAmerican else None, fixed=(idx, atHit),
                           window=T - getattr(opt, "earliestExercise", 0.0) if opt.isAmerican else None)
        res = self._greeks(gridB, vOut, x0, BigB, nR)
        if opt.isKnockOut:
            return res
        vVan = self._march(Big, np.tile(u0, nR), T)
        van = self._greeks(grid, vVan, x0, Big, nR)
        out = {k: van[k] - res[k] for k in van}
        out["value"] = max(out["value"], 0.0)                            # the difference of two grids, far from the barrier
        if opt.rebate:                                                   # the discounted probability of never hitting
            vS = self._march(BigB, np.ones(nR * gridB.n), T, fixed=(idx, 0.0))
            never = self._greeks(gridB, vS, x0, BigB, nR)
            out = {k: out[k] + opt.rebate * never[k] for k in out}
        return out

    # -- options on coupon bonds and swaps, European or Bermudan, on a short-rate grid ----------------------------
    def _rateOption(self, inst):
        m = self.model
        if not hasattr(m, "bondOnGrid"):
            raise TypeError("Bermudan and finite-difference rate options need a short-rate model with a grid (SwitchingVasicek, SwitchingHullWhite, SwitchingCoxIngersollRoss, SwitchingG2)")
        exercises, isCall, K = _exerciseTerms(inst)
        T_end = exercises[-1]
        if T_end <= 0:                                                   # a caplet that fixes today
            return self._exercisedNow(inst)
        Big, grid, _, r0, nR = self._system(inst, self.width)
        def exerciseValue(t):
            """Per regime: the bond of the remaining cash flows less the strike (call) on the rate grid, expressed in
            the grid's numeraire: the grid discounts with the factor only, the fitted drift's part is deterministic."""
            bond = sum(c * m.bondOnGrid(grid, t, S) for S, c in inst.cashflows if S > t + 1e-12)
            ex = np.maximum(bond - K, 0.0) if isCall else np.maximum(K - bond, 0.0)
            return ex / m.deterministicDiscount(t, T_end)
        steps = self.steps
        u = exerciseValue(T_end).ravel()
        t_hi = T_end
        for t_lo in reversed(exercises[:-1]):                            # each earlier exercise date, time zero included
            self.steps = max(4, int(round(steps * (t_hi - t_lo) / T_end)))
            u = self._march(Big, u, t_hi - t_lo)
            u = np.maximum(u, exerciseValue(t_lo).ravel())
            t_hi = t_lo
        if t_hi > 0:                                                     # and from the first exercise date to today
            self.steps = max(4, int(round(steps * t_hi / T_end)))
            u = self._march(Big, u, t_hi)
        self.steps = steps
        blk = slice(self.regime * grid.n, (self.regime + 1) * grid.n)
        return {"value": m.deterministicDiscount(0.0, T_end) * grid.interp(u[blk], r0)}

    # -- the same options when the regime is inferred: the belief is a state variable -----------------------------
    def _inferredRateOption(self, inst):
        """Two regimes that differ only in the level the rate reverts to. Nobody sees the regime; everybody sees the
        rate, and the belief p = P(regime 0 | the rate's path) moves only when the rate surprises:

            dr = a (b(p) - r) dt + s(r) dW,        b(p) = p b_0 + (1 - p) b_1,
            dp = (q_10 (1 - p) - q_01 p) dt + p (1 - p) a (b_0 - b_1) / s(r) dW,

        one Brownian motion for both (the innovation of the filter), s(r) = sigma for Vasicek and sigma sqrt(r) for
        CIR. The bond at (r, p) is the belief-weighted bond of the two regimes, so the exercise value is a function of
        (r, p), and the option solves one equation on the (r, p) grid, with no regime blocks."""
        from .firstorder import Grid2D
        from .models import SwitchingVasicek, SwitchingCoxIngersollRoss
        m = self.model
        if m.n != 2 or not isinstance(m, (SwitchingVasicek, SwitchingCoxIngersollRoss)):
            raise notRevealed("the price of an option on a bond or a swap")
        if isinstance(inst, CapFloor):
            return {"value": sum(w * self._inferredRateOption(o)["value"] for w, o in _caplets(inst))}
        if isinstance(inst, ZeroCouponBondOption):
            inst = CouponBondOption("call" if inst.isCall else "put", inst.strike, inst.maturity, [(inst.bondMaturity, 1.0)])
        exercises, isCall, K = _exerciseTerms(inst)
        T_end = exercises[-1]
        if T_end <= 0:
            return self._exercisedNow(inst)
        nr, npts = self.n if isinstance(self.n, tuple) else (min(int(self.n), 301), 41)
        _, _, _, rgrid, _, r0 = m.operators(inst, nr, self.width if not isinstance(self.width, tuple) or len(self.width) == 2 else None)
        grid = Grid2D(rgrid.x[0], rgrid.x[-1], nr, 0.0, 1.0, npts)
        R, P = grid.X, grid.V
        Q = m.chain.generator; q01, q10 = Q[0, 1], Q[1, 0]
        if isinstance(m, SwitchingVasicek):
            a, (b0, b1), sig = m.a, m.b, m.sigma[0]
            var = np.full_like(R, sig * sig)
        else:
            a, (b0, b1), sig = m.k, m.theta, m.sigma
            var = sig * sig * np.maximum(R, 0.0)
        level = P * b0 + (1.0 - P) * b1
        cross = P * (1.0 - P) * a * (b0 - b1)                             # s(r) times the belief's loading on dW
        floor = max(rgrid.x[1] - rgrid.x[0], 1e-12) * sig * sig           # CIR: the noise vanishes at r = 0 and the
        gamma2 = cross * cross / np.maximum(var, floor)                   # belief would learn at once; bounded here
        L = (sp.diags(a * (level - R)) @ grid.d1x + sp.diags(0.5 * var) @ grid.d2x
             + sp.diags(q10 * (1.0 - P) - q01 * P) @ grid.d1v + sp.diags(0.5 * gamma2) @ grid.d2v
             + sp.diags(cross) @ grid.d1xv - sp.diags(R)).tocsr()

        def value(t):
            """The swap or bond less the strike, at the belief-weighted bond; signed, so exercise compares like with like."""
            bond = sum(c * m.bondOnGrid(rgrid, t, S) for S, c in inst.cashflows if S > t + 1e-12)      # regimes x rates
            mixed = (np.outer(bond[0], grid.v) + np.outer(bond[1], 1.0 - grid.v)).ravel()
            return (mixed - K) if isCall else (K - mixed)
        steps = self.steps
        u = np.maximum(value(T_end), 0.0)
        t_hi = T_end
        for t_lo in reversed(exercises[:-1]):
            self.steps = max(4, int(round(steps * (t_hi - t_lo) / T_end)))
            u = self._march(L, u, t_hi - t_lo)
            u = np.maximum(u, value(t_lo))
            t_hi = t_lo
        if t_hi > 0:
            self.steps = max(4, int(round(steps * t_hi / T_end)))
            u = self._march(L, u, t_hi)
        self.steps = steps
        p0 = float(self.belief[0]) if self.belief is not None else (1.0 if self.regime == 0 else 0.0)
        return {"value": grid.interp(u, (r0, p0))}


def _localDerivatives(x, u, x0):
    """First and second derivative at x0 of the parabola through the three nodes around it. The three are interior to
    the grid, so next to a barrier (an end of the grid) the stencil is one-sided, and unequal spacing is handled."""
    i = int(np.clip(np.searchsorted(x, x0), 1, len(x) - 2))
    if i > 1 and abs(x[i - 1] - x0) < abs(x[i] - x0):
        i -= 1
    (a, b, c), (fa, fb, fc) = x[i - 1:i + 2], u[i - 1:i + 2]
    la, lb, lc = fa / ((a - b) * (a - c)), fb / ((b - a) * (b - c)), fc / ((c - a) * (c - b))
    ux = la * (2 * x0 - b - c) + lb * (2 * x0 - a - c) + lc * (2 * x0 - a - b)
    return float(ux), float(2 * (la + lb + lc))


def _exerciseTerms(inst):
    """(exercise times, whether it is a call on the bond, strike) for a swaption or an option on a coupon bond."""
    if isinstance(inst, Swaption):
        return (inst.exerciseTimes or [inst.maturity]), not inst.isPayer, inst.notional
    return [inst.maturity], inst.isCall, inst.strike


def _caplets(cap):
    """A cap or floor as weighted options on zero-coupon bonds: (1 + tau K) puts struck at 1 / (1 + tau K) per caplet."""
    out = []
    for T0, T1 in zip(cap.times[:-1], cap.times[1:]):
        tau = T1 - T0; kb = 1.0 / (1.0 + tau * cap.strike)
        option = CouponBondOption("put" if cap.isCap else "call", kb, T0, [(T1, 1.0)])
        out.append((cap.notional * (1.0 + tau * cap.strike), option))
    return out
