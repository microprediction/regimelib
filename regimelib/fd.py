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
from .instruments import VanillaOption, BarrierOption


class SwitchingFDEngine:
    supportsResults = True

    def __init__(self, model, regime=0, n=1001, steps=400, width=None):
        self.model, self.regime, self.n, self.steps, self.width = model, regime, n, steps, width

    # -- the block operator on a given grid ---------------------------------------------------------------------
    def _system(self, instrument, width):
        m = self.model
        Lbar, As, f, grid, u0, x0 = m.operators(instrument, self.n, width)
        f = np.atleast_2d(np.asarray(f, float)); pi = m.chain.stationaryDistribution(); nR = m.n
        Q = m.chain.generator; I = sp.identity(grid.n, format="csr")
        blocks = [[None] * nR for _ in range(nR)]
        for i in range(nR):
            Li = Lbar + sum((f[j, i] - f[j] @ pi) * As[j] for j in range(len(As)))
            for k in range(nR):
                blocks[i][k] = Li + Q[i, i] * I if i == k else Q[i, k] * I
        return sp.bmat(blocks, format="csr"), grid, u0, x0, nR

    def _march(self, Big, u, T, project=None, fixed=None):
        """Rannacher start then Crank-Nicolson; `project` is applied after each step, `fixed` is (indices, value)."""
        N = Big.shape[0]; I = sp.identity(N, format="csr"); dt = T / self.steps
        if fixed is not None:
            idx, val = fixed
            keep = np.ones(N); keep[idx] = 0.0
            Big = sp.diags(keep) @ Big                                  # Dirichlet rows: u stays at its value
            u = u.copy(); u[idx] = val
        half = splu((I - 0.5 * dt * Big).tocsc()); full_l = splu((I - 0.5 * dt * Big).tocsc()); full_r = I + 0.5 * dt * Big
        for k in range(self.steps):
            if k < 2:                                                    # two implicit Euler half steps
                u = half.solve(u); u = half.solve(u)
            else:
                u = full_l.solve(full_r @ u)
            if project is not None:
                u = np.maximum(u, project)
            if fixed is not None:
                u[idx] = val
        return u

    def _greeks(self, grid, v, x0, Big, nR):
        """Spot greeks from the regime block; the grid is in log price for Black-Scholes, in price for CEV."""
        blk = slice(self.regime * grid.n, (self.regime + 1) * grid.n)
        u = v[blk]; h = grid.h; i = int(np.argmin(abs(grid.x - x0)))
        ux = (u[i + 1] - u[i - 1]) / (2 * h); uxx = (u[i + 1] - 2 * u[i] + u[i - 1]) / (h * h)
        S0 = self.model.S0; logGrid = abs(math.exp(grid.x[i]) - S0) < abs(grid.x[i] - S0)
        if logGrid:
            delta, gamma = ux / S0, (uxx - ux) / (S0 * S0)
        else:
            delta, gamma = ux, uxx
        theta = -(Big @ v)[blk][i]                                       # dV/dt = -L V in calendar time
        return {"value": grid.interp(u, x0), "delta": delta, "gamma": gamma, "theta": theta}

    def calculate(self, instrument, results=False):
        if not isinstance(instrument, VanillaOption):
            raise TypeError("the switching finite-difference engine prices vanilla, American and barrier options")
        if isinstance(instrument, BarrierOption):
            out = self._barrier(instrument)
        else:
            Big, grid, u0, x0, nR = self._system(instrument, self.width)
            v = self._march(Big, np.tile(u0, nR), instrument.maturity, project=np.tile(u0, nR) if instrument.isAmerican else None)
            out = self._greeks(grid, v, x0, Big, nR)
        return out if results else out["value"]

    def _barrier(self, opt):
        m, T = self.model, opt.maturity
        logGrid = hasattr(m, "sigma") and not hasattr(m, "beta")
        b = math.log(opt.barrier) if logGrid else opt.barrier
        x0 = math.log(m.S0) if logGrid else m.S0
        if (opt.isUp and x0 >= b) or (not opt.isUp and x0 <= b):
            raise ValueError("the spot is beyond the barrier")
        # the vanilla on the default grid, and the knock-out on a grid truncated at the barrier node
        Big, grid, u0, _, nR = self._system(opt, self.width)
        L = (grid.x[-1] - grid.x[0]) / 2
        ends = (b, b + 2 * L) if not opt.isUp else (b - 2 * L, b)
        BigB, gridB, u0B, _, _ = self._system(opt, ends)
        bnode = 0 if not opt.isUp else gridB.n - 1
        idx = np.array([r * gridB.n + bnode for r in range(nR)])
        vOut = self._march(BigB, np.tile(u0B, nR), T, project=np.tile(u0B, nR) if opt.isAmerican else None, fixed=(idx, opt.rebate))
        res = self._greeks(gridB, vOut, x0, BigB, nR)
        if opt.isKnockOut:
            return res
        vVan = self._march(Big, np.tile(u0, nR), T)
        van = self._greeks(grid, vVan, x0, Big, nR)
        # rebate at expiry if never knocked in: solve for the survival value with a unit terminal payoff and zero at the barrier
        reb = 0.0
        if opt.rebate:
            vS = self._march(BigB, np.ones(nR * gridB.n), T, fixed=(idx, 0.0))
            reb = opt.rebate * gridB.interp(vS[self.regime * gridB.n:(self.regime + 1) * gridB.n], x0)
        return {k: van[k] - res[k] + (reb if k == "value" else 0.0) for k in van}
