"""The first-order tier: models whose switched parameter multiplies an operator, so the state does not factor out.

u_t = (L_bar + sum_j f~_j(y) A_j) u. To first order in the holding time,
    u(T) = e^{T L_bar} u0 + int_0^T e^{(T - s) L_bar} (sum_jk K_jk A_j A_k) e^{s L_bar} u0 ds + memory,
K the Green-Kubo matrix of the chain (integral of the autocovariance of the coefficients) and the memory term
-(Q# f~)_i . A e^{T L_bar} u0 for a start in regime i. The Duhamel integral is the second block of one matrix
exponential of [[L_bar, 0], [K A A, L_bar]] applied to (u0, 0). Finite differences on a grid the model supplies."""
import math
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from .instruments import VanillaOption


def green_kubo(chain, f):
    """K_jk = int_0^inf Cov(f_j(y_0), f_k(y_t)) dt = -pi . (f~_j (Q# f~_k)); also returns Q# f~ for the memory term."""
    Q = chain.generator; n = Q.shape[0]; pi = chain.stationaryDistribution()
    f = np.atleast_2d(np.asarray(f, float))                    # rows: coefficients j, columns: regimes
    ft = f - (f @ pi)[:, None]
    Qs = np.linalg.inv(Q - np.outer(np.ones(n), pi)) + np.outer(np.ones(n), pi)   # group inverse of Q
    M = np.array([Qs @ ft[k] for k in range(len(f))])           # Q# f~_k
    K = -np.array([[pi @ (ft[j] * M[k]) for k in range(len(f))] for j in range(len(f))])
    return K, M


def green_kubo_derivatives(chain, f, wrt):
    """Exact derivatives of the stationary distribution, the Green-Kubo matrix and the memory vectors with respect
    to one parameter: wrt = ("q", a, b) for the switching rate Q_ab (Q_aa moves with it) or ("f", j, i) for the
    forcing coefficient f_j in regime i. Returns (dpi, dK, dM). For a rate the group inverse varies as
    dQ# = -Q# dQ Q# + 1 pi dQ (Q#)^2 + (Q#)^2 dQ 1 pi and pi as dpi = -pi dQ Q#."""
    Q = chain.generator; n = Q.shape[0]; pi = chain.stationaryDistribution()
    f = np.atleast_2d(np.asarray(f, float)); ft = f - (f @ pi)[:, None]
    Qs = np.linalg.inv(Q - np.outer(np.ones(n), pi)) + np.outer(np.ones(n), pi)
    M = np.array([Qs @ ft[k] for k in range(len(f))])
    if wrt[0] == "q":
        a, b = wrt[1], wrt[2]
        dQ = np.zeros((n, n)); dQ[a, b] += 1.0; dQ[a, a] -= 1.0
        dpi = -pi @ dQ @ Qs
        P1 = np.outer(np.ones(n), pi)
        dQs = -Qs @ dQ @ Qs + P1 @ dQ @ (Qs @ Qs) + (Qs @ Qs) @ dQ @ P1
        dft = -np.outer(f @ dpi, np.ones(n))
    else:
        j, i = wrt[1], wrt[2]
        dpi = np.zeros(n); dQs = np.zeros((n, n))
        dft = np.zeros_like(f); dft[j] = np.eye(n)[i] - pi[i]
    dM = np.array([dQs @ ft[k] + Qs @ dft[k] for k in range(len(f))])
    dK = -np.array([[dpi @ (ft[j] * M[k]) + pi @ (dft[j] * M[k]) + pi @ (ft[j] * dM[k]) for k in range(len(f))]
                    for j in range(len(f))])
    return dpi, dK, dM


class Grid1D:
    def __init__(self, lo, hi, n):
        self.x = np.linspace(lo, hi, n); self.h = self.x[1] - self.x[0]; self.n = n

    def d1(self):
        n, h = self.n, self.h
        D = sp.diags([-np.ones(n - 1), np.ones(n - 1)], [-1, 1], shape=(n, n), format="lil") / (2 * h)
        D[0, :3] = [-3, 4, -1]; D[0, :3] = D[0, :3] / (2 * h); D[-1, -3:] = [1, -4, 3]; D[-1, -3:] = D[-1, -3:] / (2 * h)
        return D.tocsr()

    def d2(self):
        n, h = self.n, self.h
        D = sp.diags([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)], [-1, 0, 1], shape=(n, n), format="lil") / (h * h)
        D[0, :] = 0; D[-1, :] = 0                                # second derivative zero at the ends
        return D.tocsr()

    def interp(self, u, x0):
        return float(np.interp(x0, self.x, u))


class Grid2D:
    """Tensor grid (x, v); node index ix * nv + iv. d1x, d2x, d1v, d2v are the one-dimensional operators lifted."""
    def __init__(self, xlo, xhi, nx, vlo, vhi, nv):
        self.gx, self.gv = Grid1D(xlo, xhi, nx), Grid1D(vlo, vhi, nv)
        self.x, self.v, self.n = self.gx.x, self.gv.x, nx * nv
        Ix, Iv = sp.identity(nx, format="csr"), sp.identity(nv, format="csr")
        self.d1x, self.d2x = sp.kron(self.gx.d1(), Iv, format="csr"), sp.kron(self.gx.d2(), Iv, format="csr")
        self.d1v, self.d2v = sp.kron(Ix, self.gv.d1(), format="csr"), sp.kron(Ix, self.gv.d2(), format="csr")
        self.d1xv = sp.kron(self.gx.d1(), self.gv.d1(), format="csr")
        self.X, self.V = np.repeat(self.x, nv), np.tile(self.v, nx)

    def interp(self, u, p):
        from scipy.interpolate import RegularGridInterpolator
        f = RegularGridInterpolator((self.x, self.v), np.asarray(u).reshape(len(self.x), len(self.v)))
        return float(f([p])[0])


class FirstOrderFDEngine:
    """First-order pricing on a grid. The model supplies operators(grid) -> (L_bar, [A_j], [f_j per regime], grid,
    payoff(grid), x0)."""
    def __init__(self, model, regime=0, n=801, width=None, warnAbove=0.03):
        self.model, self.regime, self.n, self.width = model, regime, n, width
        self.averaged = self.correction = self.memory = None
        self.warnAbove = warnAbove; self.diagnostics = {}

    def calculate(self, instrument):
        if not isinstance(instrument, VanillaOption):
            raise TypeError("the first-order finite-difference engine prices vanilla options")
        m, T = self.model, instrument.maturity
        Lbar, As, f, grid, u0, x0 = m.operators(instrument, self.n, self.width)
        K, M = green_kubo(m.chain, f)
        C = sum(K[j, k] * (As[j] @ As[k]) for j in range(len(As)) for k in range(len(As)))
        Z = sp.csr_matrix((grid.n, grid.n))
        Aug = sp.bmat([[Lbar, Z], [C, Lbar]], format="csc")
        v = expm_multiply(Aug * T, np.r_[u0, np.zeros(grid.n)])
        ubar, u1 = v[:grid.n], v[grid.n:]
        mem = sum(-M[j, self.regime] * (As[j] @ ubar) for j in range(len(As)))
        self.averaged, self.correction, self.memory = grid.interp(ubar, x0), grid.interp(u1, x0), grid.interp(mem, x0)
        scale = max(abs(self.averaged), 1e-300)
        self.diagnostics = dict(holdingTime=m.chain.meanHoldingTime(), correctionRelative=abs(self.correction) / scale,
                                memoryRelative=abs(self.memory) / scale)
        rel = self.diagnostics["correctionRelative"] + self.diagnostics["memoryRelative"]
        self.diagnostics["estimatedError"] = rel * rel               # the neglected second-order term, checked against the referee
        if rel > self.warnAbove:
            from .engines import ExpansionWarning
            warnings.warn(f"the first-order correction is {rel:.1e} of the averaged value (holding time "
                          f"{m.chain.meanHoldingTime():.3g}); the neglected second-order term is about its square, {rel * rel:.1e}. "
                          "Use SwitchingFDReferee for the switching solution.", ExpansionWarning, stacklevel=3)
        return self.averaged + self.correction + self.memory


class SwitchingFDReferee:
    """The switching model solved without expansion on the same grid: u_i' = L_i u_i + sum_j Q_ij u_j."""
    def __init__(self, model, regime=0, n=801, width=None):
        self.model, self.regime, self.n, self.width = model, regime, n, width

    def calculate(self, instrument):
        m, T = self.model, instrument.maturity
        Lbar, As, f, grid, u0, x0 = m.operators(instrument, self.n, self.width)
        f = np.atleast_2d(np.asarray(f, float)); pi = m.chain.stationaryDistribution(); nR = m.n
        blocks = [[None] * nR for _ in range(nR)]
        Q = m.chain.generator; I = sp.identity(grid.n, format="csr")
        for i in range(nR):
            Li = Lbar + sum((f[j, i] - f[j] @ pi) * As[j] for j in range(len(As)))
            for k in range(nR):
                blocks[i][k] = Li + Q[i, i] * I if i == k else Q[i, k] * I
        Big = sp.bmat(blocks, format="csc")
        v = expm_multiply(Big * T, np.tile(u0, nR))
        return grid.interp(v[self.regime * grid.n:(self.regime + 1) * grid.n], x0)
