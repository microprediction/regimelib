"""The first-order tier: models whose switched parameter multiplies an operator, so the state does not factor out.

u_t = (L_bar + sum_j f~_j(y) A_j) u. To first order in the holding time,
    u(T) = e^{T L_bar} u0 + int_0^T e^{(T - s) L_bar} (sum_jk K_jk A_j A_k) e^{s L_bar} u0 ds + memory,
K the Green-Kubo matrix of the chain (integral of the autocovariance of the coefficients) and the memory term
-(Q# f~)_i . A e^{T L_bar} u0 for a start in regime i. The Duhamel integral is the second block of one matrix
exponential of [[L_bar, 0], [K A A, L_bar]] applied to (u0, 0). Finite differences on a grid the model supplies."""
import math
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


class FirstOrderFDEngine:
    """First-order pricing on a grid. The model supplies operators(grid) -> (L_bar, [A_j], [f_j per regime], grid,
    payoff(grid), x0)."""
    def __init__(self, model, regime=0, n=801, width=None):
        self.model, self.regime, self.n, self.width = model, regime, n, width
        self.averaged = self.correction = self.memory = None

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
