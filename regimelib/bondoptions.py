"""Options on coupon bonds under a switching one-factor Gaussian rate, by Jamshidian's decomposition conditioned on
the regime at expiry: in regime j the bond at expiry is sum_k c_k A_kj e^{-b_k x}, monotone in x, so the option
splits into options on the zero-coupon bonds with strikes A_kj e^{-b_k x*_j}, one crossing x*_j per regime. The
Gil-Pelaez integrals follow regimelib._engine.options.zcb_call (Kronrod panels, an error check, and the regime
indicator at expiry carried by the terminal vector)."""
import math
import cmath
import numpy as np
from scipy.optimize import brentq
from ._engine.options import _kronrod_nodes, _terminal_vectors
from ._engine.models import vasicek_terminal
from ._engine.fastswitch import FastSwitch, numerical_a_callable


def coupon_bond_call(T, cashflows, K, x0, start, kappa, thetas, sigmas, Q, order=None, U=None, tol=1e-10, panels=None):
    """Call expiring at T, strike K, on a bond paying c_k at S_k > T (cashflows: list of (S_k, c_k)), under
    dx = kappa (theta_y - x) dt + sigma_y dW. order=None uses the numerical solution; an integer the expansion."""
    Q = np.asarray(Q, float); m = len(thetas)
    wv, vl = np.linalg.eig(Q.T); pi = np.real(vl[:, np.argmin(abs(wv))]); pi = pi / pi.sum()
    var = float(pi @ np.asarray(sigmas) ** 2) / (2 * kappa) * (1 - math.exp(-2 * kappa * T))
    if U is None:
        U = 8 / math.sqrt(var)
    ET = math.exp(-kappa * T); mean_xT = x0 * ET + float(pi @ np.asarray(thetas)) * (1 - ET)

    def a_vec(t, c, a0):
        g, gf, Bc = vasicek_terminal(kappa, thetas, sigmas, c)
        a = numerical_a_callable(t, Q, gf, rtol=1e-12, a0=a0) if order is None else FastSwitch(Q, g, order=order, a0=a0).a(t, order)
        return a, Bc.value(t)
    bs = [(1 - math.exp(-kappa * (S - T))) / kappa for S, _ in cashflows]
    A = [a_vec(S - T, 0.0, np.ones(m))[0].real for S, _ in cashflows]          # A[k][j]: bond k factor in regime j
    # one crossing per expiry regime: sum_k c_k A_kj e^{-b_k x*} = K, decreasing in x
    xstar = []
    for j in range(m):
        f = lambda x, j=j: sum(c * A[k][j] * math.exp(-bs[k] * x) for k, (S, c) in enumerate(cashflows)) - K
        lo, hi = mean_xT - 60 * math.sqrt(var) - 1.0, mean_xT + 60 * math.sqrt(var) + 1.0
        xstar.append(brentq(f, lo, hi))
    cases = []
    for j in range(m):
        for k, (S, c) in enumerate(cashflows):
            cases.append((j, bs[k], c * A[k][j]))                                # A_kj e^{-b_k x} 1{x < x*_j}
        cases.append((j, 0.0, -K))                                                # -K 1{x < x*_j}
    price = 0.0
    for j, c0, weight in cases:
        a, Bv = a_vec(T, c0, np.eye(m)[j])
        price += weight * 0.5 * (a[start] * cmath.exp(-Bv * x0)).real
    rate = max(abs(xs - mean_xT) for xs in xstar) + math.sqrt(var)
    npan = max(4, math.ceil(U * rate / math.pi)) if panels is None else panels
    for _ in range(6):
        us, wk, wg = _kronrod_nodes(U, npan); nu = len(us)
        if order is None:
            cs = np.concatenate([c0 - 1j * us for _, c0, _ in cases])
            A0 = np.zeros((m, len(cases) * nu), complex)
            for kk, (j, _, _) in enumerate(cases):
                A0[j, kk * nu:(kk + 1) * nu] = 1.0
            avals = _terminal_vectors(T, Q, kappa, thetas, sigmas, cs, A0)[start].reshape(len(cases), nu)
        else:
            avals = np.array([[a_vec(T, c0 - 1j * u, np.eye(m)[j])[0][start] for u in us] for j, c0, _ in cases])
        val, err = 0.0, 0.0
        for kk, (j, c0, weight) in enumerate(cases):
            Bv = (c0 - 1j * us) * ET + (1 - ET) / kappa
            f = (np.exp(-1j * us * xstar[j]) * avals[kk] * np.exp(-Bv * x0)).imag / us
            val += weight * (-(wk @ f) / math.pi); err += abs(weight) * abs((wk - wg) @ f) / math.pi
        if err <= tol:
            return price + val
        npan *= 2
    raise ArithmeticError(f"the Gil-Pelaez integrals did not converge to {tol:.0e} (error estimate {err:.1e})")
