"""Options on coupon bonds under a switching one-factor Gaussian rate, by Jamshidian's decomposition conditioned on
the regime at expiry: in regime j the bond at expiry is sum_k c_k A_kj e^{-b_k x}, monotone in x, so the option
splits into options on the zero-coupon bonds with strikes A_kj e^{-b_k x*_j}, one crossing x*_j per regime. The
Gil-Pelaez integrals follow regimelib._engine.options.zcb_call (Kronrod panels, an error check, and the regime
indicator at expiry carried by the terminal vector)."""
import math
import cmath
import numpy as np
from scipy.optimize import brentq
from ._engine.options import _kronrod_nodes, _terminal_vectors, settled
from ._engine.models import vasicek_terminal, stable_B, ou_variance, int_B2, refuse_hidden_variance
from ._engine.fastswitch import FastSwitch, numerical_a_callable


def coupon_bond_call(T, cashflows, K, x0, start, kappa, thetas, sigmas, Q, order=None, U=None, tol=1e-10, panels=None, solve=None):
    """Call expiring at T, strike K, on a bond paying c_k at S_k > T (cashflows: list of (S_k, c_k)), under
    dx = kappa (theta_y - x) dt + sigma_y dW. order=None uses the numerical solution; an integer the expansion."""
    Q = np.asarray(Q, float); m = len(thetas)
    wv, vl = np.linalg.eig(Q.T); pi = np.real(vl[:, np.argmin(abs(wv))]); pi = pi / pi.sum()
    var = float(pi @ np.asarray(sigmas) ** 2) * ou_variance(kappa, T)
    refuse_hidden_variance(var, sigmas, "an option on a bond")
    chosen = U is None
    if U is None:
        U = 8 / math.sqrt(var)
    ET = math.exp(-kappa * T); mean_xT = x0 * ET + float(pi @ np.asarray(thetas)) * (1 - ET)

    def a_vec(t, c, a0, numerical=True, direct=False):
        g, gf, Bc = vasicek_terminal(kappa, thetas, sigmas, c, t)
        a = None if order is None or direct else solve(g, gf, t, a0) if solve is not None else FastSwitch(Q, g, order=order, a0=a0).a(t, order)
        if a is None and numerical:
            a = numerical_a_callable(t, Q, gf, rtol=1e-12, a0=a0)
        return a, Bc.value(t)
    bs = [stable_B(kappa, S - T) for S, _ in cashflows]
    A = [a_vec(S - T, 0.0, np.ones(m))[0].real for S, _ in cashflows]          # A[k][j]: bond k factor in regime j
    # one crossing per expiry regime: sum_k c_k A_kj e^{-b_k x*} = K, decreasing in x
    xstar = []
    for j in range(m):
        f = lambda x, j=j: sum(c * A[k][j] * math.exp(-bs[k] * x) for k, (S, c) in enumerate(cashflows)) - K
        # f is continuous and strictly decreasing from +inf to -K, so the root exists: widen until the signs differ
        half = 60 * math.sqrt(var) + 1.0
        lo, hi = mean_xT - half, mean_xT + half
        for _ in range(200):
            flo, fhi = f(lo), f(hi)
            if flo > 0 >= fhi:
                break
            if flo <= 0:
                lo -= half
            if fhi > 0:
                hi += half
            half *= 2
        else:
            raise ArithmeticError("no exercise boundary found for the coupon-bond option")
        xstar.append(brentq(f, lo, hi))
    cases = []
    for j in range(m):
        for k, (S, c) in enumerate(cashflows):
            cases.append((j, bs[k], c * A[k][j]))                                # A_kj e^{-b_k x} 1{x < x*_j}
        cases.append((j, 0.0, -K))                                                # -K 1{x < x*_j}
    th = np.asarray(thetas, float)
    terms = []
    for j, c0, weight in cases:
        a, Bv = a_vec(T, c0, np.eye(m)[j])
        terms.append(weight * (a[start] * cmath.exp(-Bv * x0)).real)
    s2max = float(np.max(np.asarray(sigmas, float) ** 2)); vmax = s2max * ou_variance(kappa, T)
    sure = settled(xstar, x0 * ET + th.min() * (1 - ET), x0 * ET + th.max() * (1 - ET), vmax, max(bs),
                   drift=math.sqrt(vmax * s2max * int_B2(kappa, T)), scale=max(abs(t) for t in terms))
    price = sum(t * (0.5 if sure[j] is None else sure[j]) for t, (j, _, _) in zip(terms, cases))
    cases = [case for case in cases if sure[case[0]] is None]
    if not cases:
        return price
    def beyond(U):                                                   # the largest of the transforms at the cutoff
        return max(abs(a_vec(T, c0 - 1j * U, np.eye(m)[j], direct=True)[0][start] * cmath.exp(-((c0 - 1j * U) * ET + stable_B(kappa, T)) * x0))
                   for j, c0, _ in cases)
    if chosen:
        # the stationary variance sizes a chain that mixes before expiry; from a quiet regime it may not leave, the
        # transform decays more slowly, so follow the transforms themselves
        widest = 16 * U; size = max(1.0, max(abs(t) for t in terms))     # against the terms, which a rate shift scales
        while U < widest and beyond(U) > 1e-9 * size:
            U *= 2
        if beyond(U) > 1e-9 * size:
            raise ArithmeticError("the transform from this starting regime has not decayed at sixteen times the range "
                                  "the stationary variance gives: the chain is slow against the expiry and this regime "
                                  "is far quieter than the others. Use SwitchingFDEngine for this option.")
    rate = max(abs(xs - mean_xT) for xs, known in zip(xstar, sure) if known is None) + math.sqrt(var)
    npan = max(4, math.ceil(U * rate / math.pi)) if panels is None else panels
    if panels is None and npan > 400:
        # the exercise boundary is far from the state in units of the range integrated (a bond maturing just after
        # expiry, at a strike away from par) and yet not far enough to be settled
        raise ArithmeticError(f"the Gil-Pelaez integrals would need {npan:,} panels: the exercise boundary is "
                              f"{rate:.3g} from the state's mean over a range of {U:.3g}. Use SwitchingFDEngine.")
    blownUp = False
    for _ in range(6):
        us, wk, wg = _kronrod_nodes(U, npan); nu = len(us)
        avals = None
        if order is not None and not blownUp:
            avals = np.empty((len(cases), nu), complex)
            for k, (j, c0, _) in enumerate(cases):
                for n, u in enumerate(us):
                    value = a_vec(T, c0 - 1j * u, np.eye(m)[j], numerical=False)[0]
                    if value is None:
                        blownUp = True; break                           # one method for every node: see _solver
                    avals[k, n] = np.asarray(value)[start]
                if blownUp:
                    avals = None; break
        if avals is None:
            cs = np.concatenate([c0 - 1j * us for _, c0, _ in cases])
            A0 = np.zeros((m, len(cases) * nu), complex)
            for kk, (j, _, _) in enumerate(cases):
                A0[j, kk * nu:(kk + 1) * nu] = 1.0
            avals = _terminal_vectors(T, Q, kappa, thetas, sigmas, cs, A0)[start].reshape(len(cases), nu)
        val, err = 0.0, 0.0
        for kk, (j, c0, weight) in enumerate(cases):
            Bv = (c0 - 1j * us) * ET + stable_B(kappa, T)
            f = (np.exp(-1j * us * xstar[j]) * avals[kk] * np.exp(-Bv * x0)).imag / us
            val += weight * (-(wk @ f) / math.pi); err += abs(weight) * abs((wk - wg) @ f) / math.pi
        if err <= tol * max(1.0, abs(K)):                          # of the strike: the price scales with it
            return price + val
        npan *= 2
        if npan > 800:
            break
    raise ArithmeticError(f"the Gil-Pelaez integrals did not converge to {tol:.0e} (error estimate {err:.1e})")
