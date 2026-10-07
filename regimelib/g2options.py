"""Options on zero-coupon bonds under the switching two-factor Gaussian model (G2++). With r = x + y + phi(t) the
bond at expiry T in regime j is c(T, S) A_j exp(-z), z = B_a(S - T) x_T + B_b(S - T) y_T, so the payoff is monotone
in the one variable z and the Gil-Pelaez integrals of regimelib._engine.options.zcb_call apply, with the terminal
functional E[exp(-int_0^T (x + y)) exp(-c z) 1{y_T = j}] = a_j(T) of the reduced system under the two-factor
terminal forcing g_i(tau) = (sigma_i^2 D_x^2 + eta_i^2 D_y^2 + 2 rho_i sigma_i eta_i D_x D_y) / 2,
D_x(tau) = c B_a e^{-a tau} + (1 - e^{-a tau}) / a and likewise D_y."""
import math
import cmath
import numpy as np
from ._engine.options import _kronrod_nodes, settled
from ._engine.fastswitch import FastSwitch, ExpSum, numerical_a_callable, _finished
from ._engine.models import loadings, stable_B, ou_variance, refuse_hidden_variance


def _g2_forcing(a, b, sigmas, etas, rhos, cx, cy, T=None):
    """Forcing per regime for terminal coefficients cx, cy (complex allowed) on [0, T]: exponential sums, or
    Chebyshev series when a reversion speed is small or zero; the callables evaluate the loadings stably."""
    (Dx, fx), (Dy, fy) = loadings([(a, cx), (b, cy)], T)
    g = [(Dx * Dx).scale(0.5 * s * s) + (Dy * Dy).scale(0.5 * e * e) + (Dx * Dy).scale(r * s * e)
         for s, e, r in zip(sigmas, etas, rhos)]
    gfuncs = [(lambda s, e, r: (lambda t: 0.5 * s * s * fx(t) ** 2 + 0.5 * e * e * fy(t) ** 2 + r * s * e * fx(t) * fy(t)))(s, e, r)
              for s, e, r in zip(sigmas, etas, rhos)]
    return g, gfuncs


def _g2_terminal_vectors(t, Q, a, b, sigmas, etas, rhos, cxs, cys, A0, rtol=1e-12):
    """a(t) for every terminal pair (cxs[k], cys[k]) at once, a(0) = A0[:, k]; one DOP853 solve."""
    from scipy.integrate import solve_ivp
    Q = np.asarray(Q, float); m, nc = Q.shape[0], len(cxs)
    s = np.asarray(sigmas, float)[:, None]; e = np.asarray(etas, float)[:, None]; r = np.asarray(rhos, float)[:, None]
    cxs = np.asarray(cxs, complex)[None, :]; cys = np.asarray(cys, complex)[None, :]

    def rhs(tau, y):
        A = (y[:m * nc] + 1j * y[m * nc:]).reshape(m, nc)
        Dx = cxs * math.exp(-a * tau) + stable_B(a, tau)
        Dy = cys * math.exp(-b * tau) + stable_B(b, tau)
        g = 0.5 * s * s * Dx * Dx + 0.5 * e * e * Dy * Dy + r * s * e * Dx * Dy
        d = g * A + Q @ A
        return np.concatenate([d.real.ravel(), d.imag.ravel()])
    y0 = np.asarray(A0, complex)
    y = _finished(solve_ivp(rhs, (0, t), np.concatenate([y0.real.ravel(), y0.imag.ravel()]), method="DOP853", rtol=rtol, atol=1e-14), t)
    return (y[:m * nc] + 1j * y[m * nc:]).reshape(m, nc)


def g2_zcb_call(T, S, K, start, a, b, sigmas, etas, rhos, Q, order=None, U=None, tol=1e-10, panels=None, solve=None):
    """Call expiring at T, strike K, on the unit bond maturing at S, in the zero-mean factor model (x0 = y0 = 0,
    no phi): the caller scales by the deterministic curve factors. order=None: numerical solution; else expansion."""
    Q = np.asarray(Q, float); m = len(sigmas)
    wv, vl = np.linalg.eig(Q.T); pi = np.real(vl[:, np.argmin(abs(wv))]); pi = pi / pi.sum()
    sig, eta, rho = map(lambda v: np.asarray(v, float), (sigmas, etas, rhos))
    tau = S - T
    Ba, Bb = stable_B(a, tau), stable_B(b, tau)
    Vx = float(pi @ sig ** 2) * ou_variance(a, T)
    Vy = float(pi @ eta ** 2) * ou_variance(b, T)
    Cxy = float(pi @ (rho * sig * eta)) * ou_variance((a + b) / 2, T)
    var = Ba * Ba * Vx + Bb * Bb * Vy + 2 * Ba * Bb * Cxy
    chosen = U is None
    if U is None and var > 0:
        U = 8 / math.sqrt(var)

    def a_vec(t, c, a0, numerical=True, direct=False):
        g, gf = _g2_forcing(a, b, sig, eta, rho, c * Ba, c * Bb, t)
        out = None if order is None or direct else solve(g, gf, t, a0) if solve is not None else FastSwitch(Q, g, order=order, a0=a0).a(t, order)
        if out is None and numerical:
            out = numerical_a_callable(t, Q, gf, rtol=1e-12, a0=a0)
        return out
    A = np.asarray(a_vec(tau, 0.0, np.ones(m))).real                    # bond factors at expiry per regime
    if var < -1e-12 * (Ba * Ba * Vx + Bb * Bb * Vy):
        raise ValueError("the factor covariance is not positive semidefinite")
    # the shortcut below needs the variance to vanish in every regime, not only on stationary average
    perRegime = (Ba * Ba * sig ** 2 * ou_variance(a, T) + Bb * Bb * eta ** 2 * ou_variance(b, T)
                 + 2 * Ba * Bb * rho * sig * eta * ou_variance((a + b) / 2, T))
    scale = Ba * Ba * sig ** 2 * ou_variance(a, T) + Bb * Bb * eta ** 2 * ou_variance(b, T)
    if not np.all(perRegime <= 1e-12 * np.maximum(scale, 1e-300)):
        refuse_hidden_variance(var if var > 1e-12 * (Ba * Ba * Vx + Bb * Bb * Vy) else 0.0, [1.0], "a G2 bond option")
    if var <= 1e-12 * (Ba * Ba * Vx + Bb * Bb * Vy) or var == 0.0:
        # B_a x_T + B_b y_T has no variance (no volatility, or exact cancellation): the bond at expiry is A_j in regime j
        return float(sum(max(A[j] - K, 0.0) * np.asarray(a_vec(T, 0.0, np.eye(m)[j]))[start].real for j in range(m)))
    zstar = [math.log(A[j] / K) for j in range(m)]
    cases = [(j, c0, weight) for j in range(m) for c0, weight in ((1.0, A[j]), (0.0, -K))]
    # z = B_a x_T + B_b y_T has mean zero and, given the path of the chain, at most this variance
    vmax = (abs(Ba) * np.max(np.abs(sig)) * math.sqrt(ou_variance(a, T)) + abs(Bb) * np.max(np.abs(eta)) * math.sqrt(ou_variance(b, T))) ** 2
    sure = settled(zstar, 0.0, 0.0, vmax, 1.0)
    price = 0.0
    for j, c0, weight in cases:
        price += weight * (0.5 if sure[j] is None else sure[j]) * np.asarray(a_vec(T, c0, np.eye(m)[j]))[start].real
    cases = [case for case in cases if sure[case[0]] is None]
    if not cases:
        return float(price)
    if chosen:
        # the stationary variance sizes a chain that mixes before expiry; from a quiet regime it may not leave, the
        # transform decays more slowly, so follow the transforms themselves
        beyond = lambda U: max(abs(np.asarray(a_vec(T, c0 - 1j * U, np.eye(m)[j], direct=True))[start]) for j, c0, _ in cases)
        widest = 16 * U
        while U < widest and beyond(U) > 1e-9:
            U *= 2
        if beyond(U) > 1e-9:
            raise ArithmeticError("the transform from this starting regime has not decayed at sixteen times the range "
                                  "the stationary variance gives: the chain is slow against the expiry and this regime "
                                  "is far quieter than the others. Use SwitchingFDEngine for this option.")
    rate = max(abs(zs) for zs, known in zip(zstar, sure) if known is None) + math.sqrt(var)
    npan = max(4, math.ceil(U * rate / math.pi)) if panels is None else panels
    blownUp = False
    for _ in range(6):
        us, wk, wg = _kronrod_nodes(U, npan); nu = len(us)
        avals = None
        if order is not None and not blownUp:
            avals = np.empty((len(cases), nu), complex)
            for k, (j, c0, _) in enumerate(cases):
                for n, u in enumerate(us):
                    value = a_vec(T, c0 - 1j * u, np.eye(m)[j], numerical=False)
                    if value is None:
                        blownUp = True; break                           # one method for every node: see _solver
                    avals[k, n] = np.asarray(value)[start]
                if blownUp:
                    avals = None; break
        if avals is None:
            cs = np.concatenate([c0 - 1j * us for _, c0, _ in cases])
            A0 = np.zeros((m, len(cases) * nu), complex)
            for k, (j, _, _) in enumerate(cases):
                A0[j, k * nu:(k + 1) * nu] = 1.0
            avals = _g2_terminal_vectors(T, Q, a, b, sig, eta, rho, cs * Ba, cs * Bb, A0)[start].reshape(len(cases), nu)
        val, err = 0.0, 0.0
        for k, (j, c0, weight) in enumerate(cases):
            f = (np.exp(-1j * us * zstar[j]) * avals[k]).imag / us
            val += weight * (-(wk @ f) / math.pi); err += abs(weight) * abs((wk - wg) @ f) / math.pi
        if err <= tol * max(1.0, abs(K)):                          # of the strike: the price scales with it
            return price + val
        npan *= 2
    raise ArithmeticError(f"the Gil-Pelaez integrals did not converge to {tol:.0e} (error estimate {err:.1e})")
