"""Options on zero-coupon bonds under the switching two-factor Gaussian model (G2++). With r = x + y + phi(t) the
bond at expiry T in regime j is c(T, S) A_j exp(-z), z = B_a(S - T) x_T + B_b(S - T) y_T, so the payoff is monotone
in the one variable z and the Gil-Pelaez integrals of regimelib._engine.options.zcb_call apply, with the terminal
functional E[exp(-int_0^T (x + y)) exp(-c z) 1{y_T = j}] = a_j(T) of the reduced system under the two-factor
terminal forcing g_i(tau) = (sigma_i^2 D_x^2 + eta_i^2 D_y^2 + 2 rho_i sigma_i eta_i D_x D_y) / 2,
D_x(tau) = c B_a e^{-a tau} + (1 - e^{-a tau}) / a and likewise D_y."""
import math
import cmath
import numpy as np
from ._engine.options import _kronrod_nodes
from ._engine.fastswitch import FastSwitch, ExpSum, numerical_a_callable


def _g2_forcing(a, b, sigmas, etas, rhos, cx, cy):
    """ExpSum forcing per regime for terminal coefficients cx, cy (complex allowed)."""
    Dx = ExpSum({0: 1 / a, a: cx - 1 / a}); Dy = ExpSum({0: 1 / b, b: cy - 1 / b})
    g = [(Dx * Dx).scale(0.5 * s * s) + (Dy * Dy).scale(0.5 * e * e) + (Dx * Dy).scale(r * s * e)
         for s, e, r in zip(sigmas, etas, rhos)]
    gfuncs = [(lambda gi: (lambda t: gi.value(t)))(gi) for gi in g]
    return g, gfuncs


def _g2_terminal_vectors(t, Q, a, b, sigmas, etas, rhos, cxs, cys, A0, rtol=1e-12):
    """a(t) for every terminal pair (cxs[k], cys[k]) at once, a(0) = A0[:, k]; one DOP853 solve."""
    from scipy.integrate import solve_ivp
    Q = np.asarray(Q, float); m, nc = Q.shape[0], len(cxs)
    s = np.asarray(sigmas, float)[:, None]; e = np.asarray(etas, float)[:, None]; r = np.asarray(rhos, float)[:, None]
    cxs = np.asarray(cxs, complex)[None, :]; cys = np.asarray(cys, complex)[None, :]

    def rhs(tau, y):
        A = (y[:m * nc] + 1j * y[m * nc:]).reshape(m, nc)
        Dx = cxs * math.exp(-a * tau) + (1 - math.exp(-a * tau)) / a
        Dy = cys * math.exp(-b * tau) + (1 - math.exp(-b * tau)) / b
        g = 0.5 * s * s * Dx * Dx + 0.5 * e * e * Dy * Dy + r * s * e * Dx * Dy
        d = g * A + Q @ A
        return np.concatenate([d.real.ravel(), d.imag.ravel()])
    y0 = np.asarray(A0, complex)
    y = solve_ivp(rhs, (0, t), np.concatenate([y0.real.ravel(), y0.imag.ravel()]), method="DOP853", rtol=rtol, atol=1e-14).y[:, -1]
    return (y[:m * nc] + 1j * y[m * nc:]).reshape(m, nc)


def g2_zcb_call(T, S, K, start, a, b, sigmas, etas, rhos, Q, order=None, U=None, tol=1e-10, panels=None):
    """Call expiring at T, strike K, on the unit bond maturing at S, in the zero-mean factor model (x0 = y0 = 0,
    no phi): the caller scales by the deterministic curve factors. order=None: numerical solution; else expansion."""
    Q = np.asarray(Q, float); m = len(sigmas)
    wv, vl = np.linalg.eig(Q.T); pi = np.real(vl[:, np.argmin(abs(wv))]); pi = pi / pi.sum()
    sig, eta, rho = map(lambda v: np.asarray(v, float), (sigmas, etas, rhos))
    tau = S - T
    Ba, Bb = (1 - math.exp(-a * tau)) / a, (1 - math.exp(-b * tau)) / b
    Vx = float(pi @ sig ** 2) * (1 - math.exp(-2 * a * T)) / (2 * a)
    Vy = float(pi @ eta ** 2) * (1 - math.exp(-2 * b * T)) / (2 * b)
    Cxy = float(pi @ (rho * sig * eta)) * (1 - math.exp(-(a + b) * T)) / (a + b)
    var = Ba * Ba * Vx + Bb * Bb * Vy + 2 * Ba * Bb * Cxy
    if U is None:
        U = 8 / math.sqrt(var)

    def a_vec(t, c, a0):
        g, gf = _g2_forcing(a, b, sig, eta, rho, c * Ba, c * Bb)
        if order is None:
            return numerical_a_callable(t, Q, gf, rtol=1e-12, a0=a0)
        return FastSwitch(Q, g, order=order, a0=a0).a(t, order)
    A = np.asarray(a_vec(tau, 0.0, np.ones(m))).real                    # bond factors at expiry per regime
    zstar = [math.log(A[j] / K) for j in range(m)]
    cases = [(j, c0, weight) for j in range(m) for c0, weight in ((1.0, A[j]), (0.0, -K))]
    price = 0.0
    for j, c0, weight in cases:
        price += weight * 0.5 * np.asarray(a_vec(T, c0, np.eye(m)[j]))[start].real
    rate = max(abs(zs) for zs in zstar) + math.sqrt(var)
    npan = max(4, math.ceil(U * rate / math.pi)) if panels is None else panels
    for _ in range(6):
        us, wk, wg = _kronrod_nodes(U, npan); nu = len(us)
        if order is None:
            cs = np.concatenate([c0 - 1j * us for _, c0, _ in cases])
            A0 = np.zeros((m, len(cases) * nu), complex)
            for k, (j, _, _) in enumerate(cases):
                A0[j, k * nu:(k + 1) * nu] = 1.0
            avals = _g2_terminal_vectors(T, Q, a, b, sig, eta, rho, cs * Ba, cs * Bb, A0)[start].reshape(len(cases), nu)
        else:
            avals = np.array([[np.asarray(a_vec(T, c0 - 1j * u, np.eye(m)[j]))[start] for u in us] for j, c0, _ in cases])
        val, err = 0.0, 0.0
        for k, (j, c0, weight) in enumerate(cases):
            f = (np.exp(-1j * us * zstar[j]) * avals[k]).imag / us
            val += weight * (-(wk @ f) / math.pi); err += abs(weight) * abs((wk - wg) @ f) / math.pi
        if err <= tol:
            return price + val
        npan *= 2
    raise ArithmeticError(f"the Gil-Pelaez integrals did not converge to {tol:.0e} (error estimate {err:.1e})")
