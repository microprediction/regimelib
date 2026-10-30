"""Option prices under fast regime switching: Black-Scholes with a switching volatility (Lewis's formula) and
calls on zero-coupon bonds under Vasicek with a switching mean level and volatility (Gil-Pelaez inversion)."""
import math
import cmath
import numpy as np
from .fastswitch import FastSwitch, numerical_a_callable, _finished
from .models import vasicek_terminal, stable_B, ou_variance, int_B2, refuse_hidden_variance


# 15-point Kronrod rule with the embedded 7-point Gauss rule (QUADPACK qk15): the Gauss nodes are every second
# Kronrod node, so the difference of the two rules is an error estimate at no extra cost.
_XGK = np.array([0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
                 0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
                 0.586087235467691130294144838258730, 0.405845151377397166906606412076961,
                 0.207784955007898467600689403773245, 0.0])
_WGK = np.array([0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
                 0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
                 0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
                 0.204432940075298892414161999234649, 0.209482141084727828012999174891714])
_WG = np.array([0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
                0.381830050505118944950369775488975, 0.417959183673469387755102040816327])


def _kronrod_nodes(U, panels):
    """Nodes on [0, U] split into equal panels, with the Kronrod weights and the Gauss weights (zero at the
    Kronrod-only nodes)."""
    x = np.concatenate([-_XGK[:-1], _XGK[::-1]])
    wk = np.concatenate([_WGK[:-1], _WGK[::-1]])
    wg = np.zeros(15)
    wg[1::2] = np.concatenate([_WG[:-1], _WG[-1:], _WG[-2::-1]])
    edges = np.linspace(0.0, U, panels + 1)
    h = np.diff(edges)[:, None] / 2
    mid = edges[:-1, None] + h
    return (mid + h * x).ravel(), (h * wk).ravel(), (h * wg).ravel()


def settled(boundaries, lo, hi, vmax, tilt, drift=0.0, scale=1.0):
    """Exercise is below a boundary x*_j in the state. Given the path of the chain the state is Gaussian with mean in
    [lo, hi] and variance at most vmax, and each term of the price is an expectation under a measure that moves that
    mean: discounting by exp(-int r) moves it by the covariance of the state with the integrated rate, at most
    `drift` either way, and delivery of the bond lowers it by at most tilt vmax. A boundary far enough above every
    such mean is exercise with certainty (1.0), one far enough below is none (0.0); otherwise None, and the term is
    integrated. Far enough is the number of standard deviations at which the Gaussian tail times `scale`, the largest
    term, is below 1e-40: at least fifteen, and more when the terms are large. Without this a bond maturing just
    after expiry, whose boundary recedes as 1 / (S - T), asks for as many panels."""
    deviations = max(15.0, math.sqrt(2.0 * (math.log(max(scale, 1e-300)) + 92.2))) if scale > 0 else 15.0
    reach = deviations * math.sqrt(vmax) + drift
    return [1.0 if x > hi + reach else 0.0 if x < lo - tilt * vmax - reach else None for x in boundaries]


def _terminal_vectors(t, Q, kappa, thetas, sigmas, cs, A0, rtol=1e-12):
    """a(t) for a' = (Q + diag g_c(t)) a, a(0) = A0[:, k], for every c = cs[k] at once (one DOP853 solve), with
    g_c the Vasicek terminal forcing of models.vasicek_terminal. Returns the m x len(cs) array of a(t)."""
    from scipy.integrate import solve_ivp
    Q = np.asarray(Q, float)
    m, nc = Q.shape[0], len(cs)
    th, s2 = np.asarray(thetas, float)[:, None], (np.asarray(sigmas, float) ** 2)[:, None]
    cs = np.asarray(cs, complex)[None, :]

    def rhs(r, y):
        A = (y[:m * nc] + 1j * y[m * nc:]).reshape(m, nc)
        Bc = cs * math.exp(-kappa * r) + stable_B(kappa, r)
        d = (-kappa * th * Bc + 0.5 * s2 * Bc * Bc) * A + Q @ A
        return np.concatenate([d.real.ravel(), d.imag.ravel()])
    y0 = np.asarray(A0, complex)
    y = _finished(solve_ivp(rhs, (0, t), np.concatenate([y0.real.ravel(), y0.imag.ravel()]), method='DOP853',
                  rtol=rtol, atol=1e-14), t)
    return (y[:m * nc] + 1j * y[m * nc:]).reshape(m, nc)


def zcb_call(T, S, K, x0, start, kappa, thetas, sigmas, Q, order=None, U=None, tol=1e-10, panels=None, solve=None):
    """Call expiring at T on the zero-coupon bond maturing at S. The bond at T in regime j is A_j exp(-b x_T).
    The Gil-Pelaez integrals stop at eight standard deviations of x_T, in frequency, under the averaged model.

    Each integral is computed on panels of [0, U] with the 15-point Kronrod rule; the integrand oscillates at the
    rate |x* - E x_T|, so the panels start at half a period wide, and they are halved until the embedded 7-point
    Gauss rule agrees with the Kronrod rule within tol (an absolute error on the price). A rule that does not
    converge raises rather than returning a value. order=None uses the numerical solution of the reduced system;
    an integer uses the expansion through that order, by `solve(g, gfuncs, t, a0)` if one is given (the engine's,
    which solves numerically where the series diverges and keeps the diagnostics)."""
    Q = np.asarray(Q, float)
    m = len(thetas)
    wv, vl = np.linalg.eig(Q.T)
    pi = np.real(vl[:, np.argmin(abs(wv))])
    pi = pi / pi.sum()
    var = float(pi @ np.asarray(sigmas) ** 2) * ou_variance(kappa, T)
    refuse_hidden_variance(var, sigmas, "an option on a bond")
    chosen = U is None
    if U is None:
        U = 8 / math.sqrt(var)
    b = stable_B(kappa, S - T)
    ET = math.exp(-kappa * T)
    mean_xT = x0 * ET + float(pi @ np.asarray(thetas)) * (1 - ET)

    def a_vec(t, c, a0, numerical=True, direct=False):
        g, gf, Bc = vasicek_terminal(kappa, thetas, sigmas, c, t)
        a = None if order is None or direct else solve(g, gf, t, a0) if solve is not None else FastSwitch(Q, g, order=order, a0=a0).a(t, order)
        if a is None and numerical:
            a = numerical_a_callable(t, Q, gf, rtol=1e-12, a0=a0)
        return a, Bc.value(t)
    A = a_vec(S - T, 0.0, np.ones(m))[0].real       # regime bond factors at expiry
    # the four integrands: regime j at expiry, and the two terms A_j e^{-b x} 1{x < x*} and K 1{x < x*}
    cases = [(j, c0, weight) for j in range(m) for c0, weight in ((b, A[j]), (0.0, -K))]
    xstar = [math.log(A[j] / K) / b for j in range(m)]
    th = np.asarray(thetas, float)
    terms = []
    for j, c0, weight in cases:
        a, Bv = a_vec(T, c0, np.eye(m)[j])
        terms.append(weight * (a[start] * cmath.exp(-Bv * x0)).real)
    s2max = float(np.max(np.asarray(sigmas, float) ** 2)); vmax = s2max * ou_variance(kappa, T)
    sure = settled(xstar, x0 * ET + th.min() * (1 - ET), x0 * ET + th.max() * (1 - ET), vmax, b,
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
        us, wk, wg = _kronrod_nodes(U, npan)
        nu = len(us)
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
            for k, (j, _, _) in enumerate(cases):
                A0[j, k * nu:(k + 1) * nu] = 1.0
            avals = _terminal_vectors(T, Q, kappa, thetas, sigmas, cs, A0)[start].reshape(len(cases), nu)
        val, err = 0.0, 0.0
        for k, (j, c0, weight) in enumerate(cases):
            Bv = (c0 - 1j * us) * ET + stable_B(kappa, T)
            f = (np.exp(-1j * us * xstar[j]) * avals[k] * np.exp(-Bv * x0)).imag / us
            val += weight * (-(wk @ f) / math.pi)
            err += abs(weight) * abs((wk - wg) @ f) / math.pi
        if err <= tol:
            return price + val
        npan *= 2
        if npan > 800:
            break
    raise ArithmeticError(f"the Gil-Pelaez integrals did not converge to {tol:.0e} with {npan // 2} panels "
                          f"(error estimate {err:.1e})")
