"""Option prices under fast regime switching: Black-Scholes with a switching volatility (Lewis's formula) and
calls on zero-coupon bonds under Vasicek with a switching mean level and volatility (Gil-Pelaez inversion)."""
import math
import cmath
import numpy as np
from .fastswitch import FastSwitch, numerical_a_callable
from .models import bs_switching, vasicek_terminal


def _nodes(U, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1) * U / 2, w * U / 2


def bs_call(S0, K, T, r, sigmas, Q, start, order=None, U=None, n=96):
    """European call; order=None uses the numerical solution of the reduced system. The Fourier integral stops
    where the characteristic function falls below exp(-40); beyond that the regimes' exponents differ by more
    than the switching rate and the expansion does not apply."""
    if U is None:
        Qa = np.asarray(Q, float)
        wv, vl = np.linalg.eig(Qa.T)
        pi = np.real(vl[:, np.argmin(abs(wv))])
        pi = pi / pi.sum()
        U = math.sqrt(80 / (float(pi @ np.asarray(sigmas) ** 2) * T))
    us, ws = _nodes(U, n)
    k, tot = math.log(S0 / K), 0.0
    for u, w in zip(us, ws):
        g, gf = bs_switching(u - 0.5j, r, sigmas)
        phi = numerical_a_callable(T, Q, gf, rtol=1e-12)[start] if order is None else FastSwitch(Q, g, order=order).a(T, order)[start]
        tot += w * (cmath.exp(1j * u * k) * phi).real / (u * u + 0.25)
    return S0 - math.sqrt(S0 * K) * math.exp(-r * T) / math.pi * tot


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
        Bc = cs * math.exp(-kappa * r) + (1 - math.exp(-kappa * r)) / kappa
        d = (-kappa * th * Bc + 0.5 * s2 * Bc * Bc) * A + Q @ A
        return np.concatenate([d.real.ravel(), d.imag.ravel()])
    y0 = np.asarray(A0, complex)
    y = solve_ivp(rhs, (0, t), np.concatenate([y0.real.ravel(), y0.imag.ravel()]), method='DOP853',
                  rtol=rtol, atol=1e-14).y[:, -1]
    return (y[:m * nc] + 1j * y[m * nc:]).reshape(m, nc)


def zcb_call(T, S, K, x0, start, kappa, thetas, sigmas, Q, order=None, U=None, tol=1e-10, panels=None):
    """Call expiring at T on the zero-coupon bond maturing at S. The bond at T in regime j is A_j exp(-b x_T).
    The Gil-Pelaez integrals stop at eight standard deviations of x_T, in frequency, under the averaged model.

    Each integral is computed on panels of [0, U] with the 15-point Kronrod rule; the integrand oscillates at the
    rate |x* - E x_T|, so the panels start at half a period wide, and they are halved until the embedded 7-point
    Gauss rule agrees with the Kronrod rule within tol (an absolute error on the price). A rule that does not
    converge raises rather than returning a value. order=None uses the numerical solution of the reduced system;
    an integer uses the expansion through that order."""
    Q = np.asarray(Q, float)
    m = len(thetas)
    wv, vl = np.linalg.eig(Q.T)
    pi = np.real(vl[:, np.argmin(abs(wv))])
    pi = pi / pi.sum()
    var = float(pi @ np.asarray(sigmas) ** 2) / (2 * kappa) * (1 - math.exp(-2 * kappa * T))
    if U is None:
        U = 8 / math.sqrt(var)
    b = (1 - math.exp(-kappa * (S - T))) / kappa
    ET = math.exp(-kappa * T)
    mean_xT = x0 * ET + float(pi @ np.asarray(thetas)) * (1 - ET)

    def a_vec(t, c, a0):
        g, gf, Bc = vasicek_terminal(kappa, thetas, sigmas, c)
        if order is None:
            a = numerical_a_callable(t, Q, gf, rtol=1e-12, a0=a0)
        else:
            a = FastSwitch(Q, g, order=order, a0=a0).a(t, order)
        return a, Bc.value(t)
    A = a_vec(S - T, 0.0, np.ones(m))[0].real       # regime bond factors at expiry
    # the four integrands: regime j at expiry, and the two terms A_j e^{-b x} 1{x < x*} and K 1{x < x*}
    cases = [(j, c0, weight) for j in range(m) for c0, weight in ((b, A[j]), (0.0, -K))]
    xstar = [math.log(A[j] / K) / b for j in range(m)]
    price = 0.0
    for j, c0, weight in cases:
        a, Bv = a_vec(T, c0, np.eye(m)[j])
        price += weight * 0.5 * (a[start] * cmath.exp(-Bv * x0)).real
    rate = max(abs(xs - mean_xT) for xs in xstar) + math.sqrt(var)
    npan = max(4, math.ceil(U * rate / math.pi)) if panels is None else panels
    for _ in range(6):
        us, wk, wg = _kronrod_nodes(U, npan)
        nu = len(us)
        if order is None:
            cs = np.concatenate([c0 - 1j * us for _, c0, _ in cases])
            A0 = np.zeros((m, len(cases) * nu), complex)
            for k, (j, _, _) in enumerate(cases):
                A0[j, k * nu:(k + 1) * nu] = 1.0
            avals = _terminal_vectors(T, Q, kappa, thetas, sigmas, cs, A0)[start].reshape(len(cases), nu)
        else:
            avals = np.array([[a_vec(T, c0 - 1j * u, np.eye(m)[j])[0][start] for u in us] for j, c0, _ in cases])
        val, err = 0.0, 0.0
        for k, (j, c0, weight) in enumerate(cases):
            Bv = (c0 - 1j * us) * ET + (1 - ET) / kappa
            f = (np.exp(-1j * us * xstar[j]) * avals[k] * np.exp(-Bv * x0)).imag / us
            val += weight * (-(wk @ f) / math.pi)
            err += abs(weight) * abs((wk - wg) @ f) / math.pi
        if err <= tol:
            return price + val
        npan *= 2
    raise ArithmeticError(f"the Gil-Pelaez integrals did not converge to {tol:.0e} with {npan // 2} panels "
                          f"(error estimate {err:.1e})")
