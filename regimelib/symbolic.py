"""Closed-form prices and greeks as formulas (sympy), for the cases the fast-switching expansion gives in closed form.

Two regimes switching at rate lam each way, Vasicek short rate dr = kappa (theta_y - r) dt + sigma_y dW, zero-coupon
bond to second order in eps = 1 / lam (the regime-switching page of homogenization.microprediction.org):

    log P_{1,2}(T) = -B r0 + int_0^T gbar + (eps/2) int_0^T gt^2 - (eps^2/8) gt(T)^2
                     + log(1 +/- (eps/2) gt(T) -/+ (eps^2/4) gt'(T)),
    g_i = -kappa theta_i B + sigma_i^2 B^2 / 2,  gbar = (g_1 + g_2)/2,  gt = (g_1 - g_2)/2,  B = (1 - e^{-kappa T}) / kappa,

with the upper sign for a start in regime 1. Every greek is sympy.diff of this expression."""
import sympy as sp

r0, kappa, th1, th2, s1, s2, lam, T = sp.symbols("r_0 kappa theta_1 theta_2 sigma_1 sigma_2 lambda T", positive=True)
t = sp.symbols("t", positive=True)


class VasicekTwoStateBond:
    """Symbolic bond price under a two-state switching Vasicek model, second order in 1/lambda."""
    symbols = dict(r0=r0, kappa=kappa, theta1=th1, theta2=th2, sigma1=s1, sigma2=s2, lam=lam, T=T)

    def __init__(self, regime=0):
        sign = 1 if regime == 0 else -1
        B = (1 - sp.exp(-kappa * t)) / kappa
        g1 = -kappa * th1 * B + s1 ** 2 * B ** 2 / 2
        g2 = -kappa * th2 * B + s2 ** 2 * B ** 2 / 2
        gbar, gt = (g1 + g2) / 2, (g1 - g2) / 2
        eps = 1 / lam
        I_gbar = sp.integrate(sp.expand(gbar), (t, 0, T))
        I_gt2 = sp.integrate(sp.expand(gt ** 2), (t, 0, T))
        gtT, gtpT = gt.subs(t, T), sp.diff(gt, t).subs(t, T)
        BT = B.subs(t, T)
        # the terms of log P, kept apart so that a greek can be read term by term
        self.terms = {
            "state": -BT * r0,                                          # -B(T) r0
            "averaged": I_gbar,                                         # the averaged Vasicek exponent
            "green_kubo": eps / 2 * I_gt2,                              # (eps/2) int gt^2: the first-order term
            "second_order": -eps ** 2 / 8 * gtT ** 2,                   # -(eps^2/8) gt(T)^2
            "memory": sp.log(1 + sign * eps / 2 * gtT - sign * eps ** 2 / 4 * gtpT),   # the starting regime
        }
        self.B, self.gt, self.gbar = BT, gt, gbar
        self.logPrice = sum(self.terms.values())
        self.price = sp.exp(self.logPrice)
        self._fn = {}

    def greek(self, *wrt):
        """A formula: the derivative of the price with respect to the named symbols, e.g. greek('r0'), greek('r0', 'r0'),
        greek('theta1'), greek('lam'). Returns a sympy expression."""
        e = self.price
        for name in wrt:
            e = sp.diff(e, self.symbols[name])
        return e

    def logGreekTerms(self, *wrt):
        """The derivative of log P with respect to the named symbols, one formula per term of log P; the greek of the
        price itself is the price times the sum of these (for a single derivative)."""
        out = {}
        for name, term in self.terms.items():
            e = term
            for w in wrt:
                e = sp.diff(e, self.symbols[w])
            out[name] = sp.simplify(e)
        return out

    def evaluate(self, expr, **values):
        key = sp.srepr(expr)
        if key not in self._fn:
            self._fn[key] = sp.lambdify(list(self.symbols.values()), expr, "math")
        return float(self._fn[key](*[values[n] for n in self.symbols]))

    # QuantLib-named conveniences, as formulas
    def delta(self): return self.greek("r0")
    def gamma(self): return self.greek("r0", "r0")
    def theta(self): return -self.greek("T")


# ---------------------------------------------------------------- any number of regimes, first order
Kcc, Kcd, Kdd, mc, md, thbar, s2bar = sp.symbols("K_cc K_cd K_dd m_c m_d thetabar sigma2bar", real=True)   # K_cd is the symmetric part (K_cd + K_dc) / 2


class VasicekBondFirstOrder:
    """Vasicek bond under any finite chain, first order in the holding time, as a formula.

    g_i(t) = c_i B(t) + d_i B(t)^2 with c_i = -kappa theta_i, d_i = sigma_i^2 / 2. The chain enters through
    K_ab = int_0^inf Cov(a(y_0), b(y_t)) dt for a, b in {c, d} (the Green-Kubo matrix; only its symmetric part enters
    a scalar forcing, so K_cd here means (K_cd + K_dc) / 2, which differ for a non-reversible chain) and the memory coefficients
    m_c, m_d = -(Q# c~)_i, -(Q# d~)_i of the starting regime i:

        log P_i = -B r0 - kappa thetabar I1 + sigma2bar I2 / 2 + K_cc I2 + 2 K_cd I3 + K_dd I4 + log(1 + m_c B + m_d B^2),
        I_k = int_0^T B^k dt, B = (1 - e^{-kappa T}) / kappa."""
    symbols = dict(r0=r0, kappa=kappa, T=T, thetabar=thbar, sigma2bar=s2bar, Kcc=Kcc, Kcd=Kcd, Kdd=Kdd, mc=mc, md=md)

    def __init__(self):
        B = (1 - sp.exp(-kappa * t)) / kappa
        I = {k: sp.integrate(sp.expand(B ** k), (t, 0, T)) for k in (1, 2, 3, 4)}
        BT = B.subs(t, T)
        self.terms = {
            "state": -BT * r0,
            "averaged": -kappa * thbar * I[1] + s2bar / 2 * I[2],
            "green_kubo": Kcc * I[2] + 2 * Kcd * I[3] + Kdd * I[4],
            "memory": sp.log(1 + mc * BT + md * BT ** 2),
        }
        self.logPrice = sum(self.terms.values()); self.price = sp.exp(self.logPrice)
        self._fn = {}

    @staticmethod
    def coefficients(chain, kappa_, thetas, sigmas, regime=0):
        """The numbers the chain contributes, from its generator."""
        import numpy as np
        from .firstorder import green_kubo
        c = -kappa_ * np.asarray(thetas, float); d = 0.5 * np.asarray(sigmas, float) ** 2
        K, M = green_kubo(chain, [c, d]); pi = chain.stationaryDistribution()
        return dict(thetabar=float(pi @ thetas), sigma2bar=float(pi @ np.asarray(sigmas) ** 2),
                    Kcc=float(K[0, 0]), Kcd=float(0.5 * (K[0, 1] + K[1, 0])), Kdd=float(K[1, 1]), mc=float(-M[0, regime]), md=float(-M[1, regime]))

    def greek(self, *wrt):
        e = self.price
        for name in wrt:
            e = sp.diff(e, self.symbols[name])
        return e

    def evaluate(self, expr, **values):
        key = sp.srepr(expr)
        if key not in self._fn:
            self._fn[key] = sp.lambdify(list(self.symbols.values()), expr, "math")
        return float(self._fn[key](*[values[n] for n in self.symbols]))


# ---------------------------------------------------------------- two states, constant forcing: exact, all orders
class TwoStateConstantForcing:
    """The reduced system a' = (Q + diag g) a with two regimes and constant forcing g = (g1, g2), which is every
    Black-Scholes, Merton and variance-gamma characteristic function under switching, has the exact solution
    exp(M T) 1 with M the 2 x 2 matrix, written with its two eigenvalues mu_pm = (tr M +- sqrt(tr M^2 - 4 det M)) / 2:

        a(T) = [e^{mu+ T} (M - mu- I) - e^{mu- T} (M - mu+ I)] 1 / (mu+ - mu-).

    `a(regime)` is that formula as a sympy expression in q12, q21, g1, g2, T; the characteristic function of a
    switching Black-Scholes log return is `blackScholes(regime)` with g_i = -u^2 sigma_i^2 / 2 for the martingale
    return and u kept symbolic, so that d/du under Lewis's integral gives the greeks in closed form."""
    q12, q21, g1, g2, T, u, s1, s2 = sp.symbols("q12 q21 g1 g2 T u sigma1 sigma2")
    symbols = dict(q12=q12, q21=q21, g1=g1, g2=g2, T=T)

    def __init__(self):
        q12, q21, g1, g2, T = self.q12, self.q21, self.g1, self.g2, self.T
        M = sp.Matrix([[g1 - q12, q12], [q21, g2 - q21]])
        tr, det = M.trace(), M.det()
        disc = sp.sqrt(tr ** 2 - 4 * det)
        self.mu_plus, self.mu_minus = (tr + disc) / 2, (tr - disc) / 2
        I2 = sp.eye(2)
        E = (sp.exp(self.mu_plus * T) * (M - self.mu_minus * I2) - sp.exp(self.mu_minus * T) * (M - self.mu_plus * I2)) / disc
        self.vector = E * sp.Matrix([1, 1])
        self._fn = {}

    def a(self, regime=0):
        return sp.simplify(self.vector[regime])

    def blackScholes(self, regime=0):
        """E[exp(i u X_T) | y_0 = regime] for the martingale log return of a two-state Black-Scholes model."""
        u, s1, s2 = self.u, self.s1, self.s2
        return self.vector[regime].subs({self.g1: -u * (u + sp.I) * s1 ** 2 / 2, self.g2: -u * (u + sp.I) * s2 ** 2 / 2})

    def evaluate(self, expr, **values):
        key = sp.srepr(expr)
        if key not in self._fn:
            names = sorted(str(s) for s in expr.free_symbols)
            self._fn[key] = (names, sp.lambdify([sp.Symbol(n) for n in names], expr, "mpmath"))
        names, f = self._fn[key]
        return complex(f(*[values[n] for n in names]))


# ---------------------------------------------------------------- CIR with a switching mean level, first order
class CIRBondFirstOrder:
    """CIR bond dr = k (theta_y - r) dt + sigma sqrt(r) dW under any finite chain, first order in the holding time,
    as a formula. Only the mean level switches, so g_i(t) = c_i B(t) with c_i = -k theta_i and B the Riccati solution
    B(t) = 2 (e^{ht} - 1) / ((h + k)(e^{ht} - 1) + 2h), h = sqrt(k^2 + 2 sigma^2). With K_cc the Green-Kubo integral
    of c and m_c = -(Q# c~)_i for the starting regime i:

        log P_i = -B(T) r0 - k thetabar I1 + K_cc I2 + log(1 + m_c B(T)),   I_n = int_0^T B^n dt,

    where I1 = -(2/sigma^2) log A(T) with A the classical CIR function, and the Riccati equation
    B' = 1 - k B - sigma^2 B^2 / 2 gives I2 = (2/sigma^2) (T - k I1 - B(T)) without any further integration."""
    k_, sig_, Kc_, mc_ = sp.symbols("k sigma K_cc m_c", positive=True)
    symbols = dict(r0=r0, k=k_, sigma=sig_, T=T, thetabar=thbar, Kcc=Kc_, mc=mc_)

    def __init__(self):
        k, sig = self.k_, self.sig_
        h = sp.sqrt(k ** 2 + 2 * sig ** 2)
        E = sp.exp(h * T)
        BT = 2 * (E - 1) / ((h + k) * (E - 1) + 2 * h)
        logA = sp.log(2 * h * sp.exp((h + k) * T / 2) / ((h + k) * (E - 1) + 2 * h))
        I = {1: -2 / sig ** 2 * logA}
        I[2] = 2 / sig ** 2 * (T - k * I[1] - BT)
        self.h, self.B = h, BT
        self.terms = {"state": -BT * r0, "averaged": -k * thbar * I[1], "green_kubo": self.Kc_ * I[2],
                      "memory": sp.log(1 + self.mc_ * BT)}
        self.logPrice = sum(self.terms.values()); self.price = sp.exp(self.logPrice)
        self._fn = {}

    @staticmethod
    def coefficients(chain, k, thetas, regime=0):
        import numpy as np
        from .firstorder import green_kubo
        c = -k * np.asarray(thetas, float)
        K, M = green_kubo(chain, [c]); pi = chain.stationaryDistribution()
        return dict(thetabar=float(pi @ thetas), Kcc=float(K[0, 0]), mc=float(-M[0, regime]))

    def greek(self, *wrt):
        e = self.price
        for name in wrt:
            e = sp.diff(e, self.symbols[name])
        return e

    def evaluate(self, expr, **values):
        key = sp.srepr(expr)
        if key not in self._fn:
            self._fn[key] = sp.lambdify(list(self.symbols.values()), expr, "math")
        return float(self._fn[key](*[values[n] for n in self.symbols]))


# ---------------------------------------------------------------- Vasicek with jumps at a switching intensity, first order
class VasicekJumpsBondFirstOrder:
    """Vasicek with exponential jumps of mean m at intensity l_y (SwitchingVasicekJumps) under any finite chain,
    first order in the holding time, as a formula. The forcing g_i = c_i B + d_i B^2 + l_i J with c_i = -kappa theta_i,
    d_i = sigma_i^2 / 2 and J(t) = 1 / (1 + m B(t)) - 1, so three Green-Kubo entries per pair (symmetric parts) and three
    memory coefficients enter:

        log P_i = -B r0 + cbar I1 + dbar I2 + lbar IJ
                  + K_cc I2 + 2 K_cd I3 + K_dd I4 + 2 K_cl IJB + 2 K_dl IJB2 + K_ll IJJ
                  + log(1 + m_c B + m_d B^2 + m_l J),

    every integral over [0, T] closed by the substitution u = e^{-kappa t} (rational integrands). Building the
    formula takes about half a minute of sympy."""
    m_, cbar, dbar, lbar = sp.symbols("m cbar dbar lbar", real=True)
    Kcl, Kdl, Kll, ml = sp.symbols("K_cl K_dl K_ll m_l", real=True)
    symbols = dict(r0=r0, kappa=kappa, m=m_, T=T, cbar=cbar, dbar=dbar, lbar=lbar, Kcc=Kcc, Kcd=Kcd, Kdd=Kdd,
                   Kcl=Kcl, Kdl=Kdl, Kll=Kll, mc=mc, md=md, ml=ml)

    def __init__(self):
        u = sp.symbols("u", positive=True)                                    # u = e^{-kappa t}, dt = -du / (kappa u)
        Bu = (1 - u) / kappa; Ju = 1 / (1 + self.m_ * Bu) - 1
        def I(expr):
            return sp.integrate(sp.apart(sp.together(expr / (kappa * u)), u), (u, sp.exp(-kappa * T), 1))
        I1, I2, I3, I4 = (I(Bu ** n) for n in (1, 2, 3, 4))
        IJ, IJB, IJB2, IJJ = I(Ju), I(Ju * Bu), I(Ju * Bu ** 2), I(Ju * Ju)
        BT = Bu.subs(u, sp.exp(-kappa * T)); JT = Ju.subs(u, sp.exp(-kappa * T))
        self.B, self.J = BT, JT
        self.terms = {
            "state": -BT * r0,
            "averaged": self.cbar * I1 + self.dbar * I2 + self.lbar * IJ,
            "green_kubo": Kcc * I2 + 2 * Kcd * I3 + Kdd * I4 + 2 * self.Kcl * IJB + 2 * self.Kdl * IJB2 + self.Kll * IJJ,
            "memory": sp.log(1 + mc * BT + md * BT ** 2 + self.ml * JT),
        }
        self.logPrice = sum(self.terms.values()); self.price = sp.exp(self.logPrice)
        self._fn = {}

    @staticmethod
    def coefficients(chain, kappa_, thetas, sigmas, intensities, regime=0):
        import numpy as np
        from .firstorder import green_kubo
        c = -kappa_ * np.asarray(thetas, float); d = 0.5 * np.asarray(sigmas, float) ** 2; l = np.asarray(intensities, float)
        K, M = green_kubo(chain, [c, d, l]); pi = chain.stationaryDistribution(); sym = lambda i, j: float(0.5 * (K[i, j] + K[j, i]))
        return dict(cbar=float(pi @ c), dbar=float(pi @ d), lbar=float(pi @ l), Kcc=float(K[0, 0]), Kcd=sym(0, 1), Kdd=float(K[1, 1]),
                    Kcl=sym(0, 2), Kdl=sym(1, 2), Kll=float(K[2, 2]), mc=float(-M[0, regime]), md=float(-M[1, regime]), ml=float(-M[2, regime]))

    def greek(self, *wrt):
        e = self.price
        for name in wrt:
            e = sp.diff(e, self.symbols[name])
        return e

    def evaluate(self, expr, **values):
        """The antiderivatives carry logarithms whose arguments can be negative between the limits; their imaginary
        parts cancel, so the expression is evaluated in complex arithmetic and the real part returned."""
        key = sp.srepr(expr)
        if key not in self._fn:
            self._fn[key] = sp.lambdify(list(self.symbols.values()), expr, "mpmath")
        import mpmath
        v = self._fn[key](*[mpmath.mpc(values[n]) for n in self.symbols])
        assert abs(mpmath.im(v)) < 1e-9 * max(1.0, abs(mpmath.re(v))), "imaginary parts did not cancel"
        return float(mpmath.re(v))
