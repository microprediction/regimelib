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
        self.logPrice = (-BT * r0 + I_gbar + eps / 2 * I_gt2 - eps ** 2 / 8 * gtT ** 2
                         + sp.log(1 + sign * eps / 2 * gtT - sign * eps ** 2 / 4 * gtpT))
        self.price = sp.exp(self.logPrice)
        self._fn = {}

    def greek(self, *wrt):
        """A formula: the derivative of the price with respect to the named symbols, e.g. greek('r0'), greek('r0', 'r0'),
        greek('theta1'), greek('lam'). Returns a sympy expression."""
        e = self.price
        for name in wrt:
            e = sp.diff(e, self.symbols[name])
        return e

    def evaluate(self, expr, **values):
        key = sp.srepr(expr)
        if key not in self._fn:
            self._fn[key] = sp.lambdify(list(self.symbols.values()), expr, "math")
        return float(self._fn[key](*[values[n] for n in self.symbols]))

    # QuantLib-named conveniences, as formulas
    def delta(self): return self.greek("r0")
    def gamma(self): return self.greek("r0", "r0")
    def theta(self): return -self.greek("T")
