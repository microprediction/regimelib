"""Certificate for Theorem 1: the first-order terms and the initial layer. The residual after the order-one
approximation a0 [1 + eps (int K - m_i)] + eps e^{Q1 tau/eps} Q1# g~(0) is O(eps^2) uniformly on [0, T], for a
three-state chain with a Vasicek forcing (vanishing at tau = 0) and with a constant forcing (where the layer is
needed: without it the residual is O(eps))."""
import math
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp, quad

Q1 = np.array([[-5.0, 3.0, 2.0], [4.0, -9.0, 5.0], [1.0, 6.0, -7.0]]); n = 3
Q1 = Q1 * (n / -np.trace(Q1))                                   # unit scale: mean holding time one
w, v = np.linalg.eig(Q1.T); pi = np.real(v[:, np.argmin(abs(w))]); pi /= pi.sum()
Qs = np.linalg.inv(Q1 - np.outer(np.ones(n), pi)) + np.outer(np.ones(n), pi)


def check(g, T, label):
    gbar = lambda t: pi @ g(t); gt = lambda t: g(t) - gbar(t); K = lambda t: -pi @ (gt(t) * (Qs @ gt(t)))
    ratios, ratios_nolayer = [], []
    for eps in (0.2, 0.1, 0.05, 0.025):
        sol = solve_ivp(lambda t, a: (Q1 / eps + np.diag(g(t))) @ a, (0, T), np.ones(n), rtol=1e-12, atol=1e-14, dense_output=True)
        worst = worst_nolayer = 0.0
        for tau in np.linspace(0.0, T, 41):
            a = sol.sol(tau); a0 = math.exp(quad(gbar, 0, tau)[0]); IK = quad(K, 0, tau)[0]; m = Qs @ gt(tau)
            outer = a0 * (1 + eps * (IK - m)); layer = expm(Q1 * tau / eps) @ (Qs @ gt(0))
            worst = max(worst, np.max(abs(a - outer - eps * layer))); worst_nolayer = max(worst_nolayer, np.max(abs(a - outer)))
        ratios.append(worst / eps ** 2); ratios_nolayer.append(worst_nolayer / eps)
    print(f"{label}: sup residual / eps^2 = {', '.join(f'{r:.4f}' for r in ratios)}; without the layer, / eps = {', '.join(f'{r:.4f}' for r in ratios_nolayer)}")
    assert max(ratios) < 2 * min(ratios), "the residual is not O(eps^2)"
    return ratios_nolayer


kappa, thetas, sigmas = 0.5, np.array([0.08, 0.05, 0.01]), np.array([0.015, 0.01, 0.006])
B = lambda t: (1 - math.exp(-kappa * t)) / kappa
check(lambda t: -kappa * thetas * B(t) + 0.5 * sigmas ** 2 * B(t) ** 2, 3.0, "Vasicek forcing")
r = check(lambda t: np.array([-0.06, -0.02, 0.03]), 2.0, "constant forcing")
assert min(r) > 0.01, "the layer should matter for a forcing that does not vanish at tau = 0"
print("PASS")
