"""The formulas shared by the model pages, as plain numpy. Every page evaluates its closed form with these and prints
it next to the library's price."""
import numpy as np
import regimelib as rl


def second_order(int_gbar, int_gt2, gt_T, gt_0, dgt_T, lam, sign=+1):
    """a_i(T) for a symmetric two-state chain through eps^2 = 1 / lam^2. sign=+1 is a start in regime 0."""
    eps = 1.0 / lam
    return (np.exp(int_gbar + eps / 2 * int_gt2 - eps ** 2 / 8 * (gt_T ** 2 + gt_0 ** 2))
            * (1 + sign * eps / 2 * gt_T - sign * eps ** 2 / 4 * dgt_T))


def exact_two_state(g1, g2, lam, T, sign=+1):
    """a_i(T) when the forcing is constant in time: exact, no expansion."""
    gbar, gt = (g1 + g2) / 2, (g1 - g2) / 2
    s = np.sqrt(lam ** 2 + gt ** 2)
    return np.exp((gbar - lam) * T) * (np.cosh(s * T) + (lam + sign * gt) / s * np.sinh(s * T))


def half(x):
    """Half-sum and half-difference across the two regimes."""
    return (x[0] + x[1]) / 2, (x[0] - x[1]) / 2


def powers_of_B(a, T):
    """E = e^{-aT}, B = (1 - E) / a and I_n = int_0^T B(t)^n dt for n = 1..4."""
    E = np.exp(-a * T)
    I1 = (T - (1 - E) / a) / a
    I2 = (T - 2 * (1 - E) / a + (1 - E ** 2) / (2 * a)) / a ** 2
    I3 = (T - 3 * (1 - E) / a + 3 * (1 - E ** 2) / (2 * a) - (1 - E ** 3) / (3 * a)) / a ** 3
    I4 = (T - 4 * (1 - E) / a + 3 * (1 - E ** 2) / a - 4 * (1 - E ** 3) / (3 * a) + (1 - E ** 4) / (4 * a)) / a ** 4
    return E, (1 - E) / a, I1, I2, I3, I4


def lewis_call(phi, S0, K, r, q, T, U=60.0, n=400):
    """European call from the characteristic function phi(u) of the martingale log return X, S_T = F e^X."""
    F = S0 * np.exp((r - q) * T)
    k = np.log(F / K)
    x, w = np.polynomial.legendre.leggauss(n)
    u, w = (x + 1) * U / 2, w * U / 2
    integral = np.sum(w * (np.exp(1j * u * k) * phi(u - 0.5j)).real / (u * u + 0.25))
    return np.exp(-r * T) * (F - np.sqrt(F * K) / np.pi * integral)


def library_bond(model, T, regime=0):
    bond = rl.ZeroCouponBond(T)
    bond.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=regime))
    return bond.NPV()


def library_call(model, K, T, regime=0):
    option = rl.VanillaOption(("call", K), maturity=T)
    option.setPricingEngine(rl.NumericalSwitchingEngine(model, regime=regime))
    return option.NPV()
