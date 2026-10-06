"""A finite Markov chain of regimes, given by its generator."""
import math
import numpy as np


def stateIndex(value, n, name="regime"):
    """A regime label: an integer in 0, ..., n - 1. Negative values are refused rather than counted from the end."""
    import numbers
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer regime index, got {value!r}")
    if not 0 <= value < n:
        raise ValueError(f"{name} must be between 0 and {n - 1}, got {value}")
    return int(value)


def _gth(Q):
    """Stationary law of an irreducible generator by state reduction: only additions, multiplications and divisions of
    nonnegative numbers, so no cancellation whatever the spread of the rates."""
    A = np.array(Q, float); n = A.shape[0]
    for k in range(n - 1, 0, -1):
        A[:k, k] /= A[k, :k].sum()
        A[:k, :k] += np.outer(A[:k, k], A[k, :k])
    pi = np.zeros(n); pi[0] = 1.0
    for k in range(1, n):
        pi[k] = pi[:k] @ A[:k, k]
    return pi / pi.sum()


class RegimeChain:
    def __init__(self, generator):
        raw = np.array(generator)                                  # a copy: the chain owns its generator
        if np.iscomplexobj(raw):
            if np.any(raw.imag != 0):
                raise ValueError("generator entries must be real")
            raw = raw.real
        Q = np.array(raw, dtype=float)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("the generator must be a square matrix")
        if Q.shape[0] == 0:
            raise ValueError("the generator must contain at least one regime")
        if not np.all(np.isfinite(Q)):
            raise ValueError("generator entries must be finite")
        if np.any(Q - np.diag(np.diag(Q)) < 0):
            raise ValueError("off-diagonal rates must be nonnegative")
        # rows sum to zero relative to the size of the row's own rates, so the check does not depend on the time unit
        if np.any(np.abs(Q.sum(axis=1)) > 1e-9 * np.abs(Q).sum(axis=1)):
            raise ValueError("generator rows must sum to zero")
        Q.setflags(write=False)                                    # validated once; replace the chain to change it
        self.generator = Q

    def numberOfRegimes(self):
        return self.generator.shape[0]

    def closedClasses(self):
        """The sets of regimes that communicate and cannot be left. An irreducible chain has one, all of it; a chain
        with an absorbing regime has one, that regime; a chain with more than one has no unique stationary law."""
        from scipy.sparse.csgraph import connected_components
        Q = self.generator; n = Q.shape[0]
        adjacency = (Q - np.diag(np.diag(Q))) > 0
        count, label = connected_components(adjacency, directed=True, connection="strong")
        classes = [np.flatnonzero(label == c) for c in range(count)]
        return [c for c in classes if not adjacency[np.ix_(c, np.setdiff1d(np.arange(n), c))].any()]

    def stationaryDistribution(self):
        """The stationary law, computed without subtraction (Grassmann, Taksar and Heyman), so it is nonnegative and
        accurate on stiff chains. With several closed classes there is no unique one: this returns the long-run law
        from a uniform start, and the expansion engines, which need uniqueness, refuse such a chain."""
        Q = self.generator; n = Q.shape[0]
        closed = self.closedClasses()
        pi = np.zeros(n)
        weights = [1.0]
        if len(closed) > 1:                                    # share of a uniform start that ends in each class
            inside = np.concatenate(closed); transient = np.setdiff1d(np.arange(n), inside)
            weights = []
            for c in closed:
                absorbed = 0.0
                if len(transient):
                    absorbed = float(np.linalg.solve(Q[np.ix_(transient, transient)], -Q[np.ix_(transient, c)].sum(axis=1)).sum())
                weights.append((len(c) + absorbed) / n)
        for c, weight in zip(closed, weights):
            pi[c] = weight * _gth(Q[np.ix_(c, c)])
        return pi

    def meanHoldingTime(self):
        """The expansion scale of the engine: n / -trace Q."""
        rate = -float(np.trace(self.generator))
        return math.inf if rate == 0.0 else self.numberOfRegimes() / rate      # a chain that never switches

    def transitionMatrix(self, dt):
        """P(dt) = exp(Q dt): the probabilities of each regime a time dt later, by row of the regime now."""
        from scipy.linalg import expm
        if not math.isfinite(dt) or dt < 0:
            raise ValueError(f"dt must be finite and nonnegative, got {dt}")
        return expm(self.generator * dt)

    @staticmethod
    def fromTransitionMatrix(P, dt):
        """The chain whose transition matrix over a step dt is P, for a matrix estimated at a data frequency (rows
        are the regime now and sum to one). Not every transition matrix comes from a continuous-time chain: for two
        regimes it does exactly when p11 + p22 > 1, and in general when its principal logarithm is a generator. A
        matrix that does not is refused; how to move it to one that does is a modelling choice left to the caller."""
        from scipy.linalg import logm, expm
        P = np.array(P, dtype=float)
        if P.ndim != 2 or P.shape[0] != P.shape[1] or P.shape[0] == 0:
            raise ValueError("the transition matrix must be square with at least one regime")
        if not (math.isfinite(dt) and dt > 0):
            raise ValueError(f"dt must be finite and positive, got {dt}")
        if not np.all(np.isfinite(P)) or np.any(P < 0) or np.any(np.abs(P.sum(axis=1) - 1.0) > 1e-9):
            raise ValueError("a transition matrix has nonnegative entries and rows that sum to one")
        n = P.shape[0]
        if n == 2:                                             # in closed form: exp(Q dt) has second eigenvalue 1 - p - q
            p, q = P[0, 1], P[1, 0]
            if p + q >= 1.0:
                raise ValueError(f"no continuous-time chain has this transition matrix: it needs p11 + p22 > 1, and "
                                 f"p11 + p22 = {P[0, 0] + P[1, 1]:.6g}")
            rate = 0.0 if p + q == 0.0 else -math.log1p(-(p + q)) / dt / (p + q)
            return RegimeChain.twoState(rate * p, rate * q)
        with np.errstate(all="ignore"):
            L = logm(P)
        Q = np.real(L) / dt
        off = Q - np.diag(np.diag(Q))
        scale = max(np.abs(Q).max(), 1e-300)
        if (not np.all(np.isfinite(L)) or np.abs(np.imag(L)).max() > 1e-9 * scale * dt
                or off.min() < -1e-9 * scale):
            raise ValueError("no continuous-time chain has this transition matrix: its principal logarithm is not a "
                             "generator (an off-diagonal rate would be negative or complex)")
        off = np.maximum(off, 0.0)
        Q = off - np.diag(off.sum(axis=1))
        if np.abs(expm(Q * dt) - P).max() > 1e-8:
            raise ValueError("the logarithm of this transition matrix does not reproduce it as a generator")
        return RegimeChain(Q)

    @staticmethod
    def twoState(rate12, rate21):
        return RegimeChain([[-rate12, rate12], [rate21, -rate21]])
