"""A finite Markov chain of regimes, given by its generator."""
import numpy as np


def stateIndex(value, n, name="regime"):
    """A regime label: an integer in 0, ..., n - 1. Negative values are refused rather than counted from the end."""
    import numbers
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer regime index, got {value!r}")
    if not 0 <= value < n:
        raise ValueError(f"{name} must be between 0 and {n - 1}, got {value}")
    return int(value)


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

    def stationaryDistribution(self):
        w, v = np.linalg.eig(self.generator.T)
        pi = np.real(v[:, np.argmin(abs(w))])
        return pi / pi.sum()

    def meanHoldingTime(self):
        """The expansion scale of the engine: n / -trace Q."""
        return self.numberOfRegimes() / -np.trace(self.generator)

    @staticmethod
    def twoState(rate12, rate21):
        return RegimeChain([[-rate12, rate12], [rate21, -rate21]])
