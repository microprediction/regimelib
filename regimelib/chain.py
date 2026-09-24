"""A finite Markov chain of regimes, given by its generator."""
import numpy as np


class RegimeChain:
    def __init__(self, generator):
        Q = np.asarray(generator, float)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("the generator must be a square matrix")
        if not np.allclose(Q.sum(axis=1), 0.0):
            raise ValueError("generator rows must sum to zero")
        if np.any(Q - np.diag(np.diag(Q)) < 0):
            raise ValueError("off-diagonal rates must be nonnegative")
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
