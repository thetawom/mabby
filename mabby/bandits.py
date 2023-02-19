from abc import ABC, abstractmethod

import numpy as np

from mabby.exceptions import BanditUsageError


class Bandit(ABC):
    def __init__(self):
        self._primed = False
        self._choice = None

    def prime(self, k, rounds):
        self._primed = True
        self._choice = None
        self._prime(k, rounds)

    def choose(self):
        if not self._primed:
            raise BanditUsageError("choose() can only be called on a primed bandit")
        self._choice = self._choose()
        return self._choice

    def update(self, reward):
        if self._choice is None:
            raise BanditUsageError("update() can only be called after choose()")
        self._update(self._choice, reward)
        self._choice = None

    @property
    def Qs(self):
        if not self._primed:
            raise BanditUsageError("bandit has no Q values before it is run")
        return self.compute_Qs()

    @abstractmethod
    def _prime(self, k, rounds):
        pass

    @abstractmethod
    def _choose(self):
        pass

    @abstractmethod
    def _update(self, choice, reward):
        pass

    @abstractmethod
    def compute_Qs(self):
        pass


class EpsilonGreedyBandit(Bandit):
    def __init__(self, eps):
        super().__init__()
        if eps < 0 or eps > 1:
            raise ValueError("eps must be between 0 and 1")
        self.eps = eps
        self._Qs = None
        self._Ns = None

    def _prime(self, k, rounds):
        self._Qs = np.zeros(k)
        self._Ns = np.zeros(k)

    def _choose(self):
        if np.random.rand() < self.eps:
            return np.random.randint(0, len(self._Ns))
        return np.argmax(self._Qs)

    def _update(self, choice, reward):
        self._Ns[choice] += 1
        self._Qs[choice] += (reward - self._Qs[choice]) / self._Ns[choice]

    def compute_Qs(self):
        return self._Qs
