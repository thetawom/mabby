from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from mabby.exceptions import BanditUsageError


class Bandit(ABC):
    def __init__(self, name: str | None = None):
        self._name = name
        self._primed = False
        self._choice: int | None = None

    @property
    def name(self) -> str:
        if self._name is None:
            return self.default_name()
        return self._name

    @abstractmethod
    def default_name(self) -> str:
        pass

    def prime(self, k: int, steps: int) -> None:
        self._primed = True
        self._choice = None
        self._prime(k, steps)

    def choose(self, rng: Generator) -> int:
        if not self._primed:
            raise BanditUsageError("choose() can only be called on a primed bandit")
        self._choice = self._choose(rng)
        return self._choice

    def update(self, reward: float) -> None:
        if self._choice is None:
            raise BanditUsageError("update() can only be called after choose()")
        self._update(self._choice, reward)
        self._choice = None

    @property
    def Qs(self) -> NDArray[np.float64]:
        if not self._primed:
            raise BanditUsageError("bandit has no Q values before it is run")
        return self.compute_Qs()

    @abstractmethod
    def _prime(self, k: int, steps: int) -> None:
        pass

    @abstractmethod
    def _choose(self, rng: Generator) -> int:
        pass

    @abstractmethod
    def _update(self, choice: int, reward: float) -> None:
        pass

    @abstractmethod
    def compute_Qs(self) -> NDArray[np.float64]:
        pass


class EpsilonGreedyBandit(Bandit):
    def __init__(self, eps: float, name: str | None = None):
        super().__init__(name)
        if eps < 0 or eps > 1:
            raise ValueError("eps must be between 0 and 1")
        self.eps = eps

    def default_name(self) -> str:
        return f"epsilon-greedy (eps={self.eps})"

    def _prime(self, k: int, steps: int) -> None:
        self._Qs = np.zeros(k)
        self._Ns = np.zeros(k)

    def _choose(self, rng: Generator) -> int:
        if rng.random() < self.eps:
            return rng.integers(0, len(self._Ns))
        return int(np.argmax(self._Qs))

    def _update(self, choice: int, reward: float) -> None:
        self._Ns[choice] += 1
        self._Qs[choice] += (reward - self._Qs[choice]) / self._Ns[choice]

    def compute_Qs(self) -> NDArray[np.float64]:
        return self._Qs
