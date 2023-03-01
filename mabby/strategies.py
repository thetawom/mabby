from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray


class Strategy(ABC):
    def __init__(self, **kwargs: float) -> None:
        super().__init__()

    @abstractmethod
    def prime(self, k: int, steps: int) -> None:
        """Set up bandit before a trial run"""

    @abstractmethod
    def choose(self, rng: Generator) -> int:
        """Choose an arm to play"""

    @abstractmethod
    def update(self, choice: int, reward: float) -> None:
        """Update estimates based on reward observation"""

    @property
    @abstractmethod
    def Qs(self) -> NDArray[np.float64]:
        """Compute action value estimates for each arm"""


class SemiUniformStrategy(Strategy, ABC):
    _Qs: NDArray[np.float64]
    _Ns: NDArray[np.float64]

    def __init__(self, **kwargs: float) -> None:
        super().__init__()

    def prime(self, k: int, steps: int) -> None:
        self._Qs = np.zeros(k)
        self._Ns = np.zeros(k)

    def choose(self, rng: Generator) -> int:
        if rng.random() < self.effective_eps():
            return self._explore(rng=rng)
        return self._exploit()

    def _explore(self, rng: Generator) -> int:
        return rng.integers(0, len(self._Ns))

    def _exploit(self) -> int:
        return int(np.argmax(self._Qs))

    def update(self, choice: int, reward: float) -> None:
        self._Ns[choice] += 1
        self._Qs[choice] += (reward - self._Qs[choice]) / self._Ns[choice]

    @property
    def Qs(self) -> NDArray[np.float64]:
        return self._Qs

    @abstractmethod
    def effective_eps(self) -> float:
        """Compute effective epsilon value"""


class RandomStrategy(SemiUniformStrategy):
    def __repr__(self) -> str:
        return "random"

    def effective_eps(self) -> float:
        return 1


class EpsilonGreedyStrategy(SemiUniformStrategy):
    def __init__(self, eps: float) -> None:
        super().__init__()
        if eps < 0 or eps > 1:
            raise ValueError("eps must be between 0 and 1")
        self.eps = eps

    def __repr__(self) -> str:
        return f"epsilon-greedy (eps={self.eps})"

    def effective_eps(self) -> float:
        return self.eps
