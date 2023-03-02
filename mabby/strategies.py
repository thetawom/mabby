from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from mabby.exceptions import BanditUsageError


class Strategy(ABC):
    def __init__(self, **kwargs: float) -> None:
        super().__init__()

    @abstractmethod
    def prime(self, k: int, steps: int) -> None:
        """Set up bandit before a trial run"""

    @abstractmethod
    def choose(self, rng: Generator | None = None) -> int:
        """Choose an arm to play"""

    @abstractmethod
    def update(self, choice: int, reward: float) -> None:
        """Update estimates based on reward observation"""

    @property
    @abstractmethod
    def Qs(self) -> NDArray[np.float64]:
        """Compute action value estimates for each arm"""

    @property
    @abstractmethod
    def Ns(self) -> NDArray[np.int32]:
        """Count number of plays for each arm"""


class SemiUniformStrategy(Strategy, ABC):
    _Qs: NDArray[np.float64]
    _Ns: NDArray[np.int32]

    def __init__(self, **kwargs: float) -> None:
        super().__init__()

    def prime(self, k: int, steps: int) -> None:
        self._Qs = np.zeros(k, dtype=np.float64)
        self._Ns = np.zeros(k, dtype=np.int32)

    def choose(self, rng: Generator | None = None) -> int:
        if rng is None:
            raise BanditUsageError("semi-uniform bandits require rng")
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

    @property
    def Ns(self) -> NDArray[np.int32]:
        return self._Ns

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


class UCB1Strategy(Strategy):
    _t: int
    _Qs: NDArray[np.float64]
    _Ns: NDArray[np.int32]

    def __init__(self, alpha: float) -> None:
        super().__init__()
        if alpha < 0:
            raise ValueError("alpha must be greater than 0")
        self.alpha = alpha

    def __repr__(self) -> str:
        return f"ucb-1 (alpha={self.alpha})"

    def prime(self, k: int, steps: int) -> None:
        self._t = 0
        self._Qs = np.zeros(k, dtype=np.float64)
        self._Ns = np.zeros(k, dtype=np.int32)

    def choose(self, rng: Generator | None = None) -> int:
        if self._t < len(self._Ns):
            return self._t
        return int(np.argmax(self._compute_UCBs()))

    def _compute_UCBs(self) -> NDArray[np.float64]:
        return self._Qs + self.alpha * np.sqrt(np.log(self._t) / self._Ns)

    def update(self, choice: int, reward: float) -> None:
        self._t += 1
        self._Ns[choice] += 1
        self._Qs[choice] += (reward - self._Qs[choice]) / self._Ns[choice]

    @property
    def Qs(self) -> NDArray[np.float64]:
        return self._Qs

    @property
    def Ns(self) -> NDArray[np.int32]:
        return self._Ns
