from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from mabby.agents import Agent
from mabby.exceptions import StrategyUsageError
from mabby.utils import random_argmax


class Strategy(ABC):
    def __init__(self, **kwargs: float) -> None:
        super().__init__()

    @abstractmethod
    def prime(self, k: int, steps: int) -> None:
        """Set up agent before a trial run"""

    @abstractmethod
    def choose(self, rng: Generator) -> int:
        """Choose an arm to play"""

    @abstractmethod
    def update(self, choice: int, reward: float, rng: Generator | None = None) -> None:
        """Update estimates based on reward observation"""

    @property
    @abstractmethod
    def Qs(self) -> NDArray[np.float64]:
        """Compute action value estimates for each arm"""

    @property
    @abstractmethod
    def Ns(self) -> NDArray[np.uint32]:
        """Count number of plays for each arm"""

    def agent(self, **kwargs: str) -> Agent:
        return Agent(strategy=self, **kwargs)


class SemiUniformStrategy(Strategy, ABC):
    _Qs: NDArray[np.float64]
    _Ns: NDArray[np.uint32]

    def __init__(self, **kwargs: float) -> None:
        super().__init__()

    def prime(self, k: int, steps: int) -> None:
        self._Qs = np.zeros(k, dtype=np.float64)
        self._Ns = np.zeros(k, dtype=np.uint32)

    def choose(self, rng: Generator) -> int:
        if rng.random() < self.effective_eps():
            return self._explore(rng=rng)
        return self._exploit(rng=rng)

    def _explore(self, rng: Generator) -> int:
        return rng.integers(0, len(self._Ns))

    def _exploit(self, rng: Generator) -> int:
        return random_argmax(self._Qs, rng=rng)

    def update(self, choice: int, reward: float, rng: Generator | None = None) -> None:
        self._Ns[choice] += 1
        self._Qs[choice] += (reward - self._Qs[choice]) / self._Ns[choice]

    @property
    def Qs(self) -> NDArray[np.float64]:
        return self._Qs

    @property
    def Ns(self) -> NDArray[np.uint32]:
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
        return f"eps-greedy (eps={self.eps})"

    def effective_eps(self) -> float:
        return self.eps


class UCB1Strategy(Strategy):
    _t: int
    _Qs: NDArray[np.float64]
    _Ns: NDArray[np.uint32]

    def __init__(self, alpha: float) -> None:
        super().__init__()
        if alpha < 0:
            raise ValueError("alpha must be greater than 0")
        self.alpha = alpha

    def __repr__(self) -> str:
        return f"ucb1 (alpha={self.alpha})"

    def prime(self, k: int, steps: int) -> None:
        self._t = 0
        self._Qs = np.zeros(k, dtype=np.float64)
        self._Ns = np.zeros(k, dtype=np.uint32)

    def choose(self, rng: Generator) -> int:
        if self._t < len(self._Ns):
            return self._t
        return random_argmax(self._compute_UCBs(), rng=rng)

    def _compute_UCBs(self) -> NDArray[np.float64]:
        return self._Qs + self.alpha * np.sqrt(np.log(self._t) / self._Ns)

    def update(self, choice: int, reward: float, rng: Generator | None = None) -> None:
        self._t += 1
        self._Ns[choice] += 1
        self._Qs[choice] += (reward - self._Qs[choice]) / self._Ns[choice]

    @property
    def Qs(self) -> NDArray[np.float64]:
        return self._Qs

    @property
    def Ns(self) -> NDArray[np.uint32]:
        return self._Ns


class BetaTSStrategy(Strategy):
    _a: NDArray[np.uint32]
    _b: NDArray[np.uint32]

    def __init__(self, general: bool = False):
        super().__init__()
        self.general = general

    def __repr__(self) -> str:
        return f"{'generalized ' if self.general else ''}beta ts"

    def prime(self, k: int, steps: int) -> None:
        self._a = np.ones(k, dtype=np.uint32)
        self._b = np.ones(k, dtype=np.uint32)

    def choose(self, rng: Generator) -> int:
        samples = rng.beta(a=self._a, b=self._b)
        return random_argmax(samples, rng=rng)

    def update(self, choice: int, reward: float, rng: Generator | None = None) -> None:
        if rng is None:
            raise StrategyUsageError("TS strategies require rng")
        if self.general and (reward > 1 or reward < 0):
            raise StrategyUsageError(
                "general Beta TS agents can only be used with rewards from 0 to 1"
            )
        if not self.general and reward != 0 and reward != 1:
            raise StrategyUsageError(
                "Beta TS agents can only be used with Bernoulli rewards"
            )
        pseudo_reward = rng.binomial(n=1, p=reward) if self.general else reward
        self._a[choice] += pseudo_reward
        self._b[choice] += 1 - pseudo_reward

    @property
    def Qs(self) -> NDArray[np.float64]:
        return self._a / (self._a + self._b)

    @property
    def Ns(self) -> NDArray[np.uint32]:
        return ((self._a + self._b).astype(np.int64) - 2).astype(np.uint32)
