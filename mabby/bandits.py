from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from mabby.exceptions import BanditUsageError
from mabby.strategies import EpsilonGreedyStrategy, RandomStrategy, Strategy


class Bandit:
    def __init__(self, strategy: Strategy, name: str | None = None):
        self.strategy = strategy
        self._name = name
        self._primed = False
        self._choice: int | None = None

    @property
    def name(self) -> str:
        if self._name is None:
            return str(self.strategy)
        return self._name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self.strategy)

    def prime(self, k: int, steps: int) -> None:
        self._primed = True
        self._choice = None
        self.strategy.prime(k, steps)

    def choose(self, rng: Generator) -> int:
        if not self._primed:
            raise BanditUsageError("choose() can only be called on a primed bandit")
        self._choice = self.strategy.choose(rng)
        return self._choice

    def update(self, reward: float) -> None:
        if self._choice is None:
            raise BanditUsageError("update() can only be called after choose()")
        self.strategy.update(self._choice, reward)
        self._choice = None

    @property
    def Qs(self) -> NDArray[np.float64]:
        if not self._primed:
            raise BanditUsageError("bandit has no Q values before it is run")
        return self.strategy.Qs


class RandomBandit(Bandit):
    def __init__(self, name: str | None = None):
        strategy = RandomStrategy()
        super().__init__(strategy=strategy, name=name)


class EpsilonGreedyBandit(Bandit):
    def __init__(self, eps: float, name: str | None = None):
        strategy = EpsilonGreedyStrategy(eps=eps)
        super().__init__(strategy=strategy, name=name)
