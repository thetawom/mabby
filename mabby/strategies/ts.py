from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from mabby.strategies.exceptions import StrategyUsageError
from mabby.strategies.strategy import Strategy
from mabby.utils import random_argmax


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
