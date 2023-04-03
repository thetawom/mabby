"""Provides implementations of Thompson sampling strategies."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from overrides import override

from mabby.exceptions import StrategyUsageError
from mabby.strategies.strategy import Strategy
from mabby.utils import random_argmax


class BetaTSStrategy(Strategy):
    """Thompson sampling strategy with Beta priors."""

    _a: NDArray[np.uint32]
    _b: NDArray[np.uint32]

    def __init__(self, general: bool = False):
        """Initializes a Beta Thompson sampling strategy.

        If ``general`` is ``False``, rewards used for updates must be either 0 or 1.
        Otherwise, rewards must be with support [0, 1].

        Args:
            general: Whether to use a generalized version of the strategy.
        """
        self.general = general

    @override
    def __repr__(self) -> str:
        return f"{'generalized ' if self.general else ''}beta ts"

    @override
    def prime(self, k: int, steps: int) -> None:
        self._a = np.ones(k, dtype=np.uint32)
        self._b = np.ones(k, dtype=np.uint32)

    @override
    def choose(self, rng: Generator) -> int:
        samples = rng.beta(a=self._a, b=self._b)
        return random_argmax(samples, rng=rng)

    @override
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
    @override
    def Qs(self) -> NDArray[np.float64]:
        return self._a / (self._a + self._b)

    @property
    @override
    def Ns(self) -> NDArray[np.uint32]:
        return ((self._a + self._b).astype(np.int64) - 2).astype(np.uint32)
