from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from mabby.core.exceptions import AgentUsageError

if TYPE_CHECKING:
    from mabby.strategies import Strategy


class Agent:
    _rng: Generator

    def __init__(self, strategy: Strategy, name: str | None = None):
        self.strategy = strategy
        self._name = name
        self._primed = False
        self._choice: int | None = None

    def __repr__(self) -> str:
        if self._name is None:
            return str(self.strategy)
        return self._name

    def prime(self, k: int, steps: int, rng: Generator) -> None:
        self._primed = True
        self._choice = None
        self._rng = rng
        self.strategy.prime(k, steps)

    def choose(self) -> int:
        if not self._primed:
            raise AgentUsageError("choose() can only be called on a primed agent")
        self._choice = self.strategy.choose(self._rng)
        return self._choice

    def update(self, reward: float) -> None:
        if self._choice is None:
            raise AgentUsageError("update() can only be called after choose()")
        self.strategy.update(self._choice, reward, self._rng)
        self._choice = None

    @property
    def Qs(self) -> NDArray[np.float64]:
        if not self._primed:
            raise AgentUsageError("agent has no Q values before it is run")
        return self.strategy.Qs

    @property
    def Ns(self) -> NDArray[np.uint32]:
        if not self._primed:
            raise AgentUsageError("agent has no Q values before it is run")
        return self.strategy.Ns
