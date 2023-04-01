"""Provides implementations of upper confidence bound (UCB) strategies."""
from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from overrides import override

from mabby.strategies.strategy import Strategy
from mabby.utils import random_argmax


class UCB1Strategy(Strategy):
    """Strategy using the UCB1 bandit algorithm."""

    _t: int
    _Qs: NDArray[np.float64]
    _Ns: NDArray[np.uint32]

    def __init__(self, alpha: float) -> None:
        """Initializes a UCB1 strategy.

        Args:
            alpha: The exploration parameter.
        """
        if alpha < 0:
            raise ValueError("alpha must be greater than 0")
        self.alpha = alpha

    @override
    def __repr__(self) -> str:
        return f"ucb1 (alpha={self.alpha})"

    @override
    def prime(self, k: int, steps: int) -> None:
        self._t = 0
        self._Qs = np.zeros(k, dtype=np.float64)
        self._Ns = np.zeros(k, dtype=np.uint32)

    @override
    def choose(self, rng: Generator) -> int:
        if self._t < len(self._Ns):
            return self._t
        return random_argmax(self._compute_UCBs(), rng=rng)

    def _compute_UCBs(self) -> NDArray[np.float64]:
        return self._Qs + self.alpha * np.sqrt(np.log(self._t) / self._Ns)

    @override
    def update(self, choice: int, reward: float, rng: Generator | None = None) -> None:
        self._t += 1
        self._Ns[choice] += 1
        self._Qs[choice] += (reward - self._Qs[choice]) / self._Ns[choice]

    @property
    @override
    def Qs(self) -> NDArray[np.float64]:
        return self._Qs

    @property
    @override
    def Ns(self) -> NDArray[np.uint32]:
        return self._Ns
