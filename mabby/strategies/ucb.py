from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from mabby.strategies.strategy import Strategy
from mabby.utils import random_argmax


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
