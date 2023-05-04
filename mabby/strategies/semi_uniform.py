"""Provides implementations of semi-uniform bandit strategies.

Semi-uniform strategies will choose to explore or exploit at each time step. When
exploring, a random arm will be played. When exploiting, the arm with the greatest
estimated action value will be played. ``epsilon``, the chance of exploration, is
computed differently with different semi-uniform strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from overrides import EnforceOverrides, override

from mabby.strategies.strategy import Strategy
from mabby.utils import random_argmax


class SemiUniformStrategy(Strategy, ABC, EnforceOverrides):
    """Base class for semi-uniform bandit strategies.

    Every semi-uniform strategy must implement
    [`effective_eps`][mabby.strategies.semi_uniform.SemiUniformStrategy.effective_eps]
    to compute the chance of exploration at each time step.
    """

    _Qs: NDArray[np.float64]
    _Ns: NDArray[np.uint32]

    def __init__(self) -> None:
        """Initializes a semi-uniform strategy."""

    @override
    def prime(self, k: int, steps: int) -> None:
        self._Qs = np.zeros(k, dtype=np.float64)
        self._Ns = np.zeros(k, dtype=np.uint32)

    @override
    def choose(self, rng: Generator) -> int:
        if rng.random() < self.effective_eps():
            return self._explore(rng=rng)
        return self._exploit(rng=rng)

    def _explore(self, rng: Generator) -> int:
        return rng.integers(0, len(self._Ns))

    def _exploit(self, rng: Generator) -> int:
        return random_argmax(self._Qs, rng=rng)

    @override
    def update(self, choice: int, reward: float, rng: Generator | None = None) -> None:
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

    @abstractmethod
    def effective_eps(self) -> float:
        """Returns the effective epsilon value.

        The effective epsilon value is the probability at the current time step that the
        bandit will explore rather than exploit. Depending on the strategy, the
        effective epsilon value may be different from the nominal epsilon value set.
        """


class RandomStrategy(SemiUniformStrategy):
    """Random bandit strategy.

    The random strategy chooses arms at random, i.e., it explores with 100% chance.
    """

    def __init__(self) -> None:
        """Initializes a random strategy."""
        super().__init__()

    @override
    def __repr__(self) -> str:
        return "random"

    @override
    def effective_eps(self) -> float:
        return 1


class EpsilonGreedyStrategy(SemiUniformStrategy):
    """Epsilon-greedy bandit strategy.

    The epsilon-greedy strategy has a fixed chance of exploration every time step.
    """

    def __init__(self, eps: float) -> None:
        """Initializes an epsilon-greedy strategy.

        Args:
            eps: The chance of exploration (must be between 0 and 1).
        """
        super().__init__()
        if eps < 0 or eps > 1:
            raise ValueError("eps must be between 0 and 1")
        self.eps = eps

    @override
    def __repr__(self) -> str:
        return f"eps-greedy (eps={self.eps})"

    @override
    def effective_eps(self) -> float:
        return self.eps


class EpsilonFirstStrategy(SemiUniformStrategy):
    """Epsilon-first bandit strategy.

    The epsilon-first strategy has a pure exploration phase followed by a pure
    exploitation phase.
    """

    _explore_steps_remaining: int

    def __init__(self, eps: float) -> None:
        """Initializes an epsilon-first strategy.

        Args:
            eps: The ratio of exploration steps (must be between 0 and 1).
        """
        super().__init__()
        if eps < 0 or eps > 1:
            raise ValueError("eps must be between 0 and 1")
        self.eps = eps

    @override
    def __repr__(self) -> str:
        return f"eps-first (eps={self.eps})"

    @override
    def prime(self, k: int, steps: int) -> None:
        super().prime(k, steps)
        self._explore_steps_remaining = int(self.eps * steps)

    @override
    def effective_eps(self) -> float:
        return float(self._explore_steps_remaining > 0)

    @override
    def update(self, choice: int, reward: float, rng: Generator | None = None) -> None:
        super().update(choice, reward, rng=rng)
        if self._explore_steps_remaining > 0:
            self._explore_steps_remaining -= 1
