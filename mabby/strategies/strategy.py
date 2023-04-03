"""Provides [`Strategy`][mabby.strategies.strategy.Strategy] class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from overrides import EnforceOverrides

from mabby.agent import Agent


class Strategy(ABC, EnforceOverrides):
    """Base class for a bandit strategy.

    A strategy provides the computational logic for choosing which bandit arms to play
    and updating parameter estimates.
    """

    @abstractmethod
    def __init__(self) -> None:
        """Initializes a bandit strategy."""

    @abstractmethod
    def __repr__(self) -> str:
        """Returns a string representation of the strategy."""

    @abstractmethod
    def prime(self, k: int, steps: int) -> None:
        """Primes the strategy before running a trial.

        Args:
            k: The number of bandit arms to choose from.
            steps: The number of steps to the simulation will be run.
        """

    @abstractmethod
    def choose(self, rng: Generator) -> int:
        """Returns the next arm to play.

        Args:
            rng: A random number generator.

        Returns:
            The index of the arm to play.
        """

    @abstractmethod
    def update(self, choice: int, reward: float, rng: Generator | None = None) -> None:
        """Updates internal parameter estimates based on reward observation.

        Args:
            choice: The most recent choice made.
            reward: The observed reward from the agent's most recent choice.
            rng: A random number generator.
        """

    @property
    @abstractmethod
    def Qs(self) -> NDArray[np.float64]:
        """The current estimated action values for each arm."""

    @property
    @abstractmethod
    def Ns(self) -> NDArray[np.uint32]:
        """The number of times each arm has been played."""

    def agent(self, **kwargs: str) -> Agent:
        """Creates an agent following the strategy.

        Args:
            **kwargs: Parameters for initializing the agent (see
                [`Agent`][mabby.agent.Agent])

        Returns:
            The created agent with the strategy.
        """
        return Agent(strategy=self, **kwargs)
