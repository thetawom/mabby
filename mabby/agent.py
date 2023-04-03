"""Provides [`Agent`][mabby.agent.Agent] class for bandit simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from mabby.exceptions import AgentUsageError

if TYPE_CHECKING:
    from mabby.strategies import Strategy


class Agent:
    """Agent in a multi-armed bandit simulation.

    An agent represents an autonomous entity in a bandit simulation. It wraps around a
    specified strategy and provides an interface for each part of the decision-making
    process, including making a choice then updating internal parameter estimates based
    on the observed rewards from that choice.
    """

    _rng: Generator

    def __init__(self, strategy: Strategy, name: str | None = None):
        """Initializes an agent with a given strategy.

        Args:
            strategy: The bandit strategy to use.
            name: An optional name for the agent.
        """
        self.strategy: Strategy = strategy  #: The bandit strategy to use
        self._name = name
        self._primed = False
        self._choice: int | None = None

    def __repr__(self) -> str:
        """Returns the agent's string representation.

        Uses the agent's name if set. Otherwise, the string representation of the
        agent's strategy is used by default.
        """
        if self._name is None:
            return str(self.strategy)
        return self._name

    def prime(self, k: int, steps: int, rng: Generator) -> None:
        """Primes the agent before running a trial.

        Args:
            k: The number of bandit arms for the agent to choose from.
            steps: The number of steps to the simulation will be run.
            rng: A random number generator.
        """
        self._primed = True
        self._choice = None
        self._rng = rng
        self.strategy.prime(k, steps)

    def choose(self) -> int:
        """Returns the agent's next choice based on its strategy.

        This method can only be called on a primed agent.

        Returns:
            The index of the arm chosen by the agent.

        Raises:
            AgentUsageError: If the agent has not been primed.
        """
        if not self._primed:
            raise AgentUsageError("choose() can only be called on a primed agent")
        self._choice = self.strategy.choose(self._rng)
        return self._choice

    def update(self, reward: float) -> None:
        """Updates the agent's internal parameter estimates.

        This method can only be called if the agent has previously made a choice, and
        an update based on that choice has not already been made.

        Args:
            reward: The observed reward from the agent's most recent choice.

        Raises:
            AgentUsageError: If the agent has not previously made a choice.
        """
        if self._choice is None:
            raise AgentUsageError("update() can only be called after choose()")
        self.strategy.update(self._choice, reward, self._rng)
        self._choice = None

    @property
    def Qs(self) -> NDArray[np.float64]:
        """The agent's current estimated action values (Q-values).

        The action values are only available after the agent has been primed.

        Returns:
            An array of the action values of each arm.

        Raises:
            AgentUsageError: If the agent has not been primed.
        """
        if not self._primed:
            raise AgentUsageError("agent has no Q values before it is run")
        return self.strategy.Qs

    @property
    def Ns(self) -> NDArray[np.uint32]:
        """The number of times the agent has played each arm.

        The play counts are only available after the agent has been primed.

        Returns:
            An array of the play counts of each arm.

        Raises:
            AgentUsageError: If the agent has not been primed.
        """
        if not self._primed:
            raise AgentUsageError("agent has no Q values before it is run")
        return self.strategy.Ns
