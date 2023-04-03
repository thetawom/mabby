"""Provides [`Simulation`][mabby.simulation.Simulation] class for bandit simulations."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import zip_longest
from typing import TYPE_CHECKING

import numpy as np
from numpy.random import Generator

from mabby.agent import Agent
from mabby.exceptions import SimulationUsageError
from mabby.stats import AgentStats, Metric, SimulationStats

if TYPE_CHECKING:
    from mabby.bandit import Bandit
    from mabby.strategies import Strategy


class Simulation:
    """Simulation of a multi-armed bandit problem.

    A simulation consists of multiple trials of one or more bandit strategies run on a
    configured multi-armed bandit.
    """

    def __init__(
        self,
        bandit: Bandit,
        agents: Iterable[Agent] | None = None,
        strategies: Iterable[Strategy] | None = None,
        names: Iterable[str] | None = None,
        rng: Generator | None = None,
        seed: int | None = None,
    ):
        """Initializes a simulation.

        One of ``agents`` or ``strategies`` must be supplied. If ``agents`` is supplied,
        ``strategies`` and ``names`` are ignored. Otherwise, an ``agent`` is created for
        each ``strategy`` and given a name from ``names`` if available.

        Args:
            bandit: A configured multi-armed bandit to simulate on.
            agents: A list of agents to simulate.
            strategies: A list of strategies to simulate.
            names: A list of names for agents.
            rng: A random number generator.
            seed: A seed for random number generation if ``rng`` is not provided.

        Raises:
            SimulationUsageError: If neither ``agents`` nor ``strategies`` are supplied.
        """
        self.agents = self._create_agents(agents, strategies, names)
        if len(list(self.agents)) == 0:
            raise ValueError("no strategies or agents were supplied")
        self.bandit = bandit
        if len(self.bandit) == 0:
            raise ValueError("bandit cannot be empty")
        self._rng = rng if rng else np.random.default_rng(seed)

    @staticmethod
    def _create_agents(
        agents: Iterable[Agent] | None = None,
        strategies: Iterable[Strategy] | None = None,
        names: Iterable[str] | None = None,
    ) -> Iterable[Agent]:
        if agents is not None:
            return agents
        if strategies is not None and names is not None:
            return [
                strategy.agent(name=name)
                for strategy, name in zip_longest(strategies, names)
                if strategy
            ]
        if strategies is not None:
            return [strategy.agent() for strategy in strategies]
        raise SimulationUsageError("one of agents or strategies must be supplied")

    def run(
        self, trials: int, steps: int, metrics: Iterable[Metric] | None = None
    ) -> SimulationStats:
        """Runs a simulation.

        In a simulation run, each agent or strategy is run for the specified number of
        trials, and each trial is run for the given number of steps.

        If ``metrics`` is not specified, all available metrics are tracked by default.

        Args:
            trials: The number of trials in the simulation.
            steps: The number of steps in a trial.
            metrics: A list of metrics to collect.

        Returns:
            A ``SimulationStats`` object with the results of the simulation.
        """
        sim_stats = SimulationStats(simulation=self)
        for agent in self.agents:
            agent_stats = self._run_trials_for_agent(agent, trials, steps, metrics)
            sim_stats.add(agent_stats)
        return sim_stats

    def _run_trials_for_agent(
        self,
        agent: Agent,
        trials: int,
        steps: int,
        metrics: Iterable[Metric] | None = None,
    ) -> AgentStats:
        agent_stats = AgentStats(agent, self.bandit, steps, metrics)
        for _ in range(trials):
            agent.prime(len(self.bandit), steps, self._rng)
            for step in range(steps):
                choice = agent.choose()
                reward = self.bandit.play(choice)
                agent.update(reward)
                agent_stats.update(step, choice, reward)
        return agent_stats
