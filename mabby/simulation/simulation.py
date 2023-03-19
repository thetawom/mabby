from __future__ import annotations

from collections.abc import Iterable
from itertools import zip_longest
from typing import TYPE_CHECKING

import numpy as np

from mabby.simulation.agent import Agent
from mabby.simulation.exceptions import SimulationUsageError
from mabby.simulation.stats import AgentStats, Metric, SimulationStats

if TYPE_CHECKING:
    from mabby.simulation.bandit import Bandit
    from mabby.strategies import Strategy


class Simulation:
    def __init__(
        self,
        bandit: Bandit,
        agents: Iterable[Agent] | None = None,
        strategies: Iterable[Strategy] | None = None,
        names: Iterable[str] | None = None,
        seed: int | None = None,
    ):
        self.agents = self._create_agents(agents, strategies, names)
        if len(list(self.agents)) == 0:
            raise ValueError("no strategies or agents were supplied")
        self.bandit = bandit
        if len(self.bandit) == 0:
            raise ValueError("bandit cannot be empty")
        self._rng = np.random.default_rng(seed)

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
                reward = self.bandit.play(choice, self._rng)
                agent.update(reward)
                agent_stats.update(step, choice, reward)
        return agent_stats
