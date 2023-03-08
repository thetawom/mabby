from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from mabby.agents import Agent
from mabby.stats import AgentStats, Metric, SimulationStats

if TYPE_CHECKING:
    from mabby.bandit import Bandit


class Simulation:
    def __init__(
        self, agents: Iterable[Agent], bandit: Bandit, seed: int | None = None
    ):
        self.agents = agents
        if len(bandit) == 0:
            raise ValueError("Bandit cannot be empty")
        self.bandit = bandit
        self._rng = np.random.default_rng(seed)

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
