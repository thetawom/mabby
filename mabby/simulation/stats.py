from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from mabby.exceptions import StatsUsageError

if TYPE_CHECKING:
    from mabby import Agent, Bandit, Simulation


@dataclass
class MetricMapping:
    base: Metric
    transform: Callable[[NDArray[np.float64]], NDArray[np.float64]]


class Metric(Enum):
    REGRET = "Regret"
    REWARDS = "Rewards"
    OPTIMALITY = "Optimality"
    CUM_REGRET = "Cumulative Regret", "REGRET", np.cumsum
    CUM_REWARDS = "Cumulative Rewards", "REWARDS", np.cumsum

    __MAPPING__: dict[str, Metric] = {}

    def __init__(
        self,
        label: str,
        base: str | None = None,
        transform: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    ):
        self.__class__.__MAPPING__[self._name_] = self
        self._label = label
        self._mapping: MetricMapping | None = (
            MetricMapping(base=self.__class__.__MAPPING__[base], transform=transform)
            if base and transform
            else None
        )

    def __repr__(self) -> str:
        return self._label

    def is_base(self) -> bool:
        return self._mapping is None

    @property
    def base(self) -> Metric:
        if self._mapping is not None:
            return self._mapping.base
        return self

    @classmethod
    def map_to_base(cls, metrics: Iterable[Metric]) -> Iterable[Metric]:
        return set(m.base for m in metrics)

    def transform(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._mapping is not None:
            return self._mapping.transform(values)
        return values


class SimulationStats:
    def __init__(self, simulation: Simulation):
        self._simulation: Simulation = simulation
        self._stats_dict: dict[Agent, AgentStats] = {}

    def add(self, agent_stats: AgentStats) -> None:
        self._stats_dict[agent_stats.agent] = agent_stats

    def __getitem__(self, agent: Agent) -> AgentStats:
        return self._stats_dict[agent]

    def __setitem__(self, agent: Agent, agent_stats: AgentStats) -> None:
        if agent != agent_stats.agent:
            raise StatsUsageError("agents specified in key and value don't match")
        self._stats_dict[agent] = agent_stats

    def __contains__(self, agent: Agent) -> bool:
        return agent in self._stats_dict

    def plot(self, metric: Metric) -> None:
        for agent, agent_stats in self._stats_dict.items():
            plt.plot(agent_stats[metric], label=str(agent))
        plt.legend()
        plt.show()

    def plot_regret(self, cumulative: bool = True) -> None:
        self.plot(metric=Metric.CUM_REGRET if cumulative else Metric.REGRET)

    def plot_optimality(self) -> None:
        self.plot(metric=Metric.OPTIMALITY)

    def plot_rewards(self, cumulative: bool = True) -> None:
        self.plot(metric=Metric.CUM_REWARDS if cumulative else Metric.REWARDS)


class AgentStats:
    def __init__(
        self,
        agent: Agent,
        bandit: Bandit,
        steps: int,
        metrics: Iterable[Metric] | None = None,
    ):
        self.agent = agent
        self._bandit = bandit
        self._steps = steps
        self._counts = np.zeros(steps)

        base_metrics = Metric.map_to_base(list(Metric) if metrics is None else metrics)
        self._stats = {stat: np.zeros(steps) for stat in base_metrics}

    def __len__(self) -> int:
        return self._steps

    def __getitem__(self, metric: Metric) -> NDArray[np.float64]:
        with np.errstate(divide="ignore", invalid="ignore"):
            values = self._stats[metric.base] / self._counts
        return metric.transform(values)

    def update(self, step: int, choice: int, reward: float) -> None:
        regret = self._bandit.regret(choice)
        if Metric.REGRET in self._stats:
            self._stats[Metric.REGRET][step] += regret
        if Metric.OPTIMALITY in self._stats:
            self._stats[Metric.OPTIMALITY][step] += int(self._bandit.is_opt(choice))
        if Metric.REWARDS in self._stats:
            self._stats[Metric.REWARDS][step] += reward
        self._counts[step] += 1
