"""Provides metric tracking for multi-armed bandit simulations."""
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
    """Transformation from a base metric.

    See [`Metric`][mabby.stats.Metric] for examples of metric mappings.
    """

    #: The base metric to transform from
    base: Metric

    #: The transformation function
    transform: Callable[[NDArray[np.float64]], NDArray[np.float64]]


class Metric(Enum):
    """Enum for metrics that simulations can track."""

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
        """Initializes a metric.

        Metrics can be derived from other metrics through specifying a ``base`` metric
        and a ``transform`` function. This is useful for things like defining cumulative
        versions of an existing metric, where the transformed values can be computed
        "lazily" instead of being redundantly stored.

        Args:
            label: Verbose name of the metric (title case)
            base: Name of the base metric
            transform: Transformation function from the base metric
        """
        self.__class__.__MAPPING__[self._name_] = self
        self._label = label
        self._mapping: MetricMapping | None = (
            MetricMapping(base=self.__class__.__MAPPING__[base], transform=transform)
            if base and transform
            else None
        )

    def __repr__(self) -> str:
        """Returns the verbose name of the metric."""
        return self._label

    def is_base(self) -> bool:
        """Returns whether the metric is a base metric.

        Returns:
            ``True`` if the metric is a base metric, ``False`` otherwise.
        """
        return self._mapping is None

    @property
    def base(self) -> Metric:
        """The base metric that the metric is transformed from.

        If the metric is already a base metric, the metric itself is returned.
        """
        if self._mapping is not None:
            return self._mapping.base
        return self

    @classmethod
    def map_to_base(cls, metrics: Iterable[Metric]) -> Iterable[Metric]:
        """Traces all metrics back to their base metrics.

        Args:
            metrics: A collection of metrics.

        Returns:
            A set containing the base metrics of all the input metrics.
        """
        return set(m.base for m in metrics)

    def transform(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transforms values from the base metric.

        If the metric is already a base metric, the input values are returned.

        Args:
            values: An array of input values for the base metric.

        Returns:
            An array of transformed values for the metric.
        """
        if self._mapping is not None:
            return self._mapping.transform(values)
        return values


class SimulationStats:
    """Statistics for a multi-armed bandit simulation."""

    def __init__(self, simulation: Simulation):
        """Initializes simulation statistics.

        Args:
            simulation: The simulation to track.
        """
        self._simulation: Simulation = simulation
        self._stats_dict: dict[Agent, AgentStats] = {}

    def add(self, agent_stats: AgentStats) -> None:
        """Adds statistics for an agent.

        Args:
            agent_stats: The agent statistics to add.
        """
        self._stats_dict[agent_stats.agent] = agent_stats

    def __getitem__(self, agent: Agent) -> AgentStats:
        """Gets statistics for an agent.

        Args:
            agent: The agent to get the statistics of.

        Returns:
            The statistics of the agent.
        """
        return self._stats_dict[agent]

    def __setitem__(self, agent: Agent, agent_stats: AgentStats) -> None:
        """Sets the statistics for an agent.

        Args:
            agent: The agent to set the statistics of.
            agent_stats: The agent statistics to set.
        """
        if agent != agent_stats.agent:
            raise StatsUsageError("agents specified in key and value don't match")
        self._stats_dict[agent] = agent_stats

    def __contains__(self, agent: Agent) -> bool:
        """Returns if an agent's statistics are present.

        Returns:
            ``True`` if an agent's statistics are present, ``False`` otherwise.
        """
        return agent in self._stats_dict

    def plot(self, metric: Metric) -> None:
        """Generates a plot for a simulation metric.

        Args:
            metric: The metric to plot.
        """
        for agent, agent_stats in self._stats_dict.items():
            plt.plot(agent_stats[metric], label=str(agent))
        plt.legend()
        plt.show()

    def plot_regret(self, cumulative: bool = True) -> None:
        """Generates a plot for the regret or cumulative regret metrics.

        Args:
            cumulative: Whether to use the cumulative regret.
        """
        self.plot(metric=Metric.CUM_REGRET if cumulative else Metric.REGRET)

    def plot_optimality(self) -> None:
        """Generates a plot for the optimality metric."""
        self.plot(metric=Metric.OPTIMALITY)

    def plot_rewards(self, cumulative: bool = True) -> None:
        """Generates a plot for the rewards or cumulative rewards metrics.

        Args:
            cumulative: Whether to use the cumulative rewards.
        """
        self.plot(metric=Metric.CUM_REWARDS if cumulative else Metric.REWARDS)


class AgentStats:
    """Statistics for an agent in a multi-armed bandit simulation."""

    def __init__(
        self,
        agent: Agent,
        bandit: Bandit,
        steps: int,
        metrics: Iterable[Metric] | None = None,
    ):
        """Initializes agent statistics.

        All available metrics are tracked by default. Alternatively, a specific list can
        be specified through the ``metrics`` argument.

        Args:
            agent: The agent that statistics are tracked for
            bandit: The bandit of the simulation being run
            steps: The number of steps per trial in the simulation
            metrics: A collection of metrics to track.
        """
        self.agent: Agent = agent  #: The agent that statistics are tracked for
        self._bandit = bandit
        self._steps = steps
        self._counts = np.zeros(steps)

        base_metrics = Metric.map_to_base(list(Metric) if metrics is None else metrics)
        self._stats = {stat: np.zeros(steps) for stat in base_metrics}

    def __len__(self) -> int:
        """Returns the number of steps each trial is tracked for."""
        return self._steps

    def __getitem__(self, metric: Metric) -> NDArray[np.float64]:
        """Gets values for a metric.

        If the metric is not a base metric, the values are automatically transformed.

        Args:
            metric: The metric to get the values for.

        Returns:
            An array of values for the metric.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            values = self._stats[metric.base] / self._counts
        return metric.transform(values)

    def update(self, step: int, choice: int, reward: float) -> None:
        """Updates metric values for the latest simulation step.

        Args:
            step: The number of the step.
            choice: The choice made by the agent.
            reward: The reward observed by the agent.
        """
        regret = self._bandit.regret(choice)
        if Metric.REGRET in self._stats:
            self._stats[Metric.REGRET][step] += regret
        if Metric.OPTIMALITY in self._stats:
            self._stats[Metric.OPTIMALITY][step] += int(self._bandit.is_opt(choice))
        if Metric.REWARDS in self._stats:
            self._stats[Metric.REWARDS][step] += reward
        self._counts[step] += 1
