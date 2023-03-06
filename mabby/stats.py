from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mabby import ArmSet, Bandit


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


class BanditStats:
    def __init__(
        self,
        bandit: Bandit,
        armset: ArmSet,
        steps: int,
        metrics: Collection[Metric] | None = None,
    ):
        self.bandit = bandit
        self._armset = armset
        self._steps = steps
        self._counts = np.zeros(steps)

        base_metrics = Metric.map_to_base(list(Metric) if metrics is None else metrics)
        self._stats = {stat: np.zeros(steps) for stat in base_metrics}

    def __len__(self) -> int:
        return self._steps

    def __getitem__(self, metric: Metric) -> NDArray[np.float64]:
        values = self._stats[metric.base] / self._counts
        return metric.transform(values)

    def update(self, step: int, choice: int, reward: float) -> None:
        opt_choice = self._armset.best_arm()
        regret = self._armset.regret(choice)
        if Metric.REGRET in self._stats:
            self._stats[Metric.REGRET][step] += regret
        if Metric.OPTIMALITY in self._stats:
            self._stats[Metric.OPTIMALITY][step] += int(opt_choice == choice)
        if Metric.REWARDS in self._stats:
            self._stats[Metric.REWARDS][step] += reward
        self._counts[step] += 1
