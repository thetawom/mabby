from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Callable, NamedTuple

import numpy as np
from numpy.typing import NDArray


class MetricTransform(NamedTuple):
    base: Metric
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]]


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
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    ):
        self.__class__.__MAPPING__[self._name_] = self
        self._label = label
        self._transform: MetricTransform | None = (
            MetricTransform(base=self.__class__.__MAPPING__[base], func=func)
            if base and func
            else None
        )

    def __repr__(self) -> str:
        return self._label

    def is_base(self) -> bool:
        return self._transform is None

    @property
    def base(self) -> Metric:
        if self._transform is not None:
            return self._transform.base
        return self

    @classmethod
    def filter_base(cls, metrics: Iterable[Metric]) -> Iterable[Metric]:
        return filter(lambda m: m.is_base(), metrics)

    def transform(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._transform is not None:
            return self._transform.func(values)
        return values
