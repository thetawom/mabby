from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator
from numpy.typing import NDArray

from mabby import Bandit
from mabby.arms import Arm
from mabby.strategies import Strategy


class GenericStrategy(Strategy):
    k: int

    def prime(self, k: int, steps: int) -> None:
        self.k = k

    def choose(self, rng: Generator | None = None) -> int:
        return 0

    def update(self, choice: int, reward: float) -> None:
        pass

    @property
    def Qs(self) -> NDArray[np.float64]:
        return np.zeros(self.k)

    @property
    def Ns(self) -> NDArray[np.int32]:
        return np.zeros(self.k)


class GenericArm(Arm):
    def __init__(self, mean: float = 0):
        self._mean = mean

    def play(self, rng: Generator) -> float:
        return 1

    @property
    def mean(self) -> float:
        return self._mean


@pytest.fixture
def strategy_factory():
    class GenericStrategyFactory:
        @staticmethod
        def generic():
            return GenericStrategy()

    return GenericStrategyFactory


@pytest.fixture
def bandit_factory(strategy_factory):
    class GenericBanditFactory:
        @staticmethod
        def generic():
            strategy = strategy_factory.generic()
            return Bandit(strategy=strategy, name="Generic Bandit")

    return GenericBanditFactory


@pytest.fixture
def arm_factory():
    class GenericArmFactory:
        @staticmethod
        def generic(mean: float = 0):
            return GenericArm(mean=mean)

    return GenericArmFactory
