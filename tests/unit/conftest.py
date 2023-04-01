from __future__ import annotations

import random

import numpy as np
import pytest
from numpy.random import Generator
from numpy.typing import NDArray
from overrides import override

from mabby import Agent, Arm
from mabby.strategies import Strategy


class GenericStrategy(Strategy):
    k: int

    def __init__(self) -> None:
        pass

    @override
    def __repr__(self) -> str:
        return "generic-strategy"

    @override
    def prime(self, k: int, steps: int) -> None:
        self.k = k

    @override
    def choose(self, rng: Generator) -> int:
        return 0

    @override
    def update(self, choice: int, reward: float, rng: Generator | None = None) -> None:
        pass

    @property
    @override
    def Qs(self) -> NDArray[np.float64]:
        return np.zeros(self.k)

    @property
    @override
    def Ns(self) -> NDArray[np.int32]:
        return np.zeros(self.k)


class GenericArm(Arm):
    def __init__(self, mean: float = 0):
        self._mean = mean

    @override
    def play(self, rng: Generator) -> float:
        return 1

    @property
    @override
    def mean(self) -> float:
        return self._mean

    @override
    def __repr__(self) -> str:
        return "generic-arm"


@pytest.fixture
def strategy_factory():
    class GenericStrategyFactory:
        @staticmethod
        def generic():
            return GenericStrategy()

    return GenericStrategyFactory


@pytest.fixture
def agent_factory(strategy_factory):
    class GenericAgentFactory:
        @staticmethod
        def generic():
            strategy = strategy_factory.generic()
            return Agent(strategy=strategy, name="Generic Agent")

    return GenericAgentFactory


@pytest.fixture
def arm_factory():
    class GenericArmFactory:
        @staticmethod
        def generic(mean: float | None = None):
            if mean is None:
                mean = random.random()
            return GenericArm(mean=mean)

    return GenericArmFactory
