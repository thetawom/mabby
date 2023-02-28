import numpy as np
import pytest
from numpy.random import Generator
from numpy.typing import NDArray

from mabby.arms import Arm
from mabby.bandits import Bandit


class GenericBandit(Bandit):
    def default_name(self) -> str:
        return ""

    def _prime(self, k: int, steps: int) -> None:
        self.k = k

    def _choose(self, rng: Generator) -> int:
        return 0

    def _update(self, choice: int, reward: float) -> None:
        pass

    def _compute_Qs(self) -> NDArray[np.float64]:
        return np.zeros(self.k)


class GenericArm(Arm):
    def __init__(self, **kwargs: float):
        pass

    def play(self, rng: Generator) -> float:
        return 1

    @property
    def mean(self) -> float:
        return 0


@pytest.fixture
def bandit_factory():
    class GenericBanditFactory:
        @staticmethod
        def generic():
            return GenericBandit()

    return GenericBanditFactory


@pytest.fixture
def arm_factory():
    class GenericArmFactory:
        @staticmethod
        def generic():
            return GenericArm()

    return GenericArmFactory
