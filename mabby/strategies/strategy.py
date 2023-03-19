from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from mabby.simulation.agent import Agent


class Strategy(ABC):
    def __init__(self, **kwargs: float) -> None:
        super().__init__()

    @abstractmethod
    def prime(self, k: int, steps: int) -> None:
        """Set up agent before a trial run"""

    @abstractmethod
    def choose(self, rng: Generator) -> int:
        """Choose an arm to play"""

    @abstractmethod
    def update(self, choice: int, reward: float, rng: Generator | None = None) -> None:
        """Update estimates based on reward observation"""

    @property
    @abstractmethod
    def Qs(self) -> NDArray[np.float64]:
        """Compute action value estimates for each arm"""

    @property
    @abstractmethod
    def Ns(self) -> NDArray[np.uint32]:
        """Count number of plays for each arm"""

    def agent(self, **kwargs: str) -> Agent:
        return Agent(strategy=self, **kwargs)
