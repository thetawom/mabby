from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.random import Generator

from mabby.simulation.bandit import Bandit


class Arm(ABC):
    @abstractmethod
    def __init__(self, **kwargs: float):
        pass

    @abstractmethod
    def play(self, rng: Generator) -> float:
        """Play arm and sample reward from distribution"""

    @property
    @abstractmethod
    def mean(self) -> float:
        """Compute mean of reward distribution"""

    @classmethod
    def bandit(cls, **kwargs: list[float]) -> Bandit:
        params_dicts = [dict(zip(kwargs, t)) for t in zip(*kwargs.values())]
        if len(params_dicts) == 0:
            raise ValueError("insufficient parameters to create an arm")
        return Bandit([cls(**params) for params in params_dicts])


class BernoulliArm(Arm):
    def __init__(self, p: float):
        self.p = p

    def play(self, rng: Generator) -> int:
        return rng.binomial(1, self.p)

    @property
    def mean(self) -> float:
        return self.p

    def __repr__(self) -> str:
        return f"Bernoulli(p={self.p})"


class GaussianArm(Arm):
    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale

    def play(self, rng: Generator) -> float:
        return rng.normal(self.loc, self.scale)

    @property
    def mean(self) -> float:
        return self.loc

    def __repr__(self) -> str:
        return f"Gaussian(loc={self.loc}, scale={self.scale})"
