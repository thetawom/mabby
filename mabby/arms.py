"""Provides [`Arm`][mabby.arms.Arm] base class with some common reward distributions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.random import Generator
from overrides import EnforceOverrides, override

from mabby.bandit import Bandit


class Arm(ABC, EnforceOverrides):
    """Base class for a bandit arm implementing a reward distribution.

    An arm represents one of the decision choices available to the agent in a bandit
    problem. It has a hidden reward distribution and can be played by the agent to
    generate observable rewards.
    """

    @abstractmethod
    def __init__(self, **kwargs: float):
        """Initializes an arm."""

    @abstractmethod
    def play(self, rng: Generator) -> float:
        """Plays the arm and samples a reward.

        Args:
            rng: A random number generator.

        Returns:
            The sampled reward from the arm's reward distribution.
        """

    @property
    @abstractmethod
    def mean(self) -> float:
        """The mean reward of the arm.

        Returns:
            The computed mean of the arm's reward distribution.
        """

    @abstractmethod
    def __repr__(self) -> str:
        """Returns the string representation of the arm."""

    @classmethod
    def bandit(
        cls,
        rng: Generator | None = None,
        seed: int | None = None,
        **kwargs: list[float],
    ) -> Bandit:
        """Creates a bandit with arms of the same reward distribution type.

        Args:
            rng: A random number generator.
            seed: A seed for random number generation if ``rng`` is not provided.
            **kwargs: A dictionary where keys are arm parameter names and values are
                lists of parameter values for each arm.

        Returns:
            A bandit with the specified arms.
        """
        params_dicts = [dict(zip(kwargs, t)) for t in zip(*kwargs.values())]
        if len(params_dicts) == 0:
            raise ValueError("insufficient parameters to create an arm")
        return Bandit([cls(**params) for params in params_dicts], rng, seed)


class BernoulliArm(Arm):
    """Bandit arm with a Bernoulli reward distribution."""

    def __init__(self, p: float):
        """Initializes a Bernoulli arm.

        Args:
            p: Parameter of the Bernoulli distribution.
        """
        if p < 0 or p > 1:
            raise ValueError(
                f"float {str(p)} is not a valid probability for Bernoulli distribution"
            )

        self.p: float = p  #: Parameter of the Bernoulli distribution

    @override
    def play(self, rng: Generator) -> float:
        return rng.binomial(1, self.p)

    @property
    @override
    def mean(self) -> float:
        return self.p

    @override
    def __repr__(self) -> str:
        return f"Bernoulli(p={self.p})"


class GaussianArm(Arm):
    """Bandit arm with a Gaussian reward distribution."""

    def __init__(self, loc: float, scale: float):
        """Initializes a Gaussian arm.

        Args:
            loc: Mean ("center") of the Gaussian distribution.
            scale: Standard deviation of the Gaussian distribution.
        """
        if scale < 0:
            raise ValueError(
                f"float {str(scale)} is not a valid scale for Gaussian distribution"
            )

        self.loc: float = loc  #: Mean ("center") of the Gaussian distribution
        self.scale: float = scale  #: Standard deviation of the Gaussian distribution

    @override
    def play(self, rng: Generator) -> float:
        return rng.normal(self.loc, self.scale)

    @property
    @override
    def mean(self) -> float:
        return self.loc

    @override
    def __repr__(self) -> str:
        return f"Gaussian(loc={self.loc}, scale={self.scale})"
