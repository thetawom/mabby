"""Provides [`Bandit`][mabby.bandit.Bandit] class for bandit simulations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from numpy.random import Generator

if TYPE_CHECKING:
    from mabby.arms import Arm
from mabby.utils import random_argmax


class Bandit:
    """Multi-armed bandit with one or more arms.

    This class wraps around a list of arms, each of which has a reward distribution. It
    provides an interface for interacting with the arms, such as playing a specific arm,
    querying for the optimal arm, and computing regret from a given choice.
    """

    def __init__(
        self, arms: list[Arm], rng: Generator | None = None, seed: int | None = None
    ):
        """Initializes a bandit with a given set of arms.

        Args:
            arms: A list of arms for the bandit.
            rng: A random number generator.
            seed: A seed for random number generation if ``rng`` is not provided.
        """
        self._arms = arms
        self._rng = rng if rng else np.random.default_rng(seed)

    def __len__(self) -> int:
        """Returns the number of arms."""
        return len(self._arms)

    def __repr__(self) -> str:
        """Returns a string representation of the bandit."""
        return repr(self._arms)

    def __getitem__(self, i: int) -> Arm:
        """Returns an arm by index.

        Args:
            i: The index of the arm to get.

        Returns:
            The arm at the given index.
        """
        return self._arms[i]

    def __iter__(self) -> Iterable[Arm]:
        """Returns an iterator over the bandit's arms."""
        return iter(self._arms)

    def play(self, i: int) -> float:
        """Plays an arm by index.

        Args:
            i: The index of the arm to play.

        Returns:
            The reward from playing the arm.
        """
        return self[i].play(self._rng)

    @property
    def means(self) -> list[float]:
        """The means of the arms.

        Returns:
            An array of the means of each arm.
        """
        return [arm.mean for arm in self._arms]

    def best_arm(self) -> int:
        """Returns the index of the optimal arm.

        The optimal arm is the arm with the greatest expected reward. If there are
        multiple arms with equal expected rewards, a random one is chosen.

        Returns:
            The index of the optimal arm.
        """
        return random_argmax(self.means, rng=self._rng)

    def is_opt(self, choice: int) -> bool:
        """Returns the optimality of a given choice.

        Args:
            choice: The index of the chosen arm.

        Returns:
            ``True`` if the arm has the greatest expected reward, ``False`` otherwise.
        """
        return np.max(self.means) == self._arms[choice].mean

    def regret(self, choice: int) -> float:
        """Returns the regret from a given choice.

        The regret is computed as the difference between the expected reward from the
        optimal arm and the expected reward from the chosen arm.

        Args:
            choice: The index of the chosen arm.

        Returns:
            The computed regret value.
        """
        return np.max(self.means) - self._arms[choice].mean
