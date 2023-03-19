from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from numpy.random import Generator

if TYPE_CHECKING:
    from mabby.simulation.arms import Arm
from mabby.utils import random_argmax


class Bandit:
    def __init__(
        self, arms: list[Arm], rng: Generator | None = None, seed: int | None = None
    ):
        self._arms = arms
        self._rng = rng if rng else np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._arms)

    def __repr__(self) -> str:
        return repr(self._arms)

    def __getitem__(self, i: int) -> Arm:
        return self._arms[i]

    def __iter__(self) -> Iterable[Arm]:
        return iter(self._arms)

    def play(self, i: int, rng: Generator) -> float:
        return self[i].play(rng)

    @property
    def means(self) -> list[float]:
        return [arm.mean for arm in self._arms]

    def best_arm(self) -> int:
        return random_argmax(self.means, rng=self._rng)

    def is_opt(self, choice: int) -> bool:
        return np.max(self.means) == self._arms[choice].mean

    def regret(self, choice: int) -> float:
        return np.max(self.means) - self._arms[choice].mean
