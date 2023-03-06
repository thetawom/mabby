from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from mabby.bandits import Bandit
from mabby.stats import BanditStats, Metric, SimulationStats

if TYPE_CHECKING:
    from mabby.arms import ArmSet


class Simulation:
    def __init__(
        self, bandits: Iterable[Bandit], armset: ArmSet, seed: int | None = None
    ):
        self.bandits = bandits
        if len(armset) == 0:
            raise ValueError("ArmSet cannot be empty")
        self.armset = armset
        self._rng = np.random.default_rng(seed)

    def run(
        self, trials: int, steps: int, metrics: Iterable[Metric] | None = None
    ) -> SimulationStats:
        sim_stats = SimulationStats(simulation=self)
        for bandit in self.bandits:
            bandit_stats = self._run_trials_for_bandit(bandit, trials, steps, metrics)
            sim_stats.add(bandit_stats)
        return sim_stats

    def _run_trials_for_bandit(
        self,
        bandit: Bandit,
        trials: int,
        steps: int,
        metrics: Iterable[Metric] | None = None,
    ) -> BanditStats:
        bandit_stats = BanditStats(bandit, self.armset, steps, metrics)
        for _ in range(trials):
            bandit.prime(len(self.armset), steps)
            for step in range(steps):
                choice = bandit.choose(self._rng)
                reward = self.armset.play(choice, self._rng)
                bandit.update(reward)
                bandit_stats.update(step, choice, reward)
        return bandit_stats
