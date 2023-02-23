from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from mabby import ArmSet, Bandit


class Simulation:
    def __init__(
        self, bandits: List[Bandit], armset: ArmSet, seed: Optional[int] = None
    ):
        self.bandits = bandits
        self.armset = armset
        self._rng = np.random.default_rng(seed)

    def run(self, trials: int, steps: int) -> SimStats:
        stats = SimStats(self, trials, steps)
        for bandit in self.bandits:
            for _ in range(trials):
                self._run_trial(stats, bandit, steps)
        stats.compile_all()
        return stats

    def _run_trial(self, stats: SimStats, bandit: Bandit, steps: int) -> None:
        bandit.prime(len(self.armset), steps)
        for i in range(steps):
            choice = bandit.choose(self._rng)
            reward = self.armset.play(choice)
            bandit.update(reward)
            stats.update(bandit, i, choice, reward)


SimStatOpt = DefaultDict[Bandit, Optional[NDArray[np.float64]]]
SimStat = DefaultDict[Bandit, NDArray[np.float64]]


class SimStats:
    def __init__(self, simulation: Simulation, trials: int, steps: int):
        self._sim = simulation
        self._trials = trials

        self.rewards: SimStatOpt = defaultdict(lambda: None)
        self.optimality: SimStatOpt = defaultdict(lambda: None)
        self.regret: SimStatOpt = defaultdict(lambda: None)

        self._rewards: SimStat = defaultdict(lambda: np.zeros(steps))
        self._optimality: SimStat = defaultdict(lambda: np.zeros(steps))
        self._regret: SimStat = defaultdict(lambda: np.zeros(steps))

    def update(self, bandit: Bandit, i: int, choice: int, reward: float) -> None:
        best_arm = self._sim.armset.best_arm()
        self._rewards[bandit][i] += reward
        self._optimality[bandit][i] += int(best_arm == choice)
        self._regret[bandit][i] += (
            self._sim.armset[best_arm].mean - self._sim.armset[choice].mean
        )

    def compile_all(self) -> None:
        for bandit in self._sim.bandits:
            self.compile_for_bandit(bandit)

    def compile_for_bandit(self, bandit: Bandit) -> None:
        self.rewards[bandit] = self._rewards[bandit] / self._trials
        self.optimality[bandit] = self._optimality[bandit] / self._trials
        self.regret[bandit] = np.cumsum(self._regret[bandit] / self._trials)

    def plot_rewards(self) -> None:
        for i, (bandit, stats) in enumerate(self.rewards.items()):
            plt.plot(stats, label=bandit.name)
        plt.legend()
        plt.show()

    def plot_optimality(self) -> None:
        for i, (bandit, stats) in enumerate(self.optimality.items()):
            plt.plot(stats, label=bandit.name)
        plt.legend()
        plt.show()

    def plot_regret(self) -> None:
        for i, (bandit, stats) in enumerate(self.regret.items()):
            plt.plot(stats, label=bandit.name)
        plt.legend()
        plt.show()
