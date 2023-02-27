from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, DefaultDict

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from mabby.bandits import Bandit
from mabby.exceptions import SimulationUsageError

if TYPE_CHECKING:
    from mabby.arms import ArmSet


class Simulation:
    def __init__(self, bandits: list[Bandit], armset: ArmSet, seed: int | None = None):
        self.bandits = bandits
        self.armset = armset
        self._rng = np.random.default_rng(seed)

    def run(self, trials: int, steps: int) -> SimStats:
        stats = SimStats(self, trials, steps)
        for bandit in self.bandits:
            for _ in range(trials):
                self._run_trial(stats, bandit, steps)
        return stats.compile()

    def _run_trial(self, stats: SimStats, bandit: Bandit, steps: int) -> None:
        bandit.prime(len(self.armset), steps)
        for i in range(steps):
            choice = bandit.choose(self._rng)
            reward = self.armset.play(choice, self._rng)
            bandit.update(reward)
            stats.update(bandit, i, choice, reward)


SimStat = DefaultDict[Bandit, NDArray[np.float64]]


class SimStats:
    def __init__(self, simulation: Simulation, trials: int, steps: int):
        self._sim = simulation
        self._trials = trials

        self._rewards: SimStat = defaultdict(lambda: np.zeros(steps))
        self._optimality: SimStat = defaultdict(lambda: np.zeros(steps))
        self._regret: SimStat = defaultdict(lambda: np.zeros(steps))

        self._compiled = False

    @property
    def rewards(self) -> SimStat:
        if not self._compiled:
            raise SimulationUsageError("stats have not been compiled")
        return self._rewards

    @property
    def optimality(self) -> SimStat:
        if not self._compiled:
            raise SimulationUsageError("stats have not been compiled")
        return self._optimality

    @property
    def regret(self) -> SimStat:
        if not self._compiled:
            raise SimulationUsageError("stats have not been compiled")
        return self._regret

    def update(self, bandit: Bandit, i: int, choice: int, reward: float) -> None:
        opt_arm = self._sim.armset.best_arm()
        self._rewards[bandit][i] += reward
        self._optimality[bandit][i] += int(opt_arm == choice)
        self._regret[bandit][i] += (
            self._sim.armset[opt_arm].mean - self._sim.armset[choice].mean
        )
        self._compiled = False

    def compile(self) -> SimStats:
        if not self._compiled:
            for bandit in self._sim.bandits:
                self._rewards[bandit] /= self._trials
                self._optimality[bandit] /= self._trials
                self._regret[bandit] = np.cumsum(self._regret[bandit] / self._trials)
            self._compiled = True
        return self

    def plot_rewards(self) -> None:
        for _, (bandit, stats) in enumerate(self.rewards.items()):
            plt.plot(stats, label=bandit.name)
        plt.legend()
        plt.show()

    def plot_optimality(self) -> None:
        for _, (bandit, stats) in enumerate(self.optimality.items()):
            plt.plot(stats, label=bandit.name)
        plt.legend()
        plt.show()

    def plot_regret(self) -> None:
        for _, (bandit, stats) in enumerate(self.regret.items()):
            plt.plot(stats, label=bandit.name)
        plt.legend()
        plt.show()
