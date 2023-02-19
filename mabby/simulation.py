from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


class Simulation:
    def __init__(self, bandits, armset):
        self.bandits = bandits
        self.armset = armset

    def run(self, trials, steps):
        stats = SimStats(self, trials, steps)
        for bandit in self.bandits:
            for _ in range(trials):
                self._run_trial(stats, bandit, steps)
        stats.compile_all()
        return stats

    def _run_trial(self, stats, bandit, steps):
        bandit.prime(len(self.armset), steps)
        for i in range(steps):
            choice = bandit.choose()
            reward = self.armset.play(choice)
            bandit.update(reward)
            stats.update(bandit, i, choice, reward)


class SimStats:
    def __init__(self, simulation, trials, steps):
        self._sim = simulation
        self._trials = trials

        self.rewards = defaultdict(lambda: None)
        self.optimality = defaultdict(lambda: None)
        self.regret = defaultdict(lambda: None)

        self._rewards = defaultdict(lambda: np.zeros(steps))
        self._optimality = defaultdict(lambda: np.zeros(steps))
        self._regret = defaultdict(lambda: np.zeros(steps))

    def update(self, bandit, i, choice, reward):
        best_arm = self._sim.armset.best_arm()
        self._rewards[bandit][i] += reward
        self._optimality[bandit][i] += int(best_arm == choice)
        self._regret[bandit][i] += (
            self._sim.armset[best_arm].mean - self._sim.armset[choice].mean
        )

    def compile_all(self):
        for bandit in self._sim.bandits:
            self.compile_for_bandit(bandit)

    def compile_for_bandit(self, bandit):
        self.rewards[bandit] = self._rewards[bandit] / self._trials
        self.optimality[bandit] = self._optimality[bandit] / self._trials
        self.regret[bandit] = np.cumsum(self._regret[bandit] / self._trials)

    def plot_rewards(self):
        for i, (bandit, stats) in enumerate(self.rewards.items()):
            plt.plot(stats, label=bandit.name)
        plt.legend()
        plt.show()

    def plot_optimality(self):
        for i, (bandit, stats) in enumerate(self.optimality.items()):
            plt.plot(stats, label=bandit.name)
        plt.legend()
        plt.show()

    def plot_regret(self):
        for i, (bandit, stats) in enumerate(self.regret.items()):
            plt.plot(stats, label=bandit.name)
        plt.legend()
        plt.show()
