from collections import defaultdict

import numpy as np


class Simulation:
    def __init__(self, bandits, armset):
        self.bandits = bandits
        self.armset = armset

    def run(self, trials, rounds):
        stats = SimStats(self, trials, rounds)
        for bandit in self.bandits:
            for _ in range(trials):
                self._run_trial(stats, bandit, rounds)
        stats.compile_all()
        return stats

    def _run_trial(self, stats, bandit, rounds):
        bandit.prime(len(self.armset), rounds)
        for i in range(rounds):
            choice = bandit.choose()
            reward = self.armset.play(choice)
            bandit.update(reward)
            stats.update(bandit, i, choice, reward)


class SimStats:
    def __init__(self, simulation, trials, rounds):
        self._sim = simulation
        self._trials = trials

        self.rewards = defaultdict(lambda: None)
        self.optimality = defaultdict(lambda: None)
        self.regret = defaultdict(lambda: None)

        self._rewards = defaultdict(lambda: np.zeros(rounds))
        self._optimality = defaultdict(lambda: np.zeros(rounds))
        self._regret = defaultdict(lambda: np.zeros(rounds))

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
