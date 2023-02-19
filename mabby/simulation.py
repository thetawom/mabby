class Simulation:
    def __init__(self, bandits, armset):
        self.bandits = bandits
        self.armset = armset

    def run(self, trials, rounds):
        for bandit in self.bandits:
            for _ in range(trials):
                self._run_trial(bandit, rounds)

    def _run_trial(self, bandit, rounds):
        bandit.prime(len(self.armset), rounds)
        for _ in range(rounds):
            choice = bandit.choose()
            reward = self.armset.play(choice)
            bandit.update(reward)
