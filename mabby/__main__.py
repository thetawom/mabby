import mabby as mb

if __name__ == "__main__":
    explore_bandit = mb.EpsilonGreedyBandit(eps=0.8)
    exploit_bandit = mb.EpsilonGreedyBandit(eps=0.1)

    sim = mb.Simulation(
        bandits=[explore_bandit, exploit_bandit],
        armset=mb.BernoulliArm.armset(p=[0.2, 0.6]),
    )
    sim.run(trials=10, rounds=2000)

    print("eps=0.8: ", explore_bandit.Qs)
    print("eps=0.1: ", exploit_bandit.Qs)
