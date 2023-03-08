import mabby as mb

eps_greedy_bandit = mb.EpsilonGreedyBandit(eps=0.2)
ucb_bandit = mb.UCB1Bandit(alpha=0.5)
ts_bandit = mb.BetaTSBandit(general=True)
sim = mb.Simulation(
    bandits=[eps_greedy_bandit, ucb_bandit, ts_bandit],
    armset=mb.BernoulliArm.armset(p=[0.5, 0.6, 0.7, 0.8]),
)
stats = sim.run(trials=100, steps=300)

stats.plot_regret(cumulative=True)
stats.plot_optimality()
stats.plot_rewards(cumulative=True)
