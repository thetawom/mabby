import mabby as mb

eps_greedy_strategy = mb.EpsilonGreedyStrategy(eps=0.2)
ucb1_strategy = mb.UCB1Strategy(alpha=0.5)
beta_ts_strategy = mb.BetaTSStrategy(general=True)
sim = mb.Simulation(
    strategies=[eps_greedy_strategy, ucb1_strategy, beta_ts_strategy],
    bandit=mb.BernoulliArm.bandit(p=[0.5, 0.6, 0.7, 0.8]),
)
stats = sim.run(trials=100, steps=300)

stats.plot_regret(cumulative=True)
stats.plot_optimality()
stats.plot_rewards(cumulative=True)
