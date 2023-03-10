from mabby import BetaTSStrategy, EpsilonGreedyStrategy, Simulation, UCB1Strategy
from mabby.core import BernoulliArm

eps_greedy_strategy = EpsilonGreedyStrategy(eps=0.2)
ucb1_strategy = UCB1Strategy(alpha=0.5)
beta_ts_strategy = BetaTSStrategy(general=True)
sim = Simulation(
    strategies=[eps_greedy_strategy, ucb1_strategy, beta_ts_strategy],
    bandit=BernoulliArm.bandit(p=[0.5, 0.6, 0.7, 0.8]),
)
stats = sim.run(trials=100, steps=300)

stats.plot_regret(cumulative=True)
stats.plot_optimality()
stats.plot_rewards(cumulative=True)
