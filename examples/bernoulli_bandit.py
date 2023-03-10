from mabby import BernoulliArm, Simulation
from mabby.strategies import BetaTSStrategy, EpsilonGreedyStrategy, UCB1Strategy

simulation = Simulation(
    bandit=BernoulliArm.bandit(p=[0.5, 0.6, 0.7, 0.8]),
    strategies=[
        EpsilonGreedyStrategy(eps=0.2),
        UCB1Strategy(alpha=0.5),
        BetaTSStrategy(general=True),
    ],
)
stats = simulation.run(trials=100, steps=300)

stats.plot_regret(cumulative=True)
stats.plot_optimality()
stats.plot_rewards(cumulative=True)
