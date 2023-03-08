import mabby as mb

eps_greedy_agent = mb.EpsilonGreedyAgent(eps=0.2)
ucb_agent = mb.UCB1Agent(alpha=0.5)
ts_agent = mb.BetaTSAgent(general=True)
sim = mb.Simulation(
    agents=[eps_greedy_agent, ucb_agent, ts_agent],
    bandit=mb.BernoulliArm.bandit(p=[0.5, 0.6, 0.7, 0.8]),
)
stats = sim.run(trials=100, steps=300)

stats.plot_regret(cumulative=True)
stats.plot_optimality()
stats.plot_rewards(cumulative=True)
