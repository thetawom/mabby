import numpy as np
import pytest

from mabby import BernoulliArm, GaussianArm, Simulation
from mabby.strategies import EpsilonGreedyStrategy, RandomStrategy


@pytest.mark.parametrize("p", [[0.7, 0.9]])
@pytest.mark.parametrize("num_agents", [2])
def test_random_bernoulli_bandits_simulation(p, num_agents):
    rng = np.random.default_rng(seed=15)
    bandit = BernoulliArm.bandit(p=p, rng=rng)
    agents = [RandomStrategy().agent() for _ in range(num_agents)]
    sim = Simulation(agents=agents, bandit=bandit, rng=rng)
    sim.run(trials=1, steps=100000)
    for agent in agents:
        assert np.allclose(agent.Qs, p, rtol=0.1)
        assert np.isclose(agent.Ns, np.mean(agent.Ns), rtol=0.01).all()


@pytest.mark.parametrize("loc,scale", [([2, -2], [0.5, 0.5])])
@pytest.mark.parametrize("num_agents", [2])
def test_random_gaussian_bandits_simulation(loc, scale, num_agents):
    rng = np.random.default_rng(seed=36)
    bandit = GaussianArm.bandit(loc=loc, scale=scale, rng=rng)
    agents = [RandomStrategy().agent() for _ in range(num_agents)]
    sim = Simulation(agents=agents, bandit=bandit, rng=rng)
    sim.run(trials=1, steps=100000)
    for agent in agents:
        assert np.allclose(agent.Qs, loc, rtol=0.1)
        assert np.isclose(agent.Ns, np.mean(agent.Ns), rtol=0.01).all()


@pytest.mark.parametrize("p", [[0.2, 0.6]])
@pytest.mark.parametrize("eps", [[0.1, 0.8]])
def test_epsilon_greedy_bernoulli_bandits_simulation(p, eps):
    rng = np.random.default_rng(seed=203)
    bandit = BernoulliArm.bandit(p=p, rng=rng)
    agents = [EpsilonGreedyStrategy(eps=e).agent() for e in eps]
    sim = Simulation(agents=agents, bandit=bandit, rng=rng)
    sim.run(trials=1, steps=10000)
    opt_arm = np.argmax(p)
    for agent in agents:
        assert np.allclose(agent.Qs, p, rtol=0.1)
        assert np.isclose(agent.Qs[opt_arm], p[opt_arm], rtol=0.01)


@pytest.mark.parametrize("loc,scale", [([0.2, 0.6], [0.5, 0.5])])
@pytest.mark.parametrize("eps", [[0.1, 0.3]])
def test_epsilon_greedy_gaussian_bandits_simulation(loc, scale, eps):
    rng = np.random.default_rng(seed=3028)
    bandit = GaussianArm.bandit(loc=loc, scale=scale, rng=rng)
    agents = [EpsilonGreedyStrategy(eps=e).agent() for e in eps]
    sim = Simulation(agents=agents, bandit=bandit, rng=rng)
    sim.run(trials=1, steps=100000)
    opt_arm = np.argmax(loc)
    for agent in agents:
        assert np.allclose(agent.Qs, loc, rtol=0.1)
        assert np.isclose(agent.Qs[opt_arm], loc[opt_arm], rtol=0.01)
