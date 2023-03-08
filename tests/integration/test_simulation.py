import numpy as np
import pytest

from mabby import (
    BernoulliArm,
    EpsilonGreedyStrategy,
    GaussianArm,
    RandomStrategy,
    Simulation,
)


@pytest.mark.parametrize("p", [[0.7, 0.9]])
@pytest.mark.parametrize("num_agents", [2])
def test_random_bernoulli_agents_simulation(p, num_agents):
    bandit = BernoulliArm.bandit(p=p)
    agents = [RandomStrategy().agent() for _ in range(num_agents)]
    sim = Simulation(agents=agents, bandit=bandit, seed=15)
    sim.run(trials=100, steps=1000)
    for agent in agents:
        assert np.allclose(agent.Qs, p, rtol=0.1)
        assert np.isclose(agent.Ns, np.mean(agent.Ns), rtol=0.05).all()


@pytest.mark.parametrize(("loc", "scale"), [([2, -2], [0.5, 0.5])])
@pytest.mark.parametrize("num_agents", [2])
def test_random_gaussian_agents_simulation(loc, scale, num_agents):
    bandit = GaussianArm.bandit(loc=loc, scale=scale)
    agents = [RandomStrategy().agent() for _ in range(num_agents)]
    sim = Simulation(agents=agents, bandit=bandit, seed=63)
    sim.run(trials=100, steps=1000)
    for agent in agents:
        assert np.allclose(agent.Qs, loc, rtol=0.1)
        assert np.isclose(agent.Ns, np.mean(agent.Ns), rtol=0.05).all()


@pytest.mark.parametrize("p", [[0.2, 0.6]])
@pytest.mark.parametrize("eps", [[0.1, 0.8]])
def test_epsilon_greedy_bernoulli_agents_simulation(p, eps):
    bandit = BernoulliArm.bandit(p=p)
    agents = [EpsilonGreedyStrategy(eps=e).agent() for e in eps]
    sim = Simulation(agents=agents, bandit=bandit, seed=0)
    sim.run(trials=100, steps=1000)
    opt_arm = np.argmax(p)
    for agent in agents:
        assert np.allclose(agent.Qs, p, rtol=0.3)
        assert np.isclose(agent.Qs[opt_arm], p[opt_arm], rtol=0.05)


@pytest.mark.parametrize(("loc", "scale"), [([0.2, 0.6], [0.5, 0.5])])
@pytest.mark.parametrize("eps", [[0.1, 0.3]])
def test_epsilon_greedy_gaussian_agents_simulation(loc, scale, eps):
    bandit = GaussianArm.bandit(loc=loc, scale=scale)
    agents = [EpsilonGreedyStrategy(eps=e).agent() for e in eps]
    sim = Simulation(agents=agents, bandit=bandit, seed=3028)
    sim.run(trials=100, steps=1000)
    opt_arm = np.argmax(loc)
    for agent in agents:
        assert np.allclose(agent.Qs, loc, rtol=0.4)
        assert np.isclose(agent.Qs[opt_arm], loc[opt_arm], rtol=0.05)
