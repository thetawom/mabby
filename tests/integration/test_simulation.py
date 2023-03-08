import numpy as np
import pytest

from mabby import (
    BernoulliArm,
    EpsilonGreedyAgent,
    GaussianArm,
    RandomAgent,
    Simulation,
)


@pytest.mark.parametrize("p", [[0.7, 0.9]])
@pytest.mark.parametrize("num_agents", [2])
def test_random_bernoulli_agents_simulation(p, num_agents):
    armset = BernoulliArm.armset(p=p)
    agents = [RandomAgent() for _ in range(num_agents)]
    sim = Simulation(agents=agents, armset=armset, seed=15)
    sim.run(trials=100, steps=1000)
    for agent in agents:
        assert np.allclose(agent.Qs, p, rtol=0.1)
        assert np.isclose(agent.Ns, np.mean(agent.Ns), rtol=0.05).all()


@pytest.mark.parametrize("loc,scale", [([2, -2], [0.5, 0.5])])
@pytest.mark.parametrize("num_agents", [2])
def test_random_gaussian_agents_simulation(loc, scale, num_agents):
    armset = GaussianArm.armset(loc=loc, scale=scale)
    agents = [RandomAgent() for _ in range(num_agents)]
    sim = Simulation(agents=agents, armset=armset, seed=63)
    sim.run(trials=100, steps=1000)
    for agent in agents:
        assert np.allclose(agent.Qs, loc, rtol=0.1)
        assert np.isclose(agent.Ns, np.mean(agent.Ns), rtol=0.05).all()


@pytest.mark.parametrize("p", [[0.2, 0.6]])
@pytest.mark.parametrize("eps", [[0.1, 0.8]])
def test_epsilon_greedy_bernoulli_agents_simulation(p, eps):
    armset = BernoulliArm.armset(p=p)
    agents = [EpsilonGreedyAgent(eps=e) for e in eps]
    sim = Simulation(agents=agents, armset=armset, seed=0)
    sim.run(trials=100, steps=1000)
    opt_arm = np.argmax(p)
    for agent in agents:
        assert np.allclose(agent.Qs, p, rtol=0.3)
        assert np.isclose(agent.Qs[opt_arm], p[opt_arm], rtol=0.05)


@pytest.mark.parametrize("loc,scale", [([0.2, 0.6], [0.5, 0.5])])
@pytest.mark.parametrize("eps", [[0.1, 0.3]])
def test_epsilon_greedy_gaussian_agents_simulation(loc, scale, eps):
    armset = GaussianArm.armset(loc=loc, scale=scale)
    agents = [EpsilonGreedyAgent(eps=e) for e in eps]
    sim = Simulation(agents=agents, armset=armset, seed=3028)
    sim.run(trials=100, steps=1000)
    opt_arm = np.argmax(loc)
    for agent in agents:
        assert np.allclose(agent.Qs, loc, rtol=0.4)
        assert np.isclose(agent.Qs[opt_arm], loc[opt_arm], rtol=0.05)
