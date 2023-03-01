import numpy as np
import pytest

from mabby import (
    BernoulliArm,
    EpsilonGreedyBandit,
    GaussianArm,
    RandomBandit,
    Simulation,
)


@pytest.mark.parametrize("p", [[0.7, 0.9]])
@pytest.mark.parametrize("num_bandits", [2])
def test_random_bernoulli_bandits_simulation(p, num_bandits):
    armset = BernoulliArm.armset(p=p)
    bandits = [RandomBandit() for _ in range(num_bandits)]
    sim = Simulation(bandits=bandits, armset=armset, seed=15)
    sim.run(trials=100, steps=1000)
    for bandit in bandits:
        assert np.allclose(bandit.Qs, p, rtol=0.1)
        assert np.isclose(bandit.Ns, np.mean(bandit.Ns), rtol=0.05).all()


@pytest.mark.parametrize("loc,scale", [([2, -2], [0.5, 0.5])])
@pytest.mark.parametrize("num_bandits", [2])
def test_random_gaussian_bandits_simulation(loc, scale, num_bandits):
    armset = GaussianArm.armset(loc=loc, scale=scale)
    bandits = [RandomBandit() for _ in range(num_bandits)]
    sim = Simulation(bandits=bandits, armset=armset, seed=63)
    sim.run(trials=100, steps=1000)
    for bandit in bandits:
        assert np.allclose(bandit.Qs, loc, rtol=0.1)
        assert np.isclose(bandit.Ns, np.mean(bandit.Ns), rtol=0.05).all()


@pytest.mark.parametrize("p", [[0.2, 0.6]])
@pytest.mark.parametrize("eps", [[0.1, 0.8]])
def test_epsilon_greedy_bernoulli_bandits_simulation(p, eps):
    armset = BernoulliArm.armset(p=p)
    bandits = [EpsilonGreedyBandit(eps=e) for e in eps]
    sim = Simulation(bandits=bandits, armset=armset, seed=0)
    sim.run(trials=100, steps=1000)
    opt_arm = np.argmax(p)
    for bandit in bandits:
        assert np.allclose(bandit.Qs, p, rtol=0.3)
        assert np.isclose(bandit.Qs[opt_arm], p[opt_arm], rtol=0.05)


@pytest.mark.parametrize("loc,scale", [([0.2, 0.6], [0.5, 0.5])])
@pytest.mark.parametrize("eps", [[0.1, 0.3]])
def test_epsilon_greedy_gaussian_bandits_simulation(loc, scale, eps):
    armset = GaussianArm.armset(loc=loc, scale=scale)
    bandits = [EpsilonGreedyBandit(eps=e) for e in eps]
    sim = Simulation(bandits=bandits, armset=armset, seed=3028)
    sim.run(trials=100, steps=1000)
    opt_arm = np.argmax(loc)
    for bandit in bandits:
        assert np.allclose(bandit.Qs, loc, rtol=0.4)
        assert np.isclose(bandit.Qs[opt_arm], loc[opt_arm], rtol=0.05)
