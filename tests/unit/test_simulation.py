import random

import pytest
from numpy.random import Generator

from mabby import Simulation
from mabby.arms import ArmSet
from mabby.stats import BanditStats, SimulationStats


@pytest.fixture(params=[2])
def num_bandits(request):
    return request.param


@pytest.fixture(params=[3])
def num_arms(request):
    return request.param


@pytest.fixture
def bandits(num_bandits, bandit_factory):
    return [bandit_factory.generic() for _ in range(num_bandits)]


@pytest.fixture
def bandit(bandits):
    return random.choice(bandits)


@pytest.fixture
def armset(num_arms, arm_factory):
    arms = [arm_factory.generic() for _ in range(num_arms)]
    return ArmSet(arms=arms)


@pytest.fixture
def simulation(bandits, armset):
    return Simulation(bandits=bandits, armset=armset)


@pytest.fixture(params=[{"trials": 3, "steps": 2}])
def run_params(request):
    return request.param


class TestSimulation:
    def test_init_sets_bandits_armset_and_rng(self, bandits, armset, simulation):
        assert simulation.bandits == bandits
        assert simulation.armset == armset
        assert isinstance(simulation._rng, Generator)

    def test_init_with_empty_armset_raises_error(self, bandits):
        empty_armset = ArmSet(arms=[])
        with pytest.raises(ValueError):
            Simulation(bandits=bandits, armset=empty_armset)

    def test_run_returns_sim_stats_with_bandit_stats(
        self, mocker, bandits, armset, simulation, run_params
    ):
        mocker.patch.object(
            simulation,
            "_run_trials_for_bandit",
            lambda b, _, steps, metrics: BanditStats(b, armset, steps, metrics),
        )
        run_trials_for_bandit_spy = mocker.spy(simulation, "_run_trials_for_bandit")
        sim_stats = simulation.run(**run_params)
        assert isinstance(sim_stats, SimulationStats)
        for bandit in bandits:
            assert bandit in sim_stats
        assert run_trials_for_bandit_spy.call_count == len(bandits)

    def test__run_trials_for_bandit_returns_bandit_stats(
        self, bandit, simulation, run_params
    ):
        bandit_stats = simulation._run_trials_for_bandit(bandit, **run_params)
        assert isinstance(bandit_stats, BanditStats)

    def test__run_trials_for_bandit_primes_bandit_each_trial(
        self, mocker, bandit, simulation, run_params
    ):
        prime_spy = mocker.spy(bandit, "prime")
        simulation._run_trials_for_bandit(bandit, **run_params)
        assert prime_spy.call_count == run_params["trials"]

    def test__run_trials_for_bandit_chooses_plays_updates_each_step(
        self, mocker, bandit, armset, simulation, run_params
    ):
        bandit_choose_spy = mocker.spy(bandit, "choose")
        armset_play_spy = mocker.spy(armset, "play")
        bandit_update_spy = mocker.spy(bandit, "update")
        bandit_stats_update_spy = mocker.spy(BanditStats, "update")
        simulation._run_trials_for_bandit(bandit, **run_params)
        total_count = run_params["trials"] * run_params["steps"]
        assert bandit_choose_spy.call_count == total_count
        assert armset_play_spy.call_count == total_count
        assert bandit_update_spy.call_count == total_count
        assert bandit_stats_update_spy.call_count == total_count
