import random
from unittest.mock import patch

import pytest
from numpy.random import Generator

from mabby import Simulation
from mabby.arms import ArmSet
from mabby.simulation import SimStats


class TestSimulation:
    @pytest.fixture(params=[2])
    def num_bandits(self, request):
        return request.param

    @pytest.fixture(params=[3])
    def num_arms(self, request):
        return request.param

    @pytest.fixture
    def bandits(self, num_bandits, bandit_factory):
        return [bandit_factory.generic() for _ in range(num_bandits)]

    @pytest.fixture
    def bandit(self, bandits):
        return random.choice(bandits)

    @pytest.fixture
    def armset(self, num_arms, arm_factory):
        arms = [arm_factory.generic() for _ in range(num_arms)]
        return ArmSet(arms=arms)

    @pytest.fixture
    def sim(self, bandits, armset):
        return Simulation(bandits=bandits, armset=armset)

    @pytest.fixture
    def stats(self, mocker):
        return mocker.Mock(spec=SimStats)

    @pytest.fixture(params=[{"trials": 3, "steps": 2}])
    def run_params(self, request):
        return request.param

    def test_init_sets_bandits_armset_and_rng(self, bandits, armset, sim):
        assert sim.bandits == bandits
        assert sim.armset == armset
        assert isinstance(sim._rng, Generator)

    @patch.object(Simulation, "_run_trial")
    def test_run_creates_and_returns_compiled_sim_stats(
        self, _, mocker, sim, run_params
    ):
        init_stats = mocker.spy(SimStats, "__init__")
        compile_stats = mocker.spy(SimStats, "compile_all")
        stats = sim.run(**run_params)
        assert init_stats.call_count == 1
        assert compile_stats.call_count == 1
        assert isinstance(stats, SimStats)

    @patch.object(Simulation, "_run_trial")
    def test_run_invokes__run_trial_correct_number_of_times(
        self, run_trial, bandits, sim, run_params
    ):
        sim.run(**run_params)
        total_trials = len(bandits) * run_params["trials"]
        assert run_trial.call_count == total_trials

    def test__run_trial_primes_bandit(
        self, mocker, bandit, armset, stats, sim, run_params
    ):
        bandit_prime = mocker.spy(bandit, "prime")
        sim._run_trial(stats=stats, bandit=bandit, steps=run_params["steps"])
        bandit_prime.assert_called_once_with(k=len(armset), steps=run_params["steps"])

    def test__run_trial_runs_sim_loop(
        self, mocker, bandit, armset, stats, sim, run_params
    ):
        bandit_choose = mocker.spy(bandit, "choose")
        armset_play = mocker.spy(armset, "play")
        bandit_update = mocker.spy(bandit, "update")
        stats_update = mocker.spy(stats, "update")
        sim._run_trial(stats=stats, bandit=bandit, steps=run_params["steps"])
        assert bandit_choose.call_count == run_params["steps"]
        assert armset_play.call_count == run_params["steps"]
        assert bandit_update.call_count == run_params["steps"]
        assert stats_update.call_count == run_params["steps"]
