from unittest.mock import patch

import pytest
from numpy.random import Generator

from mabby import Simulation
from mabby.simulation import SimStats


class TestSimulation:
    @pytest.fixture(params=[2])
    def mock_bandits(self, mocker, request):
        return [mocker.MagicMock() for _ in range(request.param)]

    @pytest.fixture(params=[0])
    def mock_bandit(self, request, mock_bandits):
        return mock_bandits[request.param]

    @pytest.fixture(params=[3])
    def mock_armset(self, mocker, request):
        return mocker.MagicMock(__len__=lambda _: request.param)

    @pytest.fixture
    def simulation(self, mock_bandits, mock_armset):
        return Simulation(bandits=mock_bandits, armset=mock_armset)

    @pytest.fixture
    def mock_stats(self, mocker):
        return mocker.MagicMock()

    @pytest.fixture(params=[{"trials": 3, "steps": 2}])
    def run_params(self, request):
        return request.param

    def test_init_sets_bandits_armset_and_rng(
        self, mock_bandits, mock_armset, simulation
    ):
        assert simulation.bandits == mock_bandits
        assert simulation.armset == mock_armset
        assert isinstance(simulation._rng, Generator)

    def test_run_creates_and_returns_compiled_sim_stats(
        self, mocker, mock_bandits, simulation, run_params
    ):
        mocker.patch.object(simulation, "_run_trial")
        stats_init = mocker.spy(SimStats, "__init__")
        stats_compile = mocker.spy(SimStats, "compile_all")
        stats = simulation.run(**run_params)
        stats_init.assert_called_once()
        stats_compile.assert_called_once()
        assert isinstance(stats, SimStats)

    def test_run_invokes__run_trial_correct_number_of_times(
        self, mock_bandits, simulation, run_params
    ):
        with patch.object(simulation, "_run_trial") as run_trials:
            simulation.run(**run_params)
            assert run_trials.call_count == len(mock_bandits) * run_params["trials"]

    def test__run_trial_primes_bandit(
        self, mocker, mock_bandit, mock_armset, mock_stats, simulation, run_params
    ):
        bandit_prime = mocker.spy(mock_bandit, "prime")
        simulation._run_trial(
            stats=mock_stats, bandit=mock_bandit, steps=run_params["steps"]
        )
        bandit_prime.assert_called_once_with(len(mock_armset), run_params["steps"])

    def test__run_trial_runs_sim_loop(
        self, mocker, mock_bandit, mock_armset, mock_stats, simulation, run_params
    ):
        bandit_choose = mocker.spy(mock_bandit, "choose")
        armset_play = mocker.spy(mock_armset, "play")
        bandit_update = mocker.spy(mock_bandit, "update")
        stats_update = mocker.spy(mock_stats, "update")
        simulation._run_trial(
            stats=mock_stats, bandit=mock_bandit, steps=run_params["steps"]
        )
        assert bandit_choose.call_count == run_params["steps"]
        assert armset_play.call_count == run_params["steps"]
        assert bandit_update.call_count == run_params["steps"]
        assert stats_update.call_count == run_params["steps"]
