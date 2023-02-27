import random
from collections import defaultdict
from unittest.mock import patch

import pytest
from numpy.random import Generator

from mabby import Simulation
from mabby.arms import ArmSet
from mabby.exceptions import SimulationUsageError
from mabby.simulation import SimStats


@pytest.fixture(params=[2])
def num_bandits(request):
    return request.param


@pytest.fixture(params=[3])
def num_arms(request):
    return request.param


@pytest.fixture
def bandit(bandits):
    return random.choice(bandits)


@pytest.fixture
def bandits(num_bandits, bandit_factory):
    return [bandit_factory.generic() for _ in range(num_bandits)]


@pytest.fixture
def armset(num_arms, arm_factory):
    arms = [arm_factory.generic() for _ in range(num_arms)]
    return ArmSet(arms=arms)


@pytest.fixture
def sim(bandits, armset):
    return Simulation(bandits=bandits, armset=armset)


@pytest.fixture(params=[{"trials": 3, "steps": 2}])
def run_params(request):
    return request.param


class TestSimulation:
    @pytest.fixture
    def stats(self, mocker):
        return mocker.Mock(spec=SimStats)

    def test_init_sets_bandits_armset_and_rng(self, bandits, armset, sim):
        assert sim.bandits == bandits
        assert sim.armset == armset
        assert isinstance(sim._rng, Generator)

    @patch.object(Simulation, "_run_trial")
    def test_run_creates_and_returns_compiled_sim_stats(
        self, _, mocker, sim, run_params
    ):
        init_stats = mocker.spy(SimStats, "__init__")
        compile_stats = mocker.spy(SimStats, "compile")
        stats = sim.run(**run_params)
        assert init_stats.call_count == 1
        assert compile_stats.call_count == 1
        assert isinstance(stats, SimStats)
        assert stats._compiled

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


class TestSimStats:
    @pytest.fixture(autouse=True)
    def patch_matplotlib_pyplot_show(self, mocker):
        mocker.patch("matplotlib.pyplot.show")

    @pytest.fixture
    def stats(self, sim, run_params):
        trials, steps = run_params["trials"], run_params["steps"]
        return SimStats(simulation=sim, trials=trials, steps=steps)

    @pytest.fixture
    def updated_stats(self, stats, num_arms, run_params, bandits):
        for bandit in bandits:
            for _ in range(run_params["trials"]):
                for i in range(run_params["steps"]):
                    stats.update(bandit, i, random.randint(0, num_arms - 1), 1)
        return stats

    @pytest.fixture
    def compiled_stats(self, updated_stats):
        return updated_stats.compile()

    def test_init_sets_stats_dicts(self, stats):
        assert isinstance(stats._rewards, defaultdict)
        assert isinstance(stats._optimality, defaultdict)
        assert isinstance(stats._regret, defaultdict)
        assert not stats._compiled

    def test_rewards_before_compiling_raises_error(self, updated_stats):
        with pytest.raises(SimulationUsageError):
            assert updated_stats.rewards

    def test_rewards_after_compiling_returns_rewards(self, compiled_stats):
        assert compiled_stats.rewards == compiled_stats._rewards

    def test_optimality_before_compiling_raises_error(self, updated_stats):
        with pytest.raises(SimulationUsageError):
            assert updated_stats.optimality

    def test_optimality_after_compiling_returns_rewards(self, compiled_stats):
        assert compiled_stats.optimality == compiled_stats._optimality

    def test_regret_before_compiling_raises_error(self, updated_stats):
        with pytest.raises(SimulationUsageError):
            assert updated_stats.regret

    def test_regret_after_compiling_returns_rewards(self, compiled_stats):
        assert compiled_stats.regret == compiled_stats._regret

    def test_compile_returns_self(self, updated_stats):
        compiled_stats = updated_stats.compile()
        assert compiled_stats == updated_stats

    def test_plot_rewards_before_compiling_raises_error(self, updated_stats):
        with pytest.raises(SimulationUsageError):
            updated_stats.plot_rewards()

    @patch("matplotlib.pyplot.plot")
    def test_plot_rewards_after_compiling_calls_plot(
        self, plot, compiled_stats, num_bandits
    ):
        compiled_stats.plot_rewards()
        assert plot.call_count == num_bandits

    def test_plot_optimality_before_compiling_raises_error(self, updated_stats):
        with pytest.raises(SimulationUsageError):
            updated_stats.plot_optimality()

    @patch("matplotlib.pyplot.plot")
    def test_plot_optimality_after_compiling_calls_plot(
        self, plot, compiled_stats, num_bandits
    ):
        compiled_stats.plot_optimality()
        assert plot.call_count == num_bandits

    def test_plot_regret_before_compiling_raises_error(self, updated_stats):
        with pytest.raises(SimulationUsageError):
            updated_stats.plot_regret()

    @patch("matplotlib.pyplot.plot")
    def test_plot_regret_after_compiling_calls_plot(
        self, plot, compiled_stats, num_bandits
    ):
        compiled_stats.plot_regret()
        assert plot.call_count == num_bandits
