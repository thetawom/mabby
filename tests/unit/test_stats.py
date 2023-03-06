import random
from unittest.mock import patch

import numpy as np
import pytest

from mabby import ArmSet, Simulation
from mabby.exceptions import StatsUsageError
from mabby.stats import BanditStats, Metric, SimulationStats

BASE_METRICS = [Metric.REGRET, Metric.REWARDS, Metric.OPTIMALITY]
NON_BASE_METRICS = [Metric.CUM_REGRET, Metric.CUM_REWARDS]


@pytest.fixture(params=list(Metric))
def metric(request):
    return request.param


@pytest.fixture(params=BASE_METRICS)
def base_metric(request):
    return request.param


@pytest.fixture(params=NON_BASE_METRICS)
def non_base_metric(request):
    return request.param


class TestMetric:
    def test_repr_is_title_case(self, metric):
        metric_repr = repr(metric)
        assert metric_repr.istitle()
        assert "_" not in metric_repr

    def test_is_base_returns_true_for_base_metrics(self, base_metric):
        assert base_metric.is_base()

    def test_is_base_returns_true_for_non_base_metrics(self, non_base_metric):
        assert not non_base_metric.is_base()

    def test_base_returns_self_for_base_metrics(self, base_metric):
        assert base_metric.base == base_metric

    def test_base_returns_base_for_non_base_metrics(self, non_base_metric):
        assert non_base_metric.base.is_base()

    @pytest.mark.parametrize("metrics", [list(Metric), NON_BASE_METRICS, BASE_METRICS])
    def test_map_to_base_returns_only_base_metrics(self, metrics):
        base_metrics = Metric.map_to_base(metrics)
        for metric in base_metrics:
            assert metric in BASE_METRICS

    @pytest.mark.parametrize("metrics", [list(Metric), NON_BASE_METRICS, BASE_METRICS])
    def test_map_to_base_represents_all_non_base_metrics(self, metrics):
        base_metrics = Metric.map_to_base(metrics)
        for metric in metrics:
            assert metric.base in base_metrics

    def test_transform_returns_values_for_base_metrics(self, mocker, base_metric):
        values = mocker.Mock()
        assert Metric.transform(base_metric, values) == values

    def test_transform_returns_transformed_values_for_non_base_metrics(
        self, mocker, non_base_metric
    ):
        values, return_values = mocker.Mock(), mocker.Mock()
        with patch.object(
            non_base_metric._mapping, "transform", return_value=return_values
        ) as transform:
            transformed_values = Metric.transform(non_base_metric, values)
            transform.assert_called_once_with(values)
            assert transformed_values == return_values


class TestBanditStats:
    @pytest.fixture
    def bandit(self, bandit_factory):
        return bandit_factory.generic()

    @pytest.fixture
    def armset(self, arm_factory, num_arms):
        arms = [arm_factory.generic(mean=random.random()) for _ in range(num_arms)]
        return ArmSet(arms=arms)

    @pytest.fixture(params=[10])
    def steps(self, request):
        return request.param

    @pytest.fixture
    def step(self, steps):
        return random.randint(0, steps - 1)

    @pytest.fixture(params=[3])
    def num_arms(self, request):
        return request.param

    @pytest.fixture
    def choice(self, num_arms):
        return random.randint(0, num_arms - 1)

    @pytest.fixture
    def opt_choice(self, armset):
        return armset.best_arm()

    @pytest.fixture
    def non_opt_choice(self, opt_choice):
        return 1 if opt_choice == 0 else opt_choice - 1

    @pytest.fixture(params=[1])
    def reward(self, request):
        return request.param

    @pytest.fixture
    def bandit_stats(self, bandit, armset, steps):
        return BanditStats(bandit=bandit, armset=armset, steps=steps)

    def test_init_sets_bandit_armset_and_steps(
        self, bandit_stats, bandit, armset, steps
    ):
        assert bandit_stats.bandit == bandit
        assert bandit_stats._armset == armset
        assert bandit_stats._steps == steps

    def test_init_with_no_metrics_creates_correct_stats_dictionary(
        self, mocker, bandit, armset, steps
    ):
        map_to_base_spy = mocker.spy(Metric, "map_to_base")
        bandit_stats = BanditStats(bandit=bandit, armset=armset, steps=steps)
        map_to_base_spy.assert_called_once_with(list(Metric))
        assert set(bandit_stats._stats.keys()) == set(BASE_METRICS)
        for stat_values in bandit_stats._stats.values():
            assert len(stat_values) == steps

    @pytest.mark.parametrize(
        "metrics,base_metrics",
        [([Metric.REGRET, Metric.CUM_REWARDS], [Metric.REGRET, Metric.REWARDS])],
    )
    def test_init_with_metrics_creates_correct_stats_dictionary(
        self, mocker, bandit, armset, steps, metrics, base_metrics
    ):
        map_to_base_spy = mocker.spy(Metric, "map_to_base")
        bandit_stats = BanditStats(
            bandit=bandit, armset=armset, steps=steps, metrics=metrics
        )
        map_to_base_spy.assert_called_once_with(metrics)
        assert set(bandit_stats._stats.keys()) == set(base_metrics)
        for stat_values in bandit_stats._stats.values():
            assert len(stat_values) == steps

    def test_len_returns_number_of_steps(self, bandit_stats, steps):
        assert len(bandit_stats) == steps

    @pytest.mark.parametrize("counts", [3])
    def test_getitem_returns_transformed_average(
        self, mocker, bandit_stats, metric, counts
    ):
        mocker.patch.object(bandit_stats, "_counts", counts)
        transform_spy = mocker.spy(metric, "transform")
        stats = bandit_stats[metric]
        transform_spy.assert_called_once()
        assert (stats == transform_spy.spy_return).all()

    def test_update_increments_count_for_step(self, bandit_stats, step, choice, reward):
        original_count = bandit_stats._counts[step]
        bandit_stats.update(step=step, choice=choice, reward=reward)
        assert bandit_stats._counts[step] == original_count + 1

    @pytest.mark.parametrize("metrics", [[Metric.CUM_REGRET], [Metric.REGRET]])
    def test_update_keeps_regret_when_optimal(
        self, bandit, armset, metrics, steps, step, opt_choice, reward
    ):
        bandit_stats = BanditStats(bandit, armset, steps, metrics)
        prev_regret = bandit_stats._stats[Metric.REGRET][step]
        bandit_stats.update(step=step, choice=opt_choice, reward=reward)
        assert bandit_stats._stats[Metric.REGRET][step] == prev_regret

    @pytest.mark.parametrize("metrics", [[Metric.CUM_REGRET], [Metric.REGRET]])
    def test_update_updates_regret_when_not_optimal(
        self, mocker, bandit, armset, metrics, steps, step, non_opt_choice, reward
    ):
        regret_spy = mocker.spy(armset, "regret")
        bandit_stats = BanditStats(bandit, armset, steps, metrics)
        prev_regret = bandit_stats._stats[Metric.REGRET][step]
        bandit_stats.update(step=step, choice=non_opt_choice, reward=reward)
        assert (
            bandit_stats._stats[Metric.REGRET][step]
            == prev_regret + regret_spy.spy_return
        )

    @pytest.mark.parametrize("metrics", [[Metric.OPTIMALITY]])
    def test_update_increments_optimality_when_optimal(
        self, bandit, armset, metrics, steps, step, opt_choice, reward
    ):
        bandit_stats = BanditStats(bandit, armset, steps, metrics)
        prev_optimality = bandit_stats._stats[Metric.OPTIMALITY][step]
        bandit_stats.update(step=step, choice=opt_choice, reward=reward)
        assert bandit_stats._stats[Metric.OPTIMALITY][step] == prev_optimality + 1

    @pytest.mark.parametrize("metrics", [[Metric.OPTIMALITY]])
    def test_update_keeps_optimality_when_not_optimal(
        self, bandit, armset, metrics, steps, step, non_opt_choice, reward
    ):
        bandit_stats = BanditStats(bandit, armset, steps, metrics)
        prev_optimality = bandit_stats._stats[Metric.OPTIMALITY][step]
        bandit_stats.update(step=step, choice=non_opt_choice, reward=reward)
        print("opt_choice", armset.best_arm())
        print("non_opt_choice", non_opt_choice)
        assert bandit_stats._stats[Metric.OPTIMALITY][step] == prev_optimality

    @pytest.mark.parametrize("metrics", [[Metric.CUM_REWARDS], [Metric.REWARDS]])
    def test_update_updates_rewards(
        self, bandit, armset, metrics, steps, step, choice, reward
    ):
        bandit_stats = BanditStats(bandit, armset, steps, metrics)
        prev_rewards = bandit_stats._stats[Metric.REWARDS][step]
        bandit_stats.update(step=step, choice=choice, reward=reward)
        assert bandit_stats._stats[Metric.REWARDS][step] == prev_rewards + reward


class TestSimulationStats:
    @pytest.fixture(autouse=True)
    def patch_matplotlib_pyplot_show(self, mocker):
        mocker.patch("matplotlib.pyplot.show")

    @pytest.fixture(params=[3])
    def bandits(self, request, bandit_factory):
        return [bandit_factory.generic() for _ in range(request.param)]

    @pytest.fixture
    def bandit(self, bandits):
        return random.choice(bandits)

    @pytest.fixture(params=[3])
    def armset(self, request, arm_factory):
        arms = [arm_factory.generic(mean=random.random()) for _ in range(request.param)]
        return ArmSet(arms=arms)

    @pytest.fixture
    def simulation(self, bandits, armset):
        return Simulation(bandits=bandits, armset=armset)

    @pytest.fixture(params=[10])
    def steps(self, request):
        return request.param

    @pytest.fixture
    def bandit_stats(self, bandit, armset, steps):
        return BanditStats(bandit=bandit, armset=armset, steps=steps)

    @pytest.fixture
    def sim_stats(self, simulation):
        return SimulationStats(simulation=simulation)

    @pytest.fixture
    def filled_sim_stats(self, sim_stats, bandits, armset, steps):
        for bandit in bandits:
            sim_stats.add(BanditStats(bandit, armset, steps))
        return sim_stats

    @pytest.fixture
    def plot_spy(self, mocker, filled_sim_stats):
        return mocker.spy(filled_sim_stats, "plot")

    def test_init_sets_simulation_and_stats_dict(self, sim_stats, simulation):
        assert sim_stats._simulation == simulation
        assert isinstance(sim_stats._stats_dict, dict)

    def test_add_puts_bandit_stats_in_stats_dict(self, sim_stats, bandit, bandit_stats):
        sim_stats.add(bandit_stats)
        assert sim_stats._stats_dict[bandit] == bandit_stats

    def test_getitem_returns_bandit_stats_of_bandit(
        self, sim_stats, bandit, bandit_stats
    ):
        sim_stats._stats_dict[bandit] = bandit_stats
        assert sim_stats[bandit] == bandit_stats

    def test_setitem_puts_bandit_stats_in_stats_dict(
        self, sim_stats, bandit, bandit_stats
    ):
        sim_stats[bandit] = bandit_stats
        assert sim_stats._stats_dict[bandit] == bandit_stats

    def test_setitem_raises_error_with_non_matching_bandit(
        self, sim_stats, bandit, bandits, bandit_stats
    ):
        other_bandit = random.choice(list(filter(lambda b: b != bandit, bandits)))
        with pytest.raises(StatsUsageError):
            sim_stats[other_bandit] = bandit_stats

    def test_contains_returns_true_if_bandit_tracked(self, filled_sim_stats, bandits):
        for bandit in bandits:
            assert bandit in filled_sim_stats

    def test_contains_returns_false_if_bandit_not_tracked(
        self, sim_stats, bandit_factory
    ):
        other_bandit = bandit_factory.generic()
        assert other_bandit not in sim_stats

    @patch("matplotlib.pyplot.plot")
    def test_plot_plots_stats_for_each_bandit(
        self, plot, filled_sim_stats, metric, bandits
    ):
        filled_sim_stats.plot(metric=metric)
        calls = plot.call_args_list
        for i, bandit in enumerate(bandits):
            bandit_stats = filled_sim_stats[bandit]
            np.testing.assert_array_equal(bandit_stats[metric], calls[i][0][0])
            assert calls[i][1]["label"] == str(bandit)

    def test_plot_regret_invokes_plot_when_cumulative_is_true(
        self, plot_spy, filled_sim_stats
    ):
        filled_sim_stats.plot_regret()
        plot_spy.assert_called_once_with(Metric.CUM_REGRET)

    def test_plot_regret_invokes_plot_when_cumulative_is_false(
        self, plot_spy, filled_sim_stats
    ):
        filled_sim_stats.plot_regret(cumulative=False)
        plot_spy.assert_called_once_with(Metric.REGRET)

    def test_plot_optimality_invokes_plot(self, plot_spy, filled_sim_stats):
        filled_sim_stats.plot_optimality()
        plot_spy.assert_called_once_with(Metric.OPTIMALITY)

    def test_plot_rewards_invokes_plot_when_cumulative_is_true(
        self, plot_spy, filled_sim_stats
    ):
        filled_sim_stats.plot_rewards()
        plot_spy.assert_called_once_with(Metric.CUM_REWARDS)

    def test_plot_rewards_invokes_plot_when_cumulative_is_false(
        self, plot_spy, filled_sim_stats
    ):
        filled_sim_stats.plot_rewards(cumulative=False)
        plot_spy.assert_called_once_with(Metric.REWARDS)
