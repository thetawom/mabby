import random

import numpy as np
import pytest

from mabby import Arm, Bandit
from mabby.arms import BernoulliArm, GaussianArm


@pytest.fixture()
def mock_rng(mocker):
    return mocker.Mock(choice=lambda xs: random.choice(xs))


class TestArm:
    ARM_CLASS = Arm

    @pytest.fixture(autouse=True)
    def patch_abstract_methods(self, mocker):
        mocker.patch.object(Arm, "__abstractmethods__", new_callable=set)

    @pytest.fixture
    def valid_params(self):
        pass

    @pytest.fixture
    def arm(self, valid_params):
        return self.ARM_CLASS(**valid_params)

    @pytest.fixture(params=[{"x": [1, 2]}, {"y": [1, 2, 3], "z": [1, 2]}])
    def bandit_params(self, request):
        return request.param

    @pytest.fixture(params=[{}, {"x": []}, {"y": [1], "z": []}])
    def invalid_bandit_params(self, request):
        return request.param

    @pytest.fixture
    def bandit(self, bandit_params):
        return self.ARM_CLASS.bandit(**bandit_params)

    @pytest.fixture(params=[100000])
    def sample(self, request, arm):
        rng = np.random.default_rng(seed=0)
        return [arm.play(rng) for _ in range(request.param)]

    def test_bandit_returns_bandit_with_correct_types(self, bandit):
        assert isinstance(bandit, Bandit)
        for arm in bandit:
            assert isinstance(arm, self.ARM_CLASS)

    def test_bandit_returns_bandit_with_correct_length(self, bandit_params, bandit):
        expected_bandit_length = min([len(v) for v in bandit_params.values()])
        assert len(bandit) == expected_bandit_length

    def test_bandit_with_insufficient_params_raises_error(self, invalid_bandit_params):
        with pytest.raises(ValueError):
            self.ARM_CLASS.bandit(**invalid_bandit_params)


class TestBernoulliArm(TestArm):
    ARM_CLASS = BernoulliArm

    @pytest.fixture(params=[{"p": [0.1, 0.3]}])
    def bandit_params(self, request):
        return request.param

    @pytest.fixture(params=[{"p": 0.1}])
    def valid_params(self, request):
        return request.param

    @pytest.fixture(params=[{"p": [-0.1, 0.9]}])
    def invalid_params(self, request):
        return request.param

    def test_init_sets_p(self, arm, valid_params):
        assert arm.p == valid_params["p"]

    def test_play_generates_bernoulli_distribution(self, sample, valid_params):
        assert np.logical_or(np.equal(sample, 0), np.equal(sample, 1)).any()
        assert np.isclose(np.mean(sample), valid_params["p"], rtol=0.01)

    def test_mean_equals_to_p(self, arm, valid_params):
        assert arm.mean == valid_params["p"]

    def test_repr_includes_p(self, arm, valid_params):
        assert str(valid_params["p"]) in repr(arm)

    def test_invalid_p_raises_error(self, arm, invalid_params):
        with pytest.raises(ValueError):
            self.ARM_CLASS.bandit(**invalid_params)


class TestGaussianArm(TestArm):
    ARM_CLASS = GaussianArm

    @pytest.fixture(params=[{"loc": [0.1, 0.3], "scale": [2]}])
    def bandit_params(self, request):
        return request.param

    @pytest.fixture(params=[{"loc": 0.1, "scale": 2}])
    def valid_params(self, request):
        return request.param

    @pytest.fixture(params=[{"loc": [0.1, 0.3], "scale": [-1]}])
    def invalid_params(self, request):
        return request.param

    def test_init_sets_loc_and_scale(self, arm, valid_params):
        assert arm.loc == valid_params["loc"]
        assert arm.scale == valid_params["scale"]

    def test_play_generates_normal_distribution(self, sample, valid_params):
        assert np.isclose(np.mean(sample), valid_params["loc"], rtol=0.05)
        assert np.isclose(np.std(sample), valid_params["scale"], rtol=0.05)

    def test_mean_equals_to_loc(self, arm, valid_params):
        assert arm.mean == valid_params["loc"]

    def test_repr_includes_loc_and_scale(self, arm, valid_params):
        assert str(valid_params["loc"]) in repr(arm)
        assert str(valid_params["scale"]) in repr(arm)

    def test_invalid_scale_raises_error(self, arm, invalid_params):
        with pytest.raises(ValueError):
            self.ARM_CLASS.bandit(**invalid_params)


class TestBandit:
    @pytest.fixture(params=[2, 4])
    def num_arms(self, request):
        return request.param

    @pytest.fixture
    def arms(self, num_arms, arm_factory):
        return [arm_factory.generic() for _ in range(num_arms)]

    @pytest.fixture()
    def bandit(self, arms, mock_rng):
        return Bandit(arms=arms, rng=mock_rng)

    def test_init_sets_arms_list(self, arms, bandit):
        arms_list = bandit._arms
        assert arms_list == arms

    def test_init_sets_bandit(self, bandit):
        assert bandit._rng is not None

    def test_len_returns_num_arms(self, arms, bandit):
        num_arms = len(bandit)
        assert num_arms == len(arms)

    def test_repr_returns_arm_list_repr(self, arms, bandit):
        bandit_repr = repr(bandit)
        assert bandit_repr == repr(arms)

    def test_getitem_returns_correct_arm(self, arms, bandit):
        for i, arm in enumerate(bandit):
            assert arm == arms[i]

    @pytest.mark.parametrize("choice", [0, 1])
    def test_play_invokes_play_of_correct_arm(
        self, mocker, arms, bandit, mock_rng, choice
    ):
        play_spy = mocker.spy(arms[choice], "play")
        bandit.play(choice)
        play_spy.assert_called_once_with(mock_rng)

    def test_best_arm_returns_arm_with_max_mean(self, arms, bandit):
        best_arm = bandit.best_arm()
        assert arms[best_arm].mean == max(arm.mean for arm in arms)

    def test_best_arm_returns_any_optimal_arm_if_many(self, arm_factory, num_arms):
        arms = [arm_factory.generic(mean=1) for _ in range(num_arms)]
        bandit = Bandit(arms=arms, seed=324)
        best_arm_samples = [bandit.best_arm() for _ in range(1000)]
        values, counts = np.unique(best_arm_samples, return_counts=True)
        assert len(values) == num_arms
        assert np.allclose(counts, np.mean(counts), rtol=0.1)

    def test_is_opt_returns_true_for_optimal_choice(self, arms, bandit):
        opt_choice = int(np.argmax(bandit.means))
        assert bandit.is_opt(opt_choice)

    def test_is_opt_returns_false_for_non_optimal_choice(self, arms, bandit):
        non_opt_choice = int(np.argmin(bandit.means))
        assert not bandit.is_opt(non_opt_choice)

    @pytest.mark.parametrize("choice", [0, 1])
    def test_regret_returns_difference_in_mean(self, arms, bandit, choice):
        regret = bandit.regret(choice)
        assert regret == max(arm.mean for arm in arms) - arms[choice].mean
