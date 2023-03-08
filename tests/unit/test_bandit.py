import numpy as np
import pytest

from mabby.bandit import Arm, Bandit, BernoulliArm, GaussianArm


@pytest.fixture()
def mock_rng(mocker):
    return mocker.Mock()


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

    def test_init_sets_p(self, arm, valid_params):
        assert arm.p == valid_params["p"]

    def test_play_generates_bernoulli_distribution(self, sample, valid_params):
        assert np.logical_or(np.equal(sample, 0), np.equal(sample, 1)).any()
        assert np.isclose(np.mean(sample), valid_params["p"], rtol=0.01)

    def test_mean_equals_to_p(self, arm, valid_params):
        assert arm.mean == valid_params["p"]

    def test_repr_includes_p(self, arm, valid_params):
        assert str(valid_params["p"]) in repr(arm)


class TestGaussianArm(TestArm):
    ARM_CLASS = GaussianArm

    @pytest.fixture(params=[{"loc": [0.1, 0.3], "scale": [2]}])
    def bandit_params(self, request):
        return request.param

    @pytest.fixture(params=[{"loc": 0.1, "scale": 2}])
    def valid_params(self, request):
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


class TestBandit:
    @pytest.fixture(params=[[1, 3], [0.2, 0.7, 0.3]])
    def mock_arms(self, mocker, request):
        return [mocker.Mock(mean=m) for m in request.param]

    @pytest.fixture()
    def mock_bandit(self, mock_arms):
        return Bandit(arms=mock_arms)

    def test_init_sets_arms_list(self, mock_arms, mock_bandit):
        arms_list = mock_bandit._arms
        assert arms_list == mock_arms

    def test_len_returns_num_arms(self, mock_arms, mock_bandit):
        num_arms = len(mock_bandit)
        assert num_arms == len(mock_arms)

    def test_repr_returns_arm_list_repr(self, mock_arms, mock_bandit):
        bandit_repr = repr(mock_bandit)
        assert bandit_repr == repr(mock_arms)

    def test_getitem_returns_correct_arm(self, mock_arms, mock_bandit):
        for i, arm in enumerate(mock_bandit):
            assert arm == mock_arms[i]

    @pytest.mark.parametrize("choice", [0, 1])
    def test_play_invokes_play_of_correct_arm(
        self, mock_arms, mock_bandit, mock_rng, choice
    ):
        mock_bandit.play(choice, mock_rng)
        mock_arms[choice].play.assert_called_once_with(mock_rng)
        for i in filter(lambda x: x != choice, range(len(mock_arms))):
            mock_arms[i].play.assert_not_called()

    def test_best_arm_returns_arm_with_max_mean(self, mock_arms, mock_bandit):
        best_arm = mock_bandit.best_arm()
        assert mock_arms[best_arm].mean == max(arm.mean for arm in mock_arms)

    @pytest.mark.parametrize("choice", [0, 1])
    def test_regret_returns_difference_in_mean(self, mock_arms, mock_bandit, choice):
        regret = mock_bandit.regret(choice)
        assert regret == max(arm.mean for arm in mock_arms) - mock_arms[choice].mean
