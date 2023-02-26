import numpy as np
import pytest

from mabby.arms import Arm, ArmSet, BernoulliArm, GaussianArm


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
    def armset_params(self, request):
        return request.param

    @pytest.fixture
    def armset(self, armset_params):
        return self.ARM_CLASS.armset(**armset_params)

    @pytest.fixture(params=[100000])
    def sample(self, request, arm):
        rng = np.random.default_rng(seed=0)
        return [arm.play(rng) for _ in range(request.param)]

    def test_armset_returns_armset_with_correct_types(self, armset):
        assert isinstance(armset, ArmSet)
        for arm in armset:
            assert isinstance(arm, self.ARM_CLASS)

    def test_armset_returns_armset_with_correct_length(self, armset_params, armset):
        expected_armset_length = min([len(v) for v in armset_params.values()])
        assert len(armset) == expected_armset_length


class TestBernoulliArm(TestArm):
    ARM_CLASS = BernoulliArm

    @pytest.fixture(params=[{"p": [0.1, 0.3]}])
    def armset_params(self, request):
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
    def armset_params(self, request):
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


class TestArmSet:
    @pytest.fixture(params=[[1, 3], [0.2, 0.7, 0.3]])
    def mock_arms(self, mocker, request):
        return [mocker.Mock(mean=m) for m in request.param]

    @pytest.fixture()
    def mock_armset(self, mock_arms):
        return ArmSet(arms=mock_arms)

    def test_armset_init_sets_arms_list(self, mock_arms, mock_armset):
        arms_list = mock_armset._arms
        assert arms_list == mock_arms

    def test_armset_len_returns_num_arms(self, mock_arms, mock_armset):
        num_arms = len(mock_armset)
        assert num_arms == len(mock_arms)

    def test_armset_repr_returns_arm_list_repr(self, mock_arms, mock_armset):
        armset_repr = repr(mock_armset)
        assert armset_repr == repr(mock_arms)

    def test_armset_getitem_returns_correct_arm(self, mock_arms, mock_armset):
        for i, arm in enumerate(mock_armset):
            assert arm == mock_arms[i]

    @pytest.mark.parametrize("play_choice", [0, 1])
    def test_armset_play_invokes_play_of_correct_arm(
        self, mock_arms, mock_armset, mock_rng, play_choice
    ):
        mock_armset.play(play_choice, mock_rng)
        mock_arms[play_choice].play.assert_called_once_with(mock_rng)
        for i in filter(lambda x: x != play_choice, range(len(mock_arms))):
            mock_arms[i].play.assert_not_called()

    def test_armset_best_arm_returns_arm_with_max_mean(self, mock_arms, mock_armset):
        best_arm = mock_armset.best_arm()
        assert mock_arms[best_arm].mean == max(arm.mean for arm in mock_arms)
