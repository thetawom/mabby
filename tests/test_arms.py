import pytest

from mabby.arms import ArmSet, BernoulliArm, GaussianArm


@pytest.fixture()
def mock_rng(mocker):
    return mocker.MagicMock()


class TestArmSet:
    @pytest.fixture()
    def mock_arms(self, mocker):
        MEANS = [0.2, 0.7, 0.3]
        return [mocker.MagicMock(mean=m) for m in MEANS]

    @pytest.fixture()
    def mock_armset(self, mock_arms):
        return ArmSet(arms=mock_arms)

    def test_armset_init_sets_arms_list(self, mock_arms, mock_armset):
        assert mock_armset._arms == mock_arms

    def test_armset_len_returns_num_arms(self, mock_arms, mock_armset):
        assert len(mock_armset) == len(mock_arms)

    def test_armset_repr_returns_arm_list_repr(self, mock_arms, mock_armset):
        assert repr(mock_armset) == repr(mock_arms)

    def test_armset_getitem_returns_correct_arm(self, mock_arms, mock_armset):
        for i, arm in enumerate(mock_armset):
            assert arm == mock_arms[i]

    def test_armset_play_invokes_play_of_correct_arm(
        self, mock_arms, mock_armset, mock_rng
    ):
        mock_armset.play(1, mock_rng)
        for i, arm in enumerate(mock_armset):
            if i == 1:
                arm.play.assert_called_once_with(mock_rng)
            else:
                arm.play.assert_not_called()

    def test_armset_best_arm_returns_arm_with_max_mean(self, mock_armset):
        best_arm = mock_armset.best_arm()
        assert best_arm == 1


class TestArm:
    @pytest.mark.parametrize("ps", [[0.1, 0.3]])
    def test_armset_bernoulli_armset_creates_correct_arms(self, ps):
        armset = BernoulliArm.armset(p=ps)
        for i, p in enumerate(ps):
            assert isinstance(armset[i], BernoulliArm)
            assert armset[i].p == p

    @pytest.mark.parametrize("locs", [[0.1, 0.3]])
    @pytest.mark.parametrize("scales", [[0.5, 1], [0.5]])
    def test_armset_gaussian_armset_creates_correct_arms(self, locs, scales):
        armset = GaussianArm.armset(loc=locs, scale=scales)
        for i, (loc, scale) in enumerate(zip(locs, scales)):
            assert isinstance(armset[i], GaussianArm)
            assert armset[i].loc == loc
            assert armset[i].scale == scale


@pytest.mark.parametrize("p", [0.1])
class TestBernoulliArm:
    def test_bernoulli_arm_init_sets_p(self, p):
        arm = BernoulliArm(p=p)
        assert arm.p == p

    def test_bernoulli_arm_play_invokes_rng_binomial(self, p, mock_rng):
        arm = BernoulliArm(p=p)
        arm.play(mock_rng)
        mock_rng.binomial.assert_called_once_with(1, p)

    def test_bernoulli_arm_mean_equal_to_p(self, p):
        arm = BernoulliArm(p=p)
        assert arm.mean == p

    def test_bernoulli_arm_repr_includes_p(self, p):
        arm = BernoulliArm(p=p)
        assert str(p) in repr(arm)


@pytest.mark.parametrize("loc", [0.1])
@pytest.mark.parametrize("scale", [0.5])
class TestGaussianArm:
    def test_gaussian_arm_init_sets_loc_and_scale(self, loc, scale):
        arm = GaussianArm(loc=loc, scale=scale)
        assert arm.loc == loc
        assert arm.scale == scale

    def test_gaussian_arm_play_invokes_rng_normal(self, loc, scale, mock_rng):
        arm = GaussianArm(loc=loc, scale=scale)
        arm.play(mock_rng)
        mock_rng.normal.assert_called_once_with(loc, scale)

    def test_gaussian_arm_mean_equal_to_loc(self, loc, scale):
        arm = GaussianArm(loc=loc, scale=scale)
        assert arm.mean == loc

    def test_gaussian_arm_repr_includes_loc_and_scale(self, loc, scale):
        arm = GaussianArm(loc=loc, scale=scale)
        arm_repr = repr(arm)
        assert str(loc) in arm_repr
        assert str(scale) in arm_repr
