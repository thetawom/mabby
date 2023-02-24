import pytest

from mabby.arms import ArmSet


@pytest.fixture()
def mock_arms(mocker):
    MEANS = [0.2, 0.7, 0.3]
    return [mocker.MagicMock(mean=m) for m in MEANS]


@pytest.fixture()
def mock_armset(mock_arms):
    return ArmSet(arms=mock_arms)


class TestArmSet:
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
        self, mocker, mock_arms, mock_armset
    ):
        rng = mocker.MagicMock()
        mock_armset.play(1, rng)
        for i, arm in enumerate(mock_armset):
            if i == 1:
                arm.play.assert_called_once_with(rng)
            else:
                arm.play.assert_not_called()

    def test_armset_best_arm_returns_arm_with_max_mean(self, mock_armset):
        best_arm = mock_armset.best_arm()
        assert best_arm == 1
