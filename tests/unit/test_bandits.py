import pytest
from numpy.random import Generator

from mabby.bandits import Bandit, EpsilonGreedyBandit, RandomBandit
from mabby.exceptions import BanditUsageError
from mabby.strategies import EpsilonGreedyStrategy, RandomStrategy


class TestBandit:
    BANDIT_CLASS = Bandit

    @pytest.fixture(params=[{}, {"name": "bandit-name"}])
    def valid_params(self, request, strategy_factory):
        request.param["strategy"] = strategy_factory.generic()
        return request.param

    @pytest.fixture(params=[{"k": 3, "steps": 20}])
    def prime_params(self, request):
        return request.param

    @pytest.fixture(params=[1])
    def choice(self, request):
        return request.param

    @pytest.fixture(params=[1])
    def reward(self, request):
        return request.param

    @pytest.fixture
    def bandit(self, valid_params):
        return self.BANDIT_CLASS(**valid_params)

    @pytest.fixture
    def primed_bandit(self, prime_params, bandit):
        bandit.prime(**prime_params)
        return bandit

    @pytest.fixture
    def chosen_bandit(self, choice, primed_bandit):
        primed_bandit._choice = choice
        return primed_bandit

    def test_init_sets_name(self, valid_params, bandit):
        assert bandit._name == valid_params.get("name")

    @pytest.mark.parametrize("name", ["bandit-name"])
    def test_repr_returns_custom_name_if_overridden(self, valid_params, name):
        valid_params["name"] = name
        bandit = self.BANDIT_CLASS(**valid_params)
        assert repr(bandit) == name

    def test_repr_returns_strategy_name_if_not_overridden(self, valid_params):
        valid_params["name"] = None
        bandit = self.BANDIT_CLASS(**valid_params)
        assert repr(bandit) == str(bandit.strategy)

    def test_prime_invokes_strategy_prime(self, mocker, bandit, prime_params):
        strategy_prime = mocker.spy(bandit.strategy, "prime")
        bandit.prime(**prime_params)
        strategy_prime.assert_called_once_with(**prime_params)

    def test_choose_invokes_strategy_choose(self, mocker, primed_bandit, choice):
        mock_rng = mocker.Mock(spec=Generator)
        mocker.patch.object(primed_bandit.strategy, "choose", return_value=choice)
        strategy_choose = mocker.spy(primed_bandit.strategy, "choose")
        primed_bandit.choose(mock_rng)
        strategy_choose.assert_called_once_with(mock_rng)

    def test_choose_saves_and_returns_choice(self, mocker, primed_bandit, choice):
        mock_rng = mocker.Mock(spec=Generator)
        mocker.patch.object(primed_bandit.strategy, "choose", return_value=choice)
        assert primed_bandit.choose(mock_rng) == choice
        assert primed_bandit._choice == choice

    def test_update_invokes_strategy_update(
        self, mocker, chosen_bandit, choice, reward
    ):
        strategy_update = mocker.spy(chosen_bandit.strategy, "update")
        chosen_bandit.update(reward)
        strategy_update.assert_called_once_with(choice, reward)

    def test_update_resets_choice_to_none(self, chosen_bandit, reward):
        chosen_bandit.update(reward)
        assert chosen_bandit._choice is None

    def test_Qs_returns_strategy_Qs(self, primed_bandit):
        assert (primed_bandit.Qs == primed_bandit.strategy.Qs).all()

    def test_Ns_returns_strategy_Ns(self, primed_bandit):
        assert (primed_bandit.Ns == primed_bandit.strategy.Ns).all()

    def test_choose_before_prime_raises_error(self, mocker, bandit):
        mock_rng = mocker.Mock(spec=Generator)
        with pytest.raises(BanditUsageError):
            bandit.choose(rng=mock_rng)

    def test_Qs_before_prime_raises_error(self, bandit):
        with pytest.raises(BanditUsageError):
            assert bandit.Qs is not None

    def test_Ns_before_prime_raises_error(self, bandit):
        with pytest.raises(BanditUsageError):
            assert bandit.Ns is not None

    def test_update_before_choose_raises_error(self, primed_bandit, reward):
        with pytest.raises(BanditUsageError):
            primed_bandit.update(reward=reward)


class TestRandomBandit(TestBandit):
    BANDIT_CLASS = RandomBandit

    @pytest.fixture(params=[{}, {"name": "bandit-name"}])
    def valid_params(self, request):
        return request.param

    def test_init_sets_random_strategy(self, bandit):
        assert isinstance(bandit.strategy, RandomStrategy)


class TestEpsilonGreedyBandit(TestBandit):
    BANDIT_CLASS = EpsilonGreedyBandit

    @pytest.fixture(params=[{"eps": 0.2}, {"eps": 0.5, "name": "bandit-name"}])
    def valid_params(self, request):
        return request.param

    def test_init_sets_epsilon_greedy_strategy(self, bandit):
        assert isinstance(bandit.strategy, EpsilonGreedyStrategy)
