from unittest.mock import patch

import pytest
from numpy.random import Generator

from mabby.bandits import (
    Bandit,
    BetaTSBandit,
    EpsilonGreedyBandit,
    RandomBandit,
    UCB1Bandit,
)
from mabby.exceptions import BanditUsageError
from mabby.strategies import (
    BetaTSStrategy,
    EpsilonGreedyStrategy,
    RandomStrategy,
    UCB1Strategy,
)


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
    def mock_rng(self, mocker):
        # return np.random.default_rng(seed=1234)
        return mocker.Mock(spec=Generator)

    @pytest.fixture
    def bandit(self, valid_params):
        return self.BANDIT_CLASS(**valid_params)

    @pytest.fixture
    def primed_bandit(self, prime_params, mock_rng, bandit):
        bandit.prime(**prime_params, rng=mock_rng)
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

    def test_prime_invokes_strategy_prime(self, mocker, mock_rng, bandit, prime_params):
        strategy_prime = mocker.spy(bandit.strategy, "prime")
        bandit.prime(**prime_params, rng=mock_rng)
        strategy_prime.assert_called_once_with(**prime_params)

    def test_choose_invokes_strategy_choose(
        self, mocker, mock_rng, primed_bandit, choice
    ):
        mocker.patch.object(primed_bandit.strategy, "choose", return_value=choice)
        strategy_choose = mocker.spy(primed_bandit.strategy, "choose")
        primed_bandit.choose()
        strategy_choose.assert_called_once_with(mock_rng)

    def test_choose_saves_and_returns_choice(self, mocker, primed_bandit, choice):
        mocker.patch.object(primed_bandit.strategy, "choose", return_value=choice)
        assert primed_bandit.choose() == choice
        assert primed_bandit._choice == choice

    def test_update_invokes_strategy_update(
        self, mock_rng, chosen_bandit, choice, reward
    ):
        with patch.object(chosen_bandit.strategy, "update") as strategy_update:
            chosen_bandit.update(reward)
            strategy_update.assert_called_once_with(choice, reward, mock_rng)

    def test_update_resets_choice_to_none(self, mocker, chosen_bandit, reward):
        mocker.patch.object(chosen_bandit.strategy, "update")
        chosen_bandit.update(reward)
        assert chosen_bandit._choice is None

    def test_Qs_returns_strategy_Qs(self, primed_bandit):
        assert (primed_bandit.Qs == primed_bandit.strategy.Qs).all()

    def test_Ns_returns_strategy_Ns(self, primed_bandit):
        assert (primed_bandit.Ns == primed_bandit.strategy.Ns).all()

    def test_choose_before_prime_raises_error(self, bandit):
        with pytest.raises(BanditUsageError):
            bandit.choose()

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

    def test_init_sets_epsilon_greedy_strategy(self, bandit, valid_params):
        assert isinstance(bandit.strategy, EpsilonGreedyStrategy)
        assert bandit.strategy.eps == valid_params["eps"]


class TestUCB1Bandit(TestBandit):
    BANDIT_CLASS = UCB1Bandit

    @pytest.fixture(params=[{"alpha": 2}, {"alpha": 0.5, "name": "bandit-name"}])
    def valid_params(self, request):
        return request.param

    def test_init_sets_ucb1_strategy(self, bandit, valid_params):
        assert isinstance(bandit.strategy, UCB1Strategy)
        assert bandit.strategy.alpha == valid_params["alpha"]


class TestBetaTSBandit(TestBandit):
    BANDIT_CLASS = BetaTSBandit

    @pytest.fixture(
        params=[{"general": False}, {"general": True, "name": "bandit-name"}]
    )
    def valid_params(self, request):
        return request.param

    def test_init_sets_beta_ts_strategy(self, bandit, valid_params):
        assert isinstance(bandit.strategy, BetaTSStrategy)
        assert bandit.strategy.general == valid_params["general"]
