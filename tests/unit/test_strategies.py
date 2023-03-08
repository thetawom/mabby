import numpy as np
import pytest
from numpy.random import Generator

from mabby.exceptions import BanditUsageError
from mabby.strategies import (
    BetaTSStrategy,
    EpsilonGreedyStrategy,
    RandomStrategy,
    SemiUniformStrategy,
    Strategy,
    UCB1Strategy,
)


class TestStrategy:
    STRATEGY_CLASS = Strategy

    @pytest.fixture(autouse=True)
    def patch_abstract_methods(self, mocker):
        mocker.patch.object(Strategy, "__abstractmethods__", set())

    @pytest.fixture(params=[{}])
    def valid_params(self, request):
        return request.param

    @pytest.fixture(params=[])
    def invalid_params(self, request):
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
    def strategy(self, valid_params):
        return self.STRATEGY_CLASS(**valid_params)

    @pytest.fixture
    def primed_strategy(self, strategy, prime_params):
        strategy.prime(**prime_params)
        return strategy

    def test_init_raises_error_with_invalid_params(self, invalid_params):
        with pytest.raises(ValueError):
            self.STRATEGY_CLASS(**invalid_params)


class TestSemiUniformStrategy(TestStrategy):
    STRATEGY_CLASS = SemiUniformStrategy

    @pytest.fixture(autouse=True)
    def patch_abstract_methods(self, mocker):
        mocker.patch.object(SemiUniformStrategy, "__abstractmethods__", set())

    @pytest.fixture(params=[{}])
    def valid_params(self, request):
        return request.param

    @pytest.fixture(params=[0.5])
    def effective_eps(self, mocker, request, strategy):
        mocker.patch.object(strategy, "effective_eps", return_value=request.param)
        return request.param

    @pytest.fixture(params=[[0.1, 0.5, 0.2], [2, 0]])
    def Qs(self, request):
        return request.param

    @pytest.fixture(params=[[1, 1, 1]])
    def Ns(self, request):
        return request.param

    def test_prime_inits_Qs_and_Ns(self, prime_params, primed_strategy):
        assert isinstance(primed_strategy._Qs, np.ndarray)
        assert isinstance(primed_strategy._Ns, np.ndarray)
        assert len(primed_strategy._Qs) == prime_params["k"]
        assert len(primed_strategy._Ns) == prime_params["k"]
        assert not primed_strategy._Qs.any()
        assert not primed_strategy._Ns.any()

    def test_choose_with_low_rng_explores(
        self, mocker, effective_eps, prime_params, primed_strategy
    ):
        mock_rng = mocker.Mock(random=lambda: 0.9 * effective_eps)
        explore = mocker.spy(primed_strategy, "_explore")
        primed_strategy.choose(mock_rng)
        explore.assert_called_once_with(mock_rng)

    def test_choose_with_high_rng_exploits(
        self, mocker, effective_eps, primed_strategy
    ):
        mock_rng = mocker.Mock(random=lambda: 1.1 * effective_eps)
        exploit = mocker.spy(primed_strategy, "_exploit")
        primed_strategy.choose(mock_rng)
        exploit.assert_called_once_with()

    def test_choose_without_rng_raises_error(self, primed_strategy):
        with pytest.raises(BanditUsageError):
            primed_strategy.choose()

    def test_explore_follows_uniform_distribution(self):
        pass

    def test_exploit_returns_optimal_arm(self, primed_strategy, Qs):
        primed_strategy._Qs = Qs
        choice = primed_strategy._exploit()
        assert Qs[choice] == max(Qs)

    def test_update_updates_Qs_and_Ns(
        self, prime_params, primed_strategy, choice, reward
    ):
        primed_strategy._Ns = np.ones(prime_params["k"])
        primed_strategy.update(choice, reward)
        assert primed_strategy._Qs[choice] == reward / 2
        assert primed_strategy._Ns[choice] == 2
        assert sum(primed_strategy._Ns) == prime_params["k"] + 1

    def test_Qs_returns_Qs(self, primed_strategy, Qs):
        primed_strategy._Qs = Qs
        assert primed_strategy.Qs == Qs

    def test_Ns_returns_Ns(self, primed_strategy, Ns):
        primed_strategy._Ns = Ns
        assert primed_strategy.Ns == Ns


class TestRandomStrategy(TestSemiUniformStrategy):
    STRATEGY_CLASS = RandomStrategy

    @pytest.fixture(params=[{}])
    def valid_params(self, request):
        return request.param

    def test_repr_equals_random(self, strategy):
        assert repr(strategy) == "random"

    def test_effective_eps_equals_1(self, strategy):
        assert strategy.effective_eps() == 1


class TestEpsilonGreedyStrategy(TestSemiUniformStrategy):
    STRATEGY_CLASS = EpsilonGreedyStrategy

    @pytest.fixture(params=[{"eps": 0.1}])
    def valid_params(self, request):
        return request.param

    @pytest.fixture(params=[{"eps": -1}, {"eps": 1.2}])
    def invalid_params(self, request):
        return request.param

    def test_init_sets_eps(self, valid_params, strategy):
        assert strategy.eps == valid_params["eps"]

    def test_repr_includes_eps(self, valid_params, strategy):
        assert str(valid_params["eps"]) in repr(strategy)

    def test_effective_eps_equals_eps(self, valid_params, strategy):
        assert strategy.effective_eps() == valid_params["eps"]


class TestUCB1Strategy(TestStrategy):
    STRATEGY_CLASS = UCB1Strategy

    @pytest.fixture(params=[{"alpha": 0.3}])
    def valid_params(self, request):
        return request.param

    @pytest.fixture(params=[{"alpha": -2}])
    def invalid_params(self, request):
        return request.param

    @pytest.fixture(params=[([0.1, 0.5, 0.2], [1, 1, 1]), ([2, 5], [10, 3])])
    def Qs_Ns(self, request):
        return request.param

    def test_init_sets_alpha(self, valid_params, strategy):
        assert strategy.alpha == valid_params["alpha"]

    def test_repr_includes_alpha(self, valid_params, strategy):
        assert str(valid_params["alpha"]) in repr(strategy)

    def test_prime_inits_t(self, primed_strategy):
        assert primed_strategy._t == 0

    def test_prime_inits_Qs_and_Ns(self, prime_params, primed_strategy):
        assert isinstance(primed_strategy._Qs, np.ndarray)
        assert isinstance(primed_strategy._Ns, np.ndarray)
        assert len(primed_strategy._Qs) == prime_params["k"]
        assert len(primed_strategy._Ns) == prime_params["k"]
        assert not primed_strategy._Qs.any()
        assert not primed_strategy._Ns.any()

    @pytest.mark.parametrize("UCBs", [[0.1, 0.5, 0.3]])
    def test_choose_returns_UCB_argmax_when_t_greater_than_k(
        self, mocker, prime_params, primed_strategy, UCBs
    ):
        mocker.patch.object(primed_strategy, "_compute_UCBs", return_value=UCBs)
        compute_UCBs = mocker.spy(primed_strategy, "_compute_UCBs")
        primed_strategy._t = prime_params["k"]
        choice = primed_strategy.choose()
        compute_UCBs.assert_called_once()
        assert UCBs[choice] == max(UCBs)

    def test_choose_returns_t_when_t_less_than_k(
        self, prime_params, primed_strategy, reward
    ):
        for t in range(prime_params["k"]):
            choice = primed_strategy.choose()
            primed_strategy.update(choice, reward)
            assert choice == t

    def test_compute_UCBs_computes_correct_values(
        self, valid_params, primed_strategy, Qs_Ns
    ):
        primed_strategy._Qs = Qs_Ns[0]
        primed_strategy._Ns = Qs_Ns[1]
        primed_strategy._t = np.sum(Qs_Ns[1])
        UCBs = primed_strategy._compute_UCBs()
        expected_UCBs = Qs_Ns[0] + valid_params["alpha"] * np.sqrt(
            np.log(primed_strategy._t) / Qs_Ns[1]
        )
        assert np.allclose(UCBs, expected_UCBs, rtol=0.01)

    def test_update_updates_Qs_and_Ns(
        self, prime_params, primed_strategy, choice, reward
    ):
        primed_strategy._t = prime_params["k"]
        primed_strategy._Ns = np.ones(prime_params["k"])
        primed_strategy.update(choice, reward)
        assert primed_strategy._Qs[choice] == reward / 2
        assert primed_strategy._Ns[choice] == 2
        assert sum(primed_strategy._Ns) == prime_params["k"] + 1
        assert primed_strategy._t == prime_params["k"] + 1

    def test_Qs_returns_Qs(self, primed_strategy, Qs_Ns):
        primed_strategy._Qs = Qs_Ns[0]
        assert primed_strategy.Qs == Qs_Ns[0]

    def test_Ns_returns_Ns(self, primed_strategy, Qs_Ns):
        primed_strategy._Ns = Qs_Ns[1]
        assert primed_strategy.Ns == Qs_Ns[1]


class TestBetaTSStrategy(TestStrategy):
    STRATEGY_CLASS = BetaTSStrategy

    @pytest.fixture(params=[{"general": True}, {"general": False}])
    def valid_params(self, request):
        return request.param

    @pytest.fixture(params=[([1, 4, 2], [2, 3, 1]), ([2, 6], [5, 3])])
    def a_b(self, request):
        return request.param

    def test_init_sets_general(self, valid_params, strategy):
        assert strategy.general == valid_params["general"]

    def test_repr_includes_general(self, valid_params, strategy):
        assert valid_params["general"] == ("generalized" in repr(strategy))

    def test_prime_inits_a_and_b(self, prime_params, primed_strategy):
        assert isinstance(primed_strategy._a, np.ndarray)
        assert isinstance(primed_strategy._b, np.ndarray)
        assert len(primed_strategy._a) == prime_params["k"]
        assert len(primed_strategy._b) == prime_params["k"]

    @pytest.mark.parametrize("beta_samples", [[0.1, 0.5, 0.3]])
    def test_choose_returns_beta_samples_argmax(
        self, mocker, primed_strategy, beta_samples
    ):
        mock_rng = mocker.Mock(spec=Generator, beta=lambda a, b: beta_samples)
        beta_spy = mocker.spy(mock_rng, "beta")
        choice = primed_strategy.choose(mock_rng)
        beta_spy.assert_called_once_with(a=primed_strategy._a, b=primed_strategy._b)
        assert beta_spy.spy_return[choice] == max(beta_spy.spy_return)

    def test_choose_without_rng_raises_error(self, primed_strategy):
        with pytest.raises(BanditUsageError):
            primed_strategy.choose()

    def test_update_increments_a_when_reward_is_1(
        self, mocker, primed_strategy, choice
    ):
        mock_rng = mocker.Mock(spec=Generator, binomial=lambda n, p: 1)
        prev_a, prev_b = primed_strategy._a[choice], primed_strategy._b[choice]
        primed_strategy.update(choice, 1, mock_rng)
        assert primed_strategy._a[choice] == prev_a + 1
        assert primed_strategy._b[choice] == prev_b

    def test_update_increments_b_when_reward_is_0(
        self, mocker, primed_strategy, choice
    ):
        mock_rng = mocker.Mock(spec=Generator, binomial=lambda n, p: 0)
        prev_a, prev_b = primed_strategy._a[choice], primed_strategy._b[choice]
        primed_strategy.update(choice, 0, mock_rng)
        assert primed_strategy._a[choice] == prev_a
        assert primed_strategy._b[choice] == prev_b + 1

    @pytest.mark.parametrize("invalid_reward", [-0.2, 1.3])
    def test_update_with_invalid_reward_for_general_strategy_raises_error(
        self, mocker, prime_params, choice, invalid_reward
    ):
        mock_rng = mocker.Mock(spec=Generator)
        strategy = self.STRATEGY_CLASS(general=True)
        strategy.prime(**prime_params)
        with pytest.raises(BanditUsageError):
            strategy.update(choice, invalid_reward, mock_rng)

    @pytest.mark.parametrize("invalid_reward", [-2, 0.4, 1.2])
    def test_update_with_invalid_reward_for_non_general_strategy_raises_error(
        self, mocker, prime_params, choice, invalid_reward
    ):
        mock_rng = mocker.Mock(spec=Generator)
        strategy = self.STRATEGY_CLASS(general=False)
        strategy.prime(**prime_params)
        with pytest.raises(BanditUsageError):
            strategy.update(choice, invalid_reward, mock_rng)

    def test_update_without_rng_raises_error(self, primed_strategy, choice, reward):
        with pytest.raises(BanditUsageError):
            primed_strategy.update(choice, reward)

    def test_Qs_returns_beta_mean(self, primed_strategy, a_b):
        a, b = np.array(a_b[0]), np.array(a_b[1])
        primed_strategy._a, primed_strategy._b = a, b
        np.testing.assert_array_equal(primed_strategy.Qs, a / (a + b))

    def test_Ns_returns_counts(self, primed_strategy, a_b):
        a, b = np.array(a_b[0]), np.array(a_b[1])
        primed_strategy._a, primed_strategy._b = a, b
        np.testing.assert_array_equal(primed_strategy.Ns, a + b - 2)
