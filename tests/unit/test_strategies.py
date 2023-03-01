import numpy as np
import pytest

from mabby.strategies import (
    EpsilonGreedyStrategy,
    RandomStrategy,
    SemiUniformStrategy,
    Strategy,
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


class TestRandomBandit(TestSemiUniformStrategy):
    STRATEGY_CLASS = RandomStrategy

    @pytest.fixture(params=[{}])
    def valid_params(self, request):
        return request.param

    def test_repr_equals_random(self, strategy):
        assert repr(strategy) == "random"

    def test_effective_eps_equals_1(self, strategy):
        assert strategy.effective_eps() == 1


class TestEpsilonGreedyBandit(TestSemiUniformStrategy):
    STRATEGY_CLASS = EpsilonGreedyStrategy

    @pytest.fixture(params=[{"eps": 0.1}])
    def valid_params(self, request):
        return request.param

    @pytest.fixture(params=[{"eps": -1}, {"eps": 1.2}])
    def invalid_params(self, request):
        return request.param

    def test_repr_includes_eps(self, valid_params, strategy):
        assert str(valid_params["eps"]) in repr(strategy)

    def test_effective_eps_equals_eps(self, valid_params, strategy):
        assert strategy.effective_eps() == valid_params["eps"]
