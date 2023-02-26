import numpy as np
import pytest

from mabby.bandits import Bandit, EpsilonGreedyBandit
from mabby.exceptions import BanditUsageError


@pytest.fixture
def mock_rng(mocker):
    return mocker.Mock(random=lambda: 0.5)


class TestBandit:
    BANDIT_CLASS = Bandit

    @pytest.fixture(autouse=True)
    def patch_abstract_methods(self, mocker):
        mocker.patch.object(Bandit, "__abstractmethods__", new_callable=set)

    @pytest.fixture(params=[{}, {"name": "bandit-name"}])
    def valid_params(self, request):
        return request.param

    @pytest.fixture(params=[])
    def invalid_params(self, request):
        return request.param

    @pytest.fixture
    def bandit(self, valid_params):
        return self.BANDIT_CLASS(**valid_params)

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
    def primed_bandit(self, bandit, prime_params):
        bandit.prime(k=prime_params["k"], steps=prime_params["steps"])
        return bandit

    @pytest.fixture
    def chosen_bandit(self, mocker, primed_bandit, mock_rng, choice):
        mocker.patch.object(primed_bandit, "_choose", return_value=choice)
        primed_bandit.choose(mock_rng)
        return primed_bandit

    def test_init_sets_name(self, valid_params, bandit):
        assert bandit._name == valid_params.get("name")

    def test_init_raises_error_with_invalid_params(self, invalid_params):
        with pytest.raises(ValueError):
            self.BANDIT_CLASS(**invalid_params)

    @pytest.mark.parametrize("name", ["name"])
    def test_name_returns_custom_name_if_overridden(self, valid_params, name):
        valid_params["name"] = name
        bandit = self.BANDIT_CLASS(**valid_params)
        assert bandit.name == name

    @pytest.mark.parametrize("default_name", ["default"])
    def test_name_returns_default_name_if_not_overridden(
        self, mocker, valid_params, default_name
    ):
        mocker.patch.object(
            self.BANDIT_CLASS, "default_name", return_value=default_name
        )
        valid_params["name"] = None
        bandit = self.BANDIT_CLASS(**valid_params)
        assert bandit.name == default_name

    def test_prime_invokes__prime(self, mocker, bandit, prime_params):
        spy = mocker.spy(bandit, "_prime")
        bandit.prime(**prime_params)
        assert spy.call_count == 1

    def test_choose_invokes__choose(self, mocker, primed_bandit, mock_rng):
        spy = mocker.spy(primed_bandit, "_choose")
        primed_bandit.choose(mock_rng)
        assert spy.call_count == 1

    def test_choose_saves_and_returns_choice(
        self, mocker, primed_bandit, mock_rng, choice
    ):
        mocker.patch.object(primed_bandit, "_choose", return_value=choice)
        assert primed_bandit.choose(mock_rng) == choice
        assert primed_bandit._choice == choice

    def test_update_invokes__update(self, mocker, chosen_bandit, reward):
        spy = mocker.spy(chosen_bandit, "_update")
        chosen_bandit.update(reward)
        assert spy.call_count == 1

    def test_update_resets_choice_to_none(self, chosen_bandit, reward):
        chosen_bandit.update(reward)
        assert chosen_bandit._choice is None

    @pytest.mark.parametrize("Qs", [[1, 2]])
    def test_Qs_returns_computed_Qs(self, mocker, primed_bandit, Qs):
        mocker.patch.object(primed_bandit, "compute_Qs", return_value=Qs)
        assert primed_bandit.Qs == Qs

    def test_choose_before_prime_raises_error(self, bandit, mock_rng):
        with pytest.raises(BanditUsageError):
            bandit.choose(rng=mock_rng)

    def test_Qs_before_prime_raises_error(self, bandit):
        with pytest.raises(BanditUsageError):
            assert bandit.Qs is not None

    def test_update_before_choose_raises_error(self, primed_bandit, reward):
        with pytest.raises(BanditUsageError):
            primed_bandit.update(reward=reward)


class TestEpsilonGreedyBandit(TestBandit):
    BANDIT_CLASS = EpsilonGreedyBandit

    @pytest.fixture(params=[{"eps": 0.1}, {"eps": 0.3, "name": "bandit-name"}])
    def valid_params(self, request):
        return request.param

    @pytest.fixture(params=[{"eps": -1}, {"eps": 1.2}])
    def invalid_params(self, request):
        return request.param

    @pytest.fixture(params=[[0.1, 0.5, 0.2], [2, 0]])
    def Qs(self, request):
        return request.param

    def test_default_name_contains_eps(self, valid_params, bandit):
        eps = valid_params["eps"]
        assert str(eps) in bandit.default_name()

    def test__prime_inits__Qs_and__Ns(self, prime_params, primed_bandit):
        k = prime_params["k"]
        assert isinstance(primed_bandit._Qs, np.ndarray)
        assert isinstance(primed_bandit._Ns, np.ndarray)
        assert len(primed_bandit._Qs) == k
        assert len(primed_bandit._Ns) == k
        assert not primed_bandit._Qs.any()
        assert not primed_bandit._Ns.any()

    def test__choose_explores_with_low_rng(
        self, mocker, valid_params, prime_params, primed_bandit
    ):
        mock_rng = mocker.Mock(random=lambda: 0.9 * valid_params["eps"])
        primed_bandit._choose(mock_rng)
        mock_rng.integers.assert_called_once_with(0, prime_params["k"])

    def test__choose_exploits_with_high_rng(
        self, mocker, valid_params, primed_bandit, Qs
    ):
        mock_rng = mocker.Mock(random=lambda: 1.1 * valid_params["eps"])
        primed_bandit._Qs = Qs
        choice = primed_bandit._choose(mock_rng)
        assert Qs[choice] == max(Qs)

    def test__update_updates__Qs_and__Ns(
        self, choice, prime_params, chosen_bandit, reward
    ):
        chosen_bandit._Ns = np.ones(prime_params["k"])
        chosen_bandit._update(choice, reward)
        assert chosen_bandit._Qs[choice] == reward / 2
        assert chosen_bandit._Ns[choice] == 2

    def test_compute_Qs_returns_Qs(self, primed_bandit, Qs):
        primed_bandit._Qs = Qs
        assert primed_bandit.compute_Qs() == primed_bandit._Qs
