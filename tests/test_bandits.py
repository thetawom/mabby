import numpy as np
import pytest

from mabby.bandits import Bandit, EpsilonGreedyBandit
from mabby.exceptions import BanditUsageError


@pytest.fixture
def rng(mocker):
    return mocker.MagicMock()


@pytest.fixture
def rng_low(mocker):
    return mocker.MagicMock(random=lambda: 0.1)


@pytest.fixture
def rng_high(mocker):
    return mocker.MagicMock(random=lambda: 0.9)


class TestBandit:
    BANDIT_CLASS = Bandit

    @pytest.fixture(autouse=True)
    def patch_abstract_methods(self, mocker):
        mocker.patch.object(Bandit, "__abstractmethods__", new_callable=set)

    @pytest.fixture(params=[{"name": None}, {"name": "bandit-name"}])
    def valid_params(self, request):
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
    def chosen_bandit(self, mocker, primed_bandit, rng, choice):
        mocker.patch.object(primed_bandit, "_choose", return_value=choice)
        primed_bandit.choose(rng)
        return primed_bandit

    def test_init_sets_name(self, valid_params, bandit):
        assert bandit._name == valid_params["name"]

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

    def test_choose_invokes__choose(self, mocker, primed_bandit, rng):
        spy = mocker.spy(primed_bandit, "_choose")
        primed_bandit.choose(rng)
        assert spy.call_count == 1

    def test_choose_saves_and_returns_choice(self, mocker, primed_bandit, rng, choice):
        mocker.patch.object(primed_bandit, "_choose", return_value=choice)
        assert primed_bandit.choose(rng) == choice
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

    def test_choose_before_prime_raises_error(self, bandit, rng):
        with pytest.raises(BanditUsageError):
            bandit.choose(rng=rng)

    def test_Qs_before_prime_raises_error(self, bandit):
        with pytest.raises(BanditUsageError):
            assert bandit.Qs is not None

    def test_update_before_choose_raises_error(self, primed_bandit, reward):
        with pytest.raises(BanditUsageError):
            primed_bandit.update(reward=reward)


@pytest.mark.parametrize("eps", [0.5])
class TestEpsilonGreedyBandit:
    @pytest.fixture
    def rng_low(self, mocker):
        return mocker.MagicMock(random=lambda: 0.1)

    @pytest.fixture
    def rng_high(self, mocker):
        return mocker.MagicMock(random=lambda: 0.9)

    def test_init_sets_eps_and_uses_default_name(self, eps):
        bandit = EpsilonGreedyBandit(eps=eps)
        assert bandit.eps == eps
        assert str(eps) in bandit.name

    @pytest.mark.parametrize("invalid_eps", [-0.5, 1.2])
    def test_init_with_invalid_eps_raises_error(self, invalid_eps, eps):
        with pytest.raises(ValueError):
            EpsilonGreedyBandit(eps=invalid_eps)

    @pytest.mark.parametrize("name", ["bandit-name"])
    def test_init_with_name_uses_custom_name(self, eps, name):
        bandit = EpsilonGreedyBandit(eps=eps, name=name)
        assert bandit.eps == eps
        assert bandit.name == name

    def test_default_name_contains_eps(self, eps):
        bandit = EpsilonGreedyBandit(eps=eps)
        default_name = bandit.default_name()
        assert str(eps) in default_name

    @pytest.mark.parametrize("k", [3])
    @pytest.mark.parametrize("steps", [10])
    def test_prime_inits_Qs_and_Ns(self, eps, k, steps):
        bandit = EpsilonGreedyBandit(eps=eps)
        bandit._prime(k, steps)
        assert isinstance(bandit._Qs, np.ndarray)
        assert isinstance(bandit._Ns, np.ndarray)
        assert len(bandit._Qs) == k
        assert len(bandit._Ns) == k
        assert not bandit._Qs.any()
        assert not bandit._Ns.any()

    @pytest.mark.parametrize("k", [3])
    def test_choose_with_low_rng_explores(self, eps, k, rng_low):
        bandit = EpsilonGreedyBandit(eps=eps)
        bandit._prime(k, 10)
        bandit._choose(rng_low)
        rng_low.integers.assert_called_once_with(0, k)

    @pytest.mark.parametrize("Qs", [[0.1, 0.5, 0.2], [2, 0]])
    def test_choose_with_high_rng_exploits(self, eps, Qs, rng_high):
        bandit = EpsilonGreedyBandit(eps=eps)
        bandit._Qs = Qs
        choice = bandit._choose(rng_high)
        assert Qs[choice] == max(Qs)

    @pytest.mark.parametrize("k", [3])
    @pytest.mark.parametrize("choice", [0, 1])
    @pytest.mark.parametrize("reward", [0.5])
    def test_update_updates_Ns_and_Qs(self, eps, k, choice, reward):
        bandit = EpsilonGreedyBandit(eps=eps)
        bandit._prime(k, 10)
        bandit._Ns = np.ones(k)
        bandit._update(choice, reward)
        assert bandit._Ns[choice] == 2
        assert bandit._Qs[choice] == reward / 2

    @pytest.mark.parametrize("Qs", [[0.1, 0.5, 0.2]])
    def test_compute_Qs_returns_Qs(self, eps, Qs):
        bandit = EpsilonGreedyBandit(eps=eps)
        bandit._Qs = Qs
        ret_Qs = bandit.compute_Qs()
        assert ret_Qs == Qs
