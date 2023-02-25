import numpy as np
import pytest

from mabby import EpsilonGreedyBandit


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
