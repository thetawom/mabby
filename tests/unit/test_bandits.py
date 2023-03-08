from unittest.mock import patch

import pytest
from numpy.random import Generator

from mabby.agents import (
    Agent,
    BetaTSAgent,
    EpsilonGreedyAgent,
    RandomAgent,
    UCB1Agent,
)
from mabby.exceptions import AgentUsageError
from mabby.strategies import (
    BetaTSStrategy,
    EpsilonGreedyStrategy,
    RandomStrategy,
    UCB1Strategy,
)


class TestAgent:
    AGENT_CLASS = Agent

    @pytest.fixture(params=[{}, {"name": "agent-name"}])
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
    def agent(self, valid_params):
        return self.AGENT_CLASS(**valid_params)

    @pytest.fixture
    def primed_agent(self, prime_params, mock_rng, agent):
        agent.prime(**prime_params, rng=mock_rng)
        return agent

    @pytest.fixture
    def chosen_agent(self, choice, primed_agent):
        primed_agent._choice = choice
        return primed_agent

    def test_init_sets_name(self, valid_params, agent):
        assert agent._name == valid_params.get("name")

    @pytest.mark.parametrize("name", ["agent-name"])
    def test_repr_returns_custom_name_if_overridden(self, valid_params, name):
        valid_params["name"] = name
        agent = self.AGENT_CLASS(**valid_params)
        assert repr(agent) == name

    def test_repr_returns_strategy_name_if_not_overridden(self, valid_params):
        valid_params["name"] = None
        agent = self.AGENT_CLASS(**valid_params)
        assert repr(agent) == str(agent.strategy)

    def test_prime_invokes_strategy_prime(self, mocker, mock_rng, agent, prime_params):
        strategy_prime = mocker.spy(agent.strategy, "prime")
        agent.prime(**prime_params, rng=mock_rng)
        strategy_prime.assert_called_once_with(**prime_params)

    def test_choose_invokes_strategy_choose(
        self, mocker, mock_rng, primed_agent, choice
    ):
        mocker.patch.object(primed_agent.strategy, "choose", return_value=choice)
        strategy_choose = mocker.spy(primed_agent.strategy, "choose")
        primed_agent.choose()
        strategy_choose.assert_called_once_with(mock_rng)

    def test_choose_saves_and_returns_choice(self, mocker, primed_agent, choice):
        mocker.patch.object(primed_agent.strategy, "choose", return_value=choice)
        assert primed_agent.choose() == choice
        assert primed_agent._choice == choice

    def test_update_invokes_strategy_update(
        self, mock_rng, chosen_agent, choice, reward
    ):
        with patch.object(chosen_agent.strategy, "update") as strategy_update:
            chosen_agent.update(reward)
            strategy_update.assert_called_once_with(choice, reward, mock_rng)

    def test_update_resets_choice_to_none(self, mocker, chosen_agent, reward):
        mocker.patch.object(chosen_agent.strategy, "update")
        chosen_agent.update(reward)
        assert chosen_agent._choice is None

    def test_Qs_returns_strategy_Qs(self, primed_agent):
        assert (primed_agent.Qs == primed_agent.strategy.Qs).all()

    def test_Ns_returns_strategy_Ns(self, primed_agent):
        assert (primed_agent.Ns == primed_agent.strategy.Ns).all()

    def test_choose_before_prime_raises_error(self, agent):
        with pytest.raises(AgentUsageError):
            agent.choose()

    def test_Qs_before_prime_raises_error(self, agent):
        with pytest.raises(AgentUsageError):
            assert agent.Qs is not None

    def test_Ns_before_prime_raises_error(self, agent):
        with pytest.raises(AgentUsageError):
            assert agent.Ns is not None

    def test_update_before_choose_raises_error(self, primed_agent, reward):
        with pytest.raises(AgentUsageError):
            primed_agent.update(reward=reward)


class TestRandomAgent(TestAgent):
    AGENT_CLASS = RandomAgent

    @pytest.fixture(params=[{}, {"name": "agent-name"}])
    def valid_params(self, request):
        return request.param

    def test_init_sets_random_strategy(self, agent):
        assert isinstance(agent.strategy, RandomStrategy)


class TestEpsilonGreedyAgent(TestAgent):
    AGENT_CLASS = EpsilonGreedyAgent

    @pytest.fixture(params=[{"eps": 0.2}, {"eps": 0.5, "name": "agent-name"}])
    def valid_params(self, request):
        return request.param

    def test_init_sets_epsilon_greedy_strategy(self, agent, valid_params):
        assert isinstance(agent.strategy, EpsilonGreedyStrategy)
        assert agent.strategy.eps == valid_params["eps"]


class TestUCB1Agent(TestAgent):
    AGENT_CLASS = UCB1Agent

    @pytest.fixture(params=[{"alpha": 2}, {"alpha": 0.5, "name": "agent-name"}])
    def valid_params(self, request):
        return request.param

    def test_init_sets_ucb1_strategy(self, agent, valid_params):
        assert isinstance(agent.strategy, UCB1Strategy)
        assert agent.strategy.alpha == valid_params["alpha"]


class TestBetaTSAgent(TestAgent):
    AGENT_CLASS = BetaTSAgent

    @pytest.fixture(
        params=[{"general": False}, {"general": True, "name": "agent-name"}]
    )
    def valid_params(self, request):
        return request.param

    def test_init_sets_beta_ts_strategy(self, agent, valid_params):
        assert isinstance(agent.strategy, BetaTSStrategy)
        assert agent.strategy.general == valid_params["general"]
