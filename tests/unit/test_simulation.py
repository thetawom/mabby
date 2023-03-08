import random

import pytest
from numpy.random import Generator

from mabby import Simulation
from mabby.bandit import Bandit
from mabby.exceptions import SimulationUsageError
from mabby.stats import AgentStats, SimulationStats


@pytest.fixture(params=[2])
def num_agents(request):
    return request.param


@pytest.fixture(params=[3])
def num_arms(request):
    return request.param


@pytest.fixture()
def agents(num_agents, agent_factory):
    return [agent_factory.generic() for _ in range(num_agents)]


@pytest.fixture()
def agent(agents):
    return random.choice(agents)


@pytest.fixture()
def strategies(num_agents, strategy_factory):
    return [strategy_factory.generic() for _ in range(num_agents)]


@pytest.fixture()
def bandit(num_arms, arm_factory):
    arms = [arm_factory.generic() for _ in range(num_arms)]
    return Bandit(arms=arms)


@pytest.fixture()
def simulation(agents, bandit):
    return Simulation(agents=agents, bandit=bandit)


@pytest.fixture(params=[{"trials": 3, "steps": 2}])
def run_params(request):
    return request.param


class TestSimulation:
    def test_init_sets_agents_bandit_and_rng(self, agents, bandit, simulation):
        assert simulation.agents == agents
        assert simulation.bandit == bandit
        assert isinstance(simulation._rng, Generator)

    def test_init_with_empty_bandit_raises_error(self, agents):
        empty_bandit = Bandit(arms=[])
        with pytest.raises(ValueError):
            Simulation(agents=agents, bandit=empty_bandit)

    def test_init_with_no_agents_or_strategies_raises_error(self, bandit):
        with pytest.raises(SimulationUsageError):
            Simulation(bandit=bandit)

    def test_init_with_empty_agents_raises_error(self, bandit):
        with pytest.raises(ValueError):
            Simulation(bandit=bandit, agents=[])

    def test_init_with_strategies_sets_agents(self, bandit, strategies):
        simulation = Simulation(bandit=bandit, strategies=strategies)
        assert len(simulation.agents) == len(strategies)
        for i, agent in enumerate(simulation.agents):
            assert agent.strategy == strategies[i]

    @pytest.mark.parametrize("names", [["a", "b", "c"], [], ["a", "b", "c", "d", "e"]])
    def test_init_with_strategies_and_names_sets_agents(
        self, bandit, strategies, names
    ):
        simulation = Simulation(bandit=bandit, strategies=strategies, names=names)
        assert len(simulation.agents) == len(strategies)
        for i, agent in enumerate(simulation.agents):
            assert agent.strategy == strategies[i]
            assert i >= len(names) or agent._name == names[i]

    def test_run_returns_sim_stats_with_agent_stats(
        self, mocker, agents, bandit, simulation, run_params
    ):
        mocker.patch.object(
            simulation,
            "_run_trials_for_agent",
            lambda b, _, steps, metrics: AgentStats(b, bandit, steps, metrics),
        )
        run_trials_for_agent_spy = mocker.spy(simulation, "_run_trials_for_agent")
        sim_stats = simulation.run(**run_params)
        assert isinstance(sim_stats, SimulationStats)
        for agent in agents:
            assert agent in sim_stats
        assert run_trials_for_agent_spy.call_count == len(agents)

    def test__run_trials_for_agent_returns_agent_stats(
        self, agent, simulation, run_params
    ):
        agent_stats = simulation._run_trials_for_agent(agent, **run_params)
        assert isinstance(agent_stats, AgentStats)

    def test__run_trials_for_agent_primes_agent_each_trial(
        self, mocker, agent, simulation, run_params
    ):
        prime_spy = mocker.spy(agent, "prime")
        simulation._run_trials_for_agent(agent, **run_params)
        assert prime_spy.call_count == run_params["trials"]

    def test__run_trials_for_agent_chooses_plays_updates_each_step(
        self, mocker, agent, bandit, simulation, run_params
    ):
        agent_choose_spy = mocker.spy(agent, "choose")
        bandit_play_spy = mocker.spy(bandit, "play")
        agent_update_spy = mocker.spy(agent, "update")
        agent_stats_update_spy = mocker.spy(AgentStats, "update")
        simulation._run_trials_for_agent(agent, **run_params)
        total_count = run_params["trials"] * run_params["steps"]
        assert agent_choose_spy.call_count == total_count
        assert bandit_play_spy.call_count == total_count
        assert agent_update_spy.call_count == total_count
        assert agent_stats_update_spy.call_count == total_count
