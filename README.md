<h1 align="center">
<img src="https://raw.githubusercontent.com/ew2664/mabby/main/assets/mabby-logo-title.png" width="500">
</h1>

[![PyPI](https://img.shields.io/pypi/v/mabby)](https://pypi.org/project/mabby/)
[![license](https://img.shields.io/github/license/ew2664/mabby)](https://github.com/ew2664/mabby/blob/main/LICENSE)
[![issues](https://img.shields.io/github/issues/ew2664/mabby)](https://github.com/ew2664/mabby/issues)
[![build](https://img.shields.io/github/actions/workflow/status/ew2664/mabby/build.yml)](https://github.com/ew2664/mabby/actions/workflows/build.yml)
[![docs](https://img.shields.io/github/actions/workflow/status/ew2664/mabby/docs.yml?label=docs)](https://ew2664.github.io/mabby/)
[![coverage](https://coveralls.io/repos/github/ew2664/mabby/badge.svg)](https://coveralls.io/github/ew2664/mabby)

**mabby** is a library for simulating [multi-armed bandits (MABs)](https://en.wikipedia.org/wiki/Multi-armed_bandit), a resource-allocation problem and framework in reinforcement learning. It allows users to quickly yet flexibly define and run bandit simulations, with the ability to:

- choose from a wide range of classic bandit algorithms to use
- configure environments with custom arm spaces and rewards distributions
- collect and visualize simulation metrics like regret and optimality

## Installation

Prerequisites: Python 3.9+ and `pip`

Install **mabby** with `pip`:

```bash
pip install mabby
```

## Basic Usage

First, configure the arms and rewards of the bandit problem to simulate. For example, the following creates a two-armed Bernoulli bandit.

```python
bandit = Bandit(arms=[BernoulliArm(p=0.3), BernoulliArm(p=0.6)])
```

If all arms of the bandit have rewards distributions of the same type, then the following shorthand can also be used.

```python
bandit = BernoulliArm.bandit(p=[0.3, 0.6])
```

Next, specify the bandit strategy to simulate by creating a `Strategy` with desired parameter values. For example, the following sets up an epsilon-greedy strategy with `eps=0.2`.

```python
strategy = EpsilonGreedyStrategy(eps=0.2)
```

Run a simulation with the bandit and bandit strategy for a specified number of trials and steps per trial. When creating the `Simulation`, multiple strategies can be specified to be run together for comparison.

```python
simulation = Simulation(bandit=bandit, strategies=[strategy])
stats = simulation.run(trials=100, steps=300)
```

Running the simulation outputs a `SimulationStats` object that can be used to visualize the tracked metrics.

```python
stats.plot_regret(cumulative=True)
stats.plot_optimality()
```

## Examples

- [**Bernoulli Bandits**](docs/examples/bernoulli_bandit.py): a four-armed Bernoulli bandit simulation comparing epsilon-greedy, UCB1, and Thompson sampling strategies

## Contributing

Please see [CONTRIBUTING](https://ew2664.github.io/mabby/contributing/) for more information.

## License

This software is licensed under the Apache 2.0 license. Please see [LICENSE](https://ew2664.github.io/mabby/license/) for more information.
