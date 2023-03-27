<h1 align="center">
<img src="https://raw.githubusercontent.com/ew2664/mabby/main/assets/mabby-logo-title.png" width="500">
</h1>

[![license](https://img.shields.io/github/license/ew2664/mabby)](https://github.com/ew2664/mabby/blob/main/LICENSE)
[![issues](https://img.shields.io/github/issues/ew2664/mabby)](https://github.com/ew2664/mabby/issues)
[![build](https://img.shields.io/github/actions/workflow/status/ew2664/mabby/ci.yml)](https://github.com/ew2664/mabby/actions/workflows/ci.yml)
[![coverage](https://coveralls.io/repos/github/ew2664/mabby/badge.svg)](https://coveralls.io/github/ew2664/mabby)
[![PyPI](https://img.shields.io/pypi/v/mabby)](https://pypi.org/project/mabby/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
[![poetry](https://img.shields.io/badge/packaging-poetry-008adf)](https://python-poetry.org/)
[![black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
[![mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

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

- [**Bernoulli Bandits**](./examples/bernoulli_bandit.py): a four-armed Bernoulli bandit simulation comparing epsilon-greedy, UCB1, and Thompson sampling strategies

## Contributing

Please see [CONTRIBUTING](CONTRIBUTING.md) for more information.

## License

This software is licensed under the Apache 2.0 license. Please see [LICENSE](LICENSE) for more information.
