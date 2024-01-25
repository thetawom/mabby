<h1 align="center">
<img src="https://raw.githubusercontent.com/thetawom/mabby/main/assets/mabby-logo-title.png" width="500">
</h1>

[![PyPI](https://img.shields.io/pypi/v/mabby)](https://pypi.org/project/mabby/)
[![license](https://img.shields.io/github/license/thetawom/mabby)](https://github.com/thetawom/mabby/blob/main/LICENSE)
[![issues](https://img.shields.io/github/issues/thetawom/mabby)](https://github.com/thetawom/mabby/issues)
[![build](https://img.shields.io/github/actions/workflow/status/thetawom/mabby/build.yml)](https://github.com/thetawom/mabby/actions/workflows/build.yml)
[![docs](https://img.shields.io/github/actions/workflow/status/thetawom/mabby/docs.yml?label=docs)](https://thetawom.github.io/mabby/)
[![coverage](https://coveralls.io/repos/github/thetawom/mabby/badge.svg)](https://coveralls.io/github/thetawom/mabby)

**mabby** is a library for simulating [multi-armed bandits (MABs)](https://en.wikipedia.org/wiki/Multi-armed_bandit), a resource-allocation problem and framework in reinforcement learning. It allows users to quickly yet flexibly define and run bandit simulations, with the ability to:

- choose from a wide range of classic bandit algorithms to use
- configure environments with custom arm spaces and rewards distributions
- collect and visualize simulation metrics like regret and optimality

## Installation

Prerequisites: [Python 3.9+](https://www.python.org/downloads/) and `pip`

Install **mabby** with `pip`:

```bash
pip install mabby
```

## Basic Usage

The code example below demonstrates the basic steps of running a simulation with **mabby**. For more in-depth examples, please see the [Usage Examples](https://thetawom.github.io/mabby/examples/) section of the **mabby** documentation.

```python
import mabby as mb

# configure bandit arms
bandit = mb.BernoulliArm.bandit(p=[0.3, 0.6])

# configure bandit strategy
strategy = mb.strategies.EpsilonGreedyStrategy(eps=0.2)

# setup simulation
simulation = mb.Simulation(bandit=bandit, strategies=[strategy])

# run simulation
stats = simulation.run(trials=100, steps=300)

# plot regret statistics
stats.plot_regret()
```

## Contributing

Please see [CONTRIBUTING](https://thetawom.github.io/mabby/contributing/) for more information.

## License

This software is licensed under the Apache 2.0 license. Please see [LICENSE](https://thetawom.github.io/mabby/license/) for more information.
