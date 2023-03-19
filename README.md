<h1 align="center">
<img src="https://raw.githubusercontent.com/ew2664/mabby/main/assets/mabby-logo-title.png" width="500">
</h1>

[![license](https://img.shields.io/github/license/ew2664/mabby)](https://github.com/ew2664/mabby/blob/main/LICENSE)
[![issues](https://img.shields.io/github/issues/ew2664/mabby)](https://github.com/ew2664/mabby/issues)
[![build](https://img.shields.io/github/actions/workflow/status/ew2664/mabby/ci.yml)](https://github.com/ew2664/mabby/actions/workflows/ci.yml)
[![coverage](https://coveralls.io/repos/github/ew2664/mabby/badge.svg)](https://coveralls.io/github/ew2664/mabby)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
[![poetry](https://img.shields.io/badge/packaging-poetry-008adf)](https://python-poetry.org/)
[![black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
[![mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

`mabby` is a library for simulating [multi-armed bandits (MABs)](https://en.wikipedia.org/wiki/Multi-armed_bandit), a resource-allocation problem and framework in reinforcement learning. It allows users to quickly yet flexibly define and run bandit simulations, with the ability to:

- choose from a wide range of preset classic bandit algorithms to use
- configure environments with custom arm spaces and rewards distributions
- collect, store, and visualize simulation statistics

## Example Usage

Below is an example of a four-armed Bernoulli bandit simulation using epsilon-greedy, UCB1, and Thompson sampling strategies.

```python
from mabby import Simulation, BernoulliArm
from mabby.strategies import EpsilonGreedyStrategy, UCB1Strategy, BetaTSStrategy

simulation = Simulation(
    bandit=BernoulliArm.bandit(p=[0.5, 0.6, 0.7, 0.8]),
    strategies=[
        EpsilonGreedyStrategy(eps=0.2),
        UCB1Strategy(alpha=0.5),
        BetaTSStrategy(general=True),
    ],
)
stats = simulation.run(trials=100, steps=300)

stats.plot_regret(cumulative=True)
stats.plot_optimality()
stats.plot_rewards(cumulative=True)
```
