<h1 align="center">
<img src="https://raw.githubusercontent.com/ew2664/mabby/main/assets/mabby-logo-title.png" width="500">
</h1>

[![license](https://img.shields.io/github/license/ew2664/mabby)](https://github.com/ew2664/mabby/blob/main/LICENSE)
[![issues](https://img.shields.io/github/issues/ew2664/mabby)](https://github.com/ew2664/mabby/issues)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![poetry](https://img.shields.io/badge/packaging-poetry-008adf)](https://python-poetry.org/)
[![black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
[![mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

`mabby` is a library for simulating [multi-armed bandits (MABs)](https://en.wikipedia.org/wiki/Multi-armed_bandit), a resource-allocation problem and framework in reinforcement learning. It allows users to quickly yet flexibly define and run bandit simulations, with the ability to:

- choose from a wide range of preset classic bandit algorithms to use
- configure environments with custom arm spaces and rewards distributions
- collect, store, and visualize simulation statistics

## Example Usage

Below is an example of a simple simulation comparing two epsilon-greedy bandits with different exploration parameters.

```python
import mabby as mb

explore_bandit = mb.EpsilonGreedyBandit(eps=0.8)
exploit_bandit = mb.EpsilonGreedyBandit(eps=0.1)
sim = mb.Simulation(
    bandits=[explore_bandit, exploit_bandit],
    armset=mb.BernoulliArm.armset(p=[0.2, 0.6]),
)
stats = sim.run(trials=100, steps=200)
stats.plot_regret()

print("eps=0.8: ", explore_bandit.Qs)
print("eps=0.1: ", exploit_bandit.Qs)
```
