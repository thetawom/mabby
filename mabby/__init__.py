"""A multi-armed bandit (MAB) sim library.

mabby is a library for simulating multi-armed bandits (MABs), a resource-allocation
problem and framework in reinforcement learning. It allows users to quickly yet flexibly
define and run bandit simulations, with the ability to:

* choose from a wide range of classic bandit algorithms to use
* configure environments with custom arm spaces and rewards distributions
* collect and visualize sim metrics like regret and optimality
"""
from mabby.agent import Agent
from mabby.arms import Arm, BernoulliArm, GaussianArm
from mabby.bandit import Bandit
from mabby.simulation import Simulation
from mabby.stats import Metric, SimulationStats

__all__ = [
    "Agent",
    "Arm",
    "BernoulliArm",
    "GaussianArm",
    "Bandit",
    "Simulation",
    "Metric",
    "SimulationStats",
]
