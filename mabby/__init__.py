"""A multi-armed bandit (MAB) simulation library.

mabby is a library for simulating multi-armed bandits (MABs), a resource-allocation
problem and framework in reinforcement learning. It allows users to quickly yet flexibly
define and run bandit simulations, with the ability to:

* choose from a wide range of classic bandit algorithms to use
* configure environments with custom arm spaces and rewards distributions
* collect and visualize simulation metrics like regret and optimality
"""
from mabby.simulation.agent import Agent
from mabby.simulation.arms import Arm, BernoulliArm, GaussianArm
from mabby.simulation.bandit import Bandit
from mabby.simulation.simulation import Simulation
from mabby.simulation.stats import Metric, SimulationStats

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
