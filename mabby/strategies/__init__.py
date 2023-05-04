"""Multi-armed bandit strategies.

**mabby** provides a collection of preset bandit strategies that can be plugged into
simulations. The [`Strategy`][mabby.strategies.strategy.Strategy] abstract base class
can also be sub-classed to implement custom bandit strategies.
"""

from mabby.strategies.semi_uniform import (
    EpsilonFirstStrategy,
    EpsilonGreedyStrategy,
    RandomStrategy,
    SemiUniformStrategy,
)
from mabby.strategies.strategy import Strategy
from mabby.strategies.thompson import BetaTSStrategy
from mabby.strategies.ucb import UCB1Strategy

__all__ = [
    "Strategy",
    "SemiUniformStrategy",
    "RandomStrategy",
    "EpsilonGreedyStrategy",
    "EpsilonFirstStrategy",
    "BetaTSStrategy",
    "UCB1Strategy",
]
