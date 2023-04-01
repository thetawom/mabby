"""Multi-armed bandit strategies.

**mabby** provides a collection of preset bandit strategies that can be plugged into
simulations. The :class:`strategy.Strategy` abstract base class can also be
sub-classed to implement custom bandit strategies.
"""

from mabby.strategies.semi_uniform import (
    EpsilonGreedyStrategy,
    RandomStrategy,
    SemiUniformStrategy,
)
from mabby.strategies.strategy import Strategy
from mabby.strategies.ts import BetaTSStrategy
from mabby.strategies.ucb import UCB1Strategy

__all__ = [
    "Strategy",
    "SemiUniformStrategy",
    "RandomStrategy",
    "EpsilonGreedyStrategy",
    "BetaTSStrategy",
    "UCB1Strategy",
]
