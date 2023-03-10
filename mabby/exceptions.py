from mabby.core.exceptions import AgentUsageError, SimulationUsageError
from mabby.strategies.exceptions import StrategyUsageError

__all__ = [
    "StrategyUsageError",
    "AgentUsageError",
    "SimulationUsageError",
    "StatsUsageError",
]


class StatsUsageError(Exception):
    """Raised when stats methods are used incorrectly"""

    pass
