class AgentUsageError(Exception):
    """Raised when agent methods are used incorrectly"""

    pass


class SimulationUsageError(Exception):
    """Raised when simulation methods are used incorrectly"""

    pass


class StatsUsageError(Exception):
    """Raised when stats methods are used incorrectly"""

    pass
