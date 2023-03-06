class BanditUsageError(Exception):
    """Raised when bandit functions are called out of order"""

    pass


class SimulationUsageError(Exception):
    """Raised when simulation functions are called out of order"""

    pass


class StatsUsageError(Exception):
    """Raised when stats functions are called improperly"""

    pass
