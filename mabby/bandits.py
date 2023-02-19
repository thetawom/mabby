from abc import ABC, abstractmethod

from mabby.exceptions import BanditUsageError


class Bandit(ABC):
    def __init__(self):
        self._primed = False
        self._choice = None

    def prime(self, k, rounds):
        self._primed = True
        self._choice = None
        self._prime(k, rounds)

    def choose(self):
        if not self._primed:
            raise BanditUsageError("choose() can only be called on a primed bandit")
        self._choice = self._choose()
        return self._choice

    def update(self, reward):
        if self._choice is None:
            raise BanditUsageError("update() can only be called after choose()")
        self._update(self._choice, reward)
        self._choice = None

    @abstractmethod
    def _prime(self, k, rounds):
        pass

    @abstractmethod
    def _choose(self):
        pass

    @abstractmethod
    def _update(self, choice, reward):
        pass
