from abc import ABC, abstractmethod


class Arm(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def play(self):
        pass

    @property
    @abstractmethod
    def mean(self):
        pass
