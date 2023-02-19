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

    @classmethod
    def armset(cls, **kwargs):
        params_dicts = [dict(zip(kwargs, t)) for t in zip(*kwargs.values())]
        return ArmSet([cls(**params) for params in params_dicts])


class ArmSet:
    def __init__(self, arms):
        self._arms = arms

    def __len__(self):
        return len(self._arms)

    def __repr__(self):
        return repr(self._arms)

    def __getitem__(self, i):
        return self._arms[i]

    def __iter__(self):
        return iter(self._arms)

    def play(self, i):
        return self[i].play()
