from abc import ABC, abstractmethod

import numpy as np


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

    def best_arm(self):
        return np.argmax([arm.mean for arm in self._arms])


class BernoulliArm(Arm):
    def __init__(self, p):
        self.p = p

    def play(self):
        return np.random.binomial(1, self.p)

    @property
    def mean(self):
        return self.p

    def __repr__(self):
        return f"Bernoulli(p={self.p})"


class GaussianArm(Arm):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def play(self):
        return np.random.normal(self.loc, self.scale)

    @property
    def mean(self):
        return self.loc

    def __repr__(self):
        return f"Gaussian(loc={self.loc}, scale={self.scale})"
