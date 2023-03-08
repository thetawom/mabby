from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike


def random_argmax(values: ArrayLike, rng: Generator) -> int:
    candidates = np.where(values == np.max(values))[0]
    return int(rng.choice(candidates))
