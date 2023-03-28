"""Provides commonly used utility functions."""
from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike


def random_argmax(values: ArrayLike, rng: Generator) -> int:
    """Computes random argmax of an array.

    If there are multiple maximums, the index of one is chosen at random.

    Args:
        values: An input array.
        rng: A random number generator.

    Returns:
        The random argmax of the input array.
    """
    candidates = np.where(values == np.max(values))[0]
    return int(rng.choice(candidates))
