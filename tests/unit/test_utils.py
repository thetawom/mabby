import numpy as np
import pytest

from mabby.utils import random_argmax


@pytest.fixture()
def rng():
    return np.random.default_rng(83271)


@pytest.mark.parametrize("values", [[3, 10, -2, 10]])
def test_random_argmax_returns_argmax(values, rng):
    assert values[random_argmax(values, rng=rng)] == max(values)


@pytest.mark.parametrize("values", [[3, 10, -2, 10]])
def test_random_argmax_with_rng_produces_even_distribution(values, rng):
    all_argmax = np.where(values == np.max(values))[0]
    argmax_samples = [random_argmax(values, rng=rng) for _ in range(100)]
    values, counts = np.unique(argmax_samples, return_counts=True)
    assert len(values) == len(all_argmax)
    assert np.allclose(counts, np.mean(counts), rtol=0.05)
