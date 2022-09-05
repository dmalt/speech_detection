import numpy as np
from hypothesis import given
from hypothesis.strategies import floats, integers
from numpy.testing import assert_almost_equal

from speech_detection import features


def test_energy_on_empty_input_returns_zero():
    assert features.compute_energy(np.array([])) == 0


@given(floats(min_value=-1e12, max_value=1e12))
def test_energy_on_single_value_returns_this_value_squared(val: float) -> None:
    assert features.compute_energy(np.array([val])) == val**2


@given(floats(min_value=-1e6, max_value=1e6), integers(min_value=1, max_value=1_000_000))
def test_energy_on_repeated_value_returns_this_value_squared(val: float, n: int) -> None:
    assert_almost_equal(features.compute_energy(np.array([val] * n)), val**2, decimal=0)


@given(integers(min_value=1, max_value=1_000_000))
def test_energy_on_arithmetic_progression_satisfies_closed_form_solution(n: int) -> None:
    assert_almost_equal(
        features.compute_energy(np.arange(1, n + 1)), (n + 1) * (2 * n + 1) / 6, decimal=3
    )


def test_centroid_on_empty_sequence_returns_zero():
    assert features._centroid(np.array([])) == 0


@given(floats(min_value=-1e6, max_value=1e6), integers(min_value=1, max_value=1_000_000))
def test_centroid_on_equal_values_returns_middle(val: float, n: int):
    res = features._centroid(np.array([val] * n))
    if val == 0:
        assert res == 0
    else:
        assert_almost_equal(res, (n + 1) / 2, decimal=3)
