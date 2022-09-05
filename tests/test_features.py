import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers
from numpy.testing import assert_almost_equal

from speech_detection import features


def test_energy_on_empty_input_returns_zero() -> None:
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


def test_centroid_on_empty_sequence_returns_zero() -> None:
    assert features._centroid(np.array([])) == 0


@given(floats(min_value=1e-6, max_value=1e6), integers(min_value=1, max_value=1_000_000))
def test_centroid_on_equal_values_returns_middle(val: float, n: int) -> None:
    res = features._centroid(np.array([val] * n))
    if val == 0:
        assert res == 0
    else:
        assert_almost_equal(res, (n - 1) / 2, decimal=3)


@given(floats(min_value=1e-12, max_value=1e12), integers(min_value=1, max_value=100_000))
def test_centroid_on_triangular_input_returns_peak_index(peak_val: float, n_over_2: int) -> None:
    test_array = np.concatenate(
        [
            np.linspace(0, peak_val, num=n_over_2, endpoint=False),
            np.linspace(peak_val, 0, n_over_2 + 1),
        ]
    )
    assert_almost_equal(features._centroid(test_array), n_over_2, decimal=3)


def test_centroid_on_negative_inputs_raises_bad_array_exception() -> None:
    with pytest.raises(features.BadArrayException):
        features._centroid(np.array([-1, -1]))


@given(arrays(float, 100, elements=floats(-1e12, 1e12)))
def test_amplitude_spectrum_returns_nonnegative_values(x: np.ndarray) -> None:
    assert np.all(features._amplitude_spectrum(x, 10) >= 0)


@given(arrays(float, 100, elements=floats(-1e12, 1e12)), floats(0, 1e3))
def test_amplitude_spectrum_is_normalized(x: np.ndarray, threshold: float) -> None:
    res = features._amplitude_spectrum(x, threshold)
    if np.any(res):
        assert np.max(res) == 1


@given(arrays(float, 100, elements=floats(-1e3, 1e3)))
def test_amplitude_spectrum_returns_zeros_for_high_threshold(x: np.ndarray) -> None:
    threshold = 1e8
    res = features._amplitude_spectrum(x, threshold)
    assert np.all(res == 0)


def test_spread_on_empty_input_returns_zero() -> None:
    assert features._spread(np.array([])) == 0


def test_spread_on_negative_inputs_raises_bad_array_exception() -> None:
    with pytest.raises(features.BadArrayException):
        features._spread(np.array([-1, -1]))


@given(floats(1e-6, 1e6), integers(1, 1000))
def test_spread_of_uniform_distribution_agrees_with_closed_form_solution(
    val: float, n: int
) -> None:
    assert_almost_equal(features._spread(np.array([val] * n)), np.sqrt((n ** 2 - 1) / 12))
