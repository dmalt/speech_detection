from math import ceil

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import integers

from speech_detection import windowing


def test_compute_windowed_feature_raises_assertion_error_on_negative_window_length() -> None:
    with pytest.raises(AssertionError):
        windowing.compute_windowed_feature(np.ones(10), -1, lambda _: 1)


@given(integers(1, 1000), integers(1, 1000))
def test_compute_windowed_feature_returns_array_of_proper_length(
    array_length: int, window_length: int
) -> None:
    res = windowing.compute_windowed_feature(np.ones(array_length), window_length, lambda _: 1)

    assert len(res) == ceil(array_length / window_length)
