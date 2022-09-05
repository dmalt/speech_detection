import numpy as np
import pytest

from speech_detection import windowing


def test_compute_windowed_feature_raises_assertion_error_on_negative_window_length():
    with pytest.raises(AssertionError):
        windowing.compute_windowed_feature(np.random.randn(10), -1, lambda x: 1)
