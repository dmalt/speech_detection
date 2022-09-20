from math import ceil
from typing import Callable

import numpy as np

FeatureFunc = Callable[[np.ndarray], float]


def compute_windowed_feature(
    signal: np.ndarray, window_length: int, feature_func: FeatureFunc
) -> np.ndarray:
    assert window_length > 0, "Window length must be positive integer"
    num_frames = ceil(len(signal) / window_length)
    result = np.empty(num_frames)

    lo, hi = 0, window_length
    for i in range(num_frames):
        window = signal[lo:hi]
        result[i] = feature_func(window)
        lo += window_length
        hi += window_length
    return result
