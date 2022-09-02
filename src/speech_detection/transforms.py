import numpy as np
from scipy.signal import medfilt


def normalize(signal: np.ndarray) -> np.ndarray:
    max = signal.max()
    return signal / max if max > 0 else signal


def smooth(signal, window_length: int):
    result = medfilt(signal, window_length)
    result = medfilt(result, window_length)
    return result
