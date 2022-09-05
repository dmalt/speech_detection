from math import sqrt

import numpy as np

from .transforms import normalize


class BadArrayException(Exception):
    pass


def compute_energy(signal: np.ndarray) -> float:
    return np.square(signal).mean() if len(signal) else 0


def _amplitude_spectrum(signal: np.ndarray, threshold: float) -> np.ndarray:
    signal = np.multiply(signal, np.hamming(len(signal)))

    fft = np.abs(np.fft.rfft(signal))
    fft = np.multiply(fft, fft >= threshold * len(signal))
    return normalize(fft)


def _centroid(x: np.ndarray) -> float:
    if np.any(x < 0):
        raise BadArrayException("All weights for centroid must be nonnegative")
    return np.arange(len(x)).dot(x) / np.sum(x) if np.sum(x) else 0


def _spread(x: np.ndarray) -> float:
    if np.any(x < 0):
        raise BadArrayException("All weights for centroid must be nonnegative")
    c = _centroid(x)
    x_sum = np.sum(x)
    square_dev = (np.arange(1, len(x) + 1) - c) ** 2
    s = square_dev.dot(x)
    return sqrt(s / x_sum) if x_sum else 0


def compute_spectral_centroid(signal: np.ndarray, threshold: float = 0.002) -> float:
    fft = _amplitude_spectrum(signal, threshold)
    return _centroid(fft)


def compute_spectral_spread(signal: np.ndarray, threshold: float = 0.002) -> float:
    fft = _amplitude_spectrum(signal, threshold)
    return _spread(fft)
