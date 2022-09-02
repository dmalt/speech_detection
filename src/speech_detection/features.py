from math import ceil, sqrt

import numpy as np

from .transforms import normalize


def energy(sound, window_length: int):
    assert window_length > 0, "Window length must be positive integer"
    sound = normalize(sound)

    num_frames = ceil(len(sound) / window_length)
    result = np.empty(num_frames)
    lo, hi = 0, window_length
    for i in range(num_frames):
        result[i] = np.square(sound[lo:hi]).mean()
        lo += window_length
        hi += window_length

    return result


def spectral_centroid(sound, window_length: int):
    assert window_length > 0, "Window length must be positive integer"
    num_frames = ceil(len(sound) / window_length)
    result = np.empty(num_frames)

    lo, hi = 0, window_length
    for i in range(num_frames):
        window = sound[lo:hi]
        window = np.multiply(window, np.hamming(len(window)))

        fft = np.abs(np.fft.rfft(window))
        fft = np.multiply(fft, fft >= 0.002 * window_length)
        fft = normalize(fft)

        freq_sum = sum((j + 1) * f for j, f in enumerate(fft))
        fft_sum = np.sum(fft)
        result[i] = freq_sum / fft_sum if fft_sum else 0

        lo += window_length
        hi += window_length

    return result


def spectral_spread(sound, window_length: int):
    assert window_length > 0, "Window length must be positive integer"
    num_frames = ceil(len(sound) / window_length)
    result = np.empty(num_frames)

    lo, hi = 0, window_length
    for i in range(num_frames):
        window = sound[lo:hi]
        window = np.multiply(window, np.hamming(len(window)))

        fft = np.abs(np.fft.rfft(window))
        fft = np.multiply(fft, fft >= 0.002 * window_length)
        fft = normalize(fft)

        freq_sum = sum((f + 1) * s for f, s in enumerate(fft))
        fft_sum = np.sum(fft)
        centroid = freq_sum / fft_sum if fft_sum else 0

        freq_sum = sum((f + 1 - centroid) * (f + 1 - centroid) * s for f, s in enumerate(fft))
        result[i] = sqrt(freq_sum / fft_sum if fft_sum else 0)

        lo += window_length
        hi += window_length

    return result
