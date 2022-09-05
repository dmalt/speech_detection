import numpy as np
import numpy.typing as npt

from .features import energy, spectral_spread
from .thresholding import get_local_maxima, get_weighted_average_threshold
from .transforms import normalize, smooth
from .windowing import compute_windowed_feature

Mask = npt.NDArray[np.bool_]  # of shape(n_samp,)


def calculate_mask(signal, max, th):
    return signal > th if max[0] <= max[1] else signal < th


def post_process(lr_mask: Mask, sound_length: int, window_length: int, extend_length: int) -> Mask:
    hr_mask = np.zeros(sound_length).astype(bool)

    hr_pos = 0
    for i in range(len(lr_mask)):
        next_hr_pos = hr_pos + window_length
        is_on_left_bound = i > 0 and lr_mask[i] and not lr_mask[i - 1]
        is_on_right_bound = i < (len(lr_mask) - 1) and lr_mask[i] and not lr_mask[i + 1]

        if lr_mask[i]:
            hr_mask[hr_pos:next_hr_pos] = True
        if is_on_left_bound:
            hr_mask[max(hr_pos - extend_length, 0) : hr_pos] = True
        if is_on_right_bound:
            hr_mask[next_hr_pos : next_hr_pos + extend_length] = True
        hr_pos = next_hr_pos

    return hr_mask


def detect_speech(sound, sr, draw=False):
    assert sr > 0, "Sample rate must be positive integer"
    window_length = round(0.05 * sr)
    thresholding_weight = 5.0
    sm_filter_order = 5

    nrg = compute_windowed_feature(normalize(sound), window_length, energy)
    spc = compute_windowed_feature(sound, window_length, spectral_spread)
    nrgsm = smooth(nrg, sm_filter_order)
    spcsm = smooth(spc, sm_filter_order)
    nrghst = np.histogram(np.trim_zeros(nrgsm), "fd")
    spchst = np.histogram(np.trim_zeros(spcsm), "fd")
    maxnrg = get_local_maxima(nrghst, 2)
    maxspc = get_local_maxima(spchst, 2)
    if np.corrcoef(nrgsm, spcsm)[0][1] < 0:
        maxspc = np.flip(maxspc)
    nrgth = get_weighted_average_threshold(nrgsm, maxnrg, thresholding_weight)
    spcth = get_weighted_average_threshold(spcsm, maxspc, thresholding_weight)
    nrgmsk = calculate_mask(nrgsm, maxnrg, nrgth)
    spcmsk = calculate_mask(spcsm, maxspc, spcth)
    mask = post_process(
        np.logical_and(nrgmsk, spcmsk), len(sound), window_length, 5 * window_length
    )

    print(maxnrg)
    print(nrgth)
    print(maxspc)
    print(spcth)

    if draw:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 3)
        axs[0][0].plot(sound, linewidth=0.5)
        axs[0][0].plot(np.multiply(sound, mask), linewidth=0.5)
        axs[1][0].plot(mask)
        axs[0][1].plot(nrg, linewidth=0.5, zorder=1)
        axs[0][1].plot(nrgsm, linewidth=0.5, zorder=2)
        axs[0][1].plot([0, len(nrg)], [nrgth, nrgth], linewidth=0.7, zorder=3)
        axs[1][1].plot(spc, linewidth=0.5, zorder=1)
        axs[1][1].plot(spcsm, linewidth=0.5, zorder=2)
        axs[1][1].plot([0, len(spc)], [spcth, spcth], linewidth=0.7, zorder=3)
        axs[0][2].bar(nrghst[1][:-1], nrghst[0], width=np.diff(nrghst[1]))
        axs[1][2].bar(spchst[1][:-1], spchst[0], width=np.diff(spchst[1]))
        plt.show()

    return mask


def remove_silence(sound, df):
    mask = detect_speech(sound, df)
    return np.multiply(sound, mask).astype(np.int16)
