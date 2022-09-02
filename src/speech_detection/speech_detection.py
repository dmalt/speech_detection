import numpy as np

from .features import energy, spectral_spread
from .thresholding import get_local_maxima, get_weighted_average_threshold
from .transforms import smooth


def calculate_mask(signal, max, th):
    return signal > th if max[0] <= max[1] else signal < th


def post_process(mask, sound_length, window_length, extend_length):
    res = np.zeros(sound_length).astype(bool)

    pos = 0
    for i in range(len(mask)):
        next_pos = pos + window_length
        if mask[i]:
            res[pos:next_pos] = True
        if i > 0 and mask[i] and not mask[i - 1]:
            res[max(pos - extend_length, 0) : pos] = True
        if i < (len(mask) - 1) and mask[i] and not mask[i + 1]:
            res[next_pos : next_pos + extend_length] = True
        pos = next_pos

    return res


def detect_speech(sound, sr, draw=False):
    assert sr > 0, "Sample rate must be positive integer"
    window_length = round(0.05 * sr)
    thresholding_weight = 5.0
    sm_filter_order = 5

    nrg = energy(sound, window_length)
    spc = spectral_spread(sound, window_length)
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
