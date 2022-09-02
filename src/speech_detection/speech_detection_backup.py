from math import ceil

import numpy as np
from scipy.signal import medfilt


def normalize(sound: np.ndarray) -> np.ndarray:
    max = sound.max()
    return sound / max if max > 0 else sound


def compute_signal_energy(sound, window_length_samp):
    assert window_length_samp > 0, "Window length must be positive integer"
    sound = normalize(sound)

    num_frames = ceil(len(sound) / window_length_samp)
    result = np.empty(num_frames)
    lo, hi = 0, window_length_samp
    for i in range(num_frames):
        result[i] = np.square(sound[lo:hi]).mean()
        lo += window_length_samp
        hi += window_length_samp

    return result


def compute_spectral_centroid(sound, window_length_samp):
    assert window_length_samp > 0, "Window length must be positive integer"
    num_frames = ceil(len(sound) / window_length_samp)
    result = np.empty(num_frames)
    sound = normalize(sound)

    lo, hi = 0, window_length_samp
    for i in range(num_frames):
        window = sound[lo:hi]
        window = np.multiply(window, np.hamming(len(window)))
        fft = np.fft.rfft(window)
        freq_sum = sum((j + 1) * np.abs(f) for j, f in enumerate(fft))
        fft_sum = np.abs(fft).sum()
        result[i] = freq_sum / fft_sum if fft_sum else 0

        lo += window_length_samp
        hi += window_length_samp

    return result


def smooth_signal(signal, kernel_size):
    result = medfilt(signal, kernel_size=kernel_size)
    return medfilt(result, kernel_size=kernel_size)


def get_local_maxima(hist, n_maxima):
    max_args = [-1] * n_maxima
    bin_counts, bin_edges = hist[0], hist[1]
    bound = 0.02 * np.mean(bin_counts)

    i = 0
    while i < len(bin_counts):
        max_vals = [bin_counts[j] if j >= 0 else 0 for j in max_args]
        if (
            bin_counts[i] > np.min(max_vals)
            and (i == 0 or bin_counts[i] >= bin_counts[i - 1])
            and (i == (len(bin_counts) - 1) or bin_counts[i] >= bin_counts[i + 1])
            and bin_counts[i] > bound
        ):
            max_args[np.argmin(max_vals)] = i
            i += 2
        else:
            i += 1
        print(f"{max_args=}")

    max_args = np.sort(max_args)

    print("---------------------")
    print(bin_edges)
    return np.array(
        [((bin_edges[i] + bin_edges[i + 1]) * 0.5) if i >= 0 else -1.0 for i in max_args]
    )


def get_weighted_average_threshold(v1, v2, weight):
    return (weight * v1 + v2) / (weight + 1)


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


class SpeechDetector:
    def __init__(self, sound, sr):
        window_length = round(0.025 * sr)
        num_bins = round(0.002 * sr)

        nrg = compute_signal_energy(sound, window_length)
        spc = compute_spectral_centroid(sound, window_length)
        nrgmsk = self._compute_feature_mask(nrg, num_bins)
        spcmsk = self._compute_feature_mask(spc, num_bins, above=False)
        # spcmsk = spcsm < 50
        msk = np.logical_and(nrgmsk, spcmsk)
        # msk = spcmsk
        self.mask = post_process(msk, len(sound), window_length, 7 * window_length)

    def _compute_feature_mask(self, feature, hist_nbins, above=True):
        feature_smoothed = smooth_signal(feature, kernel_size=5)
        hist = np.histogram(feature_smoothed, hist_nbins)
        local_maxima = get_local_maxima(hist, n_maxima=2)
        thresh = get_weighted_average_threshold(local_maxima[0], local_maxima[1], weight=5)
        return feature_smoothed > thresh if above else feature_smoothed < thresh

    # def draw(self):
    #     import matplotlib.pyplot as plt

    #     fig, axs = plt.subplots(2, 3)
    #     axs[0][0].plot(sound, linewidth=0.5)
    #     axs[1][0].plot(mask, linewidth=0.7, zorder=2)
    #     axs[1][0].plot(nrgmsk, linewidth=0.4, zorder=1)
    #     axs[1][0].plot(spcmsk, linewidth=0.4, zorder=1)
    #     axs[0][1].plot(nrg, linewidth=0.5, zorder=1)
    #     axs[0][1].plot(nrgsm, linewidth=0.5, zorder=2)
    #     axs[0][1].plot([0, len(nrg)], [nrgth, nrgth], linewidth=0.7, zorder=3)
    #     axs[1][1].plot(spc, linewidth=0.5, zorder=1)
    #     axs[1][1].plot(spcsm, linewidth=0.5, zorder=2)
    #     axs[1][1].plot([0, len(spc)], [spcth, spcth], linewidth=0.7, zorder=3)
    #     axs[0][2].bar(nrghst[1][:-1], nrghst[0], width=np.diff(nrghst[1]))
    #     axs[1][2].bar(spchst[1][:-1], spchst[0], width=np.diff(spchst[1]))
    #     plt.show()

# def remove_silence(sound, df):
#     mask = detect_speech(sound, df)
#     return np.multiply(sound, mask).astype(np.int16)


if __name__ == "__main__":
    import matplotlib
    import mne
    from mne import create_info
    from mne.io import RawArray
    from scipy.io.wavfile import read

    matplotlib.use("TkAgg")

    def annots_from_mask(mask: np.ndarray, sr: float, type: str) -> mne.Annotations:
        onset = []
        duration = []
        description = []
        prev_seg_start = 0 if mask[0] else None
        prev_sample = mask[0]
        for i, sample in enumerate(mask[1:], start=1):
            if not prev_sample and sample:
                prev_seg_start = i
            elif prev_sample and not sample:
                # one sample segment counts as zero-length
                assert prev_seg_start is not None
                onset.append(prev_seg_start / sr)
                duration.append((i - 1 - prev_seg_start) / sr)
                description.append(type)
                prev_seg_start = None
            prev_sample = sample
        if prev_seg_start is not None:
            onset.append(prev_seg_start / sr)
            duration.append((len(mask) - 1 - prev_seg_start) / sr)
            description.append(type)
        return mne.Annotations(onset, duration, description)

    # sr, sound = read("sub-01_task-speech_proc-align_beh.wav")
    sr, sound = read("sub-02_task-overtcovert_beh.wav")
    print(f"Read sound of length {len(sound) / sr} with sampling rate = {sr}")
    print(f"{sound=}")
    lo, hih = 30 * sr, 350 * sr
    sound = sound.astype(float)[lo:hih]
    print("Computing audio mask")
    mask = SpeechDetector(sound, sr).mask
    print("Done")
    print(f"{len(mask)=}")

    info = create_info(ch_names=["audio"], sfreq=sr)
    raw = RawArray(sound[np.newaxis, :], info)
    annots = annots_from_mask(mask, sr=sr, type="speech")
    for o, d in zip(annots.onset, annots.duration):
        print(f"onset={o}, duration={d}, offset={o+d}")

    raw.set_annotations(annots)
    print("Downsampling ")
    raw.resample(sfreq=500)
    raw.plot(block=True)
