import matplotlib
import mne
import numpy as np
from mne import create_info
from mne.io import RawArray
from scipy.io.wavfile import read
from speech_detection.speech_detection import detect_speech

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


# sr, sound = read("test_data/sub-01_task-speech_proc-align_beh.wav")
sr, sound = read("test_data/sub-02_task-overtcovert_proc-align_beh.wav")
print(f"Read sound of length {len(sound) / sr} with sampling rate = {sr}")
print(f"{sound=}")
lo, hih = 30 * sr, 350 * sr
sound = sound.astype(float)[lo:hih]
print("Computing audio mask")
mask = detect_speech(sound, sr, draw=True)
print("Done")
print(f"{len(mask)=}")

info = create_info(ch_names=["audio"], sfreq=sr)
raw = RawArray(sound[np.newaxis, :], info)
annots = annots_from_mask(mask, sr=sr, type="speech")
raw.set_annotations(annots)
print("Downsampling ")
raw.resample(sfreq=500)
raw.plot(block=True)
