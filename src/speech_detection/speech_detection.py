import numpy as np
import numpy.typing as npt
from sklearn.model_selection import cross_val_score

from .features import compute_energy, compute_spectral_spread, compute_windowed_mask
from .transforms import normalize, smooth
from .windowing import compute_windowed_feature
from .classification import thresholding

Mask = npt.NDArray[np.bool_]  # of shape(n_samp,)

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

def compute_features(sound, window_length, sm_filter_order):
    sound = normalize(sound)
    nrg = compute_windowed_feature(sound, window_length, compute_energy)
    spc = compute_windowed_feature(sound, window_length, compute_spectral_spread)
    nrgsm = smooth(nrg, sm_filter_order)
    spcsm = smooth(spc, sm_filter_order)
    return np.array([nrgsm, spcsm])

def calculate_window_length(sr):
    return round(0.05 * sr)

def train_model(data, model):
    
    features = np.array([])
    
    for sound, sr, mask in data:
        assert sr > 0, "Sample rate must be positive integer"
        window_length = calculate_window_length(sr)
        sm_filter_order = 5
        
        fs = compute_features(sound, window_length, sm_filter_order)
        fs = np.vstack([fs, compute_windowed_feature(mask, window_length, compute_windowed_mask)])
        features.shape = (len(features), fs.shape[0])
        features = np.vstack([features, fs.T])
    
    np.random.shuffle(features)
    
    model = model.fit(features[:, 0:-1], features[:, -1])
    
    score = cross_val_score(model, features[:, 0:-1], features[:, -1], scoring="f1", cv=10)
    print(f"Model f1 score {score}")
    
    return model

def detect_speech(sound, sr, model=None, draw=False):
    assert sr > 0, "Sample rate must be positive integer"
    window_length = calculate_window_length(sr)
    thresholding_weight = 5.0
    sm_filter_order = 5
    
    features = compute_features(sound, window_length, sm_filter_order)
    
    mask = []
    statistics = None
    
    if model:
        mask = model.predict(features.T)
    else:
        mask, statistics = thresholding(features, thresholding_weight)
    
    mask = post_process(mask, len(sound), window_length, 5 * window_length)

    if draw:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 3)
        
        axs[0][0].plot(sound, linewidth=0.5)
        axs[0][0].plot(np.multiply(sound, mask), linewidth=0.5)
        axs[1][0].plot(mask)
        
        if (statistics):
            ths = statistics['th']
            hists = statistics['hist']
            axs[0][1].plot(features[0], linewidth=0.5, zorder=1)
            axs[0][1].plot([0, len(features[0])], [ths[0], ths[0]], linewidth=0.7, zorder=3)
            axs[1][1].plot(features[1], linewidth=0.5, zorder=1)
            axs[1][1].plot([0, len(features[1])], [ths[1], ths[1]], linewidth=0.7, zorder=3)
            axs[0][2].bar(hists[0][1][:-1], hists[0][0], width=np.diff(hists[0][1]))
            axs[1][2].bar(hists[1][1][:-1], hists[1][0], width=np.diff(hists[1][1]))
        
        plt.show()

    return mask


def remove_silence(sound, df):
    mask = detect_speech(sound, df)
    return np.multiply(sound, mask).astype(np.int16)
