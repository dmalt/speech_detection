import numpy as np
from math import ceil
from scipy.signal import medfilt

def signal_energy(sound, window_length):
    
    num_frames = ceil(sound.size / window_length) if window_length > 0 else 0
    result = np.empty(num_frames)
    
    max = np.max(sound)
    if max > 0: sound = sound / max
    
    pos = 0
    for i in range(num_frames):
        next_pos = pos + window_length
        window = sound[pos:(next_pos if next_pos <= sound.size else sound.size)]
        result[i] = 1 / window.size * np.sum(np.square(window))
        pos = next_pos
        
    return result

def spectral_centroid(sound, window_length):
    
    num_frames = ceil(sound.size / window_length) if window_length > 0 else 0
    result = np.empty(num_frames)
    
    pos = 0
    for i in range(num_frames):
        next_pos = pos + window_length
        window = sound[pos:(next_pos if next_pos <= sound.size else sound.size)]
        
        fft = np.fft.rfft(window)
        sum = 0
        freq_sum = 0
        for j in range(fft.size):
            sum += np.abs(fft[j])
            freq_sum += (j + 1) * np.abs(fft[j])
        result[i] = (freq_sum / sum) if sum != 0 else 0
        
        pos = next_pos
    
    return result

def smooth_signal(signal, window_length):
    result = medfilt(signal, window_length)
    result = medfilt(result, window_length)
    return result

def max_values(signal, num_values):
    
    max = [0] * num_values
    hist = np.histogram(signal, bins = 'auto')
    
    for i in range(hist[0].size):
        maxv = [hist[0][j] for j in max]
        if hist[0][i] > np.min(maxv):
            max[np.argmin(maxv)] = i
    
    return [(hist[1][i] + hist[1][i + 1]) * 0.5 for i in max]

def calculate_mask(signal, max, weight):
    th = (weight * max[0] + max[1]) / (weight + 1)
    return [1 if y > th else 0 for y in signal]

def post_process(mask, sound_length, window_length, extend_length):
    
    res = [0] * sound_length
    
    def set(start, end):
        start = start if start >= 0 else 0
        end = end if end <= sound_length else sound_length
        for i in range(start, end):
            res[i] = 1
    
    pos = 0
    for i in range(mask.size):
        next_pos = pos + window_length
        if (mask[i] > 0):
            set(pos, next_pos)
        if i > 0 and mask[i] > 0 and mask[i - 1] == 0:
            set(pos - extend_length, pos)
        if i < mask.size and mask[i] > 0 and mask[i + 1] == 0:
            set(next_pos + 1, next_pos + extend_length)
        pos = next_pos
        
    return res

def detect_speech(sound, df, draw = 0):
    
    window_length = round(0.025 * df)
    
    nrg = signal_energy(sound, window_length)
    spc = spectral_centroid(sound, window_length)
    nrgsm = smooth_signal(nrg, 5)
    spcsm = smooth_signal(spc, 5)
    maxnrg = max_values(nrgsm, 2)
    maxspc = max_values(spcsm, 2)
    nrgmsk = calculate_mask(nrg, maxnrg, 5.0)
    spcmsk = calculate_mask(spc, maxspc, 5.0)
    mask = post_process(np.multiply(nrgmsk, spcmsk), sound.size, window_length, 5 * window_length)
    
    if draw:
        import matplotlib.pyplot as plt
        from librosa import display
        
        fig, axs = plt.subplots(2, 3)
        display.waveshow(sound, ax=axs[0][0])
        axs[1][0].plot(mask)
        axs[0][1].plot(nrg)
        axs[1][1].plot(nrgsm)
        axs[0][2].plot(spc)
        axs[1][2].plot(spcsm)
        plt.show()
        
    return mask

def remove_silence(sound, df):
    mask = detect_speech(sound, df)
    return np.multiply(sound, mask)