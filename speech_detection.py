import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from scipy.io.wavfile import read
from scipy.signal import medfilt
from librosa import display

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

def smooth_signal(signal):
    result = medfilt(signal, 51)
    result = medfilt(result, 51)
    return result

def max_values(signal):
    
    max = [0, 0]
    hist = np.histogram(signal, bins = 'auto')
    
    for i in range(hist[0].size):
        maxv = [hist[0][j] for j in max]
        if hist[0][i] > np.min(maxv):
            max[np.argmin(maxv)] = i
    
    return [(hist[1][i] + hist[1][i + 1]) * 0.5 for i in max]

def calculate_mask(signal, max):
    th = (5.0 * max[0] + max[1]) / 6.0
    return [1 if y > th else 0 for y in signal]

def post_process(mask, window_length):
    
    bound = -1
    for i in range(mask.size):
        if mask[i] > 0:
            if bound >= 0 and i - bound < 20:
                for j in range(bound + 1, i):
                    mask[j] = 1
            bound = i
            
    res = []
    for m in mask:
        res.extend(np.ones(window_length) if m > 0 else np.zeros(window_length))
        
    return res

sound = read('C:/Users/mrfal/Downloads/silenceRemoval/example.wav')
sound = sound[1].astype(float)

window_length = 500

nrg = signal_energy(sound, window_length)
spc = spectral_centroid(sound, window_length)
nrgsm = smooth_signal(nrg)
spcsm = smooth_signal(spc)
maxnrg = max_values(nrg)
maxspc = max_values(spc)
nrgmsk = calculate_mask(nrg, maxnrg)
spcmsk = calculate_mask(spc, maxspc)
mask = post_process(np.multiply(nrgmsk, spcmsk), window_length)

fig, axs = plt.subplots(2, 3)
display.waveshow(sound, ax=axs[0][0])
axs[1][0].plot(mask)
axs[0][1].plot(nrg)
axs[1][1].plot(nrgsm)
axs[0][2].plot(spc)
axs[1][2].plot(spcsm)
plt.show()