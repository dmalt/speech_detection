import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import medfilt
from librosa import display
from math import ceil

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

def smooth_signal(signal, bin_length):
    
    result = []
    
    bins = [signal[i:i + bin_length] for i in range(0, len(signal), bin_length)]
    for bin in bins:
        result.extend(np.full(bin.size, np.mean(bin)))
    
    result = medfilt(result, 375)
    
    return result

def maximum_values(signal):
    
    max = np.zeros(2)
    
    for v in signal:
        if v > np.min(max):
            max[np.argmin(max)] = v
    
    return max    

def calculate_threshold(weight, max):
    return (weight * max[0] + max[1]) / (weight + 1)

sound = read('C:/Users/mrfal/Downloads/silenceRemoval/example.wav')
sound = sound[1].astype(float)

window_length = 50

nrg = signal_energy(sound, window_length)
spc = spectral_centroid(sound, window_length)
nrgsm = smooth_signal(nrg, 25)
spcsm = smooth_signal(spc, 25)
maxnrg = maximum_values(nrg)
maxspc = maximum_values(spc)
tnrg = calculate_threshold(5, maxnrg)
tspc = calculate_threshold(5, maxspc)

res = []
for i in range(nrg.size):
    res.extend(np.ones(window_length) if nrg[i] > tnrg and spc[i] > tspc else np.zeros(window_length))    

print(maxnrg)
print(maxspc)
print(tnrg)
print(tspc)

fig, axs = plt.subplots(3, 2)
display.waveshow(sound, ax=axs[0][0])
axs[0][1].plot(res)
axs[1][0].plot(nrg)
axs[1][1].plot(nrgsm)
axs[2][0].plot(spc)
axs[2][1].plot(spcsm)
plt.show()