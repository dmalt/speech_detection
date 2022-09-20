import numpy as np

from .thresholding import get_local_maxima, get_weighted_average_threshold

def calculate_mask(signal, max, th):
    return signal > th if max[0] <= max[1] else signal < th

def thresholding(features, weight):
    nrg = features[0]     # signal energy is the main feature, so we compare other ones with it
    mask = np.full(len(nrg), True)
    statistics = { 'hist' : [],
                   'th'   : [] }
    
    for i, feature in enumerate(features):
        assert(len(feature) == len(mask), "Features must be the same size")
        
        hist = np.histogram(np.trim_zeros(feature), "fd")
        statistics['hist'].append(hist)
        
        max = get_local_maxima(hist, 2)
        if i > 0 and np.corrcoef(nrg, feature)[0][1] < 0.1:
            max = np.flip(max)
        th = get_weighted_average_threshold(feature, max, weight)
        statistics['th'].append(th)
        
        print(max)
        print(th)
        
        mask = np.logical_and(mask, calculate_mask(feature, max, th))
    return mask, statistics
