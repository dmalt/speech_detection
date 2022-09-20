import numpy as np


def get_local_maxima(hist, num_values):
    max_args = [-1] * num_values
    bin_counts, bin_edges = hist[0], hist[1]
    bound = 0.02 * np.mean(bin_counts)

    i = 0
    while i < len(bin_counts):
        max_vals = [bin_counts[j] if j >= 0 else 0 for j in max_args]
        if (
            bin_counts[i] > np.min(max_vals)
            and (i == 0 or bin_counts[i] > bin_counts[i - 1])
            and (i == (len(bin_counts) - 1) or bin_counts[i] > bin_counts[i + 1])
            and bin_counts[i] > bound
        ):
            max_args[np.argmin(max_vals)] = i
            i += 2
        else:
            i += 1

    max_args = sorted(max_args)

    return np.array(
        [((bin_edges[i] + bin_edges[i + 1]) * 0.5) if i >= 0 else np.nan for i in max_args]
    )


def get_weighted_average_threshold(signal, max, weight):
    if np.count_nonzero(~np.isnan(max)) < 2:
        return np.mean(signal)
    return (weight * max[0] + max[1]) / (weight + 1)
