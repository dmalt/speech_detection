import numpy as np

from speech_detection import features


def test_energy_on_empty_input_returns_zero():
    assert features.compute_energy(np.array([])) == 0
