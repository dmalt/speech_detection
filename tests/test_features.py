import numpy as np
from hypothesis import given
from hypothesis.strategies import floats

from speech_detection import features


def test_energy_on_empty_input_returns_zero():
    assert features.compute_energy(np.array([])) == 0


@given(floats(min_value=-1e12, max_value=1e12))
def test_energy_on_single_value_returns_this_value_squared(val: float) -> None:
    assert features.compute_energy(np.array([val])) == val ** 2
