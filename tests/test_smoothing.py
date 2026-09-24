"""
Unit tests for the 1€ Filter (OneEuroFilter) and Double Exponential Smoothing.
"""

import numpy as np
import pytest

from src.interaction.smoothing import DoubleExponentialFilter, LowPassFilter, OneEuroFilter


def test_one_euro_filter_stationary_jitter_reduction():
    # 100 stationary frames with additive zero-mean Gaussian jitter
    np.random.seed(42)
    true_pos = np.array([0.5, 0.5])
    raw_signal = true_pos + np.random.normal(0, 0.05, size=(100, 2))

    filter_1euro = OneEuroFilter(fc_min=1.0, beta=0.007, d_cutoff=1.0)
    filtered_signal = []

    dt = 1.0 / 30.0
    for i, p in enumerate(raw_signal):
        out = filter_1euro.filter(p, timestamp=i * dt)
        filtered_signal.append(out)

    filtered_arr = np.array(filtered_signal)

    raw_jitter_std = np.std(np.linalg.norm(raw_signal - true_pos, axis=1))
    filtered_jitter_std = np.std(np.linalg.norm(filtered_arr[10:] - true_pos, axis=1))

    # Jitter standard deviation must decrease by at least 65%
    assert filtered_jitter_std < raw_jitter_std * 0.35


def test_one_euro_filter_dynamic_adaptation():
    filter_1euro = OneEuroFilter(fc_min=1.0, beta=0.01)

    # Fast movement: step from 0.0 to 10.0 in 1 frame
    t0 = 0.0
    filter_1euro.filter(0.0, timestamp=t0)

    # Large velocity step
    t1 = t0 + (1.0 / 60.0)
    out = filter_1euro.filter(10.0, timestamp=t1)

    # Adaptive filter should respond quickly without massive multi-frame lag
    assert out > 2.5


def test_double_exponential_filter():
    double_exp = DoubleExponentialFilter(alpha=0.5, gamma=0.3)
    # Linear ramp
    ramp = np.linspace(0, 10, 20)
    outputs = [double_exp.filter(x) for x in ramp]

    # After warm up, outputs should track the ramp
    assert abs(outputs[-1] - 10.0) < 1.0
