"""
Experiment 3: Spatial Movement Stability & Jitter Spectrum Analysis.
Compares Raw Landmarks, Static EMA, and Adaptive 1€ Filter.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.interaction.smoothing import LowPassFilter, OneEuroFilter
from src.utils.metrics import compute_positional_jitter, compute_smoothness_index


def run_experiment_3():
    print("\n=======================================================")
    print("  EXPERIMENT 3: Movement Stability & Jitter Analysis   ")
    print("=======================================================")

    np.random.seed(42)
    num_frames = 600
    dt = 1.0 / 30.0

    # 1. Stationary Hand Holding Test (Jitter Evaluation)
    true_target = np.array([0.0, 0.0])
    raw_stationary = true_target + np.random.normal(0, 0.0068, size=(num_frames, 2))

    # 2. Dynamic Trajectory Test (Smooth Ballistic Arc + High Frequency Noise)
    t_vals = np.linspace(0, 4 * np.pi, num_frames)
    true_trajectory = np.column_stack([np.sin(t_vals) * 0.5, np.cos(t_vals * 0.5) * 0.5])
    raw_trajectory = true_trajectory + np.random.normal(0, 0.005, size=(num_frames, 2))

    # Filtering Strategies
    # Baseline 1: Raw
    # Baseline 2: Static EMA (alpha = 0.30)
    # Proposed: OneEuroFilter (fc_min = 1.0, beta = 0.007)

    ema_filter = LowPassFilter(alpha=0.30)
    one_euro = OneEuroFilter(fc_min=1.0, beta=0.007, d_cutoff=1.0)

    ema_stationary = []
    one_euro_stationary = []

    for i in range(num_frames):
        p = raw_stationary[i]
        ts = i * dt
        ema_stationary.append(ema_filter.filter(p))
        one_euro_stationary.append(one_euro.filter(p, timestamp=ts))

    ema_stationary_arr = np.array(ema_stationary)
    one_euro_stationary_arr = np.array(one_euro_stationary)

    # Compute Positional Jitter
    jitter_raw = compute_positional_jitter(raw_stationary)
    jitter_ema = compute_positional_jitter(ema_stationary_arr)
    jitter_one_euro = compute_positional_jitter(one_euro_stationary_arr)

    # Dynamic Smoothness Index
    ema_dyn = []
    one_euro_dyn = []
    one_euro.reset()
    ema_filter.reset()

    for i in range(num_frames):
        p = raw_trajectory[i]
        ts = i * dt
        ema_dyn.append(ema_filter.filter(p))
        one_euro_dyn.append(one_euro.filter(p, timestamp=ts))

    smooth_raw = compute_smoothness_index(raw_trajectory, dt=dt)
    smooth_ema = compute_smoothness_index(np.array(ema_dyn), dt=dt)
    smooth_one_euro = compute_smoothness_index(np.array(one_euro_dyn), dt=dt)

    jitter_reduction_pct = ((jitter_raw - jitter_one_euro) / jitter_raw) * 100.0

    print("--- STATIONARY JITTER (Standard Deviation) ---")
    print(f"Raw Landmarks:         {jitter_raw*1000:.3f} x 10^-3 norm units")
    print(f"Static EMA Filter:     {jitter_ema*1000:.3f} x 10^-3 norm units")
    print(f"Adaptive 1€ Filter:    {jitter_one_euro*1000:.3f} x 10^-3 norm units")
    print(f">> JITTER REDUCTION:   {jitter_reduction_pct:.1f}% <<\n")

    print("--- DYNAMIC INTERACTION SMOOTHNESS INDEX (0 to 1) ---")
    print(f"Raw Landmarks:         {smooth_raw:.3f}")
    print(f"Static EMA:            {smooth_ema:.3f}")
    print(f"Adaptive 1€ Filter:    {smooth_one_euro:.3f}")
    print("-------------------------------------------------------")

    return {
        "jitter_raw": jitter_raw,
        "jitter_one_euro": jitter_one_euro,
        "jitter_reduction_pct": jitter_reduction_pct,
        "smooth_one_euro": smooth_one_euro,
    }


if __name__ == "__main__":
    run_experiment_3()
