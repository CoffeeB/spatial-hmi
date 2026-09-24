"""
Master runner for all 5 research experiments.
Executes the scientific evaluation pipeline and formats quantitative research metrics.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.recognition.eval_static_gestures import run_experiment_1
from experiments.temporal.eval_temporal_intent import run_experiment_2
from experiments.stability.eval_movement_stability import run_experiment_3
from experiments.user_variation.eval_user_variation import run_experiment_4
from experiments.robustness.eval_environmental_robustness import run_experiment_5


def main():
    print("=================================================================")
    print("  SPATIAL HMI RESEARCH ENGINE: COMPREHENSIVE BENCHMARK SUITE    ")
    print("=================================================================")
    start_time = time.time()

    res1 = run_experiment_1()
    res2 = run_experiment_2()
    res3 = run_experiment_3()
    res4 = run_experiment_4()
    run_experiment_5()

    elapsed = time.time() - start_time

    print("\n=================================================================")
    print("                 EXECUTIVE RESEARCH SUMMARY                      ")
    print("=================================================================")
    print(f"Total Benchmark Runtime:        {elapsed:.2f} seconds")
    print(f"Static Gesture Macro F1-Score:  {res1['macro_f1']*100:.1f}%")
    print(f"Expected Calibration Error (ECE): {res1['ece']:.4f}")
    print(f"False Activation Reduction:     {res2['far_reduction_pct']:.1f}% (Proposed FSM vs Instant)")
    print(f"Positional Jitter Reduction:    {res3['jitter_reduction_pct']:.1f}% (1€ Filter vs Raw)")
    print(f"Dynamic Interaction Smoothness: {res3['smooth_one_euro']:.3f} / 1.000")
    print(f"Mean Confirmation Latency:      {res2['mean_latency_ms']:.1f} ms")
    print("=================================================================")


if __name__ == "__main__":
    main()
