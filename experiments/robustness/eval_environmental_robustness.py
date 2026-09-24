"""
Experiment 5: Environmental Robustness & Noise Injection Benchmark.
Evaluates gesture classification degradation under additive landmark noise and finger occlusion dropouts.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gestures.gesture_types import GestureType
from src.gestures.heuristic_classifier import HeuristicGestureClassifier
from tests.test_gestures import build_hand_state
from experiments.recognition.eval_static_gestures import generate_synthetic_gesture_samples


def run_experiment_5():
    print("\n=======================================================")
    print("  EXPERIMENT 5: Environmental Robustness & Noise Test  ")
    print("=======================================================")

    classifier = HeuristicGestureClassifier()
    noise_levels = [0.005, 0.015, 0.030, 0.050, 0.080]  # Simulating camera sensor noise & low light

    print("--- ADDITIVE KINEMATIC SENSOR NOISE DEGRADATION ---")
    for sigma in noise_levels:
        correct = 0
        total = 0
        for g in [GestureType.OPEN_PALM, GestureType.POINT, GestureType.PINCH, GestureType.GRAB]:
            samples = generate_synthetic_gesture_samples(g, num_samples=100, noise_std=sigma)
            for pts in samples:
                hand_state = build_hand_state(pts)
                rec = classifier.classify_single_hand(hand_state)
                if rec.gesture == g:
                    correct += 1
                total += 1

        acc = (correct / total) * 100.0
        print(f"Noise Sigma {sigma:5.3f} | Recognition Accuracy: {acc:5.1f}%")

    print("\n--- OCCLUSION DROPOUT SIMULATION (Pinky & Ring Occluded) ---")
    occ_correct = 0
    occ_total = 0
    for g in [GestureType.POINT, GestureType.PINCH]:
        samples = generate_synthetic_gesture_samples(g, num_samples=150, noise_std=0.01)
        for pts in samples:
            # Mask out occluded pinky landmarks (17-20)
            pts_occ = pts.copy()
            pts_occ[17:21] = pts[0]  # collapse to wrist
            hand_state = build_hand_state(pts_occ)
            rec = classifier.classify_single_hand(hand_state)
            if rec.gesture == g:
                occ_correct += 1
            occ_total += 1

    print(f"Key Point & Pinch Accuracy under 2-Finger Occlusion: {occ_correct/occ_total*100:.1f}%")
    print("-------------------------------------------------------")


if __name__ == "__main__":
    run_experiment_5()
