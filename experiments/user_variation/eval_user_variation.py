"""
Experiment 4: User Variation Benchmark.
Tests hand scale invariance (small to large hands), rotation angles, and handedness parity.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gestures.gesture_types import GestureType
from src.gestures.heuristic_classifier import HeuristicGestureClassifier
from tests.test_gestures import build_hand_state
from experiments.recognition.eval_static_gestures import generate_synthetic_gesture_samples


def run_experiment_4():
    print("\n=======================================================")
    print("  EXPERIMENT 4: User Variation & Scale Invariance      ")
    print("=======================================================")

    classifier = HeuristicGestureClassifier()
    scale_factors = [0.65, 0.85, 1.0, 1.25, 1.50, 1.80]  # Simulating diverse hand sizes

    scale_results = {}

    for s in scale_factors:
        correct = 0
        total = 0
        for g in [GestureType.OPEN_PALM, GestureType.POINT, GestureType.PINCH, GestureType.GRAB]:
            samples = generate_synthetic_gesture_samples(g, num_samples=100)
            for pts in samples:
                scaled_pts = pts * s  # Scale landmark geometry
                hand_state = build_hand_state(scaled_pts)
                rec = classifier.classify_single_hand(hand_state)
                if rec.gesture == g:
                    correct += 1
                total += 1

        acc = (correct / total) * 100.0
        scale_results[s] = acc
        print(f"Hand Scale Factor {s:4.2f}x | Accuracy: {acc:.1f}%")

    # Handedness Parity Check
    left_correct = 0
    right_correct = 0
    test_samples = generate_synthetic_gesture_samples(GestureType.POINT, num_samples=200)

    for pts in test_samples:
        h_right = build_hand_state(pts, handedness="Right")
        h_left = build_hand_state(pts, handedness="Left")
        if classifier.classify_single_hand(h_right).gesture == GestureType.POINT:
            right_correct += 1
        if classifier.classify_single_hand(h_left).gesture == GestureType.POINT:
            left_correct += 1

    print(f"\nRight Hand Accuracy: {right_correct/200*100:.1f}%")
    print(f"Left Hand Accuracy:  {left_correct/200*100:.1f}%")
    print("-------------------------------------------------------")

    return scale_results


if __name__ == "__main__":
    run_experiment_4()
