"""
Experiment 1: Static Gesture Recognition Benchmark.
Evaluates Precision, Recall, Macro-F1, Confusion Matrix, and Expected Calibration Error (ECE).
"""

import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gestures.gesture_types import GestureType
from src.gestures.heuristic_classifier import HeuristicGestureClassifier
from src.utils.metrics import compute_expected_calibration_error, compute_multiclass_metrics
from tests.test_gestures import build_hand_state


def generate_synthetic_gesture_samples(gesture: GestureType, num_samples: int = 400, noise_std: float = 0.015) -> list:
    """Generates synthetic 21-landmark poses with realistic anatomical variations."""
    samples = []
    base_pts = np.zeros((21, 3), dtype=np.float32)
    base_pts[0] = [0.5, 0.8, 0.0]
    base_pts[1] = [0.42, 0.72, 0.0]; base_pts[5] = [0.45, 0.60, 0.0]; base_pts[9] = [0.50, 0.58, 0.0]; base_pts[13] = [0.55, 0.60, 0.0]; base_pts[17] = [0.60, 0.64, 0.0]

    for _ in range(num_samples):
        pts = base_pts.copy()
        if gesture == GestureType.OPEN_PALM:
            # All fingers extended upward
            pts[4] = [0.32, 0.52, 0.0]; pts[8] = [0.45, 0.30, 0.0]; pts[12] = [0.50, 0.28, 0.0]; pts[16] = [0.55, 0.34, 0.0]; pts[20] = [0.60, 0.40, 0.0]
            pts[6] = [0.45, 0.50, 0.0]; pts[7] = [0.45, 0.40, 0.0]
            pts[10] = [0.50, 0.48, 0.0]; pts[11] = [0.50, 0.38, 0.0]
            pts[14] = [0.55, 0.50, 0.0]; pts[15] = [0.55, 0.42, 0.0]
            pts[18] = [0.60, 0.56, 0.0]; pts[19] = [0.60, 0.48, 0.0]
            pts[2] = [0.38, 0.65, 0.0]; pts[3] = [0.35, 0.58, 0.0]

        elif gesture == GestureType.POINT:
            # Index extended, middle/ring/pinky folded, thumb folded
            pts[6] = [0.45, 0.50, 0.0]; pts[7] = [0.45, 0.40, 0.0]; pts[8] = [0.45, 0.30, 0.0]
            pts[10] = [0.50, 0.64, 0.0]; pts[11] = [0.50, 0.68, 0.0]; pts[12] = [0.50, 0.72, 0.0]
            pts[14] = [0.55, 0.64, 0.0]; pts[15] = [0.55, 0.68, 0.0]; pts[16] = [0.55, 0.72, 0.0]
            pts[18] = [0.60, 0.66, 0.0]; pts[19] = [0.60, 0.70, 0.0]; pts[20] = [0.60, 0.74, 0.0]
            pts[2] = [0.42, 0.70, 0.0]; pts[3] = [0.44, 0.68, 0.0]; pts[4] = [0.46, 0.68, 0.0]

        elif gesture == GestureType.PINCH:
            # Thumb and index tips touching at (0.42, 0.45, 0.0), other fingers partially relaxed
            pts[4] = [0.42, 0.45, 0.0]; pts[8] = [0.42, 0.45, 0.0]
            pts[6] = [0.44, 0.52, 0.0]; pts[7] = [0.43, 0.48, 0.0]
            pts[2] = [0.40, 0.65, 0.0]; pts[3] = [0.41, 0.55, 0.0]
            # Middle, ring, pinky slightly curled/relaxed
            pts[10] = [0.50, 0.55, 0.0]; pts[11] = [0.50, 0.50, 0.0]; pts[12] = [0.50, 0.46, 0.0]
            pts[14] = [0.55, 0.57, 0.0]; pts[15] = [0.55, 0.52, 0.0]; pts[16] = [0.55, 0.48, 0.0]
            pts[18] = [0.60, 0.60, 0.0]; pts[19] = [0.60, 0.55, 0.0]; pts[20] = [0.60, 0.50, 0.0]

        elif gesture == GestureType.GRAB:
            # All fingers curled tightly
            pts[2] = [0.42, 0.70, 0.0]; pts[3] = [0.44, 0.68, 0.0]; pts[4] = [0.46, 0.68, 0.0]
            pts[6] = [0.45, 0.64, 0.0]; pts[7] = [0.45, 0.68, 0.0]; pts[8] = [0.45, 0.72, 0.0]
            pts[10] = [0.50, 0.64, 0.0]; pts[11] = [0.50, 0.68, 0.0]; pts[12] = [0.50, 0.72, 0.0]
            pts[14] = [0.55, 0.64, 0.0]; pts[15] = [0.55, 0.68, 0.0]; pts[16] = [0.55, 0.72, 0.0]
            pts[18] = [0.60, 0.66, 0.0]; pts[19] = [0.60, 0.70, 0.0]; pts[20] = [0.60, 0.74, 0.0]

        # Add Gaussian kinematic sensor noise
        pts += np.random.normal(0, noise_std, size=pts.shape).astype(np.float32)
        samples.append(pts)

    return samples


def run_experiment_1():
    print("\n=======================================================")
    print("  EXPERIMENT 1: Static Gesture Recognition Benchmark  ")
    print("=======================================================")

    classifier = HeuristicGestureClassifier()
    classes = [GestureType.OPEN_PALM, GestureType.POINT, GestureType.PINCH, GestureType.GRAB]

    y_true = []
    y_pred = []
    confidences = []
    accuracies = []

    samples_per_class = 400

    for target_g in classes:
        samples = generate_synthetic_gesture_samples(target_g, num_samples=samples_per_class)
        for pts in samples:
            hand_state = build_hand_state(pts)
            rec = classifier.classify_single_hand(hand_state)

            y_true.append(target_g.value)
            y_pred.append(rec.gesture.value)
            confidences.append(rec.confidence)
            accuracies.append(1.0 if rec.gesture == target_g else 0.0)

    class_names = [c.value for c in classes]
    metrics = compute_multiclass_metrics(y_true, y_pred, class_names)
    ece = compute_expected_calibration_error(np.array(confidences), np.array(accuracies), num_bins=10)

    print(f"Total Evaluation Samples: {len(y_true)}")
    print(f"Macro Precision:          {metrics['macro_precision']*100:.2f}%")
    print(f"Macro Recall:             {metrics['macro_recall']*100:.2f}%")
    print(f"Macro F1-Score:           {metrics['macro_f1']*100:.2f}%")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print("-------------------------------------------------------")

    return {
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "ece": ece,
    }


if __name__ == "__main__":
    run_experiment_1()
