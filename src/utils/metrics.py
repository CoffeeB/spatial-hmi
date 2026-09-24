"""
Scientific and statistical evaluation metrics for HCI perception systems.
"""

from typing import Dict, List, Tuple
import numpy as np


def compute_expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10,
) -> float:
    """Computes Expected Calibration Error (ECE) across confidence bins."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    if total_samples == 0:
        return 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper if i < num_bins - 1 else confidences <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_positional_jitter(positions: np.ndarray) -> float:
    """
    Computes spatial jitter standard deviation (stationary error).
    positions: shape (N, 2) or (N, 3)
    """
    if len(positions) < 2:
        return 0.0
    mean_pos = np.mean(positions, axis=0)
    deviations = np.linalg.norm(positions - mean_pos, axis=1)
    return float(np.std(deviations))


def compute_smoothness_index(trajectories: np.ndarray, dt: float = 1.0 / 30.0) -> float:
    """
    Computes the dimensionless interaction smoothness index S in [0, 1].
    S = 1 - (int |a(t)| dt) / (int |v(t)| dt + epsilon)
    """
    if len(trajectories) < 3:
        return 1.0

    velocities = np.diff(trajectories, axis=0) / dt
    accelerations = np.diff(velocities, axis=0) / dt

    v_mag = np.linalg.norm(velocities, axis=1)
    a_mag = np.linalg.norm(accelerations, axis=1)

    sum_v = np.sum(v_mag) * dt
    sum_a = np.sum(a_mag) * dt

    ratio = sum_a / (sum_v + 1e-5)
    smoothness = 1.0 / (1.0 + 0.05 * ratio)
    return float(np.clip(smoothness, 0.0, 1.0))


def compute_multiclass_metrics(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str],
) -> Dict[str, float]:
    """Computes Precision, Recall, and Macro-averaged F1."""
    confusion = {c1: {c2: 0 for c2 in classes} for c1 in classes}
    for yt, yp in zip(y_true, y_pred):
        if yt in confusion and yp in confusion[yt]:
            confusion[yt][yp] += 1

    precisions = []
    recalls = []
    f1s = []

    for c in classes:
        tp = confusion[c][c]
        fp = sum(confusion[other][c] for other in classes if other != c)
        fn = sum(confusion[c][other] for other in classes if other != c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return {
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }
