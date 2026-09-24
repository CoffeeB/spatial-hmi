"""
Continuous confidence estimation using smooth sigmoid and Gaussian activation functions.
"""

import numpy as np


class ConfidenceEstimator:
    """
    Computes continuous, differentiable confidence scores for physical and kinematic gesture metrics.
    Avoids binary step-function discretization.
    """

    @staticmethod
    def sigmoid_confidence(value: float, threshold: float, steepness: float = 12.0, invert: bool = False) -> float:
        """
        Smooth sigmoid confidence.
        If invert is False: confidence is high when value > threshold.
        If invert is True: confidence is high when value < threshold.
        """
        diff = (value - threshold) if not invert else (threshold - value)
        exponent = steepness * diff
        exponent = np.clip(exponent, -25.0, 25.0)
        return float(1.0 / (1.0 + np.exp(-exponent)))

    @staticmethod
    def gaussian_confidence(value: float, target: float, sigma: float = 0.10) -> float:
        """Gaussian radial basis confidence centered at target."""
        diff = value - target
        variance = 2.0 * (sigma**2)
        return float(np.exp(-(diff**2) / variance))

    @staticmethod
    def combine_confidences(confidences: list[float], weights: list[float] = None) -> float:
        """Computes weighted product / average of multiple sub-feature confidences."""
        if not confidences:
            return 0.0
        if weights is None:
            # Geometric mean for multi-attribute gating
            return float(np.prod(np.array(confidences)) ** (1.0 / len(confidences)))
        w = np.array(weights) / np.sum(weights)
        return float(np.sum(np.array(confidences) * w))
