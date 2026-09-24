"""
Adaptive 1€ Filter (OneEuroFilter) and Double Exponential Smoothing for spatial signals.
"""

import math
import time
from typing import Optional, Tuple, Union
import numpy as np


class LowPassFilter:
    """First-order discrete low-pass filter with exponential smoothing."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.s: Optional[np.ndarray] = None

    def reset(self):
        self.s = None

    def filter(self, value: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        a = alpha if alpha is not None else self.alpha
        if self.s is None:
            self.s = np.array(value, dtype=np.float64)
        else:
            self.s = a * np.array(value, dtype=np.float64) + (1.0 - a) * self.s
        return self.s.copy()


class OneEuroFilter:
    """
    Adaptive One-Euro filter (Casiez et al., CHI 2012).
    Dynamically adjusts cutoff frequency based on input signal derivative (velocity),
    eliminating stationary jitter while eliminating lag during fast movements.
    """

    def __init__(
        self,
        fc_min: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        self.fc_min = fc_min
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filt = LowPassFilter()
        self.dx_filt = LowPassFilter()
        self.last_timestamp: Optional[float] = None

    def _compute_alpha(self, cutoff: float, te: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + (tau / te))

    def reset(self):
        """Resets internal filter history."""
        self.x_filt.reset()
        self.dx_filt.reset()
        self.last_timestamp = None

    def filter(self, x: Union[float, np.ndarray, list], timestamp: Optional[float] = None) -> np.ndarray:
        """
        Filters input position vector or scalar x at time timestamp.
        Returns smoothed numpy array.
        """
        val = np.array(x, dtype=np.float64)
        now = timestamp if timestamp is not None else time.time()

        if self.last_timestamp is None:
            self.last_timestamp = now
            self.prev_x = val.copy()
            return self.x_filt.filter(val, alpha=1.0)

        te = now - self.last_timestamp
        if te <= 1e-5:
            # Timestamp too close -> return current filtered state
            return self.x_filt.s.copy() if self.x_filt.s is not None else val

        self.last_timestamp = now

        # 1. Estimate discrete velocity derivative
        prev_s = self.x_filt.s if self.x_filt.s is not None else val
        raw_dx = (val - prev_s) / te

        # 2. Filter velocity derivative with static cutoff d_cutoff
        alpha_d = self._compute_alpha(self.d_cutoff, te)
        filtered_dx = self.dx_filt.filter(raw_dx, alpha=alpha_d)

        # 3. Compute dynamic position cutoff frequency fc = fc_min + beta * |dx|
        dx_magnitude = np.linalg.norm(filtered_dx)
        fc = self.fc_min + self.beta * dx_magnitude

        # 4. Filter position with dynamic alpha
        alpha_x = self._compute_alpha(fc, te)
        filtered_x = self.x_filt.filter(val, alpha=alpha_x)

        return filtered_x


class DoubleExponentialFilter:
    """Holt's linear exponential trend filter for velocity and position tracking."""

    def __init__(self, alpha: float = 0.4, gamma: float = 0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.level: Optional[np.ndarray] = None
        self.trend: Optional[np.ndarray] = None

    def reset(self):
        self.level = None
        self.trend = None

    def filter(self, x: Union[float, np.ndarray]) -> np.ndarray:
        val = np.array(x, dtype=np.float64)
        if self.level is None:
            self.level = val.copy()
            self.trend = np.zeros_like(val)
            return self.level.copy()

        prev_level = self.level.copy()
        self.level = self.alpha * val + (1.0 - self.alpha) * (self.level + self.trend)
        self.trend = self.gamma * (self.level - prev_level) + (1.0 - self.gamma) * self.trend
        return self.level.copy()
