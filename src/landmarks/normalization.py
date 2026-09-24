"""
Landmark normalization for scale and translation invariance.
"""

from typing import List, Tuple
import numpy as np


class LandmarkNormalizer:
    """
    Normalizes 21 3D hand landmarks by centering on the palm centroid and
    scaling by the characteristic anatomical hand length (wrist to middle MCP joint).
    """

    def __init__(self, palm_indices: List[int] = None, wrist_idx: int = 0, middle_mcp_idx: int = 9):
        self.palm_indices = palm_indices or [0, 5, 9, 17]
        self.wrist_idx = wrist_idx
        self.middle_mcp_idx = middle_mcp_idx

    def compute_palm_center(self, raw_landmarks: np.ndarray) -> np.ndarray:
        """
        Computes the palm center as the centroid of palm anchor points.
        raw_landmarks: shape (21, 3)
        Returns: shape (3,)
        """
        palm_pts = raw_landmarks[self.palm_indices]
        return np.mean(palm_pts, axis=0)

    def compute_characteristic_scale(self, raw_landmarks: np.ndarray) -> float:
        """
        Computes the reference scale d_ref = ||p_wrist - p_middle_mcp||_2.
        Avoids division by zero using a defensive epsilon clamp.
        """
        p_wrist = raw_landmarks[self.wrist_idx]
        p_middle = raw_landmarks[self.middle_mcp_idx]
        d_ref = float(np.linalg.norm(p_wrist - p_middle))
        return max(d_ref, 1e-4)

    def normalize(self, raw_landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Applies scale and translation normalization:
        p_norm = (p_raw - p_palm) / d_ref

        Returns:
            normalized_landmarks: shape (21, 3)
            palm_center: shape (3,)
            d_ref: float
        """
        palm_center = self.compute_palm_center(raw_landmarks)
        d_ref = self.compute_characteristic_scale(raw_landmarks)
        normalized = (raw_landmarks - palm_center) / d_ref
        return normalized, palm_center, d_ref
