"""
Structured internal representation of hand landmarks and spatial state.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class HandLandmark:
    """Represents a single 3D joint landmark."""
    index: int
    x: float  # Normalized image x in [0, 1]
    y: float  # Normalized image y in [0, 1]
    z: float  # Relative depth
    visibility: float = 1.0


@dataclass
class HandState:
    """
    Structured, normalized representation of a single detected hand.
    Decouples raw MediaPipe landmarks from downstream gesture and interaction logic.
    """
    hand_id: int
    handedness: Literal["Left", "Right", "Unknown"]
    landmarks: List[HandLandmark]
    raw_landmarks_array: np.ndarray  # Shape (21, 3)
    normalized_landmarks_array: np.ndarray  # Shape (21, 3), invariant to scale and translation

    # Kinematic Palm Properties
    palm_center: Tuple[float, float, float]  # (x, y, z) in normalized camera coordinates
    palm_velocity: Tuple[float, float, float]  # (vx, vy, vz) per second
    hand_scale_ref: float  # Characteristic distance d_ref (wrist to middle MCP)
    orientation_angles: Tuple[float, float, float]  # (pitch, yaw, roll) in radians

    # Continuous Finger Metrics
    finger_extension_ratios: Dict[str, float]  # e.g., {'thumb': 1.1, 'index': 1.45, ...}
    finger_flexion_angles: Dict[str, float]  # in radians
    pinch_distance: float  # Normalized Euclidean distance between thumb tip and index tip
    pinch_confidence: float  # Continuous sigmoid confidence in [0, 1]

    # Tracking Quality
    detection_confidence: float
    timestamp: float

    def get_landmark(self, idx: int) -> HandLandmark:
        """Returns landmark by MediaPipe 0-20 index."""
        return self.landmarks[idx]

    @property
    def index_tip(self) -> Tuple[float, float, float]:
        """Convenience property for index fingertip (landmark 8)."""
        lm = self.landmarks[8]
        return (lm.x, lm.y, lm.z)

    @property
    def thumb_tip(self) -> Tuple[float, float, float]:
        """Convenience property for thumb fingertip (landmark 4)."""
        lm = self.landmarks[4]
        return (lm.x, lm.y, lm.z)
