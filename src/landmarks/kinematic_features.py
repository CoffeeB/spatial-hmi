"""
Kinematic feature extraction from normalized 3D hand landmarks.
"""

from typing import Dict, Tuple
import numpy as np


FINGER_INDICES = {
    "thumb": {"mcp": 1, "pip": 2, "dip": 3, "tip": 4},
    "index": {"mcp": 5, "pip": 6, "dip": 7, "tip": 8},
    "middle": {"mcp": 9, "pip": 10, "dip": 11, "tip": 12},
    "ring": {"mcp": 13, "pip": 14, "dip": 15, "tip": 16},
    "pinky": {"mcp": 17, "pip": 18, "dip": 19, "tip": 20},
}


class KinematicFeatureExtractor:
    """
    Extracts continuous geometric and kinematic features from 3D hand landmarks:
    - Finger extension ratios
    - Joint flexion angles
    - Pinch distance and continuous sigmoid confidence
    - Hand orientation (pitch, yaw, roll)
    - Filtered palm velocity
    """

    def __init__(self, pinch_steepness: float = 18.0, pinch_threshold: float = 0.38, velocity_alpha: float = 0.40):
        self.pinch_steepness = pinch_steepness
        self.pinch_threshold = pinch_threshold
        self.velocity_alpha = velocity_alpha
        self.prev_palm_center: Dict[int, np.ndarray] = {}
        self.prev_palm_velocity: Dict[int, np.ndarray] = {}
        self.prev_timestamp: Dict[int, float] = {}

    def compute_finger_extension_ratios(self, raw_landmarks: np.ndarray) -> Dict[str, float]:
        """
        Computes the ratio of distance(tip, wrist) to distance(mcp, wrist).
        A ratio > 1.2 indicates an extended finger, while < 0.9 indicates a curled finger.
        """
        p_wrist = raw_landmarks[0]
        ratios = {}
        for finger_name, indices in FINGER_INDICES.items():
            p_tip = raw_landmarks[indices["tip"]]
            p_mcp = raw_landmarks[indices["mcp"]]
            d_tip = np.linalg.norm(p_tip - p_wrist)
            d_mcp = max(np.linalg.norm(p_mcp - p_wrist), 1e-4)
            ratios[finger_name] = float(d_tip / d_mcp)
        return ratios

    def compute_joint_flexion_angles(self, raw_landmarks: np.ndarray) -> Dict[str, float]:
        """
        Computes the 3D joint angle at the PIP joint between MCP->PIP and PIP->TIP.
        Returns angles in radians.
        """
        angles = {}
        for finger_name, indices in FINGER_INDICES.items():
            p_mcp = raw_landmarks[indices["mcp"]]
            p_pip = raw_landmarks[indices["pip"]]
            p_tip = raw_landmarks[indices["tip"]]

            v1 = p_mcp - p_pip
            v2 = p_tip - p_pip

            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 < 1e-5 or norm_v2 < 1e-5:
                angles[finger_name] = 0.0
                continue

            cosine = np.dot(v1, v2) / (norm_v1 * norm_v2)
            cosine = np.clip(cosine, -1.0, 1.0)
            angles[finger_name] = float(np.arccos(cosine))
        return angles

    def compute_pinch_metric(self, raw_landmarks: np.ndarray, d_ref: float) -> Tuple[float, float]:
        """
        Computes normalized distance between thumb tip (4) and index tip (8).
        Calculates a continuous sigmoid confidence in [0, 1].
        """
        p_thumb = raw_landmarks[4]
        p_index = raw_landmarks[8]
        dist_raw = np.linalg.norm(p_thumb - p_index)
        normalized_dist = float(dist_raw / max(d_ref, 1e-4))

        # Sigmoid: confidence is high when normalized_dist < pinch_threshold
        # c = 1 / (1 + exp(k * (d - d_th)))
        exponent = self.pinch_steepness * (normalized_dist - self.pinch_threshold)
        exponent = np.clip(exponent, -25.0, 25.0)
        confidence = float(1.0 / (1.0 + np.exp(exponent)))

        return normalized_dist, confidence

    def compute_hand_orientation(self, raw_landmarks: np.ndarray) -> Tuple[float, float, float]:
        """
        Estimates the hand coordinate frame orientation:
        - X axis: from Index MCP (5) to Pinky MCP (17) (lateral)
        - Y axis: from Wrist (0) to Middle MCP (9) (longitudinal)
        - Z axis: normal vector (cross product X x Y)
        Returns (pitch, yaw, roll) in radians.
        """
        p_wrist = raw_landmarks[0]
        p_index_mcp = raw_landmarks[5]
        p_middle_mcp = raw_landmarks[9]
        p_pinky_mcp = raw_landmarks[17]

        v_y = p_middle_mcp - p_wrist
        v_x = p_pinky_mcp - p_index_mcp

        norm_y = np.linalg.norm(v_y)
        norm_x = np.linalg.norm(v_x)

        if norm_y < 1e-5 or norm_x < 1e-5:
            return 0.0, 0.0, 0.0

        v_y = v_y / norm_y
        v_x = v_x / norm_x
        v_z = np.cross(v_x, v_y)
        norm_z = np.linalg.norm(v_z)
        if norm_z > 1e-5:
            v_z = v_z / norm_z

        # Euler angles from rotation matrix
        pitch = float(np.arcsin(np.clip(-v_y[2], -1.0, 1.0)))
        yaw = float(np.arctan2(v_y[0], v_y[1]))
        roll = float(np.arctan2(v_x[2], v_z[2]))

        return pitch, yaw, roll

    def compute_palm_velocity(
        self, hand_id: int, current_palm_center: np.ndarray, timestamp: float
    ) -> Tuple[float, float, float]:
        """
        Computes exponentially smoothed palm velocity (vx, vy, vz) in units/second.
        """
        if hand_id not in self.prev_palm_center or hand_id not in self.prev_timestamp:
            self.prev_palm_center[hand_id] = current_palm_center.copy()
            self.prev_palm_velocity[hand_id] = np.zeros(3)
            self.prev_timestamp[hand_id] = timestamp
            return (0.0, 0.0, 0.0)

        dt = timestamp - self.prev_timestamp[hand_id]
        if dt <= 1e-4:
            vel = self.prev_palm_velocity[hand_id]
            return (float(vel[0]), float(vel[1]), float(vel[2]))

        raw_velocity = (current_palm_center - self.prev_palm_center[hand_id]) / dt
        prev_vel = self.prev_palm_velocity.get(hand_id, np.zeros(3))
        smoothed_velocity = self.velocity_alpha * raw_velocity + (1.0 - self.velocity_alpha) * prev_vel

        self.prev_palm_center[hand_id] = current_palm_center.copy()
        self.prev_palm_velocity[hand_id] = smoothed_velocity
        self.prev_timestamp[hand_id] = timestamp

        return (float(smoothed_velocity[0]), float(smoothed_velocity[1]), float(smoothed_velocity[2]))

    def reset_hand_tracking(self, hand_id: int):
        """Cleans up internal state when a hand leaves the tracking frame."""
        self.prev_palm_center.pop(hand_id, None)
        self.prev_palm_velocity.pop(hand_id, None)
        self.prev_timestamp.pop(hand_id, None)
