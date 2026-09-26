"""
Finger-Centric Perception Layer — Fingertip Tracker.

Treats each fingertip as a first-class, independently tracked perception object
with its own position, velocity, acceleration, direction, and confidence.
"""
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np

# MediaPipe landmark indices for fingertips
FINGER_TIP_INDICES: Dict[str, int] = {
    "thumb":  4,
    "index":  8,
    "middle": 12,
    "ring":   16,
    "little": 20,
}

# MediaPipe landmark indices for MCPs (proximal knuckle — base of each finger)
FINGER_MCP_INDICES: Dict[str, int] = {
    "thumb":  1,
    "index":  5,
    "middle": 9,
    "ring":   13,
    "little": 17,
}

FINGER_NAMES: List[str] = ["thumb", "index", "middle", "ring", "little"]


class FingerMotionClass(str, Enum):
    """Continuous motion observation class for a single fingertip."""
    STATIC       = "STATIC"        # Velocity below threshold
    MOVING       = "MOVING"        # General movement
    APPROACHING  = "APPROACHING"   # Moving toward camera (dz < 0 in MediaPipe)
    RECEDING     = "RECEDING"      # Moving away from camera (dz > 0)
    EXTENDING    = "EXTENDING"     # Extension ratio increasing over time
    FLEXING      = "FLEXING"       # Extension ratio decreasing over time
    TRANSLATING  = "TRANSLATING"   # Moving laterally relative to palm
    ACCELERATING = "ACCELERATING"  # Speed increasing
    DECELERATING = "DECELERATING"  # Speed decreasing
    UNKNOWN      = "UNKNOWN"


@dataclass
class FingerTipState:
    """Per-fingertip state snapshot."""
    finger_name: str
    position_3d: np.ndarray             # (3,) raw camera-space coordinates
    position_2d: Tuple[float, float]    # (x, y) normalized image coords
    velocity_3d: np.ndarray             # (3,) units / second
    acceleration_3d: np.ndarray         # (3,) units / second²
    speed: float                        # Scalar speed ||velocity_3d||
    direction_3d: np.ndarray            # (3,) unit vector of motion direction
    depth: float                        # z-component (< 0 = toward camera)
    confidence: float                   # [0, 1]
    visibility: float                   # Raw MediaPipe visibility [0, 1]
    tracking_age: int                   # Consecutive frames tracked
    motion_class: FingerMotionClass
    is_camera_facing: bool              # Dominant motion/direction is along z
    foreshortening_ratio: float         # apparent_2d / chain_3d, 0 = end-on
    timestamp: float


class FingerTipTracker:
    """
    Tracks each fingertip as an independent 3D kinematic entity.

    Maintains position / velocity / acceleration per fingertip using a rolling
    history buffer. Classifies each fingertip's motion state independently of
    palm motion, enabling detection of isolated finger movements.
    """

    def __init__(
        self,
        history_frames: int = 12,
        static_speed_threshold: float = 0.008,
        depth_motion_threshold: float = 0.010,
    ):
        self.history_frames = history_frames
        self.static_speed_threshold = static_speed_threshold
        self.depth_motion_threshold = depth_motion_threshold

        # Per-finger rolling position/timestamp history
        self._pos_history: Dict[str, Deque[Tuple[np.ndarray, float]]] = {
            f: deque(maxlen=history_frames) for f in FINGER_NAMES
        }
        self._tracking_ages: Dict[str, int] = {f: 0 for f in FINGER_NAMES}
        self._prev_speeds: Dict[str, float] = {f: 0.0 for f in FINGER_NAMES}
        self._prev_ext_ratios: Dict[str, float] = {f: 1.0 for f in FINGER_NAMES}

    def update(
        self,
        raw_landmarks: np.ndarray,               # (21, 3)
        d_ref: float,
        timestamp: float,
        visibilities: Optional[List[float]] = None,
        ext_ratios: Optional[Dict[str, float]] = None,
    ) -> Dict[str, FingerTipState]:
        """Update all fingertip trackers and return current per-finger states."""
        states: Dict[str, FingerTipState] = {}

        for finger_name in FINGER_NAMES:
            tip_idx = FINGER_TIP_INDICES[finger_name]
            mcp_idx = FINGER_MCP_INDICES[finger_name]
            tip_pos = raw_landmarks[tip_idx].copy()
            vis = float(visibilities[tip_idx]) if visibilities else 1.0

            history = self._pos_history[finger_name]

            # Velocity, acceleration, direction from history
            velocity_3d = np.zeros(3, dtype=np.float32)
            acceleration_3d = np.zeros(3, dtype=np.float32)
            speed = 0.0
            direction_3d = np.array([0.0, -1.0, 0.0], dtype=np.float32)

            if len(history) >= 1:
                prev_pos, prev_ts = history[-1]
                dt = max(timestamp - prev_ts, 1e-4)
                velocity_3d = ((tip_pos - prev_pos) / dt).astype(np.float32)
                speed = float(np.linalg.norm(velocity_3d))
                if speed > 1e-6:
                    direction_3d = (velocity_3d / speed).astype(np.float32)

            if len(history) >= 2:
                prev2_pos, prev2_ts = history[-2]
                prev1_pos, prev1_ts = history[-1]
                dt1 = max(prev1_ts - prev2_ts, 1e-4)
                dt2 = max(timestamp - prev1_ts, 1e-4)
                vel_prev = ((prev1_pos - prev2_pos) / dt1).astype(np.float32)
                acceleration_3d = ((velocity_3d - vel_prev) / dt2).astype(np.float32)

            history.append((tip_pos.copy(), timestamp))
            self._tracking_ages[finger_name] += 1

            # Foreshortening ratio: compare apparent 2D tip-MCP span to true 3D distance
            foreshorten_ratio = self._foreshortening(raw_landmarks, tip_idx, mcp_idx)

            # Camera-facing: motion direction dominated by z component
            is_camera_facing = bool(
                abs(float(direction_3d[2])) > 0.55
                and abs(float(direction_3d[2])) > abs(float(direction_3d[0])) * 1.4
                and abs(float(direction_3d[2])) > abs(float(direction_3d[1])) * 1.4
            )

            # Motion classification
            motion_class = self._classify_motion(
                finger_name, speed, velocity_3d, ext_ratios
            )

            # Confidence: visibility × age-ramp × foreshortening penalty
            age_factor = min(self._tracking_ages[finger_name] / 10.0, 1.0)
            conf = float(np.clip(
                vis * (0.35 + 0.65 * age_factor) * max(foreshorten_ratio, 0.12),
                0.0, 1.0
            ))

            self._prev_speeds[finger_name] = speed

            states[finger_name] = FingerTipState(
                finger_name=finger_name,
                position_3d=tip_pos.astype(np.float32),
                position_2d=(float(tip_pos[0]), float(tip_pos[1])),
                velocity_3d=velocity_3d,
                acceleration_3d=acceleration_3d,
                speed=speed,
                direction_3d=direction_3d,
                depth=float(tip_pos[2]),
                confidence=conf,
                visibility=vis,
                tracking_age=self._tracking_ages[finger_name],
                motion_class=motion_class,
                is_camera_facing=is_camera_facing,
                foreshortening_ratio=foreshorten_ratio,
                timestamp=timestamp,
            )

        return states

    def _classify_motion(
        self,
        finger_name: str,
        speed: float,
        velocity_3d: np.ndarray,
        ext_ratios: Optional[Dict[str, float]],
    ) -> FingerMotionClass:
        if speed < self.static_speed_threshold:
            return FingerMotionClass.STATIC

        prev_speed = self._prev_speeds.get(finger_name, speed)
        dz = float(velocity_3d[2])

        # Depth motion takes priority for camera-facing fingers
        if abs(dz) > self.depth_motion_threshold and abs(dz) > speed * 0.45:
            return FingerMotionClass.APPROACHING if dz < 0 else FingerMotionClass.RECEDING

        # Extension / flexion from ratio change
        if ext_ratios is not None:
            # Map "little" to "pinky" for compatibility with KinematicFeatureExtractor
            feat_key = "pinky" if finger_name == "little" else finger_name
            curr_ratio = ext_ratios.get(feat_key, 1.0)
            prev_ratio = self._prev_ext_ratios.get(finger_name, curr_ratio)
            delta = curr_ratio - prev_ratio
            self._prev_ext_ratios[finger_name] = curr_ratio
            if delta > 0.06:
                return FingerMotionClass.EXTENDING
            if delta < -0.06:
                return FingerMotionClass.FLEXING

        # Acceleration / deceleration
        if speed > prev_speed * 1.45 and speed > self.static_speed_threshold * 2.5:
            return FingerMotionClass.ACCELERATING
        if speed < prev_speed * 0.55 and prev_speed > self.static_speed_threshold * 2.5:
            return FingerMotionClass.DECELERATING

        return FingerMotionClass.MOVING

    @staticmethod
    def _foreshortening(
        raw_landmarks: np.ndarray,
        tip_idx: int,
        mcp_idx: int,
    ) -> float:
        """
        Returns the foreshortening ratio of the tip-MCP segment.
        1.0 = full lateral extension (no compression).
        0.0 = finger pointing directly at/away from camera (severe foreshortening).
        """
        tip = raw_landmarks[tip_idx]
        mcp = raw_landmarks[mcp_idx]
        apparent_2d = float(np.linalg.norm(tip[:2] - mcp[:2]))
        true_3d = float(np.linalg.norm(tip - mcp))
        if true_3d < 1e-5:
            return 1.0
        return float(np.clip(apparent_2d / true_3d, 0.0, 1.0))

    def reset_finger(self, finger_name: str) -> None:
        self._pos_history[finger_name].clear()
        self._tracking_ages[finger_name] = 0
        self._prev_speeds[finger_name] = 0.0
        self._prev_ext_ratios[finger_name] = 1.0

    def reset_all(self) -> None:
        for f in FINGER_NAMES:
            self.reset_finger(f)
