"""
Gestura Stability Engine — Observation Window Buffer Manager.

Maintains rolling frame observation histories across perception layers:
  - Finger: 3 frames
  - Pose: 5 frames
  - Motion: 8 frames
  - Gesture: 12 frames
  - Two-Hand: 16 frames
Configurable window lengths with majority voting, temporal consensus, and variance analysis.
"""

from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple
from src.stability.stability_config import StabilityConfig


class LayerObservationWindows:
    """
    Manages rolling frame buffers across all 5 perception layers for a single hand.
    """

    def __init__(self, config: Optional[StabilityConfig] = None):
        self.config = config or StabilityConfig()
        self.finger_buffer: deque = deque(maxlen=self.config.finger_window)
        self.pose_buffer: deque = deque(maxlen=self.config.pose_window)
        self.motion_buffer: deque = deque(maxlen=self.config.motion_window)
        self.gesture_buffer: deque = deque(maxlen=self.config.gesture_window)
        self.two_hand_buffer: deque = deque(maxlen=self.config.two_hand_window)

    def reset(self) -> None:
        """Clears all observation buffers."""
        self.finger_buffer.clear()
        self.pose_buffer.clear()
        self.motion_buffer.clear()
        self.gesture_buffer.clear()
        self.two_hand_buffer.clear()

    def update(
        self,
        finger_dict: Dict[str, str],
        pose_name: str,
        motion_primitive: str,
        gesture_name: str,
        timestamp: float,
        two_hand_gesture: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Appends instantaneous observations to rolling buffers and computes layer consensus ratios.

        Returns:
            Dict mapping layer name to consensus ratio [0.0, 1.0].
        """
        self.finger_buffer.append((finger_dict, timestamp))
        self.pose_buffer.append((pose_name, timestamp))
        self.motion_buffer.append((motion_primitive, timestamp))
        self.gesture_buffer.append((gesture_name, timestamp))
        if two_hand_gesture is not None:
            self.two_hand_buffer.append((two_hand_gesture, timestamp))

        consensus: Dict[str, float] = {}

        # 1. Pose Consensus
        if self.pose_buffer:
            poses = [p[0] for p in self.pose_buffer if p[0] not in ("NONE", "UNKNOWN")]
            if poses:
                top_pose, count = Counter(poses).most_common(1)[0]
                consensus["pose"] = count / len(self.pose_buffer)
            else:
                consensus["pose"] = 0.0
        else:
            consensus["pose"] = 0.0

        # 2. Motion Consensus
        if self.motion_buffer:
            prims = [m[0] for m in self.motion_buffer]
            top_m, count = Counter(prims).most_common(1)[0]
            consensus["motion"] = count / len(self.motion_buffer)
        else:
            consensus["motion"] = 0.0

        # 3. Gesture Consensus
        if self.gesture_buffer:
            gests = [g[0] for g in self.gesture_buffer if g[0] not in ("NONE", "IDLE")]
            if gests:
                top_g, count = Counter(gests).most_common(1)[0]
                consensus["gesture"] = count / len(self.gesture_buffer)
            else:
                consensus["gesture"] = 0.0
        else:
            consensus["gesture"] = 0.0

        return consensus

    def get_window_telemetry(self) -> Dict[str, Any]:
        """Returns window occupancy telemetry for developer inspection."""
        return {
            "finger": f"{len(self.finger_buffer)}/{self.config.finger_window}",
            "pose": f"{len(self.pose_buffer)}/{self.config.pose_window}",
            "motion": f"{len(self.motion_buffer)}/{self.config.motion_window}",
            "gesture": f"{len(self.gesture_buffer)}/{self.config.gesture_window}",
            "two_hand": f"{len(self.two_hand_buffer)}/{self.config.two_hand_window}",
        }
