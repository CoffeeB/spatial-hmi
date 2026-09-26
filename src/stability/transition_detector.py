"""
Gestura Stability Engine — Transition Detector.

Recognizes when the user is moving between poses:
  Example: Open Palm -> Curved -> Half Closed -> Closed Fist
  Outputs `TRANSITION` until the destination pose is confirmed and stable.
  Prevents intermediate frames from triggering unintended gestures.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from src.landmarks.finger_state import HandFingerStates
from src.stability.stability_config import StabilityConfig


class TransitionDetector:
    """
    Detects continuous anatomical hand transitions between canonical hand poses.
    """

    def __init__(self, config: Optional[StabilityConfig] = None):
        self.config = config or StabilityConfig()
        self._flexion_history: deque = deque(maxlen=self.config.pose_window * 2)
        self._is_transitioning: bool = False
        self._transition_start_time: float = 0.0
        self._from_pose: str = "NONE"
        self._candidate_to_pose: str = "NONE"
        self._transition_progress: float = 0.0

    def update(
        self,
        current_confirmed_pose: str,
        incoming_candidate_pose: Optional[str],
        finger_states: Optional[HandFingerStates],
        timestamp: float,
    ) -> Tuple[bool, str, str, float]:
        """
        Evaluates flexion velocity and pose candidates to detect an active transition.

        Returns:
            Tuple of:
              - is_transitioning (bool)
              - from_pose (str)
              - to_pose_candidate (str)
              - transition_progress (float in [0.0, 1.0])
        """
        # Calculate mean digit extension ratio if available
        if finger_states is not None:
            ratios = [
                finger_states.thumb.extension_ratio if finger_states.thumb else 1.0,
                finger_states.index.extension_ratio if finger_states.index else 1.0,
                finger_states.middle.extension_ratio if finger_states.middle else 1.0,
                finger_states.ring.extension_ratio if finger_states.ring else 1.0,
                finger_states.little.extension_ratio if finger_states.little else 1.0,
            ]
            mean_ext = sum(ratios) / len(ratios)
        else:
            mean_ext = 1.0

        self._flexion_history.append((mean_ext, timestamp))

        # Check if an alternate candidate pose is being observed
        is_candidate_present = (
            incoming_candidate_pose is not None
            and incoming_candidate_pose not in ("NONE", "UNKNOWN", current_confirmed_pose)
        )

        if is_candidate_present:
            if not self._is_transitioning:
                # Transition initiated
                self._is_transitioning = True
                self._transition_start_time = timestamp
                self._from_pose = current_confirmed_pose
                self._candidate_to_pose = incoming_candidate_pose or "NONE"
                self._transition_progress = 0.20
            else:
                # Transition continuing
                elapsed_ms = (timestamp - self._transition_start_time) * 1000.0
                target_ms = max(50.0, self.config.transition_min_duration_ms)
                self._transition_progress = min(0.95, 0.20 + 0.80 * (elapsed_ms / target_ms))
                self._candidate_to_pose = incoming_candidate_pose or "NONE"

                # Check timeout
                if elapsed_ms > self.config.transition_max_duration_ms:
                    self._is_transitioning = False
                    self._transition_progress = 1.0

        else:
            # No alternative candidate: either stable in current pose or transition complete
            if self._is_transitioning:
                self._is_transitioning = False
                self._transition_progress = 1.0
                self._from_pose = current_confirmed_pose
                self._candidate_to_pose = "NONE"

        return self._is_transitioning, self._from_pose, self._candidate_to_pose, self._transition_progress

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns transition telemetry for developer inspection."""
        return {
            "is_transitioning": self._is_transitioning,
            "from_pose": self._from_pose,
            "to_pose_candidate": self._candidate_to_pose,
            "transition_progress": round(self._transition_progress, 2),
        }
