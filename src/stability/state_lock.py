"""
Gestura Stability Engine — State Locking System.

Preserves active interactions against transient noise and incidental finger shifts:
  - Once an interaction becomes ACTIVE, it requires stronger evidence to leave than to enter.
  - Temporary landmark instability, digit twitching, or occlusion will not terminate active manipulation.
  - Exits active lock only upon deliberate release gestures or sustained loss of confidence.
"""

from typing import Any, Dict, Optional, Tuple
from src.stability.stability_config import StabilityConfig


class StateLockSystem:
    """
    Manages asymmetric locking for active interactions.
    """

    def __init__(self, config: Optional[StabilityConfig] = None):
        self.config = config or StabilityConfig()
        self._is_locked: bool = False
        self._locked_interaction: str = "NONE"
        self._locked_hand_id: Optional[int] = None
        self._lock_timestamp: float = 0.0
        self._instability_frames: int = 0

    def is_locked(self) -> bool:
        """Returns True if an active interaction is currently preserved."""
        return self._is_locked

    def acquire_lock(self, interaction_name: str, hand_id: int, timestamp: float) -> None:
        """Locks into an active interaction."""
        self._is_locked = True
        self._locked_interaction = interaction_name
        self._locked_hand_id = hand_id
        self._lock_timestamp = timestamp
        self._instability_frames = 0

    def release_lock(self) -> None:
        """Releases the state lock."""
        self._is_locked = False
        self._locked_interaction = "NONE"
        self._locked_hand_id = None
        self._instability_frames = 0

    def update(
        self,
        is_active_interaction: bool,
        current_pose_name: str,
        detection_confidence: float,
        is_release_gesture: bool,
        timestamp: float,
    ) -> Tuple[bool, str]:
        """
        Evaluates active state lock against evidence:

        Returns:
            Tuple of:
              - is_locked (bool)
              - locked_interaction (str)
        """
        if not self._is_locked:
            if is_active_interaction and current_pose_name not in ("NONE", "UNKNOWN"):
                self.acquire_lock(current_pose_name, 0, timestamp)
            return self._is_locked, self._locked_interaction

        # When locked: check for deliberate release gesture
        if is_release_gesture or current_pose_name in ("OPEN_PALM", "RELEASE"):
            self.release_lock()
            return False, "NONE"

        # Check for sustained loss of confidence
        if detection_confidence < self.config.lock_exit_threshold:
            self._instability_frames += 1
            if self._instability_frames >= self.config.active_preservation_frames:
                # Exceeded grace preservation window: release lock
                self.release_lock()
                return False, "NONE"
        else:
            # Landmark confidence recovered: reset instability counter
            self._instability_frames = 0

        return True, self._locked_interaction

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns state lock telemetry for developer inspection."""
        return {
            "is_locked": self._is_locked,
            "locked_interaction": self._locked_interaction,
            "instability_frames": self._instability_frames,
            "grace_frames_max": self.config.active_preservation_frames,
        }
