"""
Gestura Stability Engine — Intentional Hold Tracker.

Ensures gestures require a minimum hold dwell duration before becoming stable:
  - Point: 80 ms
  - Pinch: 120 ms
  - Fist (Closed Grab): 150 ms
  - Open Palm Anchor: 100 ms
  - Crucial HCI Rule: The hold timer begins ONLY after the pose becomes confirmed and stable!
"""

from typing import Any, Dict, Optional, Tuple
from src.stability.stability_config import StabilityConfig


class IntentionalHoldTracker:
    """
    Tracks and enforces intentional hold dwell requirements for gestures.
    """

    def __init__(self, config: Optional[StabilityConfig] = None):
        self.config = config or StabilityConfig()
        self._current_stable_pose: Optional[str] = None
        self._hold_start_timestamp: Optional[float] = None
        self._hold_duration_ms: float = 0.0
        self._is_satisfied: bool = False

    def reset(self) -> None:
        """Resets the hold timer."""
        self._current_stable_pose = None
        self._hold_start_timestamp = None
        self._hold_duration_ms = 0.0
        self._is_satisfied = False

    def update(
        self,
        stable_pose_name: str,
        is_pose_confirmed: bool,
        is_in_dead_zone_or_still: bool,
        timestamp: float,
    ) -> Tuple[bool, float, float]:
        """
        Updates the intentional hold timer:

        Args:
            stable_pose_name: Canonical name of the confirmed stable pose.
            is_pose_confirmed: True if the pose persistence manager reports confirmed stability.
            is_in_dead_zone_or_still: True if hand velocity is low / in dead zone.
            timestamp: Frame timestamp in seconds.

        Returns:
            Tuple of:
              - is_satisfied (bool): True if hold duration >= required target
              - current_hold_duration_ms (float)
              - target_hold_ms (float)
        """
        target_ms = self.config.get_hold_duration_for_pose(stable_pose_name)

        # Rule: Timer begins ONLY after the pose is confirmed and stable!
        if not is_pose_confirmed or stable_pose_name in ("NONE", "UNKNOWN"):
            self.reset()
            return False, 0.0, target_ms

        if self._current_stable_pose != stable_pose_name:
            # New stable pose established: start timer!
            self._current_stable_pose = stable_pose_name
            self._hold_start_timestamp = timestamp
            self._hold_duration_ms = 0.0
            self._is_satisfied = False

        elif is_in_dead_zone_or_still:
            # Continuing to hold steadily
            if self._hold_start_timestamp is not None:
                self._hold_duration_ms = (timestamp - self._hold_start_timestamp) * 1000.0
            else:
                self._hold_start_timestamp = timestamp
                self._hold_duration_ms = 0.0

            if self._hold_duration_ms >= target_ms:
                self._is_satisfied = True

        else:
            # Moving rapidly: do not accumulate static hold timer
            # (allow slight grace without immediately resetting to 0)
            if self._hold_start_timestamp is not None:
                elapsed = (timestamp - self._hold_start_timestamp) * 1000.0
                if elapsed > target_ms:
                    self._is_satisfied = True

        return self._is_satisfied, self._hold_duration_ms, target_ms

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns hold timer telemetry for developer inspection."""
        target = self.config.get_hold_duration_for_pose(self._current_stable_pose or "NONE")
        return {
            "pose": self._current_stable_pose or "NONE",
            "hold_duration_ms": round(self._hold_duration_ms, 1),
            "target_hold_ms": round(target, 1),
            "is_hold_satisfied": self._is_satisfied,
        }
