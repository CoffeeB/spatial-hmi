"""
Intent Lock System for Gestura.

Once an interaction begins (e.g. Pinch selection, 3D manipulation, continuous dragging),
the Intent Lock locks interpretation to that ongoing interaction.
While locked:
  - Swipes, fists, hovers, and conflicting secondary gestures are suppressed.
  - Only gestures belonging to the active interaction family remain valid.
  - Unlock occurs only after explicit or graceful release.
"""

import time
from typing import Any, Dict, List, Optional, Set
from src.gestures.gesture_types import GestureType
from src.intent.temporal_intent_config import TemporalIntentConfig


class IntentLockSystem:
    """
    Manages exclusive interaction locking to prevent mid-interaction gesture hijacking.
    """

    # Families of mutually compatible gestures while locked
    COMPATIBLE_FAMILIES: Dict[str, Set[str]] = {
        "PINCH": {
            "G006_PINCH_SELECT",
            "G007_PINCH_DRAG",
            "G022_PINCH_ZOOM_IN",
            "G023_PINCH_ZOOM_OUT",
            "PINCH",
            "DRAG",
            "RELEASE",
            "OPEN_PALM",
        },
        "BIMANUAL": {
            "TWO_HAND_PINCH",
            "BIMANUAL_NAV",
            "TWO_HAND_SPREAD",
            "TWO_HAND_ROTATION",
            "RELEASE",
            "OPEN_PALM",
        },
        "DIAL": {
            "G018_1F_ROTATE_CW",
            "G018_2F_ROTATE_CW",
            "G018_3F_ROTATE_CW",
            "G018_4F_ROTATE_CW",
            "G018_5F_ROTATE_CW",
            "G019_1F_ROTATE_CCW",
            "G019_2F_ROTATE_CCW",
            "G019_3F_ROTATE_CCW",
            "G019_4F_ROTATE_CCW",
            "G019_5F_ROTATE_CCW",
            "RELEASE",
            "OPEN_PALM",
        },
    }

    def __init__(self, config: Optional[TemporalIntentConfig] = None):
        self.config = config or TemporalIntentConfig()
        self._is_locked: bool = False
        self._locked_by: Optional[str] = None
        self._locked_family: Optional[str] = None
        self._lock_target_id: Optional[str] = None
        self._lock_timestamp: float = 0.0
        self._last_interaction_timestamp: float = 0.0

    @property
    def is_locked(self) -> bool:
        return self._is_locked

    @property
    def locked_by(self) -> Optional[str]:
        return self._locked_by

    @property
    def lock_target_id(self) -> Optional[str]:
        return self._lock_target_id

    def reset(self):
        self._is_locked = False
        self._locked_by = None
        self._locked_family = None
        self._lock_target_id = None
        self._lock_timestamp = 0.0
        self._last_interaction_timestamp = 0.0

    def acquire_lock(
        self,
        gesture_name: str,
        target_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Attempts to acquire intent lock for the given gesture interaction.
        """
        now = timestamp or time.time()
        family = self._resolve_family(gesture_name)

        if not self._is_locked:
            self._is_locked = True
            self._locked_by = gesture_name
            self._locked_family = family
            self._lock_target_id = target_id
            self._lock_timestamp = now
            self._last_interaction_timestamp = now
            return True
        elif self.is_gesture_allowed(gesture_name):
            # Refresh lock
            self._last_interaction_timestamp = now
            if target_id is not None:
                self._lock_target_id = target_id
            return True

        return False

    def release_lock(self, timestamp: Optional[float] = None) -> bool:
        """Releases the active intent lock. Returns True if a lock was active."""
        was_locked = self._is_locked
        self.reset()
        return was_locked

    def is_gesture_allowed(self, candidate_gesture: str) -> bool:
        """
        Checks if candidate_gesture is permitted under the active lock.
        If locked by PINCH, suppresses SWIPE, FIST, HOVER, etc.
        """
        if not self._is_locked:
            return True

        # Check safety timeout
        now = time.time()
        if (now - self._last_interaction_timestamp) > self.config.lock.timeout_sec:
            self.release_lock(now)
            return True

        cand_str = str(candidate_gesture).upper()
        if self._locked_family and self._locked_family in self.COMPATIBLE_FAMILIES:
            allowed = self.COMPATIBLE_FAMILIES[self._locked_family]
            return any(a in cand_str or cand_str in a for a in allowed)

        return cand_str == str(self._locked_by).upper()

    def filter_candidate(self, candidate_gesture: str) -> str:
        """
        Filters a candidate gesture: returns the gesture if allowed,
        or suppresses it (returns NONE) if blocked by the active lock.
        """
        if self.is_gesture_allowed(candidate_gesture):
            return candidate_gesture
        return "NONE"

    def get_telemetry(self) -> Dict[str, Any]:
        """Provides telemetry for debug overlays and WebSocket clients."""
        return {
            "is_locked": self._is_locked,
            "locked_by": self._locked_by or "NONE",
            "lock_family": self._locked_family or "NONE",
            "target_id": self._lock_target_id or "NONE",
            "lock_duration_sec": round(time.time() - self._lock_timestamp, 2) if self._is_locked else 0.0,
        }

    def _resolve_family(self, gesture_name: str) -> str:
        g = str(gesture_name).upper()
        if "PINCH" in g or "DRAG" in g or "ZOOM" in g:
            return "PINCH"
        elif "TWO_HAND" in g or "BIMANUAL" in g:
            return "BIMANUAL"
        elif "ROTATE" in g or "DIAL" in g:
            return "DIAL"
        return "GENERAL"
