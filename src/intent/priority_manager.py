"""
Gesture Priority Manager for Gestura.

Implements the strict 5-tier interaction priority hierarchy:
  Priority 1: Pinch on selected node / Precision Pinch (Highest)
  Priority 2: Two-hand manipulation (Bimanual zoom / rotate / pan)
  Priority 3: Swipe gestures (1F to 5F directional strokes)
  Priority 4: Point hover & cursor targeting
  Priority 5: Idle / resting hand (Lowest)

Lower-priority gestures are strictly suppressed while higher-priority interactions are candidate or active.
"""

from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple
from src.intent.temporal_intent_config import TemporalIntentConfig


class GesturePriorityTier(IntEnum):
    """Integer priority values: lower number = higher priority."""
    TIER_1_PINCH_SELECT = 1
    TIER_2_TWO_HAND_MANIP = 2
    TIER_3_SWIPE = 3
    TIER_4_POINT_HOVER = 4
    TIER_5_IDLE = 5


class GesturePriorityManager:
    """
    Arbitrates conflicts between multiple simultaneous or overlapping gesture candidates.
    """

    def __init__(self, config: Optional[TemporalIntentConfig] = None):
        self.config = config or TemporalIntentConfig()
        self._current_active_tier: GesturePriorityTier = GesturePriorityTier.TIER_5_IDLE
        self._active_gesture_name: str = "IDLE"

    def reset(self):
        self._current_active_tier = GesturePriorityTier.TIER_5_IDLE
        self._active_gesture_name = "IDLE"

    def get_tier(self, gesture_name: str) -> GesturePriorityTier:
        """Categorizes any gesture into its canonical priority tier."""
        g = str(gesture_name).upper()
        if "TWO_HAND" in g or "BIMANUAL" in g:
            return GesturePriorityTier.TIER_2_TWO_HAND_MANIP
        elif "PINCH" in g or "DRAG" in g or "GRAB" in g or "ZOOM" in g or "FM00" in g:
            return GesturePriorityTier.TIER_1_PINCH_SELECT
        elif "SWIPE" in g or "FLICK" in g:
            return GesturePriorityTier.TIER_3_SWIPE
        elif "POINT" in g or "HOVER" in g or "TAP" in g:
            return GesturePriorityTier.TIER_4_POINT_HOVER
        return GesturePriorityTier.TIER_5_IDLE

    def arbitrate(
        self,
        candidate_gesture: str,
        current_state_is_active: bool = False,
    ) -> Tuple[bool, str, GesturePriorityTier]:
        """
        Arbitrates whether candidate_gesture is allowed to proceed or should be suppressed.
        Returns:
            (is_allowed, winner_gesture, winner_tier)
        """
        cand_tier = self.get_tier(candidate_gesture)

        if not current_state_is_active:
            # When system is not in an active high-priority interaction, accept candidate
            self._current_active_tier = cand_tier
            self._active_gesture_name = candidate_gesture
            return True, candidate_gesture, cand_tier

        # When an interaction is currently ACTIVE:
        # A new candidate can only preempt if it has STRICTLY HIGHER priority (lower numerical tier)
        if cand_tier < self._current_active_tier:
            # Preemption by higher priority (e.g. Pinch pre-empting Point hover)
            self._current_active_tier = cand_tier
            self._active_gesture_name = candidate_gesture
            return True, candidate_gesture, cand_tier
        elif cand_tier == self._current_active_tier:
            # Same priority family allowed to maintain interaction
            self._active_gesture_name = candidate_gesture
            return True, candidate_gesture, cand_tier
        else:
            # Lower priority is suppressed! (e.g. Swipe attempted while Pinch is active)
            return False, self._active_gesture_name, self._current_active_tier

    def release_active(self):
        """Releases the current active tier back to idle."""
        self._current_active_tier = GesturePriorityTier.TIER_5_IDLE
        self._active_gesture_name = "IDLE"

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "active_tier": self._current_active_tier.name,
            "active_tier_rank": int(self._current_active_tier),
            "active_gesture": self._active_gesture_name,
        }
