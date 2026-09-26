"""
Gestura Stability Engine — Micro-Adjustment Filter.

Distinguishes between:
  - MICRO_ADJUSTMENT: Natural tremor, digit twitches, wrist repositioning, landmark flicker -> IGNORE
  - TRANSITION: Deliberate movement toward another pose -> OBSERVE
  - INTENTIONAL: Stable completed action -> ALLOW

Suppresses micro-adjustments so they never produce commands.
"""

from collections import deque
import math
from typing import Any, Dict, List, Optional, Tuple
from src.stability.stability_config import StabilityConfig
from src.stability.stability_lifecycle import MovementCategory


class MicroAdjustmentFilter:
    """
    Detects and classifies micro-adjustments versus deliberate human intent.
    """

    def __init__(self, config: Optional[StabilityConfig] = None):
        self.config = config or StabilityConfig()
        self._pos_history: deque = deque(maxlen=self.config.motion_window)
        self._speed_history: deque = deque(maxlen=self.config.motion_window)
        self._finger_twitch_counters: Dict[str, int] = {}
        self._last_category: MovementCategory = MovementCategory.INTENTIONAL
        self._suppression_reasons: List[str] = []

    def evaluate(
        self,
        palm_pos: Tuple[float, float, float],
        palm_speed: float,
        detection_confidence: float,
        finger_changes: Dict[str, str],
        pose_changed: bool,
        is_transitioning: bool,
        is_in_dead_zone: bool,
        timestamp: float,
    ) -> Tuple[MovementCategory, List[str]]:
        """
        Evaluates current kinematics and perception signals to classify the movement category.

        Returns:
            Tuple of (MovementCategory, list of diagnostic explanation strings)
        """
        reasons: List[str] = []
        self._pos_history.append((palm_pos, timestamp))
        self._speed_history.append(palm_speed)

        # ── 1. Single-Frame / Brief Confidence Drop ─────────────────────────
        if detection_confidence < 0.45:
            reasons.append(f"Landmark confidence drop ({detection_confidence:.2f} < 0.45)")
            self._last_category = MovementCategory.MICRO_ADJUSTMENT
            self._suppression_reasons = reasons
            return MovementCategory.MICRO_ADJUSTMENT, reasons

        # ── 2. Transition Detector has priority over micro-adjustment ────────
        if is_transitioning:
            reasons.append("Deliberate multi-digit pose transition in progress")
            self._last_category = MovementCategory.TRANSITION
            self._suppression_reasons = reasons
            return MovementCategory.TRANSITION, reasons

        # ── 3. Spatial Palm Tremor Filter ────────────────────────────────────
        # High instantaneous speed, but net displacement across the observation window is near zero
        if len(self._pos_history) >= self.config.motion_window:
            p_old, _ = self._pos_history[0]
            p_curr, _ = self._pos_history[-1]
            net_disp = math.sqrt(
                (p_curr[0] - p_old[0]) ** 2
                + (p_curr[1] - p_old[1]) ** 2
                + (p_curr[2] - p_old[2]) ** 2
            )
            mean_speed = sum(self._speed_history) / len(self._speed_history)

            # Physiological tremor check: speed is moderate/high but hand goes nowhere
            if mean_speed > self.config.tremor_speed_threshold and net_disp < self.config.tremor_max_displacement:
                reasons.append(f"Palm tremor detected (speed: {mean_speed:.2f}, net disp: {net_disp:.4f} < {self.config.tremor_max_displacement})")
                self._last_category = MovementCategory.MICRO_ADJUSTMENT
                self._suppression_reasons = reasons
                return MovementCategory.MICRO_ADJUSTMENT, reasons

        # ── 4. Isolated Single-Digit Twitch Filter ───────────────────────────
        # Exactly one finger reports a changed state while all 4 other fingers remain unchanged
        if finger_changes and len(finger_changes) == 1 and not pose_changed:
            twitching_digit = next(iter(finger_changes.keys()))
            self._finger_twitch_counters[twitching_digit] = self._finger_twitch_counters.get(twitching_digit, 0) + 1
            if self._finger_twitch_counters[twitching_digit] <= 2:
                reasons.append(f"Isolated digit twitch on {twitching_digit} (frame {self._finger_twitch_counters[twitching_digit]}/3)")
                self._last_category = MovementCategory.MICRO_ADJUSTMENT
                self._suppression_reasons = reasons
                return MovementCategory.MICRO_ADJUSTMENT, reasons
        else:
            self._finger_twitch_counters.clear()

        # ── 5. Intentional Movement ──────────────────────────────────────────
        self._last_category = MovementCategory.INTENTIONAL
        self._suppression_reasons = []
        return MovementCategory.INTENTIONAL, []

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns filter diagnostics for developer overlay."""
        return {
            "category": self._last_category.value,
            "is_micro_adjustment": self._last_category == MovementCategory.MICRO_ADJUSTMENT,
            "is_transition": self._last_category == MovementCategory.TRANSITION,
            "is_intentional": self._last_category == MovementCategory.INTENTIONAL,
            "suppression_reasons": self._suppression_reasons,
        }
