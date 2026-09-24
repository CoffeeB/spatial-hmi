"""
Gestura v2 — Temporal Intent Filter.

Algorithm: Leaky exponential accumulator (LEA) per gesture class.

    E_t(g) = λ · E_{t-1}(g) + (1-λ) · c_t(g)

With λ=0.75:
  - A gesture needs ~4 consecutive high-confidence frames to reach the
    0.50 activation threshold from zero.
  - Dropping the gesture causes evidence to decay to 0 in ~12 frames.
  - Single-frame spikes contribute only 0.25 of their value → no false triggers.

v2 additions:
  - Competing gesture decay: when gesture g fires, all *other* gestures
    receive an accelerated decay factor (0.7×) to avoid co-activation.
  - Confidence telemetry snapshot exposed for debug overlay.
  - evidence_peak tracking: remembers the highest evidence seen per gesture
    for UI feedback ("how close was I?").
"""

from typing import Dict, Optional
from src.gestures.gesture_types import GestureType


class TemporalIntentFilter:
    """
    Maintains a leaky exponential accumulator of gesture evidence.

    Args:
        evidence_lambda:   Decay factor λ ∈ (0, 1). Higher = slower decay.
        competing_factor:  Extra decay applied to non-active gestures when
                           one gesture is strongly firing. Default 0.70.
        competing_thresh:  Evidence level above which competing decay applies.
    """

    def __init__(
        self,
        evidence_lambda: float = 0.75,
        competing_factor: float = 0.70,
        competing_thresh: float = 0.55,
    ):
        self.evidence_lambda = evidence_lambda
        self.competing_factor = competing_factor
        self.competing_thresh = competing_thresh

        self.evidence_accumulators: Dict[GestureType, float] = {
            g: 0.0 for g in GestureType
        }
        self.evidence_peak: Dict[GestureType, float] = {
            g: 0.0 for g in GestureType
        }

    def update(
        self,
        detected_gesture: GestureType,
        instantaneous_conf: float,
    ) -> Dict[GestureType, float]:
        """
        Updates evidence for all gesture classes.

        The detected gesture receives instantaneous_conf as its target.
        All other gestures decay toward 0.

        When the winning gesture's evidence exceeds competing_thresh, all
        non-winning gestures receive an additional competing_factor decay
        to suppress co-activation artefacts.

        Returns a copy of the evidence accumulator map.
        """
        winning_evidence = self.evidence_accumulators.get(detected_gesture, 0.0)
        apply_competing = (
            detected_gesture != GestureType.NONE
            and winning_evidence >= self.competing_thresh
        )

        for g in GestureType:
            target = instantaneous_conf if g == detected_gesture else 0.0
            prev = self.evidence_accumulators.get(g, 0.0)

            # Standard LEA update
            new_val = self.evidence_lambda * prev + (1.0 - self.evidence_lambda) * target

            # Competing gesture suppression
            if apply_competing and g != detected_gesture:
                new_val *= self.competing_factor

            new_val = float(max(0.0, min(1.0, new_val)))
            self.evidence_accumulators[g] = new_val

            # Track peak evidence for telemetry
            if new_val > self.evidence_peak.get(g, 0.0):
                self.evidence_peak[g] = new_val

        return self.evidence_accumulators.copy()

    def get_evidence(self, gesture: GestureType) -> float:
        """Returns current cumulative evidence for a specific gesture."""
        return self.evidence_accumulators.get(gesture, 0.0)

    def get_peak(self, gesture: GestureType) -> float:
        """Returns the maximum evidence ever seen for a gesture (since last reset)."""
        return self.evidence_peak.get(gesture, 0.0)

    def snapshot(self) -> Dict[str, float]:
        """Returns a flat dict of current evidence values for debug overlay."""
        return {
            f"ev_{g.value.lower()}": round(v, 3)
            for g, v in self.evidence_accumulators.items()
            if v > 0.01  # Only include non-trivial values
        }

    def reset(self):
        """Resets all accumulated evidence and peaks to zero."""
        for g in GestureType:
            self.evidence_accumulators[g] = 0.0
            self.evidence_peak[g] = 0.0
