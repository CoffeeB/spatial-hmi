"""
Leaky temporal evidence accumulator for intention confidence filtering.
"""

from typing import Dict
from src.gestures.gesture_types import GestureType


class TemporalIntentFilter:
    """
    Maintains a leaky exponential accumulator of gesture evidence:
    E_t(g) = lambda * E_{t-1}(g) + (1 - lambda) * c_t(g)
    Prevents single-frame spurious spikes from triggering unintended actions.
    """

    def __init__(self, evidence_lambda: float = 0.82):
        self.evidence_lambda = evidence_lambda
        self.evidence_accumulators: Dict[GestureType, float] = {g: 0.0 for g in GestureType}

    def update(self, detected_gesture: GestureType, instantaneous_conf: float) -> Dict[GestureType, float]:
        """
        Updates evidence for all gesture prototypes.
        The detected gesture receives instantaneous_conf, while others receive 0.0.
        """
        for g in GestureType:
            target_val = instantaneous_conf if g == detected_gesture else 0.0
            prev_val = self.evidence_accumulators.get(g, 0.0)
            new_val = self.evidence_lambda * prev_val + (1.0 - self.evidence_lambda) * target_val
            self.evidence_accumulators[g] = float(new_val)

        return self.evidence_accumulators.copy()

    def get_evidence(self, gesture: GestureType) -> float:
        """Returns current cumulative evidence for a specific gesture."""
        return self.evidence_accumulators.get(gesture, 0.0)

    def reset(self):
        """Resets all accumulated evidence to zero."""
        for g in self.evidence_accumulators:
            self.evidence_accumulators[g] = 0.0
