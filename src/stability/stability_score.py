"""
Gestura Stability Engine — Stability Score Calculator.

Calculates a multivariate Stability Score [0.0, 1.0] combining:
  1. Recognition confidence (raw landmark & classifier quality)
  2. Temporal consistency (agreement across rolling observation window)
  3. Motion consistency (linearity, absence of tremor/jitter)
  4. Pose persistence (dwell age and confirmation lifecycle)

Formula:
  Stability = w_rec * Recognition + w_temp * Temporal + w_mot * Motion + w_pers * Persistence

Commands are only eligible when Stability exceeds the configured threshold.
"""

from typing import Any, Dict, Optional, Tuple
from src.stability.stability_config import StabilityConfig


class StabilityScoreCalculator:
    """
    Computes multivariate stability scores with configurable weights and component breakdowns.
    """

    def __init__(self, config: Optional[StabilityConfig] = None):
        self.config = config or StabilityConfig()

    def compute(
        self,
        recognition_conf: float,
        temporal_consistency: float,
        motion_consistency: float,
        persistence_score: float,
    ) -> Tuple[float, bool, Dict[str, float]]:
        """
        Computes the weighted stability score:

        Returns:
            Tuple of:
              - stability_score (float in [0.0, 1.0])
              - is_eligible_for_command (bool)
              - component_breakdown (Dict[str, float])
        """
        rec = min(1.0, max(0.0, recognition_conf))
        temp = min(1.0, max(0.0, temporal_consistency))
        mot = min(1.0, max(0.0, motion_consistency))
        pers = min(1.0, max(0.0, persistence_score))

        w_rec = self.config.weight_recognition
        w_temp = self.config.weight_temporal
        w_mot = self.config.weight_motion
        w_pers = self.config.weight_persistence
        total_w = w_rec + w_temp + w_mot + w_pers

        if total_w > 0:
            score = (w_rec * rec + w_temp * temp + w_mot * mot + w_pers * pers) / total_w
        else:
            score = 0.0

        score = min(1.0, max(0.0, score))
        is_eligible = score >= self.config.stability_threshold

        breakdown = {
            "recognition": round(rec, 3),
            "temporal": round(temp, 3),
            "motion": round(mot, 3),
            "persistence": round(pers, 3),
            "overall_stability": round(score, 3),
            "threshold": self.config.stability_threshold,
            "is_eligible": is_eligible,
        }

        return score, is_eligible, breakdown
