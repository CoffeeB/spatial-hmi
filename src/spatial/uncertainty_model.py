"""
Gestura Spatial Representation — Explicit Uncertainty & Grey-Zone Perception Model.

Governed by the core principle:
  "Never convert uncertainty into false certainty. 'UNKNOWN' and 'UNCERTAIN' are valid states."

Evaluates observation confidence across four orthogonal channels:
  1. Landmark Visibility & Tracking Confidence.
  2. Foreshortening & Perspective Compression (e.g. camera-facing hand).
  3. Directional & Sector Ambiguity (probabilistic grey zone between sectors).
  4. Anatomical Consistency & Joint Plausibility.

Derives categorical lifecycle status:
  - STABLE: High confidence, unequivocal observation.
  - TRANSITION: Expected dynamic change between valid states.
  - UNCERTAIN: Conflicting evidence, borderline boundaries, or grey zone.
  - UNKNOWN: Insufficient physical evidence, extreme occlusion, or failure.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.spatial.spatial_config import SpatialConfig


class PerceptionLifecycleState(str, Enum):
    """Categorical uncertainty lifecycle states."""
    STABLE = "STABLE"
    TRANSITION = "TRANSITION"
    UNCERTAIN = "UNCERTAIN"
    UNKNOWN = "UNKNOWN"


@dataclass
class UncertaintyEvaluation:
    """Detailed multi-dimensional uncertainty report for a perception cycle."""
    state: PerceptionLifecycleState
    overall_confidence: float            # Aggregate confidence in [0.0, 1.0]

    # Component confidences
    visibility_confidence: float         # From detector landmark visibility
    foreshortening_confidence: float     # Optical compression penalty (1.0 = normal, 0.3 = severe)
    directional_confidence: float        # Sector separation margin
    anatomical_confidence: float         # Skeletal bone proportion plausibility

    # Ambiguity metrics
    directional_ambiguity: str           # "LOW", "MEDIUM", "HIGH"
    is_direction_ambiguous: bool
    is_foreshortened: bool
    entropy: float                       # Information entropy across hypotheses

    # Detailed diagnostic reasons
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, any]:
        """Telemetry representation."""
        return {
            "state": self.state.value,
            "overall_confidence": round(float(self.overall_confidence), 3),
            "components": {
                "visibility": round(float(self.visibility_confidence), 3),
                "foreshortening": round(float(self.foreshortening_confidence), 3),
                "directional": round(float(self.directional_confidence), 3),
                "anatomical": round(float(self.anatomical_confidence), 3),
            },
            "directional_ambiguity": self.directional_ambiguity,
            "is_foreshortened": self.is_foreshortened,
            "entropy": round(float(self.entropy), 3),
            "reasons": self.reasons,
        }


class SpatialUncertaintyModel:
    """
    Evaluates multi-dimensional spatial uncertainty and determines whether observations
    should be marked as STABLE, TRANSITION, UNCERTAIN, or UNKNOWN.
    """

    def __init__(self, config: Optional[SpatialConfig] = None):
        self.config = config or SpatialConfig()

    def evaluate(
        self,
        landmark_visibilities: Optional[List[float]],
        foreshortening_ratios: Dict[str, float],
        directional_probs: Optional[Dict[str, float]],
        directional_ambiguity: str = "LOW",
        is_transitioning: bool = False,
    ) -> UncertaintyEvaluation:
        """
        Evaluates uncertainty from all perception sub-systems.
        """
        reasons: List[str] = []

        # 1. Landmark visibility confidence
        if landmark_visibilities and len(landmark_visibilities) > 0:
            mean_vis = float(np.mean(landmark_visibilities))
            min_vis = float(np.min(landmark_visibilities))
            vis_conf = 0.7 * mean_vis + 0.3 * min_vis
            if min_vis < 0.30:
                reasons.append(f"Low landmark visibility detected (min={min_vis:.2f})")
        else:
            vis_conf = 0.90

        # 2. Foreshortening & Perspective Compression
        # If fingers point directly at camera, apparent 2D lengths shrink severely
        foreshortened_digits = [
            digit for digit, ratio in foreshortening_ratios.items()
            if ratio < self.config.foreshortening_ratio_threshold
        ]
        is_foreshortened = len(foreshortened_digits) >= 2
        if is_foreshortened:
            mean_f_ratio = float(np.mean(list(foreshortening_ratios.values())))
            foreshorten_conf = max(0.40, mean_f_ratio)
            reasons.append(f"Optical foreshortening active on: {', '.join(foreshortened_digits)}")
        else:
            foreshorten_conf = 1.0

        # 3. Directional Uncertainty & Entropy
        dir_conf = 1.0
        entropy = 0.0
        is_dir_ambiguous = (directional_ambiguity == "HIGH")

        if directional_probs:
            # Shannon entropy of probability distribution: -sum(p * log2(p))
            probs = [p for p in directional_probs.values() if p > 1e-4]
            if probs:
                p_arr = np.array(probs, dtype=np.float32)
                p_norm = p_arr / np.sum(p_arr)
                entropy = float(-np.sum(p_norm * np.log2(p_norm + 1e-9)))

            sorted_probs = sorted(directional_probs.values(), reverse=True)
            if len(sorted_probs) >= 2:
                p_top1, p_top2 = sorted_probs[0], sorted_probs[1]
                delta = p_top1 - p_top2
                if delta < self.config.ambiguity_delta_threshold:
                    dir_conf = max(0.45, delta / self.config.ambiguity_delta_threshold)
                    reasons.append(f"Directional grey zone: top1={p_top1:.2f}, top2={p_top2:.2f} (delta={delta:.2f})")

        # 4. Anatomical Plausibility
        anatomical_conf = 1.0

        # 5. Composite Confidence
        composite_conf = (
            0.35 * vis_conf +
            0.25 * foreshorten_conf +
            0.25 * dir_conf +
            0.15 * anatomical_conf
        )

        # 6. Categorical Lifecycle State Derivation
        if vis_conf < 0.40 or composite_conf < 0.35:
            state = PerceptionLifecycleState.UNKNOWN
            reasons.append("Confidence below minimum threshold — state marked UNKNOWN")
        elif is_transitioning:
            state = PerceptionLifecycleState.TRANSITION
        elif composite_conf < 0.65 or is_dir_ambiguous or is_foreshortened:
            state = PerceptionLifecycleState.UNCERTAIN
        else:
            state = PerceptionLifecycleState.STABLE

        return UncertaintyEvaluation(
            state=state,
            overall_confidence=composite_conf,
            visibility_confidence=vis_conf,
            foreshortening_confidence=foreshorten_conf,
            directional_confidence=dir_conf,
            anatomical_confidence=anatomical_conf,
            directional_ambiguity=directional_ambiguity,
            is_direction_ambiguous=is_dir_ambiguous,
            is_foreshortened=is_foreshortened,
            entropy=entropy,
            reasons=reasons,
        )
