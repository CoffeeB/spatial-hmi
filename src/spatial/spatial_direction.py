"""
Gestura Spatial Representation — Continuous Spatial Direction & Uncertainty Model.

Eliminates discrete black-and-white 4-way classification (LEFT/RIGHT/UP/DOWN).
Instead:
  1. Represents direction as continuous normalized vectors (dx, dy) and continuous angles.
  2. Evaluates continuous 8-directional sectors:
       TOP, TOP-RIGHT, RIGHT, BOTTOM-RIGHT, BOTTOM, BOTTOM-LEFT, LEFT, TOP-LEFT, CENTER.
  3. Computes soft, overlapping probability distributions across all sectors using circular Gaussian kernels.
  4. Explicitly models "The Grey Zone" and ambiguity:
       Identifies primary & secondary directions, confidence delta, and ambiguity level (LOW/MED/HIGH).
       Can state: "AMBIGUOUS" or "I don't have enough evidence."
  5. Applies angular hysteresis and vector smoothing to prevent boundary chatter.
"""

from enum import Enum
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.spatial.spatial_config import SpatialConfig
from src.spatial.spatial_vector import Vector2D, Vector3D


class SpatialSector(str, Enum):
    """Derived 8-direction screen space sectors plus CENTER, AMBIGUOUS, and UNKNOWN."""
    TOP = "TOP"
    TOP_RIGHT = "TOP_RIGHT"
    RIGHT = "RIGHT"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"
    BOTTOM = "BOTTOM"
    BOTTOM_LEFT = "BOTTOM_LEFT"
    LEFT = "LEFT"
    TOP_LEFT = "TOP_LEFT"
    CENTER = "CENTER"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"


class AmbiguityLevel(str, Enum):
    """Categorical ambiguity rating for spatial decisions."""
    LOW = "LOW"         # Clear separation between primary and secondary candidates
    MEDIUM = "MEDIUM"   # Moderate proximity to sector boundary
    HIGH = "HIGH"       # Genuinely ambiguous observation in the grey zone


# Central angle in degrees for each 8-direction sector in standard screen coords (+X Right, +Y Down)
SECTOR_CENTERS_DEG: Dict[SpatialSector, float] = {
    SpatialSector.RIGHT: 0.0,
    SpatialSector.BOTTOM_RIGHT: 45.0,
    SpatialSector.BOTTOM: 90.0,
    SpatialSector.BOTTOM_LEFT: 135.0,
    SpatialSector.LEFT: 180.0,       # Also -180.0
    SpatialSector.TOP_LEFT: -135.0,
    SpatialSector.TOP: -90.0,
    SpatialSector.TOP_RIGHT: -45.0,
}


class SpatialDirectionClassifier:
    """
    Evaluates continuous 2D and 3D directional vectors into soft probability distributions,
    detecting ambiguous grey-zone observations and applying angular hysteresis.
    """

    def __init__(self, config: Optional[SpatialConfig] = None):
        self.config = config or SpatialConfig()
        self._active_sector: SpatialSector = SpatialSector.CENTER
        self._smoothed_vector: Vector2D = Vector2D(0.0, 0.0)
        self._last_angle_deg: float = 0.0
        self._accumulated_displacement: float = 0.0

    def reset(self):
        """Resets temporal smoothing and hysteresis."""
        self._active_sector = SpatialSector.CENTER
        self._smoothed_vector = Vector2D(0.0, 0.0)
        self._last_angle_deg = 0.0
        self._accumulated_displacement = 0.0

    @staticmethod
    def _shortest_angular_distance_deg(angle1_deg: float, angle2_deg: float) -> float:
        """Computes the shortest angular distance on circle [-180, 180] in degrees [0, 180]."""
        diff = (angle1_deg - angle2_deg + 180.0) % 360.0 - 180.0
        return abs(diff)

    def classify_direction(
        self,
        raw_direction: Vector2D,
        magnitude: Optional[float] = None,
        dt: float = 0.033,
    ) -> Dict[str, any]:
        """
        Classifies continuous vector into 8-sector soft probability distribution with ambiguity rating.

        Args:
            raw_direction: Continuous 2D direction vector (e.g. from displacement or velocity)
            magnitude: Optional magnitude (if None, raw_direction.magnitude is used)
            dt: Frame time delta

        Returns:
            Dict containing:
              - continuous_vector: (x, y)
              - continuous_angle_deg: float
              - primary_sector: SpatialSector
              - secondary_sector: Optional[SpatialSector]
              - primary_confidence: float [0, 1]
              - ambiguity_level: AmbiguityLevel
              - is_ambiguous: bool
              - sector_distribution: Dict[str, float]
              - classification: SpatialSector (returns AMBIGUOUS if grey zone)
        """
        mag = magnitude if magnitude is not None else raw_direction.magnitude

        # 1. Dead-Zone check: small movements remain CENTER
        if mag < self.config.dead_zone_radius_2d:
            self._smoothed_vector = self._smoothed_vector.lerp(Vector2D(0.0, 0.0), 0.3)
            self._active_sector = SpatialSector.CENTER
            dist_map = {sec.value.lower(): 0.0 for sec in SECTOR_CENTERS_DEG}
            dist_map["center"] = 1.0
            res = {
                "continuous_vector": (0.0, 0.0),
                "continuous_angle_deg": 0.0,
                "primary_sector": SpatialSector.CENTER,
                "secondary_sector": None,
                "primary_direction": SpatialSector.CENTER.value,
                "secondary_direction": None,
                "primary_confidence": 1.0 - float(mag / max(self.config.dead_zone_radius_2d, 1e-4)),
                "confidence": 1.0 - float(mag / max(self.config.dead_zone_radius_2d, 1e-4)),
                "ambiguity_level": AmbiguityLevel.LOW.value,
                "is_ambiguous": False,
                "sector_distribution": dist_map,
                "sector_probabilities": {**dist_map, **{k.upper(): v for k, v in dist_map.items()}},
                "classification": SpatialSector.CENTER,
            }
            return res

        # 2. Continuous Vector Smoothing
        unit_vec = raw_direction.normalized
        alpha = self.config.vector_smoothing_alpha
        self._smoothed_vector = self._smoothed_vector.lerp(unit_vec, alpha).normalized
        curr_angle_deg = self._smoothed_vector.angle_deg
        self._last_angle_deg = curr_angle_deg

        # 3. Soft Spatial Classification: Compute circular Gaussian probability for all 8 sectors
        sigma_deg = self.config.sector_bandwidth_deg / 2.0
        sector_scores: Dict[SpatialSector, float] = {}

        for sector, center_deg in SECTOR_CENTERS_DEG.items():
            ang_dist = self._shortest_angular_distance_deg(curr_angle_deg, center_deg)
            # Gaussian bell curve centered on sector heading
            prob = math.exp(-0.5 * (ang_dist / max(sigma_deg, 1e-3)) ** 2)
            sector_scores[sector] = float(prob)

        # Normalize probability distribution over 8 sectors so sum = 1.0
        sum_scores = sum(sector_scores.values())
        if sum_scores > 1e-6:
            sector_distribution = {sec.value.lower(): float(score / sum_scores) for sec, score in sector_scores.items()}
        else:
            sector_distribution = {sec.value.lower(): 0.125 for sec in sector_scores}
        sector_distribution["center"] = 0.0

        # 4. Rank sectors to find Primary and Secondary
        ranked_sectors = sorted(sector_scores.keys(), key=lambda s: sector_scores[s], reverse=True)
        top1_sector = ranked_sectors[0]
        top2_sector = ranked_sectors[1]

        top1_prob = sector_distribution[top1_sector.value.lower()]
        top2_prob = sector_distribution[top2_sector.value.lower()]
        delta_prob = top1_prob - top2_prob

        # 5. Model The Grey Zone (Ambiguity Detection)
        if delta_prob < self.config.ambiguity_delta_threshold:
            ambiguity = AmbiguityLevel.HIGH
            is_ambiguous = True
        elif delta_prob < (self.config.ambiguity_delta_threshold * 1.8):
            ambiguity = AmbiguityLevel.MEDIUM
            is_ambiguous = False
        else:
            ambiguity = AmbiguityLevel.LOW
            is_ambiguous = False

        # 6. Apply Angular Hysteresis to State Transitions
        # If currently locked into an active sector, require a significant angular barrier
        # before switching to a neighboring sector.
        candidate_sector = top1_sector
        if self._active_sector in SECTOR_CENTERS_DEG and candidate_sector != self._active_sector:
            active_center = SECTOR_CENTERS_DEG[self._active_sector]
            dist_to_active = self._shortest_angular_distance_deg(curr_angle_deg, active_center)
            candidate_center = SECTOR_CENTERS_DEG[candidate_sector]
            dist_to_candidate = self._shortest_angular_distance_deg(curr_angle_deg, candidate_center)

            # To exit active sector, distance to candidate must exceed active by hysteresis barrier
            if (dist_to_active - dist_to_candidate) < self.config.angular_hysteresis_deg:
                candidate_sector = self._active_sector

        self._active_sector = candidate_sector

        # If high ambiguity exists, classification can be explicitly AMBIGUOUS
        final_classification = SpatialSector.AMBIGUOUS if (is_ambiguous and delta_prob < 0.08) else candidate_sector

        dual_distribution = {**sector_distribution, **{k.upper(): v for k, v in sector_distribution.items()}}
        return {
            "continuous_vector": self._smoothed_vector.as_tuple(),
            "continuous_angle_deg": round(float(curr_angle_deg), 1),
            "primary_sector": candidate_sector,
            "secondary_sector": top2_sector,
            "primary_direction": candidate_sector.value,
            "secondary_direction": top2_sector.value if top2_sector else None,
            "primary_confidence": round(float(top1_prob), 3),
            "secondary_confidence": round(float(top2_prob), 3),
            "confidence": round(float(top1_prob), 3),
            "ambiguity_level": ambiguity.value if hasattr(ambiguity, "value") else str(ambiguity),
            "is_ambiguous": is_ambiguous,
            "sector_distribution": sector_distribution,
            "sector_probabilities": dual_distribution,
            "classification": final_classification,
        }
