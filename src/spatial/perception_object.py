"""
Gestura Spatial Representation — Master Hand Perception Object.

Decouples raw observation from discrete interpretation by unifying:
  1. Hand 3D pose, anatomical coordinate frame, and continuous orientation.
  2. Articulated 3D finger kinematics (multi-space direction, flexion, curl, foreshortening).
  3. Continuous 3D motion kinematics (velocity, acceleration, jerk, depth speed, trajectory).
  4. Soft spatial direction model (8 sectors, circular Gaussian probability, ambiguity rating).
  5. Multi-dimensional uncertainty and stability lifecycle context.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from src.spatial.finger_geometry import ArticulatedFingerGeometry
from src.spatial.hand_frame import HandCoordinateFrame
from src.spatial.motion_vector import ContinuousMotionKinematics
from src.spatial.spatial_vector import Vector2D, Vector3D
from src.spatial.uncertainty_model import UncertaintyEvaluation


@dataclass
class HandPerceptionObject:
    """
    Unified continuous 3D spatial representation of a tracked hand.
    Serves as the single structured interface between perception and gesture interpretation.
    """
    hand_id: int
    handedness: str
    timestamp: float

    # Hand anatomical geometry & orientation
    coordinate_frame: HandCoordinateFrame
    palm_center: Vector3D
    palm_normal: Vector3D
    hand_confidence: float

    # Articulated digits: 'thumb', 'index', 'middle', 'ring', 'little'
    fingers: Dict[str, ArticulatedFingerGeometry]

    # Continuous 3D motion kinematics
    motion: ContinuousMotionKinematics

    # Spatial direction model
    screen_direction: Vector2D
    hand_direction: Vector3D
    depth_direction: str
    zone_distribution: Dict[str, float]
    primary_sector: str
    secondary_sector: Optional[str]
    ambiguity_level: str

    # Explicit uncertainty evaluation
    uncertainty: UncertaintyEvaluation

    # Downstream stability & gesture contexts
    stability_state: str = "STABLE"
    stability_score: float = 1.0
    derived_pose: Optional[str] = None
    discrete_gesture: Optional[str] = None
    active_representation: str = "HAND_CENTRIC"
    viewpoint_mode: str = "NORMAL"
    finger_centric: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, any]:
        """
        Serializes perception object into structured format matching Gestura Perception Specification.
        """
        return {
            "hand": {
                "hand_id": self.hand_id,
                "handedness": self.handedness,
                "position": self.palm_center.as_tuple(),
                "orientation": {
                    "pitch_deg": round(float(self.coordinate_frame.pitch_deg), 1),
                    "yaw_deg": round(float(self.coordinate_frame.yaw_deg), 1),
                    "roll_deg": round(float(self.coordinate_frame.roll_deg), 1),
                    "palm_facing": self.coordinate_frame.palm_facing,
                },
                "axes": {
                    "origin": self.coordinate_frame.origin.as_tuple(),
                    "x_radial": self.coordinate_frame.axis_x.as_tuple(),
                    "y_distal": self.coordinate_frame.axis_y.as_tuple(),
                    "z_normal": self.coordinate_frame.axis_z.as_tuple(),
                },
                "confidence": round(float(self.hand_confidence), 3),
            },
            "fingers": {
                name: geom.to_dict() for name, geom in self.fingers.items()
            },
            "motion": self.motion.to_dict(),
            "spatial": {
                "screen_direction": self.screen_direction.as_tuple(),
                "hand_direction": self.hand_direction.as_tuple(),
                "depth_direction": self.depth_direction,
                "zone_distribution": {k: round(v, 3) for k, v in self.zone_distribution.items()},
                "primary_sector": self.primary_sector,
                "secondary_sector": self.secondary_sector,
                "ambiguity_level": self.ambiguity_level,
            },
            "uncertainty": self.uncertainty.to_dict(),
            "stability": {
                "state": self.stability_state,
                "score": round(float(self.stability_score), 3),
            },
            "interpretation": {
                "derived_pose": self.derived_pose,
                "discrete_gesture": self.discrete_gesture,
            },
            "active_representation": self.active_representation,
            "viewpoint_mode": self.viewpoint_mode,
            "finger_centric": self.finger_centric or {},
        }
