"""
Gestura Spatial Representation — Master Spatial Perception Engine.

Orchestrates the entire continuous 3D spatial perception pipeline:
  RAW LANDMARKS
        ↓
  HAND COORDINATE FRAME (Hand-Relative Basis)
        ↓
  ARTICULATED FINGER GEOMETRY (Kinematic Chains & Multi-Space Vectors)
        ↓
  TEMPORAL RECONSTRUCTION (Occlusion & Ambiguity Bridging)
        ↓
  CONTINUOUS 3D MOTION TRACKING (Kinematics, Depth, Jerk, 8-Sector Probabilities)
        ↓
  SPATIAL UNCERTAINTY MODEL (Visibility, Foreshortening, Ambiguity, Grey-Zone)
        ↓
  HAND PERCEPTION OBJECT (Unified Continuous State)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from src.spatial.finger_geometry import ArticulatedFingerGeometry, FingerGeometryEstimator
from src.spatial.hand_frame import HandCoordinateFrame, HandFrameEstimator
from src.spatial.motion_vector import ContinuousMotionKinematics, ContinuousMotionTracker
from src.spatial.perception_object import HandPerceptionObject
from src.spatial.spatial_config import SpatialConfig
from src.spatial.spatial_direction import SpatialDirectionClassifier, SpatialSector
from src.spatial.spatial_vector import Vector2D, Vector3D
from src.spatial.temporal_reconstruction import TemporalReconstructionEngine
from src.spatial.uncertainty_model import PerceptionLifecycleState, SpatialUncertaintyModel, UncertaintyEvaluation


class SpatialPerceptionEngine:
    """
    Coordinator engine providing continuous 3D articulated perception for Gestura.
    Decouples raw sensor observations from downstream gesture interpretation.
    """

    def __init__(self, config: Optional[SpatialConfig] = None):
        self.config = config or SpatialConfig()

        # Component estimators
        self.frame_estimator = HandFrameEstimator(self.config)
        self.finger_estimator = FingerGeometryEstimator(self.config)
        self.temporal_reconstructor = TemporalReconstructionEngine(self.config)
        self.uncertainty_model = SpatialUncertaintyModel(self.config)

        # Per-hand motion trackers: hand_id -> ContinuousMotionTracker
        self._motion_trackers: Dict[int, ContinuousMotionTracker] = {}

    def reset(self):
        """Resets all internal sub-system states."""
        self.finger_estimator.reset()
        self.temporal_reconstructor.reset()
        self._motion_trackers.clear()

    def prune_missing_hands(self, active_hand_ids: List[int]):
        """Prunes motion history for hands no longer in frame."""
        for hid in list(self._motion_trackers.keys()):
            if hid not in active_hand_ids:
                del self._motion_trackers[hid]

    def process_hand(
        self,
        hand_id: int,
        handedness: str,
        landmarks: np.ndarray,
        palm_center: Tuple[float, float, float],
        scale_ref: float,
        timestamp: float,
        landmark_visibilities: Optional[List[float]] = None,
        is_transitioning: bool = False,
    ) -> HandPerceptionObject:
        """
        Processes a single hand observation into a complete HandPerceptionObject.
        """
        # 1. Hand Anatomical Coordinate Frame
        coord_frame = self.frame_estimator.estimate_frame(
            landmarks=landmarks,
            handedness=handedness,
            scale_ref=scale_ref,
        )

        # 2. Articulated Finger Geometry (Kinematic chains & hand-relative vectors)
        finger_geometries = self.finger_estimator.estimate_all_fingers(
            landmarks=landmarks,
            hand_frame=coord_frame,
            palm_center=palm_center,
            visibilities=landmark_visibilities,
        )

        # 3. Temporal Reconstruction (Graceful continuity across temporary blurs)
        finger_geometries = self.temporal_reconstructor.reconstruct_all_fingers(
            finger_geometries=finger_geometries,
            timestamp=timestamp,
        )

        # 4. Continuous 3D Motion Kinematics
        if hand_id not in self._motion_trackers:
            self._motion_trackers[hand_id] = ContinuousMotionTracker(self.config)
        motion_tracker = self._motion_trackers[hand_id]
        motion_kinematics = motion_tracker.update(
            current_pos=palm_center,
            timestamp=timestamp,
            scale_ref=scale_ref,
        )

        # 5. Foreshortening ratios across all digits
        foreshortening_ratios = {
            name: geom.foreshortening_ratio
            for name, geom in finger_geometries.items()
        }

        # 6. Spatial Uncertainty Evaluation
        uncertainty_eval = self.uncertainty_model.evaluate(
            landmark_visibilities=landmark_visibilities,
            foreshortening_ratios=foreshortening_ratios,
            directional_probs=motion_kinematics.spatial_distribution,
            directional_ambiguity=motion_kinematics.ambiguity_level,
            is_transitioning=is_transitioning,
        )

        # 7. Assemble Unified HandPerceptionObject
        palm_center_vec = Vector3D(palm_center[0], palm_center[1], palm_center[2])
        screen_dir = motion_kinematics.direction_2d
        hand_dir = coord_frame.camera_to_hand_vector(motion_kinematics.direction_3d.as_tuple())

        return HandPerceptionObject(
            hand_id=hand_id,
            handedness=handedness,
            timestamp=timestamp,
            coordinate_frame=coord_frame,
            palm_center=palm_center_vec,
            palm_normal=coord_frame.axis_z,
            hand_confidence=uncertainty_eval.overall_confidence,
            fingers=finger_geometries,
            motion=motion_kinematics,
            screen_direction=screen_dir,
            hand_direction=hand_dir,
            depth_direction=motion_kinematics.depth_state,
            zone_distribution=motion_kinematics.spatial_distribution,
            primary_sector=motion_kinematics.primary_direction,
            secondary_sector=motion_kinematics.secondary_direction,
            ambiguity_level=motion_kinematics.ambiguity_level,
            uncertainty=uncertainty_eval,
        )
