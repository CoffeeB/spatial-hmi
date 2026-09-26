"""
Gestura Spatial Representation Subsystem.

Provides continuous, 3D viewpoint-invariant articulated spatial perception:
  - Vector2D, Vector3D: Continuous spatial algebra
  - SpatialConfig: Physical thresholds & configurations
  - HandCoordinateFrame, HandFrameEstimator: Anatomical hand-relative basis
  - SpatialDirectionClassifier, SpatialSector, AmbiguityLevel: Soft 8-direction model & grey zone
  - ArticulatedFingerGeometry, FingerGeometryEstimator: Continuous joint kinematics & vectors
  - ContinuousMotionKinematics, ContinuousMotionTracker: 3D velocity, acceleration, jerk & depth
  - SpatialUncertaintyModel, PerceptionLifecycleState: Explicit uncertainty & state evaluation
  - TemporalReconstructionEngine: Multi-frame occlusion & ambiguity bridging
  - HandPerceptionObject: Master continuous representation object
  - SpatialPerceptionEngine: Coordinator perception engine
"""

from src.spatial.finger_geometry import (
    ArticulatedFingerGeometry,
    ContinuousFingerState,
    FingerGeometryEstimator,
)
from src.spatial.hand_frame import (
    HandCoordinateFrame,
    HandFrameEstimator,
)
from src.spatial.motion_vector import (
    ContinuousMotionKinematics,
    ContinuousMotionTracker,
)
from src.spatial.perception_object import (
    HandPerceptionObject,
)
from src.spatial.spatial_config import (
    SpatialConfig,
)
from src.spatial.spatial_direction import (
    AmbiguityLevel,
    SpatialDirectionClassifier,
    SpatialSector,
)
from src.spatial.spatial_engine import (
    SpatialPerceptionEngine,
)
from src.spatial.spatial_vector import (
    Vector2D,
    Vector3D,
)
from src.spatial.temporal_reconstruction import (
    TemporalReconstructionEngine,
)
from src.spatial.uncertainty_model import (
    PerceptionLifecycleState,
    SpatialUncertaintyModel,
    UncertaintyEvaluation,
)

__all__ = [
    "Vector2D",
    "Vector3D",
    "SpatialConfig",
    "HandCoordinateFrame",
    "HandFrameEstimator",
    "SpatialDirectionClassifier",
    "SpatialSector",
    "AmbiguityLevel",
    "ContinuousFingerState",
    "ArticulatedFingerGeometry",
    "FingerGeometryEstimator",
    "ContinuousMotionKinematics",
    "ContinuousMotionTracker",
    "PerceptionLifecycleState",
    "UncertaintyEvaluation",
    "SpatialUncertaintyModel",
    "TemporalReconstructionEngine",
    "HandPerceptionObject",
    "SpatialPerceptionEngine",
]
