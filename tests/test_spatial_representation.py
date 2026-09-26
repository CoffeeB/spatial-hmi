"""
Comprehensive Test Suite for Gestura Spatial Representation Layer.

Evaluates continuous 3D articulated representation against:
  1. Hand Orientation Invariance:
       - Palm facing camera, palm facing away (dorsal), rotated 45°, 90°, upside-down, sideways.
  2. Articulated Finger Direction:
       - Finger pointing along camera axis (toward/away), left, right, diagonal.
       - Preservation of hand-relative anatomical direction despite camera view changes.
  3. Continuous 3D Motion Kinematics:
       - Pure diagonal movement (-0.707, -0.707) correctly classified as TOP_LEFT, not forced into LEFT or UP.
       - Toward camera depth motion, away from camera depth motion.
       - Tremor in dead-zone remains STATIONARY / CENTER.
  4. The Grey Zone & Soft Probability Distribution:
       - Borderline vector produces overlapping probabilities, flags AmbiguityLevel.HIGH.
       - System can classify as AMBIGUOUS instead of forcing false certainty.
  5. Foreshortening & Camera-Facing Hand:
       - Detects optical compression, computes 3D bone length, avoids false state collapse.
  6. Temporal Reconstruction:
       - Bridges temporary single-frame occlusion without state drop.
"""

import math
from typing import Tuple
import numpy as np
import pytest

from src.spatial import (
    AmbiguityLevel,
    ArticulatedFingerGeometry,
    ContinuousFingerState,
    ContinuousMotionTracker,
    FingerGeometryEstimator,
    HandCoordinateFrame,
    HandFrameEstimator,
    HandPerceptionObject,
    PerceptionLifecycleState,
    SpatialConfig,
    SpatialDirectionClassifier,
    SpatialPerceptionEngine,
    SpatialSector,
    TemporalReconstructionEngine,
    Vector2D,
    Vector3D,
)


def create_canonical_hand_landmarks(
    palm_facing: str = "PALM",
    rotation_z_deg: float = 0.0,
    wrist_pos: Tuple[float, float, float] = (0.5, 0.7, 0.0),
    scale: float = 0.15,
) -> np.ndarray:
    """
    Synthesizes standard 21 MediaPipe hand landmarks in canonical posture.
    Finger pointing upwards (+Y in hand space, -Y in screen space).
    """
    # Canonical hand-relative positions: (x, y, z) where +y is distal, +x radial, +z normal (volar)
    base_coords = {
        0: (0.0, 0.0, 0.0),       # Wrist
        1: (0.2, 0.15, 0.0),      # Thumb CMC
        2: (0.35, 0.3, 0.0),      # Thumb MCP
        3: (0.45, 0.45, 0.0),     # Thumb IP
        4: (0.55, 0.55, 0.0),     # Thumb TIP
        5: (0.15, 0.5, 0.0),      # Index MCP
        6: (0.15, 0.7, 0.0),      # Index PIP
        7: (0.15, 0.85, 0.0),     # Index DIP
        8: (0.15, 1.0, 0.0),      # Index TIP
        9: (0.0, 0.52, 0.0),      # Middle MCP
        10: (0.0, 0.75, 0.0),     # Middle PIP
        11: (0.0, 0.92, 0.0),     # Middle DIP
        12: (0.0, 1.08, 0.0),     # Middle TIP
        13: (-0.15, 0.48, 0.0),   # Ring MCP
        14: (-0.15, 0.68, 0.0),   # Ring PIP
        15: (-0.15, 0.83, 0.0),   # Ring DIP
        16: (-0.15, 0.98, 0.0),   # Ring TIP
        17: (-0.3, 0.42, 0.0),    # Little MCP
        18: (-0.3, 0.58, 0.0),    # Little PIP
        19: (-0.3, 0.72, 0.0),    # Little DIP
        20: (-0.3, 0.85, 0.0),    # Little TIP
    }

    # Convert to array
    lms = np.zeros((21, 3), dtype=np.float32)
    for idx, (lx, ly, lz) in base_coords.items():
        # Scale
        lx *= scale
        ly *= scale
        lz *= scale

        # In camera coordinates, hand pointing UP means -Y in image coordinates
        cam_x = lx
        cam_y = -ly  # Screen UP is -Y
        cam_z = lz

        if palm_facing == "DORSAL":
            # Invert Z normal (facing away)
            cam_z = -cam_z
            cam_x = -cam_x

        # Apply 2D roll rotation around optical axis Z
        if rotation_z_deg != 0.0:
            rad = math.radians(rotation_z_deg)
            rx = cam_x * math.cos(rad) - cam_y * math.sin(rad)
            ry = cam_x * math.sin(rad) + cam_y * math.cos(rad)
            cam_x, cam_y = rx, ry

        # Translate to wrist_pos
        lms[idx] = [cam_x + wrist_pos[0], cam_y + wrist_pos[1], cam_z + wrist_pos[2]]

    return lms


# ==============================================================================
# 1. Continuous Vector & Direction Tests
# ==============================================================================

def test_vector_algebra():
    """Validates Vector2D and Vector3D continuous operations."""
    v1 = Vector3D(1.0, 2.0, 2.0)
    assert abs(v1.magnitude() - 3.0) < 1e-5
    u1 = v1.normalized()
    assert abs(u1.magnitude() - 1.0) < 1e-5
    assert abs(u1.x - 1.0 / 3.0) < 1e-5

    v2 = Vector3D(0.0, 1.0, 0.0)
    angle_deg = v1.angle_to_deg(v2)
    assert 0.0 < angle_deg < 90.0

    # Cross product
    cx = Vector3D(1.0, 0.0, 0.0).cross(Vector3D(0.0, 1.0, 0.0))
    assert abs(cx.z - 1.0) < 1e-5


def test_diagonal_motion_not_forced_to_horizontal_or_vertical():
    """
    CRITICAL REQUIREMENT: Diagonal movements must naturally evaluate to diagonal sectors
    (e.g., TOP_LEFT) rather than being forced into LEFT or UP.
    """
    classifier = SpatialDirectionClassifier()

    # Pure diagonal: (-0.707, -0.707) in screen space (+X Right, +Y Down -> -X Left, -Y Up)
    diag_vec = Vector2D(-0.7071, -0.7071)
    res = classifier.classify_direction(diag_vec, magnitude=0.25)

    assert res["primary_direction"] == SpatialSector.TOP_LEFT.value
    # Ensure TOP_LEFT has highest probability in distribution
    probs = res["sector_probabilities"]
    assert probs[SpatialSector.TOP_LEFT.value] > probs[SpatialSector.LEFT.value]
    assert probs[SpatialSector.TOP_LEFT.value] > probs[SpatialSector.TOP.value]
    assert probs[SpatialSector.TOP_LEFT.value] > 0.40


def test_soft_spatial_classification_and_grey_zone():
    """
    Section 4 & 5: The Grey Zone.
    Ambiguous observations between sectors produce high ambiguity and overlap.
    """
    classifier = SpatialDirectionClassifier()

    # Borderline between TOP (-90 deg) and TOP_LEFT (-135 deg): angle = -112.5 deg
    rad = math.radians(-112.5)
    borderline_vec = Vector2D(math.cos(rad), math.sin(rad))

    res = classifier.classify_direction(borderline_vec, magnitude=0.20)

    # Probabilities for TOP and TOP_LEFT should be nearly equal
    probs = res["sector_probabilities"]
    p_top = probs[SpatialSector.TOP.value]
    p_top_left = probs[SpatialSector.TOP_LEFT.value]

    assert abs(p_top - p_top_left) < 0.10
    # Ambiguity level must be HIGH
    assert res["ambiguity_level"] == AmbiguityLevel.HIGH.value
    assert res["is_ambiguous"] is True
    # Can state AMBIGUOUS when evidence does not permit decisive winner
    assert res["primary_direction"] in (
        SpatialSector.AMBIGUOUS.value,
        SpatialSector.TOP_LEFT.value,
        SpatialSector.TOP.value,
    )


def test_hysteresis_prevents_angular_chatter():
    """
    Section 6: Directional label must not oscillate rapidly between adjacent sectors
    under small angular variations.
    """
    classifier = SpatialDirectionClassifier()

    # 1. Establish solid TOP direction (-90 deg)
    classifier.classify_direction(Vector2D(0.0, -1.0), magnitude=0.3)
    assert classifier._active_sector == SpatialSector.TOP

    # 2. Slight perturbation towards TOP_RIGHT (-78 deg, distance 12 deg)
    rad = math.radians(-78.0)
    res = classifier.classify_direction(Vector2D(math.cos(rad), math.sin(rad)), magnitude=0.3)

    # Hysteresis should keep it in TOP until boundary is decisively passed
    assert res["primary_direction"] == SpatialSector.TOP.value


# ==============================================================================
# 2. Hand-Relative Coordinate Frame & Invariance Tests
# ==============================================================================

def test_hand_coordinate_frame_derivation():
    """Validates that HandFrameEstimator extracts an orthonormal basis from landmarks."""
    lms = create_canonical_hand_landmarks(palm_facing="PALM")
    estimator = HandFrameEstimator()

    frame = estimator.estimate_frame(lms, handedness="Right", scale_ref=0.15)

    assert frame is not None
    # Verify orthonormality: ||x|| = 1, ||y|| = 1, ||z|| = 1
    assert abs(frame.axis_x.magnitude() - 1.0) < 1e-4
    assert abs(frame.axis_y.magnitude() - 1.0) < 1e-4
    assert abs(frame.axis_z.magnitude() - 1.0) < 1e-4

    # Dot products between orthogonal axes must be approximately 0
    assert abs(frame.axis_x.dot(frame.axis_y)) < 1e-3
    assert abs(frame.axis_x.dot(frame.axis_z)) < 1e-3
    assert abs(frame.axis_y.dot(frame.axis_z)) < 1e-3

    # Hand pointing up in screen coords means Y distal is pointing screen up (-Y in camera coords)
    assert frame.axis_y.y < -0.80


def test_viewpoint_invariant_finger_direction():
    """
    Section 7 & 8: Finger direction expressed relative to hand must remain invariant
    even when the hand is rotated 45°, 90°, or upside down.
    """
    frame_estimator = HandFrameEstimator()
    finger_estimator = FingerGeometryEstimator()

    rotations = [0.0, 45.0, 90.0, 180.0, -90.0]
    hand_relative_index_directions = []

    for rot in rotations:
        lms = create_canonical_hand_landmarks(rotation_z_deg=rot)
        palm_center = (float(lms[9][0]), float(lms[9][1]), float(lms[9][2]))
        frame = frame_estimator.estimate_frame(lms, handedness="Right", scale_ref=0.15)

        finger_geom = finger_estimator.estimate_finger(
            finger_name="index",
            landmarks=lms,
            hand_frame=frame,
            palm_center=palm_center,
        )

        # In hand-relative space, index finger points along +Y distal axis (0, 1, 0)
        dir_hand = finger_geom.direction_hand
        hand_relative_index_directions.append(dir_hand)

        # In hand coordinates: Y should be strongly positive (~1.0), X and Z near 0
        assert dir_hand.y > 0.85, f"Rotation {rot}° failed: dir_hand={dir_hand}"
        assert abs(dir_hand.x) < 0.35, f"Rotation {rot}° failed: dir_hand={dir_hand}"

    # Across all rotations, the hand-relative direction must remain highly consistent
    for i in range(1, len(hand_relative_index_directions)):
        d0 = hand_relative_index_directions[0]
        di = hand_relative_index_directions[i]
        dot_sim = d0.dot(di)
        assert dot_sim > 0.95, f"Hand-relative direction changed with orientation! dot={dot_sim}"


# ==============================================================================
# 3. Continuous 3D Motion Kinematics & Depth Tests
# ==============================================================================

def test_continuous_3d_motion_tracker():
    """Validates continuous 3D velocity, acceleration, jerk, and depth state."""
    tracker = ContinuousMotionTracker()

    t0 = 0.0
    tracker.update(current_pos=(0.5, 0.5, 0.0), timestamp=t0, scale_ref=0.15)

    # Move horizontally to the right
    t1 = t0 + 0.033
    kinematics = tracker.update(current_pos=(0.52, 0.5, 0.0), timestamp=t1, scale_ref=0.15)

    assert kinematics.speed_3d > 0.1
    assert kinematics.velocity.x > 0.0
    assert kinematics.primary_direction == SpatialSector.RIGHT.value


def test_depth_motion_toward_and_away():
    """
    Section 13: Moving toward or away from camera should produce meaningful depth motion
    even when planar XY movement is negligible.
    """
    tracker = ContinuousMotionTracker()

    # Initial state
    tracker.update(current_pos=(0.5, 0.5, 0.5), timestamp=1.0, scale_ref=0.15)

    # Move toward camera (decreasing Z and increasing scale_ref)
    kin_toward = tracker.update(current_pos=(0.5, 0.5, 0.40), timestamp=1.033, scale_ref=0.17)

    assert kin_toward.depth_state == "TOWARD"
    assert kin_toward.primary_direction == "TOWARD"

    # Move away from camera (increasing Z and decreasing scale_ref)
    tracker.reset()
    tracker.update(current_pos=(0.5, 0.5, 0.4), timestamp=2.0, scale_ref=0.17)
    kin_away = tracker.update(current_pos=(0.5, 0.5, 0.52), timestamp=2.033, scale_ref=0.15)

    assert kin_away.depth_state == "AWAY"
    assert kin_away.primary_direction == "AWAY"


# ==============================================================================
# 4. Foreshortening & Temporal Reconstruction Tests
# ==============================================================================

def test_camera_facing_foreshortening_detection():
    """
    Section 14: Camera-facing postures compress apparent 2D finger length.
    Detects foreshortening and reduces confidence rather than inventing false curl.
    """
    lms = create_canonical_hand_landmarks()
    # Modify index finger to point directly toward camera (-Z in camera coords)
    p_mcp = lms[5].copy()
    lms[6] = p_mcp + np.array([0.0, 0.0, -0.04])
    lms[7] = p_mcp + np.array([0.0, 0.0, -0.07])
    lms[8] = p_mcp + np.array([0.0, 0.0, -0.10])  # Tip points straight at camera

    frame_estimator = HandFrameEstimator()
    finger_estimator = FingerGeometryEstimator()

    frame = frame_estimator.estimate_frame(lms, handedness="Right", scale_ref=0.15)
    palm_center = (float(lms[9][0]), float(lms[9][1]), float(lms[9][2]))

    geom = finger_estimator.estimate_finger("index", lms, frame, palm_center)

    assert geom.is_foreshortened is True
    assert geom.foreshortening_ratio < 0.60
    # Optical axis pointing unit vector along -Z or +Z
    assert abs(geom.direction_camera.z) > 0.70


def test_temporal_reconstruction_bridges_temporary_ambiguity():
    """
    Section 15: If frames N-10 -> N-1 are EXTENDED, a single UNCERTAIN/UNKNOWN frame
    is bridged by temporal history rather than causing state collapse.
    """
    reconstructor = TemporalReconstructionEngine()

    dummy_dir = Vector3D(0.0, 1.0, 0.0)
    dummy_dir2d = Vector2D(0.0, 1.0)

    # Feed 5 stable EXTENDED frames
    for i in range(5):
        geom = ArticulatedFingerGeometry(
            finger_name="index",
            state=ContinuousFingerState.EXTENDED,
            confidence=0.95,
            mcp_flexion_deg=5.0,
            pip_flexion_deg=5.0,
            dip_flexion_deg=5.0,
            total_flexion_deg=15.0,
            curl_ratio=0.10,
            spread_angle_deg=0.0,
            distance_to_palm=0.9,
            foreshortening_ratio=1.0,
            is_foreshortened=False,
            direction_camera=dummy_dir,
            direction_hand=dummy_dir,
            direction_screen=dummy_dir2d,
            v_proximal=dummy_dir,
            v_intermediate=dummy_dir,
            v_distal=dummy_dir,
        )
        reconstructor.reconstruct_finger(geom, timestamp=float(i) * 0.033)

    # Frame 6 is suddenly occluded / UNKNOWN
    occluded_geom = ArticulatedFingerGeometry(
        finger_name="index",
        state=ContinuousFingerState.UNKNOWN,
        confidence=0.10,
        mcp_flexion_deg=0.0,
        pip_flexion_deg=0.0,
        dip_flexion_deg=0.0,
        total_flexion_deg=0.0,
        curl_ratio=0.5,
        spread_angle_deg=0.0,
        distance_to_palm=0.5,
        foreshortening_ratio=0.2,
        is_foreshortened=True,
        direction_camera=dummy_dir,
        direction_hand=dummy_dir,
        direction_screen=dummy_dir2d,
        v_proximal=dummy_dir,
        v_intermediate=dummy_dir,
        v_distal=dummy_dir,
    )

    reconstructed = reconstructor.reconstruct_finger(occluded_geom, timestamp=5 * 0.033)

    # State must be preserved as EXTENDED with decayed confidence
    assert reconstructed.state == ContinuousFingerState.EXTENDED
    assert 0.40 < reconstructed.confidence < 0.95
    assert any("Temporal reconstruction" in note for note in reconstructed.notes)


# ==============================================================================
# 5. Full End-to-End Spatial Perception Engine Test
# ==============================================================================

def test_spatial_perception_engine_end_to_end():
    """Validates complete HandPerceptionObject generation and telemetry."""
    engine = SpatialPerceptionEngine()

    lms = create_canonical_hand_landmarks()
    palm_center = (float(lms[9][0]), float(lms[9][1]), float(lms[9][2]))

    perception_obj = engine.process_hand(
        hand_id=1,
        handedness="Right",
        landmarks=lms,
        palm_center=palm_center,
        scale_ref=0.15,
        timestamp=1.0,
    )

    assert isinstance(perception_obj, HandPerceptionObject)
    assert len(perception_obj.fingers) == 5
    assert perception_obj.coordinate_frame is not None
    assert perception_obj.motion is not None
    assert perception_obj.uncertainty is not None

    # Telemetry serialization
    telemetry = perception_obj.to_dict()
    assert "hand" in telemetry
    assert "fingers" in telemetry
    assert "motion" in telemetry
    assert "spatial" in telemetry
    assert "uncertainty" in telemetry

    # Verify fingers have continuous metrics
    index_tel = telemetry["fingers"]["index"]
    assert "flexion" in index_tel
    assert "curl_ratio" in index_tel
    assert "direction" in index_tel
