"""
Tests for Gestura Stability Engine.

Validates:
  1. Finger State Persistence & Lifecycle State Machine
  2. Pose Persistence & Asymmetric Hysteresis
  3. Motion Dead Zone (Spatial filtering, distance scaling, smooth ramp)
  4. Micro-Adjustment Filter (Tremor and twitch suppression)
  5. Transition Detector (Continuous pose morphing)
  6. Intentional Hold Tracker (Hold dwell only begins after pose confirmation)
  7. State Lock System (Preservation of active interactions)
  8. Stability Score Calculator (Multivariate weighting)
  9. Stability Engine End-to-End Orchestration
 10. Interaction Engine Command Gating & State Lock Integration
"""

import time
import numpy as np
import pytest

from src.gestures.gesture_types import GestureType
from src.gestures.hand_pose import DerivedHandPose, FingerConfiguration, HandPoseId
from src.intent.state_machine import IntentContext, InteractionState
from src.interaction.engine import InteractionEngine
from src.interaction.mapping import SpatialCommand, SpatialCommandType
from src.landmarks.finger_state import FingerName, FingerStateDetail, FingerStateEnum, HandFingerStates
from src.landmarks.hand_state import HandState
from src.motion.motion_primitive import DynamicState, MotionPrimitive, MotionState
from src.stability.finger_persistence import FingerPersistenceManager
from src.stability.intentional_hold import IntentionalHoldTracker
from src.stability.micro_adjustment_filter import MicroAdjustmentFilter
from src.stability.motion_deadzone import MotionDeadZone
from src.stability.observation_windows import LayerObservationWindows
from src.stability.pose_persistence import PosePersistenceManager
from src.stability.stability_config import StabilityConfig
from src.stability.stability_engine import StabilityEngine
from src.stability.stability_lifecycle import MovementCategory, StabilityLifecycleState
from src.stability.stability_score import StabilityScoreCalculator
from src.stability.state_lock import StateLockSystem
from src.stability.transition_detector import TransitionDetector


# ─────────────────────────────────────────────────────────────────────────────
# Helper Fixtures & Builders
# ─────────────────────────────────────────────────────────────────────────────

def make_finger_states(default_state: FingerStateEnum = FingerStateEnum.EXTENDED) -> HandFingerStates:
    return HandFingerStates(
        thumb=FingerStateDetail(finger=FingerName.THUMB, state=default_state, confidence=0.98),
        index=FingerStateDetail(finger=FingerName.INDEX, state=default_state, confidence=0.98),
        middle=FingerStateDetail(finger=FingerName.MIDDLE, state=default_state, confidence=0.98),
        ring=FingerStateDetail(finger=FingerName.RING, state=default_state, confidence=0.98),
        little=FingerStateDetail(finger=FingerName.LITTLE, state=default_state, confidence=0.98),
    )


def make_finger_config() -> FingerConfiguration:
    return FingerConfiguration(
        thumb=FingerStateEnum.EXTENDED,
        index=FingerStateEnum.EXTENDED,
        middle=FingerStateEnum.EXTENDED,
        ring=FingerStateEnum.EXTENDED,
        little=FingerStateEnum.EXTENDED,
        num_extended_fingers=4,
        num_folded_fingers=0,
        num_curved_fingers=0,
        num_uncertain_fingers=0,
        thumb_is_extended=True,
        thumb_is_folded=False,
        thumb_is_pinching=False,
    )


def make_derived_pose(pose_id: HandPoseId, name: str, conf: float = 0.95) -> DerivedHandPose:
    return DerivedHandPose(
        pose_id=pose_id,
        canonical_name=name,
        confidence=conf,
        configuration=make_finger_config(),
    )


def make_motion_state(prim: MotionPrimitive = MotionPrimitive.STATIONARY, vel=(0.0, 0.0, 0.0), speed=0.0) -> MotionState:
    return MotionState(
        hand_id=0,
        handedness="Right",
        position=(0.5, 0.5, 0.0),
        velocity=vel,
        speed=speed,
        speed_xy=speed,
        acceleration=(0.0, 0.0, 0.0),
        acceleration_magnitude=0.0,
        tangential_acceleration=0.0,
        dynamic_state=DynamicState.STATIONARY,
        direction_vector=(0.0, 0.0, 0.0),
        heading_deg=0.0,
        primary_direction="STATIONARY",
        net_displacement=(0.0, 0.0, 0.0),
        displacement_magnitude=0.0,
        cumulative_path_length=0.0,
        linearity=1.0,
        scale_rate=0.0,
        depth_state="NEUTRAL",
        angular_velocity=0.0,
        cumulative_angle_deg=0.0,
        rotation_direction="NONE",
        stroke_duration_ms=0.0,
        dwell_duration_ms=500.0,
        is_holding=True,
        is_releasing=False,
        motion_primitive=prim,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Finger State Persistence & Lifecycle State Machine
# ─────────────────────────────────────────────────────────────────────────────

def test_finger_persistence_lifecycle():
    config = StabilityConfig(finger_window=3)
    fpm = FingerPersistenceManager(config)

    # Initially extended
    open_hand = make_finger_states(FingerStateEnum.EXTENDED)
    stab, persistent = fpm.update(open_hand, detection_confidence=0.98, timestamp=1.0)
    assert persistent["Index"].state == "extended"
    assert persistent["Index"].lifecycle == StabilityLifecycleState.STABLE

    # Single bent frame -> POSSIBLE_CHANGE, but state remains extended!
    bent_hand = make_finger_states(FingerStateEnum.EXTENDED)
    bent_hand.index.state = FingerStateEnum.FOLDED
    stab, persistent = fpm.update(bent_hand, detection_confidence=0.98, timestamp=1.033)
    assert persistent["Index"].state == "extended", "Single frame must not change stable state"
    assert persistent["Index"].lifecycle == StabilityLifecycleState.POSSIBLE_CHANGE

    # Second bent frame -> OBSERVING
    stab, persistent = fpm.update(bent_hand, detection_confidence=0.98, timestamp=1.066)
    assert persistent["Index"].state == "extended", "Second frame still observing"
    assert persistent["Index"].lifecycle == StabilityLifecycleState.OBSERVING

    # Returns to extended on 3rd frame -> Back to STABLE, no transition occurred!
    stab, persistent = fpm.update(open_hand, detection_confidence=0.98, timestamp=1.099)
    assert persistent["Index"].state == "extended"
    assert persistent["Index"].lifecycle == StabilityLifecycleState.STABLE


def test_finger_persistence_confirmation():
    config = StabilityConfig(finger_window=3)
    fpm = FingerPersistenceManager(config)

    open_hand = make_finger_states(FingerStateEnum.EXTENDED)
    fpm.update(open_hand, detection_confidence=0.98, timestamp=1.0)

    bent_hand = make_finger_states(FingerStateEnum.EXTENDED)
    bent_hand.index.state = FingerStateEnum.FOLDED

    # Feed 3 consecutive bent frames to satisfy finger_window=3
    fpm.update(bent_hand, timestamp=1.033)
    fpm.update(bent_hand, timestamp=1.066)
    stab, persistent = fpm.update(bent_hand, timestamp=1.099)

    assert persistent["Index"].state == "folded"
    assert persistent["Index"].lifecycle == StabilityLifecycleState.CONFIRMED_CHANGE


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pose Persistence & Asymmetric Hysteresis
# ─────────────────────────────────────────────────────────────────────────────

def test_pose_persistence_hysteresis():
    config = StabilityConfig(pose_window=3, enter_threshold=0.88, exit_threshold=0.62)
    ppm = PosePersistenceManager(config)

    # Frame 1-3: OPEN_PALM with confidence 0.92 (> 0.88)
    pose_open = make_derived_pose(HandPoseId.H001_OPEN_PALM, "OPEN_PALM", conf=0.92)
    ppm.update(pose_open, timestamp=1.0)
    ppm.update(pose_open, timestamp=1.033)
    stab_pose, persistent = ppm.update(pose_open, timestamp=1.066)

    assert persistent.canonical_name == "OPEN_PALM"
    assert persistent.confidence >= 0.88

    # Frame 4: Uncertain frame drops to 0.70 (below enter_threshold 0.88, but ABOVE exit_threshold 0.62)
    pose_uncertain = make_derived_pose(HandPoseId.H001_OPEN_PALM, "OPEN_PALM", conf=0.70)
    stab_pose, persistent = ppm.update(pose_uncertain, timestamp=1.099)

    # Hysteresis prevents collapse!
    assert persistent.canonical_name == "OPEN_PALM", "Uncertain frame above exit_threshold must not collapse pose"

    # Frame 5: One frame drop to UNKNOWN with confidence 0.20
    pose_drop = make_derived_pose(HandPoseId.UNKNOWN, "UNKNOWN", conf=0.20)
    stab_pose, persistent = ppm.update(pose_drop, timestamp=1.132)
    assert persistent.canonical_name == "OPEN_PALM", "Single frame dropout must be tolerated"

    # Frame 6-8: Sustained low confidence below exit_threshold (0.40) for pose_window frames
    pose_low = make_derived_pose(HandPoseId.UNKNOWN, "UNKNOWN", conf=0.40)
    ppm.update(pose_low, timestamp=1.165)
    ppm.update(pose_low, timestamp=1.198)
    stab_pose, persistent = ppm.update(pose_low, timestamp=1.231)

    assert persistent.canonical_name != "OPEN_PALM", "Sustained loss below exit_threshold must exit pose"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Motion Dead Zone (Spatial Filtering & Ramp)
# ─────────────────────────────────────────────────────────────────────────────

def test_motion_deadzone_filtering():
    config = StabilityConfig(dead_zone_px=18.0, dead_zone_ramp_width_px=10.0)
    mdz = MotionDeadZone(config)

    # Initial anchor position at screen center (0.5, 0.5, 0.0)
    pos_anchor = (0.5, 0.5, 0.0)
    f_pos, f_vel, f_disp, in_dz = mdz.update(pos_anchor, d_ref=120.0, is_active_manipulation=False)
    assert in_dz is True
    assert f_disp == 0.0

    # Small jitter movement of 3 pixels (normalized in 1280x720: dx ~ 3/1280 = 0.00234)
    pos_jitter = (0.5 + 3.0 / 1280.0, 0.5 + 2.0 / 720.0, 0.0)
    f_pos, f_vel, f_disp, in_dz = mdz.update(pos_jitter, d_ref=120.0, is_active_manipulation=False)
    assert in_dz is True, "Movement smaller than dead zone radius must be suppressed"
    assert f_vel == (0.0, 0.0, 0.0)
    assert f_disp == 0.0

    # Large intentional movement of 40 pixels (well outside 18px dead zone)
    pos_move = (0.5 + 40.0 / 1280.0, 0.5, 0.0)
    f_pos, f_vel, f_disp, in_dz = mdz.update(pos_move, d_ref=120.0, is_active_manipulation=False)
    assert in_dz is False, "Deliberate movement outside dead zone must be allowed"
    assert f_disp > 0.0


def test_motion_deadzone_active_bypass():
    config = StabilityConfig(dead_zone_px=18.0)
    mdz = MotionDeadZone(config)

    pos = (0.5, 0.5, 0.0)
    mdz.update(pos, d_ref=120.0, is_active_manipulation=False)

    # During active manipulation (e.g. 3D object manipulation), dead zone is bypassed
    pos_small = (0.5 + 4.0 / 1280.0, 0.5, 0.0)
    f_pos, f_vel, f_disp, in_dz = mdz.update(pos_small, d_ref=120.0, is_active_manipulation=True)
    assert in_dz is False, "Active manipulation must bypass dead zone"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Micro-Adjustment Filter
# ─────────────────────────────────────────────────────────────────────────────

def test_micro_adjustment_filter():
    config = StabilityConfig()
    maf = MicroAdjustmentFilter(config)

    # Case 1: Stationary hand with 1 twitching finger -> MICRO_ADJUSTMENT
    category, reasons = maf.evaluate(
        palm_pos=(0.5, 0.5, 0.0),
        palm_speed=0.01,
        detection_confidence=0.95,
        finger_changes={"Index": "curved"},
        pose_changed=False,
        is_transitioning=False,
        is_in_dead_zone=False,
        timestamp=1.0,
    )
    assert category == MovementCategory.MICRO_ADJUSTMENT
    assert any("twitch" in r for r in reasons)

    # Case 2: Hand transitioning between poses -> TRANSITION
    category, reasons = maf.evaluate(
        palm_pos=(0.5, 0.5, 0.0),
        palm_speed=0.05,
        detection_confidence=0.95,
        finger_changes={"Index": "curved", "Middle": "curved", "Ring": "curved"},
        pose_changed=True,
        is_transitioning=True,
        is_in_dead_zone=False,
        timestamp=1.0,
    )
    assert category == MovementCategory.TRANSITION

    # Case 3: Stable deliberate hand -> INTENTIONAL
    category, reasons = maf.evaluate(
        palm_pos=(0.5, 0.5, 0.0),
        palm_speed=0.02,
        detection_confidence=0.95,
        finger_changes={},
        pose_changed=False,
        is_transitioning=False,
        is_in_dead_zone=True,
        timestamp=1.0,
    )
    assert category == MovementCategory.INTENTIONAL


# ─────────────────────────────────────────────────────────────────────────────
# 5. Transition Detection
# ─────────────────────────────────────────────────────────────────────────────

def test_transition_detector():
    config = StabilityConfig()
    td = TransitionDetector(config)

    fingers = make_finger_states(FingerStateEnum.EXTENDED)
    is_trans, from_p, to_p, prog = td.update("OPEN_PALM", "CLOSED_FIST", fingers, timestamp=1.0)
    assert is_trans is True
    assert from_p == "OPEN_PALM"
    assert to_p == "CLOSED_FIST"

    # Once arrived and stable on destination pose for required duration:
    for i in range(10):
        is_trans, from_p, to_p, prog = td.update("CLOSED_FIST", "CLOSED_FIST", fingers, timestamp=1.0 + (i + 1) * 0.033)
    assert is_trans is False, "Transition must complete once destination pose is stable"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Intentional Hold Dwell
# ─────────────────────────────────────────────────────────────────────────────

def test_intentional_hold_tracker():
    config = StabilityConfig(pinch_hold_ms=120.0, point_hold_ms=80.0)
    tracker = IntentionalHoldTracker(config)

    # Pose not confirmed yet -> timer must NOT start
    is_sat, dur, target = tracker.update(
        stable_pose_name="PINCH",
        is_pose_confirmed=False,
        is_in_dead_zone_or_still=True,
        timestamp=1.0,
    )
    assert is_sat is False
    assert dur == 0.0

    # Pose confirmed at t=1.0: timer begins
    is_sat, dur, target = tracker.update("PINCH", is_pose_confirmed=True, is_in_dead_zone_or_still=True, timestamp=1.0)
    assert is_sat is False

    # After 60ms (less than 120ms requirement): not satisfied
    is_sat, dur, target = tracker.update("PINCH", is_pose_confirmed=True, is_in_dead_zone_or_still=True, timestamp=1.060)
    assert is_sat is False
    assert dur >= 59.0

    # After 130ms (exceeds 120ms requirement): SATISFIED
    is_sat, dur, target = tracker.update("PINCH", is_pose_confirmed=True, is_in_dead_zone_or_still=True, timestamp=1.130)
    assert is_sat is True
    assert dur >= 120.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. State Lock System
# ─────────────────────────────────────────────────────────────────────────────

def test_state_lock_system():
    config = StabilityConfig(active_preservation_frames=5, lock_exit_threshold=0.55)
    lock_sys = StateLockSystem(config)

    # Acquire lock on active interaction
    is_locked, locked_name = lock_sys.update(
        is_active_interaction=True,
        current_pose_name="GRAB",
        detection_confidence=0.95,
        is_release_gesture=False,
        timestamp=1.0,
    )
    assert is_locked is True
    assert locked_name == "GRAB"

    # Temporary dip in landmark confidence (e.g. 0.45 for 2 frames)
    for _ in range(2):
        is_locked, locked_name = lock_sys.update(
            is_active_interaction=True,
            current_pose_name="GRAB",
            detection_confidence=0.45,
            is_release_gesture=False,
            timestamp=1.05,
        )
        assert is_locked is True, "State lock must protect active interaction against brief landmark instability"

    # Recovery of confidence
    is_locked, _ = lock_sys.update(
        is_active_interaction=True,
        current_pose_name="GRAB",
        detection_confidence=0.92,
        is_release_gesture=False,
        timestamp=1.10,
    )
    assert is_locked is True

    # Deliberate release gesture (e.g. OPEN_PALM or RELEASE)
    is_locked, _ = lock_sys.update(
        is_active_interaction=True,
        current_pose_name="OPEN_PALM",
        detection_confidence=0.92,
        is_release_gesture=True,
        timestamp=1.15,
    )
    assert is_locked is False, "Deliberate release gesture must exit state lock"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Stability Score Calculator
# ─────────────────────────────────────────────────────────────────────────────

def test_stability_score_formula():
    config = StabilityConfig(
        weight_recognition=0.35,
        weight_temporal=0.25,
        weight_motion=0.20,
        weight_persistence=0.20,
        stability_threshold=0.90,
    )
    calc = StabilityScoreCalculator(config)

    # Perfect signals -> Score = 1.00 >= 0.90 -> Eligible
    score, eligible, breakdown = calc.compute(
        recognition_conf=1.0,
        temporal_consistency=1.0,
        motion_consistency=1.0,
        persistence_score=1.0,
    )
    assert score == pytest.approx(1.0, rel=1e-3)
    assert eligible is True

    # Degraded signals: 0.35*0.6 + 0.25*0.5 + 0.20*0.7 + 0.20*0.6 = 0.21 + 0.125 + 0.14 + 0.12 = 0.595 < 0.90
    score, eligible, breakdown = calc.compute(
        recognition_conf=0.6,
        temporal_consistency=0.5,
        motion_consistency=0.7,
        persistence_score=0.6,
    )
    assert score < 0.90
    assert eligible is False


# ─────────────────────────────────────────────────────────────────────────────
# 9. Stability Engine End-to-End Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def test_stability_engine_end_to_end():
    config = StabilityConfig()
    engine = StabilityEngine(config)

    raw_pts = np.zeros((21, 3), dtype=np.float32)
    finger_states = make_finger_states(FingerStateEnum.EXTENDED)
    pose = make_derived_pose(HandPoseId.H001_OPEN_PALM, "OPEN_PALM", conf=0.95)
    motion = make_motion_state()

    # Process 5 consecutive frames
    ctx = None
    for i in range(5):
        t = 1.0 + i * 0.033
        ctx = engine.process_hand_stability(
            hand_id=0,
            handedness="Right",
            raw_landmarks=raw_pts,
            instantaneous_finger_states=finger_states,
            instantaneous_pose=pose,
            instantaneous_motion=motion,
            candidate_gesture="OPEN_PALM",
            candidate_confidence=0.95,
            palm_center=(0.5, 0.5, 0.0),
            d_ref=120.0,
            detection_confidence=0.98,
            timestamp=t,
        )

    assert ctx is not None
    assert ctx.persistent_pose.canonical_name == "OPEN_PALM"
    assert ctx.stability_score >= 0.85
    assert ctx.telemetry is not None
    assert "finger_layer" in ctx.telemetry
    assert "pose_layer" in ctx.telemetry
    assert "motion_layer" in ctx.telemetry
    assert "stability_layer" in ctx.telemetry
    assert "intent_layer" in ctx.telemetry


# ─────────────────────────────────────────────────────────────────────────────
# 10. Interaction Engine Command Gating & State Lock Preservation
# ─────────────────────────────────────────────────────────────────────────────

def test_interaction_engine_stability_gating():
    interaction_engine = InteractionEngine()

    # Create dummy hand state
    raw_pts = np.zeros((21, 3), dtype=np.float32)
    norm_pts = np.zeros((21, 3), dtype=np.float32)
    hand = HandState(
        hand_id=0,
        handedness="Right",
        landmarks=[],
        raw_landmarks_array=raw_pts,
        normalized_landmarks_array=norm_pts,
        palm_center=(0.5, 0.5, 0.0),
        palm_velocity=(0.0, 0.0, 0.0),
        hand_scale_ref=100.0,
        orientation_angles=(0.0, 0.0, 0.0),
        finger_extension_ratios={},
        finger_flexion_angles={},
        pinch_distance=0.01,
        pinch_confidence=0.95,
        detection_confidence=0.95,
    )

    # Inject intent context that would normally map to PINCH_SELECT
    hand.intent_context = IntentContext(
        state=InteractionState.CONFIRMED,
        active_gesture=GestureType.PINCH,
        intent_confidence=0.95,
        evidence_score=0.95,
        consecutive_frames_in_state=5,
        state_duration_sec=0.15,
        timestamp=1.0,
        gesture_confidence=0.95,
    )

    # Inject mock stability context marking hand as MICRO_ADJUSTMENT (not eligible for commands)
    class MockStabilityContext:
        is_command_eligible = False
        movement_category = MovementCategory.MICRO_ADJUSTMENT
        is_state_locked = False
        stability_score = 0.45

    hand.stability_context = MockStabilityContext()

    command, intent_ctx, primary = interaction_engine.process_hands([hand], timestamp=1.0)

    # The PINCH_SELECT action command must be suppressed to HOVER with STABILITY_SUPPRESSED
    assert command.command_type == SpatialCommandType.HOVER
    assert command.interaction_state == "STABILITY_SUPPRESSED"
