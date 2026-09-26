"""
Unit tests for the Temporal Intent Engine, Observation Windows,
Intent Lock, Priority Manager, and Layered Stabilization Pipeline.
"""

import numpy as np
import pytest

from src.gestures.gesture_types import GestureType, RecognizedGesture
from src.gestures.hand_pose import DerivedHandPose, FingerConfiguration, HandPoseId
from src.intent.intent_lock import IntentLockSystem
from src.intent.observation_window import (
    FingerStateStabilizer,
    GestureCandidateBuffer,
    MotionPrimitiveAnalyzer,
    ObservationWindowManager,
    PoseConfirmationManager,
)
from src.intent.priority_manager import GesturePriorityManager, GesturePriorityTier
from src.intent.state_machine import InteractionState, InteractionStateMachine
from src.intent.temporal_intent_config import TemporalIntentConfig
from src.intent.temporal_intent_engine import TemporalIntentEngine
from src.landmarks.finger_state import FingerName, FingerStateDetail, FingerStateEnum, HandFingerStates
from src.motion.motion_primitive import MotionPrimitive


def _make_landmarks(base_x: float = 0.5, base_y: float = 0.5, noise: float = 0.0) -> np.ndarray:
    """Generate 21 hand landmarks around a base point with optional noise."""
    pts = np.zeros((21, 3), dtype=np.float32)
    for i in range(21):
        pts[i] = [base_x + i * 0.01 + np.random.uniform(-noise, noise),
                  base_y + i * 0.01 + np.random.uniform(-noise, noise),
                  -0.02 + np.random.uniform(-noise, noise)]
    return pts


def _make_finger_states(thumb: FingerStateEnum, index: FingerStateEnum,
                        middle: FingerStateEnum, ring: FingerStateEnum,
                        little: FingerStateEnum) -> HandFingerStates:
    """Helper to create HandFingerStates."""
    return HandFingerStates(
        thumb=FingerStateDetail(FingerName.THUMB, thumb, 0.95),
        index=FingerStateDetail(FingerName.INDEX, index, 0.95),
        middle=FingerStateDetail(FingerName.MIDDLE, middle, 0.95),
        ring=FingerStateDetail(FingerName.RING, ring, 0.95),
        little=FingerStateDetail(FingerName.LITTLE, little, 0.95),
    )


def _make_derived_pose(pose_id: HandPoseId, name: str, conf: float = 0.90) -> DerivedHandPose:
    """Helper to create a DerivedHandPose."""
    config = FingerConfiguration(
        thumb=FingerStateEnum.FOLDED,
        index=FingerStateEnum.EXTENDED if pose_id == HandPoseId.H004_INDEX_POINT else FingerStateEnum.FOLDED,
        middle=FingerStateEnum.FOLDED,
        ring=FingerStateEnum.FOLDED,
        little=FingerStateEnum.FOLDED,
        num_extended_fingers=1 if pose_id == HandPoseId.H004_INDEX_POINT else 0,
        num_folded_fingers=3,
        num_curved_fingers=0,
        num_uncertain_fingers=0,
        thumb_is_extended=False,
        thumb_is_folded=True,
        thumb_is_pinching=False,
    )
    return DerivedHandPose(
        pose_id=pose_id,
        canonical_name=name,
        confidence=conf,
        configuration=config,
    )


# =========================================================================
# 1. Observation Windows & Finger State Stabilization
# =========================================================================

def test_finger_state_stabilizer_jitter_rejection():
    """Verify that a 1-frame jitter/glitch does not change the stabilized digit state."""
    config = TemporalIntentConfig()
    config.windows.finger_window = 3
    stabilizer = FingerStateStabilizer(config)

    # Prime with 3 frames of stable POINT state (Thumb folded, Index extended, others folded)
    stable_fs = _make_finger_states(
        FingerStateEnum.FOLDED, FingerStateEnum.EXTENDED,
        FingerStateEnum.FOLDED, FingerStateEnum.FOLDED, FingerStateEnum.FOLDED
    )

    for i in range(3):
        stabilized_fs, summary = stabilizer.update(stable_fs, 0.95, 1.0 + i * 0.033)

    assert stabilized_fs.index.state == FingerStateEnum.EXTENDED
    assert stabilized_fs.middle.state == FingerStateEnum.FOLDED
    assert summary["is_stable"] is True

    # 1-frame noise: middle finger momentarily flickers to EXTENDED
    glitch_fs = _make_finger_states(
        FingerStateEnum.FOLDED, FingerStateEnum.EXTENDED,
        FingerStateEnum.EXTENDED, FingerStateEnum.FOLDED, FingerStateEnum.FOLDED
    )
    stabilized_fs, summary = stabilizer.update(glitch_fs, 0.95, 1.1)

    # Stabilizer majority voting rejects the 1-frame glitch
    assert stabilized_fs.middle.state == FingerStateEnum.FOLDED
    assert stabilized_fs.index.state == FingerStateEnum.EXTENDED


def test_finger_state_stabilizer_ema_smoothing():
    """Verify landmark exponential smoothing reduces high-frequency step changes."""
    config = TemporalIntentConfig()
    stabilizer = FingerStateStabilizer(config)

    pts1 = _make_landmarks(0.5, 0.5)
    pts2 = _make_landmarks(0.7, 0.7)  # Big step jump

    stabilizer.smooth_landmarks(pts1)
    smoothed_pts = stabilizer.smooth_landmarks(pts2)

    # Smoothed landmarks in X and Y must be strictly between pts1 and pts2
    assert np.all(smoothed_pts[:, :2] > pts1[:, :2])
    assert np.all(smoothed_pts[:, :2] < pts2[:, :2])


# =========================================================================
# 2. Pose Confirmation with Enter/Exit Hysteresis
# =========================================================================

def test_pose_confirmation_requires_multi_frame():
    """Verify a pose requires min_confirm_frames (3-5F) before confirmation."""
    config = TemporalIntentConfig()
    config.pose.min_confirm_frames = 3
    config.pose.enter_threshold = 0.80
    manager = PoseConfirmationManager(config)

    point_pose = _make_derived_pose(HandPoseId.H004_INDEX_POINT, "POINT", 0.90)

    # Frame 1: Candidate observed, not yet confirmed
    out_id, out_name, out_conf, is_conf, telem = manager.update(point_pose, 1.0)
    assert out_name == "NONE"
    assert telem["candidate_pose"] == "POINT"
    assert is_conf is False
    assert telem["frames_consistent"] == 1

    # Frame 2: Still not confirmed
    out_id, out_name, out_conf, is_conf, telem = manager.update(point_pose, 1.033)
    assert out_name == "NONE"
    assert is_conf is False
    assert telem["frames_consistent"] == 2

    # Frame 3: Confirmed!
    out_id, out_name, out_conf, is_conf, telem = manager.update(point_pose, 1.066)
    assert out_name == "POINT"
    assert is_conf is True
    assert telem["frames_consistent"] == 3


def test_pose_confirmation_hysteresis():
    """
    Verify enter vs exit hysteresis:
    Entering requires enter_threshold (e.g. 0.80).
    Exiting drops only below exit_threshold (e.g. 0.45).
    """
    config = TemporalIntentConfig()
    config.pose.enter_threshold = 0.80
    config.pose.exit_threshold = 0.45
    config.pose.min_confirm_frames = 2
    manager = PoseConfirmationManager(config)

    # 1. Attempt to enter with 0.70 confidence (< enter_threshold 0.80) -> rejected
    sub_pose = _make_derived_pose(HandPoseId.H005_PRECISION_PINCH, "PINCH", 0.70)
    manager.update(sub_pose, 1.0)
    _, out_name, _, is_conf, _ = manager.update(sub_pose, 1.033)
    assert is_conf is False
    assert out_name == "NONE"

    # 2. Enter with high confidence (0.92 >= 0.80)
    high_pose = _make_derived_pose(HandPoseId.H005_PRECISION_PINCH, "PINCH", 0.92)
    manager.update(high_pose, 1.066)
    _, out_name, _, is_conf, _ = manager.update(high_pose, 1.100)
    assert is_conf is True
    assert out_name == "PINCH"

    # 3. Confidence drops to 0.60 (below enter_threshold 0.80, but ABOVE exit_threshold 0.45)
    # Pose must REMAIN CONFIRMED due to hysteresis
    mid_pose = _make_derived_pose(HandPoseId.H005_PRECISION_PINCH, "PINCH", 0.60)
    _, out_name, _, is_conf, _ = manager.update(mid_pose, 1.133)
    assert is_conf is True
    assert out_name == "PINCH"

    # 4. Confidence drops below exit_threshold (0.35 < 0.45) -> deconfirmed
    low_pose = _make_derived_pose(HandPoseId.H005_PRECISION_PINCH, "PINCH", 0.35)
    _, out_name, _, is_conf, _ = manager.update(low_pose, 1.166)
    assert is_conf is False


# =========================================================================
# 3. Motion Primitive Analyzer (Linearity & Directional Consistency)
# =========================================================================

def test_motion_analyzer_linear_swipe():
    """Verify linear rightward movement achieves high consistency and linearity."""
    config = TemporalIntentConfig()
    analyzer = MotionPrimitiveAnalyzer(config)

    # Simulate smooth rightward movement (+X direction)
    t = 1.0
    for i in range(8):
        pos = (0.2 + i * 0.05, 0.5, -0.05)
        prim, metrics = analyzer.update(pos, d_ref=0.20, timestamp=t)
        t += 0.033

    assert prim in (MotionPrimitive.MOVE_RIGHT, MotionPrimitive.STATIONARY)
    assert metrics["direction_consistency"] > 0.85
    assert metrics["linearity"] > 0.85
    assert metrics["is_intentional"] is True


def test_motion_analyzer_curved_wavy_rejection():
    """Verify curved/waving trajectories have low directional consistency."""
    config = TemporalIntentConfig()
    analyzer = MotionPrimitiveAnalyzer(config)

    # Zigzag motion: alternating up and down sharply
    t = 1.0
    for i in range(10):
        pos = (0.5, 0.5 + (0.1 if i % 2 == 0 else -0.1), -0.05)
        prim, metrics = analyzer.update(pos, d_ref=0.20, timestamp=t)
        t += 0.033

    # Direction consistency must be lower due to alternating direction vectors
    assert metrics["direction_consistency"] < 0.85


# =========================================================================
# 4. Intent Lock System
# =========================================================================

def test_intent_lock_system_lifecycle():
    """Verify Intent Lock acquires, suppresses secondary gestures, and releases."""
    lock_sys = IntentLockSystem()
    assert lock_sys.is_locked is False

    # Acquire lock for PINCH interaction
    acquired = lock_sys.acquire_lock("PINCH")
    assert acquired is True
    assert lock_sys.is_locked is True
    assert lock_sys.locked_by == "PINCH"

    # Under PINCH lock, swipe / hover / grab are suppressed
    assert lock_sys.is_gesture_allowed("SWIPE_LEFT") is False
    assert lock_sys.is_gesture_allowed("POINT") is False
    assert lock_sys.is_gesture_allowed("FIST") is False

    # But PINCH family gestures are allowed
    assert lock_sys.is_gesture_allowed("PINCH") is True
    assert lock_sys.is_gesture_allowed("G007_PINCH_DRAG") is True

    # Release lock
    released = lock_sys.release_lock()
    assert released is True
    assert lock_sys.is_locked is False
    assert lock_sys.is_gesture_allowed("SWIPE_LEFT") is True


# =========================================================================
# 5. Gesture Priority Manager
# =========================================================================

def test_gesture_priority_arbitration():
    """
    Verify 5-tier priority hierarchy:
    Tier 1: Pinch Selection > Tier 2: Bimanual > Tier 3: Swipe > Tier 4: Point Hover > Tier 5: Idle
    """
    pm = GesturePriorityManager()

    assert pm.get_tier("PINCH") == GesturePriorityTier.TIER_1_PINCH_SELECT
    assert pm.get_tier("BIMANUAL_ZOOM") == GesturePriorityTier.TIER_2_TWO_HAND_MANIP
    assert pm.get_tier("SWIPE_LEFT") == GesturePriorityTier.TIER_3_SWIPE
    assert pm.get_tier("POINT") == GesturePriorityTier.TIER_4_POINT_HOVER
    assert pm.get_tier("IDLE") == GesturePriorityTier.TIER_5_IDLE

    # While Pinch is active, Swipe is suppressed
    pm.arbitrate("PINCH", current_state_is_active=False)
    allowed, winner, tier = pm.arbitrate("SWIPE_LEFT", current_state_is_active=True)
    assert allowed is False
    assert winner == "PINCH"
    assert tier == GesturePriorityTier.TIER_1_PINCH_SELECT

    # But while Point hover is active, Pinch can pre-empt it
    pm.reset()
    pm.arbitrate("POINT", current_state_is_active=False)
    allowed, winner, tier = pm.arbitrate("PINCH", current_state_is_active=True)
    assert allowed is True
    assert winner == "PINCH"
    assert tier == GesturePriorityTier.TIER_1_PINCH_SELECT


# =========================================================================
# 6. Intent State Machine & Graceful Release Easing
# =========================================================================

def test_intent_state_machine_full_lifecycle():
    """
    Test full 6-state lifecycle:
    IDLE -> OBSERVING -> CANDIDATE -> CONFIRMED -> ACTIVE -> RELEASING -> IDLE
    """
    fsm = InteractionStateMachine(
        activation_threshold=0.75,
        release_threshold=0.35,
        confirm_frames=3,
        evidence_lambda=0.80,
    )

    t = 1.0
    # 1. IDLE -> OBSERVING
    ctx = fsm.update(hand_detected=True, recognized_gesture=None, timestamp=t)
    assert ctx.state == InteractionState.OBSERVING

    def _make_g(g: GestureType, conf: float, ts: float) -> RecognizedGesture:
        return RecognizedGesture(
            gesture=g,
            confidence=conf,
            hand_id=0,
            handedness="Right",
            timestamp=ts,
        )

    # 2. OBSERVING -> CANDIDATE
    t += 0.033
    rec_g = _make_g(GestureType.PINCH, 0.88, t)
    ctx = fsm.update(hand_detected=True, recognized_gesture=rec_g, timestamp=t)
    assert ctx.state == InteractionState.CANDIDATE

    # 3. Evidence accumulates -> CONFIRMED
    for _ in range(8):
        t += 0.033
        rec_g = _make_g(GestureType.PINCH, 0.95, t)
        ctx = fsm.update(hand_detected=True, recognized_gesture=rec_g, timestamp=t)

    assert ctx.state in (InteractionState.CONFIRMED, InteractionState.ACTIVE)

    # 4. Latches ACTIVE
    t += 0.033
    ctx = fsm.update(hand_detected=True, recognized_gesture=rec_g, timestamp=t)
    assert ctx.state == InteractionState.ACTIVE
    assert ctx.easing_factor == 1.0

    # 5. Hand releases -> transitions to RELEASING
    t += 0.033
    rec_rel = _make_g(GestureType.OPEN_PALM, 0.90, t)
    ctx = fsm.update(hand_detected=True, recognized_gesture=rec_rel, timestamp=t)
    assert ctx.state == InteractionState.RELEASING
    assert ctx.easing_factor == 1.0  # Initial easing factor at release onset

    # 6. Easing factor continues decaying over cooldown frames
    t += 0.033
    ctx = fsm.update(hand_detected=True, recognized_gesture=None, timestamp=t)
    assert ctx.easing_factor < 1.0  # Decaying smoothly!
    prev_easing = ctx.easing_factor

    t += 0.033
    ctx = fsm.update(hand_detected=True, recognized_gesture=None, timestamp=t)
    assert ctx.easing_factor < prev_easing

    # Eventually returns to IDLE or OBSERVING
    for _ in range(10):
        t += 0.033
        ctx = fsm.update(hand_detected=True, recognized_gesture=None, timestamp=t)

    assert ctx.state in (InteractionState.OBSERVING, InteractionState.IDLE)


# =========================================================================
# 7. End-to-End Temporal Intent Engine Coordination
# =========================================================================

def test_temporal_intent_engine_pipeline():
    """Verify TemporalIntentEngine processes full layered pipeline without crashing."""
    config = TemporalIntentConfig()
    engine = TemporalIntentEngine(config)

    fs = _make_finger_states(
        FingerStateEnum.FOLDED, FingerStateEnum.EXTENDED,
        FingerStateEnum.FOLDED, FingerStateEnum.FOLDED, FingerStateEnum.FOLDED
    )
    pose = _make_derived_pose(HandPoseId.H004_INDEX_POINT, "POINT", 0.92)
    pts = _make_landmarks(0.5, 0.5)

    # Frame 1
    stab_fs, conf_pose, motion_prim, ctx, telem = engine.process_hand_temporal(
        hand_id=0,
        handedness="Right",
        raw_landmarks=pts,
        instantaneous_finger_states=fs,
        instantaneous_pose=pose,
        palm_center=(0.5, 0.5, -0.05),
        d_ref=0.20,
        detection_confidence=0.95,
        candidate_gesture_name="POINT",
        candidate_confidence=0.92,
        timestamp=1.0,
    )

    # Guaranteed: never confirms on first frame
    assert ctx.state != InteractionState.CONFIRMED
    assert ctx.state != InteractionState.ACTIVE
    assert telem.frame_index == 1

    # Stream several frames to confirm
    for i in range(10):
        stab_fs, conf_pose, motion_prim, ctx, telem = engine.process_hand_temporal(
            hand_id=0,
            handedness="Right",
            raw_landmarks=pts,
            instantaneous_finger_states=fs,
            instantaneous_pose=pose,
            palm_center=(0.5, 0.5, -0.05),
            d_ref=0.20,
            detection_confidence=0.95,
            candidate_gesture_name="POINT",
            candidate_confidence=0.92,
            timestamp=1.0 + (i + 1) * 0.033,
        )

    # Telemetry should be populated
    d = telem.as_dict()
    assert "finger_layer" in d
    assert "pose_layer" in d
    assert "motion_layer" in d
    assert "intent_layer" in d
    assert "performance" in d
    assert d["pose_layer"]["is_confirmed"] is True
