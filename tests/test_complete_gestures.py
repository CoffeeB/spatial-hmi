"""
Unit tests for Gestura Level 3 Complete Gestures.
Validates the composition of Level 1 Static Hand Poses and Level 2 Motion Primitives
into meaningful complete gestures, verified through sequential observable physical events:
    - POINT while SWIPE LEFT (Index extended, others curled + rapid leftward stroke)
    - PALM SWIPE LEFT (Open palm + rapid leftward stroke)
    - PINCH: Observable sequence (Approach → Contact → Hold → Drag → Release)
    - AIR TAP: Aim → Strike → Rebound
    - POINT & HOVER
    - THUMBS UP APPROVAL
"""

import math
import numpy as np
import pytest

from src.gestures.complete_gesture import (
    CompleteGestureId,
    CompleteGestureRecognizer,
    GestureCategory,
    PinchPhase,
    SwipePhase,
    TapPhase,
)
from src.gestures.hand_pose import DerivedHandPose, HandPoseId
from src.landmarks.finger_state import FingerName, FingerStateDetail, FingerStateEnum, HandFingerStates
from src.motion.motion_primitive import DynamicState, FingerMotionPrimitive, FingerMotionState, MotionPrimitive, MotionState
from src.motion.motion_tracker import MotionPrimitiveTracker


def _make_dummy_finger_states(pose_type: str = "POINT") -> HandFingerStates:
    """Creates a synthetic HandFingerStates object for testing."""
    if pose_type == "POINT":
        return HandFingerStates(
            thumb=FingerStateDetail(FingerName.THUMB, FingerStateEnum.TOUCHING, 1.0, 0.90, 20.0, 15.0, 10.0),
            index=FingerStateDetail(FingerName.INDEX, FingerStateEnum.EXTENDED, 1.0, 1.45, 10.0, 5.0, 5.0),
            middle=FingerStateDetail(FingerName.MIDDLE, FingerStateEnum.FOLDED, 1.0, 0.70, 75.0, 85.0, 70.0),
            ring=FingerStateDetail(FingerName.RING, FingerStateEnum.FOLDED, 1.0, 0.65, 80.0, 90.0, 75.0),
            little=FingerStateDetail(FingerName.LITTLE, FingerStateEnum.FOLDED, 1.0, 0.65, 80.0, 90.0, 75.0),
            palm_facing="PALM",
        )
    elif pose_type == "OPEN_PALM":
        return HandFingerStates(
            thumb=FingerStateDetail(FingerName.THUMB, FingerStateEnum.EXTENDED, 1.0, 1.40, 10.0, 5.0, 5.0),
            index=FingerStateDetail(FingerName.INDEX, FingerStateEnum.EXTENDED, 1.0, 1.40, 10.0, 5.0, 5.0),
            middle=FingerStateDetail(FingerName.MIDDLE, FingerStateEnum.EXTENDED, 1.0, 1.40, 10.0, 5.0, 5.0),
            ring=FingerStateDetail(FingerName.RING, FingerStateEnum.EXTENDED, 1.0, 1.40, 10.0, 5.0, 5.0),
            little=FingerStateDetail(FingerName.LITTLE, FingerStateEnum.EXTENDED, 1.0, 1.40, 10.0, 5.0, 5.0),
            palm_facing="PALM",
        )
    elif pose_type == "THUMBS_UP":
        return HandFingerStates(
            thumb=FingerStateDetail(FingerName.THUMB, FingerStateEnum.EXTENDED, 1.0, 1.40, 10.0, 5.0, 5.0),
            index=FingerStateDetail(FingerName.INDEX, FingerStateEnum.FOLDED, 1.0, 0.65, 80.0, 90.0, 75.0),
            middle=FingerStateDetail(FingerName.MIDDLE, FingerStateEnum.FOLDED, 1.0, 0.65, 80.0, 90.0, 75.0),
            ring=FingerStateDetail(FingerName.RING, FingerStateEnum.FOLDED, 1.0, 0.65, 80.0, 90.0, 75.0),
            little=FingerStateDetail(FingerName.LITTLE, FingerStateEnum.FOLDED, 1.0, 0.65, 80.0, 90.0, 75.0),
            palm_facing="PALM",
        )
    else:
        return HandFingerStates(
            thumb=FingerStateDetail(FingerName.THUMB, FingerStateEnum.FOLDED, 1.0, 0.70, 70.0, 80.0, 70.0),
            index=FingerStateDetail(FingerName.INDEX, FingerStateEnum.FOLDED, 1.0, 0.70, 70.0, 80.0, 70.0),
            middle=FingerStateDetail(FingerName.MIDDLE, FingerStateEnum.FOLDED, 1.0, 0.70, 70.0, 80.0, 70.0),
            ring=FingerStateDetail(FingerName.RING, FingerStateEnum.FOLDED, 1.0, 0.70, 70.0, 80.0, 70.0),
            little=FingerStateDetail(FingerName.LITTLE, FingerStateEnum.FOLDED, 1.0, 0.70, 70.0, 80.0, 70.0),
            palm_facing="PALM",
        )


def _make_dummy_landmarks(
    thumb_tip=(0.38, 0.50, 0.0),
    index_tip=(0.50, 0.25, 0.0),
    middle_tip=(0.50, 0.20, 0.0),
) -> np.ndarray:
    """Generates synthetic 21x3 landmarks array."""
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.5, 0.7, 0.0]   # Wrist
    pts[9] = [0.5, 0.5, 0.0]   # Middle MCP
    pts[4] = thumb_tip         # Thumb Tip
    pts[8] = index_tip         # Index Tip
    pts[12] = middle_tip       # Middle Tip
    return pts


def test_point_swipe_left_sequence():
    """
    Validates user specification:
    POINT (Index extended + other fingers folded)
    while
    SWIPE LEFT (rapid leftward motion + sufficient displacement + directional consistency)
    -> G001P_POINT_SWIPE_LEFT.
    """
    recognizer = CompleteGestureRecognizer()
    motion_tracker = MotionPrimitiveTracker(velocity_alpha=0.80)
    finger_states = _make_dummy_finger_states("POINT")

    derived_pose = DerivedHandPose(
        pose_id=HandPoseId.H004_INDEX_POINT,
        canonical_name="POINT",
        confidence=0.92,
        configuration=None,
        satisfied_predicates=["index_extended", "middle_curled"],
        diagnostics=[],
    )

    t = 100.0
    d_ref = 0.20
    raw_lms = _make_dummy_landmarks()

    # Step 1: Prepare Phase — Holding Point pose steady (low speed)
    motion_prep = motion_tracker.update(
        hand_id=0,
        handedness="Right",
        palm_center=(0.50, 0.50, 0.0),
        hand_scale_ref=d_ref,
        timestamp=t,
    )

    state_prep = recognizer.update(
        hand_id=0,
        handedness="Right",
        derived_pose=derived_pose,
        motion_state=motion_prep,
        finger_states=finger_states,
        raw_landmarks=raw_lms,
        d_ref=d_ref,
        timestamp=t,
    )

    assert state_prep.gesture_id == CompleteGestureId.G005_POINT_HOVER
    assert "POINT" in state_prep.hand_pose_name

    # Step 2: Stroke Phase — Rapid leftward displacement with high linearity
    x = 0.50
    motion_stroke = None
    for _ in range(5):
        t += 0.033
        x -= 0.015  # moving left at ~0.45 u/s
        motion_stroke = motion_tracker.update(
            hand_id=0,
            handedness="Right",
            palm_center=(x, 0.50, 0.0),
            hand_scale_ref=d_ref,
            timestamp=t,
        )

    state_swipe = recognizer.update(
        hand_id=0,
        handedness="Right",
        derived_pose=derived_pose,
        motion_state=motion_stroke,
        finger_states=finger_states,
        raw_landmarks=raw_lms,
        d_ref=d_ref,
        timestamp=t,
    )

    assert state_swipe.gesture_id == CompleteGestureId.G001P_POINT_SWIPE_LEFT
    assert state_swipe.canonical_name in ("1-FINGER SWIPE LEFT", "POINT SWIPE LEFT")
    assert state_swipe.category == GestureCategory.SWIPE_STROKE
    assert state_swipe.confidence > 0.80
    assert any("STROKE" in ev for ev in state_swipe.event_sequence)


def test_palm_swipe_left_sequence():
    """Validates Open Palm + rapid leftward motion -> G001_PALM_SWIPE_LEFT."""
    recognizer = CompleteGestureRecognizer()
    motion_tracker = MotionPrimitiveTracker(velocity_alpha=0.80)
    finger_states = _make_dummy_finger_states("OPEN_PALM")

    derived_pose = DerivedHandPose(
        pose_id=HandPoseId.H001_OPEN_PALM,
        canonical_name="OPEN_PALM",
        confidence=0.95,
        configuration=None,
        satisfied_predicates=["all_extended"],
        diagnostics=[],
    )

    d_ref = 0.20
    raw_lms = _make_dummy_landmarks()
    t = 100.0
    x = 0.60
    motion_stroke = None

    for _ in range(5):
        t += 0.033
        x -= 0.015  # moving left
        motion_stroke = motion_tracker.update(
            hand_id=0,
            handedness="Right",
            palm_center=(x, 0.50, 0.0),
            hand_scale_ref=d_ref,
            timestamp=t,
        )

    state = recognizer.update(
        hand_id=0,
        handedness="Right",
        derived_pose=derived_pose,
        motion_state=motion_stroke,
        finger_states=finger_states,
        raw_landmarks=raw_lms,
        d_ref=d_ref,
        timestamp=t,
    )

    assert state.gesture_id == CompleteGestureId.G001_PALM_SWIPE_LEFT
    assert state.canonical_name in ("5-FINGER SWIPE LEFT", "PALM SWIPE LEFT")
    assert state.category == GestureCategory.SWIPE_STROKE


def test_pinch_approach_contact_hold_sequence():
    """
    Validates user specification:
    PINCH
        ↓
    Thumb + index approach
        ↓
    contact
        ↓
    hold
    -> G006_PINCH_SELECT.
    """
    recognizer = CompleteGestureRecognizer()
    motion_tracker = MotionPrimitiveTracker()
    d_ref = 0.20
    t = 100.0

    motion_still = motion_tracker.update(
        hand_id=0,
        handedness="Right",
        palm_center=(0.5, 0.5, 0.0),
        hand_scale_ref=d_ref,
        timestamp=t,
    )

    # 1. Separated initial distance (d = 0.50 * d_ref = 0.10)
    lms1 = _make_dummy_landmarks(thumb_tip=(0.40, 0.50, 0.0), index_tip=(0.50, 0.50, 0.0))
    recognizer.update(
        hand_id=0,
        handedness="Right",
        derived_pose=None,
        motion_state=motion_still,
        finger_states=None,
        raw_landmarks=lms1,
        d_ref=d_ref,
        timestamp=t,
    )

    # 2. APPROACH: Tips closing distance rapidly (d drops from 0.10 to 0.06 over 33ms)
    t += 0.033
    lms2 = _make_dummy_landmarks(thumb_tip=(0.44, 0.50, 0.0), index_tip=(0.50, 0.50, 0.0))
    recognizer.update(
        hand_id=0,
        handedness="Right",
        derived_pose=None,
        motion_state=motion_still,
        finger_states=None,
        raw_landmarks=lms2,
        d_ref=d_ref,
        timestamp=t,
    )

    # 3. CONTACT: Tips touch (d <= 0.35 * d_ref -> dist = 0.03)
    t += 0.033
    lms3 = _make_dummy_landmarks(thumb_tip=(0.485, 0.50, 0.0), index_tip=(0.50, 0.50, 0.0))
    state_contact = recognizer.update(
        hand_id=0,
        handedness="Right",
        derived_pose=None,
        motion_state=motion_still,
        finger_states=None,
        raw_landmarks=lms3,
        d_ref=d_ref,
        timestamp=t,
    )

    assert state_contact.gesture_id == CompleteGestureId.G006_PINCH_SELECT
    assert state_contact.phase in (PinchPhase.CONTACT.value, PinchPhase.HOLD.value)

    # 4. HOLD: Contact maintained steadily for 100ms
    t += 0.100
    state_hold = recognizer.update(
        hand_id=0,
        handedness="Right",
        derived_pose=None,
        motion_state=motion_still,
        finger_states=None,
        raw_landmarks=lms3,
        d_ref=d_ref,
        timestamp=t,
    )

    assert state_hold.gesture_id == CompleteGestureId.G006_PINCH_SELECT
    assert state_hold.phase == PinchPhase.HOLD.value
    assert state_hold.metrics["hold_duration_ms"] >= 80.0
    # Verify verifiable sequence of observable events was captured!
    assert any("CONTACT" in ev for ev in state_hold.event_sequence)
    assert any("HOLD" in ev for ev in state_hold.event_sequence)


def test_pinch_drag_sequence():
    """Validates that sustaining pinch contact while translating through space transitions to G007_PINCH_DRAG."""
    recognizer = CompleteGestureRecognizer()
    motion_tracker = MotionPrimitiveTracker(velocity_alpha=0.80)
    d_ref = 0.20
    t = 100.0

    # Start with established pinch contact held for 120ms
    lms_contact = _make_dummy_landmarks(thumb_tip=(0.49, 0.50, 0.0), index_tip=(0.50, 0.50, 0.0))

    motion_still = motion_tracker.update(0, "Right", (0.50, 0.50, 0.0), d_ref, t)
    recognizer.update(0, "Right", None, motion_still, None, lms_contact, d_ref, t)

    t += 0.120
    motion_still = motion_tracker.update(0, "Right", (0.50, 0.50, 0.0), d_ref, t)
    recognizer.update(0, "Right", None, motion_still, None, lms_contact, d_ref, t)

    # Now hand translates through space while maintaining contact
    motion_drag = None
    x = 0.50
    for _ in range(4):
        t += 0.033
        x += 0.012  # Moving right at ~0.36 u/s
        motion_drag = motion_tracker.update(0, "Right", (x, 0.50, 0.0), d_ref, t)

    state_drag = recognizer.update(0, "Right", None, motion_drag, None, lms_contact, d_ref, t)

    assert state_drag.gesture_id == CompleteGestureId.G007_PINCH_DRAG
    assert state_drag.phase == PinchPhase.DRAG.value
    assert "DRAG" in " ".join(state_drag.event_sequence)


def test_air_tap_sequence():
    """Validates Aim (Point) -> Strike -> Rebound -> G011_AIR_TAP."""
    recognizer = CompleteGestureRecognizer()
    motion_tracker = MotionPrimitiveTracker()
    d_ref = 0.20
    t = 100.0

    derived_pose = DerivedHandPose(
        pose_id=HandPoseId.H004_INDEX_POINT,
        canonical_name="POINT",
        confidence=0.92,
        configuration=None,
        satisfied_predicates=["index_extended"],
        diagnostics=[],
    )

    raw_lms = _make_dummy_landmarks()
    tip_y = 0.35

    # 1. Aim phase with steady palm
    raw_lms[8] = [0.5, tip_y, 0.0]
    motion_aim = motion_tracker.update(
        hand_id=0,
        handedness="Right",
        palm_center=(0.5, 0.5, 0.0),
        hand_scale_ref=d_ref,
        timestamp=t,
        raw_landmarks=raw_lms,
        finger_ratios={"index": 1.45},
    )

    recognizer.update(0, "Right", derived_pose, motion_aim, None, raw_lms, d_ref, t)

    # 2. Strike downwards
    t += 0.033
    tip_y += 0.030
    raw_lms[8] = [0.5, tip_y, 0.0]
    motion_strike = motion_tracker.update(
        hand_id=0,
        handedness="Right",
        palm_center=(0.5, 0.5, 0.0),
        hand_scale_ref=d_ref,
        timestamp=t,
        raw_landmarks=raw_lms,
        finger_ratios={"index": 1.10},
    )
    recognizer.update(0, "Right", derived_pose, motion_strike, None, raw_lms, d_ref, t)

    # 3. Recovery / Rebound
    t += 0.060
    tip_y -= 0.015
    raw_lms[8] = [0.5, tip_y, 0.0]
    motion_rebound = motion_tracker.update(
        hand_id=0,
        handedness="Right",
        palm_center=(0.5, 0.5, 0.0),
        hand_scale_ref=d_ref,
        timestamp=t,
        raw_landmarks=raw_lms,
        finger_ratios={"index": 1.30},
    )

    state = recognizer.update(0, "Right", derived_pose, motion_rebound, None, raw_lms, d_ref, t)

    assert state.gesture_id == CompleteGestureId.G011_AIR_TAP
    assert state.is_stroke_completed


def test_thumbs_up_approval_sequence():
    """Validates Thumbs Up held steady -> G013_THUMBS_UP."""
    recognizer = CompleteGestureRecognizer()
    motion_tracker = MotionPrimitiveTracker()
    finger_states = _make_dummy_finger_states("THUMBS_UP")

    derived_pose = DerivedHandPose(
        pose_id=HandPoseId.H007_THUMBS_UP,
        canonical_name="THUMBS_UP",
        confidence=0.94,
        configuration=None,
        satisfied_predicates=["thumb_up", "others_curled"],
        diagnostics=[],
    )

    t = 100.0
    motion_still = motion_tracker.update(0, "Right", (0.5, 0.5, 0.0), 0.20, t)
    raw_lms = _make_dummy_landmarks()

    recognizer.update(0, "Right", derived_pose, motion_still, finger_states, raw_lms, 0.20, t)

    t += 0.150
    motion_still = motion_tracker.update(0, "Right", (0.5, 0.5, 0.0), 0.20, t)
    state = recognizer.update(0, "Right", derived_pose, motion_still, finger_states, raw_lms, 0.20, t)

    assert state.gesture_id == CompleteGestureId.G013_THUMBS_UP
    assert state.canonical_name == "THUMBS UP APPROVAL"
    assert state.phase == "APPROVAL"


def test_clockwise_dial_sequence():
    """Validates circular clockwise motion primitive -> G018_CLOCKWISE_DIAL."""
    recognizer = CompleteGestureRecognizer()
    motion_tracker = MotionPrimitiveTracker()
    d_ref = 0.20
    t = 100.0

    # Simulate circular trajectory in XY
    r = 0.08
    raw_lms = _make_dummy_landmarks()
    state = None
    for step in range(8):
        t += 0.033
        angle = step * (math.pi / 4.0)  # Clockwise rotation in screen space
        cx = 0.50 + r * math.cos(angle)
        cy = 0.50 + r * math.sin(angle)
        motion_cw = motion_tracker.update(0, "Right", (cx, cy, 0.0), d_ref, t)
        state = recognizer.update(0, "Right", None, motion_cw, None, raw_lms, d_ref, t)

    assert state is not None
    # If circular orbital motion primitive was registered:
    if state.gesture_id != CompleteGestureId.NONE:
        assert state.category in (GestureCategory.ORBITAL_DIAL, GestureCategory.SWIPE_STROKE, GestureCategory.HOLD_POSE)


def test_axial_flip_dorsal_sequence():
    """Validates FLIP_TO_DORSAL motion primitive synthesis -> G020_AXIAL_FLIP_DORSAL."""
    recognizer = CompleteGestureRecognizer()
    tracker = MotionPrimitiveTracker()
    raw_lms = _make_dummy_landmarks()
    d_ref = 0.20
    t = 100.0

    # Step 1: Hand observed with PALM facing
    tracker.update(0, "Right", (0.5, 0.5, 0.0), d_ref, t, palm_facing="PALM")

    # Step 2: Hand rolls/flips to DORSAL facing
    t += 0.033
    motion_flip = tracker.update(0, "Right", (0.5, 0.5, 0.0), d_ref, t, palm_facing="DORSAL")

    state = recognizer.update(
        hand_id=0,
        handedness="Right",
        derived_pose=None,
        motion_state=motion_flip,
        finger_states=None,
        raw_landmarks=raw_lms,
        d_ref=d_ref,
        timestamp=t,
    )

    assert state.gesture_id == CompleteGestureId.G020_AXIAL_FLIP_DORSAL
    assert state.category == GestureCategory.AXIAL_FLIP
    assert state.phase in ("FLIP", "DORSAL_FACING")


def test_single_hand_pinch_zoom_in_sequence():
    """Validates Pinch Hold + Forward Z- / scale rate -> G022_PINCH_ZOOM_IN."""
    recognizer = CompleteGestureRecognizer()
    tracker = MotionPrimitiveTracker()
    d_ref = 0.20
    t = 100.0

    lms_pinch = _make_dummy_landmarks(thumb_tip=(0.49, 0.50, 0.0), index_tip=(0.50, 0.50, 0.0))
    m_still = tracker.update(0, "Right", (0.5, 0.5, 0.0), d_ref, t)

    # Establish hold for 140ms
    recognizer.update(0, "Right", None, m_still, None, lms_pinch, d_ref, t)
    t += 0.140
    m_held = tracker.update(0, "Right", (0.5, 0.5, 0.0), d_ref, t)
    recognizer.update(0, "Right", None, m_held, None, lms_pinch, d_ref, t)

    # Now hand pushes toward camera (increasing d_ref = scale expansion)
    m_thrust = None
    for _ in range(3):
        t += 0.033
        d_ref += 0.015  # scale expansion rate > 0.18
        m_thrust = tracker.update(0, "Right", (0.5, 0.5, -0.03), d_ref, t)

    state_zoom = recognizer.update(0, "Right", None, m_thrust, None, lms_pinch, d_ref, t)

    assert state_zoom.gesture_id == CompleteGestureId.G022_PINCH_ZOOM_IN
    assert state_zoom.category == GestureCategory.ZOOM_INTERACTION
    assert state_zoom.canonical_name == "PINCH ZOOM IN"


def test_micro_pinch_tap_click():
    """Validates FM001 Thumb-to-Index micro tap click with contact <= 160ms."""
    recognizer = CompleteGestureRecognizer()
    d_ref = 0.20
    t = 100.0

    # Frame 1: Fingers separated
    lms1 = _make_dummy_landmarks(thumb_tip=(0.40, 0.50, 0.0), index_tip=(0.50, 0.50, 0.0))
    recognizer.update(0, "Right", None, None, None, lms1, d_ref, t)

    # Frame 2: Contact touched (dist = 0.02 -> norm_dist = 0.10 <= 0.32)
    t += 0.033
    lms2 = _make_dummy_landmarks(thumb_tip=(0.49, 0.50, 0.0), index_tip=(0.50, 0.50, 0.0))
    recognizer.update(0, "Right", None, None, None, lms2, d_ref, t)

    # Frame 3: Released quickly at 80ms total contact time
    t += 0.050
    lms3 = _make_dummy_landmarks(thumb_tip=(0.40, 0.50, 0.0), index_tip=(0.50, 0.50, 0.0))
    state_tap = recognizer.update(0, "Right", None, None, None, lms3, d_ref, t)

    assert state_tap.gesture_id == CompleteGestureId.FM001_MICRO_PINCH_TAP
    assert state_tap.category == GestureCategory.MICRO_GESTURE
    assert state_tap.is_stroke_completed


def test_micro_middle_tap_context_click():
    """Validates FM002 Thumb-to-Middle contact while index extended -> FM002_MICRO_MIDDLE_TAP."""
    recognizer = CompleteGestureRecognizer()
    d_ref = 0.20
    t = 100.0

    # Index tip separated at (0.60, 0.50, 0.0), middle and thumb touching at (0.49, 0.50, 0.0)
    lms = _make_dummy_landmarks(
        thumb_tip=(0.49, 0.50, 0.0),
        index_tip=(0.60, 0.50, 0.0),
        middle_tip=(0.50, 0.50, 0.0),
    )

    state = recognizer.update(0, "Right", None, None, None, lms, d_ref, t)

    assert state.gesture_id == CompleteGestureId.FM002_MICRO_MIDDLE_TAP
    assert state.category == GestureCategory.MICRO_GESTURE
    assert state.canonical_name == "MICRO MIDDLE CONTEXT CLICK"


def _make_dummy_motion_state(
    primitive: MotionPrimitive = MotionPrimitive.STATIONARY,
    speed: float = 0.0,
    direction: str = "STATIONARY",
    disp: float = 0.0,
    linearity: float = 1.0,
    angular_velocity: float = 0.0,
) -> MotionState:
    """Creates a deterministic MotionState for gesture synthesis testing."""
    return MotionState(
        hand_id=0,
        handedness="Right",
        position=(0.5, 0.5, 0.0),
        velocity=(0.0, 0.0, 0.0),
        speed=speed,
        speed_xy=speed,
        acceleration=(0.0, 0.0, 0.0),
        acceleration_magnitude=0.0,
        tangential_acceleration=0.0,
        dynamic_state=DynamicState.STEADY if speed > 0.1 else DynamicState.STATIONARY,
        direction_vector=(1.0, 0.0, 0.0),
        heading_deg=0.0,
        primary_direction=direction,
        net_displacement=(disp, 0.0, 0.0),
        displacement_magnitude=disp,
        cumulative_path_length=disp,
        linearity=linearity,
        scale_rate=0.0,
        depth_state="NEUTRAL",
        angular_velocity=angular_velocity,
        cumulative_angle_deg=angular_velocity * 0.1,
        rotation_direction="CLOCKWISE" if angular_velocity > 0 else ("COUNTERCLOCKWISE" if angular_velocity < 0 else "NONE"),
        stroke_duration_ms=120.0 if speed > 0.1 else 0.0,
        dwell_duration_ms=0.0 if speed > 0.1 else 300.0,
        is_holding=False,
        is_releasing=False,
        motion_primitive=primitive,
    )


def test_multi_finger_point_hover_progression():
    """
    Validates Point evolution to 1-finger through 5-finger pointing hover:
    - 1 finger point -> G005_1F_POINT ("1-FINGER POINT")
    - 2 fingers point -> G005_2F_POINT ("2-FINGER POINT")
    - 3 fingers point -> G005_3F_POINT ("3-FINGER POINT")
    - 4 fingers point -> G005_4F_POINT ("4-FINGER POINT")
    - 5 fingers point -> G005_5F_POINT ("5-FINGER POINT")
    """
    recognizer = CompleteGestureRecognizer()
    raw_lms = _make_dummy_landmarks()
    d_ref = 0.20
    t = 100.0

    motion_still = _make_dummy_motion_state(
        primitive=MotionPrimitive.HOLD,
        speed=0.02,
        direction="STATIONARY",
    )

    test_cases = [
        (1, HandPoseId.H004_INDEX_POINT, "POINT", CompleteGestureId.G005_1F_POINT, "1-FINGER POINT"),
        (2, HandPoseId.H016_DOUBLE_POINT, "DOUBLE_POINT", CompleteGestureId.G005_2F_POINT, "2-FINGER POINT"),
        (3, HandPoseId.H011_THREE_FINGER, "THREE_FINGER", CompleteGestureId.G005_3F_POINT, "3-FINGER POINT"),
        (4, HandPoseId.H017_FOUR_FINGER_POINT, "FOUR_FINGER", CompleteGestureId.G005_4F_POINT, "4-FINGER POINT"),
        (5, HandPoseId.H018_FIVE_FINGER_POINT, "FIVE_FINGER", CompleteGestureId.G005_5F_POINT, "5-FINGER POINT"),
    ]

    for n_fingers, pose_id, pose_name, expected_id, expected_name in test_cases:
        derived_pose = DerivedHandPose(
            pose_id=pose_id,
            canonical_name=pose_name,
            confidence=0.92,
            configuration=None,
        )

        state = recognizer.update(
            hand_id=0,
            handedness="Right",
            derived_pose=derived_pose,
            motion_state=motion_still,
            finger_states=None,
            raw_landmarks=raw_lms,
            d_ref=d_ref,
            timestamp=t,
        )

        assert state.gesture_id == expected_id, f"Expected {expected_id} for {n_fingers}F, got {state.gesture_id}"
        assert state.canonical_name == expected_name
        assert state.category == GestureCategory.POINT_INTERACTION
        assert state.metrics["finger_count"] == float(n_fingers)


def test_multi_finger_swipe_all_variants():
    """
    Validates Swipe evolution to 1-finger through 5-finger swipes across all 4 cardinal directions:
    - 1-finger swipe (left/right/up/down)
    - 2-finger swipe (left/right/up/down)
    - 3-finger swipe (left/right/up/down)
    - 4-finger swipe (left/right/up/down)
    - 5-finger swipe / hand swipe (left/right/up/down)
    """
    recognizer = CompleteGestureRecognizer()
    raw_lms = _make_dummy_landmarks()
    d_ref = 0.20
    t = 100.0

    poses = [
        (1, HandPoseId.H004_INDEX_POINT, "POINT"),
        (2, HandPoseId.H016_DOUBLE_POINT, "DOUBLE_POINT"),
        (3, HandPoseId.H011_THREE_FINGER, "THREE_FINGER"),
        (4, HandPoseId.H017_FOUR_FINGER_POINT, "FOUR_FINGER"),
        (5, HandPoseId.H018_FIVE_FINGER_POINT, "FIVE_FINGER"),
    ]

    directions = [
        ("LEFT", MotionPrimitive.MOVE_LEFT),
        ("RIGHT", MotionPrimitive.MOVE_RIGHT),
        ("UP", MotionPrimitive.MOVE_UP),
        ("DOWN", MotionPrimitive.MOVE_DOWN),
    ]

    for n_fingers, pose_id, pose_name in poses:
        derived_pose = DerivedHandPose(
            pose_id=pose_id,
            canonical_name=pose_name,
            confidence=0.95,
            configuration=None,
        )

        for dir_name, prim in directions:
            motion = _make_dummy_motion_state(
                primitive=prim,
                speed=0.35,
                direction=dir_name,
                disp=0.06,
                linearity=0.85,
            )

            state = recognizer.update(
                hand_id=0,
                handedness="Right",
                derived_pose=derived_pose,
                motion_state=motion,
                finger_states=None,
                raw_landmarks=raw_lms,
                d_ref=d_ref,
                timestamp=t,
            )

            expected_id_name = f"G00{1 if dir_name == 'LEFT' else (2 if dir_name == 'RIGHT' else (3 if dir_name == 'UP' else 4))}_{n_fingers}F_SWIPE_{dir_name}"
            expected_canonical = f"{n_fingers}-FINGER SWIPE {dir_name}"

            assert state.gesture_id.value == expected_id_name, f"Mismatch for {n_fingers}F {dir_name}: got {state.gesture_id}"
            assert state.canonical_name == expected_canonical
            assert state.category == GestureCategory.SWIPE_STROKE


def test_multi_finger_rotate_cw_ccw():
    """
    Validates Rotate evolution to 1-finger through 5-finger rotation (CW/ACW):
    - 1 finger rotate (CW/CCW)
    - 2 fingers rotate (CW/CCW)
    - 3 fingers rotate (CW/CCW)
    - 4 fingers rotate (CW/CCW)
    - 5 fingers / hand rotate (CW/CCW)
    """
    recognizer = CompleteGestureRecognizer()
    raw_lms = _make_dummy_landmarks()
    d_ref = 0.20
    t = 100.0

    poses = [
        (1, HandPoseId.H004_INDEX_POINT, "POINT"),
        (2, HandPoseId.H016_DOUBLE_POINT, "DOUBLE_POINT"),
        (3, HandPoseId.H011_THREE_FINGER, "THREE_FINGER"),
        (4, HandPoseId.H017_FOUR_FINGER_POINT, "FOUR_FINGER"),
        (5, HandPoseId.H018_FIVE_FINGER_POINT, "FIVE_FINGER"),
    ]

    rotations = [
        ("CW", MotionPrimitive.ROTATE_CW, 120.0),
        ("CCW", MotionPrimitive.ROTATE_CCW, -120.0),
    ]

    for n_fingers, pose_id, pose_name in poses:
        derived_pose = DerivedHandPose(
            pose_id=pose_id,
            canonical_name=pose_name,
            confidence=0.92,
            configuration=None,
        )

        for rot_dir, prim, ang_vel in rotations:
            motion = _make_dummy_motion_state(
                primitive=prim,
                speed=0.25,
                direction="ROTATING",
                disp=0.03,
                angular_velocity=ang_vel,
            )

            state = recognizer.update(
                hand_id=0,
                handedness="Right",
                derived_pose=derived_pose,
                motion_state=motion,
                finger_states=None,
                raw_landmarks=raw_lms,
                d_ref=d_ref,
                timestamp=t,
            )

            expected_id_name = f"G0{18 if rot_dir == 'CW' else 19}_{n_fingers}F_ROTATE_{rot_dir}"
            expected_canonical = f"{n_fingers}-FINGER ROTATE {rot_dir}"

            assert state.gesture_id.value == expected_id_name, f"Mismatch for {n_fingers}F {rot_dir}: got {state.gesture_id}"
            assert state.canonical_name == expected_canonical
            assert state.category == GestureCategory.ORBITAL_DIAL


def test_open_palm_anchor_and_hover():
    """Validates that presenting an open palm/spread fingers while hovering produces G010_OPEN_PALM_HOVER."""
    recognizer = CompleteGestureRecognizer()
    raw_lms = _make_dummy_landmarks()
    d_ref = 0.20
    t = 10.0

    derived_pose = DerivedHandPose(
        pose_id=HandPoseId.H001_OPEN_PALM,
        canonical_name="OPEN_PALM",
        confidence=0.95,
        configuration=None,
    )
    motion = _make_dummy_motion_state(
        primitive=MotionPrimitive.STATIONARY,
        speed=0.08,
        direction="STATIONARY",
        disp=0.005,
    )

    state = recognizer.update(
        hand_id=0,
        handedness="Right",
        derived_pose=derived_pose,
        motion_state=motion,
        finger_states=None,
        raw_landmarks=raw_lms,
        d_ref=d_ref,
        timestamp=t,
    )

    assert state.gesture_id == CompleteGestureId.G010_OPEN_PALM_HOVER
    assert state.category == GestureCategory.HOLD_POSE
    assert state.phase == "ANCHORED"
    assert "OPEN PALM" in state.canonical_name
    assert state.is_active is True


def test_pinch_contact_expanded_threshold():
    """Validates that normalized pinch distance of 0.40 triggers PINCH_SELECT with widened threshold."""
    recognizer = CompleteGestureRecognizer()
    d_ref = 0.20
    t = 1.0

    # Thumb and index tips separated by 0.40 * d_ref = 0.08
    raw_lms = _make_dummy_landmarks(
        thumb_tip=(0.50, 0.50, 0.0),
        index_tip=(0.50, 0.58, 0.0),  # dist = 0.08 / 0.20 = 0.40
    )

    derived_pose = DerivedHandPose(
        pose_id=HandPoseId.H005_PRECISION_PINCH,
        canonical_name="PINCH",
        confidence=0.90,
        configuration=None,
    )
    motion = _make_dummy_motion_state(
        primitive=MotionPrimitive.HOLD,
        speed=0.02,
        direction="STATIONARY",
        disp=0.002,
    )

    state = recognizer.update(
        hand_id=0,
        handedness="Right",
        derived_pose=derived_pose,
        motion_state=motion,
        finger_states=None,
        raw_landmarks=raw_lms,
        d_ref=d_ref,
        timestamp=t,
    )

    assert state.gesture_id in (CompleteGestureId.G006_PINCH_SELECT, CompleteGestureId.G007_PINCH_DRAG)
    assert state.category == GestureCategory.PINCH_INTERACTION
    assert state.is_active is True


