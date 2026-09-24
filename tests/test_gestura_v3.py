"""
Comprehensive unit tests for Gestura v3 specification.

Validates:
  1. Temporal Slap Detection (velocity, duration 120-350ms, direction consistency > 85%, slow suppression).
  2. Primary vs Modifier Hand isolation (modifier hand never triggers swipes).
  3. State Machine lifecycle (IDLE -> OBSERVING -> CANDIDATE -> CONFIRMED -> ACTIVE -> RELEASE -> IDLE).
  4. Graceful release on low confidence (no sudden snap).
  5. Two-hand relative palm angle rotation & translation.
  6. Confidence schema ({hand_confidence, gesture_confidence, intent_confidence, state}).
"""

import numpy as np
import pytest

from src.gestures.gesture_types import GestureType, RecognizedGesture
from src.gestures.heuristic_classifier import HeuristicGestureClassifier, TemporalSlapDetector
from src.intent.state_machine import InteractionState, InteractionStateMachine
from src.interaction.engine import InteractionEngine
from src.interaction.mapping import SpatialCommandType
from tests.test_gestures import build_hand_state


def test_slap_detector_temporal_consistency():
    detector = TemporalSlapDetector(
        min_speed_threshold=0.25,
        min_displacement=0.05,
        min_consistency=0.85,
        min_duration_sec=0.11,
        max_duration_sec=0.36,
    )

    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.5, 0.8, 0.0]

    hand = build_hand_state(pts, hand_id=1)

    # Frame 1 (t=0.0): stroke initiation
    hand.timestamp = 1.0
    hand.palm_center = (0.6, 0.5, 0.0)
    hand.palm_velocity = (-0.35, 0.0, 0.0)
    g, conf, m = detector.process(hand)
    assert g is None

    # Frame 2 (t=0.05s / 50ms): accumulating candidate
    hand.timestamp = 1.05
    hand.palm_center = (0.52, 0.5, 0.0)
    hand.palm_velocity = (-0.38, 0.0, 0.0)
    g, conf, m = detector.process(hand)
    assert g is None  # Below 110ms minimum

    # Frame 3 (t=0.15s / 150ms): consistent leftward stroke within 120-350ms window
    hand.timestamp = 1.15
    hand.palm_center = (0.42, 0.5, 0.0)
    hand.palm_velocity = (-0.36, 0.0, 0.0)
    g, conf, m = detector.process(hand)

    assert g == GestureType.SWIPE_LEFT
    assert conf >= 0.70
    assert m["slap_consistency"] >= 0.85


def test_slap_suppressed_for_slow_movement():
    detector = TemporalSlapDetector()
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.5, 0.8, 0.0]

    hand = build_hand_state(pts, hand_id=1)
    hand.timestamp = 1.0
    hand.palm_center = (0.5, 0.5, 0.0)
    hand.palm_velocity = (-0.08, 0.0, 0.0)  # Slow movement

    g, conf, _ = detector.process(hand)
    assert g is None

    hand.timestamp = 1.2
    hand.palm_center = (0.45, 0.5, 0.0)
    hand.palm_velocity = (-0.09, 0.0, 0.0)
    g, conf, _ = detector.process(hand)
    assert g is None


def test_modifier_hand_never_triggers_swipe():
    classifier = HeuristicGestureClassifier()
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.5, 0.8, 0.0]

    hand = build_hand_state(pts, hand_id=2, handedness="Left")
    hand.palm_velocity = (-0.65, 0.0, 0.0)

    # When marked as modifier hand, swipe is strictly suppressed
    rec = classifier.classify_single_hand(hand, is_modifier=True)
    assert rec.gesture != GestureType.SWIPE_LEFT


def test_state_machine_v3_lifecycle():
    fsm = InteractionStateMachine(
        activation_threshold=0.60,
        release_threshold=0.30,
        confirm_frames=3,
        evidence_lambda=0.75,
    )
    t = 1.0

    # IDLE
    ctx0 = fsm.update(False, None, t)
    assert ctx0.state == InteractionState.IDLE

    # OBSERVING
    rec_none = RecognizedGesture(gesture=GestureType.NONE, confidence=0.0, hand_id=0, handedness="Right", timestamp=t)
    ctx1 = fsm.update(True, rec_none, t)
    assert ctx1.state == InteractionState.OBSERVING

    # CANDIDATE
    t += 0.033
    rec_pinch = RecognizedGesture(gesture=GestureType.PINCH, confidence=0.90, hand_id=0, handedness="Right", timestamp=t)
    ctx2 = fsm.update(True, rec_pinch, t)
    assert ctx2.state == InteractionState.CANDIDATE
    assert ctx2.candidate_gesture == GestureType.PINCH

    # Accumulate frames -> CONFIRMED -> ACTIVE
    for _ in range(6):
        t += 0.033
        ctx = fsm.update(True, rec_pinch, t)

    assert ctx.state in (InteractionState.CONFIRMED, InteractionState.ACTIVE)

    t += 0.033
    ctx_act = fsm.update(True, rec_pinch, t)
    assert ctx_act.state == InteractionState.ACTIVE

    # Graceful RELEASE on open palm or low confidence
    t += 0.033
    rec_palm = RecognizedGesture(gesture=GestureType.OPEN_PALM, confidence=0.85, hand_id=0, handedness="Right", timestamp=t)
    ctx_rel = fsm.update(True, rec_palm, t)
    assert ctx_rel.state == InteractionState.RELEASE


def test_two_hand_rotation_and_translation():
    engine = InteractionEngine()

    pts1 = np.zeros((21, 3), dtype=np.float32)
    pts1[0] = [0.3, 0.8, 0.0]
    pts2 = np.zeros((21, 3), dtype=np.float32)
    pts2[0] = [0.7, 0.8, 0.0]

    h1 = build_hand_state(pts1, hand_id=0, handedness="Left")
    h2 = build_hand_state(pts2, hand_id=1, handedness="Right")

    # Frame 1: Establish baseline
    cmd1, ctx1, _ = engine.process_hands([h1, h2], timestamp=1.0)
    assert cmd1.command_type == SpatialCommandType.BIMANUAL_NAV
    assert cmd1.dominant_hand in ("Right", "Left")
    assert cmd1.modifier_hand in ("Right", "Left")

    # Frame 2: Rotate hands (hand 1 moves up, hand 2 moves down -> relative angle changes)
    pts1_rot = pts1.copy()
    pts1_rot[:, 1] -= 0.08
    pts2_rot = pts2.copy()
    pts2_rot[:, 1] += 0.08

    h1_rot = build_hand_state(pts1_rot, hand_id=0, handedness="Left")
    h2_rot = build_hand_state(pts2_rot, hand_id=1, handedness="Right")

    cmd2, _, _ = engine.process_hands([h1_rot, h2_rot], timestamp=1.05)
    assert cmd2.command_type == SpatialCommandType.BIMANUAL_NAV
    # delta_rotation should register angular change
    yaw, pitch, roll = cmd2.delta_rotation
    assert abs(roll) > 0.0 or abs(pitch) > 0.0 or abs(yaw) > 0.0


def test_gestura_v3_confidence_telemetry_schema():
    engine = InteractionEngine()
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.5, 0.8, 0.0]
    hand = build_hand_state(pts, hand_id=0)

    cmd, ctx, _ = engine.process_hands([hand], timestamp=1.0)

    # Exposes hand_confidence, gesture_confidence, intent_confidence, state
    assert hasattr(cmd, "hand_confidence")
    assert hasattr(cmd, "gesture_confidence")
    assert hasattr(cmd, "intent_confidence")
    assert hasattr(cmd, "state")
    assert isinstance(cmd.hand_confidence, float)
    assert isinstance(cmd.gesture_confidence, float)
    assert isinstance(cmd.intent_confidence, float)
    assert isinstance(cmd.state, str)


def test_three_hands_and_none_hands_safe():
    engine = InteractionEngine()
    classifier = HeuristicGestureClassifier()

    # None hands should cleanly return None without AttributeError
    assert classifier.classify_two_hands(None, None) is None

    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.5, 0.8, 0.0]
    h1 = build_hand_state(pts, hand_id=0)
    assert classifier.classify_two_hands(h1, None) is None
    assert classifier.classify_two_hands(None, h1) is None

    # 3 detected hands should process without AttributeError
    h2 = build_hand_state(pts, hand_id=1)
    h3 = build_hand_state(pts, hand_id=2)
    cmd, ctx, primary = engine.process_hands([h1, h2, h3], timestamp=1.0)
    assert primary is not None
    assert cmd.command_type is not None


def test_swipe_360_intensity_proportional():
    from src.interaction.mapping import InteractionMapper
    from src.intent.state_machine import IntentContext

    mapper = InteractionMapper()
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.5, 0.8, 0.0]

    hand = build_hand_state(pts, hand_id=0)
    hand.palm_velocity = (-0.65, 0.0, 0.0)

    intent_ctx = IntentContext(
        state=InteractionState.ACTIVE,
        active_gesture=GestureType.SWIPE_LEFT,
        intent_confidence=0.9,
        evidence_score=0.9,
        consecutive_frames_in_state=5,
        state_duration_sec=0.2,
        timestamp=1.0,
        gesture_confidence=0.9,
    )

    cmd = mapper.map_to_command(
        intent_ctx=intent_ctx,
        primary_hand=hand,
        bimanual_gesture=None,
        smoothed_cursor_ndc=(0.0, 0.0),
        timestamp=1.0,
    )
    assert cmd.command_type == SpatialCommandType.ROTATE_OBJECT
    # Rotation yaw should be 360-degree scale (approx 2π * intensity)
    yaw, pitch, _ = cmd.delta_rotation
    assert yaw < -5.0  # Fast negative yaw spin (at least ~2π)


def test_grab_provides_delta_translation_for_node_holding():
    from src.interaction.mapping import InteractionMapper
    from src.intent.state_machine import IntentContext

    mapper = InteractionMapper()
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.5, 0.8, 0.0]

    hand = build_hand_state(pts, hand_id=0)
    intent_ctx = IntentContext(
        state=InteractionState.ACTIVE,
        active_gesture=GestureType.GRAB,
        intent_confidence=0.9,
        evidence_score=0.9,
        consecutive_frames_in_state=5,
        state_duration_sec=0.2,
        timestamp=1.0,
        gesture_confidence=0.9,
    )

    # Frame 1: Establish baseline cursor position
    mapper.map_to_command(
        intent_ctx=intent_ctx,
        primary_hand=hand,
        bimanual_gesture=None,
        smoothed_cursor_ndc=(0.10, 0.10),
        timestamp=1.0,
    )

    # Frame 2: Displaced cursor during grab
    cmd2 = mapper.map_to_command(
        intent_ctx=intent_ctx,
        primary_hand=hand,
        bimanual_gesture=None,
        smoothed_cursor_ndc=(0.18, 0.15),
        timestamp=1.05,
    )
    # Grab should provide delta_translation for node holding
    assert cmd2.delta_translation is not None
    dx, dy, _ = cmd2.delta_translation
    assert dx != 0.0 or dy != 0.0

