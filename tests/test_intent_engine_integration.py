"""
End-to-end integration tests verifying that the Temporal Intent Engine
correctly resolves candidates from static poses and dynamic complete gestures,
accumulates evidence without tearing, transitions FSM states, and harmonizes
with HandDetector and InteractionEngine.
"""

import numpy as np
import pytest

from src.gestures.complete_gesture import CompleteGestureId, CompleteGestureState, GestureCategory
from src.gestures.gesture_types import GestureType
from src.gestures.hand_pose import DerivedHandPose, FingerConfiguration, HandPoseId
from src.interaction.engine import InteractionEngine
from src.intent.state_machine import InteractionState
from src.intent.temporal_intent_config import TemporalIntentConfig
from src.intent.temporal_intent_engine import TemporalIntentEngine
from src.landmarks.finger_state import FingerName, FingerStateDetail, FingerStateEnum, HandFingerStates
from src.landmarks.hand_state import HandLandmark, HandState


def _make_landmarks(base_x: float = 0.5, base_y: float = 0.5) -> np.ndarray:
    pts = np.zeros((21, 3), dtype=np.float32)
    for i in range(21):
        pts[i] = [base_x + i * 0.01, base_y + i * 0.01, -0.02]
    return pts


def _make_hand_state(pose_name: str = "POINT", hand_id: int = 0) -> HandState:
    pts = _make_landmarks(0.5, 0.5)
    lms = [HandLandmark(index=i, x=float(pts[i, 0]), y=float(pts[i, 1]), z=float(pts[i, 2])) for i in range(21)]

    # Point finger configuration
    is_point = (pose_name == "POINT")
    fs = HandFingerStates(
        thumb=FingerStateDetail(FingerName.THUMB, FingerStateEnum.FOLDED, 0.95),
        index=FingerStateDetail(FingerName.INDEX, FingerStateEnum.EXTENDED if is_point else FingerStateEnum.FOLDED, 0.95),
        middle=FingerStateDetail(FingerName.MIDDLE, FingerStateEnum.FOLDED, 0.95),
        ring=FingerStateDetail(FingerName.RING, FingerStateEnum.FOLDED, 0.95),
        little=FingerStateDetail(FingerName.LITTLE, FingerStateEnum.FOLDED, 0.95),
    )
    fconf = FingerConfiguration(
        thumb=FingerStateEnum.FOLDED,
        index=FingerStateEnum.EXTENDED if is_point else FingerStateEnum.FOLDED,
        middle=FingerStateEnum.FOLDED,
        ring=FingerStateEnum.FOLDED,
        little=FingerStateEnum.FOLDED,
        num_extended_fingers=1 if is_point else 0,
        num_folded_fingers=4 if not is_point else 3,
        num_curved_fingers=0,
        num_uncertain_fingers=0,
        thumb_is_extended=False,
        thumb_is_folded=True,
        thumb_is_pinching=False,
    )
    derived = DerivedHandPose(
        pose_id=HandPoseId.H004_INDEX_POINT if is_point else HandPoseId.H003_CLOSED_FIST,
        canonical_name=pose_name,
        confidence=0.92,
        configuration=fconf,
    )
    return HandState(
        hand_id=hand_id,
        handedness="Right",
        landmarks=lms,
        raw_landmarks_array=pts,
        normalized_landmarks_array=pts,
        palm_center=(0.5, 0.5, -0.05),
        palm_velocity=(0.0, 0.0, 0.0),
        hand_scale_ref=0.20,
        orientation_angles=(0.0, 0.0, 0.0),
        finger_extension_ratios={},
        finger_flexion_angles={},
        pinch_distance=0.15,
        pinch_confidence=0.10,
        finger_states=fs,
        derived_pose=derived,
    )


def test_candidate_resolution_fallback_to_static_pose():
    """Verify that when CompleteGesture is NONE, static DerivedHandPose acts as candidate."""
    engine = TemporalIntentEngine()
    hand = _make_hand_state("POINT")

    # Complete gesture is idle/none
    cg = CompleteGestureState(
        gesture_id=CompleteGestureId.NONE,
        canonical_name="NEUTRAL TRANSITION",
        category=GestureCategory.IDLE,
        confidence=0.0,
        hand_pose_id="H004_INDEX_POINT",
        hand_pose_name="POINT",
        motion_primitive="STATIONARY",
        phase="NEUTRAL",
        event_sequence=["POINT", "STATIONARY"],
        metrics={"speed": 0.0},
        is_active=False,
        is_stroke_completed=False,
        timestamp=1.0,
    )

    # Resolution logic as in HandDetector
    if cg and cg.gesture_id != CompleteGestureId.NONE and cg.confidence > 0.0:
        cand_name = cg.gesture_id.value
        cand_conf = cg.confidence
    elif hand.derived_pose and hand.derived_pose.pose_id != HandPoseId.UNKNOWN:
        cand_name = hand.derived_pose.canonical_name
        cand_conf = hand.derived_pose.confidence
    else:
        cand_name = "NONE"
        cand_conf = 0.0

    assert cand_name == "POINT"
    assert cand_conf == 0.92

    # Stream frames through TemporalIntentEngine
    t = 1.0
    for i in range(12):
        t += 0.033
        _, _, _, ctx, telem = engine.process_hand_temporal(
            hand_id=hand.hand_id,
            handedness=hand.handedness,
            raw_landmarks=hand.raw_landmarks_array,
            instantaneous_finger_states=hand.finger_states,
            instantaneous_pose=hand.derived_pose,
            palm_center=hand.palm_center,
            d_ref=hand.hand_scale_ref,
            detection_confidence=0.95,
            candidate_gesture_name=cand_name,
            candidate_confidence=cand_conf,
            timestamp=t,
        )

    # After 12 consistent frames of POINT, intent engine MUST confirm and activate POINT
    assert ctx.state in (InteractionState.CONFIRMED, InteractionState.ACTIVE)
    assert ctx.active_gesture == GestureType.POINT
    assert telem.confirmed_gesture == "POINT"
    assert telem.intent_confidence >= 0.50


def test_interaction_engine_uses_temporal_intent_context():
    """Verify InteractionEngine uses primary_hand.intent_context directly."""
    temporal_engine = TemporalIntentEngine()
    interaction_engine = InteractionEngine()
    hand = _make_hand_state("POINT")

    # Accumulate evidence in temporal_intent_engine
    t = 1.0
    for i in range(12):
        t += 0.033
        _, conf_pose, _, ctx, telem = temporal_engine.process_hand_temporal(
            hand_id=hand.hand_id,
            handedness=hand.handedness,
            raw_landmarks=hand.raw_landmarks_array,
            instantaneous_finger_states=hand.finger_states,
            instantaneous_pose=hand.derived_pose,
            palm_center=hand.palm_center,
            d_ref=hand.hand_scale_ref,
            detection_confidence=0.95,
            candidate_gesture_name="POINT",
            candidate_confidence=0.92,
            timestamp=t,
        )

    # Attach to hand state as detector does
    hand.intent_context = ctx
    hand.temporal_telemetry = telem.as_dict()

    cmd, engine_ctx, primary = interaction_engine.process_hands([hand], timestamp=t)

    # InteractionEngine MUST match TemporalIntentEngine output
    assert engine_ctx.state == ctx.state
    assert engine_ctx.active_gesture == ctx.active_gesture
    assert engine_ctx.active_gesture == GestureType.POINT
    assert interaction_engine.fsm.state == ctx.state


def test_hand_detector_process_frame_end_to_end():
    """Verify HandDetector.process_frame processes detected hands without NameError."""
    from src.perception.hand_detector import HandDetector

    detector = HandDetector()

    class MockLandmark:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

        def HasField(self, field):
            return False

    class MockLandmarkList:
        def __init__(self):
            # Wrist at (0.5, 0.8), fingers pointing up
            self.landmark = [MockLandmark(0.5, 0.8 - i * 0.025, -0.02) for i in range(21)]

    class MockClassification:
        def __init__(self):
            self.label = "Right"
            self.score = 0.95

    class MockClassificationList:
        def __init__(self):
            self.classification = [MockClassification()]

    class MockResults:
        def __init__(self):
            self.multi_hand_landmarks = [MockLandmarkList()]
            self.multi_handedness = [MockClassificationList()]

    # Mock MediaPipe hands_detector.process and drawing to return simulated hand
    detector.hands_detector.process = lambda frame_rgb: MockResults()
    detector.mp_drawing.draw_landmarks = lambda *args, **kwargs: None

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    hands, annotated = detector.process_frame(frame, timestamp=1.0)

    assert len(hands) == 1
    h = hands[0]
    assert h.hand_id == 1  # Right hand mapped to hand_id 1
    assert h.handedness == "Right"
    assert h.derived_pose is not None
    assert h.temporal_telemetry is not None
    assert h.intent_context is not None
    assert "intent_layer" in h.temporal_telemetry


def test_hmi_packet_serialization_with_temporal_intent_and_complete_gesture():
    """Validates that HMIPacket serializes complete_gesture and temporal_intent telemetry cleanly."""
    from src.communication.protocol import HMIPacket, HandTelemetry, SpatialCommand
    from src.interaction.mapping import SpatialCommandType

    cmd = SpatialCommand(
        command_type=SpatialCommandType.IDLE,
        interaction_state="IDLE",
        cursor_ndc=(0.0, 0.0),
        dominant_hand="Right",
    )
    telem_hand = HandTelemetry(
        hand_id=1,
        handedness="Right",
        palm_center=(0.5, 0.5, 0.0),
        pinch_confidence=0.0,
        detection_confidence=0.95,
        landmarks_normalized=[[0.5, 0.5, 0.0] for _ in range(21)],
        complete_gesture_id="G010_OPEN_PALM_HOVER",
        complete_gesture_name="OPEN PALM ANCHOR & HOVER",
        gesture_category="HOLD_POSE",
        gesture_phase="ANCHORED",
        task_intent="IDLE_MONITORING",
        predicted_next_intent="NONE",
    )
    temporal_telem = {
        "finger_layer": {"is_stable": True, "confidence": 0.95},
        "pose_layer": {"active_pose": "OPEN_PALM", "is_confirmed": True, "confidence": 0.95},
        "motion_layer": {"primitive": "STATIONARY", "speed": 0.05, "is_intentional": False},
        "intent_layer": {
            "current_state": "OBSERVING",
            "candidate_gesture": "OPEN_PALM",
            "confirmed_gesture": "NONE",
            "is_locked": False,
        },
    }

    packet = HMIPacket(
        command=cmd,
        intent_state="OBSERVING",
        active_gesture="NONE",
        candidate_gesture="OPEN_PALM",
        dominant_hand="Right",
        intent_confidence=0.85,
        hands=[telem_hand],
        fps=60.0,
        latency_ms=1.5,
        temporal_intent=temporal_telem,
        timestamp=100.0,
    )

    json_str = packet.model_dump_json()
    assert "G010_OPEN_PALM_HOVER" in json_str
    assert "OPEN PALM ANCHOR & HOVER" in json_str
    assert "temporal_intent" in json_str
    assert "intent_layer" in json_str

