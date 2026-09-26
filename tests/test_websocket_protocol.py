"""
Unit tests for the WebSocket HMIPacket telemetry protocol.
"""

import json
import pytest

from src.communication.protocol import HandTelemetry, HMIPacket
from src.interaction.mapping import SpatialCommand, SpatialCommandType


def test_hmi_packet_serialization():
    cmd = SpatialCommand(
        command_type=SpatialCommandType.ROTATE_OBJECT,
        interaction_state="ACTIVE",
        cursor_ndc=(0.25, -0.40),
        delta_rotation=(-0.05, 0.02, 0.0),
        confidence=0.88,
        handedness="Right",
        timestamp=123.456,
    )

    hand = HandTelemetry(
        hand_id=0,
        handedness="Right",
        palm_center=(0.5, 0.5, 0.0),
        pinch_confidence=0.15,
        detection_confidence=0.96,
        landmarks_normalized=[[0.0, 0.0, 0.0] for _ in range(21)],
    )

    packet = HMIPacket(
        command=cmd,
        intent_state="ACTIVE",
        active_gesture="GRAB",
        intent_confidence=0.88,
        hands=[hand],
        fps=60.0,
        latency_ms=21.4,
        timestamp=123.456,
    )

    json_str = packet.model_dump_json()
    parsed = json.loads(json_str)

    assert parsed["packet_type"] == "HMI_STATE_UPDATE"
    assert parsed["command"]["command_type"] == "ROTATE_OBJECT"
    assert parsed["intent_state"] == "ACTIVE"
    assert parsed["hands"][0]["handedness"] == "Right"
    assert len(parsed["hands"][0]["landmarks_normalized"]) == 21


def test_full_hand_telemetry_serialization_with_all_levels():
    """Verify that HandTelemetry with full Level 0, 1, 2, and 3 states serializes properly."""
    hand = HandTelemetry(
        hand_id=1,
        handedness="Left",
        palm_center=(0.45, 0.55, -0.1),
        palm_velocity=(-0.25, 0.05, 0.0),
        pinch_confidence=0.92,
        detection_confidence=0.98,
        landmarks_normalized=[[0.1, 0.2, 0.0] for _ in range(21)],
        finger_states={"Thumb": "folded", "Index": "extended", "Middle": "folded", "Ring": "folded", "Little": "folded"},
        finger_details={"Index": {"state": "extended", "extension_ratio": 1.45}},
        orientation_angles=(0.1, 0.0, 0.0),
        hand_scale_ref=0.22,
        hand_pose_id="H004_POINT",
        hand_pose_name="POINT",
        pose_predicates=["INDEX_EXTENDED"],
        finger_config_summary="T:fld | I:ext | M:fld | R:fld | L:fld",
        palm_facing="PALM",
        motion_primitive="MOVE_LEFT",
        motion_speed=0.25,
        motion_velocity=[-0.25, 0.0, 0.0],
        motion_acceleration=[-0.1, 0.0, 0.0],
        motion_tangential_accel=0.05,
        motion_dynamic_state="ACCELERATING",
        motion_direction="LEFT",
        motion_secondary_direction=None,
        motion_heading_deg=180.0,
        motion_displacement=0.08,
        motion_path_length=0.09,
        motion_linearity=0.88,
        stroke_duration_ms=120.0,
        dwell_duration_ms=0.0,
        is_holding=False,
        is_releasing=False,
        motion_summary="MOVE_LEFT (0.25 u/s)",
        is_dorsal=False,
        facing_flip="STABLE",
        roll_velocity=0.0,
        finger_motions={
            "Index": {
                "primitive": "STATIONARY",
                "tip_speed": 0.02,
                "relative_speed": 0.01,
                "extension_rate": 0.0,
                "dynamic_state": "STATIONARY",
                "is_tapping": False,
                "is_extending": False,
                "is_flexing": False,
                "is_holding": True,
            }
        },
        trajectory_points=[[0.5, 0.5, 0.0, 100.0], [0.45, 0.5, 0.0, 100.05]],
        complete_gesture_id="G001P_POINT_SWIPE_LEFT",
        complete_gesture_name="POINT SWIPE LEFT",
        gesture_category="STROKE",
        gesture_phase="STROKE",
        gesture_event_sequence=["PREPARE", "STROKE"],
        gesture_metrics={"speed": 0.25, "displacement": 0.08},
        is_stroke_completed=False,
    )

    cmd = SpatialCommand(
        command_type=SpatialCommandType.ROTATE_OBJECT,
        interaction_state="ACTIVE",
        cursor_ndc=(0.0, 0.0),
        delta_rotation=(0.0, 0.0, 0.0),
        confidence=0.9,
        handedness="Left",
        timestamp=100.0,
    )

    packet = HMIPacket(
        command=cmd,
        intent_state="ACTIVE",
        active_gesture="POINT_SWIPE_LEFT",
        intent_confidence=0.95,
        hands=[hand],
        fps=60.0,
        latency_ms=15.0,
        timestamp=100.0,
    )

    dumped = packet.model_dump_json()
    parsed = json.loads(dumped)

    h0 = parsed["hands"][0]
    assert h0["trajectory_points"] == [[0.5, 0.5, 0.0, 100.0], [0.45, 0.5, 0.0, 100.05]]
    assert h0["complete_gesture_id"] == "G001P_POINT_SWIPE_LEFT"
    assert h0["palm_velocity"] == [-0.25, 0.05, 0.0]
    assert h0["gesture_phase"] == "STROKE"

