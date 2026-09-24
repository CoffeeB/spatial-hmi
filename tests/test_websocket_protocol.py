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
