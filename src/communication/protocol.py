"""
WebSocket message schema and telemetry protocol definitions.
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

from src.interaction.mapping import SpatialCommand


class HandTelemetry(BaseModel):
    """Normalized hand data sent to the front-end for visualization and debugging."""
    hand_id: int
    handedness: str
    palm_center: Tuple[float, float, float]
    pinch_confidence: float
    detection_confidence: float
    landmarks_normalized: List[List[float]]  # List of [x, y, z] for 21 joints


class HMIPacket(BaseModel):
    """
    Complete state packet broadcasted to connected WebGL/Three.js clients.
    """
    packet_type: str = "HMI_STATE_UPDATE"
    command: SpatialCommand
    intent_state: str  # IDLE, OBSERVING, CANDIDATE, CONFIRMED, ACTIVE, RELEASING
    active_gesture: str
    intent_confidence: float
    hands: List[HandTelemetry] = []
    fps: float = 0.0
    latency_ms: float = 0.0
    video_frame_b64: Optional[str] = None  # Live annotated camera feed
    timestamp: float
