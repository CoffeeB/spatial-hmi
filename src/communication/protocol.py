"""
WebSocket message schema and telemetry protocol definitions for Gestura v3.
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

from src.interaction.mapping import SpatialCommand


class HandTelemetry(BaseModel):
    """Normalized hand data sent to the front-end for visualization and debugging."""
    hand_id: int
    handedness: str
    palm_center: Tuple[float, float, float]
    palm_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    pinch_confidence: float
    detection_confidence: float
    landmarks_normalized: List[List[float]]  # List of [x, y, z] for 21 joints
    finger_states: Dict[str, str] = {}  # Level 0 state map: {"Thumb": "folded", ...}
    finger_details: Dict[str, Dict[str, Any]] = {}  # Per-digit kinematic metrics
    orientation_angles: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (pitch, yaw, roll)
    hand_scale_ref: float = 0.20  # Scale normalizer d_ref


class HMIPacket(BaseModel):
    """
    Complete state packet broadcasted to connected WebGL/Three.js clients.
    Exposes Gestura v3 confidence model, roles, and telemetry.
    """
    packet_type: str = "HMI_STATE_UPDATE"
    command: SpatialCommand
    intent_state: str  # IDLE, OBSERVING, CANDIDATE, CONFIRMED, ACTIVE, RELEASE
    active_gesture: str
    candidate_gesture: Optional[str] = "NONE"
    dominant_hand: Optional[str] = None    # "Right" or "Left"
    modifier_hand: Optional[str] = None    # "Left" or "Right" or None
    intent_confidence: float
    hands: List[HandTelemetry] = []
    fps: float = 0.0
    latency_ms: float = 0.0
    video_frame_b64: Optional[str] = None  # Live annotated camera feed
    timestamp: float
