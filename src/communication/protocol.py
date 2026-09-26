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
    # Level 1 Static Hand Pose (Derived from Level 0 Finger Configuration)
    hand_pose_id: Optional[str] = "UNKNOWN"
    hand_pose_name: Optional[str] = "NONE"
    pose_predicates: List[str] = []
    finger_config_summary: str = ""
    palm_facing: str = "PALM"  # "PALM", "DORSAL", or "SIDE"
    pose_intention: Optional[str] = "RESTING_PALM"
    focal_digits: List[str] = []
    intended_action: Optional[str] = ""

    # Level 2 Motion Primitives ("How is the hand moving over time?")
    motion_primitive: str = "STATIONARY"
    motion_speed: float = 0.0
    motion_velocity: List[float] = [0.0, 0.0, 0.0]
    motion_acceleration: List[float] = [0.0, 0.0, 0.0]
    motion_tangential_accel: float = 0.0
    motion_dynamic_state: str = "STATIONARY"
    motion_direction: str = "STATIONARY"
    motion_secondary_direction: Optional[str] = None
    motion_heading_deg: float = 0.0
    motion_displacement: float = 0.0
    motion_path_length: float = 0.0
    motion_linearity: float = 1.0
    stroke_duration_ms: float = 0.0
    dwell_duration_ms: float = 0.0
    is_holding: bool = False
    is_releasing: bool = False
    motion_summary: str = ""
    motion_intention: Optional[str] = "STATIC_POSTURE"
    intentionality_score: float = 0.0
    is_purposeful: bool = False

    # Dorsal inclusion & Hand Flip Kinematics
    is_dorsal: bool = False
    facing_flip: str = "STABLE"
    roll_velocity: float = 0.0

    # Individual Finger Motion Primitives
    finger_motions: Dict[str, Dict[str, Any]] = {}  # { "Index": { "primitive": "EXTENDING", "speed": 0.4, ... }, ... }
    trajectory_points: List[List[float]] = []  # [[x, y, z, t], ...]

    # Level 3 Complete Gestures ("Compose Pose + Motion into observable event sequences")
    complete_gesture_id: Optional[str] = "NONE"
    complete_gesture_name: Optional[str] = "NONE"
    gesture_category: Optional[str] = "IDLE"
    gesture_phase: Optional[str] = "NEUTRAL"
    gesture_event_sequence: List[str] = []
    gesture_metrics: Dict[str, Any] = {}
    is_stroke_completed: bool = False
    task_intent: Optional[str] = "IDLE_MONITORING"
    predicted_next_intent: Optional[str] = "NONE"

    # Level 4 & 5 Temporal Intent Telemetry & Developer Debug Layer
    temporal_telemetry: Dict[str, Any] = {}

    # Stability Engine Telemetry
    stability_telemetry: Dict[str, Any] = {}

    # Continuous 3D Spatial Representation & Perception Telemetry
    spatial_telemetry: Dict[str, Any] = {}


class HMIPacket(BaseModel):
    """
    Complete state packet broadcasted to connected WebGL/Three.js clients.
    Exposes Gestura v3/v4 confidence model, roles, temporal intent, and stability telemetry.
    """
    packet_type: str = "HMI_STATE_UPDATE"
    command: SpatialCommand
    intent_state: str  # IDLE, OBSERVING, CANDIDATE, CONFIRMED, ACTIVE, RELEASING
    active_gesture: str
    candidate_gesture: Optional[str] = "NONE"
    dominant_hand: Optional[str] = None    # "Right" or "Left"
    modifier_hand: Optional[str] = None    # "Left" or "Right" or None
    intent_confidence: float
    hands: List[HandTelemetry] = []
    fps: float = 0.0
    latency_ms: float = 0.0
    video_frame_b64: Optional[str] = None  # Live annotated camera feed
    camera_zoom: float = 1.0               # Current digital zoom factor (1.0 = wide angle)
    camera_zoom_tracking: bool = False     # Whether camera is actively tracking/focusing on hands
    temporal_intent: Optional[Dict[str, Any]] = None  # Comprehensive debug telemetry for developer overlay
    stability: Optional[Dict[str, Any]] = None        # Developer Stability Panel telemetry
    timestamp: float
