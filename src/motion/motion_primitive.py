"""
Level 2 Motion Primitives Definitions & Kinematic State Data Models.
Measures and classifies pure movement properties (velocity, acceleration, direction,
distance, duration, trajectory) independently of static hand postures or gesture labels.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class MotionPrimitive(str, Enum):
    """
    Fundamental Level 2 spatial motion primitives.
    Represents pure kinetic displacement, rotational flips, and temporal behavior.
    """
    STATIONARY = "STATIONARY"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    MOVE_UP = "MOVE_UP"
    MOVE_DOWN = "MOVE_DOWN"
    MOVE_TOWARD = "MOVE_TOWARD"       # Forward / Approach (expanding scale or -Z)
    MOVE_AWAY = "MOVE_AWAY"           # Backward / Retreat (contracting scale or +Z)
    ROTATE_CW = "ROTATE_CW"           # Clockwise trajectory orbit
    ROTATE_CCW = "ROTATE_CCW"         # Counter-Clockwise trajectory orbit
    FLIP_TO_DORSAL = "FLIP_TO_DORSAL" # Pronation flip from PALM to DORSAL (back of hand)
    FLIP_TO_PALM = "FLIP_TO_PALM"     # Supination flip from DORSAL to PALM (front of hand)
    HOLD = "HOLD"                     # Stationary dwell sustained > 400ms
    RELEASE = "RELEASE"               # Abrupt deceleration / termination of moving stroke


class DynamicState(str, Enum):
    """Speed dynamics / derivative state."""
    STATIONARY = "STATIONARY"
    ACCELERATING = "ACCELERATING"
    DECELERATING = "DECELERATING"
    STEADY = "STEADY"


class FingerMotionPrimitive(str, Enum):
    """Individual digit motion primitive over time."""
    STATIONARY = "STATIONARY"         # Finger tip resting / still
    EXTENDING = "EXTENDING"           # Finger uncurling / straightening outward
    FLEXING = "FLEXING"               # Finger curling / folding into palm
    TAPPING = "TAPPING"               # Rapid downward/forward displacement and recovery
    SWIPING = "SWIPING"               # Independent lateral tip translation
    HOLD = "HOLD"                     # Sustained steady posture of the digit


@dataclass
class FingerMotionState:
    """Kinematic temporal motion state of an individual finger."""
    name: str                                     # "Thumb", "Index", "Middle", "Ring", "Little"
    tip_position: Tuple[float, float, float]      # Latest tip coordinates (x, y, z)
    tip_velocity: Tuple[float, float, float]      # Absolute tip velocity (vx, vy, vz) in units/s
    relative_velocity: Tuple[float, float, float] # Tip velocity relative to palm center (isolates finger movement)
    tip_speed: float                              # Absolute scalar speed
    relative_speed: float                         # Relative speed ||v_tip - v_palm||
    extension_rate: float                         # d(extension_ratio)/dt (+ extending, - flexing)
    dynamic_state: DynamicState                   # Dynamic state of finger tip
    motion_primitive: FingerMotionPrimitive       # Primary finger motion primitive
    is_tapping: bool = False                      # True if in active tap lifecycle
    is_extending: bool = False                    # True if actively extending
    is_flexing: bool = False                      # True if actively curling/flexing
    is_holding: bool = False                      # True if held steady
    trajectory: List[Tuple[float, float, float, float]] = field(default_factory=list)  # [(x, y, z, t)]


@dataclass
class TrajectoryPoint:
    """A discrete temporal sample along the palm trajectory."""
    timestamp: float
    x: float
    y: float
    z: float
    scale_ref: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    speed: float = 0.0

    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def velocity(self) -> Tuple[float, float, float]:
        return (self.vx, self.vy, self.vz)


@dataclass
class MotionState:
    """
    Complete measurable representation of physical movement over time.
    Separates kinematic observation from gesture semantic intent.
    """
    hand_id: int
    handedness: str

    # 1. Position & Space
    position: Tuple[float, float, float]           # Latest (x, y, z) in normalized coords

    # 2. Velocity & Speed
    velocity: Tuple[float, float, float]           # Instantaneous/smoothed velocity (vx, vy, vz) in units/sec
    speed: float                                   # Total 3D scalar speed ||v||
    speed_xy: float                                # Planar 2D speed in screen coordinates

    # 3. Acceleration Dynamics
    acceleration: Tuple[float, float, float]       # 3D acceleration vector (ax, ay, az) in units/sec^2
    acceleration_magnitude: float                  # ||a||
    tangential_acceleration: float                 # Rate of speed change d(speed)/dt (+ accelerating, - decelerating)
    dynamic_state: DynamicState                    # STATIONARY, ACCELERATING, DECELERATING, STEADY

    # 4. Direction & Heading
    direction_vector: Tuple[float, float, float]   # Normalized 3D unit direction vector (ux, uy, uz)
    heading_deg: float                             # 2D heading angle in degrees [-180, 180] in XY plane
    primary_direction: str                         # "LEFT", "RIGHT", "UP", "DOWN", "TOWARD", "AWAY", "STATIONARY"

    # 5. Distance & Geometry
    net_displacement: Tuple[float, float, float]   # (dx, dy, dz) from start of current continuous stroke
    displacement_magnitude: float                  # ||d||
    cumulative_path_length: float                  # Total path distance L along trajectory
    linearity: float                               # Tortuosity ratio ||d|| / max(L, epsilon) (1.0 = straight line)

    # 6. Depth & Approach
    scale_rate: float                              # d(d_ref)/dt rate of scale expansion/contraction
    depth_state: str                               # "TOWARD", "AWAY", "NEUTRAL"

    # 7. Rotational & Orbital Kinematics
    angular_velocity: float                        # deg/sec around trajectory centroid
    cumulative_angle_deg: float                    # Total accumulated rotation angle across buffer
    rotation_direction: str                        # "CLOCKWISE", "COUNTERCLOCKWISE", "NONE"

    # 8. Temporal Durations & Lifecycle
    stroke_duration_ms: float                      # Duration of active continuous motion stroke
    dwell_duration_ms: float                       # Duration of continuous stationarity / dwell
    is_holding: bool                               # True if held stationary for >= 400ms
    is_releasing: bool                             # True on sudden deceleration / stroke termination
    motion_primitive: MotionPrimitive              # Primary Level 2 primitive token

    # 9. Dorsal Inclusion & Hand Flip Kinematics
    palm_facing: str = "PALM"                      # "PALM", "DORSAL", or "SIDE"
    is_dorsal: bool = False                        # True if back of hand faces camera
    facing_flip: str = "STABLE"                    # "FLIP_TO_DORSAL", "FLIP_TO_PALM", "STABLE"
    roll_velocity: float = 0.0                     # Hand roll angular rate in deg/s

    # 10. Individual Finger Motion Primitives
    finger_motions: Dict[str, FingerMotionState] = field(default_factory=dict)

    confidence: float = 1.0                        # Confidence in primitive classification (0.0 to 1.0)
    timestamp: float = 0.0                         # Current sample timestamp
    secondary_direction: Optional[str] = None      # e.g. "UP" if moving diagonally UP-RIGHT
    trajectory: List[Tuple[float, float, float, float]] = field(default_factory=list)  # [(x, y, z, t), ...]

    # Level 2 Intention Deciphering & Purposefulness vs. Accidental Drift
    motion_intention: str = "STATIONARY_INSPECTION"
    intentionality_score: float = 0.0
    is_purposeful: bool = False

    def __post_init__(self):
        if self.motion_intention == "STATIONARY_INSPECTION" and self.speed > 0.0:
            from src.intent.intention_decipherer import Level2MotionIntentionDecipherer
            dec = Level2MotionIntentionDecipherer.decipher(
                speed=self.speed,
                displacement=self.displacement_magnitude,
                linearity=self.linearity,
                direction_consistency=1.0,
                motion_primitive=self.motion_primitive.value if hasattr(self.motion_primitive, "value") else str(self.motion_primitive),
                active_pose="NONE",
                duration_ms=self.stroke_duration_ms,
            )
            self.motion_intention = dec["motion_intention"]
            self.intentionality_score = dec["intentionality_score"]
            self.is_purposeful = dec["is_purposeful"]

    def summary(self) -> str:
        """Concise one-line diagnostic summary of current movement."""
        facing_tag = " [DORSAL]" if self.is_dorsal else ""
        flip_tag = f" <{self.facing_flip}>" if self.facing_flip != "STABLE" else ""
        f_acts = [
            f"{name[:1]}:{f.motion_primitive.value[:3]}"
            for name, f in self.finger_motions.items()
            if f.motion_primitive != FingerMotionPrimitive.STATIONARY
        ]
        f_summary = f" | Digits: [{', '.join(f_acts)}]" if f_acts else ""
        return (
            f"[{self.handedness}]{facing_tag}{flip_tag} {self.motion_primitive.value} ({self.dynamic_state.value}) | "
            f"spd={self.speed:.2f}u/s, dir={self.primary_direction}"
            f"{('+' + self.secondary_direction) if self.secondary_direction else ''} "
            f"({self.heading_deg:+.0f}°), lin={self.linearity:.2f}, dur={self.stroke_duration_ms:.0f}ms{f_summary}"
        )

    def as_dict(self) -> Dict[str, Any]:
        """Serializes measurable metrics for logging or debugging."""
        return {
            "hand_id": self.hand_id,
            "handedness": self.handedness,
            "motion_primitive": self.motion_primitive.value,
            "dynamic_state": self.dynamic_state.value,
            "speed": round(self.speed, 4),
            "speed_xy": round(self.speed_xy, 4),
            "velocity": [round(v, 4) for v in self.velocity],
            "acceleration": [round(a, 4) for a in self.acceleration],
            "tangential_accel": round(self.tangential_acceleration, 4),
            "primary_direction": self.primary_direction,
            "secondary_direction": self.secondary_direction,
            "heading_deg": round(self.heading_deg, 1),
            "displacement": round(self.displacement_magnitude, 4),
            "path_length": round(self.cumulative_path_length, 4),
            "linearity": round(self.linearity, 3),
            "scale_rate": round(self.scale_rate, 4),
            "depth_state": self.depth_state,
            "angular_velocity": round(self.angular_velocity, 1),
            "cumulative_angle_deg": round(self.cumulative_angle_deg, 1),
            "rotation_direction": self.rotation_direction,
            "palm_facing": self.palm_facing,
            "is_dorsal": self.is_dorsal,
            "facing_flip": self.facing_flip,
            "roll_velocity": round(self.roll_velocity, 1),
            "stroke_duration_ms": round(self.stroke_duration_ms, 1),
            "dwell_duration_ms": round(self.dwell_duration_ms, 1),
            "is_holding": self.is_holding,
            "is_releasing": self.is_releasing,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
            "motion_intention": getattr(self, "motion_intention", "STATIONARY_INSPECTION"),
            "intentionality_score": round(float(getattr(self, "intentionality_score", 0.0)), 2),
            "is_purposeful": bool(getattr(self, "is_purposeful", False)),
            "finger_motions": {
                name: {
                    "primitive": f.motion_primitive.value,
                    "tip_speed": round(f.tip_speed, 4),
                    "relative_speed": round(f.relative_speed, 4),
                    "extension_rate": round(f.extension_rate, 4),
                    "dynamic_state": f.dynamic_state.value,
                    "is_tapping": f.is_tapping,
                    "is_extending": f.is_extending,
                    "is_flexing": f.is_flexing,
                }
                for name, f in self.finger_motions.items()
            },
        }
