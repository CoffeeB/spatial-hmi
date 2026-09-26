"""
Level 3: Complete Gesture Recognition Engine for Gestura.

Implements the formal composition of:
    Hand Pose (Level 1) + Motion Primitive (Level 2)
              ↓
    Complete Gesture (Level 3: G001–G999)

Crucially, gestures in Gestura are NOT static snapshot classifications.
They are temporal sequences of observable physical events:
    - POINT while SWIPE LEFT:
          Point pose (Index extended, others folded)
          + rapid leftward motion
          + sufficient displacement
          + directional consistency
    - PINCH:
          Thumb + Index approach (distance closing)
          ↓
          Contact (pads touching)
          ↓
          Hold (steady contact) / Drag (contact + translation)
          ↓
          Release (distance opening)
    - AIR TAP:
          Aim (Point pose)
          ↓
          Strike (downward / forward excursion)
          ↓
          Bottom-out & Rebound
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
import numpy as np

from src.gestures.hand_pose import HandPoseId, POSE_CANONICAL_NAMES
from src.motion.motion_primitive import DynamicState, FingerMotionPrimitive, MotionPrimitive


class CompleteGestureId(str, Enum):
    """Canonical Level 3 Complete Gesture Identifiers (G001–G999)."""
    # ── Point Gestures (1-Finger to 5-Finger Point) ───────────────────────────
    G005_1F_POINT = "G005_1F_POINT"
    G005_2F_POINT = "G005_2F_POINT"
    G005_3F_POINT = "G005_3F_POINT"
    G005_4F_POINT = "G005_4F_POINT"
    G005_5F_POINT = "G005_5F_POINT"
    G005_POINT_HOVER = "G005_1F_POINT"

    # ── 1-Finger Swipes ───────────────────────────────────────────────────────
    G001_1F_SWIPE_LEFT = "G001_1F_SWIPE_LEFT"
    G002_1F_SWIPE_RIGHT = "G002_1F_SWIPE_RIGHT"
    G003_1F_SWIPE_UP = "G003_1F_SWIPE_UP"
    G004_1F_SWIPE_DOWN = "G004_1F_SWIPE_DOWN"
    G001P_POINT_SWIPE_LEFT = "G001_1F_SWIPE_LEFT"
    G002P_POINT_SWIPE_RIGHT = "G002_1F_SWIPE_RIGHT"
    G003P_POINT_SWIPE_UP = "G003_1F_SWIPE_UP"
    G004P_POINT_SWIPE_DOWN = "G004_1F_SWIPE_DOWN"

    # ── 2-Finger Swipes ───────────────────────────────────────────────────────
    G001_2F_SWIPE_LEFT = "G001_2F_SWIPE_LEFT"
    G002_2F_SWIPE_RIGHT = "G002_2F_SWIPE_RIGHT"
    G003_2F_SWIPE_UP = "G003_2F_SWIPE_UP"
    G004_2F_SWIPE_DOWN = "G004_2F_SWIPE_DOWN"

    # ── 3-Finger Swipes ───────────────────────────────────────────────────────
    G001_3F_SWIPE_LEFT = "G001_3F_SWIPE_LEFT"
    G002_3F_SWIPE_RIGHT = "G002_3F_SWIPE_RIGHT"
    G003_3F_SWIPE_UP = "G003_3F_SWIPE_UP"
    G004_3F_SWIPE_DOWN = "G004_3F_SWIPE_DOWN"

    # ── 4-Finger Swipes ───────────────────────────────────────────────────────
    G001_4F_SWIPE_LEFT = "G001_4F_SWIPE_LEFT"
    G002_4F_SWIPE_RIGHT = "G002_4F_SWIPE_RIGHT"
    G003_4F_SWIPE_UP = "G003_4F_SWIPE_UP"
    G004_4F_SWIPE_DOWN = "G004_4F_SWIPE_DOWN"

    # ── 5-Finger Swipes / Hand Swipes ─────────────────────────────────────────
    G001_5F_SWIPE_LEFT = "G001_5F_SWIPE_LEFT"
    G002_5F_SWIPE_RIGHT = "G002_5F_SWIPE_RIGHT"
    G003_5F_SWIPE_UP = "G003_5F_SWIPE_UP"
    G004_5F_SWIPE_DOWN = "G004_5F_SWIPE_DOWN"
    G001_PALM_SWIPE_LEFT = "G001_5F_SWIPE_LEFT"
    G002_PALM_SWIPE_RIGHT = "G002_5F_SWIPE_RIGHT"
    G003_PALM_SWIPE_UP = "G003_5F_SWIPE_UP"
    G004_PALM_SWIPE_DOWN = "G004_5F_SWIPE_DOWN"

    # Precision Pinch (Sequential Event Progression)
    G006_PINCH_SELECT = "G006_PINCH_SELECT"
    G007_PINCH_DRAG = "G007_PINCH_DRAG"

    # Closed Fist Grab
    G008_FIST_GRAB = "G008_FIST_GRAB"

    # Open Palm Anchor & Hover (G010)
    G009_WORLD_EXPAND = "G009_WORLD_EXPAND"
    G010_OPEN_PALM_RELEASE = "G010_OPEN_PALM_RELEASE"
    G010_OPEN_PALM_HOVER = "G010_OPEN_PALM_HOVER"

    # Air Tap Click
    G011_AIR_TAP = "G011_AIR_TAP"

    # Push Dwell Confirm
    G012_PUSH_CONFIRM = "G012_PUSH_CONFIRM"

    # Approval / Dismissal
    G013_THUMBS_UP = "G013_THUMBS_UP"
    G014_THUMBS_DOWN = "G014_THUMBS_DOWN"

    # Symbolic Shortcuts
    G015_PEACE_MACRO = "G015_PEACE_MACRO"
    G016_OK_LOCK = "G016_OK_LOCK"

    # Quick Flick (High-acceleration dismiss)
    G017_QUICK_FLICK = "G017_QUICK_FLICK"

    # ── Rotation (1 to 5 Fingers Rotate CW / CCW) ────────────────────────────
    G018_1F_ROTATE_CW = "G018_1F_ROTATE_CW"
    G019_1F_ROTATE_CCW = "G019_1F_ROTATE_CCW"

    G018_2F_ROTATE_CW = "G018_2F_ROTATE_CW"
    G019_2F_ROTATE_CCW = "G019_2F_ROTATE_CCW"

    G018_3F_ROTATE_CW = "G018_3F_ROTATE_CW"
    G019_3F_ROTATE_CCW = "G019_3F_ROTATE_CCW"

    G018_4F_ROTATE_CW = "G018_4F_ROTATE_CW"
    G019_4F_ROTATE_CCW = "G019_4F_ROTATE_CCW"

    G018_5F_ROTATE_CW = "G018_5F_ROTATE_CW"
    G019_5F_ROTATE_CCW = "G019_5F_ROTATE_CCW"
    G018_CLOCKWISE_DIAL = "G018_5F_ROTATE_CW"
    G019_COUNTER_CLOCKWISE_DIAL = "G019_5F_ROTATE_CCW"

    # Axial Flips
    G020_AXIAL_FLIP_DORSAL = "G020_AXIAL_FLIP_DORSAL"
    G021_AXIAL_FLIP_PALM = "G021_AXIAL_FLIP_PALM"

    # Single-Hand Pinch-to-Zoom
    G022_PINCH_ZOOM_IN = "G022_PINCH_ZOOM_IN"
    G023_PINCH_ZOOM_OUT = "G023_PINCH_ZOOM_OUT"

    # Single-Hand Finger Micro-Gestures (FM001–FM099)
    FM001_MICRO_PINCH_TAP = "FM001_MICRO_PINCH_TAP"
    FM002_MICRO_MIDDLE_TAP = "FM002_MICRO_MIDDLE_TAP"
    FM003_MICRO_DOUBLE_TAP = "FM003_MICRO_DOUBLE_TAP"
    FM004_MICRO_INDEX_TRIGGER = "FM004_MICRO_INDEX_TRIGGER"

    NONE = "NONE"


class GestureCategory(str, Enum):
    """Functional interaction category of the complete gesture."""
    SWIPE_STROKE = "SWIPE_STROKE"
    PINCH_INTERACTION = "PINCH_INTERACTION"
    POINT_INTERACTION = "POINT_INTERACTION"
    HOLD_POSE = "HOLD_POSE"
    ORBITAL_DIAL = "ORBITAL_DIAL"
    AXIAL_FLIP = "AXIAL_FLIP"
    ZOOM_INTERACTION = "ZOOM_INTERACTION"
    MICRO_GESTURE = "MICRO_GESTURE"
    IDLE = "IDLE"


class PinchPhase(str, Enum):
    """Observable temporal progression of a pinch interaction."""
    IDLE = "IDLE"           # Fingers separated, distance > 0.38
    APPROACH = "APPROACH"   # Tips actively closing in: d_pinch decreasing
    CONTACT = "CONTACT"     # Tips touching (contact threshold reached)
    HOLD = "HOLD"           # Contact sustained at rest (dwell > 80ms)
    DRAG = "DRAG"           # Contact sustained while translating through space
    RELEASE = "RELEASE"     # Tips separating


class SwipePhase(str, Enum):
    """Observable temporal progression of a directional stroke."""
    IDLE = "IDLE"
    PREPARE = "PREPARE"     # Holding stable target posture prior to movement
    STROKE = "STROKE"       # Fast directional acceleration with high linearity
    COMPLETED = "COMPLETED" # Displacement threshold met and stroke terminated


class TapPhase(str, Enum):
    """Observable temporal progression of an in-air tap."""
    IDLE = "IDLE"
    AIM = "AIM"             # Hand steady in Point pose
    STRIKE = "STRIKE"       # Rapid downward/forward excursion of index tip
    BOTTOM_OUT = "BOTTOM_OUT" # Excursion halts at peak extension
    REBOUND = "REBOUND"     # Index tip rebounds back to baseline
    COMPLETED = "COMPLETED" # Full strike-and-recover event verified


class PinchZoomPhase(str, Enum):
    """Observable temporal progression of single-hand pinch zooming."""
    IDLE = "IDLE"
    PINCHED = "PINCHED"
    ZOOMING_IN = "ZOOMING_IN"
    ZOOMING_OUT = "ZOOMING_OUT"
    STEADY = "STEADY"


class DialPhase(str, Enum):
    """Observable temporal progression of circular dialing."""
    IDLE = "IDLE"
    ORBIT_ENTER = "ORBIT_ENTER"
    ROTATING_CW = "ROTATING_CW"
    ROTATING_CCW = "ROTATING_CCW"
    DETENT = "DETENT"


class MicroGesturePhase(str, Enum):
    """Observable temporal state of sub-centimeter micro-gestures."""
    IDLE = "IDLE"
    MICRO_TOUCH = "MICRO_TOUCH"
    MICRO_CLICK = "MICRO_CLICK"
    DOUBLE_TAP = "DOUBLE_TAP"
    TRIGGER_PULL = "TRIGGER_PULL"


@dataclass
class CompleteGestureState:
    """
    Rich Level 3 complete gesture output representing the composition
    of Level 1 Hand Pose and Level 2 Motion Primitive, along with its
    active temporal phase and sequence of observable physical events.
    """
    gesture_id: CompleteGestureId
    canonical_name: str
    category: GestureCategory
    hand_pose_id: str
    hand_pose_name: str
    motion_primitive: str
    confidence: float
    phase: str
    event_sequence: List[str]
    metrics: Dict[str, float]
    is_active: bool
    is_stroke_completed: bool
    timestamp: float
    # Level 3 Complete Gesture Intention & Sequential Intent Prediction
    task_intent: str = "PASSIVE_OBSERVE"
    predicted_next_intent: str = "ENGAGE_INTERACTION"
    focal_digits: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.task_intent == "PASSIVE_OBSERVE" and self.gesture_id != CompleteGestureId.NONE:
            from src.intent.intention_decipherer import Level3GestureIntentionDecipherer
            dec = Level3GestureIntentionDecipherer.decipher(
                gesture_name=self.canonical_name,
                pose_intention=self.hand_pose_name,
                motion_intention=self.motion_primitive,
                intent_state="ACTIVE" if self.is_active else "OBSERVING",
                focal_digits=self.focal_digits,
            )
            self.task_intent = dec["task_intent"]
            self.predicted_next_intent = dec["predicted_next_intent"]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "gesture_id": self.gesture_id.value,
            "canonical_name": self.canonical_name,
            "category": self.category.value,
            "hand_pose_id": self.hand_pose_id,
            "hand_pose_name": self.hand_pose_name,
            "motion_primitive": self.motion_primitive,
            "confidence": round(self.confidence, 3),
            "phase": self.phase,
            "event_sequence": list(self.event_sequence),
            "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in self.metrics.items()},
            "is_active": self.is_active,
            "is_stroke_completed": self.is_stroke_completed,
            "timestamp": round(self.timestamp, 4),
            "task_intent": self.task_intent,
            "predicted_next_intent": self.predicted_next_intent,
            "focal_digits": list(self.focal_digits),
        }

    def summary(self) -> str:
        seq_str = " → ".join(self.event_sequence[-3:]) if self.event_sequence else self.phase
        return f"{self.canonical_name} [{self.gesture_id.value}] ({seq_str}, Conf: {self.confidence:.2f})"


class PinchSequenceTracker:
    """
    Tracks the observable sequence of physical events for a pinch:
        1. APPROACH: Thumb and index tips moving towards each other (closing velocity).
        2. CONTACT: Tips meet (distance <= 0.35 * d_ref).
        3. HOLD: Contact maintained steadily for >= 80ms (Selection).
        4. DRAG: Contact maintained while hand translates (Moving node).
        5. RELEASE: Tips separate (distance opens > 0.40 * d_ref).
    """

    def __init__(self):
        self.phase: PinchPhase = PinchPhase.IDLE
        self.distance_history: Deque[Tuple[float, float]] = deque(maxlen=15)  # (dist, timestamp)
        self.contact_start_time: Optional[float] = None
        self.last_contact_time: Optional[float] = None
        self.approach_start_time: Optional[float] = None
        self.event_log: Deque[str] = deque(maxlen=6)
        self.peak_approach_rate: float = 0.0

    def reset(self):
        self.phase = PinchPhase.IDLE
        self.distance_history.clear()
        self.contact_start_time = None
        self.last_contact_time = None
        self.approach_start_time = None
        self.event_log.clear()
        self.peak_approach_rate = 0.0

    def update(
        self,
        thumb_tip: Tuple[float, float, float],
        index_tip: Tuple[float, float, float],
        d_ref: float,
        palm_speed: float,
        net_displacement: float,
        timestamp: float,
    ) -> Tuple[PinchPhase, List[str], Dict[str, float]]:
        # Normalized Euclidean distance between thumb and index tips
        dx = thumb_tip[0] - index_tip[0]
        dy = thumb_tip[1] - index_tip[1]
        dz = thumb_tip[2] - index_tip[2]
        dist_raw = math.sqrt(dx * dx + dy * dy + dz * dz)
        d_ref_safe = max(d_ref, 0.05)
        norm_dist = dist_raw / d_ref_safe

        self.distance_history.append((norm_dist, timestamp))

        # Calculate rate of approach (d(dist)/dt)
        approach_rate = 0.0
        if len(self.distance_history) >= 2:
            prev_d, prev_t = self.distance_history[0]
            dt = max(timestamp - prev_t, 0.001)
            approach_rate = (norm_dist - prev_d) / dt  # Negative means approaching

        contact_thresh = 0.45
        release_thresh = 0.52
        metrics = {
            "pinch_distance": norm_dist,
            "approach_rate": approach_rate,
            "hold_duration_ms": 0.0,
        }

        # ── State Machine ──────────────────────────────────────────────
        if norm_dist <= contact_thresh:
            # Contact is established
            if self.contact_start_time is None:
                self.contact_start_time = timestamp
                self.event_log.append(f"CONTACT (d={norm_dist:.2f})")

            self.last_contact_time = timestamp
            hold_duration_ms = (timestamp - self.contact_start_time) * 1000.0
            metrics["hold_duration_ms"] = hold_duration_ms

            if hold_duration_ms >= 80.0:
                if palm_speed > 0.15 or net_displacement > 0.025:
                    if self.phase != PinchPhase.DRAG:
                        self.event_log.append(f"DRAG (v={palm_speed:.2f})")
                    self.phase = PinchPhase.DRAG
                else:
                    if self.phase != PinchPhase.HOLD:
                        self.event_log.append(f"HOLD ({int(hold_duration_ms)}ms)")
                    self.phase = PinchPhase.HOLD
            else:
                self.phase = PinchPhase.CONTACT

        elif norm_dist >= release_thresh:
            # Fingers separated
            if self.contact_start_time is not None:
                self.event_log.append("RELEASE")
                self.phase = PinchPhase.RELEASE
                self.contact_start_time = None
                self.approach_start_time = None
            elif approach_rate < -0.15:
                # Actively approaching
                if self.phase != PinchPhase.APPROACH:
                    self.event_log.append(f"APPROACH (rate={approach_rate:.2f})")
                self.phase = PinchPhase.APPROACH
                self.peak_approach_rate = min(self.peak_approach_rate, approach_rate)
            else:
                self.phase = PinchPhase.IDLE

        else:
            # Intermediate hysteresis band (0.35 < dist < 0.42)
            if self.phase in (PinchPhase.CONTACT, PinchPhase.HOLD, PinchPhase.DRAG):
                # Maintain contact/hold state unless distinctly opened
                hold_duration_ms = (timestamp - (self.contact_start_time or timestamp)) * 1000.0
                metrics["hold_duration_ms"] = hold_duration_ms
            elif approach_rate < -0.15:
                if self.phase != PinchPhase.APPROACH:
                    self.event_log.append(f"APPROACH (rate={approach_rate:.2f})")
                self.phase = PinchPhase.APPROACH
            else:
                self.phase = PinchPhase.IDLE

        return self.phase, list(self.event_log), metrics


class SwipeSequenceTracker:
    """
    Tracks the observable sequence of physical events for a directional swipe:
        1. PREPARE: Hand establishes steady target pose (Point or Open Palm) for >= 50ms.
        2. STROKE: Rapid directional acceleration (speed >= 0.20 u/s) with high linearity.
        3. COMPLETED: Displacement (>= 0.045 units) and linearity (>= 0.68) verified.
    """

    def __init__(self):
        self.phase: SwipePhase = SwipePhase.IDLE
        self.prepare_start_time: Optional[float] = None
        self.stroke_start_time: Optional[float] = None
        self.last_pose: str = "NONE"
        self.event_log: Deque[str] = deque(maxlen=6)
        self.completed_direction: Optional[str] = None
        self.completed_pose: Optional[str] = None

    def reset(self):
        self.phase = SwipePhase.IDLE
        self.prepare_start_time = None
        self.stroke_start_time = None
        self.last_pose = "NONE"
        self.event_log.clear()
        self.completed_direction = None
        self.completed_pose = None

    def update(
        self,
        pose_name: str,
        motion_prim: MotionPrimitive,
        speed: float,
        displacement: float,
        linearity: float,
        stroke_duration_ms: float,
        direction: str,
        timestamp: float,
    ) -> Tuple[SwipePhase, List[str], Dict[str, float]]:
        is_swipe_motion = motion_prim in (
            MotionPrimitive.MOVE_LEFT,
            MotionPrimitive.MOVE_RIGHT,
            MotionPrimitive.MOVE_UP,
            MotionPrimitive.MOVE_DOWN,
        )

        metrics = {
            "speed": speed,
            "displacement": displacement,
            "linearity": linearity,
            "duration_ms": stroke_duration_ms,
        }

        # 1. Verification of Target Pose
        is_point = pose_name in ("POINT", "DOUBLE_POINT", "GUN", "THREE_FINGER", "FOUR_FINGER", "FIVE_FINGER", "PEACE")
        is_palm = pose_name in ("OPEN_PALM", "SPREAD_FINGERS", "KNIFE_EDGE")
        valid_pose = is_point or is_palm

        if not valid_pose:
            self.prepare_start_time = None
            self.phase = SwipePhase.IDLE
            return self.phase, list(self.event_log), metrics

        # 2. Preparation Phase: Hand is in pose and steady
        if speed < 0.18:
            if self.prepare_start_time is None:
                self.prepare_start_time = timestamp
                self.event_log.append(f"PREPARE ({pose_name})")
            self.phase = SwipePhase.PREPARE
            self.last_pose = pose_name
            return self.phase, list(self.event_log), metrics

        # 3. Stroke Phase: Moving fast in a cardinal direction
        if is_swipe_motion and speed >= 0.20:
            if self.phase != SwipePhase.STROKE:
                self.stroke_start_time = timestamp
                self.event_log.append(f"STROKE ({direction})")
            self.phase = SwipePhase.STROKE

            # Check if stroke meets completion criteria
            if (
                displacement >= 0.040
                and linearity >= 0.65
                and stroke_duration_ms >= 70.0
            ):
                self.phase = SwipePhase.COMPLETED
                self.completed_direction = direction
                self.completed_pose = pose_name
                self.event_log.append(f"COMPLETED ({pose_name}·{direction})")
        else:
            if self.phase == SwipePhase.STROKE:
                self.phase = SwipePhase.IDLE

        return self.phase, list(self.event_log), metrics


class AirTapSequenceTracker:
    """
    Tracks the observable sequence of physical events for an in-air tap:
        1. AIM: Pointing pose with steady palm.
        2. STRIKE: Rapid downward/forward excursion of the index fingertip relative to palm.
        3. BOTTOM_OUT: Strike excursion reaches peak and halts (within 60–180ms).
        4. REBOUND: Index tip returns back.
        5. COMPLETED: Discrete click emitted.
    """

    def __init__(self):
        self.phase: TapPhase = TapPhase.IDLE
        self.aim_start_time: Optional[float] = None
        self.strike_start_time: Optional[float] = None
        self.peak_strike_val: float = 0.0
        self.event_log: Deque[str] = deque(maxlen=6)

    def reset(self):
        self.phase = TapPhase.IDLE
        self.aim_start_time = None
        self.strike_start_time = None
        self.peak_strike_val = 0.0
        self.event_log.clear()

    def update(
        self,
        pose_name: str,
        index_finger_motion: Optional[Any],
        palm_speed: float,
        timestamp: float,
    ) -> Tuple[TapPhase, List[str], Dict[str, float]]:
        metrics = {
            "index_rel_speed": 0.0,
            "extension_rate": 0.0,
        }

        if pose_name not in ("POINT", "DOUBLE_POINT"):
            self.phase = TapPhase.IDLE
            self.aim_start_time = None
            return self.phase, list(self.event_log), metrics

        # Palm must remain relatively stationary for an isolated tap
        if palm_speed > 0.12:
            self.phase = TapPhase.IDLE
            return self.phase, list(self.event_log), metrics

        if self.aim_start_time is None:
            self.aim_start_time = timestamp
            self.event_log.append("AIM")
            self.phase = TapPhase.AIM

        if index_finger_motion is not None:
            rel_speed = float(getattr(index_finger_motion, "relative_speed", 0.0))
            ext_rate = float(getattr(index_finger_motion, "extension_rate", 0.0))
            is_tapping = bool(getattr(index_finger_motion, "is_tapping", False))
            prim = getattr(index_finger_motion, "motion_primitive", FingerMotionPrimitive.STATIONARY)

            metrics["index_rel_speed"] = rel_speed
            metrics["extension_rate"] = ext_rate

            if is_tapping or prim == FingerMotionPrimitive.TAPPING:
                self.phase = TapPhase.COMPLETED
                self.event_log.append("STRIKE ➜ REBOUND [TAP]")
            elif ext_rate < -0.22 or rel_speed > 0.20:
                if self.phase != TapPhase.STRIKE:
                    self.event_log.append(f"STRIKE (v={rel_speed:.2f})")
                self.phase = TapPhase.STRIKE
            elif self.phase == TapPhase.STRIKE and ext_rate > 0.15:
                self.phase = TapPhase.REBOUND
                self.event_log.append("REBOUND")
            elif self.phase == TapPhase.REBOUND:
                self.phase = TapPhase.COMPLETED
                self.event_log.append("COMPLETED [AIR_TAP]")

        return self.phase, list(self.event_log), metrics


class PinchZoomSequenceTracker:
    """
    Tracks single-hand pinch-to-zoom interactions across two complementary biomechanical channels:
        Channel A (Depth Thrust): Pinching hand pushes toward camera (Z- / scale expanding) -> ZOOM IN;
                                  Pinching hand pulls away (Z+ / scale contracting) -> ZOOM OUT.
        Channel B (Aperture Scaling): Distance between thumb and index dynamically increases/decreases.
    """

    def __init__(self):
        self.phase: PinchZoomPhase = PinchZoomPhase.IDLE
        self.pinch_start_time: Optional[float] = None
        self.last_pinch_dist: Optional[float] = None
        self.last_pinch_time: Optional[float] = None
        self.event_log: Deque[str] = deque(maxlen=6)
        self.cumulative_zoom_delta: float = 0.0

    def reset(self):
        self.phase = PinchZoomPhase.IDLE
        self.pinch_start_time = None
        self.last_pinch_dist = None
        self.last_pinch_time = None
        self.event_log.clear()
        self.cumulative_zoom_delta = 0.0

    def update(
        self,
        thumb_tip: Tuple[float, float, float],
        index_tip: Tuple[float, float, float],
        d_ref: float,
        palm_velocity_z: float,
        scale_rate: float,
        pose_name: str,
        timestamp: float,
    ) -> Tuple[PinchZoomPhase, List[str], Dict[str, float]]:
        dx = thumb_tip[0] - index_tip[0]
        dy = thumb_tip[1] - index_tip[1]
        dz = thumb_tip[2] - index_tip[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        norm_dist = dist / max(0.001, d_ref)

        metrics = {
            "norm_pinch_dist": norm_dist,
            "zoom_velocity": 0.0,
            "cumulative_zoom": self.cumulative_zoom_delta,
        }

        # Pinch posture check
        is_pinching = norm_dist <= 0.44 or pose_name in ("PINCH", "KEY_PINCH", "GRAB")
        if not is_pinching:
            self.reset()
            return self.phase, list(self.event_log), metrics

        if self.pinch_start_time is None:
            self.pinch_start_time = timestamp
            self.event_log.append("PINCH_ENGAGED")
            self.phase = PinchZoomPhase.PINCHED

        depth_vel = -palm_velocity_z  # Moving towards camera (Z-) is positive depth thrust (zoom in)
        aperture_rate = 0.0
        held_duration = timestamp - self.pinch_start_time if self.pinch_start_time is not None else 0.0
        if self.last_pinch_dist is not None and self.last_pinch_time is not None and held_duration >= 0.120:
            dt = max(0.001, timestamp - self.last_pinch_time)
            aperture_rate = (norm_dist - self.last_pinch_dist) / dt

        self.last_pinch_dist = norm_dist
        self.last_pinch_time = timestamp

        # Detect Zoom In vs Zoom Out
        zoom_in_stimulus = depth_vel > 0.14 or scale_rate > 0.18 or (aperture_rate > 0.32 and held_duration >= 0.120)
        zoom_out_stimulus = depth_vel < -0.14 or scale_rate < -0.18 or (aperture_rate < -0.32 and held_duration >= 0.120)

        zoom_speed = abs(depth_vel) if abs(depth_vel) > abs(aperture_rate) else abs(aperture_rate)
        if abs(scale_rate) > zoom_speed:
            zoom_speed = abs(scale_rate)
        metrics["zoom_velocity"] = depth_vel if abs(depth_vel) > abs(aperture_rate) else aperture_rate

        if zoom_in_stimulus:
            self.cumulative_zoom_delta += zoom_speed * 0.05
            if self.phase != PinchZoomPhase.ZOOMING_IN:
                self.phase = PinchZoomPhase.ZOOMING_IN
                self.event_log.append(f"ZOOM_IN (v={zoom_speed:.2f})")
        elif zoom_out_stimulus:
            self.cumulative_zoom_delta -= zoom_speed * 0.05
            if self.phase != PinchZoomPhase.ZOOMING_OUT:
                self.phase = PinchZoomPhase.ZOOMING_OUT
                self.event_log.append(f"ZOOM_OUT (v={zoom_speed:.2f})")
        else:
            self.phase = PinchZoomPhase.STEADY

        metrics["cumulative_zoom"] = self.cumulative_zoom_delta
        return self.phase, list(self.event_log), metrics


class DialSequenceTracker:
    """
    Tracks circular dialing interactions (G018 / G019):
        1. ORBIT_ENTER: Circular trajectory detected (radius bounded).
        2. ROTATING: Continuous angular progress Delta theta.
        3. DETENT: Angle crosses 45°/90° increments.
    """

    def __init__(self):
        self.phase: DialPhase = DialPhase.IDLE
        self.cumulative_angle_deg: float = 0.0
        self.last_angle: Optional[float] = None
        self.event_log: Deque[str] = deque(maxlen=6)
        self.detent_count: int = 0

    def reset(self):
        self.phase = DialPhase.IDLE
        self.cumulative_angle_deg = 0.0
        self.last_angle = None
        self.event_log.clear()
        self.detent_count = 0

    def update(
        self,
        motion_prim: MotionPrimitive,
        angular_velocity: float,
        heading_deg: float,
        speed: float,
        timestamp: float,
    ) -> Tuple[DialPhase, List[str], Dict[str, float]]:
        metrics = {
            "cumulative_angle_deg": self.cumulative_angle_deg,
            "angular_velocity": angular_velocity,
            "detent_count": float(self.detent_count),
        }

        is_cw = motion_prim == MotionPrimitive.ROTATE_CW
        is_ccw = motion_prim == MotionPrimitive.ROTATE_CCW

        if not (is_cw or is_ccw):
            if abs(self.cumulative_angle_deg) > 15.0:
                self.cumulative_angle_deg *= 0.88
            else:
                self.reset()
            return self.phase, list(self.event_log), metrics

        if self.last_angle is None:
            self.last_angle = heading_deg
            self.event_log.append("ORBIT_ENTER")
            self.phase = DialPhase.ORBIT_ENTER
            return self.phase, list(self.event_log), metrics

        delta = heading_deg - self.last_angle
        if delta > 180.0:
            delta -= 360.0
        elif delta < -180.0:
            delta += 360.0

        self.last_angle = heading_deg
        if abs(delta) < 1.0 or abs(delta) > 90.0:
            delta = 0.0

        if is_cw and delta < 0:
            delta = abs(delta) if abs(delta) > 2.0 else 5.0
        elif is_ccw and delta > 0:
            delta = -abs(delta) if abs(delta) > 2.0 else -5.0

        if delta == 0.0 and abs(angular_velocity) > 0.5:
            delta = angular_velocity * 1.5

        self.cumulative_angle_deg += delta

        new_phase = DialPhase.ROTATING_CW if self.cumulative_angle_deg >= 0 else DialPhase.ROTATING_CCW
        if new_phase != self.phase:
            self.phase = new_phase
            tag = "CW" if is_cw else "CCW"
            self.event_log.append(f"ROTATING_{tag}")

        curr_detents = int(abs(self.cumulative_angle_deg) // 45.0)
        if curr_detents > self.detent_count:
            self.detent_count = curr_detents
            self.event_log.append(f"DETENT #{self.detent_count} ({int(self.cumulative_angle_deg)}°)")

        metrics["cumulative_angle_deg"] = round(self.cumulative_angle_deg, 1)
        metrics["detent_count"] = float(self.detent_count)
        return self.phase, list(self.event_log), metrics


class MicroGestureTracker:
    """
    Tracks subtle sub-centimeter finger micro-gestures:
        - FM001: Thumb-to-Index Tap (Air Click) with contact duration <= 160ms
        - FM002: Thumb-to-Middle Tap (Context Click) while index extended
        - FM003: Double Pinch Tap within 380ms
        - FM004: Index Knuckle Flexion (Trigger)
    """

    def __init__(self):
        self.last_tap_time: Optional[float] = None
        self.tap_count: int = 0
        self.contact_start_time: Optional[float] = None
        self.event_log: Deque[str] = deque(maxlen=6)
        self.active_micro: CompleteGestureId = CompleteGestureId.NONE

    def reset(self):
        self.last_tap_time = None
        self.tap_count = 0
        self.contact_start_time = None
        self.event_log.clear()
        self.active_micro = CompleteGestureId.NONE

    def update(
        self,
        raw_landmarks: np.ndarray,
        d_ref: float,
        finger_motions: Dict[str, Any],
        pose_name: str,
        palm_speed: float,
        timestamp: float,
    ) -> Tuple[CompleteGestureId, List[str], Dict[str, float]]:
        metrics = {
            "thumb_index_dist": 1.0,
            "thumb_middle_dist": 1.0,
            "tap_duration_ms": 0.0,
        }

        if palm_speed > 0.15:
            self.reset()
            return CompleteGestureId.NONE, list(self.event_log), metrics

        p4 = raw_landmarks[4, :3]
        p8 = raw_landmarks[8, :3]
        p12 = raw_landmarks[12, :3]

        d_ti = float(np.linalg.norm(p4 - p8)) / max(0.001, d_ref)
        d_tm = float(np.linalg.norm(p4 - p12)) / max(0.001, d_ref)

        metrics["thumb_index_dist"] = d_ti
        metrics["thumb_middle_dist"] = d_tm

        # FM002: Thumb-to-Middle Tap (Context Click)
        if d_tm <= 0.32 and d_ti >= 0.40 and pose_name not in ("OPEN_PALM", "SPREAD_FINGERS"):
            self.event_log.append("THUMB_MIDDLE_CONTACT")
            self.active_micro = CompleteGestureId.FM002_MICRO_MIDDLE_TAP
            return self.active_micro, list(self.event_log), metrics

        # FM001 / FM003: Thumb-to-Index Tap & Double Tap
        is_contact = d_ti <= 0.32
        if is_contact:
            if self.contact_start_time is None:
                self.contact_start_time = timestamp
                self.event_log.append("MICRO_TOUCH")
            metrics["tap_duration_ms"] = (timestamp - self.contact_start_time) * 1000.0
        else:
            if self.contact_start_time is not None:
                duration_ms = (timestamp - self.contact_start_time) * 1000.0
                self.contact_start_time = None
                metrics["tap_duration_ms"] = duration_ms

                if 20.0 <= duration_ms <= 160.0:
                    if self.last_tap_time is not None and (timestamp - self.last_tap_time) <= 0.38:
                        self.event_log.append("DOUBLE_TAP [FM003]")
                        self.active_micro = CompleteGestureId.FM003_MICRO_DOUBLE_TAP
                        self.last_tap_time = None
                        return self.active_micro, list(self.event_log), metrics
                    else:
                        self.last_tap_time = timestamp
                        self.event_log.append("MICRO_CLICK [FM001]")
                        self.active_micro = CompleteGestureId.FM001_MICRO_PINCH_TAP
                        return self.active_micro, list(self.event_log), metrics

        # FM004: Index Knuckle Flexion (Trigger)
        idx_fmot = finger_motions.get("Index")
        if idx_fmot is not None:
            ext_rate = float(getattr(idx_fmot, "extension_rate", 0.0))
            is_flexing = bool(getattr(idx_fmot, "is_flexing", False))
            if (ext_rate < -0.28 or is_flexing) and pose_name in ("POINT", "GUN") and d_ti > 0.42:
                self.event_log.append("TRIGGER_FLEX [FM004]")
                self.active_micro = CompleteGestureId.FM004_MICRO_INDEX_TRIGGER
                return self.active_micro, list(self.event_log), metrics

        return CompleteGestureId.NONE, list(self.event_log), metrics


class CompleteGestureRecognizer:
    """
    Level 3 Gesture Recognition Engine for Gestura.
    Composes Level 1 Static Hand Poses and Level 2 Motion Primitives into complete,
    meaningful single-hand gestures, tracked through verifiable physical event chains.
    """

    def __init__(self):
        self._pinch_trackers: Dict[int, PinchSequenceTracker] = {}
        self._swipe_trackers: Dict[int, SwipeSequenceTracker] = {}
        self._tap_trackers: Dict[int, AirTapSequenceTracker] = {}
        self._zoom_trackers: Dict[int, PinchZoomSequenceTracker] = {}
        self._dial_trackers: Dict[int, DialSequenceTracker] = {}
        self._micro_trackers: Dict[int, MicroGestureTracker] = {}
        self._hold_timers: Dict[int, Tuple[str, float]] = {}  # hand_id -> (pose_name, start_time)
        self._prev_active: Dict[int, CompleteGestureId] = {}

    def reset(self):
        self._pinch_trackers.clear()
        self._swipe_trackers.clear()
        self._tap_trackers.clear()
        self._zoom_trackers.clear()
        self._dial_trackers.clear()
        self._micro_trackers.clear()
        self._hold_timers.clear()
        self._prev_active.clear()

    def prune_missing_hands(self, active_hand_ids: Set[int]):
        for hid in list(self._pinch_trackers.keys()):
            if hid not in active_hand_ids:
                self._pinch_trackers.pop(hid, None)
                self._swipe_trackers.pop(hid, None)
                self._tap_trackers.pop(hid, None)
                self._zoom_trackers.pop(hid, None)
                self._dial_trackers.pop(hid, None)
                self._micro_trackers.pop(hid, None)
                self._hold_timers.pop(hid, None)
                self._prev_active.pop(hid, None)

    def update(
        self,
        hand_id: int,
        handedness: str,
        derived_pose: Optional[Any],
        motion_state: Optional[Any],
        finger_states: Optional[Any],
        raw_landmarks: np.ndarray,
        d_ref: float,
        timestamp: float,
    ) -> CompleteGestureState:
        """
        Synthesizes the complete Level 3 gesture by evaluating the observable event chain
        combining the static hand pose and kinematic motion primitives.
        """
        # Initialize trackers for this hand if new
        if hand_id not in self._pinch_trackers:
            self._pinch_trackers[hand_id] = PinchSequenceTracker()
            self._swipe_trackers[hand_id] = SwipeSequenceTracker()
            self._tap_trackers[hand_id] = AirTapSequenceTracker()
            self._zoom_trackers[hand_id] = PinchZoomSequenceTracker()
            self._dial_trackers[hand_id] = DialSequenceTracker()
            self._micro_trackers[hand_id] = MicroGestureTracker()

        pinch_trk = self._pinch_trackers[hand_id]
        swipe_trk = self._swipe_trackers[hand_id]
        tap_trk = self._tap_trackers[hand_id]
        zoom_trk = self._zoom_trackers[hand_id]
        dial_trk = self._dial_trackers[hand_id]
        micro_trk = self._micro_trackers[hand_id]

        pose_id_str = derived_pose.pose_id.value if (derived_pose and hasattr(derived_pose, "pose_id")) else "UNKNOWN"
        pose_name = derived_pose.canonical_name if (derived_pose and hasattr(derived_pose, "canonical_name")) else "NONE"
        pose_conf = float(derived_pose.confidence) if (derived_pose and hasattr(derived_pose, "confidence")) else 0.0

        m_prim = motion_state.motion_primitive if motion_state else MotionPrimitive.STATIONARY
        m_prim_str = m_prim.value if hasattr(m_prim, "value") else str(m_prim)
        palm_speed = float(motion_state.speed) if motion_state else 0.0
        tangential_accel = float(motion_state.tangential_acceleration) if motion_state else 0.0
        stroke_dur = float(motion_state.stroke_duration_ms) if motion_state else 0.0
        dwell_dur = float(motion_state.dwell_duration_ms) if motion_state else 0.0
        linearity = float(motion_state.linearity) if motion_state else 1.0
        net_disp = float(motion_state.displacement_magnitude) if motion_state else 0.0
        direction = str(getattr(motion_state, "primary_direction", "STATIONARY"))
        heading_deg = float(getattr(motion_state, "heading_deg", 0.0))
        angular_vel = float(getattr(motion_state, "angular_velocity", 0.0))
        vel = getattr(motion_state, "velocity", (0.0, 0.0, 0.0))
        palm_vel_z = float(vel[2]) if len(vel) > 2 else 0.0
        palm_speed_xy = float(getattr(motion_state, "speed_xy", palm_speed)) if motion_state else 0.0
        scale_rate = float(getattr(motion_state, "scale_rate", 0.0)) if motion_state else 0.0
        net_disp_vec = getattr(motion_state, "net_displacement", (0.0, 0.0, 0.0)) if motion_state else (0.0, 0.0, 0.0)
        disp_xy = math.sqrt(net_disp_vec[0]**2 + net_disp_vec[1]**2) if len(net_disp_vec) >= 2 else net_disp

        # Fingertip coordinates for pinch & micro tracking
        thumb_tip = (float(raw_landmarks[4, 0]), float(raw_landmarks[4, 1]), float(raw_landmarks[4, 2]))
        index_tip = (float(raw_landmarks[8, 0]), float(raw_landmarks[8, 1]), float(raw_landmarks[8, 2]))

        # Finger motion state for index
        f_motions = getattr(motion_state, "finger_motions", {}) if motion_state else {}
        idx_fmotion = f_motions.get("Index")

        # Resolve participating pointing / extended finger count (1 to 5)
        n_fingers = 1
        if derived_pose is not None and hasattr(derived_pose, "pointing_finger_count"):
            n_fingers = derived_pose.pointing_finger_count
        elif finger_states is not None:
            ext_c = 0
            for fn in ("index", "middle", "ring", "little"):
                fd = getattr(finger_states, fn, None)
                if fd and (getattr(fd, "state", None) == FingerStateEnum.EXTENDED or getattr(fd, "extension_ratio", 0.0) >= 1.25):
                    ext_c += 1
            t_ext = finger_states.thumb and (getattr(finger_states.thumb, "state", None) == FingerStateEnum.EXTENDED or getattr(finger_states.thumb, "extension_ratio", 0.0) >= 1.20)
            if ext_c == 4 and t_ext:
                n_fingers = 5
            elif ext_c >= 1:
                n_fingers = ext_c
        elif pose_name in ("OPEN_PALM", "SPREAD_FINGERS", "FIVE_FINGER"):
            n_fingers = 5
        elif pose_name in ("FOUR_FINGER",):
            n_fingers = 4
        elif pose_name in ("THREE_FINGER",):
            n_fingers = 3
        elif pose_name in ("DOUBLE_POINT", "PEACE"):
            n_fingers = 2
        elif pose_name in ("POINT", "GUN"):
            n_fingers = 1

        # ── 1. Evaluate Observable Event Trackers ───────────────────────
        pinch_phase, pinch_events, pinch_metrics = pinch_trk.update(
            thumb_tip=thumb_tip,
            index_tip=index_tip,
            d_ref=d_ref,
            palm_speed=palm_speed,
            net_displacement=net_disp,
            timestamp=timestamp,
        )

        swipe_phase, swipe_events, swipe_metrics = swipe_trk.update(
            pose_name=pose_name,
            motion_prim=m_prim,
            speed=palm_speed,
            displacement=net_disp,
            linearity=linearity,
            stroke_duration_ms=stroke_dur,
            direction=direction,
            timestamp=timestamp,
        )

        tap_phase, tap_events, tap_metrics = tap_trk.update(
            pose_name=pose_name,
            index_finger_motion=idx_fmotion,
            palm_speed=palm_speed,
            timestamp=timestamp,
        )

        zoom_phase, zoom_events, zoom_metrics = zoom_trk.update(
            thumb_tip=thumb_tip,
            index_tip=index_tip,
            d_ref=d_ref,
            palm_velocity_z=palm_vel_z,
            scale_rate=scale_rate,
            pose_name=pose_name,
            timestamp=timestamp,
        )

        dial_phase, dial_events, dial_metrics = dial_trk.update(
            motion_prim=m_prim,
            angular_velocity=angular_vel,
            heading_deg=heading_deg,
            speed=palm_speed,
            timestamp=timestamp,
        )

        micro_id, micro_events, micro_metrics = micro_trk.update(
            raw_landmarks=raw_landmarks,
            d_ref=d_ref,
            finger_motions=f_motions,
            pose_name=pose_name,
            palm_speed=palm_speed,
            timestamp=timestamp,
        )

        # ── 2. Track Static Hold Dwell Duration ─────────────────────────
        if (
            hand_id not in self._hold_timers
            or self._hold_timers[hand_id][0] != pose_name
            or (palm_speed > 0.28 and m_prim not in (MotionPrimitive.HOLD, MotionPrimitive.STATIONARY))
        ):
            self._hold_timers[hand_id] = (pose_name, timestamp)
        hold_duration_ms = (timestamp - self._hold_timers[hand_id][1]) * 1000.0

        # ── 3. High-Priority Event Composition Synthesis ───────────────

        # A. AXIAL HAND FLIPS (Pronation / Supination)
        if m_prim == MotionPrimitive.FLIP_TO_DORSAL:
            return CompleteGestureState(
                gesture_id=CompleteGestureId.G020_AXIAL_FLIP_DORSAL,
                canonical_name="FLIP TO DORSAL",
                category=GestureCategory.AXIAL_FLIP,
                hand_pose_id=pose_id_str,
                hand_pose_name=pose_name,
                motion_primitive=m_prim_str,
                confidence=0.92,
                phase="FLIP",
                event_sequence=["PALM_FACING", "PRONATION_ROLL", "DORSAL_FACING"],
                metrics={"roll_velocity": float(getattr(motion_state, "roll_velocity", 0.0))},
                is_active=True,
                is_stroke_completed=True,
                timestamp=timestamp,
            )
        elif m_prim == MotionPrimitive.FLIP_TO_PALM:
            return CompleteGestureState(
                gesture_id=CompleteGestureId.G021_AXIAL_FLIP_PALM,
                canonical_name="FLIP TO PALM",
                category=GestureCategory.AXIAL_FLIP,
                hand_pose_id=pose_id_str,
                hand_pose_name=pose_name,
                motion_primitive=m_prim_str,
                confidence=0.92,
                phase="FLIP",
                event_sequence=["DORSAL_FACING", "SUPINATION_ROLL", "PALM_FACING"],
                metrics={"roll_velocity": float(getattr(motion_state, "roll_velocity", 0.0))},
                is_active=True,
                is_stroke_completed=True,
                timestamp=timestamp,
            )

        # B. SINGLE-HAND FINGER MICRO-GESTURES (FM001–FM004)
        if micro_id != CompleteGestureId.NONE:
            micro_names = {
                CompleteGestureId.FM001_MICRO_PINCH_TAP: ("MICRO PINCH AIR CLICK", "CLICK"),
                CompleteGestureId.FM002_MICRO_MIDDLE_TAP: ("MICRO MIDDLE CONTEXT CLICK", "CONTEXT_CLICK"),
                CompleteGestureId.FM003_MICRO_DOUBLE_TAP: ("MICRO DOUBLE PINCH TAP", "DOUBLE_CLICK"),
                CompleteGestureId.FM004_MICRO_INDEX_TRIGGER: ("INDEX KNUCKLE TRIGGER FLEX", "TRIGGER"),
            }
            c_name, c_phase = micro_names.get(micro_id, ("FINGER MICRO GESTURE", "CLICK"))
            return CompleteGestureState(
                gesture_id=micro_id,
                canonical_name=c_name,
                category=GestureCategory.MICRO_GESTURE,
                hand_pose_id=pose_id_str,
                hand_pose_name=pose_name,
                motion_primitive=m_prim_str,
                confidence=0.94,
                phase=c_phase,
                event_sequence=micro_events,
                metrics=micro_metrics,
                is_active=True,
                is_stroke_completed=True,
                timestamp=timestamp,
            )

        # C. PINCH SEQUENCE (G006 Select / G007 Drag / G022 Zoom In / G023 Zoom Out)
        incompatible_pinch_poses = ("OPEN_PALM", "SPREAD_FINGERS", "THUMBS_UP", "THUMBS_DOWN", "PEACE")
        is_pinch_candidate = (
            pinch_phase in (PinchPhase.CONTACT, PinchPhase.HOLD, PinchPhase.DRAG)
            and pose_name not in incompatible_pinch_poses
        )
        if is_pinch_candidate:
            # Check if this held pinch is performing Depth or Aperture Zoom
            is_zooming = (
                pinch_phase != PinchPhase.CONTACT
                and zoom_phase in (PinchZoomPhase.ZOOMING_IN, PinchZoomPhase.ZOOMING_OUT)
                and not (palm_speed > 0.15 and net_disp > 0.02)
            )
            if is_zooming:
                is_zoom_in = zoom_phase == PinchZoomPhase.ZOOMING_IN
                z_id = CompleteGestureId.G022_PINCH_ZOOM_IN if is_zoom_in else CompleteGestureId.G023_PINCH_ZOOM_OUT
                z_name = "PINCH ZOOM IN" if is_zoom_in else "PINCH ZOOM OUT"
                return CompleteGestureState(
                    gesture_id=z_id,
                    canonical_name=z_name,
                    category=GestureCategory.ZOOM_INTERACTION,
                    hand_pose_id=pose_id_str,
                    hand_pose_name=pose_name,
                    motion_primitive=m_prim_str,
                    confidence=0.88,
                    phase=zoom_phase.value,
                    event_sequence=zoom_events,
                    metrics=zoom_metrics,
                    is_active=True,
                    is_stroke_completed=False,
                    timestamp=timestamp,
                )

            is_drag = pinch_phase == PinchPhase.DRAG or (palm_speed_xy > 0.15 and disp_xy > 0.02)
            g_id = CompleteGestureId.G007_PINCH_DRAG if is_drag else CompleteGestureId.G006_PINCH_SELECT
            c_name = "PINCH DRAG & HOLD" if is_drag else "PRECISION PINCH SELECT"
            conf = min(1.0, 0.70 + (0.30 if pinch_metrics["hold_duration_ms"] >= 80.0 else 0.10))
            return CompleteGestureState(
                gesture_id=g_id,
                canonical_name=c_name,
                category=GestureCategory.PINCH_INTERACTION,
                hand_pose_id=pose_id_str,
                hand_pose_name=pose_name,
                motion_primitive=m_prim_str,
                confidence=conf,
                phase=PinchPhase.DRAG.value if is_drag else pinch_phase.value,
                event_sequence=pinch_events,
                metrics=pinch_metrics,
                is_active=True,
                is_stroke_completed=False,
                timestamp=timestamp,
            )

        # D. DIRECTIONAL SWIPE STROKES & QUICK FLICK
        valid_swipe_pose = (
            pose_name in ("POINT", "DOUBLE_POINT", "GUN", "THREE_FINGER", "FOUR_FINGER", "FIVE_FINGER", "OPEN_PALM", "SPREAD_FINGERS", "KNIFE_EDGE", "PEACE")
            or n_fingers >= 1
        )

        is_stroke = valid_swipe_pose and (
            swipe_phase in (SwipePhase.STROKE, SwipePhase.COMPLETED) or (
                m_prim in (MotionPrimitive.MOVE_LEFT, MotionPrimitive.MOVE_RIGHT, MotionPrimitive.MOVE_UP, MotionPrimitive.MOVE_DOWN)
                and palm_speed >= 0.20
                and net_disp >= 0.030
            )
        )

        if not is_stroke and valid_swipe_pose and (tangential_accel >= 3.0 or (palm_speed >= 0.75 and stroke_dur <= 100.0)) and net_disp >= 0.03:
            return CompleteGestureState(
                gesture_id=CompleteGestureId.G017_QUICK_FLICK,
                canonical_name="QUICK FLICK DISMISS",
                category=GestureCategory.SWIPE_STROKE,
                hand_pose_id=pose_id_str,
                hand_pose_name=pose_name,
                motion_primitive=m_prim_str,
                confidence=0.92,
                phase="FLICK",
                event_sequence=["IMPULSE_SPIKE", f"FAST_FLICK ({direction})", "DISMISS"],
                metrics={"speed": palm_speed, "accel": tangential_accel, "duration_ms": stroke_dur},
                is_active=True,
                is_stroke_completed=True,
                timestamp=timestamp,
            )

        if is_stroke:
            dir_str = direction.upper()
            swipe_map = {
                (1, "LEFT"): (CompleteGestureId.G001_1F_SWIPE_LEFT, "1-FINGER SWIPE LEFT"),
                (1, "RIGHT"): (CompleteGestureId.G002_1F_SWIPE_RIGHT, "1-FINGER SWIPE RIGHT"),
                (1, "UP"): (CompleteGestureId.G003_1F_SWIPE_UP, "1-FINGER SWIPE UP"),
                (1, "DOWN"): (CompleteGestureId.G004_1F_SWIPE_DOWN, "1-FINGER SWIPE DOWN"),

                (2, "LEFT"): (CompleteGestureId.G001_2F_SWIPE_LEFT, "2-FINGER SWIPE LEFT"),
                (2, "RIGHT"): (CompleteGestureId.G002_2F_SWIPE_RIGHT, "2-FINGER SWIPE RIGHT"),
                (2, "UP"): (CompleteGestureId.G003_2F_SWIPE_UP, "2-FINGER SWIPE UP"),
                (2, "DOWN"): (CompleteGestureId.G004_2F_SWIPE_DOWN, "2-FINGER SWIPE DOWN"),

                (3, "LEFT"): (CompleteGestureId.G001_3F_SWIPE_LEFT, "3-FINGER SWIPE LEFT"),
                (3, "RIGHT"): (CompleteGestureId.G002_3F_SWIPE_RIGHT, "3-FINGER SWIPE RIGHT"),
                (3, "UP"): (CompleteGestureId.G003_3F_SWIPE_UP, "3-FINGER SWIPE UP"),
                (3, "DOWN"): (CompleteGestureId.G004_3F_SWIPE_DOWN, "3-FINGER SWIPE DOWN"),

                (4, "LEFT"): (CompleteGestureId.G001_4F_SWIPE_LEFT, "4-FINGER SWIPE LEFT"),
                (4, "RIGHT"): (CompleteGestureId.G002_4F_SWIPE_RIGHT, "4-FINGER SWIPE RIGHT"),
                (4, "UP"): (CompleteGestureId.G003_4F_SWIPE_UP, "4-FINGER SWIPE UP"),
                (4, "DOWN"): (CompleteGestureId.G004_4F_SWIPE_DOWN, "4-FINGER SWIPE DOWN"),

                (5, "LEFT"): (CompleteGestureId.G001_5F_SWIPE_LEFT, "5-FINGER SWIPE LEFT"),
                (5, "RIGHT"): (CompleteGestureId.G002_5F_SWIPE_RIGHT, "5-FINGER SWIPE RIGHT"),
                (5, "UP"): (CompleteGestureId.G003_5F_SWIPE_UP, "5-FINGER SWIPE UP"),
                (5, "DOWN"): (CompleteGestureId.G004_5F_SWIPE_DOWN, "5-FINGER SWIPE DOWN"),
            }

            key = (n_fingers, dir_str)
            g_id, c_name = swipe_map.get(key, (CompleteGestureId.G001_1F_SWIPE_LEFT, f"{n_fingers}-FINGER SWIPE {dir_str}"))
            conf = min(1.0, 0.65 + (0.35 * linearity))
            is_done = swipe_phase == SwipePhase.COMPLETED
            return CompleteGestureState(
                gesture_id=g_id,
                canonical_name=c_name,
                category=GestureCategory.SWIPE_STROKE,
                hand_pose_id=pose_id_str,
                hand_pose_name=pose_name,
                motion_primitive=m_prim_str,
                confidence=conf,
                phase=swipe_phase.value if swipe_phase != SwipePhase.IDLE else "STROKE",
                event_sequence=swipe_events or [f"PREPARE ({n_fingers}F)", f"STROKE ({dir_str})"],
                metrics=swipe_metrics,
                is_active=True,
                is_stroke_completed=is_done,
                timestamp=timestamp,
            )

        # F. AIR TAP CLICK (G011)
        if tap_phase == TapPhase.COMPLETED or (idx_fmotion and getattr(idx_fmotion, "is_tapping", False)):
            return CompleteGestureState(
                gesture_id=CompleteGestureId.G011_AIR_TAP,
                canonical_name="AIR TAP",
                category=GestureCategory.POINT_INTERACTION,
                hand_pose_id=pose_id_str,
                hand_pose_name=pose_name,
                motion_primitive=m_prim_str,
                confidence=0.90,
                phase="COMPLETED",
                event_sequence=tap_events or ["AIM", "STRIKE", "REBOUND"],
                metrics=tap_metrics,
                is_active=True,
                is_stroke_completed=True,
                timestamp=timestamp,
            )

        # G. ORBITAL ROTATION (1 to 5 Fingers Rotate CW / CCW)
        is_dial = dial_phase in (DialPhase.ROTATING_CW, DialPhase.ROTATING_CCW, DialPhase.DETENT) or m_prim in (MotionPrimitive.ROTATE_CW, MotionPrimitive.ROTATE_CCW)
        if is_dial:
            is_cw = dial_phase == DialPhase.ROTATING_CW or m_prim == MotionPrimitive.ROTATE_CW or dial_metrics["cumulative_angle_deg"] >= 0
            rot_dir = "CW" if is_cw else "CCW"
            rotate_map = {
                (1, "CW"): (CompleteGestureId.G018_1F_ROTATE_CW, "1-FINGER ROTATE CW"),
                (1, "CCW"): (CompleteGestureId.G019_1F_ROTATE_CCW, "1-FINGER ROTATE CCW"),

                (2, "CW"): (CompleteGestureId.G018_2F_ROTATE_CW, "2-FINGER ROTATE CW"),
                (2, "CCW"): (CompleteGestureId.G019_2F_ROTATE_CCW, "2-FINGER ROTATE CCW"),

                (3, "CW"): (CompleteGestureId.G018_3F_ROTATE_CW, "3-FINGER ROTATE CW"),
                (3, "CCW"): (CompleteGestureId.G019_3F_ROTATE_CCW, "3-FINGER ROTATE CCW"),

                (4, "CW"): (CompleteGestureId.G018_4F_ROTATE_CW, "4-FINGER ROTATE CW"),
                (4, "CCW"): (CompleteGestureId.G019_4F_ROTATE_CCW, "4-FINGER ROTATE CCW"),

                (5, "CW"): (CompleteGestureId.G018_5F_ROTATE_CW, "5-FINGER ROTATE CW"),
                (5, "CCW"): (CompleteGestureId.G019_5F_ROTATE_CCW, "5-FINGER ROTATE CCW"),
            }
            d_id, d_name = rotate_map.get((n_fingers, rot_dir), (CompleteGestureId.G018_1F_ROTATE_CW if is_cw else CompleteGestureId.G019_1F_ROTATE_CCW, f"{n_fingers}-FINGER ROTATE {rot_dir}"))
            return CompleteGestureState(
                gesture_id=d_id,
                canonical_name=d_name,
                category=GestureCategory.ORBITAL_DIAL,
                hand_pose_id=pose_id_str,
                hand_pose_name=pose_name,
                motion_primitive=m_prim_str,
                confidence=0.90,
                phase="DETENT" if dial_phase == DialPhase.DETENT else ("ROTATING_CW" if is_cw else "ROTATING_CCW"),
                event_sequence=dial_events or ["ORBIT_ENTER", f"ROTATING_{rot_dir}_{n_fingers}F"],
                metrics=dial_metrics,
                is_active=True,
                is_stroke_completed=dial_metrics.get("detent_count", 0) > 0,
                timestamp=timestamp,
            )

        # H. PUSH DWELL CONFIRM (G012)
        if pose_name in ("OPEN_PALM", "SPREAD_FINGERS") and m_prim == MotionPrimitive.MOVE_TOWARD and dwell_dur >= 200.0:
            return CompleteGestureState(
                gesture_id=CompleteGestureId.G012_PUSH_CONFIRM,
                canonical_name="PUSH DWELL CONFIRM",
                category=GestureCategory.HOLD_POSE,
                hand_pose_id=pose_id_str,
                hand_pose_name=pose_name,
                motion_primitive=m_prim_str,
                confidence=0.85,
                phase="PUSH_DWELL",
                event_sequence=["OPEN_PALM", "FORWARD_THRUST", f"DWELL ({int(dwell_dur)}ms)"],
                metrics={"speed": palm_speed, "dwell_duration_ms": dwell_dur},
                is_active=True,
                is_stroke_completed=False,
                timestamp=timestamp,
            )

        # I. POINT & HOVER (1 to 5 Fingers Point)
        is_point_posture = (
            pose_name in ("POINT", "DOUBLE_POINT", "GUN", "THREE_FINGER", "FOUR_FINGER", "FIVE_FINGER")
            or (n_fingers in (1, 2, 3, 4, 5) and pose_name not in incompatible_pinch_poses and pose_name not in ("GRAB", "THUMBS_UP", "THUMBS_DOWN"))
        )
        if is_point_posture and (palm_speed <= 0.28 or m_prim in (MotionPrimitive.HOLD, MotionPrimitive.STATIONARY)):
            point_map = {
                1: (CompleteGestureId.G005_1F_POINT, "1-FINGER POINT"),
                2: (CompleteGestureId.G005_2F_POINT, "2-FINGER POINT"),
                3: (CompleteGestureId.G005_3F_POINT, "3-FINGER POINT"),
                4: (CompleteGestureId.G005_4F_POINT, "4-FINGER POINT"),
                5: (CompleteGestureId.G005_5F_POINT, "5-FINGER POINT"),
            }
            p_id, p_name = point_map.get(n_fingers, (CompleteGestureId.G005_1F_POINT, f"{n_fingers}-FINGER POINT"))
            conf = min(1.0, 0.60 + 0.40 * min(1.0, hold_duration_ms / 300.0))
            return CompleteGestureState(
                gesture_id=p_id,
                canonical_name=p_name,
                category=GestureCategory.POINT_INTERACTION,
                hand_pose_id=pose_id_str,
                hand_pose_name=pose_name,
                motion_primitive=m_prim_str,
                confidence=conf,
                phase="POINTING",
                event_sequence=[f"{n_fingers}F_AIM", "STABLE_RAYCAST", f"HOVER ({int(hold_duration_ms)}ms)"],
                metrics={"speed": palm_speed, "finger_count": float(n_fingers), "dwell_duration_ms": hold_duration_ms},
                is_active=True,
                is_stroke_completed=False,
                timestamp=timestamp,
            )

        # H. CLOSED FIST WORLD GRAB (G008)
        if pose_name == "GRAB" and (palm_speed <= 0.20 or m_prim in (MotionPrimitive.HOLD, MotionPrimitive.STATIONARY)):
            return CompleteGestureState(
                gesture_id=CompleteGestureId.G008_FIST_GRAB,
                canonical_name="CLOSED FIST WORLD GRAB",
                category=GestureCategory.HOLD_POSE,
                hand_pose_id=pose_id_str,
                hand_pose_name=pose_name,
                motion_primitive=m_prim_str,
                confidence=pose_conf,
                phase="GRIP",
                event_sequence=["ALL_DIGITS_FLEXED", "PALM_ENCLOSED", f"GRIP_HOLD ({int(hold_duration_ms)}ms)"],
                metrics={"speed": palm_speed, "dwell_duration_ms": hold_duration_ms},
                is_active=True,
                is_stroke_completed=False,
                timestamp=timestamp,
            )

        # I. THUMBS UP / THUMBS DOWN / PEACE / OK
        if pose_name == "THUMBS_UP" and hold_duration_ms >= 100.0:
            return CompleteGestureState(
                gesture_id=CompleteGestureId.G013_THUMBS_UP,
                canonical_name="THUMBS UP APPROVAL",
                category=GestureCategory.HOLD_POSE,
                hand_pose_id=pose_id_str,
                hand_pose_name=pose_name,
                motion_primitive=m_prim_str,
                confidence=pose_conf,
                phase="APPROVAL",
                event_sequence=["THUMB_VERTICAL_UP", "FIST_ENCLOSED", f"AFFIRM_HOLD ({int(hold_duration_ms)}ms)"],
                metrics={"dwell_duration_ms": hold_duration_ms},
                is_active=True,
                is_stroke_completed=False,
                timestamp=timestamp,
            )
        elif pose_name == "THUMBS_DOWN" and hold_duration_ms >= 100.0:
            return CompleteGestureState(
                gesture_id=CompleteGestureId.G014_THUMBS_DOWN,
                canonical_name="THUMBS DOWN DISMISSAL",
                category=GestureCategory.HOLD_POSE,
                hand_pose_id=pose_id_str,
                hand_pose_name=pose_name,
                motion_primitive=m_prim_str,
                confidence=pose_conf,
                phase="DISMISSAL",
                event_sequence=["THUMB_VERTICAL_DOWN", "FIST_ENCLOSED", f"DISMISS_HOLD ({int(hold_duration_ms)}ms)"],
                metrics={"dwell_duration_ms": hold_duration_ms},
                is_active=True,
                is_stroke_completed=False,
                timestamp=timestamp,
            )
        elif pose_name == "PEACE" and hold_duration_ms >= 100.0:
            return CompleteGestureState(
                gesture_id=CompleteGestureId.G015_PEACE_MACRO,
                canonical_name="V-SIGN PEACE MACRO",
                category=GestureCategory.HOLD_POSE,
                hand_pose_id=pose_id_str,
                hand_pose_name=pose_name,
                motion_primitive=m_prim_str,
                confidence=pose_conf,
                phase="MACRO",
                event_sequence=["INDEX_MIDDLE_V_SHAPE", f"HOLD ({int(hold_duration_ms)}ms)"],
                metrics={"dwell_duration_ms": hold_duration_ms},
                is_active=True,
                is_stroke_completed=False,
                timestamp=timestamp,
            )
        elif pose_name == "OK_RING" and hold_duration_ms >= 100.0:
            return CompleteGestureState(
                gesture_id=CompleteGestureId.G016_OK_LOCK,
                canonical_name="OK SIGN LOCK",
                category=GestureCategory.HOLD_POSE,
                hand_pose_id=pose_id_str,
                hand_pose_name=pose_name,
                motion_primitive=m_prim_str,
                confidence=pose_conf,
                phase="LOCK",
                event_sequence=["THUMB_INDEX_RING", "OUTER_DIGITS_EXTENDED", f"LOCK_HOLD ({int(hold_duration_ms)}ms)"],
                metrics={"dwell_duration_ms": hold_duration_ms},
                is_active=True,
                is_stroke_completed=False,
                timestamp=timestamp,
            )

        # J. OPEN PALM ANCHOR & HOVER (G010)
        if pose_name in ("OPEN_PALM", "SPREAD_FINGERS") and (
            palm_speed <= 0.28 or m_prim in (MotionPrimitive.HOLD, MotionPrimitive.STATIONARY)
        ):
            conf = min(1.0, 0.70 + 0.30 * min(1.0, hold_duration_ms / 300.0))
            return CompleteGestureState(
                gesture_id=CompleteGestureId.G010_OPEN_PALM_HOVER,
                canonical_name="OPEN PALM ANCHOR & HOVER",
                category=GestureCategory.HOLD_POSE,
                hand_pose_id=pose_id_str,
                hand_pose_name=pose_name,
                motion_primitive=m_prim_str,
                confidence=conf,
                phase="ANCHORED",
                event_sequence=["ALL_DIGITS_EXTENDED", "PALM_FORWARD", f"ANCHOR_HOLD ({int(hold_duration_ms)}ms)"],
                metrics={"speed": palm_speed, "finger_count": 5.0, "dwell_duration_ms": hold_duration_ms},
                is_active=True,
                is_stroke_completed=False,
                timestamp=timestamp,
            )

        # K. GENERAL POSE DWELL & HOVER
        if pose_name not in ("NONE", "UNKNOWN") and (
            palm_speed <= 0.28 or m_prim in (MotionPrimitive.HOLD, MotionPrimitive.STATIONARY)
        ):
            conf = min(1.0, 0.65 + 0.35 * min(1.0, hold_duration_ms / 300.0))
            return CompleteGestureState(
                gesture_id=CompleteGestureId.NONE,
                canonical_name=f"{pose_name} HOVER",
                category=GestureCategory.HOLD_POSE,
                hand_pose_id=pose_id_str,
                hand_pose_name=pose_name,
                motion_primitive=m_prim_str,
                confidence=conf,
                phase="HOLDING",
                event_sequence=[f"POSE_{pose_name}", f"DWELL ({int(hold_duration_ms)}ms)"],
                metrics={"speed": palm_speed, "dwell_duration_ms": hold_duration_ms},
                is_active=True,
                is_stroke_completed=False,
                timestamp=timestamp,
            )

        # Default Neutral / Transitioning
        return CompleteGestureState(
            gesture_id=CompleteGestureId.NONE,
            canonical_name="NEUTRAL TRANSITION",
            category=GestureCategory.IDLE,
            hand_pose_id=pose_id_str,
            hand_pose_name=pose_name,
            motion_primitive=m_prim_str,
            confidence=0.0,
            phase="NEUTRAL",
            event_sequence=[pose_name, m_prim_str],
            metrics={"speed": palm_speed},
            is_active=False,
            is_stroke_completed=False,
            timestamp=timestamp,
        )
