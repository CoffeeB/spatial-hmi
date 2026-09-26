"""
Temporal Intent Engine for Gestura.

Central coordinator implementing the layered temporal perception pipeline:
    Camera
        ↓
    Finger States (Stabilized over 1–3 frames)
        ↓
    Hand Pose (Confirmed over 3–5 frames with hysteresis)
        ↓
    Motion Primitive (Analyzed over 5–10 frames)
        ↓
    Gesture Candidate (Accumulated over 8–15 frames)
        ↓
    Temporal Intent Engine (State Machine + Priority Manager + Intent Lock)
        ↓
    Confirmed Gesture
        ↓
    Command Dispatcher

Guarantee: The system never reacts to a single frame. Every layer must stabilize
before the next layer is evaluated.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.gestures.gesture_types import GestureType, RecognizedGesture
from src.gestures.hand_pose import DerivedHandPose, HandPoseClassifier, HandPoseId
from src.intent.intent_lock import IntentLockSystem
from src.intent.observation_window import (
    FingerStateStabilizer,
    GestureCandidateBuffer,
    MotionPrimitiveAnalyzer,
    ObservationWindowManager,
    PoseConfirmationManager,
)
from src.intent.intention_decipherer import Level2MotionIntentionDecipherer, Level3GestureIntentionDecipherer
from src.intent.priority_manager import GesturePriorityManager, GesturePriorityTier
from src.intent.state_machine import IntentContext, InteractionState, InteractionStateMachine
from src.intent.temporal_intent_config import TemporalIntentConfig
from src.landmarks.finger_state import FingerStateClassifier, HandFingerStates
from src.motion.motion_primitive import MotionPrimitive, MotionState
from src.utils.logging_config import setup_logger

logger = setup_logger("temporal_intent_engine")


@dataclass
class TemporalIntentTelemetry:
    """
    Rich real-time developer debug telemetry exposing every perception layer
    for developer overlays and client WebSockets.
    """
    # Finger Layer
    finger_states: Dict[str, str] = field(default_factory=dict)
    landmark_confidence: float = 1.0
    finger_is_stable: bool = False

    # Pose Layer
    active_pose: str = "NONE"
    candidate_pose: str = "NONE"
    pose_confidence: float = 0.0
    pose_is_confirmed: bool = False
    pose_frames_consistent: int = 0
    pose_oscillation_rate: float = 0.0

    # Motion Layer
    velocity_vector: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    speed: float = 0.0
    tangential_acceleration: float = 0.0
    net_displacement: float = 0.0
    direction_consistency: float = 1.0
    trajectory_linearity: float = 1.0
    movement_duration_ms: float = 0.0
    primary_direction: str = "STATIONARY"
    motion_primitive: str = "STATIONARY"
    motion_is_intentional: bool = False

    # Intent Layer
    current_state: str = "IDLE"
    candidate_gesture: str = "NONE"
    confirmed_gesture: str = "NONE"
    intent_confidence: float = 0.0
    active_lock: Dict[str, Any] = field(default_factory=dict)
    priority_tier: str = "TIER_5_IDLE"
    easing_factor: float = 1.0

    # Level 0–3 Multi-Level Intention Deciphering
    finger_intentions: Dict[str, str] = field(default_factory=dict)
    focal_fingers: List[str] = field(default_factory=list)
    finger_focus_weights: Dict[str, float] = field(default_factory=dict)
    pose_intention: str = "RESTING_PALM"
    pose_intended_action: str = ""
    pose_focal_digits: List[str] = field(default_factory=list)
    motion_intention: str = "STATIC_POSTURE"
    motion_intentionality_score: float = 0.0
    motion_is_purposeful: bool = False
    task_intent: str = "IDLE_MONITORING"
    predicted_next_intent: str = "NONE"

    # Performance & Diagnostics
    fps: float = 60.0
    latency_ms: float = 0.0
    timestamp: float = 0.0
    frame_index: int = 0
    diagnostics: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "finger_layer": {
                "states": self.finger_states,
                "confidence": round(self.landmark_confidence, 2),
                "is_stable": self.finger_is_stable,
            },
            "pose_layer": {
                "active_pose": self.active_pose,
                "candidate_pose": self.candidate_pose,
                "confidence": round(self.pose_confidence, 2),
                "is_confirmed": self.pose_is_confirmed,
                "frames_consistent": self.pose_frames_consistent,
                "oscillation_rate": self.pose_oscillation_rate,
            },
            "motion_layer": {
                "velocity": self.velocity_vector,
                "speed": round(self.speed, 3),
                "tangential_accel": round(self.tangential_acceleration, 2),
                "displacement": round(self.net_displacement, 4),
                "direction_consistency": round(self.direction_consistency, 2),
                "linearity": round(self.trajectory_linearity, 2),
                "duration_ms": round(self.movement_duration_ms, 1),
                "direction": self.primary_direction,
                "primitive": self.motion_primitive,
                "is_intentional": self.motion_is_intentional,
            },
            "intent_layer": {
                "current_state": self.current_state,
                "candidate_gesture": self.candidate_gesture,
                "confirmed_gesture": self.confirmed_gesture,
                "confidence": round(self.intent_confidence, 2),
                "is_locked": self.active_lock.get("is_locked", False),
                "locked_by": self.active_lock.get("locked_by", "NONE"),
                "priority_tier": self.priority_tier,
                "easing_factor": round(self.easing_factor, 3),
            },
            "intention_deciphering": {
                "level_0_finger": {
                    "intentions": self.finger_intentions,
                    "focal_fingers": self.focal_fingers,
                    "focus_weights": self.finger_focus_weights,
                },
                "level_1_pose": {
                    "intention": self.pose_intention,
                    "intended_action": self.pose_intended_action,
                    "focal_digits": self.pose_focal_digits,
                },
                "level_2_motion": {
                    "intention": self.motion_intention,
                    "intentionality_score": round(self.motion_intentionality_score, 2),
                    "is_purposeful": self.motion_is_purposeful,
                },
                "level_3_gesture": {
                    "task_intent": self.task_intent,
                    "predicted_next_intent": self.predicted_next_intent,
                },
            },
            "performance": {
                "fps": round(self.fps, 1),
                "latency_ms": round(self.latency_ms, 2),
                "frame_index": self.frame_index,
            },
            "diagnostics": self.diagnostics,
        }


class TemporalIntentEngine:
    """
    Temporal Intent Engine for Gestura.
    Guarantees evidence-based gesture recognition across observation windows.
    """

    def __init__(self, config: Optional[TemporalIntentConfig] = None):
        self.config = config or TemporalIntentConfig()

        # Observation window managers per hand_id
        self._window_managers: Dict[int, ObservationWindowManager] = {}

        # Intent State Machine, Lock, Priority Manager
        self.fsm = InteractionStateMachine(config=self.config)
        self.lock_system = self.fsm.lock_system
        self.priority_manager = self.fsm.priority_manager

        # Performance tracking
        self._frame_count: int = 0
        self._last_frame_time: float = time.time()
        self._fps: float = 60.0
        self._latest_telemetry: Optional[TemporalIntentTelemetry] = None

    def reset(self):
        """Resets all buffers and intent state."""
        self._window_managers.clear()
        self.fsm.reset()
        self._frame_count = 0
        self._latest_telemetry = None

    def get_window_manager(self, hand_id: int) -> ObservationWindowManager:
        """Retrieves or instantiates the ObservationWindowManager for the specified hand."""
        if hand_id not in self._window_managers:
            self._window_managers[hand_id] = ObservationWindowManager(self.config)
        return self._window_managers[hand_id]

    def prune_missing_hands(self, active_hand_ids: set):
        """Cleans up temporal windows for hands that have exited the interaction zone."""
        dead_ids = [hid for hid in self._window_managers if hid not in active_hand_ids]
        for hid in dead_ids:
            self._window_managers.pop(hid, None)

    def process_hand_temporal(
        self,
        hand_id: int,
        handedness: str,
        raw_landmarks: np.ndarray,
        instantaneous_finger_states: Optional[HandFingerStates],
        instantaneous_pose: Optional[DerivedHandPose],
        palm_center: Tuple[float, float, float],
        d_ref: float,
        detection_confidence: float,
        candidate_gesture_name: str,
        candidate_confidence: float,
        timestamp: float,
    ) -> Tuple[HandFingerStates, DerivedHandPose, MotionPrimitive, IntentContext, TemporalIntentTelemetry]:
        """
        Executes the full layered temporal perception pipeline:
            1. Finger State Stabilization (1–3 frames)
            2. Hand Pose Confirmation with Hysteresis (3–5 frames)
            3. Motion Primitive Analysis (5–10 frames)
            4. Gesture Candidate Validation (8–15 frames)
            5. Intent Lock & Priority Arbitration
            6. Intent State Machine (IDLE -> OBSERVING -> CANDIDATE -> CONFIRMED -> ACTIVE -> RELEASING -> IDLE)
        """
        t_start = time.perf_counter()
        self._frame_count += 1
        dt = max(timestamp - self._last_frame_time, 0.001)
        self._fps = 0.90 * self._fps + 0.10 * (1.0 / dt)
        self._last_frame_time = timestamp

        wm = self.get_window_manager(hand_id)
        diagnostics: List[str] = []

        # ── Layer 0: Finger State Stabilization (1–3 frames) ────────────────
        smoothed_landmarks = wm.finger_stabilizer.smooth_landmarks(raw_landmarks)
        stabilized_finger_states, finger_summary = wm.finger_stabilizer.update(
            finger_states=instantaneous_finger_states,
            detection_confidence=detection_confidence,
            timestamp=timestamp,
        )
        if not finger_summary.get("is_stable", False):
            diagnostics.append("Finger states transitioning / stabilizing")

        # ── Layer 1: Hand Pose Confirmation & Hysteresis (3–5 frames) ───────
        (
            conf_pose_id,
            conf_pose_name,
            conf_pose_score,
            is_pose_confirmed,
            pose_telem,
        ) = wm.pose_manager.update(
            instantaneous_pose=instantaneous_pose,
            timestamp=timestamp,
        )

        # Build stable DerivedHandPose
        if is_pose_confirmed and instantaneous_pose is not None:
            confirmed_derived_pose = DerivedHandPose(
                pose_id=conf_pose_id,
                canonical_name=conf_pose_name,
                confidence=conf_pose_score,
                configuration=instantaneous_pose.configuration,
                satisfied_predicates=instantaneous_pose.satisfied_predicates,
                diagnostics=instantaneous_pose.diagnostics + [f"Confirmed across {pose_telem['confirmed_frames']} frames"],
            )
        elif instantaneous_pose is not None:
            # Pose unconfirmed yet — mark candidate
            confirmed_derived_pose = DerivedHandPose(
                pose_id=HandPoseId.UNKNOWN,
                canonical_name="UNCONFIRMED",
                confidence=conf_pose_score,
                configuration=instantaneous_pose.configuration,
                diagnostics=["Pose candidate accumulating evidence"],
            )
            diagnostics.append(f"Pose candidate {instantaneous_pose.canonical_name} accumulating stability")
        else:
            confirmed_derived_pose = DerivedHandPose(
                pose_id=HandPoseId.UNKNOWN,
                canonical_name="NONE",
                confidence=0.0,
                configuration=None,
            )

        # ── Layer 2: Motion Primitive Analysis (5–10 frames) ────────────────
        motion_prim, motion_metrics = wm.motion_analyzer.update(
            palm_center=palm_center,
            d_ref=d_ref,
            timestamp=timestamp,
        )

        # ── Layer 3: Gesture Candidate Validation (8–15 frames) ────────────
        evidence_score, match_frames, meets_thresh = wm.gesture_buffer.update(
            candidate_gesture=candidate_gesture_name,
            instantaneous_confidence=candidate_confidence,
            timestamp=timestamp,
        )

        # ── Layer 4 & 5: Intent State Machine, Lock & Priority Arbitration ──
        # Resolve mapped GestureType for FSM
        mapped_gtype = self._map_to_gesture_type(candidate_gesture_name)
        recognized_obj = RecognizedGesture(
            gesture=mapped_gtype,
            confidence=candidate_confidence,
            hand_id=hand_id,
            handedness=handedness,
            timestamp=timestamp,
        )

        intent_ctx = self.fsm.update(
            hand_detected=True,
            recognized_gesture=recognized_obj,
            timestamp=timestamp,
        )

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        # Decipher Intentions across Level 0, 1, 2, and 3
        f_intentions = {}
        f_focus_weights = {}
        f_focal_list = []
        active_hand_states = stabilized_finger_states or instantaneous_finger_states
        if active_hand_states:
            for fname, fdetail in [
                ("Thumb", active_hand_states.thumb),
                ("Index", active_hand_states.index),
                ("Middle", active_hand_states.middle),
                ("Ring", active_hand_states.ring),
                ("Little", active_hand_states.little),
            ]:
                if fdetail:
                    f_intentions[fname] = getattr(fdetail, "intention", "STABILIZATION")
                    f_focus_weights[fname] = round(float(getattr(fdetail, "focus_weight", 0.0)), 2)
                    if getattr(fdetail, "is_focal", False):
                        f_focal_list.append(fname)

        pose_intent = getattr(confirmed_derived_pose, "pose_intention", "RESTING_PALM")
        pose_action = getattr(confirmed_derived_pose, "intended_action", "")
        pose_focal = getattr(confirmed_derived_pose, "focal_digits", [])

        mot_dec = Level2MotionIntentionDecipherer.decipher(
            speed=motion_metrics.get("speed", 0.0),
            displacement=motion_metrics.get("net_displacement", 0.0),
            linearity=motion_metrics.get("linearity", 1.0),
            direction_consistency=motion_metrics.get("direction_consistency", 1.0),
            motion_primitive=motion_prim.value if hasattr(motion_prim, "value") else str(motion_prim),
            active_pose=conf_pose_name,
            duration_ms=motion_metrics.get("duration_ms", 0.0),
        )
        mot_intent = mot_dec["motion_intention"]
        mot_score = mot_dec["intentionality_score"]
        mot_purposeful = mot_dec["is_purposeful"]

        active_or_cand_gesture = (
            intent_ctx.active_gesture.value
            if intent_ctx.state in (InteractionState.CONFIRMED, InteractionState.ACTIVE)
            else candidate_gesture_name
        )
        l3_dec = Level3GestureIntentionDecipherer.decipher(
            gesture_name=active_or_cand_gesture,
            pose_intention=pose_intent,
            motion_intention=mot_intent,
            intent_state=intent_ctx.state.value if hasattr(intent_ctx.state, "value") else str(intent_ctx.state),
            focal_digits=f_focal_list or pose_focal,
        )
        l3_intent = l3_dec["task_intent"]
        next_intent = l3_dec["predicted_next_intent"]

        # Construct Developer Debug Telemetry
        telemetry = TemporalIntentTelemetry(
            finger_states=finger_summary,
            landmark_confidence=detection_confidence,
            finger_is_stable=finger_summary.get("is_stable", False),
            active_pose=conf_pose_name,
            candidate_pose=pose_telem["candidate_pose"],
            pose_confidence=conf_pose_score,
            pose_is_confirmed=is_pose_confirmed,
            pose_frames_consistent=pose_telem["frames_consistent"],
            pose_oscillation_rate=pose_telem["oscillation_rate"],
            velocity_vector=motion_metrics["velocity"],
            speed=motion_metrics["speed"],
            tangential_acceleration=motion_metrics["tangential_acceleration"],
            net_displacement=motion_metrics["net_displacement"],
            direction_consistency=motion_metrics["direction_consistency"],
            trajectory_linearity=motion_metrics["linearity"],
            movement_duration_ms=motion_metrics["duration_ms"],
            primary_direction=motion_metrics["primary_direction"],
            motion_primitive=motion_prim.value,
            motion_is_intentional=motion_metrics["is_intentional"],
            current_state=intent_ctx.state.value,
            candidate_gesture=intent_ctx.candidate_gesture.value if intent_ctx.candidate_gesture != GestureType.NONE else candidate_gesture_name,
            confirmed_gesture=intent_ctx.active_gesture.value if intent_ctx.state in (InteractionState.CONFIRMED, InteractionState.ACTIVE) else "NONE",
            intent_confidence=intent_ctx.intent_confidence,
            active_lock=self.lock_system.get_telemetry(),
            priority_tier=intent_ctx.priority_tier or "TIER_5_IDLE",
            easing_factor=intent_ctx.easing_factor,
            finger_intentions=f_intentions,
            focal_fingers=f_focal_list,
            finger_focus_weights=f_focus_weights,
            pose_intention=pose_intent,
            pose_intended_action=pose_action,
            pose_focal_digits=pose_focal,
            motion_intention=mot_intent,
            motion_intentionality_score=mot_score,
            motion_is_purposeful=mot_purposeful,
            task_intent=l3_intent,
            predicted_next_intent=next_intent,
            fps=self._fps,
            latency_ms=latency_ms,
            timestamp=timestamp,
            frame_index=self._frame_count,
            diagnostics=diagnostics,
        )
        self._latest_telemetry = telemetry

        return (
            stabilized_finger_states or instantaneous_finger_states,
            confirmed_derived_pose,
            motion_prim,
            intent_ctx,
            telemetry,
        )

    def _map_to_gesture_type(self, gname: str) -> GestureType:
        """Translates Level 3 gesture identifiers into legacy GestureType enum safely."""
        g = str(gname).upper()
        if hasattr(GestureType, g):
            return getattr(GestureType, g)
        if "SWIPE_LEFT" in g:
            return GestureType.SWIPE_LEFT
        elif "SWIPE_RIGHT" in g:
            return GestureType.SWIPE_RIGHT
        elif "SWIPE_UP" in g:
            return GestureType.SWIPE_UP
        elif "SWIPE_DOWN" in g:
            return GestureType.SWIPE_DOWN
        elif "PINCH" in g:
            return GestureType.PINCH
        elif "GRAB" in g or "FIST" in g or "CUPPED" in g:
            return GestureType.GRAB
        elif "POINT" in g or g in ("THREE_FINGER", "FOUR_FINGER", "FIVE_FINGER", "DOUBLE_POINT"):
            return GestureType.POINT
        elif "SPREAD_FINGERS" in g or "SPREAD" in g:
            return GestureType.SPREAD_FINGERS
        elif "OPEN_PALM" in g:
            return GestureType.OPEN_PALM
        elif "THUMBS_UP" in g:
            return getattr(GestureType, "THUMBS_UP", GestureType.NONE)
        elif "THUMBS_DOWN" in g:
            return getattr(GestureType, "THUMBS_DOWN", GestureType.NONE)
        elif "PEACE" in g:
            return getattr(GestureType, "PEACE", GestureType.NONE)
        elif "OK_RING" in g:
            return getattr(GestureType, "OK_RING", GestureType.NONE)
        elif "GUN" in g:
            return getattr(GestureType, "GUN", GestureType.POINT)
        elif "SHAKA" in g:
            return getattr(GestureType, "SHAKA", GestureType.NONE)
        elif "TWO_HAND" in g or "BIMANUAL" in g:
            return getattr(GestureType, "TWO_HAND_PINCH", GestureType.NONE)
        return GestureType.NONE
