"""
Gestura Stability Engine — Master Pipeline Component.

Sits directly between perception (Finger States, Hand Pose, Motion Primitives)
and intent interpretation (Temporal Intent Engine, Confirmed Gestures, Command Dispatcher):

Camera
  ↓
Finger States
  ↓
Hand Pose
  ↓
Motion Primitive
  ↓
STABILITY ENGINE  <── Decides if change is meaningful or natural human readjustment
  ↓
Temporal Intent
  ↓
Confirmed Gesture
  ↓
Command

Key Principles:
  1. Never change state because of one frame. Every state transition requires evidence.
  2. Distinguishes MICRO_ADJUSTMENT (ignore), TRANSITION (observe), and INTENTIONAL (allow).
  3. Commands are only eligible when Stability Score >= stability_threshold.
"""

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.gestures.hand_pose import DerivedHandPose, HandPoseId
from src.landmarks.finger_state import HandFingerStates
from src.motion.motion_primitive import MotionPrimitive, MotionState
from src.stability.finger_persistence import FingerPersistenceManager
from src.stability.intentional_hold import IntentionalHoldTracker
from src.stability.micro_adjustment_filter import MicroAdjustmentFilter
from src.stability.motion_deadzone import MotionDeadZone
from src.stability.observation_windows import LayerObservationWindows
from src.stability.pose_persistence import PosePersistenceManager
from src.stability.stability_config import StabilityConfig
from src.stability.stability_lifecycle import MovementCategory, PersistentFinger, PersistentPose, StabilityLifecycleState
from src.stability.stability_score import StabilityScoreCalculator
from src.stability.state_lock import StateLockSystem
from src.stability.transition_detector import TransitionDetector


@dataclass
class StabilizedHandContext:
    """Stabilized hand representations output by the Stability Engine."""
    hand_id: int
    handedness: str
    stabilized_finger_states: HandFingerStates
    persistent_fingers: Dict[str, PersistentFinger]
    stabilized_derived_pose: DerivedHandPose
    persistent_pose: PersistentPose
    filtered_palm_pos: Tuple[float, float, float]
    filtered_palm_velocity: Tuple[float, float, float]
    filtered_displacement: float
    in_dead_zone: bool
    movement_category: MovementCategory
    is_transitioning: bool
    transition_from_pose: str
    transition_to_pose: str
    transition_progress: float
    is_hold_satisfied: bool
    hold_duration_ms: float
    target_hold_ms: float
    is_state_locked: bool
    locked_interaction: str
    stability_score: float
    is_command_eligible: bool
    suppression_reasons: List[str]
    telemetry: Dict[str, Any] = field(default_factory=dict)


class PerHandStabilityTracker:
    """Encapsulates all stability components for a single tracked hand."""

    def __init__(self, hand_id: int, config: StabilityConfig):
        self.hand_id = hand_id
        self.config = config
        self.finger_persistence = FingerPersistenceManager(config)
        self.pose_persistence = PosePersistenceManager(config)
        self.motion_deadzone = MotionDeadZone(config)
        self.micro_filter = MicroAdjustmentFilter(config)
        self.transition_detector = TransitionDetector(config)
        self.hold_tracker = IntentionalHoldTracker(config)
        self.state_lock = StateLockSystem(config)
        self.score_calc = StabilityScoreCalculator(config)
        self.observation_windows = LayerObservationWindows(config)


class StabilityEngine:
    """
    Gestura Stability Engine.

    Eliminates false positives caused by natural hand readjustments, tremor,
    incidental digit twitches, and transient occlusions.
    """

    def __init__(self, config: Optional[StabilityConfig] = None):
        self.config = config or StabilityConfig()
        self._hand_trackers: Dict[int, PerHandStabilityTracker] = {}
        self._frame_count: int = 0
        self._latest_telemetry: Dict[str, Any] = {}

    def get_tracker(self, hand_id: int) -> PerHandStabilityTracker:
        """Retrieves or instantiates tracker for a given hand ID."""
        if hand_id not in self._hand_trackers:
            self._hand_trackers[hand_id] = PerHandStabilityTracker(hand_id, self.config)
        return self._hand_trackers[hand_id]

    def prune_missing_hands(self, active_hand_ids: List[int]) -> None:
        """Clears state for hands no longer present in camera view."""
        stale_ids = [hid for hid in self._hand_trackers if hid not in active_hand_ids]
        for hid in stale_ids:
            del self._hand_trackers[hid]

    def process_hand_stability(
        self,
        hand_id: int,
        handedness: str,
        raw_landmarks: np.ndarray,
        instantaneous_finger_states: Optional[HandFingerStates],
        instantaneous_pose: Optional[DerivedHandPose],
        instantaneous_motion: Optional[MotionState],
        candidate_gesture: str,
        candidate_confidence: float,
        palm_center: Tuple[float, float, float],
        d_ref: float,
        detection_confidence: float,
        is_active_interaction: bool = False,
        is_release_gesture: bool = False,
        timestamp: float = 0.0,
    ) -> StabilizedHandContext:
        """
        Processes perception inputs through the complete stability stack:
          1. Finger State Persistence & Confidence Decay
          2. Hand Pose Persistence & Hysteresis
          3. Motion Dead-Zone Filtering
          4. Transition Detection
          5. Micro-Adjustment & Jitter Evaluation
          6. Intentional Hold Evaluation
          7. State Locking
          8. Multi-Signal Stability Score Calculation
        """
        self._frame_count += 1
        t_start = time.perf_counter()
        tracker = self.get_tracker(hand_id)

        # ── 1. Finger State Persistence ──────────────────────────────────────
        stab_fingers, persistent_fingers = tracker.finger_persistence.update(
            instantaneous_states=instantaneous_finger_states,
            detection_confidence=detection_confidence,
            timestamp=timestamp,
        )

        # ── 2. Pose Persistence with Asymmetric Hysteresis ───────────────────
        stab_pose, persistent_pose = tracker.pose_persistence.update(
            instantaneous_pose=instantaneous_pose,
            timestamp=timestamp,
        )

        # ── 3. Motion Dead Zone ──────────────────────────────────────────────
        filtered_pos, filtered_vel, filtered_disp, in_dead_zone = tracker.motion_deadzone.update(
            palm_pos=palm_center,
            d_ref=d_ref,
            is_active_manipulation=is_active_interaction,
            dt=0.033,
        )

        # ── 4. Transition Detection ──────────────────────────────────────────
        is_trans, from_p, to_p, trans_prog = tracker.transition_detector.update(
            current_confirmed_pose=persistent_pose.canonical_name,
            incoming_candidate_pose=instantaneous_pose.canonical_name if instantaneous_pose else None,
            finger_states=stab_fingers,
            timestamp=timestamp,
        )

        # ── 5. Micro-Adjustment vs Transition vs Intentional Movement ────────
        finger_changes = {
            d: pf.state for d, pf in persistent_fingers.items()
            if pf.lifecycle in (StabilityLifecycleState.POSSIBLE_CHANGE, StabilityLifecycleState.OBSERVING)
        }
        pose_changed = persistent_pose.lifecycle in (StabilityLifecycleState.POSSIBLE_CHANGE, StabilityLifecycleState.OBSERVING)

        m_category, suppression_reasons = tracker.micro_filter.evaluate(
            palm_pos=filtered_pos,
            palm_speed=math.sqrt(filtered_vel[0]**2 + filtered_vel[1]**2 + filtered_vel[2]**2),
            detection_confidence=detection_confidence,
            finger_changes=finger_changes,
            pose_changed=pose_changed,
            is_transitioning=is_trans,
            is_in_dead_zone=in_dead_zone,
            timestamp=timestamp,
        )

        # ── 6. Intentional Hold Dwell (starts ONLY after pose is confirmed) ──
        is_still = in_dead_zone or (
            instantaneous_motion is not None
            and instantaneous_motion.motion_primitive in (MotionPrimitive.STATIONARY, MotionPrimitive.HOLD)
        )
        is_hold_satisfied, hold_dur_ms, target_hold_ms = tracker.hold_tracker.update(
            stable_pose_name=persistent_pose.canonical_name,
            is_pose_confirmed=(persistent_pose.canonical_name not in ("NONE", "UNKNOWN")),
            is_in_dead_zone_or_still=is_still,
            timestamp=timestamp,
        )

        # ── 7. State Locking ─────────────────────────────────────────────────
        is_locked, locked_interaction = tracker.state_lock.update(
            is_active_interaction=is_active_interaction,
            current_pose_name=persistent_pose.canonical_name,
            detection_confidence=detection_confidence,
            is_release_gesture=is_release_gesture,
            timestamp=timestamp,
        )

        # ── 8. Observation Windows Consensus ─────────────────────────────────
        consensus = tracker.observation_windows.update(
            finger_dict={d: pf.state for d, pf in persistent_fingers.items()},
            pose_name=persistent_pose.canonical_name,
            motion_primitive=instantaneous_motion.motion_primitive.value if instantaneous_motion else "STATIONARY",
            gesture_name=candidate_gesture,
            timestamp=timestamp,
        )

        # ── 9. Multi-Signal Stability Score Calculator ───────────────────────
        mean_finger_stability = sum(pf.stability for pf in persistent_fingers.values()) / max(1, len(persistent_fingers))
        temporal_cons = (consensus.get("pose", 0.0) * 0.50 + consensus.get("gesture", 0.0) * 0.50)
        motion_cons = 1.0 if in_dead_zone else (
            float(instantaneous_motion.linearity) if (instantaneous_motion and hasattr(instantaneous_motion, "linearity")) else 0.85
        )

        stability_score, is_eligible, score_breakdown = tracker.score_calc.compute(
            recognition_conf=detection_confidence * float(getattr(instantaneous_pose, "confidence", 1.0) if instantaneous_pose else 0.8),
            temporal_consistency=temporal_cons,
            motion_consistency=motion_cons,
            persistence_score=(persistent_pose.stability * 0.60 + mean_finger_stability * 0.40),
        )

        # Command eligibility rules:
        # 1. Stability score must exceed threshold
        # 2. Movement category must be INTENTIONAL (never MICRO_ADJUSTMENT or TRANSITION)
        # 3. Intentional hold dwell must be satisfied if applicable
        if m_category == MovementCategory.MICRO_ADJUSTMENT:
            is_eligible = False
            suppression_reasons.append("Micro-adjustment suppressed from command execution")
        elif m_category == MovementCategory.TRANSITION:
            is_eligible = False
            suppression_reasons.append(f"Pose transition in progress ({from_p} -> {to_p})")

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        # Construct comprehensive developer stability telemetry
        telemetry = {
            "finger_layer": {
                "states": tracker.finger_persistence.get_finger_telemetry(),
                "mean_stability": round(mean_finger_stability, 2),
            },
            "pose_layer": {
                **tracker.pose_persistence.get_pose_telemetry(),
                "enter_threshold": self.config.enter_threshold,
                "exit_threshold": self.config.exit_threshold,
            },
            "motion_layer": {
                **tracker.motion_deadzone.get_deadzone_telemetry(),
                "filtered_velocity": [round(v, 3) for v in filtered_vel],
                "filtered_displacement": round(filtered_disp, 4),
            },
            "stability_layer": {
                "category": m_category.value,
                "is_micro_adjustment": m_category == MovementCategory.MICRO_ADJUSTMENT,
                "is_transition": m_category == MovementCategory.TRANSITION,
                "is_intentional": m_category == MovementCategory.INTENTIONAL,
                "is_locked": is_locked,
                "locked_interaction": locked_interaction,
                "is_hold_satisfied": is_hold_satisfied,
                "hold_duration_ms": round(hold_dur_ms, 1),
                "target_hold_ms": round(target_hold_ms, 1),
                "observation_windows": tracker.observation_windows.get_window_telemetry(),
                "transition_info": tracker.transition_detector.get_telemetry(),
            },
            "intent_layer": {
                "candidate_gesture": candidate_gesture,
                "stability_score": round(stability_score, 2),
                "stability_threshold": self.config.stability_threshold,
                "is_command_eligible": is_eligible,
                "breakdown": score_breakdown,
                "suppression_reasons": suppression_reasons,
            },
            "diagnostics": {
                "latency_ms": round(latency_ms, 2),
                "frame_count": self._frame_count,
            },
        }
        self._latest_telemetry = telemetry

        return StabilizedHandContext(
            hand_id=hand_id,
            handedness=handedness,
            stabilized_finger_states=stab_fingers,
            persistent_fingers=persistent_fingers,
            stabilized_derived_pose=stab_pose,
            persistent_pose=persistent_pose,
            filtered_palm_pos=filtered_pos,
            filtered_palm_velocity=filtered_vel,
            filtered_displacement=filtered_disp,
            in_dead_zone=in_dead_zone,
            movement_category=m_category,
            is_transitioning=is_trans,
            transition_from_pose=from_p,
            transition_to_pose=to_p,
            transition_progress=trans_prog,
            is_hold_satisfied=is_hold_satisfied,
            hold_duration_ms=hold_dur_ms,
            target_hold_ms=target_hold_ms,
            is_state_locked=is_locked,
            locked_interaction=locked_interaction,
            stability_score=stability_score,
            is_command_eligible=is_eligible,
            suppression_reasons=suppression_reasons,
            telemetry=telemetry,
        )

    def get_latest_telemetry(self) -> Dict[str, Any]:
        """Returns the most recent stability telemetry dictionary."""
        return self._latest_telemetry
