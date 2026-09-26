"""
Observation Window Manager for Gestura.

Implements layered temporal perception windows for:
  1. Level 0 Finger States (1–3 frames): Landmark smoothing, jitter rejection & stability.
  2. Level 1 Hand Poses (3–5 frames): Hysteresis, oscillation suppression & confirmation.
  3. Level 2 Motion Primitives (5–10 frames): Trajectory linearity, velocity, acceleration & directional consistency.
  4. Level 3 Gestures (8–15 frames): Temporal evidence accumulation for gesture candidates.
  5. Level 4 Two-Hand Gestures (10–20 frames): Bimanual synchronization and relative stability.

Ensures the core principle: "Interpretation requires evidence, not a single frame."
"""

from collections import Counter, deque
import math
import time
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np

from src.gestures.hand_pose import DerivedHandPose, HandPoseId
from src.intent.temporal_intent_config import TemporalIntentConfig
from src.landmarks.finger_state import FingerName, FingerStateDetail, FingerStateEnum, HandFingerStates
from src.motion.motion_primitive import DynamicState, MotionPrimitive, MotionState


class FingerStateStabilizer:
    """
    Stabilizes Level 0 Finger States over 1–3 frames.
    Applies exponential smoothing to raw landmarks, filters coordinate jitter,
    and requires temporal majority before flipping a finger's posture state.
    """

    def __init__(self, config: Optional[TemporalIntentConfig] = None):
        self.config = config or TemporalIntentConfig()
        self.window_size = self.config.windows.finger_window
        self.alpha = self.config.finger.smoothing_alpha
        self.min_conf = self.config.finger.min_confidence

        self._smoothed_landmarks: Optional[np.ndarray] = None
        self._state_history: Dict[str, Deque[FingerStateEnum]] = {
            "Thumb": deque(maxlen=self.window_size),
            "Index": deque(maxlen=self.window_size),
            "Middle": deque(maxlen=self.window_size),
            "Ring": deque(maxlen=self.window_size),
            "Little": deque(maxlen=self.window_size),
        }
        self._confidence_history: Deque[float] = deque(maxlen=self.window_size)
        self._last_stabilized_dict: Dict[str, str] = {}
        self._last_confidence: float = 0.0

    def reset(self):
        self._smoothed_landmarks = None
        for q in self._state_history.values():
            q.clear()
        self._confidence_history.clear()
        self._last_stabilized_dict.clear()
        self._last_confidence = 0.0

    def smooth_landmarks(self, raw_landmarks: np.ndarray) -> np.ndarray:
        """
        Validates raw coordinates and applies Exponential Moving Average (EMA)
        to suppress sub-millimeter tracking jitter.
        """
        if raw_landmarks is None or np.isnan(raw_landmarks).any():
            return self._smoothed_landmarks if self._smoothed_landmarks is not None else np.zeros((21, 3), dtype=np.float32)

        if self._smoothed_landmarks is None:
            self._smoothed_landmarks = raw_landmarks.copy().astype(np.float32)
        else:
            # Check for erratic spatial jump (> 0.40 screen units in single frame)
            diff = np.abs(raw_landmarks - self._smoothed_landmarks)
            if np.max(diff) > 0.45:
                # Sudden camera occlusion or relocation — snap to new location
                self._smoothed_landmarks = raw_landmarks.copy().astype(np.float32)
            else:
                self._smoothed_landmarks = (
                    self.alpha * raw_landmarks + (1.0 - self.alpha) * self._smoothed_landmarks
                ).astype(np.float32)

        return self._smoothed_landmarks

    def update(
        self,
        finger_states: Optional[HandFingerStates],
        detection_confidence: float,
        timestamp: float,
    ) -> Tuple[Optional[HandFingerStates], Dict[str, Any]]:
        """
        Buffers and stabilizes finger states across the observation window.
        Returns the stabilized HandFingerStates and a clean dictionary summary.
        """
        self._confidence_history.append(detection_confidence)
        mean_conf = float(np.mean(self._confidence_history)) if self._confidence_history else detection_confidence

        if finger_states is None:
            return None, {
                "thumb": "unknown",
                "index": "unknown",
                "middle": "unknown",
                "ring": "unknown",
                "little": "unknown",
                "confidence": 0.0,
                "is_stable": False,
            }

        digits = [
            ("Thumb", finger_states.thumb),
            ("Index", finger_states.index),
            ("Middle", finger_states.middle),
            ("Ring", finger_states.ring),
            ("Little", finger_states.little),
        ]

        stabilized_details: Dict[str, FingerStateDetail] = {}
        stabilized_summary: Dict[str, str] = {}
        all_stable = True

        for name, detail in digits:
            if detail is not None:
                self._state_history[name].append(detail.state)

            history = self._state_history[name]
            if len(history) == 0:
                stab_state = detail.state if detail else FingerStateEnum.UNKNOWN
            else:
                # Temporal majority voting
                counts = Counter(history)
                most_common_state, count = counts.most_common(1)[0]
                ratio = count / len(history)
                if ratio >= self.config.finger.temporal_majority_threshold:
                    stab_state = most_common_state
                else:
                    # In transition / unstable: retain last verified state if available
                    stab_state = history[-1]
                    all_stable = False

            stabilized_summary[name.lower()] = stab_state.value
            stabilized_summary[name.capitalize()] = stab_state.value
            if detail is not None:
                stabilized_details[name.lower()] = FingerStateDetail(
                    finger=detail.finger,
                    state=stab_state,
                    confidence=min(detail.confidence, mean_conf),
                    extension_ratio=detail.extension_ratio,
                    pip_flexion_deg=detail.pip_flexion_deg,
                    dip_flexion_deg=detail.dip_flexion_deg,
                    mcp_flexion_deg=detail.mcp_flexion_deg,
                    contact_target=detail.contact_target,
                    intention=detail.intention,
                    is_focal=detail.is_focal,
                    focus_weight=detail.focus_weight,
                    focal_role=detail.focal_role,
                    diagnostics=detail.diagnostics,
                )

        stabilized_hand = HandFingerStates(
            thumb=stabilized_details.get("thumb", finger_states.thumb),
            index=stabilized_details.get("index", finger_states.index),
            middle=stabilized_details.get("middle", finger_states.middle),
            ring=stabilized_details.get("ring", finger_states.ring),
            little=stabilized_details.get("little", finger_states.little),
            timestamp=timestamp,
            palm_facing=getattr(finger_states, "palm_facing", "PALM"),
        )

        self._last_stabilized_dict = stabilized_summary
        self._last_confidence = mean_conf

        summary_out = dict(stabilized_summary)
        summary_out["confidence"] = round(mean_conf, 2)
        summary_out["is_stable"] = all_stable and len(self._confidence_history) >= self.window_size

        return stabilized_hand, summary_out


class PoseConfirmationManager:
    """
    Confirms Level 1 Static Hand Poses over 3–5 frames.
    Implements Hysteresis:
      - Entering a pose requires confidence >= enter_threshold and consecutive agreement.
      - Dropping / exiting an active pose requires confidence < exit_threshold.
    Suppresses rapid pose oscillations.
    """

    def __init__(self, config: Optional[TemporalIntentConfig] = None):
        self.config = config or TemporalIntentConfig()
        self.window_size = self.config.windows.pose_window
        self.enter_thresh = self.config.pose.enter_threshold
        self.exit_thresh = self.config.pose.exit_threshold
        self.min_confirm_frames = self.config.pose.min_confirm_frames

        self._pose_history: Deque[Tuple[HandPoseId, float]] = deque(maxlen=self.window_size)
        self._active_confirmed_pose: HandPoseId = HandPoseId.UNKNOWN
        self._active_pose_name: str = "NONE"
        self._candidate_pose: HandPoseId = HandPoseId.UNKNOWN
        self._consecutive_candidate_frames: int = 0
        self._confirmed_frames: int = 0
        self._recent_pose_switches: Deque[float] = deque(maxlen=self.config.pose.max_oscillation_history)

    def reset(self):
        self._pose_history.clear()
        self._active_confirmed_pose = HandPoseId.UNKNOWN
        self._active_pose_name = "NONE"
        self._candidate_pose = HandPoseId.UNKNOWN
        self._consecutive_candidate_frames = 0
        self._confirmed_frames = 0
        self._recent_pose_switches.clear()

    def update(
        self,
        instantaneous_pose: Optional[DerivedHandPose],
        timestamp: float,
    ) -> Tuple[HandPoseId, str, float, bool, Dict[str, Any]]:
        """
        Evaluates temporal consistency and hysteresis for the static hand pose.
        Returns:
            (confirmed_pose_id, confirmed_canonical_name, confidence, is_confirmed, telemetry)
        """
        if instantaneous_pose is None or instantaneous_pose.pose_id == HandPoseId.UNKNOWN:
            self._consecutive_candidate_frames = 0
            if self._active_confirmed_pose != HandPoseId.UNKNOWN:
                # Active pose is exiting
                self._confirmed_frames = 0
                self._active_confirmed_pose = HandPoseId.UNKNOWN
                self._active_pose_name = "NONE"
            return (HandPoseId.UNKNOWN, "NONE", 0.0, False, {
                "active_pose": "NONE",
                "candidate_pose": "NONE",
                "frames_consistent": 0,
                "is_confirmed": False,
                "oscillation_rate": 0.0,
            })

        inst_id = instantaneous_pose.pose_id
        inst_name = instantaneous_pose.canonical_name
        inst_conf = instantaneous_pose.confidence

        self._pose_history.append((inst_id, inst_conf))

        # Check for oscillation (frequent switching in short time window)
        oscillation_rate = len(self._recent_pose_switches) / max(1, self.config.pose.max_oscillation_history)

        # ── 1. Candidate Tracking ─────────────────────────────────────────
        if inst_id == self._candidate_pose:
            self._consecutive_candidate_frames += 1
        else:
            if self._candidate_pose != HandPoseId.UNKNOWN:
                self._recent_pose_switches.append(timestamp)
            self._candidate_pose = inst_id
            self._consecutive_candidate_frames = 1

        # ── 2. Hysteresis Evaluation ──────────────────────────────────────
        # Entering / Promotion condition:
        if (
            inst_conf >= self.enter_thresh
            and self._consecutive_candidate_frames >= self.min_confirm_frames
        ):
            if self._active_confirmed_pose != inst_id:
                self._active_confirmed_pose = inst_id
                self._active_pose_name = inst_name
                self._confirmed_frames = 1
            else:
                self._confirmed_frames += 1

        elif self._active_confirmed_pose != HandPoseId.UNKNOWN:
            # Currently in an active confirmed pose.
            # Only drop if confidence drops below exit_thresh OR a new candidate has surpassed enter_thresh
            if inst_id == self._active_confirmed_pose:
                if inst_conf < self.exit_thresh:
                    # Dropped below lower hysteresis floor
                    self._active_confirmed_pose = HandPoseId.UNKNOWN
                    self._active_pose_name = "NONE"
                    self._confirmed_frames = 0
                else:
                    self._confirmed_frames += 1
            elif (
                self._consecutive_candidate_frames >= self.min_confirm_frames
                and inst_conf >= self.enter_thresh
            ):
                # Valid transition to new confirmed pose
                self._active_confirmed_pose = inst_id
                self._active_pose_name = inst_name
                self._confirmed_frames = 1

        is_confirmed = (
            self._active_confirmed_pose != HandPoseId.UNKNOWN
            and self._confirmed_frames >= 1
        )

        telemetry = {
            "active_pose": self._active_pose_name,
            "candidate_pose": inst_name,
            "frames_consistent": self._consecutive_candidate_frames,
            "confirmed_frames": self._confirmed_frames,
            "is_confirmed": is_confirmed,
            "confidence": inst_conf,
            "oscillation_rate": round(oscillation_rate, 2),
        }

        out_id = self._active_confirmed_pose if is_confirmed else HandPoseId.UNKNOWN
        out_name = self._active_pose_name if is_confirmed else "NONE"
        out_conf = inst_conf if is_confirmed else inst_conf * 0.5

        return out_id, out_name, out_conf, is_confirmed, telemetry


class MotionPrimitiveAnalyzer:
    """
    Measures Level 2 pure kinematics over 5–10 frames.
    Calculates velocity, acceleration, displacement, trajectory curvature,
    directional consistency, and duration completely independently of hand pose.
    Rejects slow drift, curved paths, and accidental repositioning.
    """

    def __init__(self, config: Optional[TemporalIntentConfig] = None):
        self.config = config or TemporalIntentConfig()
        self.window_size = self.config.windows.motion_window

        # Trajectory samples: [(x, y, z, timestamp)]
        self._trajectory: Deque[Tuple[float, float, float, float]] = deque(maxlen=self.window_size)
        self._velocities: Deque[Tuple[float, float, float]] = deque(maxlen=self.window_size)
        self._speeds: Deque[float] = deque(maxlen=self.window_size)
        self._stroke_start_time: Optional[float] = None
        self._stroke_start_pos: Optional[Tuple[float, float, float]] = None

    def reset(self):
        self._trajectory.clear()
        self._velocities.clear()
        self._speeds.clear()
        self._stroke_start_time = None
        self._stroke_start_pos = None

    def update(
        self,
        palm_center: Tuple[float, float, float],
        d_ref: float,
        timestamp: float,
    ) -> Tuple[MotionPrimitive, Dict[str, Any]]:
        """
        Analyzes the motion trajectory over the observation window.
        Returns:
            (motion_primitive, kinematic_metrics)
        """
        x, y, z = palm_center
        self._trajectory.append((x, y, z, timestamp))

        if len(self._trajectory) < 2:
            return MotionPrimitive.STATIONARY, {
                "velocity": (0.0, 0.0, 0.0),
                "speed": 0.0,
                "acceleration": (0.0, 0.0, 0.0),
                "tangential_acceleration": 0.0,
                "net_displacement": 0.0,
                "direction_consistency": 1.0,
                "linearity": 1.0,
                "duration_ms": 0.0,
                "primary_direction": "STATIONARY",
                "is_intentional": False,
            }

        # 1. Instantaneous velocity computation across consecutive window samples
        prev_x, prev_y, prev_z, prev_t = self._trajectory[-2]
        dt = max(timestamp - prev_t, 0.001)
        vx = (x - prev_x) / dt
        vy = (y - prev_y) / dt
        vz = (z - prev_z) / dt
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)

        self._velocities.append((vx, vy, vz))
        self._speeds.append(speed)

        # 2. Window smoothed velocity & acceleration
        mean_vx = float(np.mean([v[0] for v in self._velocities]))
        mean_vy = float(np.mean([v[1] for v in self._velocities]))
        mean_vz = float(np.mean([v[2] for v in self._velocities]))
        mean_speed = math.sqrt(mean_vx * mean_vx + mean_vy * mean_vy + mean_vz * mean_vz)

        # Tangential acceleration
        tangential_accel = 0.0
        if len(self._speeds) >= 3:
            recent_speed_diff = self._speeds[-1] - self._speeds[-3]
            recent_dt = max(self._trajectory[-1][3] - self._trajectory[-3][3], 0.001)
            tangential_accel = recent_speed_diff / recent_dt

        # 3. Stroke Displacement & Path Linearity
        start_x, start_y, start_z, start_t = self._trajectory[0]
        dx = x - start_x
        dy = y - start_y
        dz = z - start_z
        net_disp = math.sqrt(dx * dx + dy * dy + dz * dz)

        path_len = 0.0
        for i in range(1, len(self._trajectory)):
            p1 = self._trajectory[i - 1]
            p2 = self._trajectory[i]
            path_len += math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)

        linearity = net_disp / max(path_len, 0.0001)

        # 4. Directional Consistency (Average alignment between successive displacement vectors)
        cos_similarities: List[float] = []
        for i in range(2, len(self._trajectory)):
            v1 = np.array([self._trajectory[i-1][0] - self._trajectory[i-2][0],
                           self._trajectory[i-1][1] - self._trajectory[i-2][1],
                           self._trajectory[i-1][2] - self._trajectory[i-2][2]])
            v2 = np.array([self._trajectory[i][0] - self._trajectory[i-1][0],
                           self._trajectory[i][1] - self._trajectory[i-1][1],
                           self._trajectory[i][2] - self._trajectory[i-1][2]])
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 0.002 and n2 > 0.002:
                cos_sim = float(np.dot(v1, v2) / (n1 * n2))
                cos_similarities.append(max(-1.0, min(1.0, cos_sim)))

        dir_consistency = float(np.mean(cos_similarities)) if cos_similarities else 1.0

        # Stroke duration
        if mean_speed >= self.config.motion.min_velocity_threshold:
            if self._stroke_start_time is None:
                self._stroke_start_time = timestamp
                self._stroke_start_pos = (x, y, z)
            stroke_duration_ms = (timestamp - self._stroke_start_time) * 1000.0
        else:
            self._stroke_start_time = None
            self._stroke_start_pos = None
            stroke_duration_ms = 0.0

        # 5. Motion Primitive Classification
        cfg_m = self.config.motion
        is_active = mean_speed >= cfg_m.min_velocity_threshold
        is_linear = linearity >= cfg_m.curvature_reject_threshold
        is_consistent = dir_consistency >= cfg_m.direction_consistency_threshold
        is_intentional = is_active and is_linear and is_consistent and (net_disp >= cfg_m.min_displacement)

        primitive = MotionPrimitive.STATIONARY
        primary_dir = "STATIONARY"

        if is_intentional:
            abs_dx = abs(dx)
            abs_dy = abs(dy)
            abs_dz = abs(dz)

            if abs_dx >= 1.3 * abs_dy and abs_dx >= 1.3 * abs_dz:
                if dx < 0:
                    primitive = MotionPrimitive.MOVE_LEFT
                    primary_dir = "LEFT"
                else:
                    primitive = MotionPrimitive.MOVE_RIGHT
                    primary_dir = "RIGHT"
            elif abs_dy >= 1.3 * abs_dx and abs_dy >= 1.3 * abs_dz:
                # MediaPipe Y: 0 is top, 1 is bottom
                if dy < 0:
                    primitive = MotionPrimitive.MOVE_UP
                    primary_dir = "UP"
                else:
                    primitive = MotionPrimitive.MOVE_DOWN
                    primary_dir = "DOWN"
            elif abs_dz >= 1.3 * abs_dx and abs_dz >= 1.3 * abs_dy:
                if dz < 0:
                    primitive = MotionPrimitive.MOVE_TOWARD
                    primary_dir = "FORWARD"
                else:
                    primitive = MotionPrimitive.MOVE_AWAY
                    primary_dir = "BACKWARD"
            else:
                primitive = MotionPrimitive.MOVE_LEFT if dx < 0 else MotionPrimitive.MOVE_RIGHT
                primary_dir = "DIAGONAL"
        elif mean_speed <= cfg_m.stationary_velocity_threshold:
            primitive = MotionPrimitive.HOLD if len(self._speeds) >= self.window_size else MotionPrimitive.STATIONARY
            primary_dir = "STATIONARY"

        metrics = {
            "velocity": (round(mean_vx, 3), round(mean_vy, 3), round(mean_vz, 3)),
            "speed": round(mean_speed, 3),
            "acceleration": (0.0, 0.0, 0.0),
            "tangential_acceleration": round(tangential_accel, 2),
            "net_displacement": round(net_disp, 4),
            "direction_consistency": round(dir_consistency, 2),
            "linearity": round(linearity, 2),
            "duration_ms": round(stroke_duration_ms, 1),
            "primary_direction": primary_dir,
            "is_intentional": is_intentional,
        }

        return primitive, metrics


class GestureCandidateBuffer:
    """
    Validates Level 3 complete gesture candidates over 8–15 frames.
    Ensures gestures accumulate continuous empirical evidence before execution.
    """

    def __init__(self, config: Optional[TemporalIntentConfig] = None):
        self.config = config or TemporalIntentConfig()
        self.window_size = self.config.windows.gesture_window

        self._candidate_history: Deque[Tuple[str, float, float]] = deque(maxlen=self.window_size)
        self._evidence_score: float = 0.0

    def reset(self):
        self._candidate_history.clear()
        self._evidence_score = 0.0

    @staticmethod
    def _normalize_candidate(gname: str) -> str:
        if not gname:
            return "NONE"
        u = str(gname).upper()
        if u in ("NONE", "IDLE", "UNKNOWN"):
            return "NONE"
        if "SWIPE_LEFT" in u:
            return "SWIPE_LEFT"
        if "SWIPE_RIGHT" in u:
            return "SWIPE_RIGHT"
        if "SWIPE_UP" in u:
            return "SWIPE_UP"
        if "SWIPE_DOWN" in u:
            return "SWIPE_DOWN"
        if "POINT" in u or u in ("THREE_FINGER", "FOUR_FINGER", "FIVE_FINGER", "DOUBLE_POINT"):
            return "POINT"
        if "PINCH" in u:
            return "PINCH"
        if "GRAB" in u or "FIST" in u or "CUPPED" in u:
            return "GRAB"
        if "OPEN_PALM" in u or "SPREAD" in u:
            return "OPEN_PALM"
        if "THUMBS_UP" in u:
            return "THUMBS_UP"
        if "THUMBS_DOWN" in u:
            return "THUMBS_DOWN"
        if "PEACE" in u:
            return "PEACE"
        if "OK_RING" in u:
            return "OK_RING"
        if "SHAKA" in u:
            return "SHAKA"
        if "GUN" in u:
            return "GUN"
        return u

    def update(
        self,
        candidate_gesture: str,
        instantaneous_confidence: float,
        timestamp: float,
    ) -> Tuple[float, int, bool]:
        """
        Updates evidence accumulator for the candidate gesture.
        Returns:
            (evidence_score, consecutive_frames, meets_threshold)
        """
        normalized_cand = self._normalize_candidate(candidate_gesture)
        if normalized_cand in ("NONE", "IDLE"):
            # Rapid decay when no gesture is detected
            self._evidence_score *= 0.60
            self._candidate_history.clear()
            return self._evidence_score, 0, False

        self._candidate_history.append((normalized_cand, instantaneous_confidence, timestamp))

        # Check homogeneity in buffer
        matching_count = sum(1 for g, _, _ in self._candidate_history if g == normalized_cand)
        consistency_ratio = matching_count / len(self._candidate_history)

        # Exponential evidence accumulation
        lam = self.config.intent_fsm.evidence_lambda
        self._evidence_score = (lam * self._evidence_score) + ((1.0 - lam) * instantaneous_confidence * consistency_ratio)
        self._evidence_score = min(1.0, max(0.0, self._evidence_score))

        meets_threshold = (
            self._evidence_score >= self.config.intent_fsm.activation_threshold
            and matching_count >= self.config.intent_fsm.candidate_frames
        )

        return self._evidence_score, matching_count, meets_threshold


class ObservationWindowManager:
    """
    Unified manager encapsulating all 5 temporal observation windows:
      - Level 0 Finger States (1–3 frames)
      - Level 1 Hand Poses (3–5 frames)
      - Level 2 Motion Primitives (5–10 frames)
      - Level 3 Gestures (8–15 frames)
      - Level 4 Two-Hand Gestures (10–20 frames)
    """

    def __init__(self, config: Optional[TemporalIntentConfig] = None):
        self.config = config or TemporalIntentConfig()
        self.finger_stabilizer = FingerStateStabilizer(self.config)
        self.pose_manager = PoseConfirmationManager(self.config)
        self.motion_analyzer = MotionPrimitiveAnalyzer(self.config)
        self.gesture_buffer = GestureCandidateBuffer(self.config)

    def reset(self):
        self.finger_stabilizer.reset()
        self.pose_manager.reset()
        self.motion_analyzer.reset()
        self.gesture_buffer.reset()
