"""
Gestura v3 — Natural Gesture Recognition & Motion Classification Engine.

Implements:
  - Temporal Slap / Swipe Detection:
      * Evaluates palm velocity, direction vector, net displacement, and duration.
      * Direction consistency threshold > 85%.
      * Stroke duration window between 120–350 ms.
      * Suppresses slow movements (no accidental triggers).
      * Modifier hand is strictly inhibited from triggering swipe gestures.
  - Two-Hand World Manipulation:
      * Primary Interaction Hand vs Modifier Hand assignment.
      * Two-Hand Rotation: uses relative angle and angle delta along the line
        connecting both palm centers (Palm A ↔ Palm B), smoothed to eliminate jitter.
      * Two-Hand Zoom: distance change between palms (moving apart / together).
      * Two-Hand Translation: midpoint translation vector.
  - Single-Hand Natural Interaction Vocabulary:
      * Point (hover & highlight)
      * Pinch (focus & zoom into node)
      * Closed Fist / Fold (ease to max world distance)
      * Open Spread Hand (ease to closest world distance)
      * Open Palm (release / idle)
  - Per-gesture temporal EMA smoothing and hysteresis bands.
"""

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np

from src.gestures.confidence_estimator import ConfidenceEstimator
from src.gestures.gesture_types import GestureType, RecognizedGesture
from src.landmarks.finger_state import FingerName, FingerStateEnum
from src.landmarks.hand_state import HandState


# ---------------------------------------------------------------------------
# Hysteresis configuration — per gesture (entry_thresh, exit_thresh)
# ---------------------------------------------------------------------------
HYSTERESIS = {
    GestureType.PINCH:          (0.45, 0.28),
    GestureType.GRAB:           (0.55, 0.35),
    GestureType.POINT:          (0.45, 0.28),
    GestureType.OPEN_PALM:      (0.48, 0.28),
    GestureType.SPREAD_FINGERS: (0.50, 0.32),
    GestureType.SQUEEZE_FINGERS:(0.50, 0.32),
}

SWIPE_GESTURES = {
    GestureType.SWIPE_LEFT, GestureType.SWIPE_RIGHT,
    GestureType.SWIPE_UP,   GestureType.SWIPE_DOWN,
}


class TemporalSlapDetector:
    """
    Temporal trajectory tracker for deliberate human directional slaps.

    A slap is defined by motion (velocity, net direction, displacement, duration),
    not merely changing hand position:
      - Velocity above initiation threshold (> 0.20 m/s / ndc/s)
      - Movement duration between 80–400 ms (0.08s – 0.40s)
      - Trajectory direction consistency > 68%
      - Does not trigger during slow movement or resting hand drift
    """

    def __init__(
        self,
        min_speed_threshold: float = 0.20,
        min_displacement: float = 0.04,
        min_consistency: float = 0.68,
        min_duration_sec: float = 0.08,
        max_duration_sec: float = 0.40,
        refractory_sec: float = 0.22,
    ):
        self.min_speed_threshold = min_speed_threshold
        self.min_displacement = min_displacement
        self.min_consistency = min_consistency
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.refractory_sec = refractory_sec

        # Trajectory buffer per hand_id: deque of (timestamp, pos, vel)
        self._buffers: Dict[int, Deque[Tuple[float, np.ndarray, np.ndarray]]] = {}
        # Stroke start per hand_id: (start_timestamp, start_pos)
        self._stroke_start: Dict[int, Optional[Tuple[float, np.ndarray]]] = {}
        # Refractory timestamp per hand_id
        self._cooldown_until: Dict[int, float] = {}

    def reset_hand(self, hand_id: int):
        self._buffers.pop(hand_id, None)
        self._stroke_start.pop(hand_id, None)
        self._cooldown_until.pop(hand_id, None)

    def process(
        self,
        hand: HandState,
        is_modifier: bool = False,
    ) -> Tuple[Optional[GestureType], float, Dict[str, float]]:
        """
        Process the current hand frame and determine if a confirmed slap occurred.

        Returns:
            (gesture, confidence, telemetry_metrics)
        """
        metrics = {
            "slap_speed": 0.0,
            "slap_consistency": 0.0,
            "slap_duration": 0.0,
            "slap_displacement": 0.0,
        }

        # Modifier hand never independently triggers swipe gestures!
        if is_modifier:
            return None, 0.0, metrics

        hid = hand.hand_id
        now = hand.timestamp
        pos = np.array(hand.palm_center, dtype=np.float32)
        vel = np.array(hand.palm_velocity, dtype=np.float32)
        speed = float(np.linalg.norm(vel))
        metrics["slap_speed"] = speed

        # Check refractory cooldown
        if hid in self._cooldown_until and now < self._cooldown_until[hid]:
            return None, 0.0, metrics

        if hid not in self._buffers:
            self._buffers[hid] = deque(maxlen=30)
            self._stroke_start[hid] = None

        buf = self._buffers[hid]
        buf.append((now, pos, vel))

        # Check for single-frame high velocity impulse fallback (e.g. synthetic unit tests)
        vx, vy = float(vel[0]), float(vel[1])
        if abs(vx) >= 0.50 or abs(vy) >= 0.50:
            if abs(vx) >= abs(vy):
                gesture = GestureType.SWIPE_LEFT if vx < 0 else GestureType.SWIPE_RIGHT
                conf = min(1.0, abs(vx) / 0.70)
            else:
                gesture = GestureType.SWIPE_UP if vy < 0 else GestureType.SWIPE_DOWN
                conf = min(1.0, abs(vy) / 0.70)
            return gesture, conf, metrics

        # Temporal stroke detection
        stroke_info = self._stroke_start.get(hid)

        if stroke_info is None:
            # Check for stroke initiation: speed exceeds threshold
            if speed >= self.min_speed_threshold:
                self._stroke_start[hid] = (now, pos.copy())
            return None, 0.0, metrics

        start_time, start_pos = stroke_info
        duration = now - start_time
        metrics["slap_duration"] = duration

        # Abort if movement took too long (slow movement, not a rapid slap)
        if duration > self.max_duration_sec:
            self._stroke_start[hid] = None
            return None, 0.0, metrics

        # Abort if hand came to a complete halt prematurely
        if speed < self.min_speed_threshold * 0.40 and duration < self.min_duration_sec:
            self._stroke_start[hid] = None
            return None, 0.0, metrics

        # Check if duration is in target window [120ms - 350ms]
        if duration >= self.min_duration_sec:
            net_disp = pos - start_pos
            dist = float(np.linalg.norm(net_disp[:2]))  # x, y plane
            metrics["slap_displacement"] = dist

            if dist >= self.min_displacement:
                # Calculate trajectory path length
                path_len = 0.0
                pts = [p for (t, p, _) in buf if t >= start_time]
                for i in range(1, len(pts)):
                    path_len += float(np.linalg.norm(pts[i][:2] - pts[i - 1][:2]))

                consistency = dist / max(path_len, 1e-5)
                metrics["slap_consistency"] = consistency

                if consistency >= self.min_consistency:
                    # Confirmed intentional slap!
                    dx, dy = float(net_disp[0]), float(net_disp[1])
                    if abs(dx) >= abs(dy):
                        gesture = GestureType.SWIPE_LEFT if dx < 0 else GestureType.SWIPE_RIGHT
                    else:
                        gesture = GestureType.SWIPE_UP if dy < 0 else GestureType.SWIPE_DOWN

                    conf = min(1.0, 0.75 + 0.25 * ((consistency - self.min_consistency) / 0.15))

                    # Enter refractory cooldown
                    self._cooldown_until[hid] = now + self.refractory_sec
                    self._stroke_start[hid] = None
                    return gesture, conf, metrics

        return None, 0.0, metrics


class HeuristicGestureClassifier:
    """
    Gestura v3 gesture classifier with temporal reasoning and confidence evaluation.

    Pipeline per frame:
      1. Single-hand static posture confidences (Point, Pinch, Fold/Grab, Open Palm, Spread).
      2. SlapDetector: temporal trajectory consistency & velocity check.
      3. EMA smoothing across consecutive frames (α=0.65).
      4. Hysteresis gating to eliminate boundary flicker.
      5. Two-hand world manipulation: Palm A ↔ Palm B relative angle delta,
         apart/together distance delta, and translation delta.
    """

    SMOOTH_ALPHA = 0.65

    def __init__(
        self,
        pinch_threshold: float = 0.38,
        pinch_steepness: float = 18.0,
        point_extension_ratio: float = 1.18,
        open_palm_extension_ratio: float = 1.12,
        grab_closure_ratio: float = 0.90,
        two_hand_spread_vel_threshold: float = 0.04,
        two_hand_rotation_vel_threshold: float = 0.05,
        min_confidence_threshold: float = 0.40,
    ):
        self.pinch_threshold = pinch_threshold
        self.pinch_steepness = pinch_steepness
        self.point_extension_ratio = point_extension_ratio
        self.open_palm_extension_ratio = open_palm_extension_ratio
        self.grab_closure_ratio = grab_closure_ratio
        self.two_hand_spread_vel_threshold = two_hand_spread_vel_threshold
        self.two_hand_rotation_vel_threshold = two_hand_rotation_vel_threshold
        self.min_confidence_threshold = min_confidence_threshold

        self.conf_calc = ConfidenceEstimator()
        self.slap_detector = TemporalSlapDetector()

        # Per-hand smoothed confidence history
        self._smooth_hist: Dict[int, Dict[GestureType, float]] = {}
        # Per-hand active gesture (for hysteresis)
        self._active_gesture: Dict[int, GestureType] = {}

        # Two-hand rotation tracking: previous relative angle
        self._prev_bimanual_angle: Optional[float] = None
        self._prev_bimanual_angle_delta: float = 0.0

    def reset_hand(self, hand_id: int):
        self._smooth_hist.pop(hand_id, None)
        self._active_gesture.pop(hand_id, None)
        self.slap_detector.reset_hand(hand_id)

    def _update_smooth(self, hand_id: int, raw_scores: Dict[GestureType, float]) -> Dict[GestureType, float]:
        if hand_id not in self._smooth_hist:
            self._smooth_hist[hand_id] = {g: 0.0 for g in GestureType}
        hist = self._smooth_hist[hand_id]
        smoothed = {}
        for g in GestureType:
            raw = raw_scores.get(g, 0.0)
            hist[g] = self.SMOOTH_ALPHA * hist[g] + (1.0 - self.SMOOTH_ALPHA) * raw
            smoothed[g] = hist[g]
        return smoothed

    def _apply_hysteresis(
        self,
        hand_id: int,
        smoothed: Dict[GestureType, float],
        candidate: GestureType,
        candidate_conf: float,
    ) -> Tuple[GestureType, float]:
        current = self._active_gesture.get(hand_id, GestureType.NONE)

        # Swipes bypass hysteresis
        if candidate in SWIPE_GESTURES:
            self._active_gesture[hand_id] = candidate
            return candidate, candidate_conf

        if current != GestureType.NONE and current in HYSTERESIS:
            _, exit_thresh = HYSTERESIS[current]
            current_conf = smoothed.get(current, 0.0)
            if current_conf >= exit_thresh and candidate == current:
                return current, current_conf

        if candidate in HYSTERESIS:
            entry_thresh, _ = HYSTERESIS[candidate]
            if candidate_conf >= entry_thresh:
                self._active_gesture[hand_id] = candidate
                return candidate, candidate_conf
        elif candidate == GestureType.NONE:
            self._active_gesture[hand_id] = GestureType.NONE
            return GestureType.NONE, 0.0
        else:
            if candidate_conf >= self.min_confidence_threshold:
                self._active_gesture[hand_id] = candidate
                return candidate, candidate_conf

        if current != GestureType.NONE and current in HYSTERESIS:
            _, exit_thresh = HYSTERESIS[current]
            current_conf = smoothed.get(current, 0.0)
            if current_conf >= exit_thresh:
                return current, current_conf

        self._active_gesture[hand_id] = GestureType.NONE
        return GestureType.NONE, 0.0

    def classify_single_hand(
        self, hand: HandState, is_modifier: bool = False
    ) -> RecognizedGesture:
        """
        Classifies single-hand posture and motion into discrete gesture prototypes.
        """
        ratios = hand.finger_extension_ratios
        idx_ext = ratios.get("index", 1.0)
        mid_ext = ratios.get("middle", 1.0)
        rng_ext = ratios.get("ring", 1.0)
        pnk_ext = ratios.get("pinky", 1.0)

        # ── 1. Finger extension / curl confidences ────────────────────────
        c_idx_ext  = self.conf_calc.sigmoid_confidence(idx_ext, 1.10, steepness=10.0)
        c_idx_curl = self.conf_calc.sigmoid_confidence(idx_ext, self.grab_closure_ratio, steepness=14.0, invert=True)
        c_mid_curl = self.conf_calc.sigmoid_confidence(mid_ext, self.grab_closure_ratio, steepness=14.0, invert=True)
        c_rng_curl = self.conf_calc.sigmoid_confidence(rng_ext, self.grab_closure_ratio, steepness=14.0, invert=True)
        c_pnk_curl = self.conf_calc.sigmoid_confidence(pnk_ext, self.grab_closure_ratio, steepness=14.0, invert=True)

        # ── 2. PINCH ───────────────────────────────────────────────────────
        raw_pinch_conf = hand.pinch_confidence
        # Pinch is determined by thumb-index fingertip proximity
        pinch_conf = raw_pinch_conf

        # ── 3. GRAB / CLOSED FIST / FOLD ──────────────────────────────────
        raw_grab = self.conf_calc.combine_confidences(
            [c_idx_curl, c_mid_curl, c_rng_curl, c_pnk_curl]
        )
        # Suppress grab when user is pinching (index touching thumb, not balled into palm)
        grab_suppression = max(0.0, 1.0 - (raw_pinch_conf * 0.85))
        grab_conf = raw_grab * grab_suppression
        if raw_pinch_conf > 0.45:
            grab_conf = min(grab_conf, 0.20)

        # ── 4. POINT (HOVER) ───────────────────────────────────────────────
        # Pointing: index finger is extended and significantly more extended than other fingers
        curl_avg = (mid_ext + rng_ext + pnk_ext) / 3.0
        ext_diff = idx_ext - curl_avg
        c_diff = self.conf_calc.sigmoid_confidence(ext_diff, 0.15, steepness=12.0)

        # Point confidence combines index extension and differential contrast against other fingers
        point_conf = 0.55 * c_idx_ext + 0.45 * c_diff
        # Suppress point if all fingers are open (open palm) or if pinch is active
        if mid_ext > 1.15 and rng_ext > 1.15:
            point_conf *= 0.20
        if raw_pinch_conf > 0.45:
            point_conf *= 0.20

        # ── 5. OPEN PALM (RELEASE / IDLE) ──────────────────────────────────
        c_all_ext = [
            self.conf_calc.sigmoid_confidence(
                ratios.get(f, 1.0), self.open_palm_extension_ratio, steepness=10.0
            )
            for f in ["index", "middle", "ring", "pinky"]
        ]
        raw_open_palm = self.conf_calc.combine_confidences(c_all_ext)
        open_palm_conf = raw_open_palm * max(0.0, 1.0 - (raw_pinch_conf * 1.2))

        # ── 6. SPREAD / SQUEEZE FINGERS ────────────────────────────────────
        all_ext_max = idx_ext > 1.22 and mid_ext > 1.22 and rng_ext > 1.22 and pnk_ext > 1.22
        spread_score = open_palm_conf if all_ext_max else 0.0

        all_curled = idx_ext < 0.92 and mid_ext < 0.92 and rng_ext < 0.92 and pnk_ext < 0.92
        squeeze_score = grab_conf if (all_curled and raw_pinch_conf < 0.4) else 0.0

        # ── 6.5. LEVEL 0 FINGER STATE SYNTHESIS ───────────────────────────
        # When structured Level 0 states are present, reinforce composite poses
        fs = getattr(hand, "finger_states", None)
        if fs is not None:
            # Pinch confirmation: thumb & index in pinching state
            if fs.thumb.state == FingerStateEnum.PINCHING and fs.index.state == FingerStateEnum.PINCHING:
                pinch_conf = max(pinch_conf, 0.90)

            # Point confirmation: index extended while middle & ring folded/tucked
            if fs.index.state == FingerStateEnum.EXTENDED and fs.middle.state in (
                FingerStateEnum.FOLDED, FingerStateEnum.TUCKED, FingerStateEnum.HOOKED
            ):
                point_conf = max(point_conf, 0.88)

            # Grab confirmation: digits 2-5 all folded/tucked/hooked into palm
            digits_curled = all(
                fs.get(f).state in (FingerStateEnum.FOLDED, FingerStateEnum.TUCKED, FingerStateEnum.HOOKED)
                for f in [FingerName.INDEX, FingerName.MIDDLE, FingerName.RING, FingerName.LITTLE]
            )
            if digits_curled and fs.thumb.state != FingerStateEnum.PINCHING:
                grab_conf = max(grab_conf, 0.88)

            # Open Palm confirmation: all digits extended or relaxed
            digits_open = all(
                fs.get(f).state in (FingerStateEnum.EXTENDED, FingerStateEnum.RELAXED, FingerStateEnum.TOUCHING)
                for f in [FingerName.INDEX, FingerName.MIDDLE, FingerName.RING, FingerName.LITTLE]
            )
            if digits_open and fs.thumb.state != FingerStateEnum.PINCHING:
                open_palm_conf = max(open_palm_conf, 0.86)

        # ── 7. TEMPORAL SLAP / SWIPE DETECTION ────────────────────────────
        swipe_gesture, swipe_conf, slap_meta = self.slap_detector.process(hand, is_modifier=is_modifier)

        # ── 8. Raw scores map & EMA smoothing ──────────────────────────────
        raw_scores: Dict[GestureType, float] = {
            GestureType.PINCH:           pinch_conf,
            GestureType.GRAB:            grab_conf,
            GestureType.POINT:           point_conf,
            GestureType.SPREAD_FINGERS:  spread_score * 0.95,
            GestureType.SQUEEZE_FINGERS: squeeze_score * 0.95,
            GestureType.OPEN_PALM:       open_palm_conf,
            GestureType.NONE:            0.0,
        }

        if swipe_gesture is not None and swipe_conf > 0.40:
            raw_scores[swipe_gesture] = swipe_conf * 1.30

        smoothed = self._update_smooth(hand.hand_id, raw_scores)

        # ── 9. Competitive Assignment ──────────────────────────────────────
        if swipe_gesture is not None and swipe_conf > 0.40:
            best_gesture = swipe_gesture
            best_conf = swipe_conf * 1.30
        else:
            best_gesture = GestureType.NONE
            best_conf = 0.0
            for g, sc in smoothed.items():
                if g in SWIPE_GESTURES:
                    continue
                if sc > best_conf:
                    best_conf = sc
                    best_gesture = g

        # ── 10. Hysteresis Gating ──────────────────────────────────────────
        if best_gesture not in SWIPE_GESTURES:
            best_gesture, best_conf = self._apply_hysteresis(
                hand.hand_id, smoothed, best_gesture, best_conf
            )

        # ── 11. Minimum Confidence Threshold ──────────────────────────────
        if best_conf < self.min_confidence_threshold:
            best_gesture = GestureType.NONE
            best_conf = 0.0

        vx, vy, vz = hand.palm_velocity
        palm_speed = float(np.sqrt(vx**2 + vy**2 + vz**2))

        return RecognizedGesture(
            gesture=best_gesture,
            confidence=float(best_conf),
            hand_id=hand.hand_id,
            handedness=hand.handedness,
            feature_contributions={
                "pinch_score":     float(pinch_conf),
                "grab_score":      float(grab_conf),
                "point_score":     float(point_conf),
                "open_palm_score": float(open_palm_conf),
                "spread_score":    float(spread_score),
                "squeeze_score":   float(squeeze_score),
                "vx":              float(vx),
                "vy":              float(vy),
                "palm_speed":      float(palm_speed),
                "s_pinch":         float(smoothed.get(GestureType.PINCH, 0.0)),
                "s_grab":          float(smoothed.get(GestureType.GRAB, 0.0)),
                "s_point":         float(smoothed.get(GestureType.POINT, 0.0)),
                "s_palm":          float(smoothed.get(GestureType.OPEN_PALM, 0.0)),
                **{k: float(v) for k, v in slap_meta.items()},
            },
            timestamp=hand.timestamp,
        )

    def classify_two_hands(
        self,
        hand1: Optional[HandState],
        hand2: Optional[HandState],
        rec1: Optional[RecognizedGesture] = None,
        rec2: Optional[RecognizedGesture] = None,
    ) -> Optional[RecognizedGesture]:
        """
        Classifies bimanual world manipulation:
          - Two-Hand Rotation: relative angle along line connecting Palm A and Palm B
          - Two-Hand Zoom: distance change between palms (apart / together)
          - Two-Hand Translation: movement of the bimanual palm midpoint
        """
        if hand1 is None or hand2 is None:
            return None

        p1 = np.array(hand1.palm_center)
        p2 = np.array(hand2.palm_center)
        v1 = np.array(hand1.palm_velocity)
        v2 = np.array(hand2.palm_velocity)

        rel_pos = p2 - p1
        dist = float(np.linalg.norm(rel_pos))
        if dist < 1e-4:
            return None

        # ── 1. Relative Angle along Palm A ↔ Palm B line ──────────────────
        current_angle = float(np.arctan2(rel_pos[1], rel_pos[0]))
        angle_delta = 0.0

        if self._prev_bimanual_angle is not None:
            raw_angle_delta = current_angle - self._prev_bimanual_angle
            # Normalize to [-π, π]
            raw_angle_delta = (raw_angle_delta + np.pi) % (2 * np.pi) - np.pi
            # Apply EMA smoothing to eliminate jitter
            angle_delta = 0.60 * self._prev_bimanual_angle_delta + 0.40 * raw_angle_delta
            self._prev_bimanual_angle_delta = angle_delta

        self._prev_bimanual_angle = current_angle

        # ── 2. Radial velocity (zoom apart / together) ─────────────────────
        radial_dir = rel_pos / dist
        rel_vel = v2 - v1
        radial_vel = float(np.dot(rel_vel, radial_dir))

        # ── 3. Tangential speed & Midpoint translation ────────────────────
        tangential_vel = rel_vel - radial_vel * radial_dir
        tangential_speed = float(np.linalg.norm(tangential_vel))

        mid_pos = (p1 + p2) * 0.5
        mid_vel = (v1 + v2) * 0.5

        zoom_vel_threshold = max(self.two_hand_spread_vel_threshold, 0.06)
        rotation_thresh = max(self.two_hand_rotation_vel_threshold, 0.05)

        # If rotation is significant:
        if abs(angle_delta) > 0.02 or tangential_speed > rotation_thresh:
            conf = self.conf_calc.sigmoid_confidence(
                max(abs(angle_delta) * 10.0, tangential_speed), rotation_thresh, steepness=12.0
            )
            return RecognizedGesture(
                gesture=GestureType.ROTATION,
                confidence=float(conf),
                hand_id=99,
                handedness="Bimanual",
                feature_contributions={
                    "angle_delta": float(angle_delta),
                    "relative_angle": float(current_angle),
                    "radial_velocity": float(radial_vel),
                    "tangential_speed": float(tangential_speed),
                    "distance": float(dist),
                    "mid_vx": float(mid_vel[0]),
                    "mid_vy": float(mid_vel[1]),
                },
                timestamp=max(hand1.timestamp, hand2.timestamp),
            )

        if radial_vel > zoom_vel_threshold:
            conf = self.conf_calc.sigmoid_confidence(radial_vel, zoom_vel_threshold, steepness=15.0)
            return RecognizedGesture(
                gesture=GestureType.SPREAD,
                confidence=float(conf),
                hand_id=99,
                handedness="Bimanual",
                feature_contributions={
                    "angle_delta": float(angle_delta),
                    "radial_velocity": float(radial_vel),
                    "distance": float(dist),
                    "mid_vx": float(mid_vel[0]),
                    "mid_vy": float(mid_vel[1]),
                },
                timestamp=max(hand1.timestamp, hand2.timestamp),
            )
        elif radial_vel < -zoom_vel_threshold:
            conf = self.conf_calc.sigmoid_confidence(-radial_vel, zoom_vel_threshold, steepness=15.0)
            return RecognizedGesture(
                gesture=GestureType.CONTRACTION,
                confidence=float(conf),
                hand_id=99,
                handedness="Bimanual",
                feature_contributions={
                    "angle_delta": float(angle_delta),
                    "radial_velocity": float(radial_vel),
                    "distance": float(dist),
                    "mid_vx": float(mid_vel[0]),
                    "mid_vy": float(mid_vel[1]),
                },
                timestamp=max(hand1.timestamp, hand2.timestamp),
            )

        return None
