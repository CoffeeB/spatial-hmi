"""
Deterministic and kinematic heuristic gesture classifier with continuous confidence estimation.
"""

from typing import List, Optional, Tuple
import numpy as np

from src.gestures.confidence_estimator import ConfidenceEstimator
from src.gestures.gesture_types import GestureType, RecognizedGesture
from src.landmarks.hand_state import HandState


class HeuristicGestureClassifier:
    """
    Classifies HandState objects into discrete gesture prototypes while computing
    continuous multi-feature confidence scores in [0, 1].
    """

    def __init__(
        self,
        pinch_threshold: float = 0.38,
        pinch_steepness: float = 18.0,
        point_extension_ratio: float = 1.18,
        open_palm_extension_ratio: float = 1.12,
        grab_closure_ratio: float = 0.90,
        two_hand_spread_vel_threshold: float = 0.04,
        two_hand_rotation_vel_threshold: float = 0.06,
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

    def classify_single_hand(self, hand: HandState) -> RecognizedGesture:
        """
        Evaluates one hand against canonical single-hand gesture prototypes:
        - PINCH: Thumb tip and index tip in close proximity, while other fingers are NOT in a closed fist
        - POINT: Index extended while middle, ring, pinky are curled
        - OPEN_PALM: All fingers extended
        - GRAB: All fingers curled tightly into a closed fist
        - RELEASE: Transition or open state
        """
        ratios = hand.finger_extension_ratios
        idx_ext = ratios.get("index", 1.0)
        mid_ext = ratios.get("middle", 1.0)
        rng_ext = ratios.get("ring", 1.0)
        pnk_ext = ratios.get("pinky", 1.0)

        # 1. FINGER EXTENSION & CURL METRICS
        c_idx_ext = self.conf_calc.sigmoid_confidence(idx_ext, self.point_extension_ratio, steepness=12.0)
        c_idx_curl = self.conf_calc.sigmoid_confidence(idx_ext, self.grab_closure_ratio, steepness=14.0, invert=True)
        c_mid_curl = self.conf_calc.sigmoid_confidence(mid_ext, self.grab_closure_ratio, steepness=14.0, invert=True)
        c_rng_curl = self.conf_calc.sigmoid_confidence(rng_ext, self.grab_closure_ratio, steepness=14.0, invert=True)
        c_pnk_curl = self.conf_calc.sigmoid_confidence(pnk_ext, self.grab_closure_ratio, steepness=14.0, invert=True)

        # Non-index fingers curl score (middle + ring + pinky)
        non_index_curl = (c_mid_curl + c_rng_curl + c_pnk_curl) / 3.0

        # 2. GRAB EVALUATION (All 4 fingers curled into a fist)
        grab_conf = self.conf_calc.combine_confidences([c_idx_curl, c_mid_curl, c_rng_curl, c_pnk_curl])

        # 3. PINCH EVALUATION
        # In a pinch, thumb and index touch, BUT other fingers are NOT all closed in a fist.
        # If the whole hand is in a closed fist, pinch is suppressed in favor of GRAB.
        raw_pinch_conf = hand.pinch_confidence
        pinch_suppression = 1.0 - (grab_conf * 0.95)
        pinch_conf = raw_pinch_conf * max(0.0, pinch_suppression)

        # 4. POINT EVALUATION (Index extended, others curled)
        c_mid_curl_pt = self.conf_calc.sigmoid_confidence(mid_ext, 1.05, steepness=12.0, invert=True)
        c_rng_curl_pt = self.conf_calc.sigmoid_confidence(rng_ext, 1.05, steepness=12.0, invert=True)
        c_pnk_curl_pt = self.conf_calc.sigmoid_confidence(pnk_ext, 1.05, steepness=12.0, invert=True)
        point_conf = self.conf_calc.combine_confidences([c_idx_ext, c_mid_curl_pt, c_rng_curl_pt, c_pnk_curl_pt])

        # 5. OPEN PALM EVALUATION (All fingers extended, not pinching)
        c_all_ext = [
            self.conf_calc.sigmoid_confidence(ratios.get(f, 1.0), self.open_palm_extension_ratio, steepness=10.0)
            for f in ["index", "middle", "ring", "pinky"]
        ]
        raw_open_palm = self.conf_calc.combine_confidences(c_all_ext)
        open_palm_conf = raw_open_palm * max(0.0, 1.0 - (raw_pinch_conf * 1.2))

        # 6. DIRECTIONAL SLAP / SWIPE EVALUATION (Fast directional palm velocity)
        vx, vy, vz = hand.palm_velocity
        swipe_conf = 0.0
        swipe_gesture = None
        swipe_vel_thresh = 0.30

        if open_palm_conf >= 0.40 and (abs(vx) > swipe_vel_thresh or abs(vy) > swipe_vel_thresh):
            if abs(vx) >= abs(vy):
                if vx < -swipe_vel_thresh:
                    swipe_gesture = GestureType.SWIPE_LEFT
                    swipe_conf = self.conf_calc.sigmoid_confidence(-vx, swipe_vel_thresh, steepness=10.0)
                elif vx > swipe_vel_thresh:
                    swipe_gesture = GestureType.SWIPE_RIGHT
                    swipe_conf = self.conf_calc.sigmoid_confidence(vx, swipe_vel_thresh, steepness=10.0)
            else:
                if vy < -swipe_vel_thresh:
                    swipe_gesture = GestureType.SWIPE_UP
                    swipe_conf = self.conf_calc.sigmoid_confidence(-vy, swipe_vel_thresh, steepness=10.0)
                elif vy > swipe_vel_thresh:
                    swipe_gesture = GestureType.SWIPE_DOWN
                    swipe_conf = self.conf_calc.sigmoid_confidence(vy, swipe_vel_thresh, steepness=10.0)

        # 7. SINGLE HAND SPREAD & SQUEEZE FINGERS (Zoom in / out)
        spread_score = open_palm_conf if (idx_ext > 1.22 and mid_ext > 1.22 and rng_ext > 1.22 and pnk_ext > 1.22) else 0.0
        squeeze_score = grab_conf if (idx_ext < 0.92 and mid_ext < 0.92 and rng_ext < 0.92 and pnk_ext < 0.92 and raw_pinch_conf < 0.4) else 0.0

        # Multi-class competitive assignment
        candidate_scores = [
            (GestureType.PINCH, pinch_conf),
            (GestureType.GRAB, grab_conf),
            (GestureType.POINT, point_conf),
            (GestureType.SPREAD_FINGERS, spread_score * 0.95),
            (GestureType.SQUEEZE_FINGERS, squeeze_score * 0.95),
            (GestureType.OPEN_PALM, open_palm_conf),
        ]

        if swipe_gesture is not None and swipe_conf > 0.60:
            candidate_scores.insert(0, (swipe_gesture, swipe_conf * 1.2))

        # Sort by confidence descending
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        best_gesture, best_conf = candidate_scores[0]

        if best_conf < self.min_confidence_threshold:
            best_gesture = GestureType.NONE
            best_conf = 0.0

        return RecognizedGesture(
            gesture=best_gesture,
            confidence=float(best_conf),
            hand_id=hand.hand_id,
            handedness=hand.handedness,
            feature_contributions={
                "pinch_score": float(pinch_conf),
                "grab_score": float(grab_conf),
                "point_score": float(point_conf),
                "open_palm_score": float(open_palm_conf),
                "vx": float(vx),
                "vy": float(vy),
            },
            timestamp=hand.timestamp,
        )

    def classify_two_hands(
        self,
        hand1: HandState,
        hand2: HandState,
        rec1: Optional[RecognizedGesture] = None,
        rec2: Optional[RecognizedGesture] = None,
    ) -> Optional[RecognizedGesture]:
        """
        Classifies intentional bimanual zoom interaction patterns (Spread / Contraction).
        Inhibits bimanual zoom if one hand is executing a deliberate single-hand command (Grab/Point/Pinch).
        """
        # If either hand has a strong single-hand unilateral gesture (e.g. Pointing or Grabbing),
        # prioritize single-hand control and do not trigger bimanual zoom.
        if rec1 and rec1.gesture in (GestureType.POINT, GestureType.GRAB, GestureType.PINCH) and rec1.confidence >= 0.50:
            return None
        if rec2 and rec2.gesture in (GestureType.POINT, GestureType.GRAB, GestureType.PINCH) and rec2.confidence >= 0.50:
            return None

        p1 = np.array(hand1.palm_center)
        p2 = np.array(hand2.palm_center)

        v1 = np.array(hand1.palm_velocity)
        v2 = np.array(hand2.palm_velocity)

        # Relative distance and rate of change
        rel_pos = p2 - p1
        dist = np.linalg.norm(rel_pos)
        if dist < 1e-4:
            return None

        # Radial velocity (rate of change of distance between palms)
        radial_dir = rel_pos / dist
        rel_vel = v2 - v1
        radial_vel = float(np.dot(rel_vel, radial_dir))

        # Check for bimanual expansion (zoom in) or contraction (zoom out)
        # Require substantial intentional velocity (> 0.08 units/sec)
        zoom_vel_threshold = max(self.two_hand_spread_vel_threshold, 0.07)

        if radial_vel > zoom_vel_threshold:
            conf = self.conf_calc.sigmoid_confidence(radial_vel, zoom_vel_threshold, steepness=15.0)
            return RecognizedGesture(
                gesture=GestureType.SPREAD,
                confidence=float(conf),
                hand_id=99,
                handedness="Bimanual",
                feature_contributions={"radial_velocity": radial_vel, "distance": float(dist)},
                timestamp=max(hand1.timestamp, hand2.timestamp),
            )
        elif radial_vel < -zoom_vel_threshold:
            conf = self.conf_calc.sigmoid_confidence(-radial_vel, zoom_vel_threshold, steepness=15.0)
            return RecognizedGesture(
                gesture=GestureType.CONTRACTION,
                confidence=float(conf),
                hand_id=99,
                handedness="Bimanual",
                feature_contributions={"radial_velocity": radial_vel, "distance": float(dist)},
                timestamp=max(hand1.timestamp, hand2.timestamp),
            )

        return None
