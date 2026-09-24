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
        point_extension_ratio: float = 1.35,
        open_palm_extension_ratio: float = 1.25,
        grab_closure_ratio: float = 0.85,
        two_hand_spread_vel_threshold: float = 0.05,
        two_hand_rotation_vel_threshold: float = 0.08,
        min_confidence_threshold: float = 0.60,
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
        - PINCH: Thumb tip and index tip in close proximity
        - POINT: Index extended while middle, ring, pinky are curled
        - OPEN_PALM: All fingers extended
        - GRAB: All fingers curled (closure ratio < grab_closure_ratio)
        - RELEASE: Transition or open state
        """
        ratios = hand.finger_extension_ratios
        idx_ext = ratios.get("index", 1.0)
        mid_ext = ratios.get("middle", 1.0)
        rng_ext = ratios.get("ring", 1.0)
        pnk_ext = ratios.get("pinky", 1.0)

        # 1. PINCH EVALUATION
        pinch_conf = hand.pinch_confidence

        # 2. POINT EVALUATION
        # Index extended, other fingers curled
        c_idx_ext = self.conf_calc.sigmoid_confidence(idx_ext, self.point_extension_ratio, steepness=12.0)
        c_mid_curl = self.conf_calc.sigmoid_confidence(mid_ext, 1.05, steepness=12.0, invert=True)
        c_rng_curl = self.conf_calc.sigmoid_confidence(rng_ext, 1.05, steepness=12.0, invert=True)
        c_pnk_curl = self.conf_calc.sigmoid_confidence(pnk_ext, 1.05, steepness=12.0, invert=True)
        point_conf = self.conf_calc.combine_confidences([c_idx_ext, c_mid_curl, c_rng_curl, c_pnk_curl])

        # 3. OPEN PALM EVALUATION
        # All 4 fingers extended
        c_all_ext = [
            self.conf_calc.sigmoid_confidence(ratios.get(f, 1.0), self.open_palm_extension_ratio, steepness=10.0)
            for f in ["index", "middle", "ring", "pinky"]
        ]
        open_palm_conf = self.conf_calc.combine_confidences(c_all_ext)

        # 4. GRAB EVALUATION
        # All fingers curled tightly
        c_all_curl = [
            self.conf_calc.sigmoid_confidence(ratios.get(f, 1.0), self.grab_closure_ratio, steepness=14.0, invert=True)
            for f in ["index", "middle", "ring", "pinky"]
        ]
        grab_conf = self.conf_calc.combine_confidences(c_all_curl)

        # Multi-class competitive assignment
        candidate_scores = [
            (GestureType.PINCH, pinch_conf),
            (GestureType.POINT, point_conf),
            (GestureType.OPEN_PALM, open_palm_conf),
            (GestureType.GRAB, grab_conf),
        ]

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
                "point_score": float(point_conf),
                "open_palm_score": float(open_palm_conf),
                "grab_score": float(grab_conf),
            },
            timestamp=hand.timestamp,
        )

    def classify_two_hands(self, hand1: HandState, hand2: HandState) -> Optional[RecognizedGesture]:
        """
        Classifies bimanual interaction patterns (Spread, Contraction, Rotation).
        """
        p1 = np.array(hand1.palm_center)
        p2 = np.array(hand2.palm_center)

        v1 = np.array(hand1.palm_velocity)
        v2 = np.array(hand2.palm_velocity)

        # Relative distance and rate of change
        rel_pos = p2 - p1
        dist = np.linalg.norm(rel_pos)
        if dist < 1e-4:
            return None

        # Radial velocity (rate of change of distance)
        radial_dir = rel_pos / dist
        rel_vel = v2 - v1
        radial_vel = float(np.dot(rel_vel, radial_dir))

        # Check for bimanual expansion (zoom in) or contraction (zoom out)
        if radial_vel > self.two_hand_spread_vel_threshold:
            conf = self.conf_calc.sigmoid_confidence(
                radial_vel, self.two_hand_spread_vel_threshold, steepness=15.0
            )
            return RecognizedGesture(
                gesture=GestureType.SPREAD,
                confidence=float(conf),
                hand_id=99,
                handedness="Bimanual",
                feature_contributions={"radial_velocity": radial_vel, "distance": float(dist)},
                timestamp=max(hand1.timestamp, hand2.timestamp),
            )
        elif radial_vel < -self.two_hand_spread_vel_threshold:
            conf = self.conf_calc.sigmoid_confidence(
                -radial_vel, self.two_hand_spread_vel_threshold, steepness=15.0
            )
            return RecognizedGesture(
                gesture=GestureType.CONTRACTION,
                confidence=float(conf),
                hand_id=99,
                handedness="Bimanual",
                feature_contributions={"radial_velocity": radial_vel, "distance": float(dist)},
                timestamp=max(hand1.timestamp, hand2.timestamp),
            )

        # Tangential / angular velocity for two-hand rotation
        tangential_vel = rel_vel - radial_vel * radial_dir
        tangential_mag = float(np.linalg.norm(tangential_vel))
        if tangential_mag > self.two_hand_rotation_vel_threshold:
            conf = self.conf_calc.sigmoid_confidence(
                tangential_mag, self.two_hand_rotation_vel_threshold, steepness=12.0
            )
            return RecognizedGesture(
                gesture=GestureType.ROTATION,
                confidence=float(conf),
                hand_id=99,
                handedness="Bimanual",
                feature_contributions={"tangential_velocity": tangential_mag},
                timestamp=max(hand1.timestamp, hand2.timestamp),
            )

        return None
