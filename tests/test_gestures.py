"""
Unit tests for single-hand and bimanual gesture classification.
"""

import numpy as np
import pytest

from src.gestures.gesture_types import GestureType
from src.gestures.heuristic_classifier import HeuristicGestureClassifier
from src.landmarks.hand_state import HandLandmark, HandState
from src.landmarks.kinematic_features import KinematicFeatureExtractor
from src.landmarks.normalization import LandmarkNormalizer


def build_hand_state(raw_pts: np.ndarray, hand_id: int = 0, handedness: str = "Right") -> HandState:
    normalizer = LandmarkNormalizer()
    kinematics = KinematicFeatureExtractor()

    norm_pts, palm_center, d_ref = normalizer.normalize(raw_pts)
    finger_ratios = kinematics.compute_finger_extension_ratios(raw_pts)
    flexion_angles = kinematics.compute_joint_flexion_angles(raw_pts)
    pinch_dist, pinch_conf = kinematics.compute_pinch_metric(raw_pts, d_ref)
    orientation = kinematics.compute_hand_orientation(raw_pts)

    landmarks = [
        HandLandmark(index=i, x=float(p[0]), y=float(p[1]), z=float(p[2]))
        for i, p in enumerate(raw_pts)
    ]

    return HandState(
        hand_id=hand_id,
        handedness=handedness,
        landmarks=landmarks,
        raw_landmarks_array=raw_pts,
        normalized_landmarks_array=norm_pts,
        palm_center=(float(palm_center[0]), float(palm_center[1]), float(palm_center[2])),
        palm_velocity=(0.0, 0.0, 0.0),
        hand_scale_ref=d_ref,
        orientation_angles=orientation,
        finger_extension_ratios=finger_ratios,
        finger_flexion_angles=flexion_angles,
        pinch_distance=pinch_dist,
        pinch_confidence=pinch_conf,
        detection_confidence=0.95,
        timestamp=100.0,
    )


def classify_n_frames(classifier: HeuristicGestureClassifier, hand: HandState, n: int = 6):
    """
    Pre-warm the EMA smoother by classifying the same hand state N times.

    The v2 classifier uses per-gesture exponential smoothing (α=0.65) so that
    a single-frame spike cannot trigger a gesture. Real gestures accumulate
    over several frames; this helper simulates that for unit tests.

    Returns the result of the final (Nth) classification.
    """
    result = None
    for _ in range(n):
        result = classifier.classify_single_hand(hand)
    return result


def test_classify_open_palm():
    classifier = HeuristicGestureClassifier()

    # Open palm configuration
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.5, 0.8, 0.0]
    # MCPs
    pts[1] = [0.42, 0.72, 0.0]; pts[5] = [0.45, 0.60, 0.0]; pts[9] = [0.50, 0.58, 0.0]; pts[13] = [0.55, 0.60, 0.0]; pts[17] = [0.60, 0.64, 0.0]
    # Tips extended
    pts[4] = [0.32, 0.52, 0.0]; pts[8] = [0.45, 0.30, 0.0]; pts[12] = [0.50, 0.28, 0.0]; pts[16] = [0.55, 0.34, 0.0]; pts[20] = [0.60, 0.40, 0.0]
    # Intermediate joints
    pts[6] = [0.45, 0.50, 0.0]; pts[7] = [0.45, 0.40, 0.0]
    pts[10] = [0.50, 0.48, 0.0]; pts[11] = [0.50, 0.38, 0.0]
    pts[14] = [0.55, 0.50, 0.0]; pts[15] = [0.55, 0.42, 0.0]
    pts[18] = [0.60, 0.56, 0.0]; pts[19] = [0.60, 0.48, 0.0]

    hand = build_hand_state(pts)
    # v2: EMA smoother requires multiple frames to build confidence
    rec = classify_n_frames(classifier, hand, n=6)

    assert rec.gesture == GestureType.OPEN_PALM
    assert rec.confidence > 0.30


def test_classify_point():
    classifier = HeuristicGestureClassifier()

    # Point: index extended, others curled
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.5, 0.8, 0.0]
    # MCPs
    pts[1] = [0.42, 0.72, 0.0]; pts[5] = [0.45, 0.60, 0.0]; pts[9] = [0.50, 0.58, 0.0]; pts[13] = [0.55, 0.60, 0.0]; pts[17] = [0.60, 0.64, 0.0]
    # Index extended
    pts[6] = [0.45, 0.50, 0.0]; pts[7] = [0.45, 0.40, 0.0]; pts[8] = [0.45, 0.30, 0.0]
    # Middle, Ring, Pinky folded tightly near MCPs
    pts[10] = [0.50, 0.62, 0.0]; pts[11] = [0.50, 0.65, 0.0]; pts[12] = [0.50, 0.68, 0.0]
    pts[14] = [0.55, 0.62, 0.0]; pts[15] = [0.55, 0.65, 0.0]; pts[16] = [0.55, 0.68, 0.0]
    pts[18] = [0.60, 0.66, 0.0]; pts[19] = [0.60, 0.68, 0.0]; pts[20] = [0.60, 0.70, 0.0]
    # Thumb folded
    pts[2] = [0.42, 0.70, 0.0]; pts[3] = [0.44, 0.68, 0.0]; pts[4] = [0.46, 0.68, 0.0]

    hand = build_hand_state(pts)
    # v2: EMA smoother requires multiple frames to build confidence
    rec = classify_n_frames(classifier, hand, n=6)

    assert rec.gesture == GestureType.POINT
    assert rec.confidence > 0.38


def test_classify_pinch():
    classifier = HeuristicGestureClassifier()

    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.5, 0.8, 0.0]
    pts[5] = [0.45, 0.60, 0.0]; pts[9] = [0.50, 0.58, 0.0]; pts[13] = [0.55, 0.60, 0.0]; pts[17] = [0.60, 0.64, 0.0]
    # Thumb and index tips touching at (0.42, 0.45, 0.0)
    pts[4] = [0.42, 0.45, 0.0]
    pts[8] = [0.42, 0.45, 0.0]
    pts[6] = [0.44, 0.52, 0.0]; pts[7] = [0.43, 0.48, 0.0]
    pts[2] = [0.40, 0.65, 0.0]; pts[3] = [0.41, 0.55, 0.0]

    hand = build_hand_state(pts)
    # v2: EMA smoother requires multiple frames to build confidence
    rec = classify_n_frames(classifier, hand, n=6)

    assert rec.gesture == GestureType.PINCH
    assert rec.confidence > 0.50


def test_classify_directional_slaps():
    classifier = HeuristicGestureClassifier()

    # Open palm landmarks
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.5, 0.8, 0.0]
    pts[4] = [0.32, 0.52, 0.0]; pts[8] = [0.45, 0.30, 0.0]; pts[12] = [0.50, 0.28, 0.0]; pts[16] = [0.55, 0.34, 0.0]; pts[20] = [0.60, 0.40, 0.0]
    pts[5] = [0.45, 0.60, 0.0]; pts[9] = [0.50, 0.58, 0.0]; pts[13] = [0.55, 0.60, 0.0]; pts[17] = [0.60, 0.64, 0.0]

    # Slap Left (vx = -0.65)
    hand_left = build_hand_state(pts)
    hand_left.palm_velocity = (-0.65, 0.0, 0.0)
    rec_left = classifier.classify_single_hand(hand_left)
    assert rec_left.gesture == GestureType.SWIPE_LEFT

    # Slap Right (vx = +0.65)
    hand_right = build_hand_state(pts)
    hand_right.palm_velocity = (0.65, 0.0, 0.0)
    rec_right = classifier.classify_single_hand(hand_right)
    assert rec_right.gesture == GestureType.SWIPE_RIGHT

    # Slap Up (vy = -0.65)
    hand_up = build_hand_state(pts)
    hand_up.palm_velocity = (0.0, -0.65, 0.0)
    rec_up = classifier.classify_single_hand(hand_up)
    assert rec_up.gesture == GestureType.SWIPE_UP

    # Slap Down (vy = +0.65)
    hand_down = build_hand_state(pts)
    hand_down.palm_velocity = (0.0, 0.65, 0.0)
    rec_down = classifier.classify_single_hand(hand_down)
    assert rec_down.gesture == GestureType.SWIPE_DOWN


def test_classify_spread_fingers():
    classifier = HeuristicGestureClassifier()

    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.5, 0.8, 0.0]
    # Fingers extended high and spread wide
    pts[4] = [0.25, 0.45, 0.0]; pts[8] = [0.40, 0.25, 0.0]; pts[12] = [0.50, 0.22, 0.0]; pts[16] = [0.60, 0.26, 0.0]; pts[20] = [0.72, 0.35, 0.0]
    pts[5] = [0.45, 0.60, 0.0]; pts[9] = [0.50, 0.58, 0.0]; pts[13] = [0.55, 0.60, 0.0]; pts[17] = [0.60, 0.64, 0.0]

    hand = build_hand_state(pts)
    # v2: EMA smoother requires multiple frames to build confidence
    rec = classify_n_frames(classifier, hand, n=6)
    assert rec.gesture in (GestureType.SPREAD_FINGERS, GestureType.OPEN_PALM)
    assert rec.confidence > 0.30
