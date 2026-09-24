"""
Unit tests for landmark normalization and kinematic feature extraction.
"""

import numpy as np
import pytest

from src.landmarks.hand_state import HandLandmark, HandState
from src.landmarks.kinematic_features import KinematicFeatureExtractor
from src.landmarks.normalization import LandmarkNormalizer


@pytest.fixture
def mock_open_palm_landmarks():
    """Generates a realistic 21-landmark array for an open palm."""
    pts = np.zeros((21, 3), dtype=np.float32)
    # Wrist at (0.5, 0.8, 0.0)
    pts[0] = [0.5, 0.8, 0.0]

    # Metacarpals (MCPs)
    pts[1] = [0.42, 0.72, 0.0]  # Thumb CMC
    pts[2] = [0.38, 0.65, 0.0]  # Thumb MCP
    pts[3] = [0.35, 0.58, 0.0]  # Thumb IP
    pts[4] = [0.32, 0.52, 0.0]  # Thumb Tip

    # Index finger extended upwards
    pts[5] = [0.45, 0.60, 0.0]  # Index MCP
    pts[6] = [0.45, 0.50, 0.0]  # Index PIP
    pts[7] = [0.45, 0.40, 0.0]  # Index DIP
    pts[8] = [0.45, 0.30, 0.0]  # Index Tip

    # Middle finger extended upwards
    pts[9] = [0.50, 0.58, 0.0]  # Middle MCP
    pts[10] = [0.50, 0.48, 0.0]
    pts[11] = [0.50, 0.38, 0.0]
    pts[12] = [0.50, 0.28, 0.0]  # Middle Tip

    # Ring finger extended
    pts[13] = [0.55, 0.60, 0.0]
    pts[14] = [0.55, 0.50, 0.0]
    pts[15] = [0.55, 0.42, 0.0]
    pts[16] = [0.55, 0.34, 0.0]

    # Pinky finger extended
    pts[17] = [0.60, 0.64, 0.0]
    pts[18] = [0.60, 0.56, 0.0]
    pts[19] = [0.60, 0.48, 0.0]
    pts[20] = [0.60, 0.40, 0.0]

    return pts


def test_landmark_normalization_translation_invariance(mock_open_palm_landmarks):
    normalizer = LandmarkNormalizer()

    norm1, palm1, d_ref1 = normalizer.normalize(mock_open_palm_landmarks)

    # Shift all landmarks by (0.2, -0.3, 0.5)
    shifted_landmarks = mock_open_palm_landmarks + np.array([0.2, -0.3, 0.5])
    norm2, palm2, d_ref2 = normalizer.normalize(shifted_landmarks)

    # Normalized landmarks must be identical after translation
    assert np.allclose(norm1, norm2, atol=1e-5)
    assert np.isclose(d_ref1, d_ref2, atol=1e-5)
    assert np.allclose(palm2 - palm1, [0.2, -0.3, 0.5], atol=1e-5)


def test_landmark_normalization_scale_invariance(mock_open_palm_landmarks):
    normalizer = LandmarkNormalizer()

    norm1, _, d_ref1 = normalizer.normalize(mock_open_palm_landmarks)

    # Scale landmarks by 2.5x around origin
    scaled_landmarks = mock_open_palm_landmarks * 2.5
    norm2, _, d_ref2 = normalizer.normalize(scaled_landmarks)

    assert np.isclose(d_ref2, d_ref1 * 2.5, atol=1e-4)
    assert np.allclose(norm1, norm2, atol=1e-4)


def test_kinematic_finger_extension(mock_open_palm_landmarks):
    kinematics = KinematicFeatureExtractor()
    ratios = kinematics.compute_finger_extension_ratios(mock_open_palm_landmarks)

    # In open palm, all 4 fingers must have extension ratio > 1.2
    assert ratios["index"] > 1.2
    assert ratios["middle"] > 1.2
    assert ratios["ring"] > 1.2
    assert ratios["pinky"] > 1.2


def test_pinch_confidence_sigmoid(mock_open_palm_landmarks):
    kinematics = KinematicFeatureExtractor()
    normalizer = LandmarkNormalizer()
    _, _, d_ref = normalizer.normalize(mock_open_palm_landmarks)

    # 1. Open palm pinch distance is large -> pinch confidence near 0
    _, conf_open = kinematics.compute_pinch_metric(mock_open_palm_landmarks, d_ref)
    assert conf_open < 0.20

    # 2. Modify landmarks so thumb tip (4) is coincident with index tip (8)
    pinched = mock_open_palm_landmarks.copy()
    pinched[4] = pinched[8] + np.array([0.01, 0.01, 0.0])

    _, conf_pinched = kinematics.compute_pinch_metric(pinched, d_ref)
    assert conf_pinched > 0.85
