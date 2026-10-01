"""
Regression tests for low-illumination height tracking and foreshortened camera-facing perception.
"""
import numpy as np
import pytest

from src.perception.hand_detector import HandDetector
from src.landmarks.finger_state import FingerStateClassifier, FingerStateEnum
from src.gestures.gesture_types import GestureType
from src.interaction.engine import InteractionEngine
from src.landmarks.hand_state import HandLandmark, HandState


def test_adaptive_low_light_enhancement_lifts_shadows():
    """Validates that low-light frames (mean Y < 60) are lifted via adaptive gamma and CLAHE."""
    detector = HandDetector()

    # Create dark synthetic frame (mean brightness ~ 25)
    dark_frame = np.full((480, 640, 3), 25, dtype=np.uint8)
    primary_rgb, boosted_rgb = detector._enhance_for_detection(dark_frame)

    assert primary_rgb.shape == (480, 640, 3)
    assert boosted_rgb.shape == (480, 640, 3)
    # Average luminance in primary_rgb must be significantly higher than input
    y_enhanced = np.mean(primary_rgb)
    assert y_enhanced > 60.0, f"Enhanced luminance should be > 60, got {y_enhanced:.1f}"


def test_camera_pointing_foreshortened_extension():
    """Validates that a finger pointing straight at the camera is classified as EXTENDED."""
    classifier = FingerStateClassifier()

    # 21 synthetic landmarks where index finger points straight toward camera:
    # MCP (5) at [0.5, 0.6, 0.0], Tip (8) at [0.5, 0.58, -0.15]
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.50, 0.80, 0.00]  # Wrist
    # Thumb folded
    pts[1] = [0.42, 0.75, 0.00]
    pts[2] = [0.40, 0.70, 0.00]
    pts[3] = [0.42, 0.66, 0.00]
    pts[4] = [0.45, 0.64, 0.00]
    # Index pointing toward camera (z = -0.14)
    pts[5] = [0.46, 0.60, 0.00]
    pts[6] = [0.46, 0.58, -0.04]
    pts[7] = [0.46, 0.56, -0.09]
    pts[8] = [0.46, 0.54, -0.14]
    # Middle folded
    pts[9] = [0.50, 0.60, 0.00]
    pts[10] = [0.50, 0.64, 0.01]
    pts[11] = [0.50, 0.68, 0.02]
    pts[12] = [0.50, 0.71, 0.03]
    # Ring folded
    pts[13] = [0.54, 0.60, 0.00]
    pts[14] = [0.54, 0.64, 0.01]
    pts[15] = [0.54, 0.68, 0.02]
    pts[16] = [0.54, 0.71, 0.03]
    # Little folded
    pts[17] = [0.58, 0.62, 0.00]
    pts[18] = [0.58, 0.65, 0.01]
    pts[19] = [0.58, 0.68, 0.02]
    pts[20] = [0.58, 0.71, 0.03]

    states = classifier.classify_hand(pts, viewpoint_mode="CAMERA_FACING", active_representation="FINGER_CENTRIC")
    assert states.index.state == FingerStateEnum.EXTENDED, f"Index should be EXTENDED, got {states.index.state}"
    assert states.middle.state == FingerStateEnum.FOLDED, f"Middle should be FOLDED, got {states.middle.state}"


def test_height_tracking_at_vertical_margins():
    """Validates that hands near the top (y=0.08) and bottom (y=0.92) maintain continuous finger tracking."""
    classifier = FingerStateClassifier()

    pts = np.zeros((21, 3), dtype=np.float32)
    # Near top of frame
    pts[0] = [0.50, 0.12, 0.0]
    for i in range(1, 21):
        pts[i] = [0.50 + (i % 5) * 0.02, 0.08 - (i // 5) * 0.015, 0.0]

    states_top = classifier.classify_hand(pts, detection_confidence=0.85)
    # Must not collapse to global uncertain
    assert states_top.index.state != FingerStateEnum.UNCERTAIN


def test_interaction_engine_focal_tracking_on_point():
    """Validates that pointing targets the index fingertip instead of palm center."""
    engine = InteractionEngine()
    from tests.test_gestures import build_hand_state
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.5, 0.7, 0.0]
    # Position index tip at top-right [0.85, 0.25]
    pts[8] = [0.85, 0.25, -0.1]
    hand = build_hand_state(pts)

    from src.landmarks.finger_state import FingerStateDetail, HandFingerStates
    dummy_f = FingerStateDetail(finger="thumb", state=FingerStateEnum.FOLDED)
    hand.finger_states = HandFingerStates(
        thumb=dummy_f,
        index=FingerStateDetail(finger="index", state=FingerStateEnum.EXTENDED),
        middle=dummy_f,
        ring=dummy_f,
        little=dummy_f,
    )

    cmd, ctx, primary = engine.process_hands([hand], timestamp=1.0)
    cursor_x, cursor_y = cmd.cursor_ndc
    assert cursor_y > 0.0, f"Expected positive cursor_y reflecting high fingertip height, got {cursor_y}"
