"""
Unit tests for CameraZoomController.
Verifies dynamic auto-zoom, hand distance adaptation, multi-hand framing,
pan smoothing, loss recovery, and crop transformations.
"""

import numpy as np
import pytest

from src.camera.zoom_controller import CameraZoomController
from src.landmarks.hand_state import HandState


def _create_mock_hand_state(
    hand_id: int = 0,
    handedness: str = "Right",
    center: tuple = (0.5, 0.5),
    span: float = 0.10,
) -> HandState:
    """Helper to create a synthetic HandState with specified center and span."""
    cx, cy = center
    half = span / 2.0

    # 21 mock landmarks in [0, 1] relative coordinates initialized around (cx, cy)
    raw_pts = np.zeros((21, 3), dtype=np.float32)
    raw_pts[:, 0] = cx
    raw_pts[:, 1] = cy

    # Wrist at bottom
    raw_pts[0] = [cx, cy + half, 0.0]
    # MCPs in middle
    raw_pts[9] = [cx, cy, 0.0]
    # Fingertips at top
    raw_pts[8] = [cx - half, cy - half, 0.0]
    raw_pts[12] = [cx, cy - half, 0.0]
    raw_pts[20] = [cx + half, cy - half, 0.0]

    return HandState(
        hand_id=hand_id,
        handedness=handedness,
        landmarks=[],
        raw_landmarks_array=raw_pts,
        normalized_landmarks_array=raw_pts,
        palm_center=(cx, cy, 0.0),
        palm_velocity=(0.0, 0.0, 0.0),
        hand_scale_ref=span,
        orientation_angles={"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        finger_extension_ratios={},
        finger_flexion_angles={},
        pinch_distance=1.0,
        pinch_confidence=0.0,
        detection_confidence=0.90,
        timestamp=100.0,
    )


def test_zoom_controller_initial_state():
    """Controller starts at wide-angle 1.0x centered at (0.5, 0.5)."""
    controller = CameraZoomController(min_zoom=1.0, max_zoom=3.5)
    assert controller.current_zoom == 1.0
    assert controller.target_zoom == 1.0
    assert controller.current_center == (0.5, 0.5)
    assert controller.crop_rect == (0.0, 0.0, 1.0, 1.0)
    assert not controller.is_tracking


def test_zoom_in_on_distant_small_hand():
    """Small hand at distance triggers smooth auto-zoom up towards max_zoom."""
    controller = CameraZoomController(
        min_zoom=1.0,
        max_zoom=3.5,
        target_hand_scale=0.32,
        zoom_speed=0.20,
    )

    # Hand at distance with small height span of 0.08 in full frame
    hand = _create_mock_hand_state(center=(0.6, 0.4), span=0.08)

    # Run multiple update frames to simulate video stream
    for _ in range(15):
        controller.update([hand], landmarks_in_full_frame=True)

    assert controller.is_tracking
    # Target zoom should be ~ 0.32 / 0.08 = 4.0, clamped to max_zoom (3.5)
    assert controller.target_zoom == pytest.approx(3.5, abs=0.05)
    # Current zoom should have smoothly zoomed in past 2.0x
    assert controller.current_zoom > 2.5
    # Center should have smoothly panned towards (0.6, 0.4)
    assert controller.current_center[0] > 0.55
    assert controller.current_center[1] < 0.45


def test_zoom_out_on_close_large_hand():
    """Large hand close up prevents over-zooming and stays near 1.0x wide."""
    controller = CameraZoomController(
        min_zoom=1.0,
        max_zoom=3.5,
        target_hand_scale=0.32,
        zoom_speed=0.20,
    )
    # Start controller already zoomed in
    controller.current_zoom = 2.5
    controller.current_center = (0.5, 0.5)

    # Hand close up with large height span of 0.45
    hand = _create_mock_hand_state(center=(0.5, 0.5), span=0.45)

    for _ in range(25):
        controller.update([hand], landmarks_in_full_frame=True)

    # Target zoom for 0.45 span with 0.32 target scale is <= 1.0 (min_zoom)
    assert controller.target_zoom == 1.0
    assert controller.current_zoom < 1.3


def test_multi_hand_framing():
    """Two hands spread apart are both framed together with margins."""
    controller = CameraZoomController(
        min_zoom=1.0,
        max_zoom=3.5,
        zoom_speed=0.25,
        pan_speed=0.25,
    )

    # Left hand at (0.3, 0.5), Right hand at (0.7, 0.5)
    hand_left = _create_mock_hand_state(hand_id=0, handedness="Left", center=(0.3, 0.5), span=0.10)
    hand_right = _create_mock_hand_state(hand_id=1, handedness="Right", center=(0.7, 0.5), span=0.10)

    for _ in range(15):
        controller.update([hand_left, hand_right], landmarks_in_full_frame=True)

    # Centroid between (0.3, 0.5) and (0.7, 0.5) is (0.5, 0.5)
    assert controller.current_center[0] == pytest.approx(0.5, abs=0.05)
    assert controller.current_center[1] == pytest.approx(0.5, abs=0.05)
    # Span is ~0.50, so target zoom fits both hands comfortably (around 1.4x - 1.8x)
    assert 1.2 <= controller.target_zoom <= 2.2


def test_loss_grace_window_and_smooth_zoom_out():
    """When hands leave view, hold position briefly then smoothly zoom out to wide angle."""
    controller = CameraZoomController(
        min_zoom=1.0,
        max_zoom=3.5,
        hold_frames=4,
        zoom_speed=0.20,
    )
    # Put controller in zoomed state
    controller.current_zoom = 3.0
    controller.target_zoom = 3.0
    controller.current_center = (0.7, 0.3)
    controller.is_tracking = True

    # Frames 1-4: within grace window, target zoom does not reset to 1.0 yet
    for _ in range(4):
        controller.update([])
        assert controller.loss_counter <= 4
        assert controller.target_zoom == 3.0

    # Frame 5+: grace window expired, target zoom decays to 1.0
    for _ in range(15):
        controller.update([])

    assert controller.target_zoom == 1.0
    assert controller.current_zoom < 1.25


def test_fast_motion_fallback_notification():
    """Fallback notification instantly snaps center towards newly spotted hand."""
    controller = CameraZoomController(min_zoom=1.0, max_zoom=3.5)
    controller.current_zoom = 2.5
    controller.current_center = (0.2, 0.2)

    # Hand suddenly appears on opposite side in full frame
    hand_new = _create_mock_hand_state(center=(0.85, 0.85), span=0.10)
    controller.notify_full_frame_detection([hand_new])

    # Center should immediately jump halfway towards (0.85, 0.85)
    assert controller.current_center[0] > 0.50
    assert controller.current_center[1] > 0.50
    assert controller.target_center[0] == pytest.approx(0.85, abs=0.05)


def test_crop_and_resize():
    """Crop produces the exact requested dimensions without distortion."""
    controller = CameraZoomController(min_zoom=1.0, max_zoom=3.0)
    controller.current_zoom = 2.0
    controller.current_center = (0.5, 0.5)
    controller._update_crop_rect()

    # Synthetic 1280x720 frame
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128

    zoomed, roi = controller.crop_and_resize(frame, out_w=640, out_h=360)
    assert zoomed.shape == (360, 640, 3)
    x0, y0, w, h = roi
    assert w == 640  # 1280 / 2.0
    assert h == 360  # 720 / 2.0


def test_coordinate_mapping_consistency():
    """Mapping from crop space to full space maps corners correctly."""
    controller = CameraZoomController(min_zoom=1.0, max_zoom=2.0)
    controller.current_zoom = 2.0
    controller.current_center = (0.5, 0.5)
    controller._update_crop_rect()

    # In 2x centered zoom, crop_rect is [0.25, 0.25, 0.75, 0.75]
    fx0, fy0 = controller.map_crop_to_full_norm(0.0, 0.0)
    fx1, fy1 = controller.map_crop_to_full_norm(1.0, 1.0)
    fxc, fyc = controller.map_crop_to_full_norm(0.5, 0.5)

    assert fx0 == pytest.approx(0.25, abs=0.01)
    assert fy0 == pytest.approx(0.25, abs=0.01)
    assert fx1 == pytest.approx(0.75, abs=0.01)
    assert fy1 == pytest.approx(0.75, abs=0.01)
    assert fxc == pytest.approx(0.50, abs=0.01)
    assert fyc == pytest.approx(0.50, abs=0.01)
