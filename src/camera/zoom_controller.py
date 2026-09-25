"""
Dynamic Auto-Zoom and Auto-Focus Camera Controller ("Center Stage" for Hands).
Automatically focuses on, centers, and zooms in on hands to ensure maximum
trackability at distance while maintaining smooth, jitter-free framing.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.landmarks.hand_state import HandState
from src.utils.logging_config import setup_logger

logger = setup_logger("zoom_controller")


class CameraZoomController:
    """
    Intelligent digital auto-zoom and auto-framing controller.
    Dynamically magnifies distant hands to ensure the palm and individual phalanges
    receive high pixel density, enabling rock-solid hand tracking at maximum distance.
    """

    def __init__(
        self,
        enabled: bool = True,
        min_zoom: float = 1.0,
        max_zoom: float = 3.5,
        target_hand_scale: float = 0.32,
        zoom_speed: float = 0.10,
        pan_speed: float = 0.12,
        hold_frames: int = 8,
        deadband_zoom: float = 0.03,
        deadband_pan: float = 0.015,
    ):
        self.enabled = enabled
        self.min_zoom = max(1.0, float(min_zoom))
        self.max_zoom = max(self.min_zoom, float(max_zoom))
        self.target_hand_scale = float(target_hand_scale)
        self.zoom_speed = float(zoom_speed)
        self.pan_speed = float(pan_speed)
        self.hold_frames = int(hold_frames)
        self.deadband_zoom = float(deadband_zoom)
        self.deadband_pan = float(deadband_pan)

        # Internal state
        self.current_zoom: float = 1.0
        self.target_zoom: float = 1.0
        self.current_center: Tuple[float, float] = (0.5, 0.5)  # Normalized (cx, cy)
        self.target_center: Tuple[float, float] = (0.5, 0.5)

        # Normalized crop bounds [x_min, y_min, x_max, y_max] in [0, 1]
        self.crop_rect: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
        self.is_tracking: bool = False
        self.loss_counter: int = 0

    def reset(self):
        """Resets the zoom controller to default wide-angle state."""
        self.current_zoom = 1.0
        self.target_zoom = 1.0
        self.current_center = (0.5, 0.5)
        self.target_center = (0.5, 0.5)
        self.crop_rect = (0.0, 0.0, 1.0, 1.0)
        self.is_tracking = False
        self.loss_counter = 0

    def update(
        self,
        hand_states: List[HandState],
        timestamp: Optional[float] = None,
        landmarks_in_full_frame: bool = False,
    ):
        """
        Updates the target zoom and pan center based on the detected hands.
        Computes smooth exponential moving average to prevent camera jitter.
        """
        if not self.enabled:
            self.current_zoom = 1.0
            self.target_zoom = 1.0
            self.current_center = (0.5, 0.5)
            self.crop_rect = (0.0, 0.0, 1.0, 1.0)
            self.is_tracking = False
            return

        if len(hand_states) > 0:
            # Hands are in view!
            self.loss_counter = 0
            self.is_tracking = True

            # 1. Map landmarks to full-frame space
            full_pts_all: List[np.ndarray] = []
            x_min_crop, y_min_crop, x_max_crop, y_max_crop = self.crop_rect
            crop_w = x_max_crop - x_min_crop
            crop_h = y_max_crop - y_min_crop

            for hand in hand_states:
                pts = hand.normalized_landmarks_array
                if pts is None or len(pts) < 21:
                    continue
                if landmarks_in_full_frame:
                    full_pts_all.append(pts.copy())
                else:
                    full_pts = np.zeros_like(pts)
                    full_pts[:, 0] = x_min_crop + pts[:, 0] * crop_w
                    full_pts[:, 1] = y_min_crop + pts[:, 1] * crop_h
                    full_pts[:, 2] = pts[:, 2]
                    full_pts_all.append(full_pts)

            if len(full_pts_all) > 0:
                all_pts = np.vstack(full_pts_all)
                x_min = float(np.min(all_pts[:, 0]))
                x_max = float(np.max(all_pts[:, 0]))
                y_min = float(np.min(all_pts[:, 1]))
                y_max = float(np.max(all_pts[:, 1]))

                span_w = max(0.04, x_max - x_min)
                span_h = max(0.04, y_max - y_min)

                # Target center is the centroid of the combined hand bounding box
                self.target_center = (
                    float(np.clip((x_min + x_max) / 2.0, 0.1, 0.9)),
                    float(np.clip((y_min + y_max) / 2.0, 0.1, 0.9)),
                )

                # Target zoom calculation:
                if len(hand_states) == 1:
                    # Single hand: frame hand so it fills target_hand_scale of the view
                    # Using hand height span (wrist to tips)
                    computed_zoom = self.target_hand_scale / max(span_h, 0.05)
                else:
                    # Multiple hands: fit both hands with 30% margin
                    zoom_w = 0.70 / max(span_w, 0.10)
                    zoom_h = 0.70 / max(span_h, 0.10)
                    computed_zoom = min(zoom_w, zoom_h)

                self.target_zoom = float(np.clip(computed_zoom, self.min_zoom, self.max_zoom))
        else:
            # No hands detected in current view
            self.loss_counter += 1
            if self.loss_counter > self.hold_frames:
                # Hold grace window expired: smoothly ease back to wide angle view
                self.target_zoom = 1.0
                self.target_center = (0.5, 0.5)
                if abs(self.current_zoom - 1.0) < 0.05:
                    self.is_tracking = False

        # Apply smooth Exponential Moving Average (EMA) with deadband
        # 1. Zoom smoothing
        delta_zoom = self.target_zoom - self.current_zoom
        if abs(delta_zoom) > self.deadband_zoom:
            self.current_zoom += self.zoom_speed * delta_zoom
        else:
            # Inside deadband: keep steady
            pass
        self.current_zoom = float(np.clip(self.current_zoom, self.min_zoom, self.max_zoom))

        # 2. Pan smoothing
        dcx = self.target_center[0] - self.current_center[0]
        dcy = self.target_center[1] - self.current_center[1]
        dist_pan = np.hypot(dcx, dcy)
        if dist_pan > self.deadband_pan:
            # Accelerated pan if tracking hands
            speed = self.pan_speed * 1.5 if self.loss_counter == 0 else self.pan_speed
            self.current_center = (
                float(self.current_center[0] + speed * dcx),
                float(self.current_center[1] + speed * dcy),
            )

        # 3. Update normalized crop rectangle
        self._update_crop_rect()

    def notify_full_frame_detection(self, hand_states: List[HandState]):
        """
        Called when hands were missed in the zoomed crop but immediately found
        in the wide-angle fallback scan. Snaps target center to the newly spotted hand.
        """
        if not hand_states:
            return

        all_pts: List[np.ndarray] = []
        for hand in hand_states:
            pts = hand.normalized_landmarks_array
            if pts is not None and len(pts) >= 21:
                all_pts.append(pts)

        if not all_pts:
            return

        pts_cat = np.vstack(all_pts)
        x_min = float(np.min(pts_cat[:, 0]))
        x_max = float(np.max(pts_cat[:, 0]))
        y_min = float(np.min(pts_cat[:, 1]))
        y_max = float(np.max(pts_cat[:, 1]))

        # Instantly update target and accelerate pan towards new location
        self.target_center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
        # Pull center 50% of the way immediately for rapid re-acquisition
        self.current_center = (
            (self.current_center[0] + self.target_center[0]) / 2.0,
            (self.current_center[1] + self.target_center[1]) / 2.0,
        )
        self.loss_counter = 0
        self.is_tracking = True
        self._update_crop_rect()

    def _update_crop_rect(self):
        """Computes [x_min, y_min, x_max, y_max] in [0, 1] based on current_zoom and current_center."""
        z = max(1.0, self.current_zoom)
        box_w = 1.0 / z
        box_h = 1.0 / z

        cx, cy = self.current_center
        x_min = cx - box_w / 2.0
        y_min = cy - box_h / 2.0

        # Clamp within [0, 1] while preserving box size
        if x_min < 0.0:
            x_min = 0.0
        elif x_min + box_w > 1.0:
            x_min = 1.0 - box_w

        if y_min < 0.0:
            y_min = 0.0
        elif y_min + box_h > 1.0:
            y_min = 1.0 - box_h

        self.crop_rect = (float(x_min), float(y_min), float(x_min + box_w), float(y_min + box_h))

    def crop_and_resize(
        self,
        full_frame: np.ndarray,
        out_w: int = 640,
        out_h: int = 480,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Crops the full camera frame according to current zoom & pan,
        and resizes the crop to the target output dimensions.
        Returns:
            zoomed_frame: np.ndarray of shape (out_h, out_w, 3)
            pixel_roi: (x0, y0, w_crop, h_crop) in full_frame pixel coordinates
        """
        h_full, w_full = full_frame.shape[:2]
        x_min, y_min, x_max, y_max = self.crop_rect

        x0 = int(np.clip(round(x_min * w_full), 0, w_full - 1))
        y0 = int(np.clip(round(y_min * h_full), 0, h_full - 1))
        x1 = int(np.clip(round(x_max * w_full), x0 + 1, w_full))
        y1 = int(np.clip(round(y_max * h_full), y0 + 1, h_full))

        crop = full_frame[y0:y1, x0:x1]
        pixel_roi = (x0, y0, x1 - x0, y1 - y0)

        # High-quality resize for sharpness
        if (x1 - x0) == out_w and (y1 - y0) == out_h:
            return crop, pixel_roi

        interp = cv2.INTER_AREA if (x1 - x0) > out_w else cv2.INTER_LINEAR
        zoomed_frame = cv2.resize(crop, (out_w, out_h), interpolation=interp)
        return zoomed_frame, pixel_roi

    def map_crop_to_full_norm(self, x_crop: float, y_crop: float) -> Tuple[float, float]:
        """Maps a point from cropped normalized coords [0, 1] to full-frame normalized coords [0, 1]."""
        x_min, y_min, x_max, y_max = self.crop_rect
        return (
            x_min + x_crop * (x_max - x_min),
            y_min + y_crop * (y_max - y_min),
        )
