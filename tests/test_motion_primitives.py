"""
Unit tests for Gestura Level 2 Motion Primitives.
Verifies independent kinematic motion tracking: translations (left, right, up, down),
depth (toward, away), stationary hold/dwell, acceleration/deceleration,
orbital rotation (CW/CCW), release, linearity, and trajectory measurement.
"""

import math
import numpy as np
import pytest

from src.motion.motion_primitive import DynamicState, FingerMotionPrimitive, MotionPrimitive
from src.motion.motion_tracker import MotionPrimitiveTracker


def test_motion_tracker_initial_state():
    """Initial observation returns stationary state with zero velocity."""
    tracker = MotionPrimitiveTracker()
    state = tracker.update(
        hand_id=0,
        handedness="Right",
        palm_center=(0.5, 0.5, 0.0),
        hand_scale_ref=0.20,
        timestamp=100.0,
    )

    assert state.motion_primitive == MotionPrimitive.STATIONARY
    assert state.dynamic_state == DynamicState.STATIONARY
    assert state.speed == 0.0
    assert state.velocity == (0.0, 0.0, 0.0)
    assert not state.is_holding
    assert not state.is_releasing


def test_motion_cardinal_left_and_right():
    """Verifies pure horizontal translations (left and right)."""
    tracker = MotionPrimitiveTracker(velocity_alpha=0.80)

    # 1. Moving Left: X decreasing by 0.01 per 33ms (~0.30 units/s)
    x = 0.60
    t = 100.0
    state = None
    for _ in range(6):
        state = tracker.update(
            hand_id=0,
            handedness="Right",
            palm_center=(x, 0.5, 0.0),
            hand_scale_ref=0.20,
            timestamp=t,
        )
        x -= 0.01
        t += 0.033

    assert state is not None
    assert state.motion_primitive == MotionPrimitive.MOVE_LEFT
    assert state.primary_direction == "LEFT"
    assert state.velocity[0] < -0.15
    assert abs(state.heading_deg) > 150.0  # Cartesian heading pointing Left (-180 / +180)
    assert state.linearity > 0.90

    # 2. Moving Right on hand 1: X increasing
    tracker.reset()
    x = 0.40
    t = 100.0
    for _ in range(6):
        state = tracker.update(
            hand_id=1,
            handedness="Left",
            palm_center=(x, 0.5, 0.0),
            hand_scale_ref=0.20,
            timestamp=t,
        )
        x += 0.01
        t += 0.033

    assert state.motion_primitive == MotionPrimitive.MOVE_RIGHT
    assert state.primary_direction == "RIGHT"
    assert state.velocity[0] > 0.15
    assert abs(state.heading_deg) < 30.0  # Cartesian heading pointing Right (0 deg)


def test_motion_cardinal_up_and_down():
    """Verifies pure vertical translations (up and down)."""
    tracker = MotionPrimitiveTracker(velocity_alpha=0.80)

    # 1. Moving UP: In MediaPipe screen space, UP is decreasing Y
    y = 0.60
    t = 100.0
    state = None
    for _ in range(6):
        state = tracker.update(
            hand_id=0,
            handedness="Right",
            palm_center=(0.5, y, 0.0),
            hand_scale_ref=0.20,
            timestamp=t,
        )
        y -= 0.01
        t += 0.033

    assert state.motion_primitive == MotionPrimitive.MOVE_UP
    assert state.primary_direction == "UP"
    assert state.velocity[1] < -0.15
    assert 60.0 < state.heading_deg < 120.0  # Cartesian +90 deg is UP

    # 2. Moving DOWN: Increasing Y
    tracker.reset()
    y = 0.40
    t = 100.0
    for _ in range(6):
        state = tracker.update(
            hand_id=0,
            handedness="Right",
            palm_center=(0.5, y, 0.0),
            hand_scale_ref=0.20,
            timestamp=t,
        )
        y += 0.01
        t += 0.033

    assert state.motion_primitive == MotionPrimitive.MOVE_DOWN
    assert state.primary_direction == "DOWN"
    assert state.velocity[1] > 0.15
    assert -120.0 < state.heading_deg < -60.0  # Cartesian -90 deg is DOWN


def test_motion_depth_toward_and_away():
    """Verifies depth motion (approach/toward vs retreat/away)."""
    tracker = MotionPrimitiveTracker(velocity_alpha=0.80)

    # 1. Approach / Toward: Hand expands on sensor (d_ref increases)
    scale = 0.18
    t = 100.0
    state = None
    for _ in range(6):
        state = tracker.update(
            hand_id=0,
            handedness="Right",
            palm_center=(0.5, 0.5, 0.0),
            hand_scale_ref=scale,
            timestamp=t,
        )
        scale += 0.012  # Expanding ~0.36/s
        t += 0.033

    assert state.depth_state == "TOWARD"
    assert state.motion_primitive == MotionPrimitive.MOVE_TOWARD
    assert state.scale_rate > 0.20

    # 2. Retreat / Away: Hand contracts on sensor (d_ref decreases)
    tracker.reset()
    scale = 0.30
    t = 100.0
    for _ in range(6):
        state = tracker.update(
            hand_id=0,
            handedness="Right",
            palm_center=(0.5, 0.5, 0.0),
            hand_scale_ref=scale,
            timestamp=t,
        )
        scale -= 0.012  # Contracting
        t += 0.033

    assert state.depth_state == "AWAY"
    assert state.motion_primitive == MotionPrimitive.MOVE_AWAY
    assert state.scale_rate < -0.20


def test_motion_stationary_hold_dwell():
    """Holding hand stationary transitions to HOLD after dwell threshold (400ms)."""
    tracker = MotionPrimitiveTracker(dwell_hold_ms=400.0)

    # Frames 0 to 200ms: STATIONARY, not yet HOLD
    t = 100.0
    for _ in range(6):
        state = tracker.update(
            hand_id=0,
            handedness="Right",
            palm_center=(0.5, 0.5, 0.0),
            hand_scale_ref=0.20,
            timestamp=t,
        )
        t += 0.033

    assert state.motion_primitive == MotionPrimitive.STATIONARY
    assert not state.is_holding
    assert state.dwell_duration_ms > 150.0

    # Continue holding until 500ms total dwell
    for _ in range(10):
        state = tracker.update(
            hand_id=0,
            handedness="Right",
            palm_center=(0.5, 0.5, 0.0),
            hand_scale_ref=0.20,
            timestamp=t,
        )
        t += 0.033

    assert state.is_holding
    assert state.motion_primitive == MotionPrimitive.HOLD
    assert state.dwell_duration_ms >= 400.0


def test_motion_acceleration_and_deceleration():
    """Detects speed acceleration vs deceleration phases."""
    tracker = MotionPrimitiveTracker(velocity_alpha=0.90)

    # 1. Accelerating stroke: speed increases non-linearly
    x = 0.30
    t = 100.0
    step = 0.003
    for _ in range(6):
        state = tracker.update(
            hand_id=0,
            handedness="Right",
            palm_center=(x, 0.5, 0.0),
            hand_scale_ref=0.20,
            timestamp=t,
        )
        x += step
        step += 0.004  # Accelerating
        t += 0.033

    assert state.dynamic_state == DynamicState.ACCELERATING
    assert state.tangential_acceleration > 0.0

    # 2. Decelerating stroke: speed decreases
    for _ in range(5):
        step = max(0.002, step - 0.006)  # Decelerating
        x += step
        t += 0.033
        state = tracker.update(
            hand_id=0,
            handedness="Right",
            palm_center=(x, 0.5, 0.0),
            hand_scale_ref=0.20,
            timestamp=t,
        )

    assert state.dynamic_state == DynamicState.DECELERATING
    assert state.tangential_acceleration < 0.0


def test_motion_orbital_rotation_clockwise():
    """Circular trajectory generates ROTATE_CW primitive with angular velocity."""
    tracker = MotionPrimitiveTracker(buffer_capacity=30)

    # Generate circular orbit around (0.5, 0.5) with radius 0.08
    cx, cy, r = 0.5, 0.5, 0.08
    t = 100.0
    state = None

    # Step clockwise: angle increasing in screen coords (+Y down)
    for step in range(24):
        theta = step * (math.pi / 8.0)  # ~3 full radians (170 deg)
        px = cx + r * math.cos(theta)
        py = cy + r * math.sin(theta)
        state = tracker.update(
            hand_id=0,
            handedness="Right",
            palm_center=(px, py, 0.0),
            hand_scale_ref=0.20,
            timestamp=t,
        )
        t += 0.033

    assert state.rotation_direction == "CLOCKWISE"
    assert state.motion_primitive == MotionPrimitive.ROTATE_CW
    assert state.angular_velocity > 0.0
    assert abs(state.cumulative_angle_deg) >= 120.0


def test_motion_release_transition():
    """Sudden stop after fast stroke triggers RELEASE lifecycle event."""
    tracker = MotionPrimitiveTracker(velocity_alpha=0.85)

    # 1. Fast stroke
    x = 0.30
    t = 100.0
    state = None
    for _ in range(5):
        x += 0.020  # Fast: ~0.60 units/s
        t += 0.033
        state = tracker.update(
            hand_id=0,
            handedness="Right",
            palm_center=(x, 0.5, 0.0),
            hand_scale_ref=0.20,
            timestamp=t,
        )

    assert state is not None
    assert state.speed > 0.20

    # 2. Sudden dead stop at the exact same x position
    t += 0.033
    state = tracker.update(
        hand_id=0,
        handedness="Right",
        palm_center=(x, 0.5, 0.0),  # No movement from previous frame
        hand_scale_ref=0.20,
        timestamp=t,
    )

    assert state.is_releasing
    assert state.motion_primitive == MotionPrimitive.RELEASE


def test_motion_linearity_and_trajectory():
    """Straight stroke has high linearity (near 1.0); meandering has lower linearity."""
    tracker = MotionPrimitiveTracker()

    # Straight line stroke
    x = 0.20
    t = 100.0
    state = None
    for _ in range(10):
        state = tracker.update(
            hand_id=0,
            handedness="Right",
            palm_center=(x, 0.5, 0.0),
            hand_scale_ref=0.20,
            timestamp=t,
        )
        x += 0.015
        t += 0.033

    assert state.linearity > 0.95
    assert state.displacement_magnitude > 0.10
    assert len(state.trajectory) >= 10


def test_simultaneous_dual_hand_independent_motion():
    """
    Verifies that Left and Right hands moving simultaneously are tracked
    with complete physical independence and zero crosstalk.
    """
    tracker = MotionPrimitiveTracker(velocity_alpha=0.85)

    # Hand 0 (Left): moves UP (Y decreasing)
    # Hand 1 (Right): moves DOWN (Y increasing)
    y_left = 0.60
    y_right = 0.40
    t = 100.0

    state_left = None
    state_right = None

    for _ in range(8):
        y_left -= 0.015   # Moving UP
        y_right += 0.015  # Moving DOWN
        t += 0.033

        state_left = tracker.update(
            hand_id=0,
            handedness="Left",
            palm_center=(0.30, y_left, 0.0),
            hand_scale_ref=0.20,
            timestamp=t,
        )
        state_right = tracker.update(
            hand_id=1,
            handedness="Right",
            palm_center=(0.70, y_right, 0.0),
            hand_scale_ref=0.20,
            timestamp=t,
        )

    # Validate Left Hand is MOVE_UP
    assert state_left is not None
    assert state_left.motion_primitive == MotionPrimitive.MOVE_UP
    assert state_left.primary_direction == "UP"
    assert state_left.velocity[1] < -0.20
    assert state_left.speed > 0.20

    # Validate Right Hand is MOVE_DOWN
    assert state_right is not None
    assert state_right.motion_primitive == MotionPrimitive.MOVE_DOWN
    assert state_right.primary_direction == "DOWN"
    assert state_right.velocity[1] > 0.20
    assert state_right.speed > 0.20

    # Now hold Left hand still while accelerating Right hand to the RIGHT
    x_right = 0.70
    for _ in range(15):
        t += 0.033
        x_right += 0.025  # Fast rightward flick

        # Left hand stays completely still at its final y_left
        state_left = tracker.update(
            hand_id=0,
            handedness="Left",
            palm_center=(0.30, y_left, 0.0),
            hand_scale_ref=0.20,
            timestamp=t,
        )
        # Right hand moves right
        state_right = tracker.update(
            hand_id=1,
            handedness="Right",
            palm_center=(x_right, y_right, 0.0),
            hand_scale_ref=0.20,
            timestamp=t,
        )

    # Left hand should now be holding still (is_holding=True or STATIONARY/HOLD)
    assert state_left.speed < 0.05
    assert state_left.is_holding or state_left.motion_primitive in (MotionPrimitive.HOLD, MotionPrimitive.STATIONARY)

    # Right hand should be actively translating rightward
    assert state_right.motion_primitive == MotionPrimitive.MOVE_RIGHT
    assert state_right.primary_direction == "RIGHT"
    assert state_right.speed > 0.30


def test_finger_motion_extending_and_flexing():
    """Verifies that individual finger uncurling (extending) and curling (flexing) are tracked."""
    tracker = MotionPrimitiveTracker()
    raw_lms = np.zeros((21, 3), dtype=np.float32)
    # Palm at (0.5, 0.5, 0.0)
    raw_lms[0] = [0.5, 0.6, 0.0]
    raw_lms[9] = [0.5, 0.5, 0.0]

    # 1. Index finger extending: extension ratio increases rapidly from 0.70 to 1.45
    t = 100.0
    ext_ratio = 0.70
    tip_y = 0.50
    state = None

    for _ in range(5):
        ext_ratio += 0.15
        tip_y -= 0.025  # Tip moves upward (extending)
        raw_lms[8] = [0.5, tip_y, 0.0]  # Index tip (8)
        t += 0.033

        state = tracker.update(
            hand_id=0,
            handedness="Right",
            palm_center=(0.5, 0.5, 0.0),
            hand_scale_ref=0.20,
            timestamp=t,
            raw_landmarks=raw_lms,
            palm_facing="PALM",
            finger_ratios={"index": ext_ratio},
        )

    assert "Index" in state.finger_motions
    idx_motion = state.finger_motions["Index"]
    assert idx_motion.is_extending
    assert idx_motion.motion_primitive == FingerMotionPrimitive.EXTENDING
    assert idx_motion.extension_rate > 0.20

    # 2. Index finger flexing: extension ratio decreases rapidly from 1.45 to 0.65
    for _ in range(5):
        ext_ratio -= 0.15
        tip_y += 0.025  # Tip moves downward towards palm
        raw_lms[8] = [0.5, tip_y, 0.0]
        t += 0.033

        state = tracker.update(
            hand_id=0,
            handedness="Right",
            palm_center=(0.5, 0.5, 0.0),
            hand_scale_ref=0.20,
            timestamp=t,
            raw_landmarks=raw_lms,
            palm_facing="PALM",
            finger_ratios={"index": ext_ratio},
        )

    idx_motion = state.finger_motions["Index"]
    assert idx_motion.is_flexing
    assert idx_motion.motion_primitive == FingerMotionPrimitive.FLEXING
    assert idx_motion.extension_rate < -0.20


def test_finger_motion_tapping():
    """Verifies that a rapid downward stroke and recovery of the index finger is classified as TAPPING."""
    tracker = MotionPrimitiveTracker()
    raw_lms = np.zeros((21, 3), dtype=np.float32)
    raw_lms[0] = [0.5, 0.6, 0.0]
    raw_lms[9] = [0.5, 0.5, 0.0]

    # Index starts at rest
    tip_y = 0.40
    raw_lms[8] = [0.5, tip_y, 0.0]
    t = 100.0

    tracker.update(
        hand_id=0,
        handedness="Right",
        palm_center=(0.5, 0.5, 0.0),
        hand_scale_ref=0.20,
        timestamp=t,
        raw_landmarks=raw_lms,
        finger_ratios={"index": 1.2},
    )

    # Downward tap strike (+Y velocity)
    t += 0.033
    tip_y += 0.025
    raw_lms[8] = [0.5, tip_y, 0.0]
    tracker.update(
        hand_id=0,
        handedness="Right",
        palm_center=(0.5, 0.5, 0.0),
        hand_scale_ref=0.20,
        timestamp=t,
        raw_landmarks=raw_lms,
        finger_ratios={"index": 1.1},
    )

    # Recovery / halt
    t += 0.060
    tip_y -= 0.008
    raw_lms[8] = [0.5, tip_y, 0.0]
    state = tracker.update(
        hand_id=0,
        handedness="Right",
        palm_center=(0.5, 0.5, 0.0),
        hand_scale_ref=0.20,
        timestamp=t,
        raw_landmarks=raw_lms,
        finger_ratios={"index": 1.15},
    )

    idx_motion = state.finger_motions["Index"]
    assert idx_motion.is_tapping
    assert idx_motion.motion_primitive == FingerMotionPrimitive.TAPPING


def test_dorsal_motion_tracking():
    """Verifies that movement performed with the back of the hand (DORSAL) is included and recognized."""
    tracker = MotionPrimitiveTracker()

    x = 0.60
    t = 100.0
    state = None
    for _ in range(6):
        x -= 0.015  # Moving left
        t += 0.033
        state = tracker.update(
            hand_id=0,
            handedness="Right",
            palm_center=(x, 0.5, 0.0),
            hand_scale_ref=0.20,
            timestamp=t,
            palm_facing="DORSAL",
        )

    assert state.is_dorsal
    assert state.palm_facing == "DORSAL"
    assert state.motion_primitive == MotionPrimitive.MOVE_LEFT
    assert state.primary_direction == "LEFT"
    assert "DORSAL" in state.summary()


def test_hand_axial_flip_to_dorsal_and_palm():
    """Verifies that rotating the palm around its axis generates FLIP_TO_DORSAL and FLIP_TO_PALM primitives."""
    tracker = MotionPrimitiveTracker()

    t = 100.0
    # 1. Hand starts facing PALM
    state = tracker.update(
        hand_id=0,
        handedness="Right",
        palm_center=(0.5, 0.5, 0.0),
        hand_scale_ref=0.20,
        timestamp=t,
        palm_facing="PALM",
        orientation_angles=(0.0, 0.0, 0.0),
    )
    assert not state.is_dorsal

    # 2. Hand pronates / flips to DORSAL with rapid roll angular velocity
    t += 0.033
    state = tracker.update(
        hand_id=0,
        handedness="Right",
        palm_center=(0.5, 0.5, 0.0),
        hand_scale_ref=0.20,
        timestamp=t,
        palm_facing="DORSAL",
        orientation_angles=(0.0, 0.0, 1.80),  # ~103 degrees roll
    )

    assert state.is_dorsal
    assert state.facing_flip == "FLIP_TO_DORSAL"
    assert state.motion_primitive == MotionPrimitive.FLIP_TO_DORSAL

    # 3. Hand supinates / flips back to PALM
    t += 0.050
    state = tracker.update(
        hand_id=0,
        handedness="Right",
        palm_center=(0.5, 0.5, 0.0),
        hand_scale_ref=0.20,
        timestamp=t,
        palm_facing="PALM",
        orientation_angles=(0.0, 0.0, 0.10),
    )

    assert not state.is_dorsal
    assert state.facing_flip == "FLIP_TO_PALM"
    assert state.motion_primitive == MotionPrimitive.FLIP_TO_PALM


