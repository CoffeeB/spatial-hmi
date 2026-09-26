"""
Gestura Level 2 — Motion Primitive Tracker.
Maintains a rolling temporal trajectory buffer and computes velocity, acceleration,
direction, displacement, linearity, and rotational kinematics independently of static hand poses.
"""

from collections import deque
import math
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from src.motion.motion_primitive import (
    DynamicState,
    FingerMotionPrimitive,
    FingerMotionState,
    MotionPrimitive,
    MotionState,
    TrajectoryPoint,
)
from src.utils.logging_config import setup_logger

logger = setup_logger("motion_tracker")

FINGER_TIP_INDICES = {
    "Thumb": 4,
    "Index": 8,
    "Middle": 12,
    "Ring": 16,
    "Little": 20,
}


class MotionPrimitiveTracker:
    """
    Independent temporal kinematics engine for Level 2 Motion Primitives.
    Measures the physical motion of hands and individual fingers over time,
    with full dorsal (back of hand) inclusion and hand flip tracking.
    """

    def __init__(
        self,
        buffer_capacity: int = 30,
        velocity_alpha: float = 0.55,
        speed_still_threshold: float = 0.045,
        speed_move_threshold: float = 0.080,
        dwell_hold_ms: float = 400.0,
    ):
        self.buffer_capacity = buffer_capacity
        self.velocity_alpha = velocity_alpha
        self.speed_still_threshold = speed_still_threshold
        self.speed_move_threshold = speed_move_threshold
        self.dwell_hold_ms = dwell_hold_ms

        # Per-hand tracking state: hand_id -> state dict
        self._buffers: Dict[int, deque] = {}
        self._prev_velocities: Dict[int, np.ndarray] = {}
        self._prev_speeds: Dict[int, float] = {}
        self._stroke_starts: Dict[int, Optional[TrajectoryPoint]] = {}
        self._stroke_start_times: Dict[int, float] = {}
        self._stationary_start_times: Dict[int, Optional[float]] = {}
        self._last_states: Dict[int, MotionState] = {}

        # Dorsal inclusion & axial flip states: hand_id -> value
        self._prev_palm_facings: Dict[int, str] = {}
        self._facing_flip_states: Dict[int, str] = {}
        self._facing_flip_times: Dict[int, float] = {}
        self._prev_rolls: Dict[int, float] = {}

        # Individual finger tracking states: hand_id -> dict
        self._finger_buffers: Dict[int, Dict[str, deque]] = {}
        self._prev_finger_ext: Dict[int, Dict[str, float]] = {}
        self._prev_finger_vel: Dict[int, Dict[str, np.ndarray]] = {}
        self._finger_dwell_times: Dict[int, Dict[str, float]] = {}
        self._tap_states: Dict[int, Dict[str, Dict[str, float]]] = {}

    def reset(self):
        """Clears all tracking history."""
        self._buffers.clear()
        self._prev_velocities.clear()
        self._prev_speeds.clear()
        self._stroke_starts.clear()
        self._stroke_start_times.clear()
        self._stationary_start_times.clear()
        self._last_states.clear()
        self._prev_palm_facings.clear()
        self._facing_flip_states.clear()
        self._facing_flip_times.clear()
        self._prev_rolls.clear()
        self._finger_buffers.clear()
        self._prev_finger_ext.clear()
        self._prev_finger_vel.clear()
        self._finger_dwell_times.clear()
        self._tap_states.clear()

    def prune_missing_hands(self, active_hand_ids: Set[int]):
        """Cleans up internal buffers for hands that are no longer observed."""
        tracked_ids = list(self._buffers.keys())
        for hid in tracked_ids:
            if hid not in active_hand_ids:
                self._buffers.pop(hid, None)
                self._prev_velocities.pop(hid, None)
                self._prev_speeds.pop(hid, None)
                self._stroke_starts.pop(hid, None)
                self._stroke_start_times.pop(hid, None)
                self._stationary_start_times.pop(hid, None)
                self._last_states.pop(hid, None)
                self._prev_palm_facings.pop(hid, None)
                self._facing_flip_states.pop(hid, None)
                self._facing_flip_times.pop(hid, None)
                self._prev_rolls.pop(hid, None)
                self._finger_buffers.pop(hid, None)
                self._prev_finger_ext.pop(hid, None)
                self._prev_finger_vel.pop(hid, None)
                self._finger_dwell_times.pop(hid, None)
                self._tap_states.pop(hid, None)

    def update(
        self,
        hand_id: int,
        handedness: str,
        palm_center: Tuple[float, float, float],
        hand_scale_ref: float,
        timestamp: float,
        raw_landmarks: Optional[np.ndarray] = None,
        palm_facing: str = "PALM",
        orientation_angles: Optional[Tuple[float, float, float]] = None,
        finger_ratios: Optional[Dict[str, float]] = None,
    ) -> MotionState:
        """
        Updates temporal trajectory for a hand and derives all Level 2 kinematic properties.
        """
        px, py, pz = palm_center
        curr_pos = np.array([px, py, pz], dtype=np.float32)

        # Initialize per-hand buffers if first seen
        if hand_id not in self._buffers:
            self._buffers[hand_id] = deque(maxlen=self.buffer_capacity)
            self._prev_velocities[hand_id] = np.zeros(3, dtype=np.float32)
            self._prev_speeds[hand_id] = 0.0
            self._stroke_starts[hand_id] = None
            self._stroke_start_times[hand_id] = timestamp
            self._stationary_start_times[hand_id] = timestamp

        buf = self._buffers[hand_id]

        # 1. Delta time
        if len(buf) == 0:
            dt = 0.033
            prev_pt = None
        else:
            prev_pt = buf[-1]
            dt = max(1e-4, timestamp - prev_pt.timestamp)

        # 2. Velocity computation with exponential smoothing
        if prev_pt is None:
            raw_vel = np.zeros(3, dtype=np.float32)
            smoothed_vel = np.zeros(3, dtype=np.float32)
            scale_rate = 0.0
        else:
            raw_vel = (curr_pos - np.array([prev_pt.x, prev_pt.y, prev_pt.z], dtype=np.float32)) / dt
            prev_vel = self._prev_velocities[hand_id]
            smoothed_vel = self.velocity_alpha * raw_vel + (1.0 - self.velocity_alpha) * prev_vel
            scale_rate = float((hand_scale_ref - prev_pt.scale_ref) / dt)

        self._prev_velocities[hand_id] = smoothed_vel

        vx, vy, vz = float(smoothed_vel[0]), float(smoothed_vel[1]), float(smoothed_vel[2])
        speed = float(np.linalg.norm(smoothed_vel))
        speed_xy = float(math.hypot(vx, vy))

        # 3. Acceleration & Speed derivative
        prev_spd = self._prev_speeds.get(hand_id, speed)
        prev_vel_vec = self._prev_velocities.get(hand_id, smoothed_vel)
        if prev_pt is not None and dt > 1e-4:
            accel_vec = (smoothed_vel - prev_vel_vec) / dt
            tangential_accel = float((speed - prev_spd) / dt)
        else:
            accel_vec = np.zeros(3, dtype=np.float32)
            tangential_accel = 0.0

        ax, ay, az = float(accel_vec[0]), float(accel_vec[1]), float(accel_vec[2])
        accel_mag = float(np.linalg.norm(accel_vec))
        self._prev_speeds[hand_id] = speed

        # 4. Record new TrajectoryPoint
        new_pt = TrajectoryPoint(
            timestamp=timestamp,
            x=px,
            y=py,
            z=pz,
            scale_ref=hand_scale_ref,
            vx=vx,
            vy=vy,
            vz=vz,
            speed=speed,
        )
        buf.append(new_pt)

        # 5. Stationarity, Hold & Release Lifecycle
        # Effective speed incorporates planar velocity and depth scaling speed
        depth_speed = abs(scale_rate) * 0.85
        effective_speed = max(speed, depth_speed)

        is_still = effective_speed < self.speed_still_threshold
        is_moving = effective_speed >= self.speed_move_threshold

        is_releasing = False
        if prev_spd >= 0.18 and (speed < 0.15 or speed < 0.40 * prev_spd) and tangential_accel < -0.80:
            is_releasing = True

        if is_still:
            if self._stationary_start_times[hand_id] is None:
                self._stationary_start_times[hand_id] = timestamp
            dwell_ms = (timestamp - self._stationary_start_times[hand_id]) * 1000.0
            is_holding = dwell_ms >= self.dwell_hold_ms
            stroke_ms = 0.0
            self._stroke_starts[hand_id] = None
            dynamic_state = DynamicState.STATIONARY
        else:
            self._stationary_start_times[hand_id] = None
            dwell_ms = 0.0
            is_holding = False

            if self._stroke_starts[hand_id] is None:
                self._stroke_starts[hand_id] = prev_pt or new_pt
                self._stroke_start_times[hand_id] = prev_pt.timestamp if prev_pt else timestamp

            stroke_ms = (timestamp - self._stroke_start_times[hand_id]) * 1000.0

            if tangential_accel > 0.30:
                dynamic_state = DynamicState.ACCELERATING
            elif tangential_accel < -0.30:
                dynamic_state = DynamicState.DECELERATING
            else:
                dynamic_state = DynamicState.STEADY

        # 6. Direction & Heading Analysis
        # MediaPipe: +X Right, +Y Down, +Z Away from camera
        # Heading angle in standard Cartesian XY plane (+Y is Up, +X is Right)
        if speed_xy > 1e-4:
            heading_deg = float(math.degrees(math.atan2(-vy, vx)))
            ux = vx / speed
            uy = vy / speed
            uz = vz / speed
        else:
            heading_deg = 0.0
            ux, uy, uz = 0.0, 0.0, 0.0

        # Depth direction classification
        if scale_rate > 0.16 or vz < -0.18:
            depth_state = "TOWARD"
        elif scale_rate < -0.16 or vz > 0.18:
            depth_state = "AWAY"
        else:
            depth_state = "NEUTRAL"

        # Determine primary and secondary Cartesian directions
        abs_vx, abs_vy = abs(vx), abs(vy)
        primary_dir = "STATIONARY"
        secondary_dir = None

        if is_moving or (not is_still and effective_speed >= self.speed_still_threshold):
            # Check if depth motion dominates planar motion
            if depth_state != "NEUTRAL" and (abs(scale_rate) > (speed_xy * 0.7) or speed_xy < self.speed_still_threshold):
                primary_dir = depth_state
                if abs_vx >= self.speed_still_threshold:
                    secondary_dir = "RIGHT" if vx > 0 else "LEFT"
                elif abs_vy >= self.speed_still_threshold:
                    secondary_dir = "DOWN" if vy > 0 else "UP"
            else:
                # Planar dominance
                if abs_vx >= abs_vy:
                    primary_dir = "RIGHT" if vx > 0 else "LEFT"
                    if abs_vy >= 0.35 * abs_vx and abs_vy >= self.speed_still_threshold:
                        secondary_dir = "DOWN" if vy > 0 else "UP"
                else:
                    primary_dir = "DOWN" if vy > 0 else "UP"
                    if abs_vx >= 0.35 * abs_vy and abs_vx >= self.speed_still_threshold:
                        secondary_dir = "RIGHT" if vx > 0 else "LEFT"

        # 7. Distance & Trajectory Geometry over current stroke / buffer
        pts_list = list(buf)
        stroke_start = self._stroke_starts[hand_id] or (buf[0] if len(buf) > 0 else new_pt)
        dx = px - stroke_start.x
        dy = py - stroke_start.y
        dz = pz - stroke_start.z
        disp_mag = float(math.sqrt(dx * dx + dy * dy + dz * dz))

        # Accumulate path length over the active stroke
        stroke_idx = 0
        if self._stroke_starts[hand_id] is not None:
            for idx, pt in enumerate(pts_list):
                if pt.timestamp >= self._stroke_start_times[hand_id]:
                    stroke_idx = max(0, idx - 1)
                    break

        path_len = 0.0
        for i in range(stroke_idx + 1, len(pts_list)):
            p_prev = pts_list[i - 1]
            p_curr = pts_list[i]
            step_d = math.sqrt(
                (p_curr.x - p_prev.x) ** 2 +
                (p_curr.y - p_prev.y) ** 2 +
                (p_curr.z - p_prev.z) ** 2
            )
            path_len += step_d

        linearity = float(min(1.0, disp_mag / max(path_len, 1e-4))) if path_len > 0.005 else 1.0

        # 8b. Rotational & Orbital Kinematics (detecting clockwise / counterclockwise loops)
        ang_vel, cum_angle_deg, rot_dir = self._compute_orbital_rotation(pts_list)

        # 8c. Dorsal Inclusion & Axial Roll Flip Kinematics
        prev_facing = self._prev_palm_facings.get(hand_id, palm_facing)
        roll_vel = 0.0
        if orientation_angles is not None:
            curr_roll = float(orientation_angles[2]) * (180.0 / math.pi)
            if hand_id in self._prev_rolls:
                dt_roll = max(1e-4, dt) if dt > 0 else 0.033
                d_roll = curr_roll - self._prev_rolls[hand_id]
                # Normalize wrap around [-180, 180]
                while d_roll > 180.0:
                    d_roll -= 360.0
                while d_roll < -180.0:
                    d_roll += 360.0
                roll_vel = float(d_roll / dt_roll)
            self._prev_rolls[hand_id] = curr_roll

        # Detect axial flip between PALM and DORSAL
        if prev_facing != palm_facing:
            if prev_facing == "PALM" and palm_facing == "DORSAL":
                self._facing_flip_states[hand_id] = "FLIP_TO_DORSAL"
                self._facing_flip_times[hand_id] = timestamp
            elif prev_facing == "DORSAL" and palm_facing == "PALM":
                self._facing_flip_states[hand_id] = "FLIP_TO_PALM"
                self._facing_flip_times[hand_id] = timestamp
        self._prev_palm_facings[hand_id] = palm_facing

        last_flip = self._facing_flip_states.get(hand_id, "STABLE")
        last_flip_t = self._facing_flip_times.get(hand_id, 0.0)
        facing_flip = last_flip if (timestamp - last_flip_t <= 0.40) else "STABLE"
        is_dorsal = bool(palm_facing == "DORSAL")

        # 9. Classify Discrete Level 2 Motion Primitive
        if is_releasing:
            primitive = MotionPrimitive.RELEASE
            confidence = 0.85
        elif is_holding:
            primitive = MotionPrimitive.HOLD
            confidence = float(min(1.0, 0.70 + 0.30 * (dwell_ms / 1000.0)))
        elif facing_flip in ("FLIP_TO_DORSAL", "FLIP_TO_PALM") and (abs(roll_vel) >= 60.0 or speed < 0.25):
            primitive = MotionPrimitive.FLIP_TO_DORSAL if facing_flip == "FLIP_TO_DORSAL" else MotionPrimitive.FLIP_TO_PALM
            confidence = float(min(1.0, 0.75 + 0.25 * (abs(roll_vel) / 180.0)))
        elif is_still:
            primitive = MotionPrimitive.STATIONARY
            confidence = float(min(1.0, 1.0 - (speed / self.speed_still_threshold)))
        elif rot_dir in ("CLOCKWISE", "COUNTERCLOCKWISE") and abs(cum_angle_deg) >= 150.0 and linearity < 0.70:
            primitive = MotionPrimitive.ROTATE_CW if rot_dir == "CLOCKWISE" else MotionPrimitive.ROTATE_CCW
            confidence = float(min(1.0, 0.65 + 0.35 * (abs(cum_angle_deg) / 360.0)))
        else:
            # Linear translation primitives
            if primary_dir == "LEFT":
                primitive = MotionPrimitive.MOVE_LEFT
            elif primary_dir == "RIGHT":
                primitive = MotionPrimitive.MOVE_RIGHT
            elif primary_dir == "UP":
                primitive = MotionPrimitive.MOVE_UP
            elif primary_dir == "DOWN":
                primitive = MotionPrimitive.MOVE_DOWN
            elif primary_dir == "TOWARD":
                primitive = MotionPrimitive.MOVE_TOWARD
            elif primary_dir == "AWAY":
                primitive = MotionPrimitive.MOVE_AWAY
            else:
                primitive = MotionPrimitive.STATIONARY

            confidence = float(min(1.0, 0.50 + 0.50 * (speed / 0.30)))

        # 10. Individual Finger Motion Kinematics
        finger_motions: Dict[str, FingerMotionState] = {}
        if hand_id not in self._finger_buffers:
            self._finger_buffers[hand_id] = {d: deque(maxlen=20) for d in FINGER_TIP_INDICES}
            self._prev_finger_ext[hand_id] = {}
            self._prev_finger_vel[hand_id] = {}
            self._finger_dwell_times[hand_id] = {d: timestamp for d in FINGER_TIP_INDICES}
            self._tap_states[hand_id] = {}

        if raw_landmarks is not None and len(raw_landmarks) >= 21:
            for f_name, tip_idx in FINGER_TIP_INDICES.items():
                tip_pt = raw_landmarks[tip_idx]
                f_buf = self._finger_buffers[hand_id][f_name]
                f_buf.append((float(tip_pt[0]), float(tip_pt[1]), float(tip_pt[2]), timestamp))

                # Finite difference tip velocity
                if len(f_buf) >= 2:
                    p_prev = f_buf[-2]
                    dt_f = max(1e-4, timestamp - p_prev[3])
                    raw_tip_v = np.array([
                        (tip_pt[0] - p_prev[0]) / dt_f,
                        (tip_pt[1] - p_prev[1]) / dt_f,
                        (tip_pt[2] - p_prev[2]) / dt_f,
                    ], dtype=np.float32)
                    prev_tip_v = self._prev_finger_vel[hand_id].get(f_name, np.zeros(3, dtype=np.float32))
                    tip_v = self.velocity_alpha * raw_tip_v + (1.0 - self.velocity_alpha) * prev_tip_v
                else:
                    tip_v = np.zeros(3, dtype=np.float32)
                self._prev_finger_vel[hand_id][f_name] = tip_v

                tip_speed = float(np.linalg.norm(tip_v))
                # Relative tip velocity (subtracts palm center velocity to isolate finger motion)
                rel_v = tip_v - np.array([vx, vy, vz], dtype=np.float32)
                rel_speed = float(np.linalg.norm(rel_v))

                # Extension rate d(ext)/dt
                ext_rate = 0.0
                if finger_ratios and f_name.lower() in finger_ratios:
                    curr_ext = finger_ratios[f_name.lower()]
                    prev_ext = self._prev_finger_ext[hand_id].get(f_name, curr_ext)
                    dt_ext = max(1e-4, dt) if dt > 0 else 0.033
                    ext_rate = float((curr_ext - prev_ext) / dt_ext)
                    self._prev_finger_ext[hand_id][f_name] = curr_ext

                # Dynamic state of finger
                if rel_speed < 0.045 and abs(ext_rate) < 0.15:
                    f_dyn = DynamicState.STATIONARY
                elif ext_rate > 0.25 or rel_speed > 0.25:
                    f_dyn = DynamicState.ACCELERATING
                elif ext_rate < -0.25:
                    f_dyn = DynamicState.DECELERATING
                else:
                    f_dyn = DynamicState.STEADY

                # Tapping detection (Index & Thumb rapid downward or forward strike followed by halt/rebound)
                is_tapping = False
                tap_info = self._tap_states[hand_id].get(f_name)
                if f_name in ("Index", "Thumb"):
                    if rel_v[1] > 0.18 or rel_v[2] < -0.15:
                        self._tap_states[hand_id][f_name] = {"t_strike": timestamp, "y_peak": float(tip_pt[1])}
                    elif tap_info is not None:
                        t_strike = tap_info["t_strike"]
                        delta_tap = timestamp - t_strike
                        if 0.03 <= delta_tap <= 0.35 and (rel_v[1] < 0.15 or rel_speed < 0.12):
                            is_tapping = True
                            if delta_tap > 0.22:
                                self._tap_states[hand_id].pop(f_name, None)
                        elif delta_tap > 0.38:
                            self._tap_states[hand_id].pop(f_name, None)

                is_ext = ext_rate > 0.22
                is_flex = ext_rate < -0.22

                if is_tapping:
                    f_prim = FingerMotionPrimitive.TAPPING
                elif is_ext:
                    f_prim = FingerMotionPrimitive.EXTENDING
                elif is_flex:
                    f_prim = FingerMotionPrimitive.FLEXING
                elif rel_speed > 0.16:
                    f_prim = FingerMotionPrimitive.SWIPING
                elif rel_speed < 0.04:
                    dwell_f = (timestamp - self._finger_dwell_times[hand_id].get(f_name, timestamp)) * 1000.0
                    if dwell_f >= 300.0:
                        f_prim = FingerMotionPrimitive.HOLD
                    else:
                        f_prim = FingerMotionPrimitive.STATIONARY
                else:
                    self._finger_dwell_times[hand_id][f_name] = timestamp
                    f_prim = FingerMotionPrimitive.STATIONARY

                f_traj = [(p[0], p[1], p[2], p[3]) for p in list(f_buf)[-10:]]

                finger_motions[f_name] = FingerMotionState(
                    name=f_name,
                    tip_position=(float(tip_pt[0]), float(tip_pt[1]), float(tip_pt[2])),
                    tip_velocity=(float(tip_v[0]), float(tip_v[1]), float(tip_v[2])),
                    relative_velocity=(float(rel_v[0]), float(rel_v[1]), float(rel_v[2])),
                    tip_speed=tip_speed,
                    relative_speed=rel_speed,
                    extension_rate=ext_rate,
                    dynamic_state=f_dyn,
                    motion_primitive=f_prim,
                    is_tapping=is_tapping,
                    is_extending=is_ext,
                    is_flexing=is_flex,
                    is_holding=(f_prim == FingerMotionPrimitive.HOLD),
                    trajectory=f_traj,
                )

        # Format rolling trajectory coordinates for telemetry transmission [(x, y, z, t), ...]
        recent_traj = [(p.x, p.y, p.z, p.timestamp) for p in pts_list[-20:]]

        state = MotionState(
            hand_id=hand_id,
            handedness=handedness,
            position=(px, py, pz),
            velocity=(vx, vy, vz),
            speed=speed,
            speed_xy=speed_xy,
            acceleration=(ax, ay, az),
            acceleration_magnitude=accel_mag,
            tangential_acceleration=tangential_accel,
            dynamic_state=dynamic_state,
            direction_vector=(ux, uy, uz),
            heading_deg=heading_deg,
            primary_direction=primary_dir,
            secondary_direction=secondary_dir,
            net_displacement=(dx, dy, dz),
            displacement_magnitude=disp_mag,
            cumulative_path_length=path_len,
            linearity=linearity,
            scale_rate=scale_rate,
            depth_state=depth_state,
            angular_velocity=ang_vel,
            cumulative_angle_deg=cum_angle_deg,
            rotation_direction=rot_dir,
            stroke_duration_ms=stroke_ms,
            dwell_duration_ms=dwell_ms,
            is_holding=is_holding,
            is_releasing=is_releasing,
            motion_primitive=primitive,
            palm_facing=palm_facing,
            is_dorsal=is_dorsal,
            facing_flip=facing_flip,
            roll_velocity=roll_vel,
            finger_motions=finger_motions,
            confidence=confidence,
            timestamp=timestamp,
            trajectory=recent_traj,
        )

        self._last_states[hand_id] = state
        return state

    def _compute_orbital_rotation(
        self, pts: List[TrajectoryPoint]
    ) -> Tuple[float, float, str]:
        """
        Computes angular winding number around the trajectory centroid to detect
        clockwise or counter-clockwise orbital motions in the camera plane.
        """
        if len(pts) < 8:
            return 0.0, 0.0, "NONE"

        # Trajectory centroid in XY
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        radii = [math.hypot(x - cx, y - cy) for x, y in zip(xs, ys)]
        mean_r = sum(radii) / len(radii)

        # Orbit radius must be meaningful
        if mean_r < 0.025:
            return 0.0, 0.0, "NONE"

        # Check radial variance to ensure it's orbiting around a center, not a straight line
        var_r = sum((r - mean_r) ** 2 for r in radii) / len(radii)
        std_r = math.sqrt(var_r)
        if (std_r / mean_r) > 0.55:
            return 0.0, 0.0, "NONE"

        # Compute cumulative angular delta
        angles = [math.atan2(p.y - cy, p.x - cx) for p in pts]
        cum_angle = 0.0
        for i in range(1, len(angles)):
            da = angles[i] - angles[i - 1]
            # Wrap to [-pi, pi]
            while da > math.pi:
                da -= 2 * math.pi
            while da < -math.pi:
                da += 2 * math.pi
            cum_angle += da

        cum_angle_deg = math.degrees(cum_angle)
        dt_total = max(1e-3, pts[-1].timestamp - pts[0].timestamp)
        ang_vel = cum_angle_deg / dt_total

        # In screen coords (+X Right, +Y Down):
        # Positive angle step is Clockwise, negative is Counter-Clockwise
        if cum_angle_deg >= 100.0:
            rot_dir = "CLOCKWISE"
        elif cum_angle_deg <= -100.0:
            rot_dir = "COUNTERCLOCKWISE"
        else:
            rot_dir = "NONE"

        return ang_vel, cum_angle_deg, rot_dir
