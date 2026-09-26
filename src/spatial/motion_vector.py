"""
Gestura Spatial Representation — Continuous 3D Motion Kinematics & Tracking.

Replaces naive 4-way discrete motion categorization with full continuous 3D kinematics:
  1. Full 3D kinematic derivatives:
       - 3D Position: p(t) = (x, y, z)
       - 3D Velocity: v(t) = (vx, vy, vz)
       - 3D Acceleration: a(t) = (ax, ay, az)
       - 3D Jerk: j(t) = (jx, jy, jz)
  2. Planar vs Optical Depth Kinematics:
       - 3D Speed: ||v||
       - Planar Speed: ||(vx, vy)||
       - Depth Speed: vz (incorporating landmark z and anatomical scale expansion/contraction rate)
       - Depth Direction: TOWARD, AWAY, NEUTRAL
  3. Continuous Angular Dynamics:
       - Heading angle: planar azimuth in [-180°, 180°]
       - Elevation angle: optical axis tilt in [-90°, 90°]
       - Angular velocity: direction change rate in rad/s
  4. Integration with soft spatial direction probability distribution.
"""

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.spatial.spatial_config import SpatialConfig
from src.spatial.spatial_direction import SpatialDirectionClassifier, SpatialSector
from src.spatial.spatial_vector import Vector2D, Vector3D


@dataclass
class ContinuousMotionKinematics:
    """Complete continuous 3D motion kinematics for a tracked hand or digit."""
    position: Vector3D
    velocity: Vector3D
    acceleration: Vector3D
    jerk: Vector3D

    # Scalar speeds
    speed_3d: float
    speed_xy: float
    speed_z: float

    # Continuous directional vectors
    direction_3d: Vector3D               # Normalized 3D unit vector
    direction_2d: Vector2D               # Normalized 2D unit vector on image plane
    heading_deg: float                   # Planar heading [-180, 180]
    elevation_deg: float                 # Angle relative to image plane [-90, 90]
    angular_velocity_rad_s: float        # Heading change rate

    # Trajectory & stroke geometry
    displacement: Vector3D               # Net displacement from stroke start
    displacement_magnitude: float
    path_length: float
    duration_sec: float

    # Depth kinematics
    depth_state: str                     # "TOWARD", "AWAY", "NEUTRAL"
    scale_rate: float                    # Rate of change of anatomical scale d_ref

    # Soft spatial interpretation (derived from continuous measurements)
    spatial_distribution: Dict[str, float]
    primary_direction: str
    secondary_direction: Optional[str]
    ambiguity_level: str
    is_ambiguous: bool
    confidence: float

    def to_dict(self) -> Dict[str, any]:
        """Telemetry representation for visualizer and diagnostics."""
        return {
            "position": self.position.as_tuple(),
            "velocity": self.velocity.as_tuple(),
            "acceleration": self.acceleration.as_tuple(),
            "jerk": self.jerk.as_tuple(),
            "speed": {
                "speed_3d": round(float(self.speed_3d), 4),
                "speed_xy": round(float(self.speed_xy), 4),
                "speed_z": round(float(self.speed_z), 4),
            },
            "angles": {
                "heading_deg": round(float(self.heading_deg), 1),
                "elevation_deg": round(float(self.elevation_deg), 1),
                "angular_vel_rad_s": round(float(self.angular_velocity_rad_s), 2),
            },
            "displacement": {
                "vector": self.displacement.as_tuple(),
                "magnitude": round(float(self.displacement_magnitude), 4),
                "path_length": round(float(self.path_length), 4),
                "duration_sec": round(float(self.duration_sec), 3),
            },
            "depth": {
                "state": self.depth_state,
                "scale_rate": round(float(self.scale_rate), 4),
            },
            "spatial_classification": {
                "primary": self.primary_direction,
                "secondary": self.secondary_direction,
                "ambiguity": self.ambiguity_level,
                "is_ambiguous": self.is_ambiguous,
                "confidence": round(float(self.confidence), 3),
                "distribution": {k: round(v, 3) for k, v in self.spatial_distribution.items()},
            },
        }


class ContinuousMotionTracker:
    """
    Tracks continuous 3D motion over a rolling temporal window.
    Derives velocity, acceleration, jerk, angular velocity, and soft spatial sectors.
    """

    def __init__(self, config: Optional[SpatialConfig] = None, buffer_capacity: int = 30):
        self.config = config or SpatialConfig()
        self.buffer_capacity = buffer_capacity

        # Rolling history of (timestamp, Vector3D, scale_ref)
        self._history: deque = deque(maxlen=buffer_capacity)

        # Kinematic states for derivative estimation
        self._prev_velocity = Vector3D(0.0, 0.0, 0.0)
        self._prev_acceleration = Vector3D(0.0, 0.0, 0.0)
        self._prev_heading_rad: float = 0.0
        self._stroke_start_pos: Optional[Vector3D] = None
        self._stroke_start_time: float = 0.0
        self._accumulated_path: float = 0.0

        # Spatial direction classifier
        self._dir_classifier = SpatialDirectionClassifier(self.config)

    def reset(self):
        """Clears all motion history."""
        self._history.clear()
        self._prev_velocity = Vector3D(0.0, 0.0, 0.0)
        self._prev_acceleration = Vector3D(0.0, 0.0, 0.0)
        self._prev_heading_rad = 0.0
        self._stroke_start_pos = None
        self._stroke_start_time = 0.0
        self._accumulated_path = 0.0
        self._dir_classifier.reset()

    def update(
        self,
        current_pos: Tuple[float, float, float],
        timestamp: float,
        scale_ref: float,
    ) -> ContinuousMotionKinematics:
        """
        Updates motion state with a new 3D observation and computes continuous kinematics.
        """
        pos = Vector3D(current_pos[0], current_pos[1], current_pos[2])

        if not self._history:
            dt = 0.033
            prev_pos = pos
            prev_scale = scale_ref
            self._stroke_start_pos = pos
            self._stroke_start_time = timestamp
        else:
            prev_time, prev_pos, prev_scale = self._history[-1]
            dt = max(1e-4, timestamp - prev_time)

        self._history.append((timestamp, pos, scale_ref))

        # 1. 3D Velocity (smoothed)
        raw_vel = (pos - prev_pos) * (1.0 / dt)
        alpha = self.config.temporal_alpha_vector
        smooth_vel = (raw_vel * alpha) + (self._prev_velocity * (1.0 - alpha))

        # 2. 3D Acceleration (derivative of velocity)
        raw_accel = (smooth_vel - self._prev_velocity) * (1.0 / dt)
        smooth_accel = (raw_accel * alpha) + (self._prev_acceleration * (1.0 - alpha))

        # 3. 3D Jerk (derivative of acceleration)
        raw_jerk = (smooth_accel - self._prev_acceleration) * (1.0 / dt)

        self._prev_velocity = smooth_vel
        self._prev_acceleration = smooth_accel

        # 4. Speeds
        speed_3d = smooth_vel.magnitude()
        speed_xy = math.hypot(smooth_vel.x, smooth_vel.y)
        scale_rate = (scale_ref - prev_scale) / dt

        # Depth speed incorporates both landmark z velocity and scale expansion rate
        # When moving towards camera: scale_rate > 0 and vz < 0 (MediaPipe +Z away)
        speed_z = abs(smooth_vel.z) + abs(scale_rate) * 0.85

        # 5. Continuous Direction Vectors
        dir_3d = smooth_vel.normalized()
        dir_2d = Vector2D(smooth_vel.x, smooth_vel.y).normalized()

        # 6. Continuous Angles
        if speed_xy > 1e-4:
            heading_deg = math.degrees(math.atan2(smooth_vel.y, smooth_vel.x))
            heading_rad = math.atan2(smooth_vel.y, smooth_vel.x)
        else:
            heading_deg = 0.0
            heading_rad = 0.0

        # Elevation angle relative to image plane: [-90°, 90°]
        # In MediaPipe: +Z is away from camera
        if speed_3d > 1e-4:
            elevation_deg = math.degrees(math.asin(np.clip(-smooth_vel.z / speed_3d, -1.0, 1.0)))
        else:
            elevation_deg = 0.0

        # Angular velocity
        d_heading = (heading_rad - self._prev_heading_rad + math.pi) % (2 * math.pi) - math.pi
        angular_vel = abs(d_heading) / dt
        self._prev_heading_rad = heading_rad

        # 7. Stroke & Path tracking
        step_dist = (pos - prev_pos).magnitude()
        self._accumulated_path += step_dist

        # Check for stroke reset on stationarity
        is_stationary = speed_3d < self.config.dead_zone_speed
        if is_stationary:
            self._stroke_start_pos = pos
            self._stroke_start_time = timestamp
            self._accumulated_path = 0.0

        stroke_start = self._stroke_start_pos or pos
        displacement = pos - stroke_start
        disp_mag = displacement.magnitude()
        duration_sec = max(0.0, timestamp - self._stroke_start_time)

        # 8. Depth Direction Classification
        # MediaPipe: -Z or scale expansion = TOWARD camera; +Z or scale shrinkage = AWAY from camera
        vz_effective = -smooth_vel.z + (scale_rate * 1.5)
        if vz_effective > self.config.depth_speed_threshold:
            depth_state = "TOWARD"
        elif vz_effective < -self.config.depth_speed_threshold:
            depth_state = "AWAY"
        else:
            depth_state = "NEUTRAL"

        # 9. Spatial Direction Classification (derived soft probability distribution)
        spatial_res = self._dir_classifier.classify_direction(
            raw_direction=dir_2d,
            magnitude=speed_xy,
            dt=dt,
        )

        # Override primary direction if depth motion significantly dominates planar motion
        primary_dir = spatial_res["primary_direction"]
        secondary_dir = spatial_res["secondary_direction"]

        if depth_state != "NEUTRAL" and (speed_z > speed_xy * 1.4 or speed_xy < self.config.dead_zone_speed):
            secondary_dir = primary_dir if primary_dir not in (SpatialSector.CENTER.value, SpatialSector.UNKNOWN.value) else None
            primary_dir = depth_state

        return ContinuousMotionKinematics(
            position=pos,
            velocity=smooth_vel,
            acceleration=smooth_accel,
            jerk=raw_jerk,
            speed_3d=speed_3d,
            speed_xy=speed_xy,
            speed_z=speed_z,
            direction_3d=dir_3d,
            direction_2d=dir_2d,
            heading_deg=heading_deg,
            elevation_deg=elevation_deg,
            angular_velocity_rad_s=angular_vel,
            displacement=displacement,
            displacement_magnitude=disp_mag,
            path_length=self._accumulated_path,
            duration_sec=duration_sec,
            depth_state=depth_state,
            scale_rate=scale_rate,
            spatial_distribution=spatial_res["sector_probabilities"],
            primary_direction=primary_dir,
            secondary_direction=secondary_dir,
            ambiguity_level=spatial_res["ambiguity_level"],
            is_ambiguous=spatial_res["is_ambiguous"],
            confidence=spatial_res["confidence"],
        )
