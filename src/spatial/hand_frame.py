"""
Gestura Spatial Representation — Hand-Relative Orthonormal 3D Coordinate System.

Establishes an anatomical coordinate frame attached to the hand:
  - Origin: Wrist (Landmark 0) or Palm Centroid
  - Y Axis (Longitudinal): Wrist -> Middle MCP (Landmark 9)
  - Z Axis (Normal): Palmar normal pointing out of volar surface
  - X Axis (Lateral): Radial axis pointing toward index/thumb side

Transforms 3D points and vectors between:
  1. Camera coordinates (viewpoint dependent)
  2. Hand-relative coordinates (viewpoint invariant)
  3. Screen coordinates (2D projection)
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple
import numpy as np

from src.spatial.spatial_config import SpatialConfig
from src.spatial.spatial_vector import Vector2D, Vector3D


@dataclass
class HandCoordinateFrame:
    """
    Viewpoint-invariant hand-relative orthonormal basis.
    Rotates and translates with the physical hand in 3D space.
    """
    origin: Vector3D                 # Origin position in camera coordinates
    axis_x: Vector3D                 # Lateral (radial) unit axis
    axis_y: Vector3D                 # Longitudinal (distal) unit axis
    axis_z: Vector3D                 # Palmar normal unit vector (volar direction)
    scale_ref: float                 # Characteristic anatomical hand length d_ref
    handedness: str                  # "Right" or "Left"
    palm_facing: str                 # "PALM", "DORSAL", or "SIDE"
    rotation_matrix: np.ndarray      # 3x3 orthonormal rotation matrix [x, y, z]

    # Continuous orientation angles (radians and degrees)
    pitch_deg: float = 0.0           # Tilt forward/backward
    yaw_deg: float = 0.0             # Turn left/right
    roll_deg: float = 0.0            # Pronation/supination roll around longitudinal axis

    def camera_to_hand_point(self, pt_cam: Tuple[float, float, float]) -> Vector3D:
        """
        Transforms a 3D point from camera coordinates to normalized hand-relative space.
        p_hand = R^T * (p_cam - origin) / scale_ref
        """
        p = np.array(pt_cam, dtype=np.float32) - self.origin.as_array()
        # Project onto orthonormal basis
        local_coords = p @ self.rotation_matrix / max(self.scale_ref, 1e-4)
        return Vector3D(float(local_coords[0]), float(local_coords[1]), float(local_coords[2]))

    def camera_to_hand_vector(self, vec_cam: Tuple[float, float, float]) -> Vector3D:
        """
        Transforms a pure direction or velocity vector from camera coordinates to hand-relative space.
        v_hand = R^T * v_cam (normalized to preserve directional magnitude)
        """
        v = np.array(vec_cam, dtype=np.float32)
        local_v = v @ self.rotation_matrix
        return Vector3D(float(local_v[0]), float(local_v[1]), float(local_v[2]))

    def hand_to_camera_point(self, pt_hand: Tuple[float, float, float]) -> Vector3D:
        """Transforms a normalized hand-relative point back into camera space."""
        p_local = np.array(pt_hand, dtype=np.float32) * self.scale_ref
        p_cam = (p_local @ self.rotation_matrix.T) + self.origin.as_array()
        return Vector3D(float(p_cam[0]), float(p_cam[1]), float(p_cam[2]))

    def hand_to_camera_vector(self, vec_hand: Tuple[float, float, float]) -> Vector3D:
        """Transforms a hand-relative direction vector back into camera space."""
        v_local = np.array(vec_hand, dtype=np.float32)
        v_cam = v_local @ self.rotation_matrix.T
        return Vector3D(float(v_cam[0]), float(v_cam[1]), float(v_cam[2]))

    def get_telemetry(self) -> Dict[str, any]:
        """Telemetry representation for visualizer."""
        return {
            "origin": self.origin.as_tuple(),
            "axis_x": self.axis_x.as_tuple(),
            "axis_y": self.axis_y.as_tuple(),
            "axis_z": self.axis_z.as_tuple(),
            "scale_ref": round(float(self.scale_ref), 4),
            "palm_facing": self.palm_facing,
            "pitch_deg": round(float(self.pitch_deg), 1),
            "yaw_deg": round(float(self.yaw_deg), 1),
            "roll_deg": round(float(self.roll_deg), 1),
        }


class HandFrameEstimator:
    """
    Derives the anatomical HandCoordinateFrame from 21 3D landmarks.
    Works robustly across all hand orientations:
      - Palm facing camera (front)
      - Palm facing away (dorsal / back of hand)
      - Hand rotated 45°, 90°, upside down, sideways
      - Hand pointing toward/away from the camera
    """

    def __init__(self, config: Optional[SpatialConfig] = None):
        self.config = config or SpatialConfig()

    def estimate_frame(
        self,
        raw_landmarks: Optional[np.ndarray] = None,
        handedness: str = "Right",
        scale_ref: Optional[float] = None,
        landmarks: Optional[np.ndarray] = None,
    ) -> HandCoordinateFrame:
        """
        Derives orthonormal frame from anatomical landmarks:
          p0: Wrist (0)
          p5: Index MCP (5)
          p9: Middle MCP (9)
          p17: Pinky MCP (17)
        """
        lms = raw_landmarks if raw_landmarks is not None else landmarks
        if lms is None or len(lms) < 21:
            # Fallback identity frame
            origin = Vector3D(0.5, 0.5, 0.0)
            return HandCoordinateFrame(
                origin=origin,
                axis_x=Vector3D(1.0, 0.0, 0.0),
                axis_y=Vector3D(0.0, -1.0, 0.0),
                axis_z=Vector3D(0.0, 0.0, -1.0),
                scale_ref=1.0,
                handedness=handedness,
                palm_facing="PALM",
                rotation_matrix=np.eye(3, dtype=np.float32),
            )

        p0 = lms[self.config.wrist_index]
        p5 = lms[self.config.index_mcp_index]
        p9 = lms[self.config.middle_mcp_index]
        p17 = lms[self.config.pinky_mcp_index]

        origin_vec = Vector3D(float(p0[0]), float(p0[1]), float(p0[2]))

        # Characteristic anatomical hand length d_ref = ||p9 - p0||
        if scale_ref is None:
            scale_ref = float(np.linalg.norm(p9 - p0))
        scale_ref = max(scale_ref, self.config.min_hand_scale_ref)

        # 1. Longitudinal Y axis: points from Wrist (0) toward Middle MCP (9)
        v_y_raw = p9 - p0
        norm_y = np.linalg.norm(v_y_raw)
        v_y = (v_y_raw / norm_y) if norm_y > 1e-6 else np.array([0.0, -1.0, 0.0], dtype=np.float32)

        # 2. Palmar Normal Z axis:
        # Cross product of metacarpal arch (Index MCP -> Pinky MCP) with longitudinal axis (Wrist -> Middle MCP)
        # In MediaPipe coordinates (+X Right, +Y Down, +Z Away):
        if handedness == "Left":
            n_palm = np.cross(p5 - p17, p9 - p0)
        else:
            n_palm = np.cross(p17 - p5, p9 - p0)

        norm_n = np.linalg.norm(n_palm)
        if norm_n > 1e-6:
            v_z = n_palm / norm_n
        else:
            v_z = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # Classify facing state relative to camera view
        # Camera looks down +Z into scene.
        # v_z[2] < -0.18 => Normal points toward camera => PALM (front view)
        # v_z[2] > 0.18  => Normal points away from camera => DORSAL (back of hand)
        # otherwise      => SIDE (edge-on / knife-edge)
        if v_z[2] < -0.18:
            palm_facing = "PALM"
        elif v_z[2] > 0.18:
            palm_facing = "DORSAL"
        else:
            palm_facing = "SIDE"

        # 3. Lateral X axis: orthogonal to Y and Z, pointing toward radial / thumb side
        v_x_raw = np.cross(v_y, v_z)
        norm_x = np.linalg.norm(v_x_raw)
        v_x = (v_x_raw / norm_x) if norm_x > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Re-orthonormalize v_y to guarantee exact orthogonality: v_y = v_z x v_x
        v_y = np.cross(v_z, v_x)
        v_y = v_y / max(np.linalg.norm(v_y), 1e-6)

        # Build orthonormal column matrix: R = [v_x, v_y, v_z]
        R = np.column_stack([v_x, v_y, v_z]).astype(np.float32)

        # Compute continuous Euler angles (pitch, yaw, roll) in degrees
        pitch = float(math.degrees(math.asin(np.clip(-v_y[2], -1.0, 1.0))))
        yaw = float(math.degrees(math.atan2(v_y[0], v_y[1])))
        roll = float(math.degrees(math.atan2(v_x[2], v_z[2])))

        return HandCoordinateFrame(
            origin=origin_vec,
            axis_x=Vector3D(float(v_x[0]), float(v_x[1]), float(v_x[2])),
            axis_y=Vector3D(float(v_y[0]), float(v_y[1]), float(v_y[2])),
            axis_z=Vector3D(float(v_z[0]), float(v_z[1]), float(v_z[2])),
            scale_ref=scale_ref,
            handedness=handedness,
            palm_facing=palm_facing,
            rotation_matrix=R,
            pitch_deg=pitch,
            yaw_deg=yaw,
            roll_deg=roll,
        )
