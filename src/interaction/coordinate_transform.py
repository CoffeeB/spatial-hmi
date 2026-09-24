"""
Explicit 5-stage coordinate transformation pipeline from monocular camera to 3D spatial world.
"""

from typing import Optional, Tuple
import numpy as np


class CoordinateTransformer:
    """
    Transforms 2D/3D camera observations into 3D world interaction coordinates.

    Pipeline:
        Stage 1: Camera pixel space (u_px, v_px) in [0, W] x [0, H]
        Stage 2: Normalized camera image space (u_norm, v_norm) in [0, 1]^2
        Stage 3: Viewport Normalized Device Coordinates (NDC) (x_ndc, y_ndc) in [-1, 1]^2
        Stage 4: Ray-casting direction in 3D camera space
        Stage 5: Ray-Sphere intersection in 3D world space
    """

    def __init__(
        self,
        camera_fov_deg: float = 60.0,
        aspect_ratio: float = 16.0 / 9.0,
        mirror_horizontal: bool = True,
    ):
        self.camera_fov_deg = camera_fov_deg
        self.aspect_ratio = aspect_ratio
        self.mirror_horizontal = mirror_horizontal
        self.fov_rad = np.radians(camera_fov_deg)

    def camera_to_normalized(
        self, u_px: float, v_px: float, image_width: float, image_height: float
    ) -> Tuple[float, float]:
        """
        Stage 1 -> Stage 2: Converts pixel coordinates to [0, 1] normalized coordinates.
        Applies horizontal mirroring for natural front-facing ergonomic interaction.
        """
        u_norm = float(u_px / image_width)
        v_norm = float(v_px / image_height)

        if self.mirror_horizontal:
            u_norm = 1.0 - u_norm

        return float(np.clip(u_norm, 0.0, 1.0)), float(np.clip(v_norm, 0.0, 1.0))

    def normalized_to_ndc(self, u_norm: float, v_norm: float) -> Tuple[float, float]:
        """
        Stage 2 -> Stage 3: Converts [0, 1]^2 normalized coordinates to [-1, 1]^2 NDC.
        x_ndc in [-1, 1] (left to right)
        y_ndc in [-1, 1] (bottom to top in WebGL)
        """
        x_ndc = 2.0 * u_norm - 1.0
        # Invert vertical axis because image coordinates originate at top-left
        y_ndc = 1.0 - 2.0 * v_norm
        return float(x_ndc), float(y_ndc)

    def ndc_to_ray(
        self, x_ndc: float, y_ndc: float, camera_position: Tuple[float, float, float] = (0.0, 0.0, 5.0)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stage 3 -> Stage 4: Unprojects 2D NDC coordinates into a 3D ray in world space.
        Returns: (ray_origin, ray_direction_unit_vector)
        """
        cam_pos = np.array(camera_position, dtype=np.float64)

        # Compute ray direction from pinhole perspective camera model
        tan_half_fov = np.tan(self.fov_rad / 2.0)
        dir_x = x_ndc * self.aspect_ratio * tan_half_fov
        dir_y = y_ndc * tan_half_fov
        dir_z = -1.0  # Forward into screen

        ray_dir = np.array([dir_x, dir_y, dir_z], dtype=np.float64)
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        return cam_pos, ray_dir

    def ray_sphere_intersect(
        self,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        sphere_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        sphere_radius: float = 2.0,
    ) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Stage 4 -> Stage 5: Intersects 3D ray with a sphere (the 3D globe).
        Solve ||ray_origin + t * ray_dir - center||^2 = radius^2

        Returns: (has_hit, 3D_hit_point, distance_t)
        """
        center = np.array(sphere_center, dtype=np.float64)
        oc = ray_origin - center

        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - (sphere_radius**2)

        discriminant = (b**2) - (4.0 * a * c)
        if discriminant < 0.0:
            # Ray misses sphere -> project onto tangent interaction plane at z = center[2]
            t_plane = (center[2] - ray_origin[2]) / (ray_dir[2] + 1e-6)
            plane_hit = ray_origin + t_plane * ray_dir
            return False, plane_hit, float(t_plane)

        # Nearest positive root
        t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)

        t = t1 if t1 > 0.0 else t2
        if t <= 0.0:
            return False, None, 0.0

        hit_point = ray_origin + t * ray_dir
        return True, hit_point, float(t)
