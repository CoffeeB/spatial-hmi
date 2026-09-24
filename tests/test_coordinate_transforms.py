"""
Unit tests for coordinate transformation pipeline and ray-sphere intersection math.
"""

import numpy as np
import pytest

from src.interaction.coordinate_transform import CoordinateTransformer


def test_camera_to_normalized_with_mirror():
    transformer = CoordinateTransformer(mirror_horizontal=True)
    # Center pixel (320, 240) in (640, 480)
    u_norm, v_norm = transformer.camera_to_normalized(320, 240, 640, 480)
    assert np.isclose(u_norm, 0.5)
    assert np.isclose(v_norm, 0.5)

    # Left edge pixel (0, 240) -> mirrored to right edge (1.0)
    u_left, _ = transformer.camera_to_normalized(0, 240, 640, 480)
    assert np.isclose(u_left, 1.0)


def test_normalized_to_ndc():
    transformer = CoordinateTransformer()
    # Center (0.5, 0.5) -> (0.0, 0.0) NDC
    nx, ny = transformer.normalized_to_ndc(0.5, 0.5)
    assert np.isclose(nx, 0.0)
    assert np.isclose(ny, 0.0)

    # Top-Left (0.0, 0.0) -> (-1.0, 1.0) NDC
    nx_tl, ny_tl = transformer.normalized_to_ndc(0.0, 0.0)
    assert np.isclose(nx_tl, -1.0)
    assert np.isclose(ny_tl, 1.0)


def test_ndc_to_ray_and_sphere_intersection():
    transformer = CoordinateTransformer(camera_fov_deg=60.0, aspect_ratio=1.0)

    # Center ray at (0, 0)
    origin, direction = transformer.ndc_to_ray(0.0, 0.0, camera_position=(0, 0, 5))
    assert np.allclose(origin, [0, 0, 5])
    assert np.allclose(direction, [0, 0, -1])

    # Ray cast toward sphere at origin (0, 0, 0) with radius 2.0
    hit, hit_point, t = transformer.ray_sphere_intersect(origin, direction, sphere_center=(0, 0, 0), sphere_radius=2.0)
    assert hit is True
    assert np.isclose(t, 3.0)  # 5 - 2 = 3
    assert np.allclose(hit_point, [0, 0, 2.0])
