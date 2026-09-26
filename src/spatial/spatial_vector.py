"""
Gestura Spatial Representation — Continuous Vector Mathematics.

Provides Vector2D and Vector3D continuous vector primitives with continuous angle,
magnitude, projection, and angular distance calculations for spatial HCI.
Supports standard vector arithmetic operators (+, -, *, /).
"""

from dataclasses import dataclass
import math
from typing import Tuple, Union
import numpy as np


class CallableFloat(float):
    """Float that can also be called as a method: x or x()."""
    def __call__(self) -> float:
        return float(self)


@dataclass(frozen=True)
class Vector2D:
    """Continuous 2D spatial vector (e.g. screen space dx, dy)."""
    x: float
    y: float

    # Arithmetic operators
    def __add__(self, other: Union["Vector2D", float]) -> "Vector2D":
        if isinstance(other, Vector2D):
            return Vector2D(self.x + other.x, self.y + other.y)
        return Vector2D(self.x + other, self.y + other)

    def __sub__(self, other: Union["Vector2D", float]) -> "Vector2D":
        if isinstance(other, Vector2D):
            return Vector2D(self.x - other.x, self.y - other.y)
        return Vector2D(self.x - other, self.y - other)

    def __mul__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vector2D":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector2D":
        inv = 1.0 / scalar
        return Vector2D(self.x * inv, self.y * inv)

    def __neg__(self) -> "Vector2D":
        return Vector2D(-self.x, -self.y)

    @property
    def magnitude(self) -> CallableFloat:
        """Euclidean norm of vector (accessible as .magnitude or .magnitude())."""
        return CallableFloat(math.sqrt(self.x * self.x + self.y * self.y))

    @property
    def angle_rad(self) -> float:
        """
        Continuous angle in radians in range [-pi, pi].
        In screen coordinates (+X Right, +Y Down):
        - 0 rad points RIGHT (+X)
        - -pi/2 rad points TOP (-Y)
        - pi/2 rad points BOTTOM (+Y)
        - pi / -pi rad points LEFT (-X)
        """
        return math.atan2(self.y, self.x)

    @property
    def angle_deg(self) -> float:
        """Continuous angle in degrees in range [-180, 180]."""
        return math.degrees(self.angle_rad)

    @property
    def normalized(self) -> "Vector2D":
        """Returns unit vector (accessible as .normalized or .normalized())."""
        mag = float(self.magnitude)
        if mag < 1e-7:
            return Vector2D(0.0, 0.0)
        return Vector2D(self.x / mag, self.y / mag)

    def __call__(self) -> "Vector2D":
        """Allows vector properties like .normalized to be called if desired."""
        return self

    def dot(self, other: "Vector2D") -> float:
        """Scalar dot product."""
        return self.x * other.x + self.y * other.y

    def distance_to(self, other: "Vector2D") -> float:
        """Euclidean distance between endpoints."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def angular_distance_deg(self, other: "Vector2D") -> float:
        """Shortest angular distance in degrees between two vectors [0, 180]."""
        u1 = self.normalized
        u2 = other.normalized
        if float(u1.magnitude) < 1e-4 or float(u2.magnitude) < 1e-4:
            return 0.0
        dot_val = max(-1.0, min(1.0, u1.dot(u2)))
        return math.degrees(math.acos(dot_val))

    def angle_to_deg(self, other: "Vector2D") -> float:
        """Alias for angular_distance_deg."""
        return self.angular_distance_deg(other)

    def lerp(self, other: "Vector2D", alpha: float) -> "Vector2D":
        """Linear interpolation between two vectors."""
        clamped_alpha = max(0.0, min(1.0, alpha))
        return Vector2D(
            self.x + (other.x - self.x) * clamped_alpha,
            self.y + (other.y - self.y) * clamped_alpha,
        )

    def as_tuple(self) -> Tuple[float, float]:
        """Returns (x, y) tuple."""
        return (float(self.x), float(self.y))

    def as_array(self) -> np.ndarray:
        """Returns numpy array."""
        return np.array([self.x, self.y], dtype=np.float32)


@dataclass(frozen=True)
class Vector3D:
    """Continuous 3D spatial vector (e.g. camera space or hand-relative space dx, dy, dz)."""
    x: float
    y: float
    z: float

    # Arithmetic operators
    def __add__(self, other: Union["Vector3D", float]) -> "Vector3D":
        if isinstance(other, Vector3D):
            return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
        return Vector3D(self.x + other, self.y + other, self.z + other)

    def __sub__(self, other: Union["Vector3D", float]) -> "Vector3D":
        if isinstance(other, Vector3D):
            return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
        return Vector3D(self.x - other, self.y - other, self.z - other)

    def __mul__(self, scalar: float) -> "Vector3D":
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vector3D":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector3D":
        inv = 1.0 / scalar
        return Vector3D(self.x * inv, self.y * inv, self.z * inv)

    def __neg__(self) -> "Vector3D":
        return Vector3D(-self.x, -self.y, -self.z)

    @property
    def magnitude(self) -> CallableFloat:
        """Euclidean 3D norm (accessible as .magnitude or .magnitude())."""
        return CallableFloat(math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z))

    @property
    def magnitude_xy(self) -> float:
        """Planar 2D projection magnitude in screen plane."""
        return math.sqrt(self.x * self.x + self.y * self.y)

    @property
    def normalized(self) -> "Vector3D":
        """Returns unit vector (accessible as .normalized or .normalized())."""
        mag = float(self.magnitude)
        if mag < 1e-7:
            return Vector3D(0.0, 0.0, 0.0)
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)

    def __call__(self) -> "Vector3D":
        """Allows vector properties like .normalized to be called if desired."""
        return self

    @property
    def screen_vector(self) -> Vector2D:
        """Projects 3D vector onto the 2D screen/image plane (x, y)."""
        return Vector2D(self.x, self.y)

    @property
    def azimuth_deg(self) -> float:
        """Planar heading angle in degrees [-180, 180] in XY plane."""
        return math.degrees(math.atan2(self.y, self.x))

    @property
    def elevation_deg(self) -> float:
        """Elevation angle out of the XY plane in degrees [-90, 90] towards Z."""
        mag = float(self.magnitude)
        if mag < 1e-7:
            return 0.0
        return math.degrees(math.asin(max(-1.0, min(1.0, self.z / mag))))

    def dot(self, other: "Vector3D") -> float:
        """3D dot product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3D") -> "Vector3D":
        """3D cross product (self x other)."""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def distance_to(self, other: "Vector3D") -> float:
        """3D Euclidean distance."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def angular_distance_deg(self, other: "Vector3D") -> float:
        """Shortest angular distance in 3D space in degrees [0, 180]."""
        u1 = self.normalized
        u2 = other.normalized
        if float(u1.magnitude) < 1e-4 or float(u2.magnitude) < 1e-4:
            return 0.0
        dot_val = max(-1.0, min(1.0, u1.dot(u2)))
        return math.degrees(math.acos(dot_val))

    def angle_to_deg(self, other: "Vector3D") -> float:
        """Alias for angular_distance_deg."""
        return self.angular_distance_deg(other)

    def lerp(self, other: "Vector3D", alpha: float) -> "Vector3D":
        """Linear interpolation in 3D."""
        clamped_alpha = max(0.0, min(1.0, alpha))
        return Vector3D(
            self.x + (other.x - self.x) * clamped_alpha,
            self.y + (other.y - self.y) * clamped_alpha,
            self.z + (other.z - self.z) * clamped_alpha,
        )

    def project_onto_plane(self, plane_normal: "Vector3D") -> "Vector3D":
        """Projects this vector onto a plane defined by its unit normal."""
        n = plane_normal.normalized
        if float(n.magnitude) < 1e-4:
            return self
        proj = n.scale(self.dot(n))
        return Vector3D(self.x - proj.x, self.y - proj.y, self.z - proj.z)

    def scale(self, scalar: float) -> "Vector3D":
        """Scales vector by a constant."""
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def as_tuple(self) -> Tuple[float, float, float]:
        """Returns (x, y, z) tuple."""
        return (float(self.x), float(self.y), float(self.z))

    def as_array(self) -> np.ndarray:
        """Returns numpy array."""
        return np.array([self.x, self.y, self.z], dtype=np.float32)
