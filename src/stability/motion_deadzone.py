"""
Gestura Stability Engine — Motion Dead Zone.

Suppresses human hand tremor and sub-threshold palm jitter:
  - Ignores movements smaller than the dead-zone radius.
  - Never accumulates tremor jitter into stroke/swipe displacement distance.
  - Dead zone scales dynamically with camera distance (d_ref).
  - Smoothly exits the dead zone with continuous ramp-out instead of abrupt velocity snapping.
  - Applies to idle or hover states, bypassed during active manipulation.
"""

import math
from typing import Any, Dict, Optional, Tuple
from src.stability.stability_config import StabilityConfig


class MotionDeadZone:
    """
    Spatial palm dead-zone filter with distance-adaptive scaling and smooth exit transitions.
    """

    NOMINAL_D_REF = 0.20  # Nominal scale reference at ~60 cm from camera

    def __init__(self, config: Optional[StabilityConfig] = None):
        self.config = config or StabilityConfig()
        self._anchor_pos: Optional[Tuple[float, float, float]] = None
        self._last_raw_pos: Optional[Tuple[float, float, float]] = None
        self._in_dead_zone: bool = True
        self._effective_radius: float = self.config.dead_zone_ndc
        self._cumulative_filtered_displacement: float = 0.0

    def reset(self) -> None:
        """Resets anchor and history."""
        self._anchor_pos = None
        self._last_raw_pos = None
        self._in_dead_zone = True
        self._cumulative_filtered_displacement = 0.0

    def update(
        self,
        palm_pos: Tuple[float, float, float],
        d_ref: float = 0.20,
        is_active_manipulation: bool = False,
        dt: float = 0.033,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float, bool]:
        """
        Applies spatial dead-zone filtering to the raw palm position:

        Args:
            palm_pos: (x, y, z) in normalized camera coordinates.
            d_ref: Scale normalization reference (smaller = further from camera).
            is_active_manipulation: True when currently executing an active interaction (pinch drag, etc.).
            dt: Time delta in seconds.

        Returns:
            Tuple of:
              - filtered_pos (Tuple[float, float, float])
              - filtered_velocity (Tuple[float, float, float])
              - filtered_displacement (float, step displacement)
              - in_dead_zone (bool)
        """
        if self._anchor_pos is None:
            self._anchor_pos = palm_pos
            self._last_raw_pos = palm_pos
            self._in_dead_zone = True
            return palm_pos, (0.0, 0.0, 0.0), 0.0, True

        # Calculate distance-adaptive dead-zone radius:
        # Hands further away (smaller d_ref) occupy fewer pixels; dead zone in NDC expands slightly
        # to prevent pixel quantization noise from triggering rogue movement.
        base_radius = self.config.dead_zone_ndc
        if self.config.dead_zone_scale_with_dref and d_ref > 0.01:
            scale_factor = min(2.0, max(0.6, self.NOMINAL_D_REF / d_ref))
            effective_radius = base_radius * scale_factor
        else:
            effective_radius = base_radius

        self._effective_radius = effective_radius

        # Distance from current spatial anchor
        dx = palm_pos[0] - self._anchor_pos[0]
        dy = palm_pos[1] - self._anchor_pos[1]
        dz = palm_pos[2] - self._anchor_pos[2]
        dist_from_anchor = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Active manipulation bypasses idle dead zone
        if is_active_manipulation:
            self._in_dead_zone = False
            self._anchor_pos = palm_pos
            step_disp = math.sqrt(
                (palm_pos[0] - self._last_raw_pos[0]) ** 2
                + (palm_pos[1] - self._last_raw_pos[1]) ** 2
                + (palm_pos[2] - self._last_raw_pos[2]) ** 2
            )
            vel = (
                (palm_pos[0] - self._last_raw_pos[0]) / max(dt, 0.001),
                (palm_pos[1] - self._last_raw_pos[1]) / max(dt, 0.001),
                (palm_pos[2] - self._last_raw_pos[2]) / max(dt, 0.001),
            )
            self._last_raw_pos = palm_pos
            self._cumulative_filtered_displacement += step_disp
            return palm_pos, vel, step_disp, False

        # ── Dead-Zone Logic for Idle / Hover ─────────────────────────────────
        if dist_from_anchor <= effective_radius:
            # Inside dead zone: lock position to anchor, suppress velocity and displacement
            self._in_dead_zone = True
            self._last_raw_pos = palm_pos
            return self._anchor_pos, (0.0, 0.0, 0.0), 0.0, True

        else:
            # Exiting dead zone: smoothly ramp out rather than abruptly snapping!
            # Rather than jumping by full `dist_from_anchor`, we subtract the dead-zone radius
            # along the motion trajectory vector so velocity starts smoothly from near 0.
            self._in_dead_zone = False
            excess = dist_from_anchor - effective_radius
            ratio = excess / max(dist_from_anchor, 0.0001)

            # Smooth output position: anchor + excess vector
            filtered_x = self._anchor_pos[0] + dx * ratio
            filtered_y = self._anchor_pos[1] + dy * ratio
            filtered_z = self._anchor_pos[2] + dz * ratio
            filtered_pos = (filtered_x, filtered_y, filtered_z)

            # Step displacement outside dead zone
            prev_x, prev_y, prev_z = self._last_raw_pos
            step_disp = math.sqrt(
                (filtered_x - prev_x) ** 2
                + (filtered_y - prev_y) ** 2
                + (filtered_z - prev_z) ** 2
            )
            vel = (
                (filtered_x - prev_x) / max(dt, 0.001),
                (filtered_y - prev_y) / max(dt, 0.001),
                (filtered_z - prev_z) / max(dt, 0.001),
            )

            # Move anchor forward so dead zone follows continuous motion
            self._anchor_pos = (
                self._anchor_pos[0] * 0.30 + filtered_x * 0.70,
                self._anchor_pos[1] * 0.30 + filtered_y * 0.70,
                self._anchor_pos[2] * 0.30 + filtered_z * 0.70,
            )
            self._last_raw_pos = filtered_pos
            self._cumulative_filtered_displacement += step_disp

            return filtered_pos, vel, step_disp, False

    def get_deadzone_telemetry(self) -> Dict[str, Any]:
        """Returns dead-zone diagnostic metrics for developer HUD."""
        return {
            "in_dead_zone": self._in_dead_zone,
            "effective_radius": round(self._effective_radius, 4),
            "dead_zone_px": round(self.config.dead_zone_px, 1),
            "cumulative_displacement": round(self._cumulative_filtered_displacement, 4),
        }
