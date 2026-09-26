"""
Configuration schema for the Gestura Stability Engine.

Eliminates magic numbers by exposing all observation windows, hysteresis thresholds,
dead-zone radii, hold durations, and stability weights in a centralized structure.
"""

from pathlib import Path
from typing import Dict, Optional
import yaml
from pydantic import BaseModel, Field


class StabilityConfig(BaseModel):
    """Configuration settings for the Gestura Stability Engine."""

    # ── 1. Observation Window Buffers (in frames) ───────────────────────────
    finger_window: int = Field(default=3, description="Frames to observe finger state before confirmation")
    pose_window: int = Field(default=5, description="Frames to observe hand pose before confirmation")
    motion_window: int = Field(default=8, description="Frames for motion primitive and dead-zone evaluation")
    gesture_window: int = Field(default=12, description="Frames for complete gesture validation")
    two_hand_window: int = Field(default=16, description="Frames for two-hand bimanual stabilization")

    # ── 2. Spatial Palm Motion Dead Zone ────────────────────────────────────
    dead_zone_px: float = Field(default=18.0, description="Spatial dead-zone radius in pixels (at 720p base)")
    dead_zone_ndc: float = Field(default=0.025, description="Spatial dead-zone radius in normalized coordinates")
    dead_zone_scale_with_dref: bool = Field(default=True, description="Whether dead-zone radius scales with hand distance d_ref")
    dead_zone_smoothing: float = Field(default=0.85, description="Smoothing factor for gradual dead-zone exit")

    # ── 3. Pose Hysteresis Thresholds ───────────────────────────────────────
    enter_threshold: float = Field(default=0.88, description="Confidence required to enter a new stable pose")
    exit_threshold: float = Field(default=0.62, description="Confidence floor required to remain in current pose")
    confidence_decay_rate: float = Field(default=0.85, description="Rate of gradual confidence decay on uncertain frame")

    # ── 4. Intentional Hold Durations (milliseconds) ────────────────────────
    # The hold timer begins only AFTER the pose becomes stable
    point_hold_ms: float = Field(default=80.0, description="Hold dwell required for POINT gesture")
    pinch_hold_ms: float = Field(default=120.0, description="Hold dwell required for PINCH gesture")
    fist_hold_ms: float = Field(default=150.0, description="Hold dwell required for CLOSED FIST GRAB")
    open_palm_hold_ms: float = Field(default=100.0, description="Hold dwell required for OPEN PALM ANCHOR")
    default_hold_ms: float = Field(default=100.0, description="Default hold dwell for other static poses")

    # ── 5. Multi-Signal Stability Score Weights ─────────────────────────────
    # Stability = w_rec * Recognition + w_temp * Temporal + w_mot * Motion + w_pers * Persistence
    weight_recognition: float = Field(default=0.35, description="Weight of raw landmark & model confidence")
    weight_temporal: float = Field(default=0.25, description="Weight of temporal consistency across window")
    weight_motion: float = Field(default=0.20, description="Weight of motion consistency & jitter absence")
    weight_persistence: float = Field(default=0.20, description="Weight of confirmed pose age / stability duration")
    stability_threshold: float = Field(default=0.85, description="Minimum stability score required to execute commands")

    # ── 6. Micro-Adjustment & Jitter Filter ─────────────────────────────────
    tremor_speed_threshold: float = Field(default=0.18, description="Palm speed threshold below which tremor is filtered")
    tremor_max_displacement: float = Field(default=0.020, description="Net displacement ceiling for palm tremor in window")
    twitch_duration_ms: float = Field(default=70.0, description="Maximum duration of an incidental finger twitch")

    # ── 7. Pose Transition Detector ─────────────────────────────────────────
    transition_min_duration_ms: float = Field(default=60.0, description="Minimum duration of deliberate pose transition")
    transition_max_duration_ms: float = Field(default=350.0, description="Maximum duration before transition times out")

    # ── 8. Active Interaction State Locking ─────────────────────────────────
    lock_exit_threshold: float = Field(default=0.45, description="Lower threshold required to break an ACTIVE state lock")
    active_preservation_frames: int = Field(default=6, description="Grace frames preserved during temporary landmark drops")

    def get_hold_duration_for_pose(self, pose_name: str) -> float:
        """Returns the configured intentional hold duration in milliseconds for a pose."""
        p = str(pose_name).upper()
        if "POINT" in p:
            return self.point_hold_ms
        elif "PINCH" in p:
            return self.pinch_hold_ms
        elif "GRAB" in p or "FIST" in p:
            return self.fist_hold_ms
        elif "OPEN_PALM" in p or "SPREAD" in p:
            return self.open_palm_hold_ms
        return self.default_hold_ms

    @classmethod
    def from_yaml(cls, path: str) -> "StabilityConfig":
        """Loads StabilityConfig from a YAML file."""
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # If wrapped under 'stability:' key
        if "stability" in data and isinstance(data["stability"], dict):
            data = data["stability"]
        return cls(**data)
