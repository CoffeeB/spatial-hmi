"""
Gestura Spatial Representation — Configuration Model.
"""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
import yaml


class SpatialConfig(BaseModel):
    """Configuration parameters for continuous 3D spatial representation and uncertainty model."""

    # Screen-space and 3D dead-zone thresholds
    dead_zone_radius_2d: float = Field(0.02, description="Normalized screen dead zone below which direction is CENTER")
    dead_zone_radius_3d: float = Field(0.025, description="Normalized 3D dead zone below which motion is STATIONARY")
    min_displacement: float = Field(0.015, description="Minimum accumulated displacement before directional label locks")

    # Soft directional sector classification
    num_sectors: int = Field(8, description="Number of radial angular sectors around circle (8 = 45 deg each)")
    sector_bandwidth_deg: float = Field(45.0, description="Angular kernel width for soft probability calculation")
    ambiguity_delta_threshold: float = Field(0.18, description="Top-2 sector probability delta below which direction is AMBIGUOUS")
    angular_hysteresis_deg: float = Field(12.0, description="Angular barrier required to exit an active sector into a neighboring one")
    vector_smoothing_alpha: float = Field(0.55, description="Exponential smoothing factor for continuous direction vectors")
    temporal_alpha_vector: float = Field(0.55, description="Smoothing alpha for continuous vectors")
    dead_zone_speed: float = Field(0.045, description="Scalar speed below which motion is STATIONARY")

    # 3D Depth motion thresholds
    depth_speed_threshold: float = Field(0.12, description="Speed threshold for toward/away camera motion")
    depth_dominance_ratio: float = Field(0.65, description="Ratio of depth velocity to planar velocity to dominate direction")

    # Hand-relative coordinate frame
    wrist_index: int = Field(0, description="MediaPipe landmark index for wrist origin")
    index_mcp_index: int = Field(5, description="Landmark index for index MCP joint")
    middle_mcp_index: int = Field(9, description="Landmark index for middle MCP joint")
    pinky_mcp_index: int = Field(17, description="Landmark index for pinky MCP joint")
    min_hand_scale_ref: float = Field(0.001, description="Minimum scale reference clamp")

    # Articulated finger geometry
    curl_max_flexion_deg: float = Field(155.0, description="Flexion angle corresponding to 100% curl")
    foreshortening_ratio_threshold: float = Field(0.60, description="Apparent 2D length compression ratio indicating foreshortening")
    foreshortening_z_threshold: float = Field(0.45, description="Z-component magnitude threshold for pointing towards/away from camera")

    # Uncertainty & Temporal reconstruction
    visibility_dropout_threshold: float = Field(0.40, description="Landmark visibility below which digit is UNCERTAIN")
    temporal_reconstruction_window: int = Field(10, description="Frames of memory for bridging temporary landmark ambiguity")
    decay_rate_on_uncertainty: float = Field(0.88, description="Confidence decay factor when recovering through ambiguous frames")

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "SpatialConfig":
        """Loads configuration from YAML file or returns default."""
        path = config_path or "configs/spatial_config.yaml"
        p = Path(path)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                spatial_data = data.get("spatial", data)
                return cls(**spatial_data)
            except Exception:
                return cls()
        return cls()
