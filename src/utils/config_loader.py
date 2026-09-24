"""
Configuration loader with Pydantic schema validation.
"""

from pathlib import Path
from typing import List, Literal, Optional
import yaml
from pydantic import BaseModel, Field


class CameraConfig(BaseModel):
    device_index: int = 0
    width: int = 640
    height: int = 480
    target_fps: int = 30
    auto_reconnect: bool = True
    reconnect_delay_sec: float = 1.0


class PerceptionConfig(BaseModel):
    max_num_hands: int = 2
    min_detection_confidence: float = 0.65
    min_tracking_confidence: float = 0.60
    model_complexity: int = 1


class LandmarksConfig(BaseModel):
    palm_landmark_indices: List[int] = Field(default_factory=lambda: [0, 5, 9, 17])
    wrist_index: int = 0
    middle_mcp_index: int = 9
    velocity_alpha: float = 0.40


class GesturesConfig(BaseModel):
    pinch_threshold: float = 0.38
    pinch_steepness: float = 18.0
    point_extension_ratio: float = 1.18
    open_palm_extension_ratio: float = 1.12
    grab_closure_ratio: float = 0.90
    two_hand_spread_vel_threshold: float = 0.04
    two_hand_rotation_vel_threshold: float = 0.06
    min_confidence_threshold: float = 0.40


class IntentConfig(BaseModel):
    evidence_lambda: float = 0.55
    activation_threshold: float = 0.50
    release_threshold: float = 0.25
    confirm_frames: int = 2
    candidate_frames: int = 1
    hand_loss_timeout_sec: float = 0.25


class OneEuroConfig(BaseModel):
    fc_min: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0


class SmoothingConfig(BaseModel):
    filter_type: Literal["one_euro", "ema", "none"] = "one_euro"
    one_euro: OneEuroConfig = Field(default_factory=OneEuroConfig)
    dead_zone_radius: float = 0.003
    velocity_gate: float = 0.001


class InteractionConfig(BaseModel):
    rotation_sensitivity: float = 2.8
    zoom_sensitivity: float = 1.8
    translation_sensitivity: float = 2.2
    screen_aspect_ratio: float = 16.0 / 9.0


class CommunicationConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    broadcast_interval_ms: float = 16.66


class HMIConfig(BaseModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    landmarks: LandmarksConfig = Field(default_factory=LandmarksConfig)
    gestures: GesturesConfig = Field(default_factory=GesturesConfig)
    intent: IntentConfig = Field(default_factory=IntentConfig)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    interaction: InteractionConfig = Field(default_factory=InteractionConfig)
    communication: CommunicationConfig = Field(default_factory=CommunicationConfig)


def load_config(config_path: Optional[str] = None) -> HMIConfig:
    """Loads configuration from YAML file or returns default configuration."""
    if not config_path:
        default_path = Path(__file__).parent.parent.parent / "configs" / "default_config.yaml"
        if default_path.exists():
            config_path = str(default_path)
        else:
            return HMIConfig()

    path = Path(config_path)
    if not path.exists():
        return HMIConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}

    return HMIConfig(**raw_data)
