"""
Temporal Intent Engine Configuration for Gestura.

Defines all temporal observation windows, hysteresis thresholds, motion filters,
intent lock policies, and priority weights. No magic numbers inside perception code.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ObservationWindowsConfig(BaseModel):
    """Temporal observation window frame counts for each perception layer."""
    finger_window: int = Field(default=3, description="Frames for finger state stabilization (1–3)")
    pose_window: int = Field(default=5, description="Frames for hand pose confirmation (3–5)")
    motion_window: int = Field(default=8, description="Frames for motion primitive measurement (5–10)")
    gesture_window: int = Field(default=12, description="Frames for complete gesture validation (8–15)")
    two_hand_window: int = Field(default=16, description="Frames for bimanual synchronization (10–20)")


class FingerStabilizationConfig(BaseModel):
    """Parameters for Level 0 finger state smoothing and filtering."""
    smoothing_alpha: float = Field(default=0.65, description="EMA smoothing alpha for landmark coordinates")
    min_confidence: float = Field(default=0.80, description="Minimum confidence for valid finger state")
    temporal_majority_threshold: float = Field(default=0.66, description="Fraction of window required for state flip")
    jitter_deadzone_ndc: float = Field(default=0.0035, description="Coordinate jitter deadband in NDC")


class PoseConfirmationConfig(BaseModel):
    """Hysteresis thresholds for Level 1 static hand pose confirmation."""
    enter_threshold: float = Field(default=0.80, description="Confidence threshold to enter a new pose candidate")
    exit_threshold: float = Field(default=0.45, description="Confidence threshold to drop / exit an active pose")
    min_confirm_frames: int = Field(default=4, description="Consecutive frames in pose before CONFIRMED")
    max_oscillation_history: int = Field(default=10, description="History buffer for detecting rapid pose oscillation")


class MotionAnalyzerConfig(BaseModel):
    """Parameters for Level 2 motion primitive kinematic evaluation."""
    min_velocity_threshold: float = Field(default=0.20, description="Minimum scalar velocity (u/s) for active motion")
    stationary_velocity_threshold: float = Field(default=0.08, description="Velocity below which motion is STATIONARY")
    direction_consistency_threshold: float = Field(default=0.82, description="Cosine consistency required for linear strokes")
    min_displacement: float = Field(default=0.035, description="Minimum net displacement for directional strokes")
    min_duration_ms: float = Field(default=80.0, description="Minimum duration for intentional stroke (ms)")
    max_duration_ms: float = Field(default=380.0, description="Maximum duration before stroke is considered drift (ms)")
    curvature_reject_threshold: float = Field(default=0.68, description="Minimum linearity ratio (disp/path_length)")


class SwipeValidationConfig(BaseModel):
    """Strict validation thresholds for directional swipe strokes."""
    min_velocity: float = Field(default=0.22, description="Peak velocity threshold (m/s or u/s)")
    min_distance: float = Field(default=0.040, description="Minimum net stroke displacement")
    min_direction_consistency: float = Field(default=0.85, description="Trajectory vector alignment ratio")
    min_duration_ms: float = Field(default=100.0, description="Minimum temporal duration (ms)")
    max_duration_ms: float = Field(default=350.0, description="Maximum temporal duration (ms)")
    min_linearity: float = Field(default=0.70, description="Linearity ratio ||d|| / L")
    refractory_period_ms: float = Field(default=220.0, description="Cooldown refractory period after swipe (ms)")


class IntentStateMachineConfig(BaseModel):
    """Parameters for the 6-state Intent State Machine lifecycle."""
    activation_threshold: float = Field(default=0.50, description="Evidence score to transition CANDIDATE -> CONFIRMED")
    release_threshold: float = Field(default=0.25, description="Evidence score below which state falls to RELEASING")
    evidence_lambda: float = Field(default=0.75, description="Exponential decay weight for temporal evidence filter")
    candidate_frames: int = Field(default=2, description="Consecutive frames required before promoting to CANDIDATE")
    confirm_frames: int = Field(default=4, description="Consecutive frames required before promoting to CONFIRMED")
    hand_loss_timeout_sec: float = Field(default=0.80, description="Grace timeout before hand absence drops interaction")
    release_cooldown_frames: int = Field(default=5, description="Frames held in RELEASING cooldown before IDLE")
    easing_decay_rate: float = Field(default=0.85, description="Decay multiplier per frame during RELEASING interpolation")


class IntentLockConfig(BaseModel):
    """Policy for exclusive interaction locking."""
    lock_on_pinch: bool = Field(default=True, description="Automatically lock intent upon confirmed pinch interaction")
    lock_on_drag: bool = Field(default=True, description="Lock intent during active spatial dragging")
    suppress_secondary_gestures: bool = Field(default=True, description="Suppress swipes/hovers while locked")
    timeout_sec: float = Field(default=3.0, description="Safety timeout after which lock auto-releases if inactive")


class PriorityHierarchyConfig(BaseModel):
    """Hierarchy priorities (Rank 1 = highest priority)."""
    priority_order: List[str] = Field(
        default_factory=lambda: [
            "PINCH_SELECTION",     # Priority 1: Pinch on selected node / precision pinch
            "TWO_HAND_MANIPULATION", # Priority 2: Two-hand bimanual scale/rotate
            "SWIPE_GESTURES",      # Priority 3: Directional swipe strokes
            "POINT_HOVER",         # Priority 4: Pointing & cursor targeting
            "IDLE",                # Priority 5: Idle / resting hand
        ]
    )


class TemporalIntentConfig(BaseModel):
    """Central configuration for Gestura's Temporal Intent Engine."""
    windows: ObservationWindowsConfig = Field(default_factory=ObservationWindowsConfig)
    finger: FingerStabilizationConfig = Field(default_factory=FingerStabilizationConfig)
    pose: PoseConfirmationConfig = Field(default_factory=PoseConfirmationConfig)
    motion: MotionAnalyzerConfig = Field(default_factory=MotionAnalyzerConfig)
    swipe: SwipeValidationConfig = Field(default_factory=SwipeValidationConfig)
    intent_fsm: IntentStateMachineConfig = Field(default_factory=IntentStateMachineConfig)
    lock: IntentLockConfig = Field(default_factory=IntentLockConfig)
    priority: PriorityHierarchyConfig = Field(default_factory=PriorityHierarchyConfig)
