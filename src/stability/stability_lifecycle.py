"""
Gestura Stability Engine — Lifecycle States and Movement Categories.

Defines the state transitions required before any finger, pose, motion,
or gesture can be considered stable, intentional, and eligible for execution.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MovementCategory(str, Enum):
    """
    Distinguishes natural human movement before gesture interpretation:
      - MICRO_ADJUSTMENT: Natural tremor, twitches, repositioning -> IGNORE (never produce commands)
      - TRANSITION: User moving toward another pose -> OBSERVE (do not trigger intermediate gestures)
      - INTENTIONAL: Stable completed action -> ALLOW (eligible for execution)
    """
    MICRO_ADJUSTMENT = "MICRO_ADJUSTMENT"
    TRANSITION = "TRANSITION"
    INTENTIONAL = "INTENTIONAL"


class StabilityLifecycleState(str, Enum):
    """
    Temporal lifecycle state for every finger, pose, motion, and gesture:
      STABLE -> POSSIBLE_CHANGE -> OBSERVING -> CONFIRMED_CHANGE -> NEW_STABLE
    """
    STABLE = "STABLE"
    POSSIBLE_CHANGE = "POSSIBLE_CHANGE"
    OBSERVING = "OBSERVING"
    CONFIRMED_CHANGE = "CONFIRMED_CHANGE"
    NEW_STABLE = "NEW_STABLE"


class PersistentFinger(BaseModel):
    """
    Maintains per-digit state memory across frames.
    """
    finger: str                           # "Thumb", "Index", etc.
    state: str                            # Current confirmed state (e.g. "extended")
    previous_state: str                   # Previous confirmed state
    confidence: float = 1.0               # Decayed confidence
    stability: float = 1.0                # Per-digit stability score [0.0, 1.0]
    age: int = 1                          # Number of frames in current confirmed state
    last_changed: float = 0.0             # Timestamp when confirmed state last changed
    lifecycle: StabilityLifecycleState = StabilityLifecycleState.STABLE
    candidate_state: Optional[str] = None # Candidate state accumulating evidence
    candidate_frames: int = 0             # Consecutive frames candidate has matched


class PersistentPose(BaseModel):
    """
    Maintains static hand pose memory across frames.
    """
    pose_id: str = "UNKNOWN"
    canonical_name: str = "NONE"
    previous_pose: str = "NONE"
    confidence: float = 0.0
    stability: float = 0.0
    age: int = 0
    last_changed: float = 0.0
    lifecycle: StabilityLifecycleState = StabilityLifecycleState.STABLE
    candidate_pose: Optional[str] = None
    candidate_frames: int = 0
    observation_timer_ms: float = 0.0
