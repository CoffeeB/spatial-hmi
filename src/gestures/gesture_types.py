"""
Gesture taxonomy and classification data structures.
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel


class GestureType(str, Enum):
    """Canonical gesture primitives supported by the spatial HMI system."""
    NONE = "NONE"
    OPEN_PALM = "OPEN_PALM"
    PINCH = "PINCH"
    POINT = "POINT"
    GRAB = "GRAB"
    RELEASE = "RELEASE"
    SPREAD = "SPREAD"  # Bimanual expansion
    CONTRACTION = "CONTRACTION"  # Bimanual contraction (zoom in all nodes / zoom)
    ROTATION = "ROTATION"  # Bimanual or angular rotation
    SWIPE_LEFT = "SWIPE_LEFT"  # Slap / swipe left
    SWIPE_RIGHT = "SWIPE_RIGHT"  # Slap / swipe right
    SWIPE_UP = "SWIPE_UP"  # Slap / swipe up
    SWIPE_DOWN = "SWIPE_DOWN"  # Slap / swipe down
    SPREAD_FINGERS = "SPREAD_FINGERS"  # Spreading fingers wide -> zoom in
    SQUEEZE_FINGERS = "SQUEEZE_FINGERS"  # Squeezing fingers together -> zoom out
    THUMBS_UP = "THUMBS_UP"  # Affirmative / confirm
    THUMBS_DOWN = "THUMBS_DOWN"  # Dismiss / decline
    PEACE = "PEACE"  # Two-finger V selection
    TWO_HAND_PINCH = "TWO_HAND_PINCH"  # Bimanual pinch
    OK_RING = "OK_RING"  # Thumb + index ring
    GUN = "GUN"  # Thumb up + index forward
    SHAKA = "SHAKA"  # Call-me / thumb + pinky


class RecognizedGesture(BaseModel):
    """Output structure of the gesture classifier."""
    gesture: GestureType
    confidence: float  # In [0, 1]
    hand_id: int
    handedness: str
    feature_contributions: Dict[str, float] = {}
    timestamp: float
