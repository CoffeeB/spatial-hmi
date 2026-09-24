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
    SPREAD = "SPREAD"  # Bimanual expansion (zoom in)
    CONTRACTION = "CONTRACTION"  # Bimanual contraction (zoom out)
    ROTATION = "ROTATION"  # Bimanual or angular rotation


class RecognizedGesture(BaseModel):
    """Output structure of the gesture classifier."""
    gesture: GestureType
    confidence: float  # In [0, 1]
    hand_id: int
    handedness: str
    feature_contributions: Dict[str, float] = {}
    timestamp: float
