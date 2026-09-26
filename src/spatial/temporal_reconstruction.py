"""
Gestura Spatial Representation — Temporal Reconstruction Engine.

Bridges temporary occlusions and single-frame visual ambiguities without state collapse:
  "If Frames N-10 -> N-1 are EXTENDED and Frame N is UNCERTAIN, maintain EXTENDED with
   temporarily reduced confidence rather than switching states."

Maintains temporal continuity across:
  - Per-digit finger states and directions.
  - Hand poses and orientations.
  - Spatial motion vectors.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from src.spatial.finger_geometry import ArticulatedFingerGeometry, ContinuousFingerState
from src.spatial.spatial_config import SpatialConfig


@dataclass
class DigitHistoryEntry:
    """Historical record for a single digit at a frame."""
    timestamp: float
    state: ContinuousFingerState
    confidence: float
    curl_ratio: float


class TemporalReconstructionEngine:
    """
    Combines multi-frame temporal evidence with spatial uncertainty to bridge
    temporary occlusions, camera-facing foreshortening blurs, and rapid transitions.
    """

    def __init__(self, config: Optional[SpatialConfig] = None, window_size: int = 15, max_gap_frames: int = 8):
        self.config = config or SpatialConfig()
        self.window_size = window_size
        self.max_gap_frames = max_gap_frames

        # Per-digit history buffers: digit_name -> deque of DigitHistoryEntry
        self._digit_histories: Dict[str, deque] = {
            "thumb": deque(maxlen=window_size),
            "index": deque(maxlen=window_size),
            "middle": deque(maxlen=window_size),
            "ring": deque(maxlen=window_size),
            "little": deque(maxlen=window_size),
        }

        # Consecutive uncertain frame counters: digit_name -> count
        self._uncertain_frame_counts: Dict[str, int] = {
            "thumb": 0, "index": 0, "middle": 0, "ring": 0, "little": 0
        }

    def reset(self):
        """Clears all temporal reconstruction buffers."""
        for q in self._digit_histories.values():
            q.clear()
        for k in self._uncertain_frame_counts:
            self._uncertain_frame_counts[k] = 0

    def reconstruct_finger(
        self,
        geom: ArticulatedFingerGeometry,
        timestamp: float,
    ) -> ArticulatedFingerGeometry:
        """
        Applies temporal reconstruction to a digit observation.
        If current frame is UNCERTAIN/UNKNOWN but preceding history was STABLE,
        preserves the stable prior state with gracefully decayed confidence.
        """
        name = geom.finger_name
        history = self._digit_histories[name]
        is_ambiguous = (geom.state in (ContinuousFingerState.UNKNOWN, ContinuousFingerState.TRANSITION)) or (geom.confidence < 0.45)

        if not is_ambiguous:
            # Current frame is clear and stable
            self._uncertain_frame_counts[name] = 0
            history.append(DigitHistoryEntry(
                timestamp=timestamp,
                state=geom.state,
                confidence=geom.confidence,
                curl_ratio=geom.curl_ratio,
            ))
            return geom

        # Current frame is ambiguous / occluded
        self._uncertain_frame_counts[name] += 1
        gap_count = self._uncertain_frame_counts[name]

        # Check if we have sufficient prior evidence to reconstruct
        if len(history) >= 3 and gap_count <= self.max_gap_frames:
            # Check if recent history was consistent
            recent_states = [entry.state for entry in list(history)[-5:]]
            most_common_state = max(set(recent_states), key=recent_states.count)
            consistency = recent_states.count(most_common_state) / len(recent_states)

            if consistency >= 0.60 and most_common_state != ContinuousFingerState.UNKNOWN:
                # Reconstruct state from temporal evidence with decayed confidence
                decay_factor = float(0.85 ** gap_count)
                mean_prior_conf = float(np.mean([e.confidence for e in history]))
                reconstructed_conf = mean_prior_conf * decay_factor

                geom.notes.append(
                    f"Temporal reconstruction active: maintained {most_common_state.value} "
                    f"across {gap_count} ambiguous frames (conf decayed to {reconstructed_conf:.2f})"
                )

                # Return reconstructed geometry preserving stable state
                geom.state = most_common_state
                geom.confidence = reconstructed_conf
                return geom

        # Gap is too long or history is insufficient: acknowledge uncertainty
        history.append(DigitHistoryEntry(
            timestamp=timestamp,
            state=geom.state,
            confidence=geom.confidence,
            curl_ratio=geom.curl_ratio,
        ))
        return geom

    def reconstruct_all_fingers(
        self,
        finger_geometries: Dict[str, ArticulatedFingerGeometry],
        timestamp: float,
    ) -> Dict[str, ArticulatedFingerGeometry]:
        """Applies temporal reconstruction across all 5 digits."""
        reconstructed = {}
        for digit_name, geom in finger_geometries.items():
            reconstructed[digit_name] = self.reconstruct_finger(geom, timestamp)
        return reconstructed
