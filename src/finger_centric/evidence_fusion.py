"""
Finger-Centric Perception Layer — Evidence Fusion.

Adaptively weights palm, finger, motion, and temporal evidence based on
the current viewpoint quality. Selects the active perception representation.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple
import numpy as np

from src.finger_centric.viewpoint_classifier import ViewpointMode, ViewpointState


class ActiveRepresentation(str, Enum):
    """Which evidence source is currently dominant."""
    HAND_CENTRIC   = "HAND_CENTRIC"    # Palm / hand-level geometry
    FINGER_CENTRIC = "FINGER_CENTRIC"  # Finger chain / tip geometry
    COMBINED       = "COMBINED"        # Both sources weighted together
    TEMPORAL       = "TEMPORAL"        # Relying on ghost / historical data
    UNCERTAIN      = "UNCERTAIN"       # Insufficient evidence


@dataclass
class EvidenceWeights:
    """Normalised contribution of each evidence source."""
    palm_weight:     float   # Palm / hand-level geometry
    finger_weight:   float   # Finger chain geometry
    motion_weight:   float   # Velocity / acceleration evidence
    temporal_weight: float   # Historical / ghost evidence


@dataclass
class FusedPerceptionState:
    """Output of the evidence fusion layer."""
    active_representation: ActiveRepresentation
    weights: EvidenceWeights
    combined_confidence: float        # Weighted aggregate confidence [0,1]
    hand_geometry_confidence: float
    finger_geometry_confidence: float
    motion_confidence: float
    temporal_confidence: float
    viewpoint_mode: ViewpointMode
    fusion_rationale: str


# -------------------------------------------------------------------
# Base weight presets per viewpoint.  These are BEFORE confidence
# modulation.  Actual weights are scaled by the observed confidence
# of each source and then renormalised to sum to 1.
# -------------------------------------------------------------------
_PRESETS: Dict[ViewpointMode, EvidenceWeights] = {
    ViewpointMode.NORMAL: EvidenceWeights(
        palm_weight=0.40, finger_weight=0.35, motion_weight=0.15, temporal_weight=0.10
    ),
    ViewpointMode.FORESHORTENED: EvidenceWeights(
        palm_weight=0.20, finger_weight=0.48, motion_weight=0.22, temporal_weight=0.10
    ),
    ViewpointMode.CAMERA_FACING: EvidenceWeights(
        palm_weight=0.08, finger_weight=0.45, motion_weight=0.27, temporal_weight=0.20
    ),
    ViewpointMode.CAMERA_AWAY: EvidenceWeights(
        palm_weight=0.28, finger_weight=0.42, motion_weight=0.20, temporal_weight=0.10
    ),
    ViewpointMode.SIDE_VIEW: EvidenceWeights(
        palm_weight=0.12, finger_weight=0.52, motion_weight=0.26, temporal_weight=0.10
    ),
    ViewpointMode.OCCLUDED: EvidenceWeights(
        palm_weight=0.08, finger_weight=0.22, motion_weight=0.25, temporal_weight=0.45
    ),
    ViewpointMode.UNCERTAIN: EvidenceWeights(
        palm_weight=0.25, finger_weight=0.25, motion_weight=0.25, temporal_weight=0.25
    ),
}


class EvidenceFusion:
    """
    Combines palm, finger, motion, and temporal evidence into a single
    fused perception state.

    Weights are:
    1. Initialised from viewpoint-mode presets.
    2. Scaled by the observed confidence of each source.
    3. Renormalised so they always sum to 1.

    The active representation (HAND_CENTRIC / FINGER_CENTRIC / COMBINED /
    TEMPORAL / UNCERTAIN) is then selected from the resulting weight profile.
    """

    def fuse(
        self,
        viewpoint: ViewpointState,
        motion_confidence: float = 0.80,
        temporal_confidence: float = 0.85,
    ) -> FusedPerceptionState:
        preset = _PRESETS.get(viewpoint.mode, _PRESETS[ViewpointMode.UNCERTAIN])

        # Scale each preset weight by observed source confidence
        raw = np.array([
            preset.palm_weight     * viewpoint.hand_geometry_confidence,
            preset.finger_weight   * viewpoint.finger_geometry_confidence,
            preset.motion_weight   * motion_confidence,
            preset.temporal_weight * temporal_confidence,
        ], dtype=np.float32)

        total = float(raw.sum())
        norm = (raw / total) if total > 1e-5 else np.full(4, 0.25, dtype=np.float32)

        weights = EvidenceWeights(
            palm_weight=float(norm[0]),
            finger_weight=float(norm[1]),
            motion_weight=float(norm[2]),
            temporal_weight=float(norm[3]),
        )

        # Combined confidence = weighted average of source confidences
        combined_conf = float(
            norm[0] * viewpoint.hand_geometry_confidence
            + norm[1] * viewpoint.finger_geometry_confidence
            + norm[2] * motion_confidence
            + norm[3] * temporal_confidence
        )

        rep, rationale = self._select_representation(viewpoint, weights, combined_conf)

        return FusedPerceptionState(
            active_representation=rep,
            weights=weights,
            combined_confidence=float(np.clip(combined_conf, 0.0, 1.0)),
            hand_geometry_confidence=viewpoint.hand_geometry_confidence,
            finger_geometry_confidence=viewpoint.finger_geometry_confidence,
            motion_confidence=motion_confidence,
            temporal_confidence=temporal_confidence,
            viewpoint_mode=viewpoint.mode,
            fusion_rationale=rationale,
        )

    @staticmethod
    def _select_representation(
        viewpoint: ViewpointState,
        weights: EvidenceWeights,
        combined_conf: float,
    ) -> Tuple[ActiveRepresentation, str]:
        if combined_conf < 0.22:
            return (
                ActiveRepresentation.UNCERTAIN,
                f"Total evidence too low (combined={combined_conf:.2f})"
            )

        if weights.temporal_weight > 0.38:
            return (
                ActiveRepresentation.TEMPORAL,
                f"Temporal evidence dominant (w={weights.temporal_weight:.2f})"
            )

        hc = viewpoint.hand_geometry_confidence
        fc = viewpoint.finger_geometry_confidence
        diff = hc - fc

        if diff > 0.22:
            return (
                ActiveRepresentation.HAND_CENTRIC,
                f"Hand geometry dominant (hand={hc:.2f} vs finger={fc:.2f})"
            )
        if diff < -0.18:
            return (
                ActiveRepresentation.FINGER_CENTRIC,
                f"Finger geometry dominant (finger={fc:.2f} vs hand={hc:.2f})"
            )

        return (
            ActiveRepresentation.COMBINED,
            f"Combined evidence (hand={hc:.2f}, finger={fc:.2f})"
        )
