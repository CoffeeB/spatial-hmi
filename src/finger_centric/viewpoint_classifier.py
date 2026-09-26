"""
Finger-Centric Perception Layer — Viewpoint Classifier.

Estimates the camera viewpoint condition from palm normal, finger foreshortening,
and depth geometry. Drives evidence-weight selection in the fusion layer.
"""
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, Optional, Tuple
import numpy as np

from src.finger_centric.finger_chain import FingerChainState


class ViewpointMode(str, Enum):
    """Perception viewpoint condition."""
    NORMAL        = "NORMAL"         # Clear frontal or dorsal view; palm geometry reliable
    FORESHORTENED = "FORESHORTENED"  # Moderate compression; mixed evidence
    CAMERA_FACING = "CAMERA_FACING"  # Hand / fingers pointing toward camera
    CAMERA_AWAY   = "CAMERA_AWAY"    # Dorsal; hand pointing away from camera
    SIDE_VIEW     = "SIDE_VIEW"      # Edge-on; palm normal is lateral
    OCCLUDED      = "OCCLUDED"       # Significant landmark occlusion
    UNCERTAIN     = "UNCERTAIN"      # Cannot determine reliably


@dataclass
class ViewpointState:
    """Current viewpoint assessment with confidence signals."""
    mode: ViewpointMode
    hand_geometry_confidence: float     # Reliability of palm-level geometry [0,1]
    finger_geometry_confidence: float   # Reliability of finger chain geometry [0,1]
    palm_facing: str                    # "PALM", "DORSAL", or "SIDE"
    foreshortening_severity: float      # 0 = none, 1 = maximum
    camera_facing_score: float          # Mean camera-facing score across fingers
    palm_normal_z: float                # z-component of palm normal vector
    active_finger_count: int            # Fingers with chain_confidence > 0.50
    uncertainty_reason: str


class ViewpointClassifier:
    """
    Classifies the current camera-hand viewpoint from multiple signals:
    - Palm normal direction
    - Mean / minimum finger chain foreshortening ratio
    - Camera-facing score across fingers
    - Active reliable finger count

    Output drives adaptive weight selection in EvidenceFusion, so the system
    shifts toward finger-centric evidence when palm geometry degrades.
    """

    def __init__(
        self,
        history_frames: int = 5,
        foreshorten_severe: float = 0.30,    # < this = severe camera-facing
        foreshorten_moderate: float = 0.55,  # < this = moderate foreshortening
        cam_facing_threshold: float = 0.62,  # mean score to declare CAMERA_FACING
    ):
        self.history_frames = history_frames
        self.foreshorten_severe = foreshorten_severe
        self.foreshorten_moderate = foreshorten_moderate
        self.cam_facing_threshold = cam_facing_threshold
        self._mode_history: Deque[ViewpointMode] = deque(maxlen=history_frames)

    def classify(
        self,
        raw_landmarks: np.ndarray,               # (21, 3)
        chain_states: Dict[str, FingerChainState],
        handedness: str,
        d_ref: float,
        palm_facing: str,
    ) -> ViewpointState:
        """Classify current viewpoint from palm + finger chain evidence."""

        # 1. Palm normal (recompute for consistency, independent of spatial engine)
        p0  = raw_landmarks[0]
        p5  = raw_landmarks[5]
        p9  = raw_landmarks[9]
        p17 = raw_landmarks[17]
        n_raw = (
            np.cross(p5 - p17, p9 - p0) if handedness == "Left"
            else np.cross(p17 - p5, p9 - p0)
        )
        n_norm = float(np.linalg.norm(n_raw))
        v_z = (n_raw / n_norm).astype(np.float32) if n_norm > 1e-5 else np.array([0.0, 0.0, -1.0])
        palm_normal_z = float(v_z[2])

        # 2. Finger foreshortening statistics
        fsr = [cs.foreshortening_ratio for cs in chain_states.values()]
        mean_fsr = float(np.mean(fsr)) if fsr else 1.0
        min_fsr  = min(fsr) if fsr else 1.0
        foreshorten_severity = float(np.clip(1.0 - mean_fsr, 0.0, 1.0))

        # 3. Camera-facing finger scores
        cfs = [cs.camera_facing_score for cs in chain_states.values()]
        mean_cfs = float(np.mean(cfs)) if cfs else 0.0
        max_cfs  = max(cfs) if cfs else 0.0

        # 4. Active reliable fingers
        active_count = sum(
            1 for cs in chain_states.values() if cs.chain_confidence > 0.48
        )

        # 5. Hand geometry confidence
        # Reliable when palm is clearly facing forward or backward
        hand_geom_conf = float(np.clip(
            (0.30 + abs(palm_normal_z) * 0.70) * max(mean_fsr, 0.08),
            0.0, 1.0
        ))

        # 6. Finger geometry confidence
        finger_geom_conf = float(np.mean([
            cs.chain_confidence for cs in chain_states.values()
        ]) if chain_states else 0.0)

        # 7. Classify
        mode, reason = self._classify_mode(
            palm_normal_z, palm_facing,
            mean_fsr, min_fsr,
            mean_cfs, max_cfs,
            active_count, foreshorten_severity,
        )
        self._mode_history.append(mode)

        return ViewpointState(
            mode=mode,
            hand_geometry_confidence=float(np.clip(hand_geom_conf, 0.0, 1.0)),
            finger_geometry_confidence=float(np.clip(finger_geom_conf, 0.0, 1.0)),
            palm_facing=palm_facing,
            foreshortening_severity=foreshorten_severity,
            camera_facing_score=mean_cfs,
            palm_normal_z=palm_normal_z,
            active_finger_count=active_count,
            uncertainty_reason=reason,
        )

    def _classify_mode(
        self,
        palm_normal_z: float,
        palm_facing: str,
        mean_fsr: float,
        min_fsr: float,
        mean_cfs: float,
        max_cfs: float,
        active_count: int,
        foreshorten_severity: float,
    ) -> Tuple[ViewpointMode, str]:

        # Severe camera-facing: finger tips pointing at/away from camera
        if max_cfs > self.cam_facing_threshold or mean_fsr < self.foreshorten_severe:
            direction = "toward" if palm_normal_z > 0.0 else "away from"
            return (
                ViewpointMode.CAMERA_FACING,
                f"Hand pointing {direction} camera — severe foreshortening (fsr={mean_fsr:.2f}, cfs={mean_cfs:.2f})"
            )

        # Dorsal view
        if palm_facing == "DORSAL" and abs(palm_normal_z) > 0.28:
            if min_fsr < self.foreshorten_moderate:
                return (
                    ViewpointMode.FORESHORTENED,
                    f"Dorsal view with finger foreshortening (min_fsr={min_fsr:.2f})"
                )
            return (
                ViewpointMode.CAMERA_AWAY,
                f"Back-of-hand dorsal view (palm_normal_z={palm_normal_z:.2f})"
            )

        # Side / edge-on view
        if abs(palm_normal_z) < 0.22 and palm_facing == "SIDE":
            return (
                ViewpointMode.SIDE_VIEW,
                "Edge-on lateral view — palm normal nearly perpendicular to camera"
            )

        # Landmark occlusion
        if active_count < 2:
            return (
                ViewpointMode.OCCLUDED,
                f"Only {active_count} finger chains with reliable data"
            )

        # Moderate foreshortening
        if foreshorten_severity > 0.32:
            return (
                ViewpointMode.FORESHORTENED,
                f"Moderate foreshortening (severity={foreshorten_severity:.2f})"
            )

        # Normal frontal view
        if palm_facing == "PALM" and abs(palm_normal_z) > 0.28:
            return (
                ViewpointMode.NORMAL,
                "Clear frontal palm view"
            )

        # Sufficient finger data without clear palm geometry
        if active_count >= 3:
            return (
                ViewpointMode.NORMAL,
                f"Normal — {active_count} active finger chains"
            )

        return (
            ViewpointMode.UNCERTAIN,
            "Cannot reliably classify viewpoint from available signals"
        )
