"""
Finger-Centric Perception Layer — Main Engine.

Orchestrates fingertip tracking, finger chain analysis, relationship model,
viewpoint classification, and evidence fusion into a single FingerCentricState
per frame per hand.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from src.finger_centric.fingertip_tracker import FingerTipTracker, FingerTipState, FINGER_NAMES
from src.finger_centric.finger_chain import FingerChainAnalyzer, FingerChainState
from src.finger_centric.relationship_model import FingertipRelationshipModel, FingertipRelationshipState
from src.finger_centric.viewpoint_classifier import ViewpointClassifier, ViewpointState
from src.finger_centric.evidence_fusion import EvidenceFusion, FusedPerceptionState


@dataclass
class FingerCentricState:
    """
    Complete finger-centric perception output for a single hand frame.

    Unifies:
    - Per-fingertip kinematics (position, velocity, direction, motion class)
    - Per-finger chain geometry (direction_3d, foreshortening, flexion)
    - Inter-fingertip relationships (spread, convergence, relative velocity)
    - Viewpoint classification (NORMAL / FORESHORTENED / CAMERA_FACING / ...)
    - Evidence fusion (active representation + weighted confidence)
    - Independent palm kinematics (for separating hand vs finger motion)
    """
    hand_id: int
    handedness: str

    tip_states: Dict[str, FingerTipState]
    chain_states: Dict[str, FingerChainState]
    relationships: FingertipRelationshipState
    viewpoint: ViewpointState
    fusion: FusedPerceptionState

    palm_velocity_3d: np.ndarray   # (3,) velocity of palm centroid
    palm_speed: float

    timestamp: float

    # ------------------------------------------------------------------
    def finger_is_independent_of_palm(self, finger_name: str, threshold: float = 0.25) -> bool:
        """
        Returns True if the fingertip is moving significantly differently from the palm.
        Useful to distinguish "index finger flick" from "whole hand swipe".
        """
        ts = self.tip_states.get(finger_name)
        if ts is None:
            return False
        if self.palm_speed < 1e-4:
            return ts.speed > threshold
        ratio = ts.speed / max(self.palm_speed, 1e-4)
        return ratio > 2.0 or abs(ts.speed - self.palm_speed) > threshold

    def to_dict(self) -> dict:
        """Serialisable telemetry dict for the visualiser WebSocket."""
        return {
            "hand_id": self.hand_id,
            "handedness": self.handedness,
            "viewpoint_mode": self.viewpoint.mode.value,
            "hand_geom_conf": round(self.viewpoint.hand_geometry_confidence, 3),
            "finger_geom_conf": round(self.viewpoint.finger_geometry_confidence, 3),
            "active_representation": self.fusion.active_representation.value,
            "combined_confidence": round(self.fusion.combined_confidence, 3),
            "foreshortening_severity": round(self.viewpoint.foreshortening_severity, 3),
            "camera_facing_score": round(self.viewpoint.camera_facing_score, 3),
            "active_finger_count": self.viewpoint.active_finger_count,
            "fusion_rationale": self.fusion.fusion_rationale,
            "uncertainty_reason": self.viewpoint.uncertainty_reason,
            "evidence_weights": {
                "palm":     round(self.fusion.weights.palm_weight,     3),
                "finger":   round(self.fusion.weights.finger_weight,   3),
                "motion":   round(self.fusion.weights.motion_weight,   3),
                "temporal": round(self.fusion.weights.temporal_weight, 3),
            },
            "spread_score": round(self.relationships.spread_score, 3),
            "convergence_score": round(float(self.relationships.convergence_score), 4),
            "palm_speed": round(self.palm_speed, 4),
            "tips": {
                name: {
                    "position":            [round(float(x), 4) for x in ts.position_3d],
                    "velocity":            [round(float(x), 4) for x in ts.velocity_3d],
                    "direction":           [round(float(x), 3) for x in ts.direction_3d],
                    "depth":               round(ts.depth, 4),
                    "speed":               round(ts.speed, 4),
                    "confidence":          round(ts.confidence, 3),
                    "tracking_age":        ts.tracking_age,
                    "motion_class":        ts.motion_class.value,
                    "is_camera_facing":    ts.is_camera_facing,
                    "foreshortening_ratio": round(ts.foreshortening_ratio, 3),
                    "independent_of_palm": self.finger_is_independent_of_palm(name),
                }
                for name, ts in self.tip_states.items()
            },
            "chains": {
                name: {
                    "direction_3d":         [round(float(x), 3) for x in cs.direction_3d],
                    "direction_dip_to_tip": [round(float(x), 3) for x in cs.direction_dip_to_tip],
                    "foreshortening_ratio": round(cs.foreshortening_ratio, 3),
                    "camera_facing_score":  round(cs.camera_facing_score, 3),
                    "orientation_class":    cs.orientation_class.value,
                    "chain_confidence":     round(cs.chain_confidence, 3),
                    "depth_spread":         round(cs.depth_spread, 4),
                    "mcp_flex_deg":         round(cs.mcp_flexion_deg, 1),
                    "pip_flex_deg":         round(cs.pip_flexion_deg, 1),
                    "dip_flex_deg":         round(cs.dip_flexion_deg, 1),
                }
                for name, cs in self.chain_states.items()
            },
            "relationships": {
                key: {
                    "distance_3d":              round(pr.distance_3d, 3),
                    "distance_2d":              round(pr.distance_2d, 3),
                    "angle_between_directions": round(pr.angle_between_directions, 1),
                    "convergence_rate":         round(float(pr.convergence_rate), 4),
                    "relative_speed":           round(pr.relative_speed, 4),
                }
                for key, pr in self.relationships.pairs.items()
            },
        }


class FingerCentricPerceptionEngine:
    """
    Finger-Centric Perception Engine.

    Architecture
    ============
    For every frame and every detected hand, this engine:

    1. Updates per-fingertip kinematic tracking (position / velocity /
       acceleration / direction / motion classification).
    2. Analyzes each finger's full MCP→PIP→DIP→TIP chain.
    3. Computes inter-fingertip relationships (spread, convergence, relative velocity).
    4. Classifies the current camera viewpoint.
    5. Fuses all evidence streams with adaptive weights.
    6. Tracks independent palm velocity to separate hand-level from finger-level motion.

    This layer does NOT replace hand-centric perception.  It runs in parallel and
    is attached to HandState.finger_centric.  Downstream consumers choose which
    evidence stream to trust based on the active_representation field.
    """

    def __init__(self) -> None:
        self._tip_trackers:      Dict[int, FingerTipTracker]         = {}
        self._rel_models:        Dict[int, FingertipRelationshipModel] = {}
        self._palm_pos_history:  Dict[int, list]                     = {}

        # Stateless analyzers / classifiers (shared across all hands)
        self._chain_analyzer  = FingerChainAnalyzer()
        self._vp_classifier   = ViewpointClassifier()
        self._evidence_fusion = EvidenceFusion()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_hand(
        self,
        hand_id: int,
        handedness: str,
        raw_landmarks: np.ndarray,     # (21, 3)
        d_ref: float,
        palm_facing: str,
        timestamp: float,
        visibilities: Optional[List[float]] = None,
        ext_ratios: Optional[Dict[str, float]] = None,
        motion_confidence: float = 0.80,
        temporal_confidence: float = 0.85,
    ) -> FingerCentricState:
        """
        Run the full finger-centric perception pipeline for one hand.

        Parameters
        ----------
        hand_id : int
            Unique ID for this hand (0 or 1 in two-hand scenarios).
        handedness : str
            "Left" or "Right".
        raw_landmarks : ndarray (21, 3)
            Raw MediaPipe landmark positions in normalised camera space.
        d_ref : float
            Characteristic hand scale (wrist → middle MCP distance).
        palm_facing : str
            "PALM" / "DORSAL" / "SIDE" — from the existing spatial engine.
        timestamp : float
            Current frame timestamp (seconds).
        visibilities : list of float, optional
            Per-landmark MediaPipe visibility scores.
        ext_ratios : dict, optional
            Per-finger extension ratios from KinematicFeatureExtractor.
        motion_confidence : float
            External estimate of motion evidence reliability.
        temporal_confidence : float
            External estimate of temporal / ghost evidence reliability.
        """
        tracker   = self._get_tracker(hand_id)
        rel_model = self._get_rel_model(hand_id)

        # 1. Fingertip kinematics
        tip_states = tracker.update(
            raw_landmarks=raw_landmarks,
            d_ref=d_ref,
            timestamp=timestamp,
            visibilities=visibilities,
            ext_ratios=ext_ratios,
        )

        # 2. Finger chain analysis
        chain_states = self._chain_analyzer.analyze(
            raw_landmarks=raw_landmarks,
            d_ref=d_ref,
            handedness=handedness,
            visibilities=visibilities,
        )

        # 3. Fingertip relationship model
        relationships = rel_model.update(
            tip_states=tip_states,
            d_ref=d_ref,
            timestamp=timestamp,
        )

        # 4. Viewpoint classification
        viewpoint = self._vp_classifier.classify(
            raw_landmarks=raw_landmarks,
            chain_states=chain_states,
            handedness=handedness,
            d_ref=d_ref,
            palm_facing=palm_facing,
        )

        # 5. Evidence fusion
        fusion = self._evidence_fusion.fuse(
            viewpoint=viewpoint,
            motion_confidence=motion_confidence,
            temporal_confidence=temporal_confidence,
        )

        # 6. Independent palm velocity
        palm_vel, palm_speed = self._palm_velocity(hand_id, raw_landmarks, timestamp)

        return FingerCentricState(
            hand_id=hand_id,
            handedness=handedness,
            tip_states=tip_states,
            chain_states=chain_states,
            relationships=relationships,
            viewpoint=viewpoint,
            fusion=fusion,
            palm_velocity_3d=palm_vel,
            palm_speed=palm_speed,
            timestamp=timestamp,
        )

    def prune_missing_hands(self, active_hand_ids: List[int]) -> None:
        """Remove all per-hand state for hands that are no longer in frame."""
        for hand_id in list(self._tip_trackers.keys()):
            if hand_id not in active_hand_ids:
                del self._tip_trackers[hand_id]
        for hand_id in list(self._rel_models.keys()):
            if hand_id not in active_hand_ids:
                del self._rel_models[hand_id]
        for hand_id in list(self._palm_pos_history.keys()):
            if hand_id not in active_hand_ids:
                del self._palm_pos_history[hand_id]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_tracker(self, hand_id: int) -> FingerTipTracker:
        if hand_id not in self._tip_trackers:
            self._tip_trackers[hand_id] = FingerTipTracker()
        return self._tip_trackers[hand_id]

    def _get_rel_model(self, hand_id: int) -> FingertipRelationshipModel:
        if hand_id not in self._rel_models:
            self._rel_models[hand_id] = FingertipRelationshipModel()
        return self._rel_models[hand_id]

    def _palm_velocity(
        self,
        hand_id: int,
        raw_landmarks: np.ndarray,
        timestamp: float,
    ) -> tuple:
        """Compute palm centroid velocity, independent of fingertip velocities."""
        palm_center = raw_landmarks[[0, 5, 9, 17]].mean(axis=0)
        hist = self._palm_pos_history.setdefault(hand_id, [])
        velocity = np.zeros(3, dtype=np.float32)
        speed = 0.0
        if hist:
            prev_pos, prev_ts = hist[-1]
            dt = max(timestamp - prev_ts, 1e-4)
            velocity = ((palm_center - prev_pos) / dt).astype(np.float32)
            speed = float(np.linalg.norm(velocity))
        hist.append((palm_center.copy(), timestamp))
        if len(hist) > 12:
            hist.pop(0)
        return velocity, speed
