"""
Finger-Centric Perception Layer — Finger Chain Analyzer.

Tracks the complete MCP→PIP→DIP→TIP articulated chain for each finger,
computing 3D direction, foreshortening, joint angles, and orientation class.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import numpy as np

# Full chain indices per finger
FINGER_CHAIN_INDICES: Dict[str, Dict[str, int]] = {
    "thumb":  {"mcp": 1,  "pip": 2,  "dip": 3,  "tip": 4},
    "index":  {"mcp": 5,  "pip": 6,  "dip": 7,  "tip": 8},
    "middle": {"mcp": 9,  "pip": 10, "dip": 11, "tip": 12},
    "ring":   {"mcp": 13, "pip": 14, "dip": 15, "tip": 16},
    "little": {"mcp": 17, "pip": 18, "dip": 19, "tip": 20},
}

# Weighted contribution of each segment toward the dominant direction.
# DIP→TIP has the highest weight as it most clearly indicates "where the finger points".
_SEG_WEIGHTS = (0.15, 0.25, 0.60)  # MCP→PIP, PIP→DIP, DIP→TIP


class FingerOrientationClass(str, Enum):
    """Dominant 3D orientation of the finger relative to camera."""
    LATERAL       = "LATERAL"        # Pointing mostly left or right in image
    VERTICAL      = "VERTICAL"       # Pointing mostly up or down in image
    CAMERA_FACING = "CAMERA_FACING"  # Pointing toward camera (foreshortened in 2D)
    CAMERA_AWAY   = "CAMERA_AWAY"    # Pointing away from camera
    OBLIQUE       = "OBLIQUE"        # Diagonal mix of lateral + depth
    UNKNOWN       = "UNKNOWN"


@dataclass
class FingerChainState:
    """Full geometric analysis of a single finger's articulated chain."""
    finger_name: str

    # Raw 3D positions of each joint
    mcp_3d: np.ndarray
    pip_3d: np.ndarray
    dip_3d: np.ndarray
    tip_3d: np.ndarray

    # Dominant 3D direction of the entire finger chain
    direction_3d: np.ndarray          # Normalized (3,) weighted over all segments
    # Local direction from the distal two joints (most informative endpoint signal)
    direction_dip_to_tip: np.ndarray  # Normalized (3,)

    # Geometry
    chain_length_3d: float            # Sum of 3D segment lengths
    apparent_length_2d: float         # ||tip_2d - mcp_2d|| — what the camera "sees"
    foreshortening_ratio: float       # apparent_2d / chain_3d: 1 = none, ~0 = end-on

    # Joint flexion (deviation from 180° straight)
    mcp_flexion_deg: float
    pip_flexion_deg: float
    dip_flexion_deg: float

    # Orientation
    orientation_class: FingerOrientationClass
    camera_facing_score: float        # 0 = lateral, 1 = pointing at/away from camera
    depth_spread: float               # |tip_z - mcp_z| — depth range along chain

    # Reliability
    chain_confidence: float           # [0, 1] penalised by foreshortening + visibility


class FingerChainAnalyzer:
    """
    Analyzes the complete articulated chain of each finger.

    Provides viewpoint-invariant finger direction estimation by deriving the dominant
    3D direction from the weighted MCP→PIP→DIP→TIP segment chain rather than any
    single pair of landmarks. Detects foreshortening and camera-facing orientations.
    """

    @staticmethod
    def compute_chain_extension_ratio(chain: FingerChainState) -> float:
        """
        Computes continuous extension ratio derived strictly from 3D articulated chain geometry.
        Unlike 2D projected dist(tip, wrist) / dist(mcp, wrist), this remains invariant
        under severe camera-facing foreshortening.
        
        Returns:
            ratio: ~1.40-1.75 when straight/extended, ~1.05-1.20 when relaxed, ~0.50-0.80 when curled/folded.
        """
        chord = float(np.linalg.norm(chain.tip_3d - chain.mcp_3d))
        straightness = chord / max(chain.chain_length_3d, 1e-4)
        flex_pen = (chain.pip_flexion_deg + chain.dip_flexion_deg) / 180.0

        # When finger points straight toward camera:
        # straightness is high (~0.90-0.98), flex_pen is low (< 0.25).
        # When balled into fist: straightness is low (< 0.50), flex_pen is high (> 0.70).
        ratio = 0.65 + 1.10 * straightness - 0.55 * flex_pen
        return float(np.clip(ratio, 0.40, 2.10))

    def compute_all_chain_extension_ratios(self, chain_states: Dict[str, FingerChainState]) -> Dict[str, float]:
        """Maps finger names to kinematic feature keys ('thumb', 'index', 'middle', 'ring', 'pinky')."""
        out = {}
        for name, cs in chain_states.items():
            key = "pinky" if name == "little" else name
            out[key] = self.compute_chain_extension_ratio(cs)
        return out

    def analyze(
        self,
        raw_landmarks: np.ndarray,      # (21, 3)
        d_ref: float,
        handedness: str = "Right",
        visibilities: Optional[List[float]] = None,
    ) -> Dict[str, FingerChainState]:
        """Analyze all five finger chains and return per-finger states."""
        return {
            name: self._analyze_chain(name, raw_landmarks, indices, d_ref, visibilities)
            for name, indices in FINGER_CHAIN_INDICES.items()
        }

    def _analyze_chain(
        self,
        finger_name: str,
        raw_pts: np.ndarray,
        indices: Dict[str, int],
        d_ref: float,
        visibilities: Optional[List[float]],
    ) -> FingerChainState:
        mcp = raw_pts[indices["mcp"]].copy()
        pip = raw_pts[indices["pip"]].copy()
        dip = raw_pts[indices["dip"]].copy()
        tip = raw_pts[indices["tip"]].copy()

        # Segment vectors and lengths
        seg1 = pip - mcp
        seg2 = dip - pip
        seg3 = tip - dip
        len1 = max(float(np.linalg.norm(seg1)), 1e-5)
        len2 = max(float(np.linalg.norm(seg2)), 1e-5)
        len3 = max(float(np.linalg.norm(seg3)), 1e-5)
        chain_length_3d = len1 + len2 + len3

        # Weighted dominant direction over entire chain
        w1, w2, w3 = _SEG_WEIGHTS
        composite = (seg1 / len1) * w1 + (seg2 / len2) * w2 + (seg3 / len3) * w3
        comp_norm = float(np.linalg.norm(composite))
        direction_3d = (
            (composite / comp_norm).astype(np.float32) if comp_norm > 1e-5
            else np.array([0.0, -1.0, 0.0], dtype=np.float32)
        )

        # Local tip direction (DIP→TIP, most precise endpoint signal)
        direction_dip_tip = (
            (seg3 / len3).astype(np.float32) if len3 > 1e-5
            else direction_3d.copy()
        )

        # Apparent 2D length (camera projection — ignores z)
        apparent_2d = float(np.linalg.norm(tip[:2] - mcp[:2]))
        foreshorten_ratio = float(np.clip(apparent_2d / chain_length_3d, 0.0, 1.0))

        # Joint flexion angles
        mcp_flex = 180.0 - self._angle_deg(raw_pts[0], mcp, pip)
        pip_flex = 180.0 - self._angle_deg(mcp, pip, dip)
        dip_flex = 180.0 - self._angle_deg(pip, dip, tip)

        # Depth analysis
        depth_spread = abs(float(tip[2]) - float(mcp[2]))

        # Camera-facing score: fraction of direction in the z axis
        dz_abs = abs(float(direction_3d[2]))
        dxy = float(np.sqrt(direction_3d[0] ** 2 + direction_3d[1] ** 2))
        camera_facing_score = dz_abs / max(dz_abs + dxy, 1e-5)

        # Orientation classification
        orientation_class = self._classify_orientation(direction_3d, camera_facing_score)

        # Chain confidence
        vis_score = 1.0
        if visibilities:
            tip_vis = float(visibilities[indices["tip"]])
            dip_vis = float(visibilities[indices["dip"]])
            vis_score = (tip_vis + dip_vis) / 2.0
        chain_confidence = float(np.clip(
            vis_score * (0.25 + 0.75 * max(foreshorten_ratio, 0.08)), 0.0, 1.0
        ))

        return FingerChainState(
            finger_name=finger_name,
            mcp_3d=mcp.astype(np.float32),
            pip_3d=pip.astype(np.float32),
            dip_3d=dip.astype(np.float32),
            tip_3d=tip.astype(np.float32),
            direction_3d=direction_3d,
            direction_dip_to_tip=direction_dip_tip,
            chain_length_3d=chain_length_3d,
            apparent_length_2d=apparent_2d,
            foreshortening_ratio=foreshorten_ratio,
            mcp_flexion_deg=mcp_flex,
            pip_flexion_deg=pip_flex,
            dip_flexion_deg=dip_flex,
            orientation_class=orientation_class,
            camera_facing_score=camera_facing_score,
            depth_spread=depth_spread,
            chain_confidence=chain_confidence,
        )

    @staticmethod
    def _angle_deg(p1: np.ndarray, p_vertex: np.ndarray, p2: np.ndarray) -> float:
        """3D angle in degrees at p_vertex between (p1-p_vertex) and (p2-p_vertex)."""
        v1 = p1 - p_vertex
        v2 = p2 - p_vertex
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            return 180.0
        cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_a)))

    @staticmethod
    def _classify_orientation(
        direction_3d: np.ndarray,
        camera_facing_score: float,
    ) -> FingerOrientationClass:
        dx = abs(float(direction_3d[0]))
        dy = abs(float(direction_3d[1]))
        dz_raw = float(direction_3d[2])

        if camera_facing_score > 0.68:
            return (
                FingerOrientationClass.CAMERA_FACING if dz_raw < 0
                else FingerOrientationClass.CAMERA_AWAY
            )
        if camera_facing_score < 0.28:
            return FingerOrientationClass.VERTICAL if dy > dx else FingerOrientationClass.LATERAL
        return FingerOrientationClass.OBLIQUE
