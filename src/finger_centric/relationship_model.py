"""
Finger-Centric Perception Layer — Fingertip Relationship Model.

Tracks inter-fingertip geometry, convergence/divergence, and relative motion.
Provides spatial configuration signals that survive heavy palm foreshortening.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.finger_centric.fingertip_tracker import FingerTipState, FINGER_NAMES

# Pairs tracked for relationships
TRACKED_PAIRS: List[Tuple[str, str]] = [
    ("thumb",  "index"),
    ("thumb",  "middle"),
    ("index",  "middle"),
    ("middle", "ring"),
    ("ring",   "little"),
    ("index",  "ring"),
    ("index",  "little"),
]


@dataclass
class TipPairRelation:
    """Relationship between two fingertips."""
    finger_a: str
    finger_b: str
    distance_3d: float                # Euclidean distance, normalized by d_ref
    distance_2d: float                # 2D projected distance (image plane)
    angle_between_directions: float   # Degrees between motion direction vectors
    convergence_rate: float           # > 0 = moving toward each other, < 0 = apart
    relative_velocity: np.ndarray     # vel_b - vel_a  (3,)
    relative_speed: float             # ||relative_velocity||


@dataclass
class FingertipRelationshipState:
    """Complete inter-fingertip relationship snapshot."""
    pairs: Dict[str, TipPairRelation]   # key = "fingerA:fingerB"
    spread_score: float                 # 0 = all pinched, 1 = fully fanned
    convergence_score: float            # Mean convergence across adjacent pairs
    dominant_pair_distance: float       # Mean tip spread across all tracked pairs
    timestamp: float

    def get_pair(self, finger_a: str, finger_b: str) -> Optional[TipPairRelation]:
        return self.pairs.get(f"{finger_a}:{finger_b}") or self.pairs.get(f"{finger_b}:{finger_a}")


class FingertipRelationshipModel:
    """
    Tracks spatial and kinematic relationships between fingertip pairs.

    Provides convergence, divergence, separation, and relative motion signals
    that remain informative even when palm geometry is heavily foreshortened.
    Useful for detecting pinches, spreads, and multi-finger configurations
    from pure finger/tip geometry.
    """

    def __init__(self, history_frames: int = 8):
        self._dist_history: Dict[str, List[float]] = {
            f"{a}:{b}": [] for a, b in TRACKED_PAIRS
        }
        self.history_frames = history_frames

    def update(
        self,
        tip_states: Dict[str, FingerTipState],
        d_ref: float,
        timestamp: float,
    ) -> FingertipRelationshipState:
        """Compute all pairwise relationships from current fingertip states."""
        pairs: Dict[str, TipPairRelation] = {}

        for finger_a, finger_b in TRACKED_PAIRS:
            if finger_a not in tip_states or finger_b not in tip_states:
                continue
            sa = tip_states[finger_a]
            sb = tip_states[finger_b]
            key = f"{finger_a}:{finger_b}"

            # 3D Euclidean distance, normalized by d_ref
            diff_3d = sb.position_3d - sa.position_3d
            dist_3d = float(np.linalg.norm(diff_3d)) / max(d_ref, 1e-4)

            # 2D projected distance
            dist_2d = float(np.linalg.norm(
                np.array(sb.position_2d) - np.array(sa.position_2d)
            ))

            # Angle between motion direction vectors
            cos_ang = float(np.clip(np.dot(sa.direction_3d, sb.direction_3d), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cos_ang)))

            # Convergence rate: project relative velocity onto separation vector
            rel_vel = (sb.velocity_3d - sa.velocity_3d).astype(np.float32)
            rel_speed = float(np.linalg.norm(rel_vel))
            diff_norm = float(np.linalg.norm(diff_3d))
            if diff_norm > 1e-5:
                # Positive = approaching (tips moving together)
                convergence_rate = float(-np.dot(rel_vel, diff_3d / diff_norm))
            else:
                convergence_rate = 0.0

            # Rolling distance history
            hist = self._dist_history[key]
            hist.append(dist_3d)
            if len(hist) > self.history_frames:
                hist.pop(0)

            pairs[key] = TipPairRelation(
                finger_a=finger_a,
                finger_b=finger_b,
                distance_3d=dist_3d,
                distance_2d=dist_2d,
                angle_between_directions=angle_deg,
                convergence_rate=convergence_rate,
                relative_velocity=rel_vel,
                relative_speed=rel_speed,
            )

        # Spread score from adjacent fingers (index-middle, middle-ring, ring-little)
        adjacent_keys = ["index:middle", "middle:ring", "ring:little"]
        adj_dists = [pairs[k].distance_3d for k in adjacent_keys if k in pairs]
        # Scale so ~0.5 d_ref spacing = score 1.0 (fully open)
        spread_score = float(np.clip(np.mean(adj_dists) * 3.5, 0.0, 1.0)) if adj_dists else 0.5

        # Global convergence score
        conv_vals = [p.convergence_rate for p in pairs.values()]
        convergence_score = float(np.mean(conv_vals)) if conv_vals else 0.0

        # Dominant tip separation (mean across all pairs)
        all_dists = [p.distance_3d for p in pairs.values()]
        dominant_pair_distance = float(np.mean(all_dists)) if all_dists else 0.0

        return FingertipRelationshipState(
            pairs=pairs,
            spread_score=spread_score,
            convergence_score=convergence_score,
            dominant_pair_distance=dominant_pair_distance,
            timestamp=timestamp,
        )
