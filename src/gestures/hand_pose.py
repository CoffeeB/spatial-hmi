"""
Level 1: Static Hand Pose Derivation Engine for Gestura.

Implements the formal 3-tier hierarchical architecture:
    Finger States (Level 0)
          ↓
    Finger Configuration (Topological Composition)
          ↓
    Hand Pose (Level 1: H001–H016)

Static hand poses are NOT isolated black-box classifications. They are deterministically
derived from the underlying Level 0 finger state vocabulary and inter-digit geometry,
in strict accordance with the Gestura Gesture Bible (Part IV: H001–H099).
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Ensure project root is available
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.landmarks.finger_state import FingerName, FingerStateEnum, HandFingerStates


class HandPoseId(str, Enum):
    """Canonical Level 1 Hand Pose Identifiers defined in the Gesture Bible (H001–H016)."""
    H001_OPEN_PALM = "H001_OPEN_PALM"
    H002_OPEN_PALM_SPREAD = "H002_OPEN_PALM_SPREAD"
    H003_CLOSED_FIST = "H003_CLOSED_FIST"
    H004_INDEX_POINT = "H004_INDEX_POINT"
    H005_PRECISION_PINCH = "H005_PRECISION_PINCH"
    H006_LATERAL_PINCH = "H006_LATERAL_PINCH"
    H007_THUMBS_UP = "H007_THUMBS_UP"
    H008_THUMBS_DOWN = "H008_THUMBS_DOWN"
    H009_PEACE = "H009_PEACE"
    H010_OK_RING = "H010_OK_RING"
    H011_THREE_FINGER = "H011_THREE_FINGER"
    H012_SHAKA = "H012_SHAKA"
    H013_GUN = "H013_GUN"
    H014_CUPPED = "H014_CUPPED"
    H015_KNIFE_EDGE = "H015_KNIFE_EDGE"
    H016_DOUBLE_POINT = "H016_DOUBLE_POINT"
    UNKNOWN = "UNKNOWN"


# Mapping from HandPoseId to canonical short names for interaction mapping
POSE_CANONICAL_NAMES: Dict[HandPoseId, str] = {
    HandPoseId.H001_OPEN_PALM: "OPEN_PALM",
    HandPoseId.H002_OPEN_PALM_SPREAD: "SPREAD_FINGERS",
    HandPoseId.H003_CLOSED_FIST: "GRAB",
    HandPoseId.H004_INDEX_POINT: "POINT",
    HandPoseId.H005_PRECISION_PINCH: "PINCH",
    HandPoseId.H006_LATERAL_PINCH: "KEY_PINCH",
    HandPoseId.H007_THUMBS_UP: "THUMBS_UP",
    HandPoseId.H008_THUMBS_DOWN: "THUMBS_DOWN",
    HandPoseId.H009_PEACE: "PEACE",
    HandPoseId.H010_OK_RING: "OK_RING",
    HandPoseId.H011_THREE_FINGER: "THREE_FINGER",
    HandPoseId.H012_SHAKA: "CALL_ME_SHAKA",
    HandPoseId.H013_GUN: "GUN",
    HandPoseId.H014_CUPPED: "CUPPED_HAND",
    HandPoseId.H015_KNIFE_EDGE: "KNIFE_EDGE",
    HandPoseId.H016_DOUBLE_POINT: "DOUBLE_POINT",
    HandPoseId.UNKNOWN: "NONE",
}


@dataclass
class FingerConfiguration:
    """
    Intermediate Representation: Finger Configuration.
    Aggregates the 5 discrete digit states with inter-digit topological relationships.
    """
    thumb: FingerStateEnum
    index: FingerStateEnum
    middle: FingerStateEnum
    ring: FingerStateEnum
    little: FingerStateEnum

    num_extended_fingers: int  # Among digits 2-5
    num_folded_fingers: int    # Among digits 2-5
    num_curved_fingers: int    # Among digits 2-5
    num_uncertain_fingers: int # Among digits 2-5 (not in view / occluded)
    thumb_is_extended: bool
    thumb_is_folded: bool
    thumb_is_pinching: bool

    # Topological contact targets
    thumb_contact_target: Optional[str] = None
    index_contact_target: Optional[str] = None

    # Spatial relationships
    are_fingers_spread: bool = False
    are_fingers_adducted: bool = False
    is_thumb_upward: bool = False
    is_thumb_downward: bool = False
    index_middle_divergence_deg: float = 0.0
    thumb_index_angle_deg: float = 0.0
    is_edge_on: bool = False
    inner_hand_visible: bool = True
    palm_facing: str = "PALM"  # "PALM", "DORSAL", or "SIDE"

    def summary(self) -> str:
        """One-line concise topological signature."""
        view_tag = self.palm_facing.lower()
        return (
            f"T:{self.thumb.value}|I:{self.index.value}|M:{self.middle.value}|"
            f"R:{self.ring.value}|L:{self.little.value} "
            f"[{view_tag}, ext:{self.num_extended_fingers}, fld:{self.num_folded_fingers}]"
        )


@dataclass
class DerivedHandPose:
    """
    Complete Level 1 Hand Pose derivation result with explainability diagnostics.
    """
    pose_id: HandPoseId
    canonical_name: str
    confidence: float  # In [0, 1]
    configuration: FingerConfiguration
    satisfied_predicates: List[str] = field(default_factory=list)
    unmet_predicates: List[str] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)

    def is_pose(self, pose_id: HandPoseId) -> bool:
        return self.pose_id == pose_id


class HandPoseClassifier:
    """
    Deterministic Level 1 Hand Pose Classifier.
    Derives hand poses from Level 0 Finger States and Finger Configurations.
    """
    _FOLDED_LIKE = (
        FingerStateEnum.FOLDED,
        FingerStateEnum.TUCKED,
        FingerStateEnum.HOOKED,
        FingerStateEnum.UNCERTAIN,
    )

    def __init__(self, min_confidence: float = 0.45):
        self.min_confidence = min_confidence

    @staticmethod
    def _compute_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    def build_configuration(
        self,
        finger_states: HandFingerStates,
        raw_landmarks: np.ndarray,
        d_ref: float,
        orientation_angles: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> FingerConfiguration:
        """
        Builds the Finger Configuration intermediate representation from Level 0 states and landmarks.
        """
        s_thumb = finger_states.thumb.state
        s_index = finger_states.index.state
        s_middle = finger_states.middle.state
        s_ring = finger_states.ring.state
        s_little = finger_states.little.state

        digits_2to5 = [s_index, s_middle, s_ring, s_little]
        num_ext = sum(1 for s in digits_2to5 if s == FingerStateEnum.EXTENDED)
        num_fld = sum(1 for s in digits_2to5 if s in (FingerStateEnum.FOLDED, FingerStateEnum.TUCKED, FingerStateEnum.HOOKED))
        num_crv = sum(1 for s in digits_2to5 if s in (FingerStateEnum.CURVED, FingerStateEnum.RELAXED))
        num_unc = sum(1 for s in digits_2to5 if s == FingerStateEnum.UNCERTAIN)

        thumb_ext = (s_thumb == FingerStateEnum.EXTENDED) or (
            s_thumb in (FingerStateEnum.CURVED, FingerStateEnum.RELAXED) and finger_states.thumb.extension_ratio >= 1.15
        )
        thumb_fld = (s_thumb in (FingerStateEnum.FOLDED, FingerStateEnum.TUCKED))
        thumb_pinch = (s_thumb == FingerStateEnum.PINCHING or s_index == FingerStateEnum.PINCHING)

        # Inter-digit spacing between fingertips 8, 12, 16, 20
        p8, p12, p16, p20 = raw_landmarks[8], raw_landmarks[12], raw_landmarks[16], raw_landmarks[20]
        sep_im = np.linalg.norm(p12 - p8) / d_ref
        sep_mr = np.linalg.norm(p16 - p12) / d_ref
        sep_rl = np.linalg.norm(p20 - p16) / d_ref
        mean_sep = float((sep_im + sep_mr + sep_rl) / 3.0)

        are_spread = mean_sep >= 0.34
        are_adducted = mean_sep <= 0.16

        # Index & Middle separation angle (from wrist)
        v_idx = raw_landmarks[8] - raw_landmarks[5]
        v_mid = raw_landmarks[12] - raw_landmarks[9]
        idx_mid_deg = self._compute_angle_between_vectors(v_idx, v_mid)

        # Thumb direction vectors (from MCP 2 to Tip 4 and CMC 1 to Tip 4)
        v_thumb = raw_landmarks[4] - raw_landmarks[2]
        v_thumb_cmc = raw_landmarks[4] - raw_landmarks[1]

        norm_thumb = np.linalg.norm(v_thumb)
        norm_cmc = np.linalg.norm(v_thumb_cmc)
        unit_thumb_y = (v_thumb[1] / norm_thumb) if norm_thumb > 1e-6 else 0.0
        unit_cmc_y = (v_thumb_cmc[1] / norm_cmc) if norm_cmc > 1e-6 else 0.0

        # Physical elevation checks:
        tip_above_knuckle = bool(raw_landmarks[4][1] < min(raw_landmarks[5][1], raw_landmarks[9][1]))
        tip_above_wrist = bool(raw_landmarks[4][1] < raw_landmarks[0][1])
        tip_below_wrist = bool(raw_landmarks[4][1] > raw_landmarks[0][1])

        # Thumb upward: points into upper hemisphere and elevated above knuckles or wrist
        is_thumb_up = (unit_thumb_y <= -0.38 or unit_cmc_y <= -0.42) and (tip_above_knuckle or tip_above_wrist)
        # Thumb downward: points into lower hemisphere and below wrist
        is_thumb_down = (unit_thumb_y >= 0.38 or unit_cmc_y >= 0.42) and tip_below_wrist

        # Thumb to Index angle
        thumb_idx_deg = self._compute_angle_between_vectors(v_thumb, v_idx)

        # Edge-on check from roll angle
        pitch, yaw, roll = orientation_angles
        is_edge = abs(np.cos(roll)) < 0.35

        # Determine if inner hand (palm) or dorsal side faces camera
        palm_facing = getattr(finger_states, "palm_facing", "PALM")
        inner_visible = bool(palm_facing == "PALM")

        return FingerConfiguration(
            thumb=s_thumb,
            index=s_index,
            middle=s_middle,
            ring=s_ring,
            little=s_little,
            num_extended_fingers=num_ext,
            num_folded_fingers=num_fld,
            num_curved_fingers=num_crv,
            num_uncertain_fingers=num_unc,
            thumb_is_extended=thumb_ext,
            thumb_is_folded=thumb_fld,
            thumb_is_pinching=thumb_pinch,
            thumb_contact_target=finger_states.thumb.contact_target,
            index_contact_target=finger_states.index.contact_target,
            are_fingers_spread=are_spread,
            are_fingers_adducted=are_adducted,
            is_thumb_upward=is_thumb_up,
            is_thumb_downward=is_thumb_down,
            index_middle_divergence_deg=idx_mid_deg,
            thumb_index_angle_deg=thumb_idx_deg,
            is_edge_on=is_edge,
            inner_hand_visible=inner_visible,
            palm_facing=palm_facing,
        )

    def classify_pose(
        self,
        finger_states: HandFingerStates,
        raw_landmarks: np.ndarray,
        d_ref: float,
        orientation_angles: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> DerivedHandPose:
        """
        Derives the active Level 1 Hand Pose from the Finger Configuration.
        Evaluates rule-based predicates with continuous confidence scoring and explainability.
        """
        cfg = self.build_configuration(finger_states, raw_landmarks, d_ref, orientation_angles)

        # -------------------------------------------------------------------
        # Rule 1: H005 Precision Pinch (Thumb + Index touching)
        # -------------------------------------------------------------------
        # Thumb & Index pinching, while digits 3-5 are NOT all extended
        if (
            cfg.thumb_is_pinching
            or (cfg.thumb == FingerStateEnum.TOUCHING and cfg.thumb_contact_target in ("index", "index_tip"))
        ):
            if cfg.num_extended_fingers >= 3:
                # Disambiguate with H010 OK Ring
                pass
            else:
                conf = 0.95 if cfg.thumb == FingerStateEnum.PINCHING else 0.88
                return DerivedHandPose(
                    pose_id=HandPoseId.H005_PRECISION_PINCH,
                    canonical_name=POSE_CANONICAL_NAMES[HandPoseId.H005_PRECISION_PINCH],
                    confidence=conf,
                    configuration=cfg,
                    satisfied_predicates=[
                        "Thumb and Index in pinching contact opposition",
                        f"Digits 3–5 resting/curled (num_ext={cfg.num_extended_fingers})",
                    ],
                    diagnostics=["Derived from Pinch topology (F008 Thumb + F008 Index)"],
                )

        # -------------------------------------------------------------------
        # Rule 2: H010 OK Ring (Pinch ring + Digits 3-5 extended)
        # -------------------------------------------------------------------
        if cfg.thumb_is_pinching and cfg.num_extended_fingers >= 3:
            return DerivedHandPose(
                pose_id=HandPoseId.H010_OK_RING,
                canonical_name=POSE_CANONICAL_NAMES[HandPoseId.H010_OK_RING],
                confidence=0.94,
                configuration=cfg,
                satisfied_predicates=[
                    "Thumb and Index touching in circular ring",
                    "Middle, Ring, and Little extended outward",
                ],
                diagnostics=["Derived from Pinch ring + extended outer digits"],
            )

        # -------------------------------------------------------------------
        # Rule 3: H004 Index Point (Index extended, others curled into palm)
        # -------------------------------------------------------------------
        if cfg.index == FingerStateEnum.EXTENDED:
            # Point requires middle, ring, little to be folded/tucked, curved, or occluded/uncertain
            mid_curled = cfg.middle in self._FOLDED_LIKE or (cfg.middle == FingerStateEnum.CURVED and finger_states.middle.extension_ratio <= 1.05)
            rng_curled = cfg.ring in self._FOLDED_LIKE or (cfg.ring == FingerStateEnum.CURVED and finger_states.ring.extension_ratio <= 1.05)
            lit_curled = cfg.little in self._FOLDED_LIKE or (cfg.little == FingerStateEnum.CURVED and finger_states.little.extension_ratio <= 1.05)

            if mid_curled and (rng_curled or lit_curled):
                # Disambiguation with H013 Gun: Gun has thumb extended radially
                if cfg.thumb_is_extended and cfg.thumb_index_angle_deg >= 65.0:
                    pass  # Defer to Gun below
                else:
                    conf = 0.94 if (rng_curled and lit_curled) else 0.82
                    return DerivedHandPose(
                        pose_id=HandPoseId.H004_INDEX_POINT,
                        canonical_name=POSE_CANONICAL_NAMES[HandPoseId.H004_INDEX_POINT],
                        confidence=conf,
                        configuration=cfg,
                        satisfied_predicates=[
                            "Index finger extended along pointing ray",
                            "Middle, Ring, and Little curled into palm",
                            f"Thumb neutral/folded (state={cfg.thumb.value})",
                        ],
                        diagnostics=["Derived from Index extension with isolated lateral curl"],
                    )

        # -------------------------------------------------------------------
        # Rule 4: H013 Gun / L-Shape Point (Thumb up + Index forward, others folded)
        # -------------------------------------------------------------------
        if (
            cfg.index == FingerStateEnum.EXTENDED
            and cfg.thumb_is_extended
            and cfg.middle in self._FOLDED_LIKE
            and cfg.ring in self._FOLDED_LIKE
            and cfg.thumb_index_angle_deg >= 55.0
        ):
            return DerivedHandPose(
                pose_id=HandPoseId.H013_GUN,
                canonical_name=POSE_CANONICAL_NAMES[HandPoseId.H013_GUN],
                confidence=0.92,
                configuration=cfg,
                satisfied_predicates=[
                    "Index finger extended forward",
                    "Thumb extended radially in L-formation",
                    f"Thumb-Index separation angle {cfg.thumb_index_angle_deg:.1f}° >= 55°",
                    "Digits 3–5 curled into palm",
                ],
                diagnostics=["Derived from L-shape orthogonal digit abduction"],
            )

        # -------------------------------------------------------------------
        # Rule 5: H016 Finger Guns Double Point (Index + Middle extended together)
        # -------------------------------------------------------------------
        if (
            cfg.index == FingerStateEnum.EXTENDED
            and cfg.middle == FingerStateEnum.EXTENDED
            and cfg.thumb_is_extended
            and cfg.ring in self._FOLDED_LIKE
            and cfg.little in self._FOLDED_LIKE
            and cfg.index_middle_divergence_deg < 12.0
        ):
            return DerivedHandPose(
                pose_id=HandPoseId.H016_DOUBLE_POINT,
                canonical_name=POSE_CANONICAL_NAMES[HandPoseId.H016_DOUBLE_POINT],
                confidence=0.91,
                configuration=cfg,
                satisfied_predicates=[
                    "Index and Middle extended tightly parallel",
                    "Thumb extended upward",
                    "Ring and Little curled into palm",
                ],
                diagnostics=["Derived from parallel dual-ray extension with thumb cocked"],
            )

        # -------------------------------------------------------------------
        # Rule 6: H009 Peace / Victory (Index + Middle extended in V-sign)
        # -------------------------------------------------------------------
        if (
            cfg.index == FingerStateEnum.EXTENDED
            and cfg.middle == FingerStateEnum.EXTENDED
            and (cfg.ring in self._FOLDED_LIKE or (cfg.ring == FingerStateEnum.CURVED and finger_states.ring.extension_ratio <= 1.05))
            and (cfg.little in self._FOLDED_LIKE or (cfg.little == FingerStateEnum.CURVED and finger_states.little.extension_ratio <= 1.05))
            and cfg.index_middle_divergence_deg >= 8.0
        ):
            return DerivedHandPose(
                pose_id=HandPoseId.H009_PEACE,
                canonical_name=POSE_CANONICAL_NAMES[HandPoseId.H009_PEACE],
                confidence=0.93,
                configuration=cfg,
                satisfied_predicates=[
                    "Index and Middle extended outward in V-formation",
                    f"Inter-digit divergence {cfg.index_middle_divergence_deg:.1f}° >= 8°",
                    "Ring and Little folded into palm",
                ],
                diagnostics=["Derived from V-formation extension with lateral curl"],
            )

        # -------------------------------------------------------------------
        # Rule 7: H011 Three-Finger Spread (Index, Middle, Ring extended)
        # -------------------------------------------------------------------
        if (
            cfg.index == FingerStateEnum.EXTENDED
            and cfg.middle == FingerStateEnum.EXTENDED
            and cfg.ring == FingerStateEnum.EXTENDED
            and cfg.little in self._FOLDED_LIKE
            and not cfg.thumb_is_extended
        ):
            return DerivedHandPose(
                pose_id=HandPoseId.H011_THREE_FINGER,
                canonical_name=POSE_CANONICAL_NAMES[HandPoseId.H011_THREE_FINGER],
                confidence=0.90,
                configuration=cfg,
                satisfied_predicates=[
                    "Index, Middle, Ring extended",
                    "Little folded and Thumb curled/neutral",
                ],
                diagnostics=["Derived from trident 3-digit extension"],
            )

        # -------------------------------------------------------------------
        # Rule 8: H012 Call Me / Shaka (Thumb + Little extended, central digits folded)
        # -------------------------------------------------------------------
        if (
            cfg.thumb_is_extended
            and cfg.little == FingerStateEnum.EXTENDED
            and cfg.index in self._FOLDED_LIKE
            and cfg.middle in self._FOLDED_LIKE
            and cfg.ring in self._FOLDED_LIKE
        ):
            return DerivedHandPose(
                pose_id=HandPoseId.H012_SHAKA,
                canonical_name=POSE_CANONICAL_NAMES[HandPoseId.H012_SHAKA],
                confidence=0.94,
                configuration=cfg,
                satisfied_predicates=[
                    "Thumb and Little finger extended outward",
                    "Central three digits (Index, Middle, Ring) folded into palm",
                ],
                diagnostics=["Derived from extreme peripheral digit extension (Shaka)"],
            )

        # -------------------------------------------------------------------
        # Rule 9: H007 Thumbs Up / H008 Thumbs Down
        # -------------------------------------------------------------------
        thumb_isolated = (
            cfg.thumb_is_extended
            or (cfg.thumb in (FingerStateEnum.CURVED, FingerStateEnum.RELAXED) and finger_states.thumb.extension_ratio >= 1.08)
        ) and (cfg.thumb != FingerStateEnum.TUCKED)

        fingers_balled = (cfg.num_extended_fingers == 0) and (
            cfg.num_folded_fingers >= 3
            or (cfg.num_folded_fingers + cfg.num_curved_fingers >= 4)
            or (not cfg.inner_hand_visible and (cfg.num_folded_fingers + cfg.num_curved_fingers + cfg.num_uncertain_fingers >= 4) and cfg.num_folded_fingers >= 2)
        )

        if thumb_isolated and fingers_balled:
            if cfg.is_thumb_upward:
                return DerivedHandPose(
                    pose_id=HandPoseId.H007_THUMBS_UP,
                    canonical_name=POSE_CANONICAL_NAMES[HandPoseId.H007_THUMBS_UP],
                    confidence=0.95,
                    configuration=cfg,
                    satisfied_predicates=[
                        "Thumb isolated and pointing upward (-Y)",
                        "All four fingers balled into closed fist",
                    ],
                    diagnostics=["Derived from isolated vertical thumb extension (Thumbs Up)"],
                )
            elif cfg.is_thumb_downward:
                return DerivedHandPose(
                    pose_id=HandPoseId.H008_THUMBS_DOWN,
                    canonical_name=POSE_CANONICAL_NAMES[HandPoseId.H008_THUMBS_DOWN],
                    confidence=0.95,
                    configuration=cfg,
                    satisfied_predicates=[
                        "Thumb isolated and pointing downward (+Y)",
                        "All four fingers balled into closed fist",
                    ],
                    diagnostics=["Derived from isolated downward thumb extension (Thumbs Down)"],
                )

        # -------------------------------------------------------------------
        # Rule 10: H006 Lateral Pinch / Key Pinch (Thumb pad touching Index side)
        # -------------------------------------------------------------------
        if (
            cfg.thumb == FingerStateEnum.TOUCHING
            and cfg.thumb_contact_target in ("index_base", "index")
            and cfg.middle in self._FOLDED_LIKE
        ):
            return DerivedHandPose(
                pose_id=HandPoseId.H006_LATERAL_PINCH,
                canonical_name=POSE_CANONICAL_NAMES[HandPoseId.H006_LATERAL_PINCH],
                confidence=0.86,
                configuration=cfg,
                satisfied_predicates=[
                    "Thumb pad pressed against lateral radial side of Index",
                    "Remaining digits curled into palm",
                ],
                diagnostics=["Derived from lateral index base contact"],
            )

        # -------------------------------------------------------------------
        # Rule 11: H003 Closed Fist (All digits balled into palm)
        # -------------------------------------------------------------------
        if (
            (cfg.num_folded_fingers >= 3 or (cfg.num_folded_fingers >= 2 and cfg.num_folded_fingers + cfg.num_curved_fingers >= 4))
            and not cfg.thumb_is_pinching
            and not cfg.thumb_is_extended
        ):
            if cfg.num_extended_fingers == 0:
                return DerivedHandPose(
                    pose_id=HandPoseId.H003_CLOSED_FIST,
                    canonical_name=POSE_CANONICAL_NAMES[HandPoseId.H003_CLOSED_FIST],
                    confidence=0.93,
                    configuration=cfg,
                    satisfied_predicates=[
                        "All fingers 2–5 curled/folded into palm",
                        f"Thumb folded or tucked (state={cfg.thumb.value})",
                        "Zero active pinch opposition",
                    ],
                    diagnostics=["Derived from full digit flexion across palm"],
                )

        # -------------------------------------------------------------------
        # Rule 12: H015 Flat Hand Knife Edge (Extended + tightly adducted + edge-on)
        # -------------------------------------------------------------------
        if cfg.num_extended_fingers == 4 and cfg.are_fingers_adducted and cfg.is_edge_on:
            return DerivedHandPose(
                pose_id=HandPoseId.H015_KNIFE_EDGE,
                canonical_name=POSE_CANONICAL_NAMES[HandPoseId.H015_KNIFE_EDGE],
                confidence=0.88,
                configuration=cfg,
                satisfied_predicates=[
                    "All digits extended and tightly parallel (adducted)",
                    "Hand oriented edge-on to camera plane",
                ],
                diagnostics=["Derived from planar knife-edge configuration"],
            )

        # -------------------------------------------------------------------
        # Rule 13: H002 Open Palm Spread (All digits extended + abducted)
        # -------------------------------------------------------------------
        if cfg.num_extended_fingers == 4 and (cfg.are_fingers_spread or (cfg.thumb_is_extended and cfg.thumb_index_angle_deg >= 45.0)):
            return DerivedHandPose(
                pose_id=HandPoseId.H002_OPEN_PALM_SPREAD,
                canonical_name=POSE_CANONICAL_NAMES[HandPoseId.H002_OPEN_PALM_SPREAD],
                confidence=0.92,
                configuration=cfg,
                satisfied_predicates=[
                    "All 5 digits extended outward",
                    "Digits abducted laterally with wide inter-digit separation",
                ],
                diagnostics=["Derived from maximal digit extension and lateral spread"],
            )

        # -------------------------------------------------------------------
        # Rule 14: H001 Open Palm Neutral (All digits extended without forced spread)
        # -------------------------------------------------------------------
        if cfg.num_extended_fingers >= 3 and cfg.num_folded_fingers == 0 and not cfg.thumb_is_pinching:
            return DerivedHandPose(
                pose_id=HandPoseId.H001_OPEN_PALM,
                canonical_name=POSE_CANONICAL_NAMES[HandPoseId.H001_OPEN_PALM],
                confidence=0.88,
                configuration=cfg,
                satisfied_predicates=[
                    f"Fingers 2–5 extended or relaxed (num_ext={cfg.num_extended_fingers})",
                    f"Thumb natural resting or extended (state={cfg.thumb.value})",
                    "Zero digits balled into palm",
                ],
                diagnostics=["Derived from open resting planar posture"],
            )

        # -------------------------------------------------------------------
        # Rule 15: H014 Cupped Hand (All digits semi-flexed / curved concavity)
        # -------------------------------------------------------------------
        if cfg.num_curved_fingers >= 3 and cfg.num_folded_fingers == 0 and cfg.num_extended_fingers <= 1:
            return DerivedHandPose(
                pose_id=HandPoseId.H014_CUPPED,
                canonical_name=POSE_CANONICAL_NAMES[HandPoseId.H014_CUPPED],
                confidence=0.85,
                configuration=cfg,
                satisfied_predicates=[
                    "All digits semi-flexed / curved forming concave palmar bowl",
                    f"Curved digits count: {cfg.num_curved_fingers}",
                ],
                diagnostics=["Derived from palmar cupping flexion (F003/F004)"],
            )

        # -------------------------------------------------------------------
        # Default: Unknown / Transitioning
        # -------------------------------------------------------------------
        return DerivedHandPose(
            pose_id=HandPoseId.UNKNOWN,
            canonical_name=POSE_CANONICAL_NAMES[HandPoseId.UNKNOWN],
            confidence=0.20,
            configuration=cfg,
            satisfied_predicates=[],
            unmet_predicates=[
                "No standard static hand pose criteria satisfied",
                f"Configuration: {cfg.summary()}",
            ],
            diagnostics=["Transitioning between stable postures or intermediate state"],
        )
