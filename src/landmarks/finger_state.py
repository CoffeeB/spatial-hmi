"""
Level 0: Finger State Classification Engine for Gestura.
Answers the foundational question: "What is every individual finger doing?"

Classifies each digit (Thumb, Index, Middle, Ring, Little) into one of 10 canonical states:
- extended
- folded
- curved
- relaxed
- tucked
- hooked
- touching
- pinching
- crossed
- uncertain

Designed to be invariant to:
- Camera perspective and hand orientation (via hand-local orthonormal 3D frame)
- Distance from sensor (via scale reference d_ref)
- Lighting / low tracking confidence (graceful fallback to 'uncertain')
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Ensure project root is in sys.path when script is invoked directly
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from src.landmarks.kinematic_features import FINGER_INDICES
except ModuleNotFoundError:
    from kinematic_features import FINGER_INDICES


class FingerName(str, Enum):
    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    LITTLE = "little"


DIGIT_TO_FEATURE_KEY = {
    FingerName.THUMB: "thumb",
    FingerName.INDEX: "index",
    FingerName.MIDDLE: "middle",
    FingerName.RING: "ring",
    FingerName.LITTLE: "pinky",
}


class FingerStateEnum(str, Enum):
    EXTENDED = "extended"
    FOLDED = "folded"
    CURVED = "curved"
    RELAXED = "relaxed"
    TUCKED = "tucked"
    HOOKED = "hooked"
    TOUCHING = "touching"
    PINCHING = "pinching"
    CROSSED = "crossed"
    UNCERTAIN = "uncertain"


@dataclass
class FingerStateDetail:
    """Detailed state analysis for a single digit."""
    finger: FingerName
    state: FingerStateEnum
    confidence: float = 1.0  # In [0, 1]
    extension_ratio: float = 1.0
    mcp_flexion_deg: float = 0.0
    pip_flexion_deg: float = 0.0
    dip_flexion_deg: float = 0.0
    contact_target: Optional[str] = None
    diagnostics: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.finger.value.capitalize():<8}: {self.state.value}"


@dataclass
class HandFingerStates:
    """Complete Level 0 state collection for all 5 digits of a hand."""
    thumb: FingerStateDetail
    index: FingerStateDetail
    middle: FingerStateDetail
    ring: FingerStateDetail
    little: FingerStateDetail
    timestamp: float = 0.0
    palm_facing: str = "PALM"  # "PALM" (front), "DORSAL" (back), or "SIDE" (edge-on)

    def get(self, finger: FingerName) -> FingerStateDetail:
        return getattr(self, finger.value)

    def as_dict(self) -> Dict[str, str]:
        """Returns standard dict mapping digit name to state string."""
        return {
            "Thumb": self.thumb.state.value,
            "Index": self.index.state.value,
            "Middle": self.middle.state.value,
            "Ring": self.ring.state.value,
            "Little": self.little.state.value,
        }

    def as_details_dict(self) -> Dict[str, Dict[str, Any]]:
        """Returns detailed kinematic and diagnostic metrics for each digit."""
        res = {}
        for f in [FingerName.THUMB, FingerName.INDEX, FingerName.MIDDLE, FingerName.RING, FingerName.LITTLE]:
            d = self.get(f)
            res[f.value.capitalize()] = {
                "state": d.state.value,
                "confidence": round(float(d.confidence), 2),
                "extension_ratio": round(float(d.extension_ratio), 2),
                "mcp_deg": round(float(d.mcp_flexion_deg), 1),
                "pip_deg": round(float(d.pip_flexion_deg), 1),
                "dip_deg": round(float(d.dip_flexion_deg), 1),
                "contact_target": d.contact_target,
                "diagnostics": d.diagnostics,
            }
        return res

    def summary(self) -> str:
        """
        Produces the canonical Gestura Level 0 human-readable readout:
        Thumb:    folded
        Index:    extended
        Middle:   folded
        Ring:     folded
        Little:   folded
        """
        lines = [
            f"Thumb:    {self.thumb.state.value}",
            f"Index:    {self.index.state.value}",
            f"Middle:   {self.middle.state.value}",
            f"Ring:     {self.ring.state.value}",
            f"Little:   {self.little.state.value}",
        ]
        return "\n".join(lines)

    def all_are(self, state: FingerStateEnum) -> bool:
        """Checks if all 5 digits share the same state."""
        return all(
            d.state == state
            for d in (self.thumb, self.index, self.middle, self.ring, self.little)
        )

    def is_pattern(
        self,
        thumb: Optional[FingerStateEnum] = None,
        index: Optional[FingerStateEnum] = None,
        middle: Optional[FingerStateEnum] = None,
        ring: Optional[FingerStateEnum] = None,
        little: Optional[FingerStateEnum] = None,
    ) -> bool:
        """Checks if hand satisfies a specified pattern of finger states."""
        if thumb is not None and self.thumb.state != thumb:
            return False
        if index is not None and self.index.state != index:
            return False
        if middle is not None and self.middle.state != middle:
            return False
        if ring is not None and self.ring.state != ring:
            return False
        if little is not None and self.little.state != little:
            return False
        return True


class FingerStateClassifier:
    """
    Robust geometric and kinematic classifier for Level 0 Finger States.
    
    Transforms 3D hand landmarks into a hand-local coordinate frame invariant to
    global rotation, pitch, yaw, scale, and distance, computing 3D joint angles,
    relative depths, planar projections, and inter-digit proximities.
    """

    def __init__(
        self,
        pinch_distance_threshold: float = 0.36,
        touch_distance_threshold: float = 0.24,
        min_visibility_threshold: float = 0.45,
    ):
        self.pinch_thresh = pinch_distance_threshold
        self.touch_thresh = touch_distance_threshold
        self.min_vis = min_visibility_threshold

    @staticmethod
    def _compute_angle_deg(p1: np.ndarray, p_vertex: np.ndarray, p2: np.ndarray) -> float:
        """Computes 3D angle at p_vertex between (p1 - p_vertex) and (p2 - p_vertex) in degrees."""
        v1 = p1 - p_vertex
        v2 = p2 - p_vertex
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        cosine = np.dot(v1, v2) / (norm1 * norm2)
        cosine = np.clip(cosine, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    def _build_hand_local_frame(
        self, raw_pts: np.ndarray, handedness: str = "Right"
    ) -> Tuple[np.ndarray, np.ndarray, float, str]:
        """
        Builds hand-local orthonormal frame invariant to palm vs dorsal view:
        - Origin: Wrist (Landmark 0)
        - Y: Longitudinal axis from Wrist to Middle MCP (Landmark 9)
        - Normal Z: Palmar normal (pointing out of palm, volar direction)
        - X: Lateral axis pointing towards radial / thumb side
        Returns:
            local_pts: (21, 3) landmarks in local frame normalized by d_ref
            rotation_matrix: (3, 3) transform matrix
            d_ref: characteristic distance (wrist to middle MCP)
            palm_facing: "PALM" (front), "DORSAL" (back), or "SIDE" (edge-on)
        """
        p0 = raw_pts[0]   # Wrist
        p5 = raw_pts[5]   # Index MCP
        p9 = raw_pts[9]   # Middle MCP
        p17 = raw_pts[17] # Pinky MCP

        d_ref = float(np.linalg.norm(p9 - p0))
        if d_ref < 1e-5:
            d_ref = 1.0

        v_y = (p9 - p0) / d_ref

        # Compute anatomical palmar normal pointing OUT of the palm in camera coordinates:
        # In MediaPipe camera coords (+X Right, +Y Down, +Z into screen):
        # For Right hand with palm facing camera: (p17 - p5) points Right (+X), (p9 - p0) points Up (-Y).
        # (+X) x (-Y) = -Z (pointing towards camera).
        # For Left hand with palm facing camera: (p5 - p17) points Right (+X), (p9 - p0) points Up (-Y).
        if handedness == "Left":
            n_palm = np.cross(p5 - p17, p9 - p0)
        else:
            n_palm = np.cross(p17 - p5, p9 - p0)

        norm_n = np.linalg.norm(n_palm)
        if norm_n < 1e-5:
            v_z = np.array([0.0, 0.0, -1.0])
            palm_facing = "PALM"
        else:
            v_z = n_palm / norm_n
            # In camera coords, camera is at origin looking down +Z.
            # v_z[2] < -0.15 => Palmar normal points towards camera => PALM (front of hand)
            # v_z[2] > 0.15  => Palmar normal points away from camera => DORSAL (back of hand)
            # else          => Edge-on / SIDE
            if v_z[2] < -0.15:
                palm_facing = "PALM"
            elif v_z[2] > 0.15:
                palm_facing = "DORSAL"
            else:
                palm_facing = "SIDE"

        # Lateral axis v_x pointing towards the radial side (towards the thumb)
        v_x = np.cross(v_y, v_z)
        norm_x = np.linalg.norm(v_x)
        if norm_x > 1e-5:
            v_x = v_x / norm_x
        else:
            v_x = np.array([1.0, 0.0, 0.0])

        # R transforms world vectors into local hand frame: local = (p - p0) @ R / d_ref
        R = np.column_stack([v_x, v_y, v_z])
        local_pts = ((raw_pts - p0) @ R) / d_ref

        return local_pts, R, d_ref, palm_facing

    def classify_hand(
        self,
        raw_landmarks: np.ndarray,
        handedness: str = "Right",
        detection_confidence: float = 1.0,
        visibilities: Optional[List[float]] = None,
        timestamp: float = 0.0,
    ) -> HandFingerStates:
        """
        Classifies all 5 fingers of the hand into their discrete states.
        
        Args:
            raw_landmarks: (21, 3) 3D landmark array.
            handedness: 'Right' or 'Left'.
            detection_confidence: Overall detector confidence [0, 1].
            visibilities: Optional per-landmark visibility scores [0, 1].
            timestamp: Time of frame capture.
        """
        local_pts, _, d_ref, palm_facing = self._build_hand_local_frame(raw_landmarks, handedness)

        # Check frame edge clipping or low detector confidence -> uncertain
        p0 = raw_landmarks[0]
        p5 = raw_landmarks[5]
        p9 = raw_landmarks[9]
        p17 = raw_landmarks[17]

        inner_hand_visible = bool(palm_facing == "PALM")
        is_edge = (p0[0] < 0.04 or p0[0] > 0.96 or p0[1] < 0.04 or p0[1] > 0.96)
        global_uncertain = (detection_confidence < 0.40) or is_edge

        # Tips & MCP references
        p_thumb_tip = raw_landmarks[4]
        p_index_tip = raw_landmarks[8]
        p_middle_tip = raw_landmarks[12]
        p_ring_tip = raw_landmarks[16]
        p_little_tip = raw_landmarks[20]

        tips = {
            FingerName.THUMB: p_thumb_tip,
            FingerName.INDEX: p_index_tip,
            FingerName.MIDDLE: p_middle_tip,
            FingerName.RING: p_ring_tip,
            FingerName.LITTLE: p_little_tip,
        }

        # Palm center reference in world coords
        p_palm_center = (raw_landmarks[0] + raw_landmarks[5] + raw_landmarks[9] + raw_landmarks[17]) / 4.0

        # Calculate extension ratios: dist(tip, wrist) / dist(mcp, wrist) for fingers 2-5,
        # and biomechanically grounded continuous ratio for thumb.
        p_wrist = raw_landmarks[0]
        p_index_mcp = raw_landmarks[5]

        ext_ratios = {}
        for f_name, idx_map in FINGER_INDICES.items():
            if f_name == "thumb":
                p1 = raw_landmarks[1]
                p2 = raw_landmarks[2]
                p3 = raw_landmarks[3]
                p4 = raw_landmarks[4]
                L_thumb = np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2) + np.linalg.norm(p4 - p3)
                straightness_thumb = float(np.linalg.norm(p4 - p1) / max(L_thumb, 1e-4))
                abduction_thumb = float(np.linalg.norm(p4 - p_index_mcp) / d_ref)
                d_thumb_palm_init = float(np.linalg.norm(p4 - p_palm_center) / d_ref)
                thumb_ratio = straightness_thumb * (0.50 + 1.10 * abduction_thumb + 0.40 * d_thumb_palm_init)
                ext_ratios[f_name] = float(np.clip(thumb_ratio, 0.35, 2.20))
            else:
                d_tip = np.linalg.norm(raw_landmarks[idx_map["tip"]] - p_wrist)
                d_mcp = max(np.linalg.norm(raw_landmarks[idx_map["mcp"]] - p_wrist), 1e-4)
                ext_ratios[f_name] = float(d_tip / d_mcp)

        # Calculate inter-digit tip-to-thumb distances normalized by d_ref
        thumb_tip_dist = {
            FingerName.INDEX: float(np.linalg.norm(p_index_tip - p_thumb_tip) / d_ref),
            FingerName.MIDDLE: float(np.linalg.norm(p_middle_tip - p_thumb_tip) / d_ref),
            FingerName.RING: float(np.linalg.norm(p_ring_tip - p_thumb_tip) / d_ref),
            FingerName.LITTLE: float(np.linalg.norm(p_little_tip - p_thumb_tip) / d_ref),
        }

        # 1. Classify Fingers 2-5 (Index, Middle, Ring, Little)
        classified_digits: Dict[FingerName, FingerStateDetail] = {}
        pinching_digits: List[FingerName] = []

        for f_name in [FingerName.INDEX, FingerName.MIDDLE, FingerName.RING, FingerName.LITTLE]:
            feat_key = DIGIT_TO_FEATURE_KEY[f_name]
            indices = FINGER_INDICES[feat_key]
            mcp_idx = indices["mcp"]
            pip_idx = indices["pip"]
            dip_idx = indices["dip"]
            tip_idx = indices["tip"]

            # Visibility check
            if visibilities:
                digit_vis = min(
                    visibilities[mcp_idx],
                    visibilities[pip_idx],
                    visibilities[dip_idx],
                    visibilities[tip_idx],
                )
            else:
                digit_vis = 1.0

            if global_uncertain:
                classified_digits[f_name] = FingerStateDetail(
                    finger=f_name,
                    state=FingerStateEnum.UNCERTAIN,
                    confidence=0.30,
                    extension_ratio=ext_ratios[feat_key],
                    mcp_flexion_deg=0.0,
                    pip_flexion_deg=0.0,
                    dip_flexion_deg=0.0,
                    diagnostics=["Low tracking confidence or frame edge clipping"],
                )
                continue

            # If finger is not in view / occluded from camera, do not assume position or track it
            if digit_vis < self.min_vis:
                classified_digits[f_name] = FingerStateDetail(
                    finger=f_name,
                    state=FingerStateEnum.UNCERTAIN,
                    confidence=0.0,
                    extension_ratio=ext_ratios[feat_key],
                    mcp_flexion_deg=0.0,
                    pip_flexion_deg=0.0,
                    dip_flexion_deg=0.0,
                    diagnostics=["Not in view / occluded from sensor"],
                )
                continue

            # 3D Joint Flexions (180 deg = straight, smaller = flexed)
            # We measure flexion angle as deviation from 180 (0 = fully extended)
            angle_mcp = 180.0 - self._compute_angle_deg(raw_landmarks[0], raw_landmarks[mcp_idx], raw_landmarks[pip_idx])
            angle_pip = 180.0 - self._compute_angle_deg(raw_landmarks[mcp_idx], raw_landmarks[pip_idx], raw_landmarks[dip_idx])
            angle_dip = 180.0 - self._compute_angle_deg(raw_landmarks[pip_idx], raw_landmarks[dip_idx], raw_landmarks[tip_idx])

            ext_r = ext_ratios[feat_key]
            d_to_palm = float(np.linalg.norm(tips[f_name] - p_palm_center) / d_ref)
            d_to_thumb = thumb_tip_dist[f_name]

            # Hand-local coordinates for digit
            loc_tip = local_pts[tip_idx]
            loc_mcp = local_pts[mcp_idx]
            loc_pip = local_pts[pip_idx]

            state = FingerStateEnum.UNCERTAIN
            confidence = 0.85
            contact_target = None
            diags = []

            # --- 1. Check CROSSED (Middle over Index, etc. - requires extended digits) ---
            if f_name == FingerName.MIDDLE and angle_pip <= 45.0:
                loc_index_tip = local_pts[8]
                cross_offset = (loc_tip[0] - loc_index_tip[0]) if handedness == "Right" else (loc_index_tip[0] - loc_tip[0])
                if cross_offset < -0.04 and abs(loc_tip[1] - loc_index_tip[1]) < 0.22:
                    state = FingerStateEnum.CROSSED
                    contact_target = "index"
                    confidence = 0.90
                    diags.append("Middle finger crossed over index ray")

            elif f_name == FingerName.INDEX and FingerName.MIDDLE in classified_digits:
                if classified_digits[FingerName.MIDDLE].state == FingerStateEnum.CROSSED:
                    state = FingerStateEnum.CROSSED
                    contact_target = "middle"
                    confidence = 0.90
                    diags.append("Index crossed under middle")

            # --- 2. Check HOOKED (Claw: MCP open <= 45 deg, PIP/DIP sharply flexed >= 65 deg, but fingertip NOT curled into palm) ---
            if state == FingerStateEnum.UNCERTAIN:
                if angle_mcp <= 45.0 and angle_pip >= 65.0 and angle_dip >= 45.0 and d_to_palm > 0.42:
                    state = FingerStateEnum.HOOKED
                    confidence = 0.92
                    diags.append(f"Claw hook: MCP={angle_mcp:.1f}deg open, PIP={angle_pip:.1f}deg bent")

            # --- 3. Check FOLDED / TUCKED (Balled into fist: All joints curled deep into palm) ---
            if state == FingerStateEnum.UNCERTAIN:
                # Curled check works for both palm and dorsal views
                # When dorsal side faces camera, MCP and PIP angles are flexed towards palm
                is_curled = (
                    (ext_r <= 0.96 or d_to_palm <= 0.54)
                    and (angle_pip >= 55.0 or angle_mcp >= 40.0)
                    and (d_to_palm <= 0.62 or ext_r <= 0.92)
                ) or (
                    # Direct dorsal view curled knuckle signature
                    palm_facing == "DORSAL" and angle_mcp >= 40.0 and angle_pip >= 48.0 and ext_r <= 1.05
                )

                if is_curled:
                    d_thumb_knuckle = float(np.linalg.norm(tips[f_name] - raw_landmarks[2]) / d_ref)
                    if d_thumb_knuckle < 0.28 and loc_tip[2] > 0.02:
                        state = FingerStateEnum.TUCKED
                        contact_target = "thumb_base"
                        confidence = 0.88
                        diags.append("Curled and tucked under thumb/knuckles")
                    else:
                        state = FingerStateEnum.FOLDED
                        confidence = float(np.clip(0.65 + (1.0 - min(ext_r, 1.0)) * 0.4, 0.70, 0.98))
                        diags.append(f"Curled into palm (ext_r={ext_r:.2f}, PIP={angle_pip:.1f}deg, MCP={angle_mcp:.1f}deg)")

            # --- 4. Check PINCHING (Opposing thumb tip while not balled into palm) ---
            if state == FingerStateEnum.UNCERTAIN:
                if d_to_thumb <= self.pinch_thresh:
                    state = FingerStateEnum.PINCHING
                    contact_target = "thumb"
                    confidence = float(np.clip(1.0 - (d_to_thumb / self.pinch_thresh) * 0.4, 0.65, 0.99))
                    pinching_digits.append(f_name)
                    diags.append(f"Tip close to thumb ({d_to_thumb:.2f} d_ref)")

            # --- 5. Check TOUCHING (Adducted straight fingers touching, or pad touching palm) ---
            if state == FingerStateEnum.UNCERTAIN:
                neighbor_dists = []
                for other_f, other_tip in tips.items():
                    if other_f != f_name and other_f != FingerName.THUMB:
                        neighbor_dists.append((other_f.value, float(np.linalg.norm(tips[f_name] - other_tip) / d_ref)))
                closest_neighbor, closest_dist = min(neighbor_dists, key=lambda x: x[1])

                if closest_dist <= self.touch_thresh and ext_r >= 0.96 and angle_pip <= 38.0:
                    state = FingerStateEnum.TOUCHING
                    contact_target = closest_neighbor
                    confidence = 0.85
                    diags.append(f"Adducted touching {closest_neighbor} ({closest_dist:.2f} d_ref)")
                elif d_to_palm <= self.touch_thresh and ext_r < 0.92 and angle_pip < 80.0:
                    state = FingerStateEnum.TOUCHING
                    contact_target = "palm"
                    confidence = 0.82
                    diags.append(f"Tip touching palm ({d_to_palm:.2f} d_ref)")

            # --- 6. Check EXTENDED / CURVED / RELAXED ---
            if state == FingerStateEnum.UNCERTAIN:
                if ext_r >= 1.15 and angle_pip <= 28.0 and angle_dip <= 25.0:
                    state = FingerStateEnum.EXTENDED
                    confidence = float(np.clip(0.70 + (ext_r - 1.15) * 0.8, 0.75, 0.99))
                    diags.append(f"Straight extended (ext_r={ext_r:.2f}, PIP={angle_pip:.1f}deg)")
                elif 32.0 <= angle_pip <= 78.0 and 22.0 <= angle_dip <= 65.0 and 0.88 <= ext_r <= 1.14:
                    state = FingerStateEnum.CURVED
                    confidence = 0.85
                    diags.append(f"Curved arc across joints (PIP={angle_pip:.1f}deg, DIP={angle_dip:.1f}deg)")
                elif 12.0 <= angle_pip <= 38.0 and 8.0 <= angle_dip <= 32.0 and 0.98 <= ext_r <= 1.18:
                    state = FingerStateEnum.RELAXED
                    confidence = 0.82
                    diags.append(f"Neutral resting drop (PIP={angle_pip:.1f}deg)")
                elif ext_r >= 1.10:
                    state = FingerStateEnum.EXTENDED
                    confidence = 0.65
                elif ext_r <= 0.90:
                    state = FingerStateEnum.FOLDED
                    confidence = 0.65
                else:
                    state = FingerStateEnum.RELAXED
                    confidence = 0.60

            classified_digits[f_name] = FingerStateDetail(
                finger=f_name,
                state=state,
                confidence=confidence,
                extension_ratio=ext_r,
                mcp_flexion_deg=angle_mcp,
                pip_flexion_deg=angle_pip,
                dip_flexion_deg=angle_dip,
                contact_target=contact_target,
                diagnostics=diags,
            )

        # 2. Classify Thumb (Unique biomechanics & occlusion checks)
        p1 = raw_landmarks[1]
        p2 = raw_landmarks[2]
        p3 = raw_landmarks[3]
        p4 = raw_landmarks[4]
        p5 = raw_landmarks[5]

        # Anatomical segment lengths normalized by d_ref
        L1 = float(np.linalg.norm(p2 - p1) / d_ref)
        L2 = float(np.linalg.norm(p3 - p2) / d_ref)
        L3 = float(np.linalg.norm(p4 - p3) / d_ref)
        L_thumb_norm = L1 + L2 + L3

        loc_thumb_tip = local_pts[4]
        loc_thumb_ip = local_pts[3]
        loc_thumb_mcp = local_pts[2]

        angle_thumb_mcp = 180.0 - self._compute_angle_deg(raw_landmarks[1], raw_landmarks[2], raw_landmarks[3])
        angle_thumb_ip = 180.0 - self._compute_angle_deg(raw_landmarks[2], raw_landmarks[3], raw_landmarks[4])
        ext_r_thumb = ext_ratios["thumb"]

        # Distances to palm and knuckles
        d_thumb_palm = float(np.linalg.norm(p_thumb_tip - p_palm_center) / d_ref)
        d_thumb_to_index_mcp = float(np.linalg.norm(p_thumb_tip - p5) / d_ref)
        d_thumb_to_pinky_mcp = float(np.linalg.norm(p_thumb_tip - raw_landmarks[17]) / d_ref)

        # Multi-tiered obstruction tests:
        # A) Optical visibility from camera frame
        thumb_vis = 1.0
        if visibilities and len(visibilities) > 4:
            thumb_vis = min(visibilities[1], visibilities[2], visibilities[3], visibilities[4])

        # B) Kinematic segment collapse / regression hallucination
        is_segment_collapsed = (L3 < 0.03 or L_thumb_norm < 0.35 or L_thumb_norm > 2.40)

        # C) Behind-the-palm depth occlusion from camera viewpoint
        # If camera sees PALM, thumb is behind palm if loc_z < -0.12 (on dorsal side behind palm)
        # If camera sees DORSAL, thumb is on radial border (+X); it's only hidden if tucked deep inside palm (loc_z > 0.22 and centered)
        if palm_facing == "PALM":
            is_behind_palm = bool(loc_thumb_tip[2] < -0.12 and abs(loc_thumb_tip[0]) < 0.32 and loc_thumb_tip[1] > 0.05)
        elif palm_facing == "DORSAL":
            is_behind_palm = bool(loc_thumb_tip[2] > 0.22 and abs(loc_thumb_tip[0]) < 0.25 and loc_thumb_tip[1] > 0.05)
        else:
            is_behind_palm = False

        other_fingers_curled = (
            classified_digits[FingerName.INDEX].state in (FingerStateEnum.FOLDED, FingerStateEnum.TUCKED)
            or classified_digits[FingerName.MIDDLE].state in (FingerStateEnum.FOLDED, FingerStateEnum.TUCKED)
            or (palm_facing == "DORSAL" and classified_digits[FingerName.INDEX].extension_ratio <= 1.0)
        )

        # --- 1. Check PINCHING (High-priority active micro-interaction) ---
        if pinching_digits:
            classified_digits[FingerName.THUMB] = FingerStateDetail(
                finger=FingerName.THUMB,
                state=FingerStateEnum.PINCHING,
                confidence=0.96,
                extension_ratio=ext_r_thumb,
                mcp_flexion_deg=angle_thumb_mcp,
                pip_flexion_deg=angle_thumb_ip,
                dip_flexion_deg=0.0,
                contact_target=pinching_digits[0].value,
                diagnostics=[f"Pinching with {pinching_digits[0].value}"],
            )
        elif global_uncertain:
            classified_digits[FingerName.THUMB] = FingerStateDetail(
                finger=FingerName.THUMB,
                state=FingerStateEnum.UNCERTAIN,
                confidence=0.25,
                extension_ratio=min(ext_r_thumb, 0.40),
                mcp_flexion_deg=angle_thumb_mcp,
                pip_flexion_deg=angle_thumb_ip,
                dip_flexion_deg=0.0,
                diagnostics=["Global tracking uncertainty or frame edge clipping"],
            )
        # If thumb is not in view / occluded / collapsed, do not assume position or track it
        elif thumb_vis < self.min_vis or is_segment_collapsed or (is_behind_palm and not other_fingers_curled):
            classified_digits[FingerName.THUMB] = FingerStateDetail(
                finger=FingerName.THUMB,
                state=FingerStateEnum.UNCERTAIN,
                confidence=0.0,
                extension_ratio=ext_r_thumb,
                mcp_flexion_deg=0.0,
                pip_flexion_deg=0.0,
                dip_flexion_deg=0.0,
                diagnostics=["Thumb not in view / occluded from sensor"],
            )
        else:
            thumb_state = FingerStateEnum.UNCERTAIN
            thumb_conf = 0.85
            thumb_contact = None
            thumb_diags = []

            # --- Check TUCKED (Thumb tucked inside fist / under fingers across palm) ---
            if (
                (d_thumb_to_pinky_mcp < 0.60 and d_thumb_palm < 0.40 and other_fingers_curled)
                or (is_behind_palm and other_fingers_curled)
            ):
                thumb_state = FingerStateEnum.TUCKED
                thumb_contact = "palm_core"
                thumb_conf = 0.92
                thumb_diags.append("Tucked across palm beneath curled fingers")

            # --- Check CROSSED (Thumb crossed over closed fingers) ---
            elif (
                d_thumb_to_index_mcp < 0.42
                and other_fingers_curled
            ):
                thumb_state = FingerStateEnum.CROSSED
                thumb_contact = "index_knuckle"
                thumb_conf = 0.88
                thumb_diags.append("Crossed over index/middle folded knuckles")

            # --- Check TOUCHING (Touching side of index finger without pinch) ---
            elif d_thumb_to_index_mcp < 0.28 and angle_thumb_ip < 35.0:
                thumb_state = FingerStateEnum.TOUCHING
                thumb_contact = "index_base"
                thumb_conf = 0.84
                thumb_diags.append("Touching index lateral base")

            # --- Check HOOKED (IP flexed acutely while MCP abducted away from palm) ---
            elif angle_thumb_ip >= 50.0 and angle_thumb_mcp <= 35.0 and d_thumb_palm >= 0.40:
                thumb_state = FingerStateEnum.HOOKED
                thumb_conf = 0.88
                thumb_diags.append(f"Hooked thumb IP={angle_thumb_ip:.1f}deg")

            # --- Check EXTENDED (Hitchhiker / radial abduction / spread) ---
            elif (
                (ext_r_thumb >= 1.15 and angle_thumb_ip <= 48.0 and d_thumb_to_index_mcp >= 0.35)
                or (ext_r_thumb >= 1.30 and angle_thumb_ip <= 45.0)
                or (palm_facing == "DORSAL" and ext_r_thumb >= 1.12 and d_thumb_to_index_mcp >= 0.34)
            ):
                thumb_state = FingerStateEnum.EXTENDED
                thumb_conf = float(np.clip(0.75 + (ext_r_thumb - 1.15) * 0.8, 0.75, 0.99))
                thumb_diags.append(f"Extended outward (ext_r={ext_r_thumb:.2f}, IP={angle_thumb_ip:.1f}deg)")

            # --- Check FOLDED (Curled against palm) ---
            elif ext_r_thumb <= 0.85 or d_thumb_palm <= 0.38:
                thumb_state = FingerStateEnum.FOLDED
                thumb_conf = 0.88
                thumb_diags.append(f"Folded into lateral palm (ext_r={ext_r_thumb:.2f})")

            # --- Check CURVED ---
            elif 25.0 <= angle_thumb_ip <= 55.0:
                thumb_state = FingerStateEnum.CURVED
                thumb_conf = 0.82
                thumb_diags.append(f"Curved thumb IP={angle_thumb_ip:.1f}deg")

            # --- Check RELAXED (Adducted resting beside palm) ---
            else:
                thumb_state = FingerStateEnum.RELAXED
                thumb_conf = 0.78
                thumb_diags.append(f"Resting neutral thumb (ext_r={ext_r_thumb:.2f})")

            classified_digits[FingerName.THUMB] = FingerStateDetail(
                finger=FingerName.THUMB,
                state=thumb_state,
                confidence=thumb_conf,
                extension_ratio=ext_r_thumb,
                mcp_flexion_deg=angle_thumb_mcp,
                pip_flexion_deg=angle_thumb_ip,
                dip_flexion_deg=0.0,
                contact_target=thumb_contact,
                diagnostics=thumb_diags,
            )

        return HandFingerStates(
            thumb=classified_digits[FingerName.THUMB],
            index=classified_digits[FingerName.INDEX],
            middle=classified_digits[FingerName.MIDDLE],
            ring=classified_digits[FingerName.RING],
            little=classified_digits[FingerName.LITTLE],
            timestamp=timestamp,
            palm_facing=palm_facing,
        )


if __name__ == "__main__":
    print("=" * 64)
    print("GESTURA LEVEL 0: FINGER STATE CLASSIFICATION ENGINE")
    print("Vocabulary: extended, folded, curved, relaxed, tucked, hooked,")
    print("            touching, pinching, crossed, uncertain")
    print("=" * 64)

    # Synthetic Pointing Hand Landmarks
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.5, 0.70, 0.0]  # Wrist
    # Thumb folded
    pts[1] = [0.46, 0.65, 0.0]
    pts[2] = [0.45, 0.62, 0.0]
    pts[3] = [0.47, 0.60, 0.0]
    pts[4] = [0.48, 0.58, 0.0]
    # Index extended
    pts[5] = [0.47, 0.52, 0.0]
    pts[6] = [0.47, 0.44, 0.0]
    pts[7] = [0.47, 0.38, 0.0]
    pts[8] = [0.47, 0.32, 0.0]
    # Middle folded
    pts[9] = [0.50, 0.50, 0.0]
    pts[10] = [0.50, 0.46, -0.02]
    pts[11] = [0.50, 0.52, -0.05]
    pts[12] = [0.50, 0.56, -0.05]
    # Ring folded
    pts[13] = [0.53, 0.52, 0.0]
    pts[14] = [0.53, 0.48, -0.02]
    pts[15] = [0.53, 0.54, -0.05]
    pts[16] = [0.53, 0.56, -0.05]
    # Pinky folded
    pts[17] = [0.56, 0.55, 0.0]
    pts[18] = [0.56, 0.51, -0.02]
    pts[19] = [0.56, 0.57, -0.05]
    pts[20] = [0.56, 0.56, -0.05]

    classifier = FingerStateClassifier()
    result = classifier.classify_hand(pts)

    print("\n[Canonical Pointing Hand Observation]:")
    print(result.summary())
    print("\n[Detailed Diagnostics]:")
    for fname in [FingerName.THUMB, FingerName.INDEX, FingerName.MIDDLE, FingerName.RING, FingerName.LITTLE]:
        detail = result.get(fname)
        print(f"  {fname.value.capitalize():<7} -> State: {detail.state.value:<10} (Conf: {detail.confidence:.2f}, Ext: {detail.extension_ratio:.2f})")
    print("=" * 64)
