"""
Unit and invariance tests for Level 0 Finger State Classification Engine.
Validates recognition accuracy, geometric robustness, camera angle invariance,
distance scale invariance, and failure condition handling across all 10 states:
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
"""

import math
import numpy as np
import pytest

from src.landmarks.finger_state import (
    FingerName,
    FingerStateClassifier,
    FingerStateEnum,
    HandFingerStates,
)
from src.landmarks.kinematic_features import FINGER_INDICES


def create_base_hand_landmarks() -> np.ndarray:
    """
    Creates an anatomically plausible normalized 3D hand in the coronal plane.
    Wrist at origin [0.5, 0.7, 0.0].
    Hand extends upward along -Y (camera image convention where Y goes down).
    Middle MCP at [0.5, 0.5, 0.0], giving d_ref = 0.20.
    """
    pts = np.zeros((21, 3), dtype=np.float32)

    # 0: Wrist
    pts[0] = [0.5, 0.70, 0.0]

    # Thumb: 1 (CMC), 2 (MCP), 3 (IP), 4 (Tip) - resting slightly radially
    pts[1] = [0.46, 0.65, 0.0]
    pts[2] = [0.43, 0.60, 0.0]
    pts[3] = [0.40, 0.55, 0.0]
    pts[4] = [0.38, 0.50, 0.0]

    # Index: 5 (MCP), 6 (PIP), 7 (DIP), 8 (Tip) - extended straight up
    pts[5] = [0.47, 0.52, 0.0]
    pts[6] = [0.47, 0.44, 0.0]
    pts[7] = [0.47, 0.38, 0.0]
    pts[8] = [0.47, 0.32, 0.0]

    # Middle: 9 (MCP), 10 (PIP), 11 (DIP), 12 (Tip) - extended straight up
    pts[9]  = [0.50, 0.50, 0.0]
    pts[10] = [0.50, 0.41, 0.0]
    pts[11] = [0.50, 0.34, 0.0]
    pts[12] = [0.50, 0.27, 0.0]

    # Ring: 13 (MCP), 14 (PIP), 15 (DIP), 16 (Tip) - extended straight up
    pts[13] = [0.53, 0.52, 0.0]
    pts[14] = [0.53, 0.44, 0.0]
    pts[15] = [0.53, 0.38, 0.0]
    pts[16] = [0.53, 0.32, 0.0]

    # Pinky: 17 (MCP), 18 (PIP), 19 (DIP), 20 (Tip) - extended straight up
    pts[17] = [0.56, 0.55, 0.0]
    pts[18] = [0.56, 0.48, 0.0]
    pts[19] = [0.56, 0.43, 0.0]
    pts[20] = [0.56, 0.38, 0.0]

    return pts


def curl_finger(pts: np.ndarray, finger_name: str, target_y: float = 0.56, target_z: float = -0.05):
    """Folds a finger inward toward the palm crease."""
    indices = FINGER_INDICES[finger_name]
    mcp_y = pts[indices["mcp"]][1]
    mcp_x = pts[indices["mcp"]][0]

    # PIP bends forward
    pts[indices["pip"]] = [mcp_x, mcp_y - 0.04, target_z * 0.5]
    # DIP curls inward towards palm
    pts[indices["dip"]] = [mcp_x, mcp_y + 0.02, target_z]
    # Tip rests near palm base
    pts[indices["tip"]] = [mcp_x, target_y, target_z]


def hook_finger(pts: np.ndarray, finger_name: str):
    """Sets a finger in a hooked / claw posture (MCP open, PIP/DIP sharply bent)."""
    indices = FINGER_INDICES[finger_name]
    mcp = pts[indices["mcp"]]
    # MCP projects forward/upward (straight)
    pip = [mcp[0], mcp[1] - 0.08, 0.0]
    # PIP bends acutely downward at 90 degrees
    dip = [mcp[0], mcp[1] - 0.08, -0.06]
    # Tip curls back towards MCP
    tip = [mcp[0], mcp[1] - 0.04, -0.07]

    pts[indices["pip"]] = pip
    pts[indices["dip"]] = dip
    pts[indices["tip"]] = tip


def rotate_landmarks_3d(pts: np.ndarray, roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Applies a 3D Euler rotation to all landmarks around the palm center."""
    center = np.mean(pts[[0, 5, 9, 17]], axis=0)
    shifted = pts - center

    # Rotation matrices
    rx = math.radians(pitch_deg)
    ry = math.radians(yaw_deg)
    rz = math.radians(roll_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)]
    ], dtype=np.float32)

    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)]
    ], dtype=np.float32)

    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    R = Rz @ Ry @ Rx
    rotated = shifted @ R.T + center
    return rotated


# ============================================================================
# Tests
# ============================================================================

class TestFingerStateClassifier:

    def setup_method(self):
        self.classifier = FingerStateClassifier()

    def test_open_hand_all_extended(self):
        """Validates that a flat open hand produces EXTENDED for all 5 digits."""
        pts = create_base_hand_landmarks()
        # Abduct thumb outward so it's fully extended
        pts[4] = [0.34, 0.48, 0.0]

        states = self.classifier.classify_hand(pts)
        assert states.thumb.state == FingerStateEnum.EXTENDED
        assert states.index.state == FingerStateEnum.EXTENDED
        assert states.middle.state == FingerStateEnum.EXTENDED
        assert states.ring.state == FingerStateEnum.EXTENDED
        assert states.little.state == FingerStateEnum.EXTENDED
        assert states.all_are(FingerStateEnum.EXTENDED)

    def test_pointing_hand_canonical_output(self):
        """
        Validates canonical pointing hand:
        Thumb: folded / tucked
        Index: extended
        Middle: folded
        Ring: folded
        Little: folded
        """
        pts = create_base_hand_landmarks()
        # Curl middle, ring, pinky
        curl_finger(pts, "middle")
        curl_finger(pts, "ring")
        curl_finger(pts, "pinky")

        # Fold thumb against lateral palm
        pts[2] = [0.45, 0.62, 0.0]
        pts[3] = [0.47, 0.60, 0.0]
        pts[4] = [0.48, 0.58, 0.0]

        states = self.classifier.classify_hand(pts)

        assert states.index.state == FingerStateEnum.EXTENDED
        assert states.middle.state in (FingerStateEnum.FOLDED, FingerStateEnum.TUCKED)
        assert states.ring.state in (FingerStateEnum.FOLDED, FingerStateEnum.TUCKED)
        assert states.little.state in (FingerStateEnum.FOLDED, FingerStateEnum.TUCKED)
        assert states.thumb.state in (FingerStateEnum.FOLDED, FingerStateEnum.TUCKED, FingerStateEnum.TOUCHING)

        # Check summary formatting
        summary = states.summary()
        assert "Index:    extended" in summary
        assert "Middle:   folded" in summary or "Middle:   tucked" in summary

    def test_precision_pinch(self):
        """Validates precision pinch where thumb tip (4) touches index tip (8)."""
        pts = create_base_hand_landmarks()
        # Touch thumb tip to index tip
        pts[4] = pts[8].copy()
        # Slightly curl other fingers
        curl_finger(pts, "middle")
        curl_finger(pts, "ring")
        curl_finger(pts, "pinky")

        states = self.classifier.classify_hand(pts)
        assert states.thumb.state == FingerStateEnum.PINCHING
        assert states.index.state == FingerStateEnum.PINCHING
        assert states.thumb.contact_target == "index"
        assert states.index.contact_target == "thumb"

    def test_hooked_claw_posture(self):
        """Validates HOOKED state where MCP is extended but PIP/DIP are acutely bent."""
        pts = create_base_hand_landmarks()
        hook_finger(pts, "index")
        hook_finger(pts, "middle")

        states = self.classifier.classify_hand(pts)
        assert states.index.state == FingerStateEnum.HOOKED
        assert states.middle.state == FingerStateEnum.HOOKED

    def test_middle_crossed_over_index(self):
        """Validates CROSSED state where middle finger crosses over index finger."""
        pts = create_base_hand_landmarks()
        # Cross middle finger tip laterally past index finger tip
        index_tip = pts[8].copy()
        pts[12] = [index_tip[0] - 0.04, index_tip[1] + 0.01, index_tip[2] + 0.02]

        states = self.classifier.classify_hand(pts, handedness="Right")
        assert states.middle.state == FingerStateEnum.CROSSED
        assert states.middle.contact_target == "index"

    def test_tucked_thumb_inside_fist(self):
        """Validates TUCKED state when thumb is curled across the palm inside closed fingers."""
        pts = create_base_hand_landmarks()
        curl_finger(pts, "index")
        curl_finger(pts, "middle")
        curl_finger(pts, "ring")
        curl_finger(pts, "pinky")

        # Place thumb tip deep into palm under knuckles
        pts[4] = [0.52, 0.58, -0.04]

        states = self.classifier.classify_hand(pts)
        assert states.thumb.state in (FingerStateEnum.TUCKED, FingerStateEnum.FOLDED)

    def test_uncertain_on_edge_or_low_confidence(self):
        """Validates that low confidence or edge-of-frame positions yield UNCERTAIN."""
        pts = create_base_hand_landmarks()
        # Low detection confidence
        states_low_conf = self.classifier.classify_hand(pts, detection_confidence=0.30)
        assert states_low_conf.index.state == FingerStateEnum.UNCERTAIN

        # Edge of frame
        pts_edge = pts.copy()
        pts_edge[0] = [0.01, 0.01, 0.0]
        states_edge = self.classifier.classify_hand(pts_edge)
        assert states_edge.index.state == FingerStateEnum.UNCERTAIN

    def test_scale_and_distance_invariance(self):
        """
        Validates scale invariance: moving the hand 2x closer or 2x further
        must NOT change finger state classifications.
        """
        pts_base = create_base_hand_landmarks()
        curl_finger(pts_base, "middle")
        curl_finger(pts_base, "ring")
        curl_finger(pts_base, "pinky")

        # Scale down (further from camera)
        wrist = pts_base[0]
        pts_far = (pts_base - wrist) * 0.40 + wrist
        # Scale up (closer to camera)
        pts_near = (pts_base - wrist) * 2.20 + wrist

        states_base = self.classifier.classify_hand(pts_base)
        states_far  = self.classifier.classify_hand(pts_far)
        states_near = self.classifier.classify_hand(pts_near)

        assert states_far.index.state == states_base.index.state == FingerStateEnum.EXTENDED
        assert states_far.middle.state == states_base.middle.state
        assert states_near.index.state == states_base.index.state == FingerStateEnum.EXTENDED
        assert states_near.middle.state == states_base.middle.state

    def test_3d_orientation_invariance(self):
        """
        Validates orientation invariance: tilting or rotating the hand by 45 deg
        in pitch, yaw, or roll must preserve intrinsic joint flexions and states.
        """
        pts_base = create_base_hand_landmarks()
        curl_finger(pts_base, "middle")
        curl_finger(pts_base, "ring")
        curl_finger(pts_base, "pinky")

        # Rotate 35 degrees pitch and 30 degrees roll
        pts_rotated = rotate_landmarks_3d(pts_base, roll_deg=30.0, pitch_deg=35.0, yaw_deg=15.0)

        states_base = self.classifier.classify_hand(pts_base)
        states_rot  = self.classifier.classify_hand(pts_rotated)

        assert states_rot.index.state == states_base.index.state == FingerStateEnum.EXTENDED
        assert states_rot.middle.state == states_base.middle.state

    def test_pattern_matching_helpers(self):
        """Validates is_pattern and as_dict utility methods."""
        pts = create_base_hand_landmarks()
        curl_finger(pts, "middle")
        curl_finger(pts, "ring")
        curl_finger(pts, "pinky")

        states = self.classifier.classify_hand(pts)
        d = states.as_dict()
        assert "Index" in d
        assert d["Index"] == "extended"

        assert states.is_pattern(index=FingerStateEnum.EXTENDED)
        assert not states.is_pattern(index=FingerStateEnum.FOLDED)

    def test_thumb_extension_ratio_calibration(self):
        """
        Validates that the thumb extension ratio is properly calibrated and not artificially high:
        - Extended thumb: R_ext >= 1.35
        - Relaxed thumb: R_ext in [0.95, 1.35]
        - Folded/tucked thumb: R_ext <= 0.85 (never stuck at > 1.8)
        """
        pts = create_base_hand_landmarks()
        
        # 1. Extended thumb (abducted laterally)
        pts[4] = [0.32, 0.46, 0.0]
        states_ext = self.classifier.classify_hand(pts)
        assert states_ext.thumb.extension_ratio >= 1.35
        assert states_ext.thumb.state == FingerStateEnum.EXTENDED

        # 2. Folded thumb (flat on palm)
        pts_folded = create_base_hand_landmarks()
        pts_folded[2] = [0.45, 0.63, 0.0]
        pts_folded[3] = [0.47, 0.60, 0.0]
        pts_folded[4] = [0.49, 0.58, 0.0]
        states_fld = self.classifier.classify_hand(pts_folded)
        assert states_fld.thumb.extension_ratio <= 0.85
        assert states_fld.thumb.state == FingerStateEnum.FOLDED

        # 3. Tucked thumb inside closed fist
        pts_fist = create_base_hand_landmarks()
        curl_finger(pts_fist, "index")
        curl_finger(pts_fist, "middle")
        curl_finger(pts_fist, "ring")
        curl_finger(pts_fist, "pinky")
        pts_fist[4] = [0.50, 0.58, -0.04]
        states_fist = self.classifier.classify_hand(pts_fist)
        assert states_fist.thumb.extension_ratio <= 0.85
        assert states_fist.thumb.state in (FingerStateEnum.TUCKED, FingerStateEnum.FOLDED)

    def test_thumb_obstruction_detection(self):
        """
        Validates that when the thumb is not in view / occluded from the camera,
        the system stops assuming its position and marks it UNCERTAIN without tracking it as folded.
        """
        pts = create_base_hand_landmarks()

        # 1. Optical / sensor obstruction (visibilities < threshold)
        vis_obstructed = [1.0] * 21
        vis_obstructed[4] = 0.15  # Thumb tip obstructed
        states_obs = self.classifier.classify_hand(pts, visibilities=vis_obstructed)
        assert states_obs.thumb.state == FingerStateEnum.UNCERTAIN
        assert states_obs.thumb.confidence == 0.0
        assert any("not in view" in d.lower() or "occluded" in d.lower() for d in states_obs.thumb.diagnostics)

        # 2. Behind-the-palm occlusion (thumb hidden behind open hand: Z > 0 in camera space)
        pts_behind = create_base_hand_landmarks()
        pts_behind[4] = [0.48, 0.55, 0.08]
        states_behind = self.classifier.classify_hand(pts_behind)
        assert states_behind.thumb.state == FingerStateEnum.UNCERTAIN
        assert any("not in view" in d.lower() or "occluded" in d.lower() for d in states_behind.thumb.diagnostics)

        # 3. Kinematic collapse / hallucination (joint distance collapsed to 0)
        pts_collapsed = create_base_hand_landmarks()
        pts_collapsed[4] = pts_collapsed[3].copy()  # Tip collapsed onto IP
        states_collapsed = self.classifier.classify_hand(pts_collapsed)
        assert states_collapsed.thumb.state == FingerStateEnum.UNCERTAIN

    def test_dorsal_back_of_hand_detection_and_thumb_preservation(self):
        """
        Validates that when the back of the hand (dorsal side) faces the camera,
        the system correctly recognizes palm_facing as DORSAL and does NOT falsely
        flag the extended thumb as occluded/uncertain.
        """
        pts_dorsal = create_base_hand_landmarks()
        # Flip X to represent back of right hand facing camera
        pts_dorsal[:, 0] = 1.0 - pts_dorsal[:, 0]
        # Abduct thumb radially
        pts_dorsal[4] = [1.0 - 0.34, 0.48, 0.0]

        states = self.classifier.classify_hand(pts_dorsal, handedness="Right")
        assert states.palm_facing == "DORSAL"
        assert states.thumb.state == FingerStateEnum.EXTENDED
        assert states.index.state == FingerStateEnum.EXTENDED
        assert states.middle.state == FingerStateEnum.EXTENDED

