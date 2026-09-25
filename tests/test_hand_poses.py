"""
Comprehensive unit tests for Level 1 Static Hand Pose Derivation Engine.

Validates the formal 3-tier architectural chain:
    Finger States (Level 0)
          ↓
    Finger Configuration (Topological Composition)
          ↓
    Hand Pose (Level 1: H001–H016)

Tests cover canonical Gesture Bible poses (H001 through H016), explainability diagnostics,
and topological disambiguations:
- POINT vs PEACE
- PINCH vs OK_RING
- POINT vs GUN
- PEACE vs DOUBLE_POINT
- CLOSED_FIST vs THUMBS_UP
- OPEN_PALM vs OPEN_PALM_SPREAD
"""

import math
import numpy as np
import pytest

from src.gestures.hand_pose import (
    DerivedHandPose,
    FingerConfiguration,
    HandPoseClassifier,
    HandPoseId,
    POSE_CANONICAL_NAMES,
)
from src.landmarks.finger_state import (
    FingerName,
    FingerStateDetail,
    FingerStateEnum,
    HandFingerStates,
)
from src.landmarks.kinematic_features import FINGER_INDICES
from tests.test_finger_states import curl_finger


def create_base_hand_landmarks() -> np.ndarray:
    """
    Creates an anatomically plausible normalized 3D hand in coronal plane.
    Wrist at [0.5, 0.70, 0.0], Middle MCP at [0.5, 0.50, 0.0], d_ref = 0.20.
    Hand extends upward along -Y.
    """
    pts = np.zeros((21, 3), dtype=np.float32)

    # 0: Wrist
    pts[0] = [0.5, 0.70, 0.0]

    # Thumb: 1 (CMC), 2 (MCP), 3 (IP), 4 (Tip)
    pts[1] = [0.46, 0.65, 0.0]
    pts[2] = [0.43, 0.60, 0.0]
    pts[3] = [0.40, 0.55, 0.0]
    pts[4] = [0.38, 0.50, 0.0]

    # Index: 5-8
    pts[5] = [0.47, 0.52, 0.0]
    pts[6] = [0.47, 0.44, 0.0]
    pts[7] = [0.47, 0.38, 0.0]
    pts[8] = [0.47, 0.32, 0.0]

    # Middle: 9-12
    pts[9]  = [0.50, 0.50, 0.0]
    pts[10] = [0.50, 0.41, 0.0]
    pts[11] = [0.50, 0.34, 0.0]
    pts[12] = [0.50, 0.27, 0.0]

    # Ring: 13-16
    pts[13] = [0.53, 0.52, 0.0]
    pts[14] = [0.53, 0.44, 0.0]
    pts[15] = [0.53, 0.38, 0.0]
    pts[16] = [0.53, 0.32, 0.0]

    # Pinky: 17-20
    pts[17] = [0.56, 0.55, 0.0]
    pts[18] = [0.56, 0.48, 0.0]
    pts[19] = [0.56, 0.43, 0.0]
    pts[20] = [0.56, 0.38, 0.0]

    return pts


def make_finger_states(
    thumb: FingerStateEnum = FingerStateEnum.RELAXED,
    index: FingerStateEnum = FingerStateEnum.EXTENDED,
    middle: FingerStateEnum = FingerStateEnum.EXTENDED,
    ring: FingerStateEnum = FingerStateEnum.EXTENDED,
    little: FingerStateEnum = FingerStateEnum.EXTENDED,
    thumb_contact: str = None,
    index_contact: str = None,
) -> HandFingerStates:
    """Helper to construct HandFingerStates with specified Level 0 finger states."""
    return HandFingerStates(
        thumb=FingerStateDetail(
            finger=FingerName.THUMB,
            state=thumb,
            confidence=0.92,
            contact_target=thumb_contact,
        ),
        index=FingerStateDetail(
            finger=FingerName.INDEX,
            state=index,
            confidence=0.92,
            contact_target=index_contact,
        ),
        middle=FingerStateDetail(
            finger=FingerName.MIDDLE,
            state=middle,
            confidence=0.92,
        ),
        ring=FingerStateDetail(
            finger=FingerName.RING,
            state=ring,
            confidence=0.92,
        ),
        little=FingerStateDetail(
            finger=FingerName.LITTLE,
            state=little,
            confidence=0.92,
        ),
    )


class TestHandPoseDerivation:
    """Validates the 3-step hierarchy: Finger States -> Finger Configuration -> Hand Pose."""

    @pytest.fixture
    def classifier(self):
        return HandPoseClassifier(min_confidence=0.45)

    @pytest.fixture
    def base_landmarks(self):
        return create_base_hand_landmarks()

    def test_finger_configuration_intermediate_representation(self, classifier, base_landmarks):
        """FingerConfiguration must correctly aggregate finger states and compute topological metrics."""
        finger_states = make_finger_states(
            thumb=FingerStateEnum.FOLDED,
            index=FingerStateEnum.EXTENDED,
            middle=FingerStateEnum.FOLDED,
            ring=FingerStateEnum.FOLDED,
            little=FingerStateEnum.FOLDED,
        )
        cfg = classifier.build_configuration(finger_states, base_landmarks, d_ref=0.20)

        assert cfg.thumb == FingerStateEnum.FOLDED
        assert cfg.index == FingerStateEnum.EXTENDED
        assert cfg.num_extended_fingers == 1
        assert cfg.num_folded_fingers == 3
        assert cfg.thumb_is_folded is True
        assert cfg.thumb_is_extended is False
        assert "T:folded|I:extended" in cfg.summary()

    def test_derive_point_pose(self, classifier, base_landmarks):
        """
        Index extended, all other fingers folded -> H004_INDEX_POINT ('POINT').
        Verify deterministic derivation, not isolated class guessing.
        """
        finger_states = make_finger_states(
            thumb=FingerStateEnum.FOLDED,
            index=FingerStateEnum.EXTENDED,
            middle=FingerStateEnum.FOLDED,
            ring=FingerStateEnum.FOLDED,
            little=FingerStateEnum.FOLDED,
        )
        pose = classifier.classify_pose(finger_states, base_landmarks, d_ref=0.20)

        assert pose.pose_id == HandPoseId.H004_INDEX_POINT
        assert pose.canonical_name == "POINT"
        assert pose.confidence >= 0.80
        assert any("pointing ray" in p.lower() for p in pose.satisfied_predicates)
        assert len(pose.diagnostics) > 0

    def test_derive_pinch_pose(self, classifier, base_landmarks):
        """
        Thumb + Index pinching, digits 3-5 folded/relaxed -> H005_PRECISION_PINCH ('PINCH').
        """
        finger_states = make_finger_states(
            thumb=FingerStateEnum.PINCHING,
            index=FingerStateEnum.PINCHING,
            middle=FingerStateEnum.FOLDED,
            ring=FingerStateEnum.FOLDED,
            little=FingerStateEnum.FOLDED,
            thumb_contact="index_tip",
            index_contact="thumb_tip",
        )
        pose = classifier.classify_pose(finger_states, base_landmarks, d_ref=0.20)

        assert pose.pose_id == HandPoseId.H005_PRECISION_PINCH
        assert pose.canonical_name == "PINCH"
        assert pose.confidence >= 0.85
        assert pose.configuration.thumb_is_pinching is True

    def test_disambiguate_pinch_vs_ok_ring(self, classifier, base_landmarks):
        """
        Pinch with outer digits extended -> H010_OK_RING.
        Pinch with outer digits curled -> H005_PRECISION_PINCH.
        """
        # Outer digits curled -> PINCH
        fs_pinch = make_finger_states(
            thumb=FingerStateEnum.PINCHING,
            index=FingerStateEnum.PINCHING,
            middle=FingerStateEnum.FOLDED,
            ring=FingerStateEnum.FOLDED,
            little=FingerStateEnum.FOLDED,
        )
        pose_pinch = classifier.classify_pose(fs_pinch, base_landmarks, d_ref=0.20)
        assert pose_pinch.pose_id == HandPoseId.H005_PRECISION_PINCH

        # Outer digits extended -> OK_RING
        fs_ok = make_finger_states(
            thumb=FingerStateEnum.PINCHING,
            index=FingerStateEnum.PINCHING,
            middle=FingerStateEnum.EXTENDED,
            ring=FingerStateEnum.EXTENDED,
            little=FingerStateEnum.EXTENDED,
        )
        pose_ok = classifier.classify_pose(fs_ok, base_landmarks, d_ref=0.20)
        assert pose_ok.pose_id == HandPoseId.H010_OK_RING
        assert pose_ok.canonical_name == "OK_RING"

    def test_disambiguate_point_vs_gun(self, classifier, base_landmarks):
        """
        Index extended, digits 3-5 folded:
        - Thumb folded -> POINT (H004)
        - Thumb extended radially (>=55 deg) -> GUN (H013)
        """
        pts = base_landmarks.copy()

        # Folded thumb -> POINT
        fs_point = make_finger_states(
            thumb=FingerStateEnum.FOLDED,
            index=FingerStateEnum.EXTENDED,
            middle=FingerStateEnum.FOLDED,
            ring=FingerStateEnum.FOLDED,
            little=FingerStateEnum.FOLDED,
        )
        pose_point = classifier.classify_pose(fs_point, pts, d_ref=0.20)
        assert pose_point.pose_id == HandPoseId.H004_INDEX_POINT

        # Extended radial thumb (abducted along X axis) -> GUN
        pts[4] = [0.30, 0.60, 0.0]  # Thumb extended far to the side (90 deg to index)
        fs_gun = make_finger_states(
            thumb=FingerStateEnum.EXTENDED,
            index=FingerStateEnum.EXTENDED,
            middle=FingerStateEnum.FOLDED,
            ring=FingerStateEnum.FOLDED,
            little=FingerStateEnum.FOLDED,
        )
        pose_gun = classifier.classify_pose(fs_gun, pts, d_ref=0.20)
        assert pose_gun.pose_id == HandPoseId.H013_GUN
        assert pose_gun.canonical_name == "GUN"

    def test_disambiguate_point_vs_peace(self, classifier, base_landmarks):
        """
        Index extended:
        - Middle folded -> POINT (H004)
        - Middle extended in V-formation -> PEACE (H009)
        """
        pts = base_landmarks.copy()
        # Set middle finger tip divergent to create >10 deg V-divergence
        pts[12] = [0.55, 0.27, 0.0]  # Angle away from index

        fs_peace = make_finger_states(
            thumb=FingerStateEnum.FOLDED,
            index=FingerStateEnum.EXTENDED,
            middle=FingerStateEnum.EXTENDED,
            ring=FingerStateEnum.FOLDED,
            little=FingerStateEnum.FOLDED,
        )
        pose_peace = classifier.classify_pose(fs_peace, pts, d_ref=0.20)
        assert pose_peace.pose_id == HandPoseId.H009_PEACE
        assert pose_peace.canonical_name == "PEACE"

    def test_disambiguate_peace_vs_double_point(self, classifier, base_landmarks):
        """
        Index + Middle extended, Ring + Little folded:
        - Wide divergence -> PEACE (H009)
        - Tight parallel divergence (<12 deg) + Thumb extended -> DOUBLE_POINT (H016)
        """
        pts = base_landmarks.copy()
        # Parallel index and middle
        pts[8]  = [0.48, 0.28, 0.0]
        pts[12] = [0.50, 0.28, 0.0]

        fs_dp = make_finger_states(
            thumb=FingerStateEnum.EXTENDED,
            index=FingerStateEnum.EXTENDED,
            middle=FingerStateEnum.EXTENDED,
            ring=FingerStateEnum.FOLDED,
            little=FingerStateEnum.FOLDED,
        )
        pose_dp = classifier.classify_pose(fs_dp, pts, d_ref=0.20)
        assert pose_dp.pose_id == HandPoseId.H016_DOUBLE_POINT
        assert pose_dp.canonical_name == "DOUBLE_POINT"

    def test_derive_thumbs_up_and_down(self, classifier, base_landmarks):
        """
        Fingers 2-5 balled into fist:
        - Thumb pointing upward (-Y) -> THUMBS_UP (H007)
        - Thumb pointing downward (+Y) -> THUMBS_DOWN (H008)
        """
        pts = base_landmarks.copy()

        # Thumb upward (-Y)
        pts[2] = [0.45, 0.60, 0.0]
        pts[4] = [0.45, 0.40, 0.0]  # Tip is higher (-Y) than MCP
        fs_up = make_finger_states(
            thumb=FingerStateEnum.EXTENDED,
            index=FingerStateEnum.FOLDED,
            middle=FingerStateEnum.FOLDED,
            ring=FingerStateEnum.FOLDED,
            little=FingerStateEnum.FOLDED,
        )
        pose_up = classifier.classify_pose(fs_up, pts, d_ref=0.20)
        assert pose_up.pose_id == HandPoseId.H007_THUMBS_UP
        assert pose_up.canonical_name == "THUMBS_UP"

        # Thumb downward (+Y)
        pts[4] = [0.45, 0.80, 0.0]  # Tip is lower (+Y) than MCP
        pose_down = classifier.classify_pose(fs_up, pts, d_ref=0.20)
        assert pose_down.pose_id == HandPoseId.H008_THUMBS_DOWN
        assert pose_down.canonical_name == "THUMBS_DOWN"

    def test_derive_thumbs_up_hitchhiker_and_tilted(self, classifier, base_landmarks):
        """
        Validates Thumbs Up recognition across natural human variations:
        - Hitchhiker thumb with curved IP joint (angle ~35 deg)
        - Hand tilted up to 50 degrees from vertical
        - Digits 2-5 clenched into fist with slight curve
        """
        pts = base_landmarks.copy()
        curl_finger(pts, "index")
        curl_finger(pts, "middle")
        curl_finger(pts, "ring")
        curl_finger(pts, "pinky")

        pts[1] = [0.46, 0.65, 0.0]
        pts[2] = [0.45, 0.58, 0.0]
        pts[3] = [0.45, 0.48, 0.0]
        pts[4] = [0.45, 0.38, 0.0]

        # 1. Hitchhiker thumb (IP bent 35 deg outward)
        pts_hitch = pts.copy()
        rad = math.radians(35)
        pts_hitch[4] = [0.45 + 0.10 * math.sin(rad), 0.48 - 0.10 * math.cos(rad), 0.0]
        fs_hitch = make_finger_states(
            thumb=FingerStateEnum.CURVED,
            index=FingerStateEnum.FOLDED,
            middle=FingerStateEnum.FOLDED,
            ring=FingerStateEnum.FOLDED,
            little=FingerStateEnum.FOLDED,
        )
        fs_hitch.thumb.extension_ratio = 1.35
        pose_hitch = classifier.classify_pose(fs_hitch, pts_hitch, d_ref=0.20)
        assert pose_hitch.pose_id == HandPoseId.H007_THUMBS_UP
        assert pose_hitch.canonical_name == "THUMBS_UP"

        # 2. Hand tilted 45 degrees
        pts_tilt = pts.copy()
        theta = math.radians(45)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        wrist = pts_tilt[0, :2].copy()
        for i in range(len(pts_tilt)):
            dx = pts_tilt[i, 0] - wrist[0]
            dy = pts_tilt[i, 1] - wrist[1]
            pts_tilt[i, 0] = wrist[0] + dx * cos_t - dy * sin_t
            pts_tilt[i, 1] = wrist[1] + dx * sin_t + dy * cos_t

        fs_tilt = make_finger_states(
            thumb=FingerStateEnum.EXTENDED,
            index=FingerStateEnum.FOLDED,
            middle=FingerStateEnum.FOLDED,
            ring=FingerStateEnum.FOLDED,
            little=FingerStateEnum.FOLDED,
        )
        pose_tilt = classifier.classify_pose(fs_tilt, pts_tilt, d_ref=0.20)
        assert pose_tilt.pose_id == HandPoseId.H007_THUMBS_UP
        assert pose_tilt.canonical_name == "THUMBS_UP"

    def test_derive_closed_fist_grab(self, classifier, base_landmarks):
        """
        All digits balled into fist -> H003_CLOSED_FIST ('GRAB').
        """
        fs_fist = make_finger_states(
            thumb=FingerStateEnum.FOLDED,
            index=FingerStateEnum.FOLDED,
            middle=FingerStateEnum.FOLDED,
            ring=FingerStateEnum.FOLDED,
            little=FingerStateEnum.FOLDED,
        )
        pose_fist = classifier.classify_pose(fs_fist, base_landmarks, d_ref=0.20)
        assert pose_fist.pose_id == HandPoseId.H003_CLOSED_FIST
        assert pose_fist.canonical_name == "GRAB"

    def test_derive_open_palm_neutral_and_spread(self, classifier, base_landmarks):
        """
        All fingers extended:
        - Wide spacing / abduction -> H002_OPEN_PALM_SPREAD ('SPREAD_FINGERS')
        - Regular spacing -> H001_OPEN_PALM ('OPEN_PALM')
        """
        pts = base_landmarks.copy()

        # Spread fingers laterally
        pts[4]  = [0.25, 0.50, 0.0]
        pts[8]  = [0.40, 0.30, 0.0]
        pts[12] = [0.50, 0.25, 0.0]
        pts[16] = [0.60, 0.30, 0.0]
        pts[20] = [0.72, 0.35, 0.0]

        fs_open = make_finger_states(
            thumb=FingerStateEnum.EXTENDED,
            index=FingerStateEnum.EXTENDED,
            middle=FingerStateEnum.EXTENDED,
            ring=FingerStateEnum.EXTENDED,
            little=FingerStateEnum.EXTENDED,
        )
        pose_spread = classifier.classify_pose(fs_open, pts, d_ref=0.20)
        assert pose_spread.pose_id == HandPoseId.H002_OPEN_PALM_SPREAD
        assert pose_spread.canonical_name == "SPREAD_FINGERS"

    def test_derive_shaka_call_me(self, classifier, base_landmarks):
        """
        Thumb + Little extended, central 3 digits folded -> H012_SHAKA ('CALL_ME_SHAKA').
        """
        fs_shaka = make_finger_states(
            thumb=FingerStateEnum.EXTENDED,
            index=FingerStateEnum.FOLDED,
            middle=FingerStateEnum.FOLDED,
            ring=FingerStateEnum.FOLDED,
            little=FingerStateEnum.EXTENDED,
        )
        pose_shaka = classifier.classify_pose(fs_shaka, base_landmarks, d_ref=0.20)
        assert pose_shaka.pose_id == HandPoseId.H012_SHAKA
        assert pose_shaka.canonical_name == "CALL_ME_SHAKA"

    def test_derive_three_finger(self, classifier, base_landmarks):
        """
        Index, Middle, Ring extended, Little folded, Thumb neutral -> H011_THREE_FINGER.
        """
        fs_three = make_finger_states(
            thumb=FingerStateEnum.FOLDED,
            index=FingerStateEnum.EXTENDED,
            middle=FingerStateEnum.EXTENDED,
            ring=FingerStateEnum.EXTENDED,
            little=FingerStateEnum.FOLDED,
        )
        pose_three = classifier.classify_pose(fs_three, base_landmarks, d_ref=0.20)
        assert pose_three.pose_id == HandPoseId.H011_THREE_FINGER
        assert pose_three.canonical_name == "THREE_FINGER"

    def test_derive_cupped_hand(self, classifier, base_landmarks):
        """
        All fingers semi-flexed / curved -> H014_CUPPED ('CUPPED_HAND').
        """
        fs_cupped = make_finger_states(
            thumb=FingerStateEnum.CURVED,
            index=FingerStateEnum.CURVED,
            middle=FingerStateEnum.CURVED,
            ring=FingerStateEnum.CURVED,
            little=FingerStateEnum.CURVED,
        )
        pose_cupped = classifier.classify_pose(fs_cupped, base_landmarks, d_ref=0.20)
        assert pose_cupped.pose_id == HandPoseId.H014_CUPPED
        assert pose_cupped.canonical_name == "CUPPED_HAND"

    def test_derive_knife_edge(self, classifier, base_landmarks):
        """
        All digits extended & adducted tightly + edge-on roll angle -> H015_KNIFE_EDGE.
        """
        pts = base_landmarks.copy()
        # Tight adduction between fingertips
        pts[8]  = [0.49, 0.30, 0.0]
        pts[12] = [0.50, 0.28, 0.0]
        pts[16] = [0.51, 0.30, 0.0]
        pts[20] = [0.52, 0.32, 0.0]

        fs_knife = make_finger_states(
            thumb=FingerStateEnum.EXTENDED,
            index=FingerStateEnum.EXTENDED,
            middle=FingerStateEnum.EXTENDED,
            ring=FingerStateEnum.EXTENDED,
            little=FingerStateEnum.EXTENDED,
        )
        # Roll angle around 90 degrees (pi/2) -> edge-on
        pose_knife = classifier.classify_pose(
            fs_knife, pts, d_ref=0.20, orientation_angles=(0.0, 0.0, float(math.pi / 2.0))
        )
        assert pose_knife.pose_id == HandPoseId.H015_KNIFE_EDGE
        assert pose_knife.canonical_name == "KNIFE_EDGE"

    def test_derive_lateral_key_pinch(self, classifier, base_landmarks):
        """
        Thumb pad touching index base/side while digits 3-5 folded -> H006_LATERAL_PINCH.
        """
        fs_lateral = make_finger_states(
            thumb=FingerStateEnum.TOUCHING,
            index=FingerStateEnum.FOLDED,
            middle=FingerStateEnum.FOLDED,
            ring=FingerStateEnum.FOLDED,
            little=FingerStateEnum.FOLDED,
            thumb_contact="index_base",
        )
        pose_lateral = classifier.classify_pose(fs_lateral, base_landmarks, d_ref=0.20)
        assert pose_lateral.pose_id == HandPoseId.H006_LATERAL_PINCH
        assert pose_lateral.canonical_name == "KEY_PINCH"

    def test_derive_point_when_inner_hand_not_visible(self, classifier, base_landmarks):
        """
        Validates that when the back of the hand faces the camera (inner hand not visible),
        pointing reads accurately as POINT, assuming non-visible digits are folded.
        """
        # Create dorsal view landmarks: flip x so pinky is on left, index on right
        dorsal_pts = base_landmarks.copy()
        dorsal_pts[:, 0] = 1.0 - dorsal_pts[:, 0]
        # Index extended, others folded/curled
        fs = make_finger_states(
            thumb=FingerStateEnum.FOLDED,
            index=FingerStateEnum.EXTENDED,
            middle=FingerStateEnum.FOLDED,
            ring=FingerStateEnum.FOLDED,
            little=FingerStateEnum.FOLDED,
        )
        pose = classifier.classify_pose(fs, dorsal_pts, d_ref=0.20)
        assert pose.pose_id == HandPoseId.H004_INDEX_POINT
        assert pose.canonical_name == "POINT"

    def test_non_visible_digits_assumed_folded_derives_point(self, classifier, base_landmarks):
        """
        Validates the user requirement: 'any finger not visible to the camera is to be assumed as folded'.
        Even if curled digits were marked UNCERTAIN due to sensor occlusion, POINT is correctly derived.
        """
        fs_occluded = make_finger_states(
            thumb=FingerStateEnum.UNCERTAIN,
            index=FingerStateEnum.EXTENDED,
            middle=FingerStateEnum.UNCERTAIN,
            ring=FingerStateEnum.UNCERTAIN,
            little=FingerStateEnum.UNCERTAIN,
        )
        pose = classifier.classify_pose(fs_occluded, base_landmarks, d_ref=0.20)
        assert pose.pose_id == HandPoseId.H004_INDEX_POINT
        assert pose.canonical_name == "POINT"

    def test_derive_fist_when_inner_hand_not_visible(self, classifier, base_landmarks):
        """
        Validates that when the back of the hand faces the camera and all fingers
        are curled into a fist (or occluded/uncertain), it correctly reads as GRAB / CLOSED_FIST.
        """
        dorsal_pts = base_landmarks.copy()
        dorsal_pts[:, 0] = 1.0 - dorsal_pts[:, 0]
        fs_fist = make_finger_states(
            thumb=FingerStateEnum.FOLDED,
            index=FingerStateEnum.FOLDED,
            middle=FingerStateEnum.FOLDED,
            ring=FingerStateEnum.FOLDED,
            little=FingerStateEnum.FOLDED,
        )
        pose = classifier.classify_pose(fs_fist, dorsal_pts, d_ref=0.20)
        assert pose.pose_id == HandPoseId.H003_CLOSED_FIST
        assert pose.canonical_name == "GRAB"

    def test_derive_thumbs_up_from_dorsal_back_of_hand(self, classifier, base_landmarks):
        """
        Validates that THUMBS_UP is recognized with high confidence when the user shows
        the back of their hand with thumb pointing up and all fingers curled into a fist.
        """
        dorsal_pts = base_landmarks.copy()
        # Flip X to represent back of right hand
        dorsal_pts[:, 0] = 1.0 - dorsal_pts[:, 0]
        # Point thumb upward (-Y): Tip (4) is above knuckles (5) and wrist (0)
        dorsal_pts[4] = [dorsal_pts[2][0] + 0.02, dorsal_pts[0][1] - 0.22, 0.0]

        fs_thumbs_up = make_finger_states(
            thumb=FingerStateEnum.EXTENDED,
            index=FingerStateEnum.FOLDED,
            middle=FingerStateEnum.FOLDED,
            ring=FingerStateEnum.FOLDED,
            little=FingerStateEnum.FOLDED,
        )
        fs_thumbs_up.palm_facing = "DORSAL"

        pose = classifier.classify_pose(fs_thumbs_up, dorsal_pts, d_ref=0.20)
        assert pose.pose_id == HandPoseId.H007_THUMBS_UP
        assert pose.canonical_name == "THUMBS_UP"
        assert pose.confidence >= 0.90
        assert pose.configuration.palm_facing == "DORSAL"

    def test_derive_index_point_from_dorsal_back_of_hand(self, classifier, base_landmarks):
        """
        Validates that INDEX_POINT is recognized cleanly when viewed from the back of the hand.
        """
        dorsal_pts = base_landmarks.copy()
        dorsal_pts[:, 0] = 1.0 - dorsal_pts[:, 0]

        fs_point = make_finger_states(
            thumb=FingerStateEnum.FOLDED,
            index=FingerStateEnum.EXTENDED,
            middle=FingerStateEnum.FOLDED,
            ring=FingerStateEnum.FOLDED,
            little=FingerStateEnum.FOLDED,
        )
        fs_point.palm_facing = "DORSAL"

        pose = classifier.classify_pose(fs_point, dorsal_pts, d_ref=0.20)
        assert pose.pose_id == HandPoseId.H004_INDEX_POINT
        assert pose.canonical_name == "POINT"
        assert pose.confidence >= 0.90
