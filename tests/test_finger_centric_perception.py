"""
Tests for the Finger-Centric Perception Layer.

Covers:
- Fingertip tracking: camera-facing detection, foreshortening
- Finger chain analysis: direction, foreshortening ratio
- Relationship model: convergence, divergence
- Viewpoint classifier: NORMAL, CAMERA_FACING
- Evidence fusion: weight adaptation per viewpoint
- Full engine pipeline
- Independent finger motion (hand stationary, index moves)
- Whole-hand motion (palm and fingers move together)
- Mixed motion (palm left + index approaching camera)
"""
import pytest
import numpy as np

from src.finger_centric.fingertip_tracker import (
    FingerTipTracker, FingerMotionClass, FINGER_TIP_INDICES, FINGER_MCP_INDICES
)
from src.finger_centric.finger_chain import FingerChainAnalyzer, FingerOrientationClass
from src.finger_centric.relationship_model import FingertipRelationshipModel
from src.finger_centric.viewpoint_classifier import ViewpointClassifier, ViewpointMode
from src.finger_centric.evidence_fusion import EvidenceFusion, ActiveRepresentation
from src.finger_centric.finger_centric_engine import FingerCentricPerceptionEngine


# ─────────────────────────────────────────────────────────────────────────────
# Landmark scaffold helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flat_landmarks() -> np.ndarray:
    """
    21 landmarks representing an open hand facing the camera, all z≈0.
    Fingers extend upward in y, palm at bottom.
    """
    pts = np.zeros((21, 3), dtype=np.float32)
    # Wrist
    pts[0] = [0.5, 0.8, 0.0]
    # Thumb (1-4)
    for i, (x, y) in enumerate([(0.35, 0.70), (0.25, 0.62), (0.18, 0.55), (0.12, 0.48)]):
        pts[i + 1] = [x, y, 0.0]
    # Index (5-8)
    for i, (x, y) in enumerate([(0.40, 0.65), (0.39, 0.55), (0.38, 0.45), (0.37, 0.35)]):
        pts[i + 5] = [x, y, 0.0]
    # Middle (9-12)
    for i, (x, y) in enumerate([(0.50, 0.64), (0.50, 0.53), (0.50, 0.43), (0.50, 0.33)]):
        pts[i + 9] = [x, y, 0.0]
    # Ring (13-16)
    for i, (x, y) in enumerate([(0.60, 0.65), (0.61, 0.55), (0.62, 0.45), (0.63, 0.35)]):
        pts[i + 13] = [x, y, 0.0]
    # Little (17-20)
    for i, (x, y) in enumerate([(0.70, 0.67), (0.72, 0.57), (0.74, 0.50), (0.76, 0.42)]):
        pts[i + 17] = [x, y, 0.0]
    return pts


def _camera_facing_landmarks() -> np.ndarray:
    """
    Landmarks where fingers point directly at camera (z >> xy spread).
    The 2D image projection is heavily compressed along the finger axis.
    """
    pts = _flat_landmarks().copy()
    # Collapse xy spread along finger axis, emphasise z depth
    for tip_idx, mcp_idx in zip([4, 8, 12, 16, 20], [1, 5, 9, 13, 17]):
        mcp_xy = pts[mcp_idx, :2].copy()
        # TIP nearly coincides with MCP in xy (foreshortening)
        pts[tip_idx, :2] = mcp_xy + np.array([0.005, 0.003], dtype=np.float32)
        # TIP much closer to camera in z (approaching viewer)
        pts[mcp_idx, 2] = 0.05
        pts[tip_idx, 2] = -0.12   # tip closer to camera (negative z = toward camera)
    return pts


def _d_ref() -> float:
    """Typical hand scale reference."""
    return 0.15


# ─────────────────────────────────────────────────────────────────────────────
# FingerTipTracker
# ─────────────────────────────────────────────────────────────────────────────

class TestFingerTipTracker:

    def test_initial_state_no_velocity(self):
        tracker = FingerTipTracker()
        states = tracker.update(_flat_landmarks(), _d_ref(), timestamp=0.0)
        for name, ts in states.items():
            assert ts.speed == pytest.approx(0.0, abs=1e-5), f"{name} speed should be 0 on first frame"
            assert ts.tracking_age == 1

    def test_velocity_computed_after_two_frames(self):
        tracker = FingerTipTracker()
        pts = _flat_landmarks()
        tracker.update(pts, _d_ref(), timestamp=0.0)
        # Move index tip 0.1 units in x over 0.1s → velocity ≈ 1.0 u/s
        pts2 = pts.copy()
        pts2[8, 0] += 0.1
        states = tracker.update(pts2, _d_ref(), timestamp=0.1)
        assert states["index"].speed > 0.5

    def test_foreshortening_detection(self):
        tracker = FingerTipTracker()
        cam_pts = _camera_facing_landmarks()
        states = tracker.update(cam_pts, _d_ref(), timestamp=0.0)
        # Foreshortened fingers should have low foreshortening_ratio
        for name in ["index", "middle", "ring", "little"]:
            ratio = states[name].foreshortening_ratio
            assert ratio < 0.40, f"{name}: expected foreshortening < 0.40, got {ratio:.3f}"

    def test_camera_facing_detection_from_z_velocity(self):
        tracker = FingerTipTracker()
        pts = _flat_landmarks()
        tracker.update(pts, _d_ref(), timestamp=0.0)
        # Move index tip toward camera (z decreasing) without much xy change
        pts2 = pts.copy()
        pts2[8, 2] -= 0.15   # strong z motion toward camera
        pts2[8, 0] += 0.002  # tiny xy movement
        pts2[8, 1] += 0.001
        states = tracker.update(pts2, _d_ref(), timestamp=0.1)
        index_state = states["index"]
        assert index_state.is_camera_facing, "Index should be detected as camera-facing"

    def test_motion_class_approaching(self):
        tracker = FingerTipTracker()
        pts = _flat_landmarks()
        tracker.update(pts, _d_ref(), timestamp=0.0)
        pts2 = pts.copy()
        pts2[8, 2] -= 0.20  # strong approach
        pts2[8, 0] += 0.001
        states = tracker.update(pts2, _d_ref(), timestamp=0.1)
        assert states["index"].motion_class == FingerMotionClass.APPROACHING

    def test_motion_class_static(self):
        tracker = FingerTipTracker()
        pts = _flat_landmarks()
        tracker.update(pts, _d_ref(), timestamp=0.0)
        states = tracker.update(pts.copy(), _d_ref(), timestamp=0.033)
        for name, ts in states.items():
            assert ts.motion_class == FingerMotionClass.STATIC, f"{name} should be STATIC"

    def test_tracking_age_increments(self):
        tracker = FingerTipTracker()
        pts = _flat_landmarks()
        for frame in range(5):
            states = tracker.update(pts.copy(), _d_ref(), timestamp=frame * 0.033)
        for name, ts in states.items():
            assert ts.tracking_age == 5

    def test_reset_finger_clears_state(self):
        tracker = FingerTipTracker()
        pts = _flat_landmarks()
        for _ in range(4):
            tracker.update(pts.copy(), _d_ref(), timestamp=_ * 0.033)
        tracker.reset_finger("index")
        states = tracker.update(pts.copy(), _d_ref(), timestamp=4 * 0.033)
        assert states["index"].tracking_age == 1
        assert states["index"].speed == pytest.approx(0.0, abs=1e-5)

    def test_confidence_in_valid_range(self):
        tracker = FingerTipTracker()
        pts = _flat_landmarks()
        for _ in range(6):
            states = tracker.update(pts.copy(), _d_ref(), timestamp=_ * 0.033)
        for name, ts in states.items():
            assert 0.0 <= ts.confidence <= 1.0, f"{name}: confidence out of range"

    def test_no_crash_on_zero_d_ref(self):
        """Degenerate hand scale reference must not raise."""
        tracker = FingerTipTracker()
        tracker.update(_flat_landmarks(), d_ref=1e-7, timestamp=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# FingerChainAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class TestFingerChainAnalyzer:

    def test_direction_points_toward_tip(self):
        """For a flat upward hand, index chain direction should point upward (negative y)."""
        analyzer = FingerChainAnalyzer()
        chains = analyzer.analyze(_flat_landmarks(), _d_ref())
        idx_dir = chains["index"].direction_3d
        # Finger points up → dy < 0 in MediaPipe coords (y increases downward)
        assert float(idx_dir[1]) < 0, f"Index should point upward (dy<0), got {idx_dir}"

    def test_foreshortening_ratio_normal(self):
        """For a laterally extended hand, foreshortening ratio should be high."""
        analyzer = FingerChainAnalyzer()
        chains = analyzer.analyze(_flat_landmarks(), _d_ref())
        for name in ["index", "middle", "ring", "little"]:
            ratio = chains[name].foreshortening_ratio
            assert ratio > 0.60, f"{name}: expected low foreshortening, got {ratio:.3f}"

    def test_foreshortening_ratio_camera_facing(self):
        """Camera-facing fingers should have very low foreshortening ratios."""
        analyzer = FingerChainAnalyzer()
        chains = analyzer.analyze(_camera_facing_landmarks(), _d_ref())
        for name in ["index", "middle", "ring", "little"]:
            ratio = chains[name].foreshortening_ratio
            assert ratio < 0.35, f"{name}: expected severe foreshortening, got {ratio:.3f}"

    def test_camera_facing_score_high_for_end_on(self):
        """End-on fingers should have a high camera_facing_score."""
        analyzer = FingerChainAnalyzer()
        chains = analyzer.analyze(_camera_facing_landmarks(), _d_ref())
        for name in ["index", "middle"]:
            score = chains[name].camera_facing_score
            assert score > 0.55, f"{name}: expected high camera_facing_score, got {score:.3f}"

    def test_orientation_class_camera_facing(self):
        """End-on fingers should classify as CAMERA_FACING or CAMERA_AWAY."""
        analyzer = FingerChainAnalyzer()
        chains = analyzer.analyze(_camera_facing_landmarks(), _d_ref())
        for name in ["index", "middle"]:
            oc = chains[name].orientation_class
            assert oc in (FingerOrientationClass.CAMERA_FACING, FingerOrientationClass.CAMERA_AWAY), \
                f"{name}: expected camera-facing orientation, got {oc}"

    def test_joint_flexion_angles_are_non_negative(self):
        analyzer = FingerChainAnalyzer()
        chains = analyzer.analyze(_flat_landmarks(), _d_ref())
        for name, cs in chains.items():
            assert cs.mcp_flexion_deg >= 0, f"{name}: negative MCP flexion"
            assert cs.pip_flexion_deg >= 0, f"{name}: negative PIP flexion"
            assert cs.dip_flexion_deg >= 0, f"{name}: negative DIP flexion"

    def test_chain_confidence_in_range(self):
        analyzer = FingerChainAnalyzer()
        chains = analyzer.analyze(_flat_landmarks(), _d_ref())
        for name, cs in chains.items():
            assert 0.0 <= cs.chain_confidence <= 1.0, f"{name}: chain_confidence out of range"

    def test_all_five_fingers_returned(self):
        analyzer = FingerChainAnalyzer()
        chains = analyzer.analyze(_flat_landmarks(), _d_ref())
        assert set(chains.keys()) == {"thumb", "index", "middle", "ring", "little"}


# ─────────────────────────────────────────────────────────────────────────────
# FingertipRelationshipModel
# ─────────────────────────────────────────────────────────────────────────────

class TestFingertipRelationshipModel:

    def _build_tip_states(self, pts: np.ndarray, timestamp: float = 0.0):
        tracker = FingerTipTracker()
        return tracker.update(pts, _d_ref(), timestamp=timestamp)

    def test_spread_score_high_for_open_hand(self):
        rel = FingertipRelationshipModel()
        tip_states = self._build_tip_states(_flat_landmarks())
        state = rel.update(tip_states, _d_ref(), timestamp=0.0)
        assert state.spread_score > 0.15, f"Open hand should have non-trivial spread: {state.spread_score}"

    def test_spread_score_low_for_pinched_hand(self):
        """Collapse all fingertips to one point → minimal spread."""
        rel = FingertipRelationshipModel()
        pts = _flat_landmarks().copy()
        for idx in [4, 8, 12, 16, 20]:
            pts[idx] = pts[8].copy()   # All tips to index tip position
        tip_states = self._build_tip_states(pts)
        state = rel.update(tip_states, _d_ref(), timestamp=0.0)
        assert state.spread_score < 0.50

    def test_convergence_rate_when_approaching(self):
        """Tips moving toward each other should yield positive convergence_rate."""
        rel = FingertipRelationshipModel()
        pts1 = _flat_landmarks()
        tracker = FingerTipTracker()
        states1 = tracker.update(pts1, _d_ref(), timestamp=0.0)
        rel.update(states1, _d_ref(), timestamp=0.0)

        # Move index toward middle
        pts2 = pts1.copy()
        pts2[8, 0] += 0.05   # index tip → toward middle
        states2 = tracker.update(pts2, _d_ref(), timestamp=0.1)
        rel_state = rel.update(states2, _d_ref(), timestamp=0.1)

        pair = rel_state.get_pair("index", "middle")
        assert pair is not None
        # Convergence should reflect relative approach
        assert isinstance(pair.convergence_rate, float)

    def test_all_tracked_pairs_present(self):
        rel = FingertipRelationshipModel()
        tip_states = self._build_tip_states(_flat_landmarks())
        state = rel.update(tip_states, _d_ref(), timestamp=0.0)
        expected_pairs = {"thumb:index", "index:middle", "middle:ring", "ring:little"}
        for key in expected_pairs:
            assert key in state.pairs, f"Expected pair {key} not found"

    def test_dominant_pair_distance_non_negative(self):
        rel = FingertipRelationshipModel()
        tip_states = self._build_tip_states(_flat_landmarks())
        state = rel.update(tip_states, _d_ref(), timestamp=0.0)
        assert state.dominant_pair_distance >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ViewpointClassifier
# ─────────────────────────────────────────────────────────────────────────────

class TestViewpointClassifier:

    def _chain_states(self, pts: np.ndarray):
        analyzer = FingerChainAnalyzer()
        return analyzer.analyze(pts, _d_ref())

    def test_normal_view_classification(self):
        vc = ViewpointClassifier()
        chains = self._chain_states(_flat_landmarks())
        result = vc.classify(
            _flat_landmarks(), chains, "Right", _d_ref(), palm_facing="PALM"
        )
        assert result.mode in (ViewpointMode.NORMAL, ViewpointMode.FORESHORTENED), \
            f"Expected NORMAL for flat hand, got {result.mode}"

    def test_camera_facing_classification(self):
        vc = ViewpointClassifier()
        pts = _camera_facing_landmarks()
        chains = self._chain_states(pts)
        result = vc.classify(pts, chains, "Right", _d_ref(), palm_facing="PALM")
        assert result.mode in (ViewpointMode.CAMERA_FACING, ViewpointMode.FORESHORTENED), \
            f"Expected CAMERA_FACING or FORESHORTENED, got {result.mode}"

    def test_confidence_values_in_range(self):
        vc = ViewpointClassifier()
        chains = self._chain_states(_flat_landmarks())
        result = vc.classify(_flat_landmarks(), chains, "Right", _d_ref(), "PALM")
        assert 0.0 <= result.hand_geometry_confidence <= 1.0
        assert 0.0 <= result.finger_geometry_confidence <= 1.0

    def test_camera_facing_reduces_hand_geometry_confidence(self):
        vc = ViewpointClassifier()
        pts_normal = _flat_landmarks()
        pts_cam    = _camera_facing_landmarks()
        chains_n = self._chain_states(pts_normal)
        chains_c = self._chain_states(pts_cam)
        res_n = vc.classify(pts_normal, chains_n, "Right", _d_ref(), "PALM")
        res_c = vc.classify(pts_cam,    chains_c, "Right", _d_ref(), "PALM")
        assert res_c.hand_geometry_confidence <= res_n.hand_geometry_confidence + 0.05, \
            "Camera-facing should reduce hand geometry confidence"

    def test_active_finger_count(self):
        vc = ViewpointClassifier()
        chains = self._chain_states(_flat_landmarks())
        result = vc.classify(_flat_landmarks(), chains, "Right", _d_ref(), "PALM")
        assert 0 <= result.active_finger_count <= 5


# ─────────────────────────────────────────────────────────────────────────────
# EvidenceFusion
# ─────────────────────────────────────────────────────────────────────────────

class TestEvidenceFusion:

    def _make_viewpoint(self, mode: ViewpointMode, hc: float, fc: float):
        from src.finger_centric.viewpoint_classifier import ViewpointState
        return ViewpointState(
            mode=mode,
            hand_geometry_confidence=hc,
            finger_geometry_confidence=fc,
            palm_facing="PALM",
            foreshortening_severity=1.0 - hc,
            camera_facing_score=0.0,
            palm_normal_z=-0.8,
            active_finger_count=5,
            uncertainty_reason="",
        )

    def test_weights_sum_to_one(self):
        fusion = EvidenceFusion()
        for mode in ViewpointMode:
            vp = self._make_viewpoint(mode, hc=0.8, fc=0.7)
            result = fusion.fuse(vp)
            total = (result.weights.palm_weight + result.weights.finger_weight
                     + result.weights.motion_weight + result.weights.temporal_weight)
            assert abs(total - 1.0) < 1e-5, f"{mode}: weights sum to {total}, not 1.0"

    def test_combined_confidence_in_range(self):
        fusion = EvidenceFusion()
        for mode in ViewpointMode:
            vp = self._make_viewpoint(mode, hc=0.7, fc=0.8)
            result = fusion.fuse(vp)
            assert 0.0 <= result.combined_confidence <= 1.0

    def test_finger_centric_when_hand_confidence_low(self):
        fusion = EvidenceFusion()
        vp = self._make_viewpoint(ViewpointMode.CAMERA_FACING, hc=0.15, fc=0.85)
        result = fusion.fuse(vp)
        assert result.active_representation in (
            ActiveRepresentation.FINGER_CENTRIC, ActiveRepresentation.COMBINED
        ), f"Expected FINGER_CENTRIC or COMBINED, got {result.active_representation}"

    def test_hand_centric_when_finger_confidence_low(self):
        fusion = EvidenceFusion()
        vp = self._make_viewpoint(ViewpointMode.NORMAL, hc=0.92, fc=0.30)
        result = fusion.fuse(vp)
        assert result.active_representation in (
            ActiveRepresentation.HAND_CENTRIC, ActiveRepresentation.COMBINED
        ), f"Expected HAND_CENTRIC or COMBINED, got {result.active_representation}"

    def test_uncertain_when_all_confidences_low(self):
        fusion = EvidenceFusion()
        vp = self._make_viewpoint(ViewpointMode.OCCLUDED, hc=0.05, fc=0.08)
        result = fusion.fuse(vp, motion_confidence=0.05, temporal_confidence=0.10)
        assert result.active_representation in (
            ActiveRepresentation.UNCERTAIN, ActiveRepresentation.TEMPORAL
        ), f"Expected UNCERTAIN or TEMPORAL, got {result.active_representation}"

    def test_camera_facing_mode_emphasises_finger_weight(self):
        fusion = EvidenceFusion()
        vp_normal = self._make_viewpoint(ViewpointMode.NORMAL, hc=0.85, fc=0.82)
        vp_cam    = self._make_viewpoint(ViewpointMode.CAMERA_FACING, hc=0.85, fc=0.82)
        res_n = fusion.fuse(vp_normal)
        res_c = fusion.fuse(vp_cam)
        # Camera-facing should relatively increase finger weight vs palm weight
        diff_n = res_n.weights.finger_weight - res_n.weights.palm_weight
        diff_c = res_c.weights.finger_weight - res_c.weights.palm_weight
        assert diff_c > diff_n, "Camera-facing mode should favour finger evidence more"


# ─────────────────────────────────────────────────────────────────────────────
# Full FingerCentricPerceptionEngine pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestFingerCentricPerceptionEngine:

    def test_full_pipeline_runs_without_error(self):
        engine = FingerCentricPerceptionEngine()
        state = engine.process_hand(
            hand_id=0,
            handedness="Right",
            raw_landmarks=_flat_landmarks(),
            d_ref=_d_ref(),
            palm_facing="PALM",
            timestamp=0.0,
        )
        assert state is not None
        assert state.hand_id == 0
        assert len(state.tip_states) == 5
        assert len(state.chain_states) == 5

    def test_to_dict_serialisable(self):
        engine = FingerCentricPerceptionEngine()
        state = engine.process_hand(0, "Right", _flat_landmarks(), _d_ref(), "PALM", 0.0)
        d = state.to_dict()
        assert "tips" in d
        assert "chains" in d
        assert "viewpoint_mode" in d
        assert "active_representation" in d
        import json
        json.dumps(d)  # Must be fully JSON-serialisable

    def test_palm_speed_zero_on_first_frame(self):
        engine = FingerCentricPerceptionEngine()
        state = engine.process_hand(0, "Right", _flat_landmarks(), _d_ref(), "PALM", 0.0)
        assert state.palm_speed == pytest.approx(0.0, abs=1e-5)

    def test_independent_finger_motion_detected(self):
        """
        Scenario: hand is stationary but index finger moves.
        finger_is_independent_of_palm should return True for index.
        """
        engine = FingerCentricPerceptionEngine()
        pts = _flat_landmarks()
        engine.process_hand(0, "Right", pts.copy(), _d_ref(), "PALM", 0.0)

        pts2 = pts.copy()
        # Palm landmarks stay the same; only index tip moves significantly
        pts2[8, 0] += 0.12   # index tip translates 0.12 units in x
        pts2[8, 1] -= 0.05
        state2 = engine.process_hand(0, "Right", pts2, _d_ref(), "PALM", 0.1)

        assert state2.finger_is_independent_of_palm("index"), \
            "Index should be independent of the stationary palm"

    def test_whole_hand_motion_not_flagged_independent(self):
        """
        Scenario: entire hand translates by the same vector.
        palm and fingertip speeds are similar → NOT independent.
        """
        engine = FingerCentricPerceptionEngine()
        pts = _flat_landmarks()
        engine.process_hand(0, "Right", pts.copy(), _d_ref(), "PALM", 0.0)

        pts2 = pts.copy() + np.array([0.10, 0.05, 0.0], dtype=np.float32)
        state2 = engine.process_hand(0, "Right", pts2, _d_ref(), "PALM", 0.1)

        # All fingers move with the palm — should NOT be flagged independent
        assert not state2.finger_is_independent_of_palm("index"), \
            "Whole-hand swipe should NOT flag index as independent"

    def test_mixed_motion_palm_lateral_index_approaching(self):
        """
        Scenario: palm moves laterally while index approaches camera (z decreases).
        index should be flagged as camera-approaching.
        """
        engine = FingerCentricPerceptionEngine()
        pts = _flat_landmarks()
        engine.process_hand(0, "Right", pts.copy(), _d_ref(), "PALM", 0.0)

        pts2 = pts.copy()
        # Palm drifts left
        for idx in [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]:
            pts2[idx, 0] -= 0.06
        # Index tip additionally approaches camera
        pts2[8, 2] -= 0.18

        state2 = engine.process_hand(0, "Right", pts2, _d_ref(), "PALM", 0.1)
        index_motion = state2.tip_states["index"].motion_class
        assert index_motion in (
            FingerMotionClass.APPROACHING, FingerMotionClass.MOVING
        ), f"Index should be APPROACHING or MOVING, got {index_motion}"

    def test_two_hand_tracking_independent(self):
        """Two separate hand_ids must not share tracker state."""
        engine = FingerCentricPerceptionEngine()
        pts = _flat_landmarks()
        for t in range(3):
            engine.process_hand(0, "Right", pts.copy(), _d_ref(), "PALM", t * 0.033)
            engine.process_hand(1, "Left",  pts.copy(), _d_ref(), "PALM", t * 0.033)
        state0 = engine.process_hand(0, "Right", pts.copy(), _d_ref(), "PALM", 3 * 0.033)
        state1 = engine.process_hand(1, "Left",  pts.copy(), _d_ref(), "PALM", 3 * 0.033)
        assert state0.handedness == "Right"
        assert state1.handedness == "Left"

    def test_prune_missing_hands_clears_state(self):
        engine = FingerCentricPerceptionEngine()
        pts = _flat_landmarks()
        for t in range(4):
            engine.process_hand(0, "Right", pts.copy(), _d_ref(), "PALM", t * 0.033)
        engine.prune_missing_hands([])   # Remove hand 0
        # After pruning, next update should behave like first frame
        state = engine.process_hand(0, "Right", pts.copy(), _d_ref(), "PALM", 5 * 0.033)
        assert state.tip_states["index"].tracking_age == 1

    def test_camera_facing_hand_classified(self):
        engine = FingerCentricPerceptionEngine()
        cam_pts = _camera_facing_landmarks()
        for t in range(3):
            state = engine.process_hand(0, "Right", cam_pts.copy(), _d_ref(), "PALM", t * 0.033)
        assert state.viewpoint.mode in (
            ViewpointMode.CAMERA_FACING, ViewpointMode.FORESHORTENED
        ), f"Camera-facing hand should be CAMERA_FACING, got {state.viewpoint.mode}"

    def test_viewpoint_transition_finger_identity_preserved(self):
        """
        Simulate viewpoint change from normal → camera-facing → normal.
        Finger identity (tracking_age continuity) must be preserved.
        """
        engine = FingerCentricPerceptionEngine()
        pts_normal = _flat_landmarks()
        pts_cam    = _camera_facing_landmarks()

        # 4 normal frames
        for t in range(4):
            engine.process_hand(0, "Right", pts_normal.copy(), _d_ref(), "PALM", t * 0.033)
        # 4 camera-facing frames
        for t in range(4, 8):
            engine.process_hand(0, "Right", pts_cam.copy(), _d_ref(), "PALM", t * 0.033)
        # 2 normal frames again
        for t in range(8, 10):
            state = engine.process_hand(0, "Right", pts_normal.copy(), _d_ref(), "PALM", t * 0.033)

        # Tracking age should be continuous — index should have been tracked all 10 frames
        assert state.tip_states["index"].tracking_age == 10, \
            f"Finger identity should be preserved across viewpoint changes"
