"""
Gestura Spatial Representation — Articulated Finger Geometry & Kinematics.

Transforms discrete 2D appearance classifications into continuous 3D articulated bone geometry:
  1. Articulated kinematic chain: MCP -> PIP -> DIP -> TIP.
  2. Multi-space vector representation:
       - Camera Space: Vector3D (viewpoint-dependent observation)
       - Hand Space: Vector3D (viewpoint-invariant anatomical direction)
       - Screen Space: Vector2D (2D planar projection)
  3. Continuous joint flexion angles (MCP, PIP, DIP in degrees/radians), curl ratio [0, 1],
     abduction/spread angles relative to hand longitudinal axis.
  4. Explicit 3D foreshortening & elevation angle tracking for camera-facing postures.
  5. Soft state categorization with continuous-first measurements:
       EXTENDED, CURVED, FOLDED, TUCKED, UNKNOWN, TRANSITION.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.spatial.hand_frame import HandCoordinateFrame
from src.spatial.spatial_config import SpatialConfig
from src.spatial.spatial_vector import Vector2D, Vector3D


class ContinuousFingerState(str, Enum):
    """Continuous-derived soft finger states."""
    EXTENDED = "EXTENDED"
    CURVED = "CURVED"
    FOLDED = "FOLDED"
    TUCKED = "TUCKED"
    TRANSITION = "TRANSITION"
    UNKNOWN = "UNKNOWN"


# MediaPipe 21 landmark indices per digit: (CMC/MCP, PIP/MCP2, DIP/IP, TIP)
DIGIT_JOINT_INDICES: Dict[str, Tuple[int, int, int, int]] = {
    "thumb": (1, 2, 3, 4),      # CMC, MCP, IP, TIP
    "index": (5, 6, 7, 8),      # MCP, PIP, DIP, TIP
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "little": (17, 18, 19, 20),
}


@dataclass
class ArticulatedFingerGeometry:
    """
    Continuous 3D geometric representation for a single articulated digit.
    Contains both continuous physical measurements and derived soft states.
    """
    finger_name: str
    state: ContinuousFingerState
    confidence: float                    # In [0.0, 1.0]

    # Continuous joint flexion angles in degrees
    mcp_flexion_deg: float
    pip_flexion_deg: float
    dip_flexion_deg: float
    total_flexion_deg: float

    # Continuous volumetric & posture metrics
    curl_ratio: float                    # 0.0 = fully straight/extended, 1.0 = fully curled into fist
    spread_angle_deg: float              # Abduction angle relative to middle finger (+ = radial, - = ulnar)
    distance_to_palm: float              # Tip to palm center normalized by hand scale ref
    foreshortening_ratio: float          # Apparent 2D length / expected 3D length (1.0 = planar, <0.6 = camera-pointing)
    is_foreshortened: bool               # True if digit points heavily along camera optical axis (+Z or -Z)

    # Multi-space pointing vectors (MCP -> TIP direction)
    direction_camera: Vector3D           # Unit vector in raw camera coordinate frame
    direction_hand: Vector3D             # Unit vector in viewpoint-invariant hand coordinate frame
    direction_screen: Vector2D           # 2D projection unit vector on image plane

    # Segment vectors in hand-relative coordinates
    v_proximal: Vector3D                 # MCP -> PIP
    v_intermediate: Vector3D             # PIP -> DIP
    v_distal: Vector3D                   # DIP -> TIP

    # Diagnostic metadata
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, any]:
        """Telemetry representation for visualizer and logging."""
        return {
            "finger": self.finger_name,
            "state": self.state.value,
            "confidence": round(float(self.confidence), 3),
            "flexion": {
                "mcp_deg": round(float(self.mcp_flexion_deg), 1),
                "pip_deg": round(float(self.pip_flexion_deg), 1),
                "dip_deg": round(float(self.dip_flexion_deg), 1),
                "total_deg": round(float(self.total_flexion_deg), 1),
            },
            "curl_ratio": round(float(self.curl_ratio), 3),
            "spread_deg": round(float(self.spread_angle_deg), 1),
            "distance_to_palm": round(float(self.distance_to_palm), 3),
            "foreshortening": {
                "ratio": round(float(self.foreshortening_ratio), 3),
                "is_foreshortened": self.is_foreshortened,
            },
            "direction": {
                "camera": self.direction_camera.as_tuple(),
                "hand_relative": self.direction_hand.as_tuple(),
                "screen_2d": self.direction_screen.as_tuple(),
            },
        }


class FingerGeometryEstimator:
    """
    Extracts continuous articulated 3D kinematics for all 5 digits from raw landmarks.
    Computes hand-relative vectors invariant to camera perspective and hand orientation.
    """

    def __init__(self, config: Optional[SpatialConfig] = None):
        self.config = config or SpatialConfig()
        # Per-digit state hysteresis trackers: finger_name -> last state
        self._prev_states: Dict[str, ContinuousFingerState] = {}
        self._prev_curls: Dict[str, float] = {}

    def reset(self):
        """Resets temporal smoothing history."""
        self._prev_states.clear()
        self._prev_curls.clear()

    @staticmethod
    def _compute_joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Computes 3D joint angle at vertex b between vectors (a - b) and (c - b).
        Straight line = 180 deg (0 deg flexion), folded back = 0 deg (180 deg flexion).
        Returns flexion angle: 180.0 - internal_angle in degrees.
        """
        v1 = a - b
        v2 = c - b
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        internal_angle_deg = float(math.degrees(math.acos(cos_theta)))
        flexion_deg = max(0.0, 180.0 - internal_angle_deg)
        return flexion_deg

    def estimate_finger(
        self,
        finger_name: str,
        landmarks: np.ndarray,
        hand_frame: HandCoordinateFrame,
        palm_center: Tuple[float, float, float],
        landmark_visibilities: Optional[List[float]] = None,
    ) -> ArticulatedFingerGeometry:
        """
        Estimates continuous 3D geometry for a specific digit.

        Args:
            finger_name: 'thumb', 'index', 'middle', 'ring', or 'little'
            landmarks: Array of 21 (x, y, z) 3D points
            hand_frame: Current hand-relative orthonormal coordinate frame
            palm_center: 3D coordinates of palm center
            landmark_visibilities: Optional list of confidence/visibility scores
        """
        j_mcp, j_pip, j_dip, j_tip = DIGIT_JOINT_INDICES[finger_name]
        p_mcp = landmarks[j_mcp]
        p_pip = landmarks[j_pip]
        p_dip = landmarks[j_dip]
        p_tip = landmarks[j_tip]

        # 1. Joint flexion angles
        # For thumb: CMC -> MCP -> IP -> TIP
        # For other digits: Wrist (0) -> MCP -> PIP -> DIP -> TIP
        base_pt = landmarks[0] if finger_name != "thumb" else landmarks[1]
        mcp_flexion = self._compute_joint_angle_deg(base_pt, p_mcp, p_pip)
        pip_flexion = self._compute_joint_angle_deg(p_mcp, p_pip, p_dip)
        dip_flexion = self._compute_joint_angle_deg(p_pip, p_dip, p_tip)
        total_flexion = mcp_flexion + pip_flexion + dip_flexion

        # 2. Articulated bone vectors in Camera Space
        v_prox_cam = p_pip - p_mcp
        v_inter_cam = p_dip - p_pip
        v_dist_cam = p_tip - p_dip
        v_full_cam = p_tip - p_mcp

        # 3. Transform to Hand-Relative Coordinates
        v_prox_hand = hand_frame.camera_to_hand_vector(tuple(v_prox_cam))
        v_inter_hand = hand_frame.camera_to_hand_vector(tuple(v_inter_cam))
        v_dist_hand = hand_frame.camera_to_hand_vector(tuple(v_dist_cam))
        v_full_hand = hand_frame.camera_to_hand_vector(tuple(v_full_cam))

        # 4. Multi-space unit pointing vectors
        dir_cam = Vector3D(float(v_full_cam[0]), float(v_full_cam[1]), float(v_full_cam[2])).normalized()
        dir_hand = v_full_hand.normalized()
        dir_screen = Vector2D(float(v_full_cam[0]), float(v_full_cam[1])).normalized()

        # 5. Continuous Curl Ratio: ratio of tip-to-mcp distance vs total articulated chain length
        seg1_len = float(np.linalg.norm(v_prox_cam))
        seg2_len = float(np.linalg.norm(v_inter_cam))
        seg3_len = float(np.linalg.norm(v_dist_cam))
        total_chain_len = seg1_len + seg2_len + seg3_len
        direct_len = float(np.linalg.norm(v_full_cam))

        if total_chain_len > 1e-5:
            # When fully straight: direct_len approx total_chain_len -> curl = 0.0
            # When fully folded: direct_len approx 0.25 * total_chain_len -> curl = 1.0
            straightness = np.clip(direct_len / total_chain_len, 0.2, 1.0)
            raw_curl = float(1.0 - (straightness - 0.2) / 0.8)
        else:
            raw_curl = 0.5

        # Temporal smoothing on curl
        prev_curl = self._prev_curls.get(finger_name, raw_curl)
        alpha = self.config.temporal_alpha_vector
        smoothed_curl = alpha * raw_curl + (1.0 - alpha) * prev_curl
        self._prev_curls[finger_name] = smoothed_curl

        # 6. Abduction / Spread Angle relative to Middle Finger axis
        # Middle finger longitudinal axis is in the direction of middle MCP in hand frame (+Y axis)
        # Spread is angle between projection of digit onto XY plane and +Y axis
        spread_rad = math.atan2(dir_hand.x, max(1e-4, dir_hand.y))
        spread_deg = float(math.degrees(spread_rad))

        # 7. Normalized Distance to Palm Center
        d_palm = float(np.linalg.norm(p_tip - np.array(palm_center))) / max(hand_frame.scale_ref, 1e-4)

        # 8. Foreshortening & Perspective Compression Detection
        # Compare 2D screen length to 3D length
        len_2d = math.hypot(float(v_full_cam[0]), float(v_full_cam[1]))
        foreshortening_ratio = len_2d / max(direct_len, 1e-5)
        # If the finger points substantially into or out of camera (|dir_cam.z| > 0.70)
        is_foreshortened = (foreshortening_ratio < self.config.foreshortening_ratio_threshold) or (abs(dir_cam.z) > 0.72)

        # 9. Confidence Calculation
        vis_scores = []
        if landmark_visibilities and len(landmark_visibilities) >= 21:
            vis_scores = [landmark_visibilities[idx] for idx in (j_mcp, j_pip, j_dip, j_tip)]
            mean_vis = float(np.mean(vis_scores))
        else:
            mean_vis = 1.0

        # Confidence degrades under extreme foreshortening or low visibility
        confidence = mean_vis
        if is_foreshortened:
            confidence *= max(0.40, foreshortening_ratio)

        # 10. Continuous-Derived Soft Categorical State with Hysteresis
        state = self._derive_finger_state(finger_name, smoothed_curl, total_flexion, d_palm, confidence)

        return ArticulatedFingerGeometry(
            finger_name=finger_name,
            state=state,
            confidence=confidence,
            mcp_flexion_deg=mcp_flexion,
            pip_flexion_deg=pip_flexion,
            dip_flexion_deg=dip_flexion,
            total_flexion_deg=total_flexion,
            curl_ratio=smoothed_curl,
            spread_angle_deg=spread_deg,
            distance_to_palm=d_palm,
            foreshortening_ratio=foreshortening_ratio,
            is_foreshortened=is_foreshortened,
            direction_camera=dir_cam,
            direction_hand=dir_hand,
            direction_screen=dir_screen,
            v_proximal=v_prox_hand,
            v_intermediate=v_inter_hand,
            v_distal=v_dist_hand,
        )

    def _derive_finger_state(
        self,
        finger_name: str,
        curl: float,
        total_flexion_deg: float,
        dist_to_palm: float,
        confidence: float,
    ) -> ContinuousFingerState:
        """
        Derives soft categorical state from continuous metrics with hysteresis.
        Supports: EXTENDED, CURVED, FOLDED, TUCKED, TRANSITION, UNKNOWN.
        """
        if confidence < 0.35:
            return ContinuousFingerState.UNKNOWN

        prev_state = self._prev_states.get(finger_name, ContinuousFingerState.EXTENDED)

        # Thumb has distinct range of motion (more radial abduction / adduction)
        is_thumb = (finger_name == "thumb")
        curl_straight_thresh = 0.38 if is_thumb else 0.32
        curl_folded_thresh = 0.65 if is_thumb else 0.68

        # Hysteresis margins
        hyst = 0.05
        if prev_state == ContinuousFingerState.EXTENDED:
            curl_straight_thresh += hyst
        elif prev_state in (ContinuousFingerState.FOLDED, ContinuousFingerState.TUCKED):
            curl_folded_thresh -= hyst

        # Derive state
        if curl <= curl_straight_thresh and total_flexion_deg < 65.0:
            derived = ContinuousFingerState.EXTENDED
        elif curl >= curl_folded_thresh or total_flexion_deg > 160.0:
            if dist_to_palm < 0.45:
                derived = ContinuousFingerState.TUCKED
            else:
                derived = ContinuousFingerState.FOLDED
        elif 0.35 <= curl <= 0.65:
            # Check for transition zone
            if abs(curl - 0.50) < 0.06:
                derived = ContinuousFingerState.TRANSITION
            else:
                derived = ContinuousFingerState.CURVED
        else:
            derived = ContinuousFingerState.CURVED

        self._prev_states[finger_name] = derived
        return derived

    def estimate_all_fingers(
        self,
        landmarks: np.ndarray,
        hand_frame: HandCoordinateFrame,
        palm_center: Tuple[float, float, float],
        visibilities: Optional[List[float]] = None,
    ) -> Dict[str, ArticulatedFingerGeometry]:
        """Estimates continuous 3D geometry for all five digits."""
        results = {}
        for digit in ("thumb", "index", "middle", "ring", "little"):
            results[digit] = self.estimate_finger(
                finger_name=digit,
                landmarks=landmarks,
                hand_frame=hand_frame,
                palm_center=palm_center,
                landmark_visibilities=visibilities,
            )
        return results
