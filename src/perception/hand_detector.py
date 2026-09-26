"""
MediaPipe Hands perception module with landmark extraction and kinematic state estimation.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import cv2
import mediapipe as mp
import numpy as np

from src.gestures.complete_gesture import CompleteGestureId, CompleteGestureRecognizer
from src.gestures.hand_pose import HandPoseClassifier, HandPoseId
from src.intent.temporal_intent_engine import TemporalIntentEngine
from src.landmarks.finger_state import FingerStateClassifier, HandFingerStates
from src.landmarks.hand_state import HandLandmark, HandState
from src.landmarks.kinematic_features import KinematicFeatureExtractor
from src.landmarks.normalization import LandmarkNormalizer
from src.motion import MotionPrimitiveTracker
from src.spatial import SpatialPerceptionEngine
from src.stability.stability_engine import StabilityEngine
from src.finger_centric import FingerCentricPerceptionEngine
from src.utils.logging_config import setup_logger

logger = setup_logger("hand_detector")


class HandDetector:
    """
    Perception module wrapping Google MediaPipe Hands.
    Processes BGR frames into structured, normalized HandState objects.
    """

    # ── Ghost Buffer Constants ────────────────────────────────────────────────
    # How many consecutive lost-detection frames to replay last-known landmarks.
    # Camera-facing (end-on) orientations cause MediaPipe to miss 3-12 frames;
    # 8 frames at 30 fps = ~267 ms of bridging — enough to survive most transients.
    _GHOST_MAX_FRAMES: int = 8
    # Confidence applied to ghost frames decays linearly from this starting value.
    _GHOST_BASE_CONFIDENCE: float = 0.45

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.50,
        min_tracking_confidence: float = 0.45,
        model_complexity: int = 1,
    ):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            model_complexity=self.model_complexity,
        )

        self.normalizer = LandmarkNormalizer()
        self.kinematics = KinematicFeatureExtractor()
        self.finger_classifier = FingerStateClassifier()
        self.pose_classifier = HandPoseClassifier()
        self.motion_tracker = MotionPrimitiveTracker()
        self.complete_gesture_recognizer = CompleteGestureRecognizer()
        self.stability_engine = StabilityEngine()
        self.temporal_intent_engine = TemporalIntentEngine()
        self.spatial_engine = SpatialPerceptionEngine()
        self.finger_centric_engine = FingerCentricPerceptionEngine()
        self._active_interactions: Dict[int, bool] = {}

        # ── Temporal Ghost Buffer ──────────────────────────────────────────────
        # Keyed by hand_id (0=Left, 1=Right).
        # Stores the last known raw landmarks and handedness so we can replay
        # them when MediaPipe loses the hand during camera-facing orientation.
        self._ghost_landmarks: Dict[int, np.ndarray] = {}   # hand_id -> (21, 3)
        self._ghost_handedness: Dict[int, str] = {}          # hand_id -> "Left"|"Right"
        self._ghost_frames_remaining: Dict[int, int] = {}    # hand_id -> frames left

        # CLAHE for adaptive contrast enhancement (helps MediaPipe find
        # end-on / camera-facing hands whose silhouette has low contrast).
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        logger.info("MediaPipe HandDetector initialized with Spatial Perception, Stability Engine, and TemporalIntentEngine.")

    def notify_interaction_state(self, hand_id: int, is_active: bool) -> None:
        """Notifies perception of downstream active interaction status for state locking."""
        self._active_interactions[hand_id] = is_active

    def _enhance_for_detection(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Applies adaptive histogram equalization (CLAHE) on the luminance channel
        before feeding frames to MediaPipe. This enhances skin-edge contrast for
        camera-facing (end-on) hands whose silhouette is otherwise washed out.
        Returns an RGB frame ready for MediaPipe.
        """
        # Convert BGR -> YCrCb, apply CLAHE only on Y (luma), convert back
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = self._clahe.apply(ycrcb[:, :, 0])
        enhanced_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

    def process_frame(
        self, frame_bgr: np.ndarray, timestamp: Optional[float] = None
    ) -> Tuple[List[HandState], np.ndarray]:
        """
        Extracts 21 3D landmarks for all hands in the frame.
        Returns:
            hand_states: List of structured HandState instances
            annotated_frame: Copy of frame with landmarks drawn for debug mode
        """
        now = timestamp or time.time()
        annotated_frame = frame_bgr.copy()
        h, w, _ = frame_bgr.shape

        # Apply CLAHE contrast enhancement then convert to RGB for MediaPipe
        frame_rgb = self._enhance_for_detection(frame_bgr)
        frame_rgb.flags.writeable = False
        results = self.hands_detector.process(frame_rgb)
        frame_rgb.flags.writeable = True

        hand_states: List[HandState] = []

        if not results.multi_hand_landmarks:
            # ── Temporal Ghost Buffer: replay last-known landmarks ────────────
            # When MediaPipe loses the hand (e.g. camera-facing orientation),
            # we hold the last detected landmarks for up to _GHOST_MAX_FRAMES
            # frames, feeding them through the full pipeline at decayed confidence.
            ghost_states, ghost_annotated = self._process_ghost_frames(frame_bgr, now)
            if ghost_states:
                return ghost_states, ghost_annotated

            # Truly no hand present — prune all downstream buffers
            self._ghost_landmarks.clear()
            self._ghost_handedness.clear()
            self._ghost_frames_remaining.clear()
            self.spatial_engine.prune_missing_hands([])
            self.motion_tracker.prune_missing_hands(set())
            self.kinematics.reset_hand_tracking(0)
            self.kinematics.reset_hand_tracking(1)
            self.temporal_intent_engine.prune_missing_hands(set())
            return hand_states, annotated_frame

        active_hand_ids: Set[int] = set()

        # Determine stable hand IDs across frames based on handedness
        # Left Hand -> hand_id 0, Right Hand -> hand_id 1
        # Disambiguate if duplicate handedness labels occur in rare detector edge cases.
        assigned_hand_ids: List[int] = []
        used_ids: Set[int] = set()
        for handedness_proto in results.multi_handedness:
            label = handedness_proto.classification[0].label
            preferred_id = 0 if label == "Left" else 1
            if preferred_id not in used_ids:
                assigned_hand_ids.append(preferred_id)
                used_ids.add(preferred_id)
            else:
                alt_id = 1 if preferred_id == 0 else 0
                if alt_id not in used_ids:
                    assigned_hand_ids.append(alt_id)
                    used_ids.add(alt_id)
                else:
                    fallback_id = max(used_ids) + 1 if used_ids else 0
                    assigned_hand_ids.append(fallback_id)
                    used_ids.add(fallback_id)

        # MediaPipe found hands — clear ghost state for detected hand IDs
        # (ghost will be re-seeded below after landmark extraction)
        for hid in assigned_hand_ids:
            self._ghost_frames_remaining.pop(hid, None)

        for loop_idx, (landmarks_proto, handedness_proto) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness)
        ):
            hand_id = assigned_hand_ids[loop_idx]

            # Parse handedness label & confidence
            label = handedness_proto.classification[0].label
            handedness = "Left" if label == "Left" else "Right"
            detection_conf = float(handedness_proto.classification[0].score)

            # Extract 21 3D landmarks as numpy array
            raw_pts = np.zeros((21, 3), dtype=np.float32)
            landmark_objs: List[HandLandmark] = []

            for i, lm in enumerate(landmarks_proto.landmark):
                raw_pts[i] = [lm.x, lm.y, lm.z]
                # In MediaPipe Hands protobuf, lm.visibility is not populated by the Hand model;
                # reading lm.visibility without HasField gives 0.0, corrupting confidence.
                vis_val = float(lm.visibility) if lm.HasField("visibility") else 1.0
                landmark_objs.append(
                    HandLandmark(
                        index=i,
                        x=float(lm.x),
                        y=float(lm.y),
                        z=float(lm.z),
                        visibility=vis_val,
                    )
                )

            # Scale and translation normalization
            normalized_pts, palm_center_arr, d_ref = self.normalizer.normalize(raw_pts)
            palm_center = (float(palm_center_arr[0]), float(palm_center_arr[1]), float(palm_center_arr[2]))

            # Kinematic features
            finger_ratios = self.kinematics.compute_finger_extension_ratios(raw_pts)
            flexion_angles = self.kinematics.compute_joint_flexion_angles(raw_pts)
            pinch_dist, pinch_conf = self.kinematics.compute_pinch_metric(raw_pts, d_ref)
            orientation = self.kinematics.compute_hand_orientation(raw_pts)
            palm_velocity = self.kinematics.compute_palm_velocity(hand_id, palm_center_arr, now)

            # Per-landmark visibility scores
            vis_list = [float(lm.visibility) for lm in landmark_objs]

            # Level 0 Finger States Classification ("What is every individual finger doing?")
            finger_states = self.finger_classifier.classify_hand(
                raw_landmarks=raw_pts,
                handedness=handedness,
                detection_confidence=detection_conf,
                visibilities=vis_list,
                timestamp=now,
            )

            # Level 1 Static Hand Pose Derivation (Finger States -> Finger Configuration -> Hand Pose)
            derived_pose = self.pose_classifier.classify_pose(
                finger_states=finger_states,
                raw_landmarks=raw_pts,
                d_ref=d_ref,
                orientation_angles=orientation,
            )

            # Level 2 Motion Primitives ("How is the hand and every individual finger moving over time?")
            # Hand tracking is completely independent per stable hand_id with dorsal inclusion
            motion_state = self.motion_tracker.update(
                hand_id=hand_id,
                handedness=handedness,
                palm_center=palm_center,
                hand_scale_ref=d_ref,
                timestamp=now,
                raw_landmarks=raw_pts,
                palm_facing=finger_states.palm_facing,
                orientation_angles=orientation,
                finger_ratios=finger_ratios,
            )
            active_hand_ids.add(hand_id)

            # Level 3 Complete Gestures ("Compose Pose + Motion into observable event sequences")
            complete_gesture = self.complete_gesture_recognizer.update(
                hand_id=hand_id,
                handedness=handedness,
                derived_pose=derived_pose,
                motion_state=motion_state,
                finger_states=finger_states,
                raw_landmarks=raw_pts,
                d_ref=d_ref,
                timestamp=now,
            )

            # Level 4 & 5 Temporal Intent Engine (Observation Windows, Stability, Intent Telemetry)
            # Prioritize active dynamic complete gesture; fall back to static derived hand pose
            if (
                complete_gesture is not None
                and hasattr(complete_gesture, "gesture_id")
                and complete_gesture.gesture_id != CompleteGestureId.NONE
                and getattr(complete_gesture, "confidence", 0.0) > 0.0
            ):
                candidate_gname = complete_gesture.gesture_id.value
                candidate_gconf = complete_gesture.confidence
            elif (
                derived_pose is not None
                and hasattr(derived_pose, "pose_id")
                and derived_pose.pose_id != HandPoseId.UNKNOWN
                and getattr(derived_pose, "confidence", 0.0) > 0.0
            ):
                candidate_gname = derived_pose.canonical_name
                candidate_gconf = derived_pose.confidence
            else:
                candidate_gname = "NONE"
                candidate_gconf = 0.0

            # ── Level 3.5: STABILITY ENGINE ──────────────────────────────────────────
            # Evaluates finger persistence, pose hysteresis, dead-zone, and micro-adjustments
            stab_ctx = self.stability_engine.process_hand_stability(
                hand_id=hand_id,
                handedness=handedness,
                raw_landmarks=raw_pts,
                instantaneous_finger_states=finger_states,
                instantaneous_pose=derived_pose,
                instantaneous_motion=motion_state,
                candidate_gesture=candidate_gname,
                candidate_confidence=candidate_gconf,
                palm_center=palm_center,
                d_ref=d_ref,
                detection_confidence=detection_conf,
                is_active_interaction=self._active_interactions.get(hand_id, False),
                timestamp=now,
            )

            (
                stab_finger_states,
                conf_derived_pose,
                conf_motion_prim,
                intent_ctx,
                temporal_telem,
            ) = self.temporal_intent_engine.process_hand_temporal(
                hand_id=hand_id,
                handedness=handedness,
                raw_landmarks=raw_pts,
                instantaneous_finger_states=stab_ctx.stabilized_finger_states,
                instantaneous_pose=stab_ctx.stabilized_derived_pose,
                palm_center=stab_ctx.filtered_palm_pos,
                d_ref=d_ref,
                detection_confidence=detection_conf,
                candidate_gesture_name=candidate_gname,
                candidate_confidence=candidate_gconf,
                timestamp=now,
            )

            # ── SPATIAL REPRESENTATION & PERCEPTION OBJECT ────────────────────────────
            # Computes continuous 3D articulated state, multi-space vectors, and uncertainty
            perception_obj = self.spatial_engine.process_hand(
                hand_id=hand_id,
                handedness=handedness,
                landmarks=raw_pts,
                palm_center=stab_ctx.filtered_palm_pos,
                scale_ref=d_ref,
                timestamp=now,
                landmark_visibilities=vis_list,
                is_transitioning=stab_ctx.is_transitioning,
            )
            perception_obj.stability_state = "TRANSITION" if stab_ctx.is_transitioning else "STABLE"
            perception_obj.stability_score = stab_ctx.stability_score
            perception_obj.derived_pose = (
                conf_derived_pose.canonical_name
                if (conf_derived_pose and conf_derived_pose.pose_id != HandPoseId.UNKNOWN)
                else (derived_pose.canonical_name if derived_pose else None)
            )
            perception_obj.discrete_gesture = (
                complete_gesture.gesture_id.value
                if (complete_gesture and complete_gesture.gesture_id != CompleteGestureId.NONE)
                else None
            )

            # ── FINGER-CENTRIC PERCEPTION (parallel evidence stream) ────────────────
            finger_centric_state = self.finger_centric_engine.process_hand(
                hand_id=hand_id,
                handedness=handedness,
                raw_landmarks=raw_pts,
                d_ref=d_ref,
                palm_facing=finger_states.palm_facing,
                timestamp=now,
                visibilities=vis_list,
                ext_ratios=finger_ratios,
                motion_confidence=float(getattr(motion_state, "confidence", 0.80) or 0.80),
                temporal_confidence=float(stab_ctx.stability_score or 0.80),
            )

            # Build immutable HandState
            state = HandState(
                hand_id=hand_id,
                handedness=handedness,
                landmarks=landmark_objs,
                raw_landmarks_array=raw_pts,
                normalized_landmarks_array=normalized_pts,
                palm_center=stab_ctx.filtered_palm_pos,
                palm_velocity=stab_ctx.filtered_palm_velocity,
                hand_scale_ref=d_ref,
                orientation_angles=orientation,
                palm_facing=finger_states.palm_facing,
                finger_extension_ratios=finger_ratios,
                finger_flexion_angles=flexion_angles,
                pinch_distance=pinch_dist,
                pinch_confidence=pinch_conf,
                finger_states=stab_finger_states or finger_states,
                derived_pose=conf_derived_pose if (conf_derived_pose and conf_derived_pose.pose_id != HandPoseId.UNKNOWN) else derived_pose,
                motion_state=motion_state,
                complete_gesture=complete_gesture,
                temporal_telemetry=temporal_telem.as_dict() if temporal_telem else {},
                stability_context=stab_ctx,
                stability_telemetry=stab_ctx.telemetry,
                perception=perception_obj,
                spatial_telemetry=perception_obj.to_dict(),
                finger_centric=finger_centric_state,
                finger_centric_telemetry=finger_centric_state.to_dict(),
                intent_context=intent_ctx,
                detection_confidence=detection_conf,
                timestamp=now,
            )
            hand_states.append(state)

            # Draw landmarks onto annotated debug frame
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                landmarks_proto,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

        # Prune motion buffers, complete gestures, and kinematics for hands no longer in frame
        self.spatial_engine.prune_missing_hands(list(active_hand_ids))
        self.motion_tracker.prune_missing_hands(active_hand_ids)
        self.complete_gesture_recognizer.prune_missing_hands(active_hand_ids)
        self.stability_engine.prune_missing_hands(active_hand_ids)
        self.temporal_intent_engine.prune_missing_hands(active_hand_ids)
        self.finger_centric_engine.prune_missing_hands(list(active_hand_ids))
        for hid in [0, 1]:
            if hid not in active_hand_ids:
                self.kinematics.reset_hand_tracking(hid)

        # Seed ghost buffer with current live detections for next-frame bridging
        for state in hand_states:
            self._ghost_landmarks[state.hand_id] = state.raw_landmarks_array.copy()
            self._ghost_handedness[state.hand_id] = state.handedness
            self._ghost_frames_remaining[state.hand_id] = self._GHOST_MAX_FRAMES

        return hand_states, annotated_frame

    def _process_ghost_frames(
        self, frame_bgr: np.ndarray, now: float
    ) -> Tuple[List[HandState], np.ndarray]:
        """
        Replays last-known landmarks through the full pipeline for any hand_id
        that still has ghost frames remaining. Confidence is linearly decayed
        to signal downstream components that this is estimated, not measured data.

        This bridges orientation-transition gaps where MediaPipe's palm detector
        temporarily loses the hand (e.g. fingers pointing toward camera).
        """
        ghost_states: List[HandState] = []
        annotated_frame = frame_bgr.copy()
        active_ghost_ids: Set[int] = set()

        for hand_id in list(self._ghost_frames_remaining.keys()):
            remaining = self._ghost_frames_remaining[hand_id]
            if remaining <= 0:
                # Ghost expired — remove from buffer
                self._ghost_landmarks.pop(hand_id, None)
                self._ghost_handedness.pop(hand_id, None)
                del self._ghost_frames_remaining[hand_id]
                continue

            raw_pts = self._ghost_landmarks.get(hand_id)
            handedness = self._ghost_handedness.get(hand_id, "Right")
            if raw_pts is None:
                continue

            # Decay confidence linearly: full at frame 8, near-zero at frame 1
            decay = remaining / self._GHOST_MAX_FRAMES
            ghost_conf = self._GHOST_BASE_CONFIDENCE * decay

            self._ghost_frames_remaining[hand_id] = remaining - 1
            active_ghost_ids.add(hand_id)

            # Run the full landmark pipeline with decayed confidence
            state = self._process_single_hand(
                hand_id=hand_id,
                handedness=handedness,
                raw_pts=raw_pts,
                detection_conf=ghost_conf,
                now=now,
                annotated_frame=annotated_frame,
                draw_landmarks=False,  # Don't draw ghost landmarks on debug frame
            )
            if state is not None:
                ghost_states.append(state)

        # Prune downstream buffers for hands no longer ghosted
        if active_ghost_ids:
            self.spatial_engine.prune_missing_hands(list(active_ghost_ids))
            self.motion_tracker.prune_missing_hands(active_ghost_ids)
            self.complete_gesture_recognizer.prune_missing_hands(active_ghost_ids)
            self.stability_engine.prune_missing_hands(active_ghost_ids)
            self.temporal_intent_engine.prune_missing_hands(active_ghost_ids)
            self.finger_centric_engine.prune_missing_hands(list(active_ghost_ids))

        return ghost_states, annotated_frame

    def _process_single_hand(
        self,
        hand_id: int,
        handedness: str,
        raw_pts: np.ndarray,
        detection_conf: float,
        now: float,
        annotated_frame: np.ndarray,
        draw_landmarks: bool = True,
    ) -> Optional[HandState]:
        """
        Runs the full perception pipeline for a single hand given its raw landmark array.
        Extracted from process_frame to allow ghost-frame replay without code duplication.
        """
        from src.gestures.hand_pose import HandPoseId
        from src.gestures.complete_gesture import CompleteGestureId

        # Parse landmark objects from raw array
        landmark_objs: List[HandLandmark] = [
            HandLandmark(index=i, x=float(raw_pts[i, 0]), y=float(raw_pts[i, 1]),
                         z=float(raw_pts[i, 2]), visibility=1.0)
            for i in range(21)
        ]
        vis_list = [1.0] * 21

        # Scale and translation normalization
        normalized_pts, palm_center_arr, d_ref = self.normalizer.normalize(raw_pts)
        palm_center = (float(palm_center_arr[0]), float(palm_center_arr[1]), float(palm_center_arr[2]))

        # Kinematic features
        finger_ratios = self.kinematics.compute_finger_extension_ratios(raw_pts)
        flexion_angles = self.kinematics.compute_joint_flexion_angles(raw_pts)
        pinch_dist, pinch_conf = self.kinematics.compute_pinch_metric(raw_pts, d_ref)
        orientation = self.kinematics.compute_hand_orientation(raw_pts)
        palm_velocity = self.kinematics.compute_palm_velocity(hand_id, palm_center_arr, now)

        # Level 0 Finger States
        finger_states = self.finger_classifier.classify_hand(
            raw_landmarks=raw_pts,
            handedness=handedness,
            detection_confidence=detection_conf,
            visibilities=vis_list,
            timestamp=now,
        )

        # Level 1 Static Hand Pose
        derived_pose = self.pose_classifier.classify_pose(
            finger_states=finger_states,
            raw_landmarks=raw_pts,
            d_ref=d_ref,
            orientation_angles=orientation,
        )

        # Level 2 Motion Primitives
        motion_state = self.motion_tracker.update(
            hand_id=hand_id,
            handedness=handedness,
            palm_center=palm_center,
            hand_scale_ref=d_ref,
            timestamp=now,
            raw_landmarks=raw_pts,
            palm_facing=finger_states.palm_facing,
            orientation_angles=orientation,
            finger_ratios=finger_ratios,
        )

        # Level 3 Complete Gestures
        complete_gesture = self.complete_gesture_recognizer.update(
            hand_id=hand_id,
            handedness=handedness,
            derived_pose=derived_pose,
            motion_state=motion_state,
            finger_states=finger_states,
            raw_landmarks=raw_pts,
            d_ref=d_ref,
            timestamp=now,
        )

        # Candidate gesture for temporal engine
        if (
            complete_gesture is not None
            and hasattr(complete_gesture, "gesture_id")
            and complete_gesture.gesture_id != CompleteGestureId.NONE
            and getattr(complete_gesture, "confidence", 0.0) > 0.0
        ):
            candidate_gname = complete_gesture.gesture_id.value
            candidate_gconf = complete_gesture.confidence
        elif (
            derived_pose is not None
            and hasattr(derived_pose, "pose_id")
            and derived_pose.pose_id != HandPoseId.UNKNOWN
            and getattr(derived_pose, "confidence", 0.0) > 0.0
        ):
            candidate_gname = derived_pose.canonical_name
            candidate_gconf = derived_pose.confidence
        else:
            candidate_gname = "NONE"
            candidate_gconf = 0.0

        # Stability Engine
        stab_ctx = self.stability_engine.process_hand_stability(
            hand_id=hand_id,
            handedness=handedness,
            raw_landmarks=raw_pts,
            instantaneous_finger_states=finger_states,
            instantaneous_pose=derived_pose,
            instantaneous_motion=motion_state,
            candidate_gesture=candidate_gname,
            candidate_confidence=candidate_gconf,
            palm_center=palm_center,
            d_ref=d_ref,
            detection_confidence=detection_conf,
            is_active_interaction=self._active_interactions.get(hand_id, False),
            timestamp=now,
        )

        (
            stab_finger_states,
            conf_derived_pose,
            conf_motion_prim,
            intent_ctx,
            temporal_telem,
        ) = self.temporal_intent_engine.process_hand_temporal(
            hand_id=hand_id,
            handedness=handedness,
            raw_landmarks=raw_pts,
            instantaneous_finger_states=stab_ctx.stabilized_finger_states,
            instantaneous_pose=stab_ctx.stabilized_derived_pose,
            palm_center=stab_ctx.filtered_palm_pos,
            d_ref=d_ref,
            detection_confidence=detection_conf,
            candidate_gesture_name=candidate_gname,
            candidate_confidence=candidate_gconf,
            timestamp=now,
        )

        # Spatial Perception
        perception_obj = self.spatial_engine.process_hand(
            hand_id=hand_id,
            handedness=handedness,
            landmarks=raw_pts,
            palm_center=stab_ctx.filtered_palm_pos,
            scale_ref=d_ref,
            timestamp=now,
            landmark_visibilities=vis_list,
            is_transitioning=stab_ctx.is_transitioning,
        )
        perception_obj.stability_state = "TRANSITION" if stab_ctx.is_transitioning else "STABLE"
        perception_obj.stability_score = stab_ctx.stability_score
        perception_obj.derived_pose = (
            conf_derived_pose.canonical_name
            if (conf_derived_pose and conf_derived_pose.pose_id != HandPoseId.UNKNOWN)
            else (derived_pose.canonical_name if derived_pose else None)
        )
        perception_obj.discrete_gesture = (
            complete_gesture.gesture_id.value
            if (complete_gesture and complete_gesture.gesture_id != CompleteGestureId.NONE)
            else None
        )

        # Finger-Centric Perception (ghost path)
        finger_centric_state = self.finger_centric_engine.process_hand(
            hand_id=hand_id,
            handedness=handedness,
            raw_landmarks=raw_pts,
            d_ref=d_ref,
            palm_facing=finger_states.palm_facing,
            timestamp=now,
            visibilities=vis_list,
            ext_ratios=finger_ratios,
            motion_confidence=float(getattr(motion_state, "confidence", 0.60) or 0.60),
            temporal_confidence=float(detection_conf),  # ghost conf carries temporal quality
        )

        return HandState(
            hand_id=hand_id,
            handedness=handedness,
            landmarks=landmark_objs,
            raw_landmarks_array=raw_pts,
            normalized_landmarks_array=normalized_pts,
            palm_center=stab_ctx.filtered_palm_pos,
            palm_velocity=stab_ctx.filtered_palm_velocity,
            hand_scale_ref=d_ref,
            orientation_angles=orientation,
            palm_facing=finger_states.palm_facing,
            finger_extension_ratios=finger_ratios,
            finger_flexion_angles=flexion_angles,
            pinch_distance=pinch_dist,
            pinch_confidence=pinch_conf,
            finger_states=stab_finger_states or finger_states,
            derived_pose=conf_derived_pose if (conf_derived_pose and conf_derived_pose.pose_id != HandPoseId.UNKNOWN) else derived_pose,
            motion_state=motion_state,
            complete_gesture=complete_gesture,
            temporal_telemetry=temporal_telem.as_dict() if temporal_telem else {},
            stability_context=stab_ctx,
            stability_telemetry=stab_ctx.telemetry,
            perception=perception_obj,
            spatial_telemetry=perception_obj.to_dict(),
            finger_centric=finger_centric_state,
            finger_centric_telemetry=finger_centric_state.to_dict(),
            intent_context=intent_ctx,
            detection_confidence=detection_conf,
            timestamp=now,
        )

    def close(self):
        """Releases the MediaPipe resources."""
        self.hands_detector.close()
