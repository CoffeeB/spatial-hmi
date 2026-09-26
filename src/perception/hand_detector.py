"""
MediaPipe Hands perception module with landmark extraction and kinematic state estimation.
"""

import time
from typing import List, Optional, Set, Tuple
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
from src.stability.stability_engine import StabilityEngine
from src.utils.logging_config import setup_logger

logger = setup_logger("hand_detector")


class HandDetector:
    """
    Perception module wrapping Google MediaPipe Hands.
    Processes BGR frames into structured, normalized HandState objects.
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.65,
        min_tracking_confidence: float = 0.60,
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
        self._active_interactions: Dict[int, bool] = {}
        logger.info("MediaPipe HandDetector initialized with Stability Engine and TemporalIntentEngine.")

    def notify_interaction_state(self, hand_id: int, is_active: bool) -> None:
        """Notifies perception of downstream active interaction status for state locking."""
        self._active_interactions[hand_id] = is_active

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

        # MediaPipe requires RGB input
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands_detector.process(frame_rgb)
        frame_rgb.flags.writeable = True

        hand_states: List[HandState] = []

        if not results.multi_hand_landmarks:
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
        self.motion_tracker.prune_missing_hands(active_hand_ids)
        self.complete_gesture_recognizer.prune_missing_hands(active_hand_ids)
        self.stability_engine.prune_missing_hands(active_hand_ids)
        self.temporal_intent_engine.prune_missing_hands(active_hand_ids)
        for hid in [0, 1]:
            if hid not in active_hand_ids:
                self.kinematics.reset_hand_tracking(hid)

        return hand_states, annotated_frame

    def close(self):
        """Releases the MediaPipe resources."""
        self.hands_detector.close()
