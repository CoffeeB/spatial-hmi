"""
MediaPipe Hands perception module with landmark extraction and kinematic state estimation.
"""

import time
from typing import List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

from src.gestures.hand_pose import HandPoseClassifier
from src.landmarks.finger_state import FingerStateClassifier, HandFingerStates
from src.landmarks.hand_state import HandLandmark, HandState
from src.landmarks.kinematic_features import KinematicFeatureExtractor
from src.landmarks.normalization import LandmarkNormalizer
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
        logger.info("MediaPipe HandDetector initialized with Level 0 FingerStateClassifier and Level 1 HandPoseClassifier.")

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
            return hand_states, annotated_frame

        for hand_idx, (landmarks_proto, handedness_proto) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness)
        ):
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
            palm_velocity = self.kinematics.compute_palm_velocity(hand_idx, palm_center_arr, now)

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

            # Build immutable HandState
            state = HandState(
                hand_id=hand_idx,
                handedness=handedness,
                landmarks=landmark_objs,
                raw_landmarks_array=raw_pts,
                normalized_landmarks_array=normalized_pts,
                palm_center=palm_center,
                palm_velocity=palm_velocity,
                hand_scale_ref=d_ref,
                orientation_angles=orientation,
                palm_facing=finger_states.palm_facing,
                finger_extension_ratios=finger_ratios,
                finger_flexion_angles=flexion_angles,
                pinch_distance=pinch_dist,
                pinch_confidence=pinch_conf,
                finger_states=finger_states,
                derived_pose=derived_pose,
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

        return hand_states, annotated_frame

    def close(self):
        """Releases the MediaPipe resources."""
        self.hands_detector.close()
