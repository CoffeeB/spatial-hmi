"""
MediaPipe Hands perception module with landmark extraction and kinematic state estimation.
"""

import time
from typing import List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

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
        logger.info("MediaPipe HandDetector initialized with Level 0 FingerStateClassifier.")

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

            # Optical verification for thumb tip: detect physical obstruction (cards, objects, covers)
            vis_list = [float(lm.visibility) for lm in landmark_objs]
            thumb_px_x = int(np.clip(raw_pts[4, 0] * w, 0, w - 1))
            thumb_px_y = int(np.clip(raw_pts[4, 1] * h, 0, h - 1))
            rad = 6
            x1, x2 = max(0, thumb_px_x - rad), min(w, thumb_px_x + rad + 1)
            y1, y2 = max(0, thumb_px_y - rad), min(h, thumb_px_y + rad + 1)
            if x2 > x1 and y2 > y1:
                patch = frame_bgr[y1:y2, x1:x2]
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(hsv, np.array([0, 25, 45]), np.array([28, 240, 255]))
                mask2 = cv2.inRange(hsv, np.array([168, 25, 45]), np.array([180, 240, 255]))
                skin_ratio = float(np.mean((mask1 > 0) | (mask2 > 0)))
                if skin_ratio < 0.20:
                    vis_list[4] = 0.15
                    landmark_objs[4] = HandLandmark(
                        index=4,
                        x=landmark_objs[4].x,
                        y=landmark_objs[4].y,
                        z=landmark_objs[4].z,
                        visibility=0.15,
                    )

            # Level 0 Finger States Classification ("What is every individual finger doing?")
            finger_states = self.finger_classifier.classify_hand(
                raw_landmarks=raw_pts,
                handedness=handedness,
                detection_confidence=detection_conf,
                visibilities=vis_list,
                timestamp=now,
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
                finger_extension_ratios=finger_ratios,
                finger_flexion_angles=flexion_angles,
                pinch_distance=pinch_dist,
                pinch_confidence=pinch_conf,
                finger_states=finger_states,
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

            # Overlay Level 0 Finger States HUD on debug frame
            box_x = 20 if hand_idx == 0 else (w - 200)
            box_y = 30
            cv2.rectangle(
                annotated_frame,
                (box_x - 10, box_y - 20),
                (box_x + 180, box_y + 115),
                (15, 15, 25),
                -1,
            )
            cv2.rectangle(
                annotated_frame,
                (box_x - 10, box_y - 20),
                (box_x + 180, box_y + 115),
                (0, 200, 255),
                1,
            )
            cv2.putText(
                annotated_frame,
                f"Hand {hand_idx} ({handedness})",
                (box_x, box_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            for idx_f, (f_label, f_detail) in enumerate([
                ("Thumb", finger_states.thumb),
                ("Index", finger_states.index),
                ("Middle", finger_states.middle),
                ("Ring", finger_states.ring),
                ("Little", finger_states.little),
            ]):
                state_color = (0, 255, 120) if f_detail.state.value == "extended" else (
                    (0, 200, 255) if f_detail.state.value == "pinching" else (
                        (200, 200, 200) if f_detail.state.value == "folded" else (180, 180, 255)
                    )
                )
                cv2.putText(
                    annotated_frame,
                    f"{f_label:<7}: {f_detail.state.value}",
                    (box_x, box_y + 18 * (idx_f + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    state_color,
                    1,
                    cv2.LINE_AA,
                )

        return hand_states, annotated_frame

    def close(self):
        """Releases the MediaPipe resources."""
        self.hands_detector.close()
