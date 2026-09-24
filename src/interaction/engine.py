"""
Core spatial interaction engine orchestrating perception, intent, smoothing, and command mapping.
"""

import time
from typing import List, Optional, Tuple
import numpy as np

from src.gestures.heuristic_classifier import HeuristicGestureClassifier
from src.intent.state_machine import IntentContext, InteractionStateMachine
from src.interaction.coordinate_transform import CoordinateTransformer
from src.interaction.mapping import InteractionMapper, SpatialCommand
from src.interaction.smoothing import OneEuroFilter
from src.landmarks.hand_state import HandState
from src.utils.config_loader import HMIConfig
from src.utils.logging_config import setup_logger

logger = setup_logger("interaction_engine")


class InteractionEngine:
    """
    Central orchestration engine for the Spatial Human-Machine Interface.
    Maintains clean separation between perceptual tracking and downstream application commands.
    """

    def __init__(self, config: Optional[HMIConfig] = None):
        self.config = config or HMIConfig()

        self.classifier = HeuristicGestureClassifier(
            pinch_threshold=self.config.gestures.pinch_threshold,
            pinch_steepness=self.config.gestures.pinch_steepness,
            point_extension_ratio=self.config.gestures.point_extension_ratio,
            open_palm_extension_ratio=self.config.gestures.open_palm_extension_ratio,
            grab_closure_ratio=self.config.gestures.grab_closure_ratio,
            two_hand_spread_vel_threshold=self.config.gestures.two_hand_spread_vel_threshold,
            two_hand_rotation_vel_threshold=self.config.gestures.two_hand_rotation_vel_threshold,
            min_confidence_threshold=self.config.gestures.min_confidence_threshold,
        )

        self.fsm = InteractionStateMachine(
            activation_threshold=self.config.intent.activation_threshold,
            release_threshold=self.config.intent.release_threshold,
            confirm_frames=self.config.intent.confirm_frames,
            candidate_frames=self.config.intent.candidate_frames,
            evidence_lambda=self.config.intent.evidence_lambda,
            hand_loss_timeout_sec=self.config.intent.hand_loss_timeout_sec,
        )

        self.transformer = CoordinateTransformer(
            aspect_ratio=self.config.interaction.screen_aspect_ratio,
            mirror_horizontal=True,
        )

        self.cursor_filter = OneEuroFilter(
            fc_min=self.config.smoothing.one_euro.fc_min,
            beta=self.config.smoothing.one_euro.beta,
            d_cutoff=self.config.smoothing.one_euro.d_cutoff,
        )

        self.mapper = InteractionMapper(
            rotation_sensitivity=self.config.interaction.rotation_sensitivity,
            zoom_sensitivity=self.config.interaction.zoom_sensitivity,
            translation_sensitivity=self.config.interaction.translation_sensitivity,
            dead_zone_radius=self.config.smoothing.dead_zone_radius,
        )

        logger.info("InteractionEngine initialized.")

    def process_hands(
        self, hand_states: List[HandState], timestamp: Optional[float] = None
    ) -> Tuple[SpatialCommand, IntentContext, Optional[HandState]]:
        """
        Processes detected hand states through classification, intent FSM,
        filtering, and mapping to generate a SpatialCommand.
        """
        now = timestamp or time.time()

        if not hand_states:
            intent_ctx = self.fsm.update(hand_detected=False, recognized_gesture=None, timestamp=now)
            command = self.mapper.map_to_command(
                intent_ctx=intent_ctx,
                primary_hand=None,
                bimanual_gesture=None,
                smoothed_cursor_ndc=(0.0, 0.0),
                timestamp=now,
            )
            return command, intent_ctx, None

        # Select primary dominant hand (or first hand detected)
        primary_hand = hand_states[0]

        # 1. Single Hand Classification
        single_gesture = self.classifier.classify_single_hand(primary_hand)

        # 2. Bimanual Classification if two hands present
        bimanual_gesture = None
        if len(hand_states) >= 2:
            bimanual_gesture = self.classifier.classify_two_hands(hand_states[0], hand_states[1])

        # Active recognized gesture for FSM update
        eval_gesture = bimanual_gesture if bimanual_gesture is not None else single_gesture

        # 3. Temporal Intent FSM Update
        intent_ctx = self.fsm.update(
            hand_detected=True,
            recognized_gesture=eval_gesture,
            timestamp=now,
        )

        # 4. Coordinate Transformation & Spatial Smoothing
        # Use index tip or palm center as interaction focal point
        focal_x = primary_hand.palm_center[0]
        focal_y = primary_hand.palm_center[1]

        # Convert normalized camera coords [0, 1] to NDC [-1, 1]
        # (Camera coordinates are already normalized with mirror flip by LandmarkNormalizer / Transformer)
        raw_ndc_x, raw_ndc_y = self.transformer.normalized_to_ndc(focal_x, focal_y)

        # Apply 1€ Adaptive Smoothing
        smoothed_ndc = self.cursor_filter.filter([raw_ndc_x, raw_ndc_y], timestamp=now)
        smoothed_tuple = (float(smoothed_ndc[0]), float(smoothed_ndc[1]))

        # 5. Map to Generic Spatial Command
        command = self.mapper.map_to_command(
            intent_ctx=intent_ctx,
            primary_hand=primary_hand,
            bimanual_gesture=bimanual_gesture,
            smoothed_cursor_ndc=smoothed_tuple,
            timestamp=now,
        )

        return command, intent_ctx, primary_hand
