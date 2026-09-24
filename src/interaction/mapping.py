"""
Interaction mapping engine translating intents and continuous hand kinematics to generic 3D spatial commands.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel

from src.gestures.gesture_types import GestureType, RecognizedGesture
from src.intent.state_machine import IntentContext, InteractionState
from src.landmarks.hand_state import HandState


class SpatialCommandType(str, Enum):
    """Generic spatial commands decoupled from specific 3D applications."""
    IDLE = "IDLE"
    HOVER = "HOVER"
    SELECT = "SELECT"
    ROTATE_OBJECT = "ROTATE_OBJECT"
    SCALE_OBJECT = "SCALE_OBJECT"
    TRANSLATE_NODE = "TRANSLATE_NODE"
    RELEASE_OBJECT = "RELEASE_OBJECT"
    CANCEL_INTERACTION = "CANCEL_INTERACTION"


class SpatialCommand(BaseModel):
    """
    Standardized payload transmitted to the 3D visualization or application engine.
    """
    command_type: SpatialCommandType
    interaction_state: str  # IDLE, OBSERVING, CANDIDATE, CONFIRMED, ACTIVE, RELEASING
    cursor_ndc: Tuple[float, float]  # Normalized device coordinates (-1 to 1)
    delta_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (yaw, pitch, roll)
    delta_scale: float = 1.0  # Multiplicative zoom factor
    delta_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (dx, dy, dz)
    target_node_id: Optional[str] = None
    confidence: float = 1.0
    handedness: str = "Right"
    is_pinch_active: bool = False
    timestamp: float = 0.0


class InteractionMapper:
    """
    Maps (IntentContext, HandState, Kinematics) into a clean SpatialCommand.
    Enforces dead-zones and movement thresholding.
    """

    def __init__(
        self,
        rotation_sensitivity: float = 2.8,
        zoom_sensitivity: float = 1.8,
        translation_sensitivity: float = 2.2,
        dead_zone_radius: float = 0.003,
    ):
        self.rotation_sensitivity = rotation_sensitivity
        self.zoom_sensitivity = zoom_sensitivity
        self.translation_sensitivity = translation_sensitivity
        self.dead_zone_radius = dead_zone_radius

        self.prev_cursor_ndc: Optional[Tuple[float, float]] = None

    def map_to_command(
        self,
        intent_ctx: IntentContext,
        primary_hand: Optional[HandState],
        bimanual_gesture: Optional[RecognizedGesture],
        smoothed_cursor_ndc: Tuple[float, float],
        timestamp: float,
    ) -> SpatialCommand:
        """Translates current perceptual context to a high-level spatial command."""
        state = intent_ctx.state
        active_g = intent_ctx.active_gesture

        # Default IDLE / HOVER command
        if state == InteractionState.IDLE or primary_hand is None:
            self.prev_cursor_ndc = None
            return SpatialCommand(
                command_type=SpatialCommandType.IDLE,
                interaction_state=state.value,
                cursor_ndc=(0.0, 0.0),
                confidence=0.0,
                timestamp=timestamp,
            )

        curr_x, curr_y = smoothed_cursor_ndc
        dx, dy = 0.0, 0.0
        if self.prev_cursor_ndc is not None:
            raw_dx = curr_x - self.prev_cursor_ndc[0]
            raw_dy = curr_y - self.prev_cursor_ndc[1]
            dist = (raw_dx**2 + raw_dy**2) ** 0.5
            if dist > self.dead_zone_radius:
                dx = raw_dx
                dy = raw_dy

        self.prev_cursor_ndc = (curr_x, curr_y)

        # 1. Bimanual Zoom / Rotation
        if bimanual_gesture is not None and bimanual_gesture.gesture in (
            GestureType.SPREAD,
            GestureType.CONTRACTION,
            GestureType.ROTATION,
        ):
            if bimanual_gesture.gesture == GestureType.SPREAD:
                scale_factor = 1.0 + self.zoom_sensitivity * abs(bimanual_gesture.feature_contributions.get("radial_velocity", 0.1))
                return SpatialCommand(
                    command_type=SpatialCommandType.SCALE_OBJECT,
                    interaction_state=state.value,
                    cursor_ndc=(curr_x, curr_y),
                    delta_scale=float(scale_factor),
                    confidence=bimanual_gesture.confidence,
                    timestamp=timestamp,
                )
            elif bimanual_gesture.gesture == GestureType.CONTRACTION:
                scale_factor = 1.0 - self.zoom_sensitivity * abs(bimanual_gesture.feature_contributions.get("radial_velocity", 0.1))
                return SpatialCommand(
                    command_type=SpatialCommandType.SCALE_OBJECT,
                    interaction_state=state.value,
                    cursor_ndc=(curr_x, curr_y),
                    delta_scale=max(0.1, float(scale_factor)),
                    confidence=bimanual_gesture.confidence,
                    timestamp=timestamp,
                )

        # 2. Single-Hand Interaction Mapping based on FSM State
        if state in (InteractionState.OBSERVING, InteractionState.CANDIDATE):
            cmd_type = SpatialCommandType.HOVER
            return SpatialCommand(
                command_type=cmd_type,
                interaction_state=state.value,
                cursor_ndc=(curr_x, curr_y),
                confidence=intent_ctx.intent_confidence,
                handedness=primary_hand.handedness,
                is_pinch_active=False,
                timestamp=timestamp,
            )

        elif state == InteractionState.CONFIRMED:
            if active_g == GestureType.POINT:
                return SpatialCommand(
                    command_type=SpatialCommandType.SELECT,
                    interaction_state=state.value,
                    cursor_ndc=(curr_x, curr_y),
                    confidence=intent_ctx.intent_confidence,
                    handedness=primary_hand.handedness,
                    is_pinch_active=False,
                    timestamp=timestamp,
                )
            elif active_g == GestureType.PINCH:
                return SpatialCommand(
                    command_type=SpatialCommandType.TRANSLATE_NODE,
                    interaction_state=state.value,
                    cursor_ndc=(curr_x, curr_y),
                    confidence=intent_ctx.intent_confidence,
                    handedness=primary_hand.handedness,
                    is_pinch_active=True,
                    timestamp=timestamp,
                )
            elif active_g == GestureType.GRAB:
                return SpatialCommand(
                    command_type=SpatialCommandType.ROTATE_OBJECT,
                    interaction_state=state.value,
                    cursor_ndc=(curr_x, curr_y),
                    confidence=intent_ctx.intent_confidence,
                    handedness=primary_hand.handedness,
                    timestamp=timestamp,
                )

        elif state == InteractionState.ACTIVE:
            if active_g == GestureType.GRAB:
                # Grab + movement -> Rotate 3D Globe
                delta_yaw = -dx * self.rotation_sensitivity
                delta_pitch = dy * self.rotation_sensitivity
                return SpatialCommand(
                    command_type=SpatialCommandType.ROTATE_OBJECT,
                    interaction_state=state.value,
                    cursor_ndc=(curr_x, curr_y),
                    delta_rotation=(float(delta_yaw), float(delta_pitch), 0.0),
                    confidence=intent_ctx.intent_confidence,
                    handedness=primary_hand.handedness,
                    timestamp=timestamp,
                )
            elif active_g == GestureType.PINCH:
                # Pinch + movement -> Translate/Manipulate selected spatial node
                return SpatialCommand(
                    command_type=SpatialCommandType.TRANSLATE_NODE,
                    interaction_state=state.value,
                    cursor_ndc=(curr_x, curr_y),
                    delta_translation=(float(dx * self.translation_sensitivity), float(dy * self.translation_sensitivity), 0.0),
                    confidence=intent_ctx.intent_confidence,
                    handedness=primary_hand.handedness,
                    is_pinch_active=True,
                    timestamp=timestamp,
                )
            else:
                return SpatialCommand(
                    command_type=SpatialCommandType.HOVER,
                    interaction_state=state.value,
                    cursor_ndc=(curr_x, curr_y),
                    confidence=intent_ctx.intent_confidence,
                    timestamp=timestamp,
                )

        elif state == InteractionState.RELEASING:
            return SpatialCommand(
                command_type=SpatialCommandType.RELEASE_OBJECT,
                interaction_state=state.value,
                cursor_ndc=(curr_x, curr_y),
                confidence=intent_ctx.intent_confidence,
                timestamp=timestamp,
            )

        return SpatialCommand(
            command_type=SpatialCommandType.IDLE,
            interaction_state=state.value,
            cursor_ndc=(curr_x, curr_y),
            timestamp=timestamp,
        )
