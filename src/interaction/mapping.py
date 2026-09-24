"""
Gestura v3 — Natural Interaction Mapping Engine.

Translates (IntentContext, HandState, Kinematics) → abstract SpatialCommand.

Core Principles:
  1. Primary vs Modifier Hand:
     - Dominant hand -> Gesture input (primary).
     - Secondary hand -> Spatial modifier (scale, angle rotation, translation).
     - Modifier hand NEVER independently triggers swipe gestures.
  2. Interaction Priority:
     - 1. Pinch on highlighted node (Focus Mode / fly camera into node).
     - 2. Two-hand zoom/rotation/translation.
     - 3. Swipe gestures (velocity-proportional directional slaps).
     - 4. Point hover (highlights node, no selection).
     - 5. Open palm release (steady, idle, freeze).
  3. Closed Fist & Fully Open Hand:
     - Closed fist -> WORLD_ZOOM_MAX (ease camera to farthest world distance).
     - Fully open hand -> WORLD_ZOOM_MIN (ease camera to closest world distance).
  4. Two-Hand Manipulation:
     - Rotation uses line connecting both palm centers (relative angle delta).
     - Translation uses midpoint delta of palms (move globe in space).
     - Scale uses distance change between palms.
  5. Telemetry & Confidence:
     - Exposes hand_confidence, gesture_confidence, intent_confidence, and state.
"""

from enum import Enum
from typing import Dict, Optional, Tuple
import numpy as np
from pydantic import BaseModel

from src.gestures.gesture_types import GestureType, RecognizedGesture
from src.intent.state_machine import IntentContext, InteractionState
from src.landmarks.hand_state import HandState


class SpatialCommandType(str, Enum):
    """Abstract spatial commands decoupled from 3D engine."""
    IDLE               = "IDLE"
    HOVER              = "HOVER"            # Pointing highlights node; no selection
    SELECT             = "SELECT"
    FOCUS_NODE         = "FOCUS_NODE"      # Pinch on node -> locks onto node & zooms in
    WORLD_ZOOM_MAX     = "WORLD_ZOOM_MAX"  # Closed fist -> ease to max distance
    WORLD_ZOOM_MIN     = "WORLD_ZOOM_MIN"  # Spread hand -> ease to min distance
    ROTATE_OBJECT      = "ROTATE_OBJECT"   # Grab drag or directional swipe
    SCALE_OBJECT       = "SCALE_OBJECT"
    BIMANUAL_NAV       = "BIMANUAL_NAV"    # Two-hand simultaneous rotation + scale + translation
    TRANSLATE_NODE     = "TRANSLATE_NODE"  # Pinch-drag to reposition a node
    RELEASE_OBJECT     = "RELEASE_OBJECT"  # Interaction ended, easing out / freeze
    CANCEL_INTERACTION = "CANCEL_INTERACTION"


class SpatialCommand(BaseModel):
    """
    Standardized command payload sent to the 3D visualizer.
    """
    command_type: SpatialCommandType
    interaction_state: str
    cursor_ndc: Tuple[float, float]                      # NDC [-1, 1]

    delta_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (yaw, pitch, roll) in radians
    delta_scale: float = 1.0                                       # multiplicative scale factor
    delta_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0) # (dx, dy, dz) normalized scene

    target_node_id: Optional[str] = None
    handedness: str = "Right"
    dominant_hand: Optional[str] = None
    modifier_hand: Optional[str] = None
    candidate_gesture: Optional[str] = None
    is_pinch_active: bool = False
    timestamp: float = 0.0

    # Gestura v3 Confidence Model Schema
    hand_confidence: float = 1.0
    gesture_confidence: float = 0.0
    intent_confidence: float = 0.0
    state: str = "IDLE"

    # Backward compatibility fields
    confidence: float = 1.0
    confidence_hand: float = 1.0
    confidence_gesture: float = 0.0
    confidence_intent: float = 0.0
    confidence_tracking: float = 1.0


class InteractionMapper:
    """
    Maps current perceptual context to high-level Gestura v3 SpatialCommands.
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
        self.prev_bimanual_midpoint: Optional[Tuple[float, float]] = None
        self.prev_bimanual_dist: Optional[float] = None
        self.prev_bimanual_angle: Optional[float] = None
        self._smoothed_bimanual_angle_delta: float = 0.0

    def map_to_command(
        self,
        intent_ctx: IntentContext,
        primary_hand: Optional[HandState],
        bimanual_gesture: Optional[RecognizedGesture],
        smoothed_cursor_ndc: Tuple[float, float],
        timestamp: float,
        secondary_hand: Optional[HandState] = None,
    ) -> SpatialCommand:
        """Translate perceptual context to a Gestura v3 SpatialCommand."""
        state = intent_ctx.state
        active_g = intent_ctx.active_gesture

        # Hand roles
        dom_hand_str = primary_hand.handedness if primary_hand else None
        mod_hand_str = secondary_hand.handedness if secondary_hand else None
        cand_str = intent_ctx.candidate_gesture.value if hasattr(intent_ctx, "candidate_gesture") else "NONE"

        # Telemetry bundle
        hand_conf = primary_hand.detection_confidence if primary_hand else 0.0
        telem = dict(
            hand_confidence=float(hand_conf),
            gesture_confidence=float(intent_ctx.gesture_confidence),
            intent_confidence=float(intent_ctx.intent_confidence),
            state=state.value,
            dominant_hand=dom_hand_str,
            modifier_hand=mod_hand_str,
            candidate_gesture=cand_str,
            confidence_hand=float(hand_conf),
            confidence_gesture=float(intent_ctx.gesture_confidence),
            confidence_intent=float(intent_ctx.intent_confidence),
            confidence_tracking=float(intent_ctx.tracking_confidence),
            confidence=float(intent_ctx.intent_confidence),
        )

        # ── IDLE / no hand → zero command ─────────────────────────────────
        if state == InteractionState.IDLE or primary_hand is None:
            self._reset_bimanual()
            self.prev_cursor_ndc = None
            return SpatialCommand(
                command_type=SpatialCommandType.IDLE,
                interaction_state=state.value,
                cursor_ndc=(0.0, 0.0),
                timestamp=timestamp,
                **telem,
            )

        curr_x, curr_y = smoothed_cursor_ndc

        # ── Cursor displacement (dead-zone filtered) ──────────────────────
        dx, dy = 0.0, 0.0
        if self.prev_cursor_ndc is not None:
            raw_dx = curr_x - self.prev_cursor_ndc[0]
            raw_dy = curr_y - self.prev_cursor_ndc[1]
            if (raw_dx**2 + raw_dy**2) ** 0.5 > self.dead_zone_radius:
                dx = raw_dx
                dy = raw_dy
        self.prev_cursor_ndc = (curr_x, curr_y)

        # ── PRIORITY 1: Pinch on highlighted node / Focus Mode ─────────────
        # If in active pinch, trigger Focus Mode or translate node
        if active_g == GestureType.PINCH and state in (
            InteractionState.CONFIRMED, InteractionState.ACTIVE
        ):
            return SpatialCommand(
                command_type=SpatialCommandType.FOCUS_NODE,
                interaction_state=state.value,
                cursor_ndc=(curr_x, curr_y),
                delta_translation=(dx * self.translation_sensitivity, dy * self.translation_sensitivity, 0.0),
                handedness=primary_hand.handedness,
                is_pinch_active=True,
                timestamp=timestamp,
                **telem,
            )

        # ── PRIORITY 2: Two-Hand World Manipulation ────────────────────────
        if secondary_hand is not None:
            return self._bimanual_nav(
                primary_hand, secondary_hand, bimanual_gesture, state, curr_x, curr_y, timestamp, telem
            )
        else:
            self._reset_bimanual()

        # ── PRIORITY 3: Swipe Gestures (Directional Slaps) ─────────────────
        if active_g in (
            GestureType.SWIPE_LEFT, GestureType.SWIPE_RIGHT,
            GestureType.SWIPE_UP,   GestureType.SWIPE_DOWN,
        ):
            return self._swipe_command(
                active_g, primary_hand, state, curr_x, curr_y, timestamp, telem
            )

        # ── Single-Hand World Zoom: Closed Fist (Max) & Spread Hand (Min) ──
        if active_g == GestureType.GRAB and state in (
            InteractionState.CONFIRMED, InteractionState.ACTIVE
        ):
            # If user drags fist, provide delta_rotation and delta_translation (for node holding)
            drag_dist = (dx**2 + dy**2) ** 0.5
            trans_delta = (dx * self.translation_sensitivity, dy * self.translation_sensitivity, 0.0)
            if drag_dist > 0.015:
                return SpatialCommand(
                    command_type=SpatialCommandType.ROTATE_OBJECT,
                    interaction_state=state.value,
                    cursor_ndc=(curr_x, curr_y),
                    delta_rotation=(dx * self.rotation_sensitivity, -dy * self.rotation_sensitivity, 0.0),
                    delta_translation=trans_delta,
                    handedness=primary_hand.handedness,
                    timestamp=timestamp,
                    **telem,
                )
            else:
                return SpatialCommand(
                    command_type=SpatialCommandType.WORLD_ZOOM_MAX,
                    interaction_state=state.value,
                    cursor_ndc=(curr_x, curr_y),
                    delta_translation=trans_delta,
                    handedness=primary_hand.handedness,
                    timestamp=timestamp,
                    **telem,
                )

        if active_g == GestureType.SPREAD_FINGERS and state in (
            InteractionState.CONFIRMED, InteractionState.ACTIVE
        ):
            return SpatialCommand(
                command_type=SpatialCommandType.WORLD_ZOOM_MIN,
                interaction_state=state.value,
                cursor_ndc=(curr_x, curr_y),
                handedness=primary_hand.handedness,
                timestamp=timestamp,
                **telem,
            )

        # ── PRIORITY 4: Point Hover (Highlight node, no selection) ─────────
        if active_g == GestureType.POINT or state in (
            InteractionState.OBSERVING, InteractionState.CANDIDATE
        ):
            return SpatialCommand(
                command_type=SpatialCommandType.HOVER,
                interaction_state=state.value,
                cursor_ndc=(curr_x, curr_y),
                handedness=primary_hand.handedness,
                is_pinch_active=False,
                timestamp=timestamp,
                **telem,
            )

        # ── PRIORITY 5: Open Palm Release ──────────────────────────────────
        if active_g == GestureType.OPEN_PALM or state == InteractionState.RELEASE:
            return SpatialCommand(
                command_type=SpatialCommandType.RELEASE_OBJECT,
                interaction_state=state.value,
                cursor_ndc=(curr_x, curr_y),
                timestamp=timestamp,
                **telem,
            )

        # ── Default Fallback ───────────────────────────────────────────────
        return SpatialCommand(
            command_type=SpatialCommandType.IDLE,
            interaction_state=state.value,
            cursor_ndc=(curr_x, curr_y),
            timestamp=timestamp,
            **telem,
        )

    def _reset_bimanual(self):
        self.prev_bimanual_midpoint = None
        self.prev_bimanual_dist = None
        self.prev_bimanual_angle = None
        self._smoothed_bimanual_angle_delta = 0.0

    def _bimanual_nav(
        self,
        primary: HandState,
        secondary: HandState,
        bimanual_gesture: Optional[RecognizedGesture],
        state: InteractionState,
        curr_x: float,
        curr_y: float,
        timestamp: float,
        telem: dict,
    ) -> SpatialCommand:
        """
        Two-hand spatial world manipulation:
          - Line connecting Palm A and Palm B -> angle delta for rotation
          - Midpoint delta -> translation of globe in space
          - Inter-palm distance change -> scale (hands apart / together)
        """
        p1 = primary.palm_center
        p2 = secondary.palm_center

        # 1. Midpoint (translation)
        mid_x = (p1[0] + p2[0]) * 0.5
        mid_y = (p1[1] + p2[1]) * 0.5
        trans_x, trans_y = 0.0, 0.0
        if self.prev_bimanual_midpoint is not None:
            raw_mx = -(mid_x - self.prev_bimanual_midpoint[0])
            raw_my =  (mid_y - self.prev_bimanual_midpoint[1])
            if (raw_mx**2 + raw_my**2) ** 0.5 > self.dead_zone_radius:
                trans_x = raw_mx * self.translation_sensitivity * 2.0
                trans_y = raw_my * self.translation_sensitivity * 2.0
        self.prev_bimanual_midpoint = (mid_x, mid_y)

        # 2. Relative Angle along Palm A ↔ Palm B line (rotation)
        dx_palms = p2[0] - p1[0]
        dy_palms = p2[1] - p1[1]
        current_angle = float(np.arctan2(dy_palms, dx_palms))
        angle_delta = 0.0

        if self.prev_bimanual_angle is not None:
            raw_angle_delta = current_angle - self.prev_bimanual_angle
            raw_angle_delta = (raw_angle_delta + np.pi) % (2 * np.pi) - np.pi
            if abs(raw_angle_delta) > 0.005:
                # Exponential smoothing to eliminate jitter
                self._smoothed_bimanual_angle_delta = (
                    0.65 * self._smoothed_bimanual_angle_delta + 0.35 * raw_angle_delta
                )
                angle_delta = self._smoothed_bimanual_angle_delta * self.rotation_sensitivity * 1.5

        self.prev_bimanual_angle = current_angle

        # 3. Inter-palm distance (zoom)
        inter_dist = (dx_palms**2 + dy_palms**2) ** 0.5
        scale_factor = 1.0

        if self.prev_bimanual_dist is not None:
            dist_delta = inter_dist - self.prev_bimanual_dist
            if dist_delta > 0.002:
                # Moving apart -> scale > 1.0
                scale_factor = min(1.20, 1.0 + abs(dist_delta) * self.zoom_sensitivity * 5.0)
            elif dist_delta < -0.002:
                # Moving together -> scale < 1.0
                scale_factor = max(0.80, 1.0 - abs(dist_delta) * self.zoom_sensitivity * 5.0)

        self.prev_bimanual_dist = inter_dist

        # Combined yaw/pitch from translation + roll from line angle
        delta_yaw = trans_x * 1.8
        delta_pitch = trans_y * 1.8
        delta_roll = angle_delta

        return SpatialCommand(
            command_type=SpatialCommandType.BIMANUAL_NAV,
            interaction_state=state.value,
            cursor_ndc=(curr_x, curr_y),
            delta_rotation=(float(delta_yaw), float(delta_pitch), float(delta_roll)),
            delta_scale=float(scale_factor),
            delta_translation=(float(trans_x), float(trans_y), 0.0),
            handedness="Bimanual",
            timestamp=timestamp,
            **telem,
        )

    def _swipe_command(
        self,
        active_g: GestureType,
        primary: HandState,
        state: InteractionState,
        curr_x: float,
        curr_y: float,
        timestamp: float,
        telem: dict,
    ) -> SpatialCommand:
        """
        Velocity-proportional directional swipe rotation (360-degree spin).
        The swipe gesture intensity directly determines the swipe rotation intensity.
        """
        vx, vy, vz = primary.palm_velocity
        palm_speed = (vx**2 + vy**2 + vz**2) ** 0.5
        # Base 360 rotation: 2π radians (~6.28 rad), scaled proportionally to swipe velocity
        intensity = max(0.75, min(3.5, palm_speed / 0.28))
        spin_360 = (2.0 * np.pi) * intensity

        rotations = {
            GestureType.SWIPE_LEFT:  (-spin_360, 0.0, 0.0),
            GestureType.SWIPE_RIGHT: ( spin_360, 0.0, 0.0),
            GestureType.SWIPE_UP:    ( 0.0,  spin_360, 0.0),
            GestureType.SWIPE_DOWN:  ( 0.0, -spin_360, 0.0),
        }

        return SpatialCommand(
            command_type=SpatialCommandType.ROTATE_OBJECT,
            interaction_state=state.value,
            cursor_ndc=(curr_x, curr_y),
            delta_rotation=rotations.get(active_g, (0.0, 0.0, 0.0)),
            delta_translation=(0.0, 0.0, 0.0),
            handedness=primary.handedness,
            timestamp=timestamp,
            **telem,
        )
