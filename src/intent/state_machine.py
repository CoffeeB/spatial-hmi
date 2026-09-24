"""
Multi-stage temporal intent interaction state machine.
"""

from enum import Enum
import time
from typing import Dict, Optional, Tuple
from pydantic import BaseModel

from src.gestures.gesture_types import GestureType, RecognizedGesture
from src.intent.intent_filter import TemporalIntentFilter
from src.utils.logging_config import setup_logger

logger = setup_logger("intent_fsm")


class InteractionState(str, Enum):
    """Six-state interaction lifecycle model."""
    IDLE = "IDLE"  # No hand detected or hand at rest
    OBSERVING = "OBSERVING"  # Hand detected and tracked; hover/cursor tracking
    CANDIDATE = "CANDIDATE"  # Gesture evidence is accumulating above initial threshold
    CONFIRMED = "CONFIRMED"  # Evidence sustained across required confirmation frames
    ACTIVE = "ACTIVE"  # Interaction command latched and continuously executing
    RELEASING = "RELEASING"  # Interaction completing/releasing, cooldown transition


class IntentContext(BaseModel):
    """Encapsulates the current intent state and telemetry."""
    state: InteractionState
    active_gesture: GestureType
    intent_confidence: float
    evidence_score: float
    consecutive_frames_in_state: int
    target_object_id: Optional[str] = None
    state_duration_sec: float
    timestamp: float


class InteractionStateMachine:
    """
    Finite State Machine managing temporal intent verification.
    Prevents false activations by enforcing evidence accumulation and state hysteresis.
    """

    def __init__(
        self,
        activation_threshold: float = 0.50,
        release_threshold: float = 0.25,
        confirm_frames: int = 2,
        candidate_frames: int = 1,
        evidence_lambda: float = 0.55,
        hand_loss_timeout_sec: float = 0.25,
    ):
        self.activation_threshold = activation_threshold
        self.release_threshold = release_threshold
        self.confirm_frames = confirm_frames
        self.candidate_frames = candidate_frames
        self.hand_loss_timeout_sec = hand_loss_timeout_sec

        self.filter = TemporalIntentFilter(evidence_lambda=evidence_lambda)
        self.state: InteractionState = InteractionState.IDLE
        self.active_gesture: GestureType = GestureType.NONE
        self.target_object_id: Optional[str] = None

        self.consecutive_frames: int = 0
        self.state_enter_timestamp: float = time.time()
        self.last_hand_seen_timestamp: float = 0.0

    def update(
        self,
        hand_detected: bool,
        recognized_gesture: Optional[RecognizedGesture],
        timestamp: Optional[float] = None,
    ) -> IntentContext:
        """
        Transitions state based on hand presence, gesture recognition, and evidence accumulation.
        """
        now = timestamp if timestamp is not None else time.time()

        if not hand_detected or recognized_gesture is None:
            # Graceful timeout if hand disappears momentarily
            if now - self.last_hand_seen_timestamp > self.hand_loss_timeout_sec:
                if self.state in (InteractionState.ACTIVE, InteractionState.CONFIRMED, InteractionState.CANDIDATE):
                    self._transition_to(InteractionState.RELEASING, GestureType.RELEASE, now)
                elif self.state == InteractionState.RELEASING:
                    self._transition_to(InteractionState.IDLE, GestureType.NONE, now)
                else:
                    self._transition_to(InteractionState.IDLE, GestureType.NONE, now)
            self.consecutive_frames += 1
            return self._build_context(now, 0.0, 0.0)

        self.last_hand_seen_timestamp = now
        curr_g = recognized_gesture.gesture
        curr_conf = recognized_gesture.confidence

        # Update leaky evidence accumulator
        evidence_map = self.filter.update(curr_g, curr_conf)
        current_evidence = evidence_map.get(curr_g, 0.0)

        # FSM TRANSITION LOGIC
        if self.state == InteractionState.IDLE:
            # Hand newly detected
            self._transition_to(InteractionState.OBSERVING, GestureType.NONE, now)

        elif self.state == InteractionState.OBSERVING:
            # Check if gesture evidence crosses candidate threshold
            if curr_g not in (GestureType.NONE, GestureType.OPEN_PALM) and curr_conf >= 0.45:
                self._transition_to(InteractionState.CANDIDATE, curr_g, now)
            elif curr_g == GestureType.OPEN_PALM:
                # Open palm maintains hovering state
                pass

        elif self.state == InteractionState.CANDIDATE:
            if curr_g == self.active_gesture and current_evidence >= self.activation_threshold:
                if self.consecutive_frames >= self.confirm_frames:
                    self._transition_to(InteractionState.CONFIRMED, curr_g, now)
            elif current_evidence < (self.activation_threshold * 0.30) and curr_conf < 0.35:
                # Evidence evaporated -> fall back to OBSERVING
                self._transition_to(InteractionState.OBSERVING, GestureType.NONE, now)

        elif self.state == InteractionState.CONFIRMED:
            # Automatic transition to ACTIVE interaction
            self._transition_to(InteractionState.ACTIVE, self.active_gesture, now)

        elif self.state == InteractionState.ACTIVE:
            # Check for release condition (e.g. open palm, release gesture, or confidence collapse)
            if curr_g == GestureType.OPEN_PALM or current_evidence < self.release_threshold:
                self._transition_to(InteractionState.RELEASING, GestureType.RELEASE, now)

        elif self.state == InteractionState.RELEASING:
            # Cooldown before returning to OBSERVING or IDLE
            if self.consecutive_frames >= 2:
                self._transition_to(InteractionState.OBSERVING, GestureType.NONE, now)

        self.consecutive_frames += 1
        return self._build_context(now, curr_conf, current_evidence)

    def _transition_to(self, new_state: InteractionState, gesture: GestureType, timestamp: float):
        """Internal helper for clean state transitions."""
        if self.state != new_state:
            # logger.debug(f"State transition: {self.state} -> {new_state} [Gesture: {gesture}]")
            self.state = new_state
            self.active_gesture = gesture
            self.consecutive_frames = 0
            self.state_enter_timestamp = timestamp
            if new_state == InteractionState.IDLE:
                self.target_object_id = None
                self.filter.reset()

    def _build_context(self, timestamp: float, inst_conf: float, evidence: float) -> IntentContext:
        """Constructs an immutable snapshot of intent telemetry."""
        duration = timestamp - self.state_enter_timestamp
        return IntentContext(
            state=self.state,
            active_gesture=self.active_gesture,
            intent_confidence=float(evidence),
            evidence_score=float(evidence),
            consecutive_frames_in_state=self.consecutive_frames,
            target_object_id=self.target_object_id,
            state_duration_sec=float(duration),
            timestamp=timestamp,
        )
