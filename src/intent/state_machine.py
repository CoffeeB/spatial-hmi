"""
Gestura v3 — Natural Interaction State Machine.

Lifecycle:
  IDLE → OBSERVING → CANDIDATE → CONFIRMED → ACTIVE → RELEASE → IDLE

Core Principle:
  - Gesture Candidate: a possible gesture detected (accumulating evidence)
  - Intent: confidence that user truly meant it (temporal consistency confirmed)
  - Action: executed in 3D world only when CONFIRMED or ACTIVE

Graceful Degradation:
  - If confidence drops below threshold: freeze interaction and gracefully transition
    to RELEASE, never snapping unpredictably.
  - Drop guard: when hand tracking is lost, active gesture is zeroed immediately to
    prevent stale velocity from leaking, holding position during grace timeout.
"""

import time
from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel

from src.gestures.gesture_types import GestureType, RecognizedGesture
from src.intent.intent_filter import TemporalIntentFilter
from src.utils.logging_config import setup_logger

logger = setup_logger("intent_fsm_v3")


SWIPE_GESTURES = {
    GestureType.SWIPE_LEFT,
    GestureType.SWIPE_RIGHT,
    GestureType.SWIPE_UP,
    GestureType.SWIPE_DOWN,
}


class InteractionState(str, Enum):
    """Gestura v3 interaction state lifecycle."""
    IDLE       = "IDLE"        # No hand or hand at rest / dropped
    OBSERVING  = "OBSERVING"   # Hand visible, observing posture & cursor tracking only
    CANDIDATE  = "CANDIDATE"   # Possible gesture detected, temporal evidence accumulating
    CONFIRMED  = "CONFIRMED"   # Evidence threshold reached, gesture confirmed as intentional
    ACTIVE     = "ACTIVE"      # Interaction actively driving 3D manipulation
    RELEASE    = "RELEASE"     # Graceful release & cooldown before return
    RELEASING  = "RELEASE"     # Backward-compatible alias for existing tests


class IntentContext(BaseModel):
    """Immutable snapshot of intent state and telemetry."""
    state: InteractionState
    active_gesture: GestureType
    candidate_gesture: GestureType = GestureType.NONE
    intent_confidence: float        # Evidence accumulator value ∈ [0, 1]
    evidence_score: float           # Alias for intent_confidence
    consecutive_frames_in_state: int
    target_object_id: Optional[str] = None
    state_duration_sec: float
    timestamp: float

    # Confidence telemetry (Gestura v3 schema)
    hand_confidence: float = 1.0      # MediaPipe detection confidence
    gesture_confidence: float = 0.0   # Instantaneous classifier score
    tracking_confidence: float = 1.0  # Quality estimate


class InteractionStateMachine:
    """
    Finite State Machine managing temporal intent verification.

    Prevents false activations by requiring temporal consistency over multiple
    frames (via TemporalIntentFilter) before confirming actions.
    """

    def __init__(
        self,
        activation_threshold: float = 0.50,
        release_threshold: float = 0.25,
        confirm_frames: int = 4,
        candidate_frames: int = 2,
        evidence_lambda: float = 0.75,
        hand_loss_timeout_sec: float = 0.80,
    ):
        self.activation_threshold = activation_threshold
        self.release_threshold = release_threshold
        self.confirm_frames = confirm_frames
        self.candidate_frames = candidate_frames
        self.hand_loss_timeout_sec = hand_loss_timeout_sec

        self.filter = TemporalIntentFilter(evidence_lambda=evidence_lambda)
        self.state: InteractionState = InteractionState.IDLE
        self.active_gesture: GestureType = GestureType.NONE
        self.candidate_gesture: GestureType = GestureType.NONE
        self.target_object_id: Optional[str] = None

        self.consecutive_frames: int = 0
        self.state_enter_timestamp: float = time.time()
        self.last_hand_seen_timestamp: float = 0.0

        # Telemetry
        self._last_inst_conf: float = 0.0
        self._last_tracking_conf: float = 1.0

    def update(
        self,
        hand_detected: bool,
        recognized_gesture: Optional[RecognizedGesture],
        timestamp: Optional[float] = None,
    ) -> IntentContext:
        """
        Advances the FSM by one frame with temporal consistency verification.
        """
        now = timestamp if timestamp is not None else time.time()

        # ── Hand absent / lost ─────────────────────────────────────────────
        if not hand_detected or recognized_gesture is None:
            time_since_last = now - self.last_hand_seen_timestamp

            if time_since_last > self.hand_loss_timeout_sec:
                if self.state in (
                    InteractionState.ACTIVE,
                    InteractionState.CONFIRMED,
                    InteractionState.CANDIDATE,
                ):
                    self._transition_to(InteractionState.RELEASE, GestureType.NONE, now)
                elif self.state == InteractionState.RELEASE:
                    self._transition_to(InteractionState.IDLE, GestureType.NONE, now)
                else:
                    self._transition_to(InteractionState.IDLE, GestureType.NONE, now)
            else:
                # Within grace window: freeze active gesture immediately to hold steady
                self.active_gesture = GestureType.NONE

            self._last_inst_conf = 0.0
            self.consecutive_frames += 1
            return self._build_context(now, 0.0, 0.0)

        # ── Hand present ──────────────────────────────────────────────────
        self.last_hand_seen_timestamp = now
        curr_g = recognized_gesture.gesture
        curr_conf = recognized_gesture.confidence

        self._last_inst_conf = curr_conf
        self._last_tracking_conf = curr_conf

        # Update evidence accumulator
        evidence_map = self.filter.update(curr_g, curr_conf)
        current_evidence = evidence_map.get(curr_g, 0.0)

        # ── State Transitions ─────────────────────────────────────────────
        if self.state == InteractionState.IDLE:
            self._transition_to(InteractionState.OBSERVING, GestureType.NONE, now)

        elif self.state == InteractionState.OBSERVING:
            # Swipe gestures are temporally vetted by TemporalSlapDetector — trigger ACTIVE immediately
            if curr_g in SWIPE_GESTURES and curr_conf >= 0.35:
                self.candidate_gesture = curr_g
                self._transition_to(InteractionState.ACTIVE, curr_g, now)
            elif self.consecutive_frames >= 1:
                if curr_g not in (GestureType.NONE, GestureType.OPEN_PALM) and curr_conf >= 0.38:
                    self.candidate_gesture = curr_g
                    self._transition_to(InteractionState.CANDIDATE, curr_g, now)

        elif self.state == InteractionState.CANDIDATE:
            if curr_g in SWIPE_GESTURES and curr_conf >= 0.35:
                self.candidate_gesture = curr_g
                self._transition_to(InteractionState.ACTIVE, curr_g, now)
            elif curr_g == self.active_gesture and current_evidence >= self.activation_threshold:
                if self.consecutive_frames >= self.confirm_frames:
                    self._transition_to(InteractionState.CONFIRMED, curr_g, now)
            elif (
                current_evidence < self.activation_threshold * 0.28
                and curr_conf < 0.35
            ):
                # Candidate evidence collapsed -> fall back to OBSERVING
                self.candidate_gesture = GestureType.NONE
                self._transition_to(InteractionState.OBSERVING, GestureType.NONE, now)

        elif self.state == InteractionState.CONFIRMED:
            # Confirmed intent promotes directly to ACTIVE action
            self._transition_to(InteractionState.ACTIVE, self.active_gesture, now)

        elif self.state == InteractionState.ACTIVE:
            # Swipes are transient impulses: complete after 1 active frame and return to OBSERVING
            if self.active_gesture in SWIPE_GESTURES:
                self.candidate_gesture = GestureType.NONE
                self._transition_to(InteractionState.OBSERVING, GestureType.NONE, now)
            # Release conditions:
            # (a) Open palm release detected
            # (b) Confidence falls below release threshold (graceful release, never snap)
            elif curr_g == GestureType.OPEN_PALM or current_evidence < self.release_threshold:
                self.candidate_gesture = GestureType.NONE
                self._transition_to(InteractionState.RELEASE, GestureType.RELEASE, now)

        elif self.state == InteractionState.RELEASE:
            # Extended cooldown to prevent immediate jittery reactivation
            if self.consecutive_frames >= 5:
                self.candidate_gesture = GestureType.NONE
                self._transition_to(InteractionState.OBSERVING, GestureType.NONE, now)

        self.consecutive_frames += 1
        return self._build_context(now, curr_conf, current_evidence)

    def _transition_to(
        self,
        new_state: InteractionState,
        gesture: GestureType,
        timestamp: float,
    ):
        """Clean state transition with reset of frame counter."""
        if self.state != new_state:
            self.state = new_state
            self.active_gesture = gesture
            self.consecutive_frames = 0
            self.state_enter_timestamp = timestamp
            if new_state == InteractionState.IDLE:
                self.target_object_id = None
                self.candidate_gesture = GestureType.NONE
                self.filter.reset()

    def _build_context(
        self,
        timestamp: float,
        inst_conf: float,
        evidence: float,
    ) -> IntentContext:
        """Constructs an immutable telemetry snapshot."""
        duration = timestamp - self.state_enter_timestamp
        return IntentContext(
            state=self.state,
            active_gesture=self.active_gesture,
            candidate_gesture=self.candidate_gesture,
            intent_confidence=float(evidence),
            evidence_score=float(evidence),
            consecutive_frames_in_state=self.consecutive_frames,
            target_object_id=self.target_object_id,
            state_duration_sec=float(duration),
            timestamp=timestamp,
            hand_confidence=float(self._last_tracking_conf),
            gesture_confidence=float(inst_conf),
            tracking_confidence=float(self._last_tracking_conf),
        )
