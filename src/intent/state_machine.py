"""
Gestura v3 & v4 — Natural Interaction State Machine & Temporal Intent Engine.

Lifecycle:
  IDLE → OBSERVING → CANDIDATE → CONFIRMED → ACTIVE → RELEASING → IDLE

Core Principles:
  - Interpretation requires evidence, not a single frame.
  - Candidate: accumulates temporal consensus across observation windows.
  - Confirmed: satisfies all thresholds, dispatches exactly one interaction.
  - Active: continuous 3D manipulation.
  - Releasing: graceful decay with interpolation / easing, never snapping abruptly.
  - Intent Lock: prevents mid-interaction gesture hijacking.
  - Priority Engine: Pinch > Two-Hand > Swipe > Hover > Idle.
"""

from enum import Enum
import time
from typing import Any, Dict, Optional, Set

from pydantic import BaseModel, Field

from src.gestures.gesture_types import GestureType, RecognizedGesture
from src.intent.intent_filter import TemporalIntentFilter
from src.intent.intent_lock import IntentLockSystem
from src.intent.priority_manager import GesturePriorityManager, GesturePriorityTier
from src.intent.temporal_intent_config import TemporalIntentConfig
from src.utils.logging_config import setup_logger

logger = setup_logger("intent_fsm_v3")


SWIPE_GESTURES: Set[GestureType] = {
    GestureType.SWIPE_LEFT,
    GestureType.SWIPE_RIGHT,
    GestureType.SWIPE_UP,
    GestureType.SWIPE_DOWN,
}


class InteractionState(str, Enum):
    """Gestura interaction state lifecycle."""
    IDLE       = "IDLE"        # No hand detected or hand at rest
    OBSERVING  = "OBSERVING"   # Hand visible, collecting temporal evidence, no commands
    CANDIDATE  = "CANDIDATE"   # Possible gesture detected, temporal evidence accumulating
    CONFIRMED  = "CONFIRMED"   # Evidence threshold reached, dispatched exactly once
    ACTIVE     = "ACTIVE"      # Continuous 3D manipulation (dragging, rotating, scaling)
    RELEASING  = "RELEASING"   # Graceful decay with easing interpolation, never snaps abruptly
    RELEASE    = "RELEASING"   # Backward-compatible alias for existing test suites


class IntentContext(BaseModel):
    """Immutable snapshot of temporal intent state and telemetry."""
    state: InteractionState
    active_gesture: GestureType
    candidate_gesture: GestureType = GestureType.NONE
    intent_confidence: float        # Evidence accumulator value ∈ [0, 1]
    evidence_score: float           # Alias for intent_confidence
    consecutive_frames_in_state: int
    target_object_id: Optional[str] = None
    state_duration_sec: float
    timestamp: float

    # Confidence telemetry
    hand_confidence: float = 1.0      # MediaPipe detection confidence
    gesture_confidence: float = 0.0   # Instantaneous classifier score
    tracking_confidence: float = 1.0  # Quality estimate

    # Easing and Locking telemetry
    easing_factor: float = 1.0        # Smooth easing multiplier for camera/object interpolation [0, 1]
    is_locked: bool = False           # Whether intent lock is actively engaged
    locked_by: Optional[str] = None   # Gesture family holding the lock
    priority_tier: Optional[str] = None # Current active priority tier


class InteractionStateMachine:
    """
    Finite State Machine managing temporal intent verification.
    Requires temporal consistency over multiple frames before confirming actions.
    Integrates Intent Lock and Priority Management.
    """

    def __init__(
        self,
        activation_threshold: float = 0.50,
        release_threshold: float = 0.25,
        confirm_frames: int = 4,
        candidate_frames: int = 2,
        evidence_lambda: float = 0.75,
        hand_loss_timeout_sec: float = 0.80,
        config: Optional[TemporalIntentConfig] = None,
    ):
        self.config = config or TemporalIntentConfig()

        # Allow override from explicit constructor args for backward compatibility
        self.activation_threshold = activation_threshold
        self.release_threshold = release_threshold
        self.confirm_frames = confirm_frames
        self.candidate_frames = candidate_frames
        self.hand_loss_timeout_sec = hand_loss_timeout_sec
        self.evidence_lambda = evidence_lambda

        self.filter = TemporalIntentFilter(evidence_lambda=self.evidence_lambda)
        self.lock_system = IntentLockSystem(self.config)
        self.priority_manager = GesturePriorityManager(self.config)

        self.state: InteractionState = InteractionState.IDLE
        self.active_gesture: GestureType = GestureType.NONE
        self.candidate_gesture: GestureType = GestureType.NONE
        self.target_object_id: Optional[str] = None

        self.consecutive_frames: int = 0
        self.state_enter_timestamp: float = time.time()
        self.last_hand_seen_timestamp: float = 0.0

        # Easing decay during RELEASING
        self._easing_factor: float = 1.0

        # Telemetry
        self._last_inst_conf: float = 0.0
        self._last_tracking_conf: float = 1.0

    def reset(self):
        """Resets the state machine back to IDLE."""
        self.state = InteractionState.IDLE
        self.active_gesture = GestureType.NONE
        self.candidate_gesture = GestureType.NONE
        self.target_object_id = None
        self.consecutive_frames = 0
        self.state_enter_timestamp = time.time()
        self.last_hand_seen_timestamp = 0.0
        self._easing_factor = 1.0
        self.filter.reset()
        self.lock_system.reset()
        self.priority_manager.reset()

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

        if hand_detected and recognized_gesture is None:
            recognized_gesture = RecognizedGesture(
                gesture=GestureType.NONE,
                confidence=0.0,
                hand_id=0,
                handedness="Right",
                timestamp=now,
            )

        # ── 1. Hand absent / lost ─────────────────────────────────────────
        if not hand_detected or recognized_gesture is None:
            time_since_last = now - self.last_hand_seen_timestamp

            if time_since_last > self.hand_loss_timeout_sec:
                if self.state in (
                    InteractionState.ACTIVE,
                    InteractionState.CONFIRMED,
                    InteractionState.CANDIDATE,
                ):
                    self._transition_to(InteractionState.RELEASING, GestureType.NONE, now)
                elif self.state == InteractionState.RELEASING:
                    self._transition_to(InteractionState.IDLE, GestureType.NONE, now)
                else:
                    self._transition_to(InteractionState.IDLE, GestureType.NONE, now)
            else:
                # Within grace window: freeze active gesture to hold steady during repositioning
                self.active_gesture = GestureType.NONE

            # During RELEASING, smoothly ease decay
            if self.state == InteractionState.RELEASING:
                self._easing_factor *= self.config.intent_fsm.easing_decay_rate

            self._last_inst_conf = 0.0
            self.consecutive_frames += 1
            return self._build_context(now, 0.0, 0.0)

        # ── 2. Hand present ───────────────────────────────────────────────
        self.last_hand_seen_timestamp = now
        curr_g = recognized_gesture.gesture
        curr_conf = recognized_gesture.confidence

        self._last_inst_conf = curr_conf
        self._last_tracking_conf = curr_conf

        # ── 3. Intent Lock & Priority Filtering ───────────────────────────
        # Check if gesture is allowed under active intent lock
        if self.lock_system.is_locked:
            if not self.lock_system.is_gesture_allowed(curr_g.value):
                # Suppressed by lock: force gesture to NONE
                curr_g = GestureType.NONE
                curr_conf = 0.0

        # Arbitrate priority against current state
        is_active = self.state == InteractionState.ACTIVE
        is_allowed, _, _ = self.priority_manager.arbitrate(curr_g.value, current_state_is_active=is_active)
        if not is_allowed:
            curr_g = GestureType.NONE
            curr_conf = 0.0

        # Update evidence accumulator
        evidence_map = self.filter.update(curr_g, curr_conf)
        current_evidence = evidence_map.get(curr_g, 0.0)

        # ── 4. State Transitions (IDLE -> OBSERVING -> CANDIDATE -> CONFIRMED -> ACTIVE -> RELEASING -> IDLE)
        if self.state == InteractionState.IDLE:
            self._transition_to(InteractionState.OBSERVING, GestureType.NONE, now)

        elif self.state == InteractionState.OBSERVING:
            self._easing_factor = 1.0
            # Swipes are temporally vetted by sequence trackers — trigger ACTIVE
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
            elif curr_g == self.candidate_gesture and current_evidence >= self.activation_threshold:
                if self.consecutive_frames >= self.confirm_frames:
                    # Promotes to CONFIRMED
                    self._transition_to(InteractionState.CONFIRMED, curr_g, now)
            elif (
                current_evidence < self.activation_threshold * 0.28
                and curr_conf < 0.35
            ):
                # Candidate evidence collapsed -> fall back to OBSERVING
                self.candidate_gesture = GestureType.NONE
                self._transition_to(InteractionState.OBSERVING, GestureType.NONE, now)

        elif self.state == InteractionState.CONFIRMED:
            # Confirmed intent dispatches interaction and promotes to ACTIVE manipulation
            # Acquire lock on Pinch / Continuous Drag
            if curr_g in (GestureType.PINCH, GestureType.GRAB) and self.config.lock.lock_on_pinch:
                self.lock_system.acquire_lock(curr_g.value, self.target_object_id, now)

            self._transition_to(InteractionState.ACTIVE, self.active_gesture, now)

        elif self.state == InteractionState.ACTIVE:
            self._easing_factor = 1.0
            # Swipes are transient impulses: complete after active frame and return to OBSERVING
            if self.active_gesture in SWIPE_GESTURES:
                self.candidate_gesture = GestureType.NONE
                self.lock_system.release_lock(now)
                self.priority_manager.release_active()
                self._transition_to(InteractionState.OBSERVING, GestureType.NONE, now)
            # Release conditions:
            # (a) Open palm release detected
            # (b) Confidence falls below release threshold (graceful release with easing)
            elif curr_g == GestureType.OPEN_PALM or current_evidence < self.release_threshold:
                self.candidate_gesture = GestureType.NONE
                self.lock_system.release_lock(now)
                self.priority_manager.release_active()
                self._transition_to(InteractionState.RELEASING, GestureType.RELEASE, now)

        elif self.state == InteractionState.RELEASING:
            # Graceful release with exponential easing decay
            self._easing_factor *= self.config.intent_fsm.easing_decay_rate

            if self.consecutive_frames >= self.config.intent_fsm.release_cooldown_frames:
                self.candidate_gesture = GestureType.NONE
                self._easing_factor = 0.0
                self._transition_to(InteractionState.OBSERVING, GestureType.NONE, now)

        self.consecutive_frames += 1
        return self._build_context(now, curr_conf, current_evidence)

    def _transition_to(
        self,
        new_state: InteractionState,
        gesture: GestureType,
        timestamp: float,
    ):
        """Clean state transition with reset of frame counter and easing setup."""
        if self.state != new_state:
            self.state = new_state
            self.active_gesture = gesture
            self.consecutive_frames = 0
            self.state_enter_timestamp = timestamp

            if new_state == InteractionState.ACTIVE:
                self._easing_factor = 1.0
            elif new_state == InteractionState.RELEASING:
                self._easing_factor = 1.0
            elif new_state == InteractionState.IDLE:
                self.target_object_id = None
                self.candidate_gesture = GestureType.NONE
                self._easing_factor = 0.0
                self.filter.reset()
                self.lock_system.reset()
                self.priority_manager.reset()

    def _build_context(
        self,
        timestamp: float,
        inst_conf: float,
        evidence: float,
    ) -> IntentContext:
        """Constructs an immutable telemetry snapshot."""
        duration = timestamp - self.state_enter_timestamp
        lock_tel = self.lock_system.get_telemetry()
        priority_tel = self.priority_manager.get_telemetry()

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
            easing_factor=float(self._easing_factor),
            is_locked=bool(lock_tel["is_locked"]),
            locked_by=lock_tel["locked_by"] if lock_tel["is_locked"] else None,
            priority_tier=priority_tel["active_tier"],
        )
