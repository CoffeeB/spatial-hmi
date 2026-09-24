"""
Unit tests for the Interaction State Machine and temporal evidence accumulation.
"""

import pytest

from src.gestures.gesture_types import GestureType, RecognizedGesture
from src.intent.state_machine import InteractionState, InteractionStateMachine


def make_gesture(g: GestureType, conf: float = 0.90, ts: float = 1.0) -> RecognizedGesture:
    return RecognizedGesture(
        gesture=g,
        confidence=conf,
        hand_id=0,
        handedness="Right",
        timestamp=ts,
    )


def test_fsm_initial_state_idle():
    fsm = InteractionStateMachine()
    ctx = fsm.update(hand_detected=False, recognized_gesture=None, timestamp=0.0)
    assert ctx.state == InteractionState.IDLE


def test_fsm_lifecycle_transition_to_active():
    fsm = InteractionStateMachine(
        activation_threshold=0.75,
        release_threshold=0.35,
        confirm_frames=3,
        evidence_lambda=0.80,
    )

    t = 1.0

    # Frame 1: Hand observed -> IDLE to OBSERVING
    ctx1 = fsm.update(hand_detected=True, recognized_gesture=make_gesture(GestureType.NONE, 0.0, t), timestamp=t)
    assert ctx1.state == InteractionState.OBSERVING

    # Frame 2: Point gesture starts -> OBSERVING to CANDIDATE
    t += 0.033
    ctx2 = fsm.update(hand_detected=True, recognized_gesture=make_gesture(GestureType.POINT, 0.85, t), timestamp=t)
    assert ctx2.state == InteractionState.CANDIDATE

    # Frames: Evidence accumulates across leaky filter time constant -> CANDIDATE to CONFIRMED
    for _ in range(8):
        t += 0.033
        ctx = fsm.update(hand_detected=True, recognized_gesture=make_gesture(GestureType.POINT, 0.95, t), timestamp=t)

    assert ctx.state in (InteractionState.CONFIRMED, InteractionState.ACTIVE)

    # Subsequent frame latches ACTIVE
    t += 0.033
    ctx_active = fsm.update(hand_detected=True, recognized_gesture=make_gesture(GestureType.POINT, 0.90, t), timestamp=t)
    assert ctx_active.state == InteractionState.ACTIVE


def test_fsm_single_frame_spike_rejected():
    fsm = InteractionStateMachine(
        activation_threshold=0.75,
        confirm_frames=3,
        evidence_lambda=0.80,
    )

    t = 1.0
    fsm.update(hand_detected=True, recognized_gesture=make_gesture(GestureType.NONE, 0.0, t), timestamp=t)

    # Single frame spike of Pinch
    t += 0.033
    ctx_spike = fsm.update(hand_detected=True, recognized_gesture=make_gesture(GestureType.PINCH, 0.95, t), timestamp=t)
    # May transition to CANDIDATE, but must NOT transition to CONFIRMED or ACTIVE
    assert ctx_spike.state in (InteractionState.OBSERVING, InteractionState.CANDIDATE)
    assert ctx_spike.state != InteractionState.ACTIVE

    # Next frame reverts to NONE
    t += 0.033
    ctx_revert = fsm.update(hand_detected=True, recognized_gesture=make_gesture(GestureType.NONE, 0.0, t), timestamp=t)
    assert ctx_revert.state in (InteractionState.OBSERVING, InteractionState.CANDIDATE)


def test_fsm_release_transition():
    fsm = InteractionStateMachine(confirm_frames=2, evidence_lambda=0.70)
    t = 1.0

    # Fast forward to ACTIVE
    fsm.update(True, make_gesture(GestureType.NONE, 0.0, t), t)
    for _ in range(6):
        t += 0.033
        fsm.update(True, make_gesture(GestureType.GRAB, 0.95, t), t)

    # Assert latched ACTIVE
    assert fsm.state == InteractionState.ACTIVE

    # Open Palm triggered -> must transition to RELEASING
    t += 0.033
    ctx_rel = fsm.update(True, make_gesture(GestureType.OPEN_PALM, 0.90, t), t)
    assert ctx_rel.state == InteractionState.RELEASING
