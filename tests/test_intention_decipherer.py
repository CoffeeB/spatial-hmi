"""
Unit tests for Gestura Multi-Level Intention Decipherer (Levels 0, 1, 2, and 3).
Verifies:
  - Level 0: Finger Intention, Focal Digit Identification, Focus Weight, and Focal Role.
  - Level 1: Hand Pose Intention, Intended Action, and Focal Digits.
  - Level 2: Motion Intention, Intentionality Score, and Accidental Repositioning Filter.
  - Level 3: Task Intent Synthesis and Sequential Next-Intent Prediction.
  - Master Intention Decipherer: Synchronized multi-level hierarchy analysis.
"""

import pytest
from src.gestures.hand_pose import HandPoseId
from src.intent.intention_decipherer import (
    FingerIntention,
    FocalRole,
    Level0FingerIntentionDecipherer,
    Level1PoseIntentionDecipherer,
    Level2MotionIntentionDecipherer,
    Level3GestureIntentionDecipherer,
    MasterIntentionDecipherer,
    MotionIntention,
    PoseIntention,
    TaskIntention,
)
from src.landmarks.finger_state import FingerName, FingerStateDetail, FingerStateEnum, HandFingerStates
from src.motion.motion_primitive import MotionPrimitive


def _make_hand_states(states_dict):
    """Helper to construct HandFingerStates with mock states."""
    details = {}
    for name_str, state_val in states_dict.items():
        fname = getattr(FingerName, name_str.upper())
        s_enum = FingerStateEnum(state_val)
        ext_ratio = 1.0 if state_val == "extended" else 0.2
        details[name_str.lower()] = FingerStateDetail(
            finger=fname,
            state=s_enum,
            confidence=0.95,
            extension_ratio=ext_ratio,
            pip_flexion_deg=10.0 if state_val == "extended" else 85.0,
            dip_flexion_deg=5.0 if state_val == "extended" else 75.0,
            mcp_flexion_deg=10.0 if state_val == "extended" else 80.0,
            contact_target="Index" if name_str == "Thumb" and state_val == "contact" else None,
        )
    return HandFingerStates(
        thumb=details["thumb"],
        index=details["index"],
        middle=details["middle"],
        ring=details["ring"],
        little=details["little"],
    )


# =========================================================================
# 1. Level 0: Finger Intention Tests
# =========================================================================

def test_level_0_pointing_focal_digit():
    """Verify single finger point isolates Index as primary focal actor."""
    hand = _make_hand_states({
        "thumb": "folded",
        "index": "extended",
        "middle": "folded",
        "ring": "folded",
        "little": "folded",
    })

    deciphered = Level0FingerIntentionDecipherer.decipher_hand(hand)

    assert deciphered["Index"]["is_focal"] is True
    assert deciphered["Index"]["intention"] == FingerIntention.POINTING_TARGETING.value
    assert deciphered["Index"]["focus_weight"] >= 0.90
    assert deciphered["Index"]["focal_role"] == FocalRole.PRIMARY_ACTOR.value

    # Folded digits must be support base and not focal
    for non_focal in ["Middle", "Ring", "Little"]:
        assert deciphered[non_focal]["is_focal"] is False
        assert deciphered[non_focal]["focal_role"] == FocalRole.SUPPORT_BASE.value


def test_level_0_pinch_focal_digits():
    """Verify precision pinch isolates Thumb and Index in contact opposition."""
    hand = _make_hand_states({
        "thumb": "touching",
        "index": "touching",
        "middle": "folded",
        "ring": "folded",
        "little": "folded",
    })

    deciphered = Level0FingerIntentionDecipherer.decipher_hand(hand)

    assert deciphered["Thumb"]["is_focal"] is True
    assert deciphered["Thumb"]["intention"] == FingerIntention.CONTACT_OPPOSITION.value
    assert deciphered["Thumb"]["focus_weight"] >= 0.90

    assert deciphered["Index"]["is_focal"] is True
    assert deciphered["Index"]["intention"] == FingerIntention.CONTACT_OPPOSITION.value
    assert deciphered["Index"]["focus_weight"] >= 0.90


def test_level_0_thumbs_up_focal_digit():
    """Verify thumbs up isolates Thumb as primary affirmative actor."""
    hand = _make_hand_states({
        "thumb": "extended",
        "index": "folded",
        "middle": "folded",
        "ring": "folded",
        "little": "folded",
    })

    deciphered = Level0FingerIntentionDecipherer.decipher_hand(hand)

    assert deciphered["Thumb"]["is_focal"] is True
    assert deciphered["Thumb"]["intention"] == FingerIntention.ISOLATED_EMPHASIS.value
    assert deciphered["Thumb"]["focus_weight"] >= 0.95
    assert deciphered["Index"]["is_focal"] is False


# =========================================================================
# 2. Level 1: Hand Pose Intention Tests
# =========================================================================

def test_level_1_pose_intention_point():
    """Verify H004 Pointing translates to TARGETING_RAYCAST intent."""
    res = Level1PoseIntentionDecipherer.decipher(
        pose_id=HandPoseId.H004_INDEX_POINT,
        pose_name="POINT",
        focal_digits=["Index"],
        confidence=0.92,
    )
    assert res["pose_intention"] == PoseIntention.TARGETING_RAYCAST.value
    assert "raycast" in res["intended_action"].lower()
    assert res["focal_digits"] == ["Index"]


def test_level_1_pose_intention_pinch():
    """Verify H005 Precision Pinch translates to SELECTION_PREPARATION intent."""
    res = Level1PoseIntentionDecipherer.decipher(
        pose_id=HandPoseId.H005_PRECISION_PINCH,
        pose_name="PINCH",
        focal_digits=["Thumb", "Index"],
        confidence=0.95,
    )
    assert res["pose_intention"] == PoseIntention.SELECTION_PREPARATION.value
    assert "grasp" in res["intended_action"].lower() or "select" in res["intended_action"].lower()
    assert "Thumb" in res["focal_digits"] and "Index" in res["focal_digits"]


def test_level_1_pose_intention_thumbs_up():
    """Verify Thumbs Up translates to SYSTEM_CONFIRMATION intent."""
    res = Level1PoseIntentionDecipherer.decipher(
        pose_id=HandPoseId.H007_THUMBS_UP,
        pose_name="THUMBS_UP",
        focal_digits=["Thumb"],
        confidence=0.96,
    )
    assert res["pose_intention"] == PoseIntention.SYSTEM_CONFIRMATION.value
    assert "affirmative" in res["intended_action"].lower()


# =========================================================================
# 3. Level 2: Motion Intention Tests
# =========================================================================

def test_level_2_stationary_inspection():
    """Verify low speed and minimal displacement are classified as stationary inspection."""
    res = Level2MotionIntentionDecipherer.decipher(
        speed=0.05,
        displacement=0.01,
        linearity=0.95,
        direction_consistency=0.98,
        motion_primitive="STATIONARY",
        active_pose="POINT",
    )
    assert res["motion_intention"] == MotionIntention.STATIONARY_INSPECTION.value
    assert res["is_purposeful"] is False


def test_level_2_purposeful_navigation_stroke():
    """Verify high-velocity linear movement is recognized as purposeful navigation stroke."""
    res = Level2MotionIntentionDecipherer.decipher(
        speed=0.85,
        displacement=0.22,
        linearity=0.92,
        direction_consistency=0.90,
        motion_primitive="MOVE_LEFT",
        active_pose="OPEN_PALM",
    )
    assert res["motion_intention"] == MotionIntention.NAVIGATIONAL_STROKE.value
    assert res["is_purposeful"] is True
    assert res["intentionality_score"] >= 0.75


def test_level_2_accidental_repositioning_filter():
    """Verify wandering, non-linear drift is suppressed as accidental repositioning."""
    res = Level2MotionIntentionDecipherer.decipher(
        speed=0.35,
        displacement=0.05,
        linearity=0.35,
        direction_consistency=0.40,
        motion_primitive="MOVE_RIGHT",
        active_pose="NONE",
    )
    assert res["motion_intention"] == MotionIntention.ACCIDENTAL_REPOSITIONING.value
    assert res["is_purposeful"] is False


def test_level_2_rotational_dial():
    """Verify circular motion primitive is recognized as intentional dial."""
    res = Level2MotionIntentionDecipherer.decipher(
        speed=0.60,
        displacement=0.10,
        linearity=0.60,
        direction_consistency=0.75,
        motion_primitive="ROTATE_CW",
        active_pose="PINCH",
    )
    assert res["motion_intention"] == MotionIntention.ROTATIONAL_DIAL.value
    assert res["is_purposeful"] is True


# =========================================================================
# 4. Level 3: Complete Task Intent & Sequence Prediction Tests
# =========================================================================

def test_level_3_drag_and_drop_synthesis():
    """Verify pinch while dragging translates to DRAG_AND_DROP task intent."""
    res = Level3GestureIntentionDecipherer.decipher(
        gesture_name="PINCH_DRAG",
        pose_intention=PoseIntention.MANIPULATION_ENGAGED.value,
        motion_intention=MotionIntention.OBJECT_TRANSLATION.value,
        intent_state="ACTIVE",
        focal_digits=["Thumb", "Index"],
    )
    assert res["task_intent"] == TaskIntention.DRAG_AND_DROP.value
    assert res["predicted_next_intent"] == "RELEASE_AND_PLACE"


def test_level_3_swipe_navigation():
    """Verify swipe gesture maps to SWIPE_NAVIGATE with neutral next intent."""
    res = Level3GestureIntentionDecipherer.decipher(
        gesture_name="SWIPE_LEFT",
        pose_intention=PoseIntention.WORKSPACE_PANNING.value,
        motion_intention=MotionIntention.NAVIGATIONAL_STROKE.value,
        intent_state="CONFIRMED",
        focal_digits=["Index", "Middle", "Ring", "Little"],
    )
    assert res["task_intent"] == TaskIntention.SWIPE_NAVIGATE.value
    assert res["predicted_next_intent"] == "OBSERVATION_NEUTRAL"


def test_level_3_air_tap():
    """Verify tap click maps to AIR_TAP_TRIGGER predicting return to aim."""
    res = Level3GestureIntentionDecipherer.decipher(
        gesture_name="AIR_TAP",
        pose_intention=PoseIntention.TARGETING_RAYCAST.value,
        motion_intention=MotionIntention.APPROACH_ENGAGEMENT.value,
        intent_state="ACTIVE",
        focal_digits=["Index"],
    )
    assert res["task_intent"] == TaskIntention.AIR_TAP_TRIGGER.value
    assert res["predicted_next_intent"] == "RETURN_TO_AIM"


# =========================================================================
# 5. Master Orchestrator Synchronized Analysis Tests
# =========================================================================

def test_master_intention_decipherer_full_hierarchy():
    """Verify master orchestrator coordinates Levels 0–3 into a unified payload."""
    hand = _make_hand_states({
        "thumb": "folded",
        "index": "extended",
        "middle": "folded",
        "ring": "folded",
        "little": "folded",
    })

    master_result = MasterIntentionDecipherer.decipher_full_hierarchy(
        finger_states=hand,
        pose_id=HandPoseId.H004_INDEX_POINT,
        pose_name="POINT",
        pose_confidence=0.94,
        motion_primitive=MotionPrimitive.MOVE_TOWARD,
        motion_speed=0.45,
        motion_displacement=0.08,
        motion_linearity=0.88,
        motion_direction_consistency=0.91,
        motion_duration_ms=220.0,
        candidate_gesture_name="AIR_TAP",
        intent_state="ACTIVE",
    )

    assert "level_0_fingers" in master_result
    assert "level_1_pose" in master_result
    assert "level_2_motion" in master_result
    assert "level_3_gesture" in master_result

    # L0
    assert master_result["level_0_fingers"]["focal_digits"] == ["Index"]
    assert master_result["level_0_fingers"]["digits"]["Index"]["intention"] == FingerIntention.POINTING_TARGETING.value

    # L1
    assert master_result["level_1_pose"]["pose_intention"] == PoseIntention.TARGETING_RAYCAST.value

    # L2
    assert master_result["level_2_motion"]["is_purposeful"] is True
    assert master_result["level_2_motion"]["motion_intention"] == MotionIntention.APPROACH_ENGAGEMENT.value

    # L3
    assert master_result["level_3_gesture"]["task_intent"] == TaskIntention.AIR_TAP_TRIGGER.value
    assert master_result["level_3_gesture"]["predicted_next_intent"] == "RETURN_TO_AIM"
