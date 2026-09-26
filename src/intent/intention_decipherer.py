"""
Multi-Level Intention Deciphering Engine for Gestura.

Deciphers human interaction intent across all 4 perception tiers:
    - Level 0: Individual Finger Intentions & Focal Salience
    - Level 1: Static Hand Pose Intentions & Intended Actions
    - Level 2: Motion Primitive Intentions & Purposefulness vs. Accidental Drift
    - Level 3: Complete Gesture & High-Level Task Intentions (with sequential prediction)

Core Philosophy:
    Every digit, posture, and kinetic stroke conveys purposeful human intent.
    Intention tracking must occur at EVERY level so the system knows what the user
    is trying to achieve and precisely which finger(s) to focus visual and physical
    interaction attention upon.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Set, Tuple


# =========================================================================
# Level 0: Finger Intention Taxonomy
# =========================================================================

class FingerIntention(str, Enum):
    """The physiological and interaction purpose behind an individual digit's posture."""
    POINTING_TARGETING = "POINTING_TARGETING"    # Extended to direct gaze, raycast, or aim
    CONTACT_OPPOSITION = "CONTACT_OPPOSITION"    # Opposing thumb in pad-to-pad pinch, touch, or grip
    SUPPORT_BASE = "SUPPORT_BASE"                # Curled/folded against palm providing stability/clear line of sight
    DYNAMIC_TRIGGER = "DYNAMIC_TRIGGER"          # Poised/hooked ready to tap, click, or pull trigger
    ISOLATED_EMPHASIS = "ISOLATED_EMPHASIS"      # Lone extended digit expressing symbolic affirmation or call
    ABDUCTED_EXPANSION = "ABDUCTED_EXPANSION"    # Splayed wide to indicate maximum scale, reset, or span
    PASSIVE_ADDUCTION = "PASSIVE_ADDUCTION"      # Curled or extended symmetrically in parallel with neighbors
    PASSIVE_RESTING = "PASSIVE_RESTING"          # Neutral relaxed resting state


class FocalRole(str, Enum):
    """The hierarchical role of the digit in the active interaction."""
    PRIMARY_ACTOR = "PRIMARY_ACTOR"              # Main driver of current interaction (e.g. index in point, thumb in thumbs-up)
    SECONDARY_ACTOR = "SECONDARY_ACTOR"          # Co-actor in dual-digit interactions (e.g. thumb in pinch, middle in peace)
    SUPPORT_BASE = "SUPPORT_BASE"                # Stabilizing digit folded into palm
    PASSIVE_RESTING = "PASSIVE_RESTING"          # Non-participating digit


# =========================================================================
# Level 1: Pose Intention Taxonomy
# =========================================================================

class PoseIntention(str, Enum):
    """The static semantic interaction intent conveyed by hand configuration."""
    TARGETING_RAYCAST = "TARGETING_RAYCAST"          # Aiming 3D cursor at spatial nodes
    SELECTION_PREPARATION = "SELECTION_PREPARATION"  # Pre-contact pinch shaping ready to grasp
    MANIPULATION_ENGAGED = "MANIPULATION_ENGAGED"    # Clutched fist holding or dragging an acquired node
    SYSTEM_CONFIRMATION = "SYSTEM_CONFIRMATION"      # Explicit affirmative command (thumbs up, affirm)
    SYSTEM_DISMISSAL = "SYSTEM_DISMISSAL"            # Explicit rejection / dismiss command (thumbs down)
    DELIMITER_RELEASE = "DELIMITER_RELEASE"          # Palm opening to terminate or drop interaction
    OBSERVATION_NEUTRAL = "OBSERVATION_NEUTRAL"      # Open resting hand in neutral attention
    NUMERIC_INPUT = "NUMERIC_INPUT"                  # Count-based selection (2F peace, 3F point, etc.)
    WORKSPACE_PANNING = "WORKSPACE_PANNING"          # Planar edge-on hand scanning or cupped boundary
    COMMUNICATION_SIGN = "COMMUNICATION_SIGN"        # Symbolic gesture (shaka, ok-ring)


# =========================================================================
# Level 2: Motion Intention Taxonomy
# =========================================================================

class MotionIntention(str, Enum):
    """The mechanical and spatial intent behind dynamic hand travel."""
    NAVIGATIONAL_STROKE = "NAVIGATIONAL_STROKE"      # High-speed ballistic swipe to slide/paginate
    OBJECT_TRANSLATION = "OBJECT_TRANSLATION"        # Smooth controlled dragging of acquired element
    ROTATIONAL_DIAL = "ROTATIONAL_DIAL"              # Circular orbital winding around a target
    APPROACH_ENGAGEMENT = "APPROACH_ENGAGEMENT"      # Moving forward into interaction volume (+depth)
    RETREAT_DISENGAGEMENT = "RETREAT_DISENGAGEMENT"  # Pulling hand away to disengage (-depth)
    STATIONARY_INSPECTION = "STATIONARY_INSPECTION"  # Holding motionless to inspect / hover
    ACCIDENTAL_REPOSITIONING = "ACCIDENTAL_REPOSITIONING"  # Arm drops, fidgeting, fatigue drift (SUPPRESSED)


# =========================================================================
# Level 3: Complete Task Intention Taxonomy
# =========================================================================

class TaskIntention(str, Enum):
    """High-level cognitive goal inferred by composing L0, L1, and L2."""
    SELECT_NODE = "SELECT_NODE"                      # Precision pick on interactive node
    DRAG_AND_DROP = "DRAG_AND_DROP"                  # Continuous spatial movement of selected node
    SWIPE_NAVIGATE = "SWIPE_NAVIGATE"                # Paginate screen / carousel / workspace
    ORBIT_CAMERA = "ORBIT_CAMERA"                    # Rotate 3D perspective around visual scene
    AIR_TAP_TRIGGER = "AIR_TAP_TRIGGER"              # Click / activate button in air
    CONFIRM_DECISION = "CONFIRM_DECISION"            # Affirmative commit of pending action
    DISMISS_VIEW = "DISMISS_VIEW"                    # Cancel / dismiss active modal or interface
    SCALE_WORKSPACE = "SCALE_WORKSPACE"              # Zoom in or out of 3D spatial field
    PASSIVE_OBSERVE = "PASSIVE_OBSERVE"              # Non-intrusive monitoring of hand presence


# =========================================================================
# 1. Level 0 Decipherer: Finger Intentions & Focal Salience
# =========================================================================

class Level0FingerIntentionDecipherer:
    """
    Deciphers the intention behind each of the 5 finger states and computes
    which finger(s) should be actively focused upon by the spatial HMI.
    """

    @classmethod
    def decipher_hand(cls, hand_states: Any) -> Dict[str, Dict[str, Any]]:
        """Convenience helper to decipher from a HandFingerStates object directly."""
        fs_dict = hand_states.as_dict() if hasattr(hand_states, "as_dict") else dict(hand_states)
        fd_dict = hand_states.as_details_dict() if hasattr(hand_states, "as_details_dict") else None
        facing = getattr(hand_states, "palm_facing", "PALM")
        return cls.decipher(fs_dict, fd_dict, facing)

    @staticmethod
    def decipher(
        finger_states_dict: Dict[str, str],
        finger_details_dict: Optional[Dict[str, Any]] = None,
        palm_facing: str = "PALM",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluates digit states and determines intention, focal status, focus weight,
        and focal role for Thumb, Index, Middle, Ring, and Little.
        Returns:
            Dict[digit_name, {
                "intention": str,
                "is_focal": bool,
                "focus_weight": float,
                "focal_role": str,
                "diagnostics": str,
            }]
        """
        details = finger_details_dict or {}
        results: Dict[str, Dict[str, Any]] = {}

        thumb_state = str(finger_states_dict.get("Thumb", "folded")).lower()
        index_state = str(finger_states_dict.get("Index", "folded")).lower()
        middle_state = str(finger_states_dict.get("Middle", "folded")).lower()
        ring_state = str(finger_states_dict.get("Ring", "folded")).lower()
        little_state = str(finger_states_dict.get("Little", "folded")).lower()

        ext_count = sum(
            1 for s in [thumb_state, index_state, middle_state, ring_state, little_state]
            if s in ("extended", "relaxing")
        )
        folded_count = sum(
            1 for s in [thumb_state, index_state, middle_state, ring_state, little_state]
            if s in ("folded", "tucked", "curved")
        )

        # Check for pinching opposition
        thumb_contact = details.get("Thumb", {}).get("contact_target") or ""
        index_contact = details.get("Index", {}).get("contact_target") or ""
        is_pinching = (
            "index" in thumb_contact.lower()
            or "thumb" in index_contact.lower()
            or thumb_state in ("pinching", "touching")
            or index_state in ("pinching", "touching")
        )

        # ── Thumb ─────────────────────────────────────────────────────────────
        if is_pinching:
            t_intent = FingerIntention.CONTACT_OPPOSITION
            t_focal = True
            t_weight = 0.95
            t_role = FocalRole.PRIMARY_ACTOR
            t_diag = "Pinch contact opposition with index pad"
        elif thumb_state == "extended" and folded_count >= 3:
            t_intent = FingerIntention.ISOLATED_EMPHASIS
            t_focal = True
            t_weight = 1.0
            t_role = FocalRole.PRIMARY_ACTOR
            t_diag = "Isolated thumb extension expressing confirmation or shaka"
        elif thumb_state in ("folded", "tucked"):
            t_intent = FingerIntention.SUPPORT_BASE
            t_focal = False
            t_weight = 0.05
            t_role = FocalRole.SUPPORT_BASE
            t_diag = "Folded across palm to clear line-of-sight for other digits"
        elif ext_count >= 4:
            t_intent = FingerIntention.ABDUCTED_EXPANSION
            t_focal = True
            t_weight = 0.70
            t_role = FocalRole.SECONDARY_ACTOR
            t_diag = "Extended as part of wide open hand configuration"
        else:
            t_intent = FingerIntention.PASSIVE_RESTING
            t_focal = False
            t_weight = 0.15
            t_role = FocalRole.PASSIVE_RESTING
            t_diag = "Neutral relaxed posture"

        results["Thumb"] = {
            "intention": t_intent.value,
            "is_focal": t_focal,
            "focus_weight": round(t_weight, 2),
            "focal_role": t_role.value,
            "diagnostics": t_diag,
        }

        # ── Index ─────────────────────────────────────────────────────────────
        if is_pinching:
            i_intent = FingerIntention.CONTACT_OPPOSITION
            i_focal = True
            i_weight = 0.95
            i_role = FocalRole.PRIMARY_ACTOR
            i_diag = "Pinch contact opposition with thumb pad"
        elif index_state == "extended" and middle_state in ("folded", "tucked", "curved") and ring_state in ("folded", "tucked"):
            i_intent = FingerIntention.POINTING_TARGETING
            i_focal = True
            i_weight = 1.0
            i_role = FocalRole.PRIMARY_ACTOR
            i_diag = "Primary targeting cursor raycast; user attention focal digit"
        elif index_state in ("curved", "hooked") and folded_count >= 2:
            i_intent = FingerIntention.DYNAMIC_TRIGGER
            i_focal = True
            i_weight = 0.85
            i_role = FocalRole.PRIMARY_ACTOR
            i_diag = "Poised to actuate in-air tap or trigger pull"
        elif index_state == "extended" and middle_state == "extended" and ring_state in ("folded", "tucked"):
            i_intent = FingerIntention.POINTING_TARGETING
            i_focal = True
            i_weight = 0.90
            i_role = FocalRole.PRIMARY_ACTOR
            i_diag = "Dual-finger coordinate targeting raycast"
        elif ext_count >= 4:
            i_intent = FingerIntention.ABDUCTED_EXPANSION
            i_focal = True
            i_weight = 0.75
            i_role = FocalRole.SECONDARY_ACTOR
            i_diag = "Planar spread digit"
        elif index_state in ("folded", "tucked"):
            i_intent = FingerIntention.SUPPORT_BASE
            i_focal = False
            i_weight = 0.05
            i_role = FocalRole.SUPPORT_BASE
            i_diag = "Clutched inside fist"
        else:
            i_intent = FingerIntention.PASSIVE_RESTING
            i_focal = False
            i_weight = 0.20
            i_role = FocalRole.PASSIVE_RESTING
            i_diag = "Resting neutral posture"

        results["Index"] = {
            "intention": i_intent.value,
            "is_focal": i_focal,
            "focus_weight": round(i_weight, 2),
            "focal_role": i_role.value,
            "diagnostics": i_diag,
        }

        # ── Middle ────────────────────────────────────────────────────────────
        if middle_state == "extended" and index_state == "extended" and ring_state in ("folded", "tucked"):
            m_intent = FingerIntention.POINTING_TARGETING
            m_focal = True
            m_weight = 0.90
            m_role = FocalRole.SECONDARY_ACTOR
            m_diag = "Co-acting in two-finger point or peace selection"
        elif middle_state == "extended" and index_state == "extended" and ring_state == "extended" and little_state in ("folded", "tucked"):
            m_intent = FingerIntention.POINTING_TARGETING
            m_focal = True
            m_weight = 0.80
            m_role = FocalRole.SECONDARY_ACTOR
            m_diag = "Co-acting in three-finger pointing group"
        elif middle_state in ("folded", "tucked"):
            m_intent = FingerIntention.SUPPORT_BASE
            m_focal = False
            m_weight = 0.05
            m_role = FocalRole.SUPPORT_BASE
            m_diag = "Folded into palm providing anatomical stabilization"
        elif ext_count >= 4:
            m_intent = FingerIntention.ABDUCTED_EXPANSION
            m_focal = True
            m_weight = 0.75
            m_role = FocalRole.SECONDARY_ACTOR
            m_diag = "Planar spread digit"
        else:
            m_intent = FingerIntention.PASSIVE_RESTING
            m_focal = False
            m_weight = 0.15
            m_role = FocalRole.PASSIVE_RESTING
            m_diag = "Passive digit"

        results["Middle"] = {
            "intention": m_intent.value,
            "is_focal": m_focal,
            "focus_weight": round(m_weight, 2),
            "focal_role": m_role.value,
            "diagnostics": m_diag,
        }

        # ── Ring ──────────────────────────────────────────────────────────────
        if ring_state == "extended" and middle_state == "extended" and index_state == "extended":
            r_intent = FingerIntention.POINTING_TARGETING
            r_focal = True
            r_weight = 0.75
            r_role = FocalRole.SECONDARY_ACTOR
            r_diag = "Co-acting in multi-finger grouping"
        elif ring_state in ("folded", "tucked"):
            r_intent = FingerIntention.SUPPORT_BASE
            r_focal = False
            r_weight = 0.05
            r_role = FocalRole.SUPPORT_BASE
            r_diag = "Folded into palm providing anatomical stabilization"
        elif ext_count >= 4:
            r_intent = FingerIntention.ABDUCTED_EXPANSION
            r_focal = True
            r_weight = 0.75
            r_role = FocalRole.SECONDARY_ACTOR
            r_diag = "Planar spread digit"
        else:
            r_intent = FingerIntention.PASSIVE_ADDUCTION
            r_focal = False
            r_weight = 0.10
            r_role = FocalRole.PASSIVE_RESTING
            r_diag = "Following adjacent digit motion passively"

        results["Ring"] = {
            "intention": r_intent.value,
            "is_focal": r_focal,
            "focus_weight": round(r_weight, 2),
            "focal_role": r_role.value,
            "diagnostics": r_diag,
        }

        # ── Little ────────────────────────────────────────────────────────────
        if little_state == "extended" and thumb_state == "extended" and index_state in ("folded", "tucked"):
            l_intent = FingerIntention.ISOLATED_EMPHASIS
            l_focal = True
            l_weight = 0.95
            l_role = FocalRole.PRIMARY_ACTOR
            l_diag = "Co-acting in Shaka / Call-me symbolic posture"
        elif little_state in ("folded", "tucked"):
            l_intent = FingerIntention.SUPPORT_BASE
            l_focal = False
            l_weight = 0.05
            l_role = FocalRole.SUPPORT_BASE
            l_diag = "Folded into palm providing anatomical stabilization"
        elif ext_count >= 4:
            l_intent = FingerIntention.ABDUCTED_EXPANSION
            l_focal = True
            l_weight = 0.75
            l_role = FocalRole.SECONDARY_ACTOR
            l_diag = "Planar spread digit"
        else:
            l_intent = FingerIntention.PASSIVE_ADDUCTION
            l_focal = False
            l_weight = 0.10
            l_role = FocalRole.PASSIVE_RESTING
            l_diag = "Following ring digit passively"

        results["Little"] = {
            "intention": l_intent.value,
            "is_focal": l_focal,
            "focus_weight": round(l_weight, 2),
            "focal_role": l_role.value,
            "diagnostics": l_diag,
        }

        return results


# =========================================================================
# 2. Level 1 Decipherer: Pose Intention & Target Action
# =========================================================================

class Level1PoseIntentionDecipherer:
    """
    Deciphers the interaction intent behind static hand poses (H001–H018)
    and maps which digits dominate the posture.
    """

    @staticmethod
    def decipher(
        pose_name: Optional[str] = None,
        focal_digits: Optional[List[str]] = None,
        confidence: float = 1.0,
        pose_id: Optional[Any] = None,
    ) -> Dict[str, Any]:
        p = str(pose_name or (pose_id.value if hasattr(pose_id, "value") else str(pose_id)) or "").upper()

        if "PINCH" in p or "OK_RING" in p:
            intention = PoseIntention.SELECTION_PREPARATION
            digits = focal_digits or ["Thumb", "Index"]
            action = "Preparing to grasp, select, or click interactive spatial element"
        elif "POINT" in p:
            intention = PoseIntention.TARGETING_RAYCAST
            digits = focal_digits or ["Index"]
            action = "Directing visual raycast or cursor targeting toward spatial coordinates"
        elif "GRAB" in p or "FIST" in p:
            intention = PoseIntention.MANIPULATION_ENGAGED
            digits = focal_digits or ["Thumb", "Index", "Middle", "Ring", "Little"]
            action = "Maintaining continuous clutch/drag on acquired object"
        elif "THUMBS_UP" in p:
            intention = PoseIntention.SYSTEM_CONFIRMATION
            digits = focal_digits or ["Thumb"]
            action = "Emitting affirmative system confirmation / committing dialog"
        elif "THUMBS_DOWN" in p:
            intention = PoseIntention.SYSTEM_DISMISSAL
            digits = focal_digits or ["Thumb"]
            action = "Emitting dismiss command / rejecting proposal"
        elif "PEACE" in p:
            intention = PoseIntention.NUMERIC_INPUT
            digits = focal_digits or ["Index", "Middle"]
            action = "Binary dual-selection or peace modal trigger"
        elif "SHAKA" in p:
            intention = PoseIntention.COMMUNICATION_SIGN
            digits = focal_digits or ["Thumb", "Little"]
            action = "Triggering auxiliary menu or quick dashboard"
        elif "SPREAD" in p:
            intention = PoseIntention.OBSERVATION_NEUTRAL
            digits = focal_digits or ["Thumb", "Index", "Middle", "Ring", "Little"]
            action = "Maximally expanding hand span / preparing bimanual scaling"
        elif "OPEN_PALM" in p:
            intention = PoseIntention.DELIMITER_RELEASE
            digits = focal_digits or ["Thumb", "Index", "Middle", "Ring", "Little"]
            action = "Neutral resting observation or dropping active interaction"
        elif "KNIFE_EDGE" in p or "CUPPED" in p:
            intention = PoseIntention.WORKSPACE_PANNING
            digits = focal_digits or ["Index", "Middle", "Ring", "Little"]
            action = "Scanning workspace plane or gathering spatial nodes"
        else:
            intention = PoseIntention.OBSERVATION_NEUTRAL
            digits = []
            action = "Neutral observation awaiting decisive posture"

        return {
            "pose_intention": intention.value,
            "intended_action": action,
            "focal_digits": digits,
            "intention_confidence": round(confidence, 2),
        }


# =========================================================================
# 3. Level 2 Decipherer: Motion Intention & Purposefulness
# =========================================================================

class Level2MotionIntentionDecipherer:
    """
    Deciphers the kinetic purpose behind hand motion, determining whether
    movement is intentional navigation, object translation, rotation, or
    merely accidental drift/fatigue repositioning.
    """

    @staticmethod
    def decipher(
        speed: float,
        displacement: float,
        linearity: float,
        direction_consistency: float,
        motion_primitive: str,
        active_pose: str = "NONE",
        duration_ms: float = 0.0,
    ) -> Dict[str, Any]:
        prim = str(motion_primitive).upper()
        pose = str(active_pose).upper()

        # Compute composite intentionality score (0.0 to 1.0)
        # Factors: speed, displacement, trajectory linearity, and directional consistency
        speed_factor = min(1.0, speed / 1.5)
        disp_factor = min(1.0, displacement / 0.12)
        lin_factor = max(0.0, min(1.0, linearity))
        cons_factor = max(0.0, min(1.0, direction_consistency))

        intentionality_score = (
            0.30 * speed_factor +
            0.30 * disp_factor +
            0.20 * lin_factor +
            0.20 * cons_factor
        )

        # Accidental repositioning filter
        if speed < 0.20 and displacement < 0.04:
            intention = MotionIntention.STATIONARY_INSPECTION
            is_purposeful = False
            diag = "Holding steady to inspect target; kinematics at rest"
        elif intentionality_score < 0.45 or (direction_consistency < 0.65 and linearity < 0.50):
            intention = MotionIntention.ACCIDENTAL_REPOSITIONING
            is_purposeful = False
            diag = "Wandering or curved drift; suppressed as accidental repositioning"
        elif "CW" in prim or "CCW" in prim or "ROTATE" in prim:
            intention = MotionIntention.ROTATIONAL_DIAL
            is_purposeful = True
            diag = "Orbital trajectory intended for rotational dialing or angle scrubbing"
        elif "TOWARD" in prim:
            intention = MotionIntention.APPROACH_ENGAGEMENT
            is_purposeful = True
            diag = "Translating toward camera; deepening spatial engagement"
        elif "AWAY" in prim:
            intention = MotionIntention.RETREAT_DISENGAGEMENT
            is_purposeful = True
            diag = "Translating away from camera; withdrawing from interaction zone"
        elif "GRAB" in pose or "PINCH" in pose:
            intention = MotionIntention.OBJECT_TRANSLATION
            is_purposeful = True
            diag = "Translating active selection through 3D coordinate space"
        elif linearity >= 0.70 and direction_consistency >= 0.80 and speed >= 1.0:
            intention = MotionIntention.NAVIGATIONAL_STROKE
            is_purposeful = True
            diag = f"High-velocity linear {prim} stroke intended for pagination / sliding"
        else:
            intention = MotionIntention.NAVIGATIONAL_STROKE
            is_purposeful = intentionality_score >= 0.50
            diag = f"Directional movement along {prim}"

        return {
            "motion_intention": intention.value,
            "intentionality_score": round(intentionality_score, 2),
            "is_purposeful": is_purposeful,
            "diagnostics": diag,
        }


# =========================================================================
# 4. Level 3 Decipherer: Complete Task Intent & Sequence Prediction
# =========================================================================

class Level3GestureIntentionDecipherer:
    """
    Synthesizes L0 (Focal Digits) + L1 (Hand Pose) + L2 (Motion) + Intent State
    into high-level user task intent and predicts anticipated next actions.
    """

    @staticmethod
    def decipher(
        gesture_name: str,
        pose_intention: str,
        motion_intention: str,
        intent_state: str,
        focal_digits: List[str],
    ) -> Dict[str, Any]:
        g = str(gesture_name).upper()
        p_int = str(pose_intention).upper()
        m_int = str(motion_intention).upper()
        st = str(intent_state).upper()

        if "PINCH" in g and "DRAG" in g:
            task = TaskIntention.DRAG_AND_DROP
            next_intent = "RELEASE_AND_PLACE"
            summary = "User is translating gripped 3D element across interaction space"
        elif "PINCH" in g:
            task = TaskIntention.SELECT_NODE
            next_intent = "DRAG_AND_DROP"
            summary = "User is pinching target element to pick up or inspect"
        elif "SWIPE" in g or m_int == MotionIntention.NAVIGATIONAL_STROKE.value:
            task = TaskIntention.SWIPE_NAVIGATE
            next_intent = "OBSERVATION_NEUTRAL"
            summary = "User is executing high-velocity directional sweep to paginate or slide view"
        elif "DIAL" in g or "ROTATE" in g or m_int == MotionIntention.ROTATIONAL_DIAL.value:
            task = TaskIntention.ORBIT_CAMERA
            next_intent = "HOLD_PERSPECTIVE"
            summary = "User is scrubbing rotation dial or orbiting 3D camera"
        elif "TAP" in g or "TRIGGER" in g:
            task = TaskIntention.AIR_TAP_TRIGGER
            next_intent = "RETURN_TO_AIM"
            summary = "User is executing in-air tap click on focused element"
        elif "THUMBS_UP" in g or p_int == PoseIntention.SYSTEM_CONFIRMATION.value:
            task = TaskIntention.CONFIRM_DECISION
            next_intent = "DELIMITER_RELEASE"
            summary = "User is affirming proposal or committing dialog state"
        elif "THUMBS_DOWN" in g or p_int == PoseIntention.SYSTEM_DISMISSAL.value:
            task = TaskIntention.DISMISS_VIEW
            next_intent = "DELIMITER_RELEASE"
            summary = "User is dismissing view or cancelling action"
        elif "ZOOM" in g or "SPREAD" in g:
            task = TaskIntention.SCALE_WORKSPACE
            next_intent = "STABILIZE_SCALE"
            summary = "User is expanding or contracting spatial workspace scale"
        elif "POINT" in g or p_int == PoseIntention.TARGETING_RAYCAST.value:
            task = TaskIntention.PASSIVE_OBSERVE
            next_intent = "SELECT_NODE"
            summary = "User is casting targeting cursor; anticipating selection click"
        else:
            task = TaskIntention.PASSIVE_OBSERVE
            next_intent = "ENGAGE_INTERACTION"
            summary = "Neutral presence; awaiting intentional interaction onset"

        focal_str = ", ".join(focal_digits) if focal_digits else "Hand"
        hierarchy = f"L0[{focal_str}] ➔ L1[{p_int}] ➔ L2[{m_int}] ➔ L3[{task.value}]"

        return {
            "task_intent": task.value,
            "predicted_next_intent": next_intent,
            "hierarchy_summary": hierarchy,
            "explanation": summary,
        }


# =========================================================================
# Unified Multi-Level Intention Engine
# =========================================================================

class MasterIntentionDecipherer:
    """
    Central orchestrator providing unified, synchronized intention analysis
    across Level 0 (Fingers), Level 1 (Pose), Level 2 (Motion), and Level 3 (Gestures).
    """

    def __init__(self):
        self.l0 = Level0FingerIntentionDecipherer()
        self.l1 = Level1PoseIntentionDecipherer()
        self.l2 = Level2MotionIntentionDecipherer()
        self.l3 = Level3GestureIntentionDecipherer()

    def process_all_levels(
        self,
        finger_states_dict: Dict[str, str],
        finger_details_dict: Optional[Dict[str, Any]],
        pose_name: str,
        motion_primitive: str,
        speed: float,
        displacement: float,
        linearity: float,
        direction_consistency: float,
        gesture_name: str,
        intent_state: str,
        palm_facing: str = "PALM",
        duration_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Processes every perception layer and produces a comprehensive multi-level
        intention breakdown for telemetry, command dispatch, and developer HUDs.
        """
        # 1. Level 0: Finger Intentions & Focus
        l0_out = self.l0.decipher(finger_states_dict, finger_details_dict, palm_facing)
        focal_digits = [d for d, info in l0_out.items() if info["is_focal"]]
        # Sort focal digits by focus weight descending
        focal_digits.sort(key=lambda d: l0_out[d]["focus_weight"], reverse=True)

        # 2. Level 1: Pose Intention
        l1_out = self.l1.decipher(pose_name, focal_digits)

        # 3. Level 2: Motion Intention & Purposefulness
        l2_out = self.l2.decipher(
            speed=speed,
            displacement=displacement,
            linearity=linearity,
            direction_consistency=direction_consistency,
            motion_primitive=motion_primitive,
            active_pose=pose_name,
            duration_ms=duration_ms,
        )

        # 4. Level 3: Complete Task Intent & Sequence Prediction
        l3_out = self.l3.decipher(
            gesture_name=gesture_name,
            pose_intention=l1_out["pose_intention"],
            motion_intention=l2_out["motion_intention"],
            intent_state=intent_state,
            focal_digits=focal_digits,
        )

        return {
            "level_0_fingers": {
                "digits": l0_out,
                "focal_digits": focal_digits,
                "primary_focal_digit": focal_digits[0] if focal_digits else "None",
            },
            "level_1_pose": l1_out,
            "level_2_motion": l2_out,
            "level_3_gesture": l3_out,
            "hierarchy_summary": l3_out["hierarchy_summary"],
        }

    @classmethod
    def decipher_full_hierarchy(
        cls,
        finger_states: Any,
        pose_id: Any = None,
        pose_name: str = "NONE",
        pose_confidence: float = 1.0,
        motion_primitive: Any = "STATIONARY",
        motion_speed: float = 0.0,
        motion_displacement: float = 0.0,
        motion_linearity: float = 1.0,
        motion_direction_consistency: float = 1.0,
        motion_duration_ms: float = 0.0,
        candidate_gesture_name: str = "NONE",
        intent_state: str = "IDLE",
        palm_facing: str = "PALM",
    ) -> Dict[str, Any]:
        """Convenience classmethod to execute full hierarchy from objects or primitives."""
        engine = cls()
        fs_dict = finger_states.as_dict() if hasattr(finger_states, "as_dict") else dict(finger_states)
        fd_dict = finger_states.as_details_dict() if hasattr(finger_states, "as_details_dict") else None
        m_prim_str = motion_primitive.value if hasattr(motion_primitive, "value") else str(motion_primitive)
        p_name = pose_name if pose_name != "NONE" else (pose_id.value if hasattr(pose_id, "value") else str(pose_id or "NONE"))
        return engine.process_all_levels(
            finger_states_dict=fs_dict,
            finger_details_dict=fd_dict,
            pose_name=p_name,
            motion_primitive=m_prim_str,
            speed=motion_speed,
            displacement=motion_displacement,
            linearity=motion_linearity,
            direction_consistency=motion_direction_consistency,
            gesture_name=candidate_gesture_name,
            intent_state=intent_state,
            palm_facing=palm_facing,
            duration_ms=motion_duration_ms,
        )
