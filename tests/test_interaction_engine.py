"""
Unit tests for the InteractionEngine and spatial command mapping.
"""

import numpy as np
import pytest

from src.interaction.engine import InteractionEngine
from src.interaction.mapping import SpatialCommandType
from tests.test_gestures import build_hand_state


def test_interaction_engine_idle_when_no_hands():
    engine = InteractionEngine()
    command, intent_ctx, primary_hand = engine.process_hands([], timestamp=0.0)

    assert command.command_type == SpatialCommandType.IDLE
    assert intent_ctx.state.value == "IDLE"
    assert primary_hand is None


def test_interaction_engine_observing_on_hand_present():
    engine = InteractionEngine()

    # Open palm landmarks
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.5, 0.8, 0.0]
    pts[4] = [0.32, 0.52, 0.0]; pts[8] = [0.45, 0.30, 0.0]; pts[12] = [0.50, 0.28, 0.0]; pts[16] = [0.55, 0.34, 0.0]; pts[20] = [0.60, 0.40, 0.0]
    hand = build_hand_state(pts)

    command, intent_ctx, primary = engine.process_hands([hand], timestamp=1.0)
    assert intent_ctx.state.value in ("OBSERVING", "CANDIDATE")
    assert command.command_type == SpatialCommandType.HOVER
    assert primary is not None


def test_interaction_engine_two_hands_dominant_selection():
    engine = InteractionEngine()

    # Hand 1: Resting open palm
    pts1 = np.zeros((21, 3), dtype=np.float32)
    pts1[0] = [0.2, 0.8, 0.0]
    pts1[4] = [0.15, 0.55, 0.0]; pts1[8] = [0.18, 0.35, 0.0]; pts1[12] = [0.22, 0.33, 0.0]; pts1[16] = [0.25, 0.38, 0.0]; pts1[20] = [0.28, 0.42, 0.0]
    hand1 = build_hand_state(pts1, hand_id=0)

    # Hand 2: Active fist (grab)
    pts2 = np.zeros((21, 3), dtype=np.float32)
    pts2[0] = [0.8, 0.8, 0.0]
    pts2[5] = [0.75, 0.60, 0.0]; pts2[9] = [0.80, 0.58, 0.0]; pts2[13] = [0.85, 0.60, 0.0]; pts2[17] = [0.90, 0.64, 0.0]
    pts2[4] = [0.74, 0.68, 0.0]; pts2[8] = [0.76, 0.66, 0.0]; pts2[12] = [0.80, 0.65, 0.0]; pts2[16] = [0.84, 0.67, 0.0]; pts2[20] = [0.88, 0.70, 0.0]
    hand2 = build_hand_state(pts2, hand_id=1)

    command, intent_ctx, primary = engine.process_hands([hand1, hand2], timestamp=1.0)
    assert primary is not None
    assert primary.hand_id == 1  # Hand 2 with active fist should be selected as dominant


def test_interaction_engine_two_hands_bimanual_nav():
    engine = InteractionEngine()

    # Hand 1: Open palm on left
    pts1 = np.zeros((21, 3), dtype=np.float32)
    pts1[0] = [0.3, 0.8, 0.0]
    pts1[4] = [0.22, 0.52, 0.0]; pts1[8] = [0.25, 0.30, 0.0]; pts1[12] = [0.30, 0.28, 0.0]; pts1[16] = [0.35, 0.34, 0.0]; pts1[20] = [0.40, 0.40, 0.0]
    hand1 = build_hand_state(pts1, hand_id=0)

    # Hand 2: Open palm on right
    pts2 = np.zeros((21, 3), dtype=np.float32)
    pts2[0] = [0.7, 0.8, 0.0]
    pts2[4] = [0.62, 0.52, 0.0]; pts2[8] = [0.65, 0.30, 0.0]; pts2[12] = [0.70, 0.28, 0.0]; pts2[16] = [0.75, 0.34, 0.0]; pts2[20] = [0.80, 0.40, 0.0]
    hand2 = build_hand_state(pts2, hand_id=1)

    # Step 1: Initial calibration frame
    cmd1, _, _ = engine.process_hands([hand1, hand2], timestamp=1.0)
    assert cmd1.command_type == SpatialCommandType.BIMANUAL_NAV

    # Step 2: Hands move together towards center (distance decreasing)
    pts1_together = pts1.copy()
    pts1_together[:, 0] += 0.05  # moved right towards center
    pts2_together = pts2.copy()
    pts2_together[:, 0] -= 0.05  # moved left towards center

    hand1_next = build_hand_state(pts1_together, hand_id=0)
    hand2_next = build_hand_state(pts2_together, hand_id=1)

    cmd2, _, _ = engine.process_hands([hand1_next, hand2_next], timestamp=1.1)
    assert cmd2.command_type == SpatialCommandType.BIMANUAL_NAV
    assert cmd2.delta_scale > 1.0  # Hands coming together zooms in (scale > 1.0)

