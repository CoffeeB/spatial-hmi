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
