"""
Gestura Stability Engine — Finger State Persistence Manager.

Maintains temporal memory for every digit across frames:
  - Tracks Current State, Previous State, Confidence, Stability Score, Age, and Last Changed.
  - Implements the lifecycle: STABLE -> POSSIBLE_CHANGE -> OBSERVING -> CONFIRMED_CHANGE -> NEW_STABLE.
  - Never overwrites stable states immediately on a single low-confidence frame; decays confidence gradually.
  - Filters out finger twitches and momentary landmark anomalies.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from src.landmarks.finger_state import FingerName, FingerStateDetail, FingerStateEnum, HandFingerStates
from src.stability.stability_config import StabilityConfig
from src.stability.stability_lifecycle import PersistentFinger, StabilityLifecycleState


class FingerPersistenceManager:
    """
    Manages temporal persistence and stabilization lifecycle for all 5 digits of a hand.
    """

    DIGIT_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]

    def __init__(self, config: Optional[StabilityConfig] = None):
        self.config = config or StabilityConfig()
        self._fingers: Dict[str, PersistentFinger] = {}
        self._history: Dict[str, deque] = {d: deque(maxlen=self.config.finger_window * 3) for d in self.DIGIT_NAMES}
        self._initialize_fingers()

    def _initialize_fingers(self) -> None:
        """Initializes persistent finger states to default baseline."""
        for d in self.DIGIT_NAMES:
            self._fingers[d] = PersistentFinger(
                finger=d,
                state="uncertain",
                previous_state="uncertain",
                confidence=0.50,
                stability=0.50,
                age=0,
                last_changed=0.0,
                lifecycle=StabilityLifecycleState.STABLE,
                candidate_state=None,
                candidate_frames=0,
            )

    def update(
        self,
        instantaneous_states: Optional[HandFingerStates],
        detection_confidence: float = 1.0,
        timestamp: float = 0.0,
    ) -> Tuple[HandFingerStates, Dict[str, PersistentFinger]]:
        """
        Processes instantaneous finger states through the persistence lifecycle:
          - If incoming state matches current confirmed state: increments age, recovers confidence.
          - If incoming state differs: enters POSSIBLE_CHANGE -> OBSERVING.
          - If candidate remains consistent for `finger_window` frames: CONFIRMED_CHANGE -> NEW_STABLE.
          - If finger returns before confirmation: reverts to STABLE with zero state change.
        """
        if instantaneous_states is None:
            # Decay confidence when hand is not clearly seen
            for d in self.DIGIT_NAMES:
                pf = self._fingers[d]
                pf.confidence *= self.config.confidence_decay_rate
                pf.stability = max(0.0, pf.stability - 0.05)
            return self._build_stabilized_hand_states(instantaneous_states, timestamp), self._fingers

        digit_details: Dict[str, Optional[FingerStateDetail]] = {
            "Thumb": instantaneous_states.thumb,
            "Index": instantaneous_states.index,
            "Middle": instantaneous_states.middle,
            "Ring": instantaneous_states.ring,
            "Little": instantaneous_states.little,
        }

        window_size = max(2, self.config.finger_window)

        for d in self.DIGIT_NAMES:
            detail = digit_details.get(d)
            raw_state_str = detail.state.value if (detail and hasattr(detail.state, "value")) else "uncertain"
            raw_conf = float(detail.confidence) if detail else 0.50

            pf = self._fingers[d]
            self._history[d].append(raw_state_str)

            # If this is the very first frame or uninitialized
            if pf.age == 0 or pf.state == "uncertain":
                pf.state = raw_state_str
                pf.previous_state = raw_state_str
                pf.confidence = raw_conf
                pf.stability = 0.80
                pf.age = 1
                pf.last_changed = timestamp
                pf.lifecycle = StabilityLifecycleState.STABLE
                pf.candidate_state = None
                pf.candidate_frames = 0
                continue

            # Compare raw incoming state with current confirmed state
            if raw_state_str == pf.state:
                # Consistent with confirmed state
                pf.age += 1
                # Smoothly restore confidence
                pf.confidence = min(1.0, pf.confidence * 0.80 + raw_conf * 0.20)
                # Boost stability with age (asymptotes to 1.0)
                age_bonus = min(0.30, pf.age * 0.03)
                pf.stability = min(1.0, 0.70 + age_bonus)

                if pf.lifecycle in (StabilityLifecycleState.POSSIBLE_CHANGE, StabilityLifecycleState.OBSERVING):
                    # User returned to stable state before confirming change: cancel transition!
                    pf.lifecycle = StabilityLifecycleState.STABLE
                    pf.candidate_state = None
                    pf.candidate_frames = 0
                elif pf.lifecycle == StabilityLifecycleState.NEW_STABLE:
                    pf.lifecycle = StabilityLifecycleState.STABLE

            else:
                # State differs from confirmed state!
                # Do NOT overwrite stable state immediately.
                if raw_conf < 0.40:
                    # Low-confidence flicker: decay confidence slightly and retain current stable state
                    pf.confidence *= self.config.confidence_decay_rate
                    pf.age += 1
                    continue

                if pf.candidate_state != raw_state_str:
                    # New distinct candidate begins observation
                    pf.candidate_state = raw_state_str
                    pf.candidate_frames = 1
                    pf.lifecycle = StabilityLifecycleState.POSSIBLE_CHANGE
                    # Jitter / possible change drops stability slightly
                    pf.stability = max(0.20, pf.stability - 0.15)
                else:
                    # Sustained candidate
                    pf.candidate_frames += 1
                    pf.lifecycle = StabilityLifecycleState.OBSERVING

                    if pf.candidate_frames >= window_size:
                        # Required evidence threshold satisfied! Transition confirmed.
                        pf.previous_state = pf.state
                        pf.state = raw_state_str
                        pf.confidence = raw_conf
                        pf.age = 1
                        pf.last_changed = timestamp
                        pf.lifecycle = StabilityLifecycleState.CONFIRMED_CHANGE
                        pf.stability = 0.88
                        pf.candidate_state = None
                        pf.candidate_frames = 0

        return self._build_stabilized_hand_states(instantaneous_states, timestamp), self._fingers

    def _build_stabilized_hand_states(
        self,
        raw_states: Optional[HandFingerStates],
        timestamp: float,
    ) -> HandFingerStates:
        """Constructs an updated HandFingerStates object using stabilized persistent finger values."""
        def make_detail(name: str, raw_detail: Optional[FingerStateDetail]) -> FingerStateDetail:
            pf = self._fingers[name]
            try:
                state_enum = FingerStateEnum(pf.state)
            except (ValueError, KeyError):
                state_enum = FingerStateEnum.UNKNOWN

            if raw_detail is not None:
                return FingerStateDetail(
                    finger=raw_detail.finger,
                    state=state_enum,
                    confidence=pf.confidence,
                    extension_ratio=raw_detail.extension_ratio,
                    pip_flexion_deg=raw_detail.pip_flexion_deg,
                    dip_flexion_deg=raw_detail.dip_flexion_deg,
                    mcp_flexion_deg=raw_detail.mcp_flexion_deg,
                    contact_target=raw_detail.contact_target,
                    intention=raw_detail.intention,
                    is_focal=raw_detail.is_focal,
                    focus_weight=raw_detail.focus_weight,
                    focal_role=raw_detail.focal_role,
                    diagnostics=raw_detail.diagnostics,
                )
            else:
                fname_enum = getattr(FingerName, name.upper(), FingerName.INDEX)
                return FingerStateDetail(
                    finger=fname_enum,
                    state=state_enum,
                    confidence=pf.confidence,
                )

        return HandFingerStates(
            thumb=make_detail("Thumb", raw_states.thumb if raw_states else None),
            index=make_detail("Index", raw_states.index if raw_states else None),
            middle=make_detail("Middle", raw_states.middle if raw_states else None),
            ring=make_detail("Ring", raw_states.ring if raw_states else None),
            little=make_detail("Little", raw_states.little if raw_states else None),
            timestamp=timestamp,
            palm_facing=getattr(raw_states, "palm_facing", "PALM") if raw_states else "PALM",
        )

    def get_finger_telemetry(self) -> Dict[str, Dict[str, Any]]:
        """Returns per-digit persistence telemetry for developer inspection."""
        return {
            d: {
                "finger": pf.finger,
                "state": pf.state,
                "previous_state": pf.previous_state,
                "confidence": round(pf.confidence, 2),
                "stability": round(pf.stability, 2),
                "age": pf.age,
                "lifecycle": pf.lifecycle.value,
                "candidate": pf.candidate_state or "--",
                "candidate_frames": pf.candidate_frames,
            }
            for d, pf in self._fingers.items()
        }
