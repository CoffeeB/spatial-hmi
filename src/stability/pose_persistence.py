"""
Gestura Stability Engine — Pose Persistence Manager.

Implements temporal pose persistence with asymmetric hysteresis:
  - enter_threshold (0.88): Entering a new pose requires sustained high confidence across `pose_window`.
  - exit_threshold (0.62): Leaving an active stable pose requires confidence to drop significantly.
  - An isolated uncertain or flicker frame never collapses an established pose.
  - Enforces the lifecycle: STABLE -> POSSIBLE_CHANGE -> OBSERVING -> CONFIRMED_CHANGE -> NEW_STABLE.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from src.gestures.hand_pose import DerivedHandPose, HandPoseId
from src.stability.stability_config import StabilityConfig
from src.stability.stability_lifecycle import PersistentPose, StabilityLifecycleState


class PosePersistenceManager:
    """
    Manages temporal persistence and hysteresis for Level 1 static hand poses.
    """

    def __init__(self, config: Optional[StabilityConfig] = None):
        self.config = config or StabilityConfig()
        self._persistent = PersistentPose()
        self._history: deque = deque(maxlen=max(10, self.config.pose_window * 3))
        self._last_timestamp: float = 0.0

    def update(
        self,
        instantaneous_pose: Optional[DerivedHandPose],
        timestamp: float,
    ) -> Tuple[DerivedHandPose, PersistentPose]:
        """
        Evaluates the instantaneous pose through hysteresis and rolling observation buffers:
          - Prevents rapid oscillation between similar poses.
          - Rejects single-frame dropouts or uncertain classifications.
          - Emits stabilized DerivedHandPose and persistent pose metadata.
        """
        dt = max(0.001, timestamp - self._last_timestamp) if self._last_timestamp > 0 else 0.033
        self._last_timestamp = timestamp

        raw_id_str = instantaneous_pose.pose_id.value if (instantaneous_pose and hasattr(instantaneous_pose.pose_id, "value")) else "UNKNOWN"
        raw_name = instantaneous_pose.canonical_name if instantaneous_pose else "NONE"
        raw_conf = float(getattr(instantaneous_pose, "confidence", 0.0)) if instantaneous_pose else 0.0

        p = self._persistent
        self._history.append((raw_name, raw_conf))

        # Initial frame
        if p.age == 0 or p.canonical_name in ("NONE", "UNKNOWN"):
            if raw_conf >= self.config.enter_threshold and raw_name not in ("NONE", "UNKNOWN"):
                p.pose_id = raw_id_str
                p.canonical_name = raw_name
                p.previous_pose = raw_name
                p.confidence = raw_conf
                p.stability = 0.85
                p.age = 1
                p.last_changed = timestamp
                p.lifecycle = StabilityLifecycleState.STABLE
                p.candidate_pose = None
                p.candidate_frames = 0
                p.observation_timer_ms = 0.0
            return self._build_derived_pose(instantaneous_pose, is_confirmed=(p.age > 0)), p

        # ── Case A: Incoming pose matches current confirmed pose ───────────
        if raw_name == p.canonical_name:
            p.age += 1
            p.observation_timer_ms = 0.0
            # Blend confidence smoothly
            p.confidence = min(1.0, p.confidence * 0.80 + raw_conf * 0.20)
            # Stability grows with sustained hold
            age_bonus = min(0.25, p.age * 0.02)
            p.stability = min(1.0, 0.75 + age_bonus)

            if p.lifecycle in (StabilityLifecycleState.POSSIBLE_CHANGE, StabilityLifecycleState.OBSERVING):
                # User returned to current confirmed pose before completing candidate observation
                p.lifecycle = StabilityLifecycleState.STABLE
                p.candidate_pose = None
                p.candidate_frames = 0
            elif p.lifecycle == StabilityLifecycleState.NEW_STABLE:
                p.lifecycle = StabilityLifecycleState.STABLE

        # ── Case B: Incoming frame is uncertain or dropped ──────────────────
        elif raw_name in ("NONE", "UNKNOWN") or raw_conf < 0.40:
            # Single-frame dropout / occlusion: do NOT collapse pose!
            # Decay confidence gradually according to configured decay rate
            p.confidence *= self.config.confidence_decay_rate
            p.age += 1
            p.observation_timer_ms += dt * 1000.0

            if p.confidence < self.config.exit_threshold:
                # Confidence has decayed below exit threshold across multiple dropped frames
                p.previous_pose = p.canonical_name
                p.canonical_name = "NONE"
                p.pose_id = "UNKNOWN"
                p.confidence = 0.0
                p.stability = 0.0
                p.age = 0
                p.last_changed = timestamp
                p.lifecycle = StabilityLifecycleState.STABLE
                p.candidate_pose = None
                p.candidate_frames = 0
            else:
                # Still within hysteresis grace window: preserve pose!
                p.stability = max(0.30, p.stability - 0.10)

        # ── Case C: Incoming pose is a different candidate pose ─────────────
        else:
            p.observation_timer_ms += dt * 1000.0

            if raw_conf >= self.config.enter_threshold:
                # High-confidence candidate
                if p.candidate_pose != raw_name:
                    p.candidate_pose = raw_name
                    p.candidate_frames = 1
                    p.lifecycle = StabilityLifecycleState.POSSIBLE_CHANGE
                    p.stability = max(0.40, p.stability - 0.15)
                else:
                    p.candidate_frames += 1
                    p.lifecycle = StabilityLifecycleState.OBSERVING

                    if p.candidate_frames >= self.config.pose_window:
                        # Candidate has met enter_threshold for pose_window frames!
                        p.previous_pose = p.canonical_name
                        p.pose_id = raw_id_str
                        p.canonical_name = raw_name
                        p.confidence = raw_conf
                        p.age = 1
                        p.last_changed = timestamp
                        p.lifecycle = StabilityLifecycleState.CONFIRMED_CHANGE
                        p.stability = 0.90
                        p.candidate_pose = None
                        p.candidate_frames = 0
                        p.observation_timer_ms = 0.0
            else:
                # Candidate has low/marginal confidence: does not exceed enter_threshold!
                # Keep current pose stable, slightly penalize stability
                p.age += 1
                p.stability = max(0.50, p.stability - 0.05)

        is_confirmed = (
            p.canonical_name not in ("NONE", "UNKNOWN")
            and p.confidence >= self.config.exit_threshold
            and p.age >= 1
        )
        return self._build_derived_pose(instantaneous_pose, is_confirmed=is_confirmed), p

    def _build_derived_pose(
        self,
        raw_pose: Optional[DerivedHandPose],
        is_confirmed: bool,
    ) -> DerivedHandPose:
        """Emits a stabilized DerivedHandPose reflecting hysteresis state."""
        p = self._persistent
        if not is_confirmed or p.canonical_name in ("NONE", "UNKNOWN"):
            return DerivedHandPose(
                pose_id=HandPoseId.UNKNOWN,
                canonical_name="NONE",
                confidence=p.confidence,
                configuration=raw_pose.configuration if raw_pose else None,
                satisfied_predicates=[],
                diagnostics=["Awaiting stable pose confirmation (enter_threshold: 0.88)"],
            )

        try:
            pose_id_enum = HandPoseId(p.pose_id)
        except (ValueError, KeyError):
            pose_id_enum = HandPoseId.UNKNOWN

        return DerivedHandPose(
            pose_id=pose_id_enum,
            canonical_name=p.canonical_name,
            confidence=p.confidence,
            configuration=raw_pose.configuration if raw_pose else None,
            satisfied_predicates=raw_pose.satisfied_predicates if raw_pose else [],
            diagnostics=[f"Stabilized across {p.age} frames (Stability: {p.stability:.2f})"],
        )

    def get_pose_telemetry(self) -> Dict[str, Any]:
        """Returns pose persistence telemetry for developer inspection."""
        p = self._persistent
        return {
            "pose": p.canonical_name,
            "previous_pose": p.previous_pose,
            "confidence": round(p.confidence, 2),
            "stability": round(p.stability, 2),
            "age": p.age,
            "lifecycle": p.lifecycle.value,
            "candidate": p.candidate_pose or "--",
            "candidate_frames": p.candidate_frames,
            "observation_timer_ms": round(p.observation_timer_ms, 1),
        }
