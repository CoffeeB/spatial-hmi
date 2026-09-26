# Gestura Stability Engine — Architectural Specification & Engineering Guide

## Executive Overview

The **Gestura Stability Engine** serves as the authoritative sensory arbiter between raw perception (Level 0 Finger States, Level 1 Hand Poses, Level 2 Motion Primitives) and high-level intention interpretation (Level 4 Temporal Intent, Level 5 Confirmed Gestures, Command Dispatch).

Prior to the introduction of the Stability Engine, perceptual pipelines reacted immediately to frame-by-frame landmark updates. Because human hands exhibit involuntary micro-tremor, incidental repositioning twitches, and camera sensor jitter, naive frame-by-frame classification produced false positives, erratic gesture flipping, and unexpected command dispatches.

The fundamental axiom of the Stability Engine is:

> **Never change state because of one frame. Every state transition requires evidence.**

```
Camera Sensor (60 FPS)
       ↓
Level 0: Finger States (Extension, Flexion, Abduction)
       ↓
Level 1: Hand Pose (Topological Configuration H001–H018)
       ↓
Level 2: Motion Primitives (Linearity, Velocity, Direction)
       ↓
┌────────────────────────────────────────────────────────┐
│                   STABILITY ENGINE                     │
│  - Finger State Persistence & Confidence Decay         │
│  - Hand Pose Persistence & Asymmetric Hysteresis       │
│  - Motion Dead-Zone with Cubic Hermite Ramp-Out        │
│  - Micro-Adjustment Filter (Tremor / Twitch / Dropout) │
│  - Transition Detector (Continuous Flexion Morphing)   │
│  - Intentional Hold Dwell (Post-Stability Delay)       │
│  - State Locking System (Active Manipulation Shield)   │
│  - Multi-Signal Stability Score Calculator             │
└────────────────────────────────────────────────────────┘
       ↓
Level 4: Temporal Intent Consensus
       ↓
Level 5: Confirmed Gesture
       ↓
Spatial Command Dispatch
```

---

## 1. Why False Positives Occur

In natural human–computer interaction via spatial vision, false positives originate from three primary physical and computational phenomena:

### 1.1 Involuntary Physiological Hand Tremor
Human motor physiology inherently exhibits micro-tremor (typically 6–12 Hz) caused by alternating motor unit contractions in the extrinsic flexors and intrinsic lumbrical muscles. Even when a user believes their hand is completely motionless, palm and finger coordinates oscillate within a 2–6 mm radius in physical space (~8–25 pixels on camera sensors).

### 1.2 Incidental Digit Readjustments & Muscle Interdependence
Human fingers share common tendon sheaths and flexor digitorum profundus (FDP) muscle bellies. When an index finger deliberately extends to perform a pointing gesture, the middle and ring fingers naturally curl or twitch slightly due to passive biomechanical coupling. Naive classifiers detect this passive twitch as a transition to a different gesture (e.g., misclassifying `POINT` as `GUN` or `VICTORY`).

### 1.3 Vision-Based Sensor Noise & Perspective Jitter
Single-frame monocular landmark estimators (MediaPipe Hands / BlazeHand) solve an underconstrained 3D inverse problem from 2D RGB pixels. Optical phenomena such as motion blur, rolling shutter artifacts, edge defocus, partial self-occlusion (especially in edge-on knife poses or pinch postures), and fluctuating indoor lighting cause individual landmark coordinates to flicker by ±3–15% in confidence between adjacent 16.6ms frames.

Without a temporal stability layer, a single flickering landmark collapses a high-level confirmed state into `UNKNOWN` or triggers erroneous swipe and grab commands.

---

## 2. Micro-Adjustments vs Transitions vs Intentional Actions

The Stability Engine categorizes all perceptual movement into three mutually exclusive categories **before** any gesture or command logic is permitted to evaluate:

| Category | Kinetic / Postural Meaning | Downstream Command Execution? | System Reaction |
|---|---|:---:|---|
| **`MICRO_ADJUSTMENT`** | Natural physiological tremor, isolated digit twitch (< 3 frames), palm repositioning within hover dead-zone, camera landmark dropout. | ❌ **Strictly Ignored** | Preserves existing stable state; suppresses all action commands; records suppression reason in telemetry. |
| **`TRANSITION`** | User is actively moving between distinct poses (e.g. morphing from `OPEN_PALM` toward `CLOSED_FIST`). | ⏳ **Observed** | Labels state as `TRANSITION`; blocks intermediate hybrid frames from triggering rogue gestures; waits for destination pose stabilization. |
| **`INTENTIONAL`** | Completed, deliberate, and temporally stable hand posture or motion primitive with multi-frame consensus. | ✅ **Allowed** | Confirms state change; computes multivariate Stability Score; dispatches spatial commands once dwell and threshold criteria are satisfied. |

### 2.1 The Stability State Machine

Every individual finger, pose, and gesture follows this deterministic lifecycle:

```
              ┌───────────────┐
              │    STABLE     │
              └───────┬───────┘
                      │ Change detected in single frame
                      ▼
              ┌───────────────┐
              │POSSIBLE_CHANGE│
              └───────┬───────┘
                      │ Change persists across 2nd frame
                      ▼
              ┌───────────────┐
              │   OBSERVING   │
              └───────┬───────┘
                      │ Evidence satisfies observation window
                      ▼
              ┌───────────────┐
              │CONFIRMED_CHG  │
              └───────┬───────┘
                      │ State finalized
                      ▼
              ┌───────────────┐
              │  NEW_STABLE   │
              └───────────────┘
```

If a candidate change reverts at any point during `POSSIBLE_CHANGE` or `OBSERVING`, the state machine immediately aborts back to `STABLE` without ever dispatching a state transition event.

---

## 3. Stability Score Methodology

Every detected hand element receives an immutable, multivariate **Stability Score** $S \in [0.0, 1.0]$. The score combines four orthogonal signals reflecting both perceptual clarity and temporal consensus:

$$S = w_{\text{rec}} \cdot S_{\text{rec}} + w_{\text{temp}} \cdot S_{\text{temp}} + w_{\text{mot}} \cdot S_{\text{mot}} + w_{\text{pers}} \cdot S_{\text{pers}}$$

### 3.1 Weight Distribution and Default Configuration

All weights are configured via `configs/stability_config.yaml` and sum to 1.0:

| Parameter | Default Weight | Signal Source |
|---|:---:|---|
| $w_{\text{rec}}$ (**Recognition Confidence**) | **0.35** | Product of MediaPipe landmark tracking confidence and instantaneous pose classifier probability. |
| $w_{\text{temp}}$ (**Temporal Consistency**) | **0.25** | Percentage consensus of the candidate gesture across the rolling 12-frame observation buffer. |
| $w_{\text{mot}}$ (**Motion Consistency**) | **0.20** | Linearity ($L$) and directional coherence of trajectory; evaluates to 1.0 when within the palm dead zone. |
| $w_{\text{pers}}$ (**Persistence Score**) | **0.20** | Weighted combination of pose stability ($60\%$) and the mean stability of all five persistent digits ($40\%$). |

### 3.2 Command Eligibility Gating

An application command is **only eligible** for dispatch when:

$$S \ge \theta_{\text{stability}} \quad (\text{default: } 0.90)$$
$$\text{and} \quad \text{MovementCategory} == \text{INTENTIONAL}$$
$$\text{and} \quad \text{HoldDwellSatisfied} == \text{True}$$

If $S < 0.90$ or the hand is labeled `MICRO_ADJUSTMENT` or `TRANSITION`, all active commands (such as `PINCH_SELECT`, `MANIPULATE_NODE`, `SWIPE_CARD`) are suppressed to `HOVER` or `IDLE`, accompanied by explicit diagnostic suppression strings.

---

## 4. Finger State & Pose Persistence

### 4.1 Finger State Persistence
Every finger maintains an active memory structure across frames:

```json
{
  "finger": "Index",
  "state": "extended",
  "previous_state": "extended",
  "confidence": 0.96,
  "stability": 0.94,
  "age": 24,
  "last_changed": 1727378412.35
}
```

- **Confidence Decay**: When a frame delivers low landmark visibility or tracking dropout, the persistent finger does *not* immediately revert to `uncertain`. Instead, its confidence decays gradually:
  $$C_t = \max(C_{\text{instantaneous}}, C_{t-1} \cdot \lambda_{\text{decay}})$$
  where $\lambda_{\text{decay}} = 0.85$.
- **Twitch Suppression**: An isolated state change on a single digit while the other four digits remain stable is labeled a twitch and suppressed for up to 2 frames before undergoing transition.

### 4.2 Pose Persistence
Hand poses require temporal agreement over an observation buffer of 5 frames. A single frame reporting `UNKNOWN` or an erroneous pose during an established `OPEN_PALM` or `POINT` posture will not collapse the confirmed pose.

---

## 5. Asymmetric Hysteresis

To eliminate rapid border oscillation (chatter) between adjacent hand poses (e.g. `POINT` vs `OPEN_PALM`), the Stability Engine implements **dual-threshold asymmetric hysteresis**:

```
Confidence
   1.0 ──────────────────────────────────────────
       ▲
       │  [ ENTER THRESHOLD: 0.88 ]  Pose entered & confirmed
       ├──────────────────────────────────────────
       │
       │  [ HYSTERESIS BAND ]        Pose maintained
       │                             (single frame dips tolerated)
       ├──────────────────────────────────────────
       │  [ EXIT THRESHOLD: 0.62 ]   Pose exits & reverts
       ▼
   0.0 ──────────────────────────────────────────
```

- **Entering a Pose**: Requires instantaneous confidence $\ge 0.88$ sustained over the observation window.
- **Maintaining a Pose**: The pose remains active as long as confidence stays above the lower threshold of $0.62$.
- **Exiting a Pose**: Confidence must fall below $0.62$ for multiple consecutive frames before the pose is exited.

This asymmetric energy barrier ensures that hands do not flicker between states when hovering near decision boundaries.

---

## 6. Motion Dead-Zone Mathematics

Humans cannot hold their hands in free space without low-frequency spatial drift. The Stability Engine enforces a spatial dead zone centered on the palm anchor position $\mathbf{p}_{\text{anchor}} = (x_0, y_0, z_0)$.

### 6.1 Distance-Scaled Dead-Zone Radius
Because a hand further from the camera occupies fewer pixels, a fixed pixel dead zone would be too large at distance and too small up close. The effective dead-zone radius $R_{\text{eff}}$ dynamically scales with the characteristic hand scale reference $d_{\text{ref}}$ (distance from wrist to middle MCP):

$$R_{\text{eff}} = R_{\text{base}} \cdot \left(\frac{d_{\text{ref}}}{d_{\text{ref, base}}}\right) \quad (\text{where } R_{\text{base}} = 18\text{ px, } d_{\text{ref, base}} = 120\text{ px})$$

### 6.2 Smooth Hermite Ramp-Out (No Snapping)
Abrupt step functions create harsh snapping artifacts when a hand leaves the dead zone. The engine computes a smooth transition factor $\alpha \in [0.0, 1.0]$ using a cubic Hermite spline across ramp width $W_{\text{ramp}} = 10\text{ px}$:

Let distance $d = \|\mathbf{p}_t - \mathbf{p}_{\text{anchor}}\|$.
- If $d \le R_{\text{eff}}$: Hand is **Inside Dead Zone**; velocity $\mathbf{v} = \mathbf{0}$, displacement $\Delta = 0$, $\alpha = 0.0$.
- If $R_{\text{eff}} < d < R_{\text{eff}} + W_{\text{ramp}}$:
  $$u = \frac{d - R_{\text{eff}}}{W_{\text{ramp}}}, \quad \alpha = 3u^2 - 2u^3$$
  Filtered displacement: $\mathbf{d}_{\text{filtered}} = \alpha \cdot (\mathbf{p}_t - \mathbf{p}_{\text{anchor}})$.
- If $d \ge R_{\text{eff}} + W_{\text{ramp}}$: Hand is **Fully Active**; $\alpha = 1.0$, anchor position smoothly tracks the hand.

### 6.3 Active Manipulation Bypass
The dead zone applies exclusively to idle, hovering, and stationary inspection states. When the hand enters active object manipulation (e.g. grabbing a node or zooming a globe), the dead zone is bypassed to allow high-precision 1:1 manipulation.

---

## 7. State Locking System

When an interaction enters the `ACTIVE` state (e.g., node manipulation via `GRAB` or precision zooming via `PINCH`), the user's primary focus is on spatial manipulation, not maintaining a rigid finger posture. Incidental digit twitching or partial landmark occlusion must not break the active connection.

The **State Lock System** enforces asymmetric lock preservation:

1. **Lock Acquisition**: When an interaction transitions to `ACTIVE`, the engine locks the active mode:
   $$\text{Lock}(\text{gesture} = \text{"GRAB"}, \text{hand\_id} = 0)$$
2. **Lock Maintenance**: While locked, temporary drops in finger extension, brief landmark occlusions, or micro-adjustments are ignored. Downstream commands continue uninterrupted.
3. **Grace Preservation Window**: If tracking drops below the lock exit threshold ($0.55$), the engine maintains the lock for up to $6$ grace frames before releasing.
4. **Deliberate Release**: The lock is instantly and gracefully released only upon a deliberate release gesture (e.g. `OPEN_PALM` or `RELEASE`).

---

## 8. Observation Windows & Rolling Frame Buffers

Each hierarchical layer operates over a specialized rolling frame buffer tuned to the physical inertia of that layer:

| Layer | Buffer Size | Purpose |
|---|:---:|---|
| **Finger Layer** | **3 Frames** (~50 ms) | Detects single-digit flexion changes while ignoring isolated 1-frame twitches. |
| **Pose Layer** | **5 Frames** (~83 ms) | Validates topological composition and enforces enter/exit hysteresis. |
| **Motion Layer** | **8 Frames** (~133 ms) | Computes trajectory linearity, cumulative path length, and directional consensus. |
| **Gesture Layer** | **12 Frames** (~200 ms) | Establishes high-confidence intent consensus across combined pose and motion tokens. |
| **Two-Hand Layer**| **16 Frames** (~266 ms) | Synchronizes bimanual interactions (spread, pinch-zoom, dual rotation). |

All window lengths are fully configurable in `configs/stability_config.yaml`—no hardcoded constants exist in the perception pipeline.

---

## 9. Intentional Hold Dwell Times

Certain high-impact gestures require a minimum hold duration before being eligible to trigger commands. Crucially, the hold dwell timer **begins only after the pose is confirmed stable**:

| Gesture | Default Hold Duration | Justification |
|---|:---:|---|
| **Point** | **80 ms** | Rapid targeting selection; short dwell to ensure responsive cursor targeting. |
| **Open Palm** | **100 ms** | Primary neutral resting pose; prevents accidental activation during hand entry. |
| **Pinch** | **120 ms** | Precision object picking; ensures deliberate thumb-index contact. |
| **Closed Fist (Grab)** | **150 ms** | High-energy physical grab; prevents accidental triggers when curling fingers. |

---

## 10. Dedicated Configuration Schema

All parameters are externalized in `configs/stability_config.yaml`:

```yaml
stability:
  # Observation Windows (Rolling Frame Buffers)
  finger_window: 3
  pose_window: 5
  motion_window: 8
  gesture_window: 12
  two_hand_window: 16

  # Motion Dead Zone (Pixels & Ramp)
  dead_zone_px: 18.0
  dead_zone_ramp_width_px: 10.0
  dead_zone_ref_scale_px: 120.0

  # Asymmetric Hysteresis Thresholds
  enter_threshold: 0.88
  exit_threshold: 0.62

  # Intentional Hold Dwell Durations (Milliseconds)
  point_hold_ms: 80.0
  pinch_hold_ms: 120.0
  fist_hold_ms: 150.0
  open_palm_hold_ms: 100.0

  # Multi-Signal Stability Weights
  weight_recognition: 0.35
  weight_temporal: 0.25
  weight_motion: 0.20
  weight_persistence: 0.20

  # Overall Eligibility Threshold
  stability_threshold: 0.90

  # Micro-Adjustment & Tremor Suppression
  tremor_speed_threshold: 0.15
  tremor_max_displacement: 0.025
  decay_rate: 0.85

  # State Lock System
  active_preservation_frames: 6
  lock_exit_threshold: 0.55
```

---

## 11. Developer Stability Panel UI

The visualizer includes a dedicated Developer Stability Panel (`#stability-engine-overlay`) with real-time HUD telemetry:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ ● STABILITY ENGINE   [INTENTIONAL]   [UNLOCKED]   [ELIGIBLE]       SCORE: 0.95   0.8ms │
├─────────────────┬─────────────────┬──────────────────┬─────────────────┬───────────────┤
│ FINGERS (3F)    │ POSE (5F)       │ MOTION (8F)      │ LIFECYCLE       │ INTENT (12F)  │
│ THUMB  EXT 0.95 │ POINT     0.94  │ DEAD ZONE        │ [M] [T] [S] [L] │ POINT    0.95 │
│ INDEX  EXT 0.98 │ Obs: 120ms (80) │ VEL: 0.00        │ Windows:        │ Cand: POINT   │
│ MIDDLE FLX 0.94 │ Age: 14f        │ DISP: 0.000      │ F:3 P:5 M:8 G:12│ Thresh: 0.90  │
│ RING   FLX 0.92 │ Enter: 0.88     │ RADIUS: 18px     │ Trans: STABLE 0%│ Formula:      │
│ LITTLE FLX 0.90 │ Exit:  0.62     │ Status: DEADZONE │ Hold: 120/80ms  │ 0.35R + 0.25T │
├─────────────────┴─────────────────┴──────────────────┴─────────────────┴───────────────┤
│ STATUS / REJECTION: INTENT CONFIRMED: Stability score (0.95) meets threshold (0.90).   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

The panel provides clear explainability by detailing **why** an action was rejected (e.g. `REJECTED: Micro-adjustment suppressed | Isolated digit twitch on Ring | Stability score 0.78 < 0.90`).

---

## 12. Verification & Testing Methodology

The Stability Engine is verified across a 12-suite automated regression test battery (`tests/test_stability_engine.py`) covering all functional layers:

1. **`test_finger_persistence_lifecycle`**: Verifies that a single bent frame enters `POSSIBLE_CHANGE` without changing confirmed state, and returns to `STABLE` without transition.
2. **`test_finger_persistence_confirmation`**: Confirms that sustained digit flexion over 3 frames successfully transitions to `CONFIRMED_CHANGE`.
3. **`test_pose_persistence_hysteresis`**: Verifies entering at confidence $> 0.88$, sustaining state through uncertain frames ($0.70$), and only exiting when falling below $0.62$.
4. **`test_motion_deadzone_filtering`**: Validates suppression of sub-18px palm tremor and accumulation prevention.
5. **`test_motion_deadzone_active_bypass`**: Confirms dead zone is bypassed during active 3D manipulation.
6. **`test_micro_adjustment_filter`**: Validates suppression of isolated finger twitches and rapid palm tremor.
7. **`test_transition_detector`**: Validates continuous pose morphing detection and suppresses intermediate frames.
8. **`test_intentional_hold_tracker`**: Confirms timer does not start until pose stability is confirmed, and enforces gesture-specific dwell durations.
9. **`test_state_lock_system`**: Verifies that active interactions remain locked during landmark dips and only release upon deliberate release postures.
10. **`test_stability_score_formula`**: Asserts mathematical correctness of the 4-component weighted score.
11. **`test_stability_engine_end_to_end`**: Tests full pipeline processing across 5 frames with comprehensive telemetry generation.
12. **`test_interaction_engine_stability_gating`**: Validates that unconfirmed or micro-adjustment hand states are suppressed to `HOVER` with `STABILITY_SUPPRESSED` state.
