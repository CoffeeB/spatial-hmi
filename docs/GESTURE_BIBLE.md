# Gestura Gesture Bible — Standard Spatial Interaction Specification
**Document ID:** GESTURA-SPEC-2026-V1  
**Classification:** Canonical Specification / Single Source of Truth  
**Target File:** `docs/GESTURE_BIBLE.md`  
**Status:** Approved Standard  
**Maintainer:** Lead HCI, Computer Vision, and AI Architect  

---

## Document Control & Revision History

| Version | Date | Status | Description |
| :--- | :--- | :--- | :--- |
| **1.0.0** | 2026-09-24 | Approved | Initial canonical release of the Gestura Gesture Bible. Full specification across Levels 0–4, Micro-Gestures, Spatial Interaction Principles, Context Mappings, and Multimodal Voice Integration. |

**Companion Living Documents:**
- 📊 **[GESTURE_LOGBOOK.md](file:///Users/mac/Desktop/gebashmi/docs/GESTURE_LOGBOOK.md)** — Empirical registry of all gestures tested, benchmarks, accuracy metrics, and failure post-mortems.
- 🔬 **[RESEARCH_JOURNAL.md](file:///Users/mac/Desktop/gebashmi/docs/RESEARCH_JOURNAL.md)** — Chronological record of daily discoveries, mathematical formulations, failures, and future ideas.

---

# Part I: Research Documentation & Interaction Architecture

## 1. Abstract

Spatial Human–Machine Interfaces (Spatial HMIs) transcend physical peripherals by turning the human body into an expressive interaction device. However, optical gesture systems frequently fail due to jitter, tracking loss, ambiguous intent, and physical fatigue. 

The **Gestura Gesture Bible** establishes an authoritative, research-grade gesture language for spatial computing. Operating over commodity monocular webcams augmented by contextual audio, Gestura treats human gestures not as discrete button clicks, but as continuous, confidence-aware, temporal intent trajectories. This specification formally decouples physical finger anatomy, raw static hand poses, and kinetic motion primitives from application-specific commands, creating an operating-system-independent interaction standard comparable to W3C DOM events or ISO/IEC ergonomic standards.

---

## 2. Design Philosophy

Every gesture and spatial interaction defined within Gestura complies with five immutable architectural principles:

```
  ┌─────────────────────────────────────────────────────────────┐
  │                 GESTURA DESIGN PHILOSOPHY                   │
  └─────────────────────────────────────────────────────────────┘
         │               │               │               │
         ▼               ▼               ▼               ▼
   [Physics-Grounded] [Intent-Driven] [Zero-Fatigue] [Decoupled]
```

1. **Physics-Grounded Continuity**: Digital objects possess virtual mass, drag, and spatial inertia. Interactions must avoid instantaneous position snaps; gestures establish kinematic constraints and transfer momentum with smooth mathematical continuity.
2. **Intent Before Action**: Single-frame classification is prohibited. The engine distinguishes between an involuntary transition movement (*Gesture Candidate*), sustained physiological alignment (*Intent Confidence*), and confirmed execution (*Action*).
3. **Ergonomic Neutrality & Zero-Fatigue ("Gorilla Arm" Prevention)**: Gestures must function within the relaxed arm envelope (forearm resting on desk or lap, hand oriented 30°–60° from vertical). Interactions requiring sustained shoulder flexion or extreme wrist extension are structurally disqualified.
4. **Asymmetric Bimanual Cooperativity**: Derived from Guiard’s Kinematic Chain Model, two-hand interactions divide labor: the non-dominant hand establishes the spatial reference frame (orientation, scale, coordinate origin), while the dominant hand performs precision manipulations (pointing, pinching, holding).
5. **Decoupled Semantic Abstraction**: Hand tracking and gesture classification produce abstract semantic interaction tokens (e.g., `SPATIAL_SWIPE_HORIZONTAL`, `NODE_HOLD_INITIATE`). Client environments (3D visualizers, desktop window managers, robotic interfaces) interpret tokens according to their local domain grammar.

---

## 3. Recognition Hierarchy

Gestura models human hand interaction as a 5-layer hierarchical stack. Higher layers consume primitives from lower layers to build deterministic interaction semantics.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ LEVEL 4: TWO-HAND COOPERATIVE GESTURES (T001–T099)                      │
│ Asymmetric dual-hand manipulation: Bimanual scale, rotation, framing    │
├─────────────────────────────────────────────────────────────────────────┤
│ LEVEL 3: COMPLETE GESTURES (G001–G999)                                  │
│ Composition: Hand Pose + Motion Primitive + Intent Lifecycle            │
├─────────────────────────────────────────────────────────────────────────┤
│ LEVEL 2: MOTION PRIMITIVES (M001–M099)                                  │
│ Kinematic trajectories: Direction vector, velocity, displacement        │
├─────────────────────────────────────────────────────────────────────────┤
│ LEVEL 1: STATIC HAND POSES (H001–H099)                                  │
│ Spatial configurations: Open palm, closed fist, pinch, point, thumbs-up │
├─────────────────────────────────────────────────────────────────────────┤
│ LEVEL 0: FINGER STATES (F001–F099)                                      │
│ Discrete digit posture: Flexion angles, extension ratios, abduction     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Gesture Taxonomy

Gestures in Gestura are taxonomically classified along four orthogonal axes:

1. **Temporal Horizon**:
   - *Static Postures*: Steady states held over $\Delta t \ge 150\,\text{ms}$ (e.g., `H004 Index Point`).
   - *Dynamic Strokes*: Trajectory-based transient actions over $80\,\text{ms} \le \Delta t \le 400\,\text{ms}$ (e.g., `G001 Swipe Left`).
   - *Continuous Streams*: Real-time tracking streams with 1:1 spatial coupling (e.g., `G007 Pinch Drag & Hold`).
2. **Contact Topology**:
   - *Free-Space*: Digits operate without self-contact or physical surface interaction.
   - *Self-Contact (Chiral)*: Digits of the same hand touch (e.g., thumb-index pad pinch, thumb tap).
   - *Bimanual Contact*: Digits or palms of opposing hands meet.
3. **Symmetry**:
   - *Unimanual*: Single hand acting in isolation.
   - *Bimanual Symmetric*: Both hands perform mirror actions (e.g., dual-hand inward contraction).
   - *Bimanual Asymmetric*: Left hand sets context, right hand acts (Guiard chain).
4. **Spatial Granularity**:
   - *Macro-Gestures*: Gross motor movements involving wrist, forearm, and palm displacement.
   - *Micro-Gestures*: Fine motor movements localized to metacarpophalangeal (MCP) and interphalangeal (IP) joints.

---

## 5. Intent vs. Gesture

A fundamental flaw of legacy computer vision interfaces is equating a detected posture with user intent. In Gestura:

$$\text{Action Execution} = f(\text{Pose}, \text{Kinematics}, \text{Temporal Consistency}, \text{Spatial Context})$$

* **Gesture Candidate ($S_{\text{cand}}$)**: An instantaneous classification match from raw landmarks. It generates zero side effects.
* **Intent Confidence ($C_{\text{intent}} \in [0, 1]$)**: A time-decaying evidence accumulator tracking how steadily the user sustains the posture or motion trajectory against background noise.
* **Action ($A_{\text{exec}}$)**: State transition triggered only when $C_{\text{intent}}$ crosses the activation threshold $\theta_{\text{act}}$ and satisfies minimum duration and stability constraints.

```
Landmark Stream ──► Feature Extraction ──► Candidate Pose / Motion
                                                   │
                                                   ▼
Idle/Observing ◄── Low Confidence ◄── Intent Evidence Filter
                                                   │
                                           Sustained Evidence
                                                   │
                                                   ▼
                                        CONFIRMED / ACTIVE
                                                   │
                                                   ▼
                                        Emit Abstract Command
```

---

## 6. Continuous Confidence Model

All geometric and kinematic parameters are mapped through continuous, differentiable activation functions (sigmoids and Gaussians) to avoid brittle binary step thresholds.

### 6.1. Sigmoid Confidence Estimation
For metric $x$ with target threshold $x_0$ and steepness $k$:

$$C_{\text{sig}}(x; x_0, k, \text{invert}) = \frac{1}{1 + \exp\left( -k \cdot (x - x_0) \cdot (-1)^{\text{invert}} \right)}$$

### 6.2. Multi-Feature Fusion
Individual feature confidences $c_1, c_2, \dots, c_n$ are combined via weighted arithmetic consensus:

$$C_{\text{composite}} = \sum_{i=1}^n w_i c_i, \quad \text{where } \sum_{i=1}^n w_i = 1$$

Geometric products ($\prod c_i$) are strictly prohibited for primary gating because any single noisy landmark will collapse the entire confidence to zero.

---

## 7. Temporal Recognition Model

Gestura enforces a 6-state finite state machine (FSM) for all gesture lifecycles:

```
[ IDLE ] ──► [ OBSERVING ] ──► [ CANDIDATE ] ──► [ CONFIRMED ] ──► [ ACTIVE ] ──► [ RELEASE ]
   ▲               │                 │                                  │              │
   │               ▼                 ▼                                  │              │
   └───────────────┴─────────────────┴──────────────────────────────────┴──────────────┘
```

1. **IDLE**: No hand detected or hand at rest below interaction boundary.
2. **OBSERVING**: Hand detected. Visual tracking, spatial raycasting, and pointer hover active. No action committed.
3. **CANDIDATE**: Gesture posture detected. Evidence accumulation buffer begins filling.
4. **CONFIRMED**: Evidence threshold $\theta_{\text{act}}$ reached for $\ge N_{\text{confirm}}$ frames. User intent verified.
5. **ACTIVE**: Gesture command actively driving target object or spatial canvas. 1:1 kinematic tracking.
6. **RELEASE**: Posture dissolved or confidence dropped below $\theta_{\text{rel}}$. Spatial momentum gracefully decoupled. Objects anchor without snapback. Cooldown timer initiated.

### Transient vs. Sustained Actions
* **Transient Gestures (e.g., Slaps, Swipes, Air Taps)**: Vetted by kinematic velocity and trajectory linearity over a sliding window ($80\,\text{ms} \le \Delta t \le 400\,\text{ms}$). Upon reaching `CONFIRMED`, they emit a single impulse command in `ACTIVE` and transition immediately back to `OBSERVING`.
* **Sustained Gestures (e.g., Pinches, Grabs, Points)**: Remain in `ACTIVE` for arbitrary durations while the posture is held, feeding continuous delta translation/rotation streams until explicit `RELEASE`.

---

## 8. Dominant Hand Strategy

In dual-hand scenarios:

1. **Primary Interaction Hand (Dominant)**: Selected dynamically based on intent energy:
   $$E_{\text{hand}} = w_g C_{\text{gesture}} + w_d (1 - d_{\text{center}}) + w_v v_{\text{palm}}$$
   The hand executing explicit manipulation postures (Pinch, Point, Grab) takes primary status.
2. **Modifier Hand (Non-Dominant)**: Acts as spatial anchor or parameter modifier.
3. **Isolation Rule**: The modifier hand **never** independently triggers directional swipe gestures or global application shortcuts.
4. **Bimanual Transition Rule**: When both hands concurrently engage in cooperative postures (e.g., open palms apart/together or dual pinch), the engine transitions from single-hand mapping to Level 4 Bimanual Navigation (`T001–T099`).

---

## 9. Future Expansion Strategy & Deprecation Lifecycle

1. **Immutability of IDs**: Once an ID (`F001`, `H001`, `M001`, `G001`, `T001`, `C001`) is assigned, its fundamental anatomical definition is frozen.
2. **Deprecation**: Deprecated gestures are marked `[DEPRECATED in vX.Y.Z]` with a migration pointer. IDs are never recycled.
3. **Extensibility**: Specialized interaction profiles (e.g., Medical VR, Automotive HMI, Robotic Teleoperation) extend the specification by adding IDs within reserved ranges without modifying core levels.

---

# Part II: Permanent Numbering System & Landmark Schema

## 1. ID Allocation Namespace

```
F001–F099   Level 0: Finger States (Anatomical digit postures)
H001–H099   Level 1: Static Hand Poses (Still configurations)
M001–M099   Level 2: Motion Primitives (Kinematic strokes)
G001–G999   Level 3: Complete Gestures (Intent compositions)
T001–T099   Level 4: Two-Hand Gestures (Cooperative bimanual)
FM01–FM99   Finger Micro-Gestures (Fine-motor digit interactions)
C001–C999   Context Mappings (Abstract command bindings)
```

## 2. Anatomical Reference Landmark Schema

Gestura utilizes the standard 21 3D hand landmarks:

```
                  [8] Tip
                  │
                 [7] DIP
                  │
                 [6] PIP
                  │
                 [5] MCP
                  │
    [4] Tip ───┐  │      [12] Tip
         │     │  │        │
       [3] IP  │  │       [11] DIP   [16] Tip
         │     │  │        │           │
       [2] MCP └──┼───────[10] PIP    [15] DIP   [20] Tip
         │        │        │           │           │
       [1] CMC ───┴───────[9] MCP ───[14] PIP    [19] DIP
         │                             │           │
         └───────────────────────────[13] MCP ───[18] PIP
                  │                    │           │
                  └────────────────────┴─────────[17] MCP
                            │
                           [0] WRIST
```

* **Thumb**: Landmarks 1 (CMC), 2 (MCP), 3 (IP), 4 (Tip)
* **Index**: Landmarks 5 (MCP), 6 (PIP), 7 (DIP), 8 (Tip)
* **Middle**: Landmarks 9 (MCP), 10 (PIP), 11 (DIP), 12 (Tip)
* **Ring**: Landmarks 13 (MCP), 14 (PIP), 15 (DIP), 16 (Tip)
* **Little (Pinky)**: Landmarks 17 (MCP), 18 (PIP), 19 (DIP), 20 (Tip)
* **Reference Metric ($d_{\text{ref}}$)**: Euclidean distance $\|P_0 - P_9\|$ (Wrist to Middle MCP). Used as scale-invariant hand normalizer.

---

# Part III: Level 0 — Finger States (F001–F099)

### F001 — Extended
* **Description**: Finger straightened fully along its longitudinal axis.
* **Anatomical Definition**: PIP joint angle $\theta_{\text{PIP}} < 20^\circ$, DIP joint angle $\theta_{\text{DIP}} < 20^\circ$. Extension ratio $R_{\text{ext}} = \|P_{\text{tip}} - P_0\| / \|P_{\text{mcp}} - P_0\| \ge 1.15$.
* **Recognition Criteria**: $R_{\text{ext}} \ge 1.15$ with confidence $C_{\text{sig}}(R_{\text{ext}}; 1.12, 10.0)$.
* **Ambiguities**: May be confused with F003 (Relaxed) if camera viewing angle foreshortens the finger.

### F002 — Folded / Curled
* **Description**: Finger flexed inward toward the palm.
* **Anatomical Definition**: PIP joint angle $\theta_{\text{PIP}} > 90^\circ$, DIP joint angle $\theta_{\text{DIP}} > 70^\circ$. Fingertip distance to palm center $\|P_{\text{tip}} - P_{\text{palm}}\| < 0.45 \cdot d_{\text{ref}}$.
* **Recognition Criteria**: Extension ratio $R_{\text{ext}} \le 0.90$.
* **Ambiguities**: Thumb folded across palm can occlude index MCP joint.

### F003 — Relaxed / Semi-Flexed
* **Description**: Natural physiological resting state; digits curved without muscular tension.
* **Anatomical Definition**: $25^\circ \le \theta_{\text{PIP}} \le 65^\circ$. Extension ratio $0.95 \le R_{\text{ext}} \le 1.14$.
* **Recognition Criteria**: Gaussian confidence centered at $R_{\text{ext}} = 1.05$ with $\sigma = 0.08$.
* **Ambiguities**: Easily transitions into F001 or F002; must not trigger state commits.

### F004 — Hyper-Extended
* **Description**: Finger straightened beyond $180^\circ$ planar alignment (dorsal deflection).
* **Anatomical Definition**: Angle between vector $(P_{\text{mcp}} - P_{\text{pip}})$ and $(P_{\text{pip}} - P_{\text{tip}}) < -5^\circ$ dorsally.
* **Recognition Criteria**: Detected via 3D ray plane normal projection.
* **Ambiguities**: Often caused by camera perspective warp; treated as F001 in standard pipelines.

### F005 — Hooked / Flexed PIP
* **Description**: MCP extended, but PIP and DIP joints acutely flexed (claw posture).
* **Anatomical Definition**: Angle at MCP $\le 30^\circ$, but PIP angle $\ge 80^\circ$.
* **Recognition Criteria**: Distance $\|P_{\text{tip}} - P_{\text{mcp}}\| \le 0.50 \cdot \|P_{\text{pip}} - P_{\text{mcp}}\|$.
* **Ambiguities**: Can be confused with F002 under steep pitch angles.

### F006 — Abducted (Spread)
* **Description**: Digits separated laterally away from the hand midline (Middle finger axis).
* **Anatomical Definition**: Inter-digit tip angles $\angle(P_{\text{tip}, i}, P_0, P_{\text{tip}, i+1}) \ge 18^\circ$.
* **Recognition Criteria**: Average adjacent tip separation $\|P_{\text{tip}, i} - P_{\text{tip}, i+1}\| \ge 0.38 \cdot d_{\text{ref}}$.
* **Ambiguities**: Dependent on hand distance from lens.

### F007 — Adducted (Closed)
* **Description**: Straightened digits held tightly parallel touching each other.
* **Anatomical Definition**: Inter-digit tip angles $\le 8^\circ$. Adjacent fingertip separation $< 0.18 \cdot d_{\text{ref}}$.
* **Recognition Criteria**: Variance of finger lateral vectors $\sigma^2_x \le 0.02$.
* **Ambiguities**: Landmark fusion by MediaPipe where adjacent fingertips overlap.

### F008 — Pinching Contact
* **Description**: Fingertip pad in physical or sub-centimeter contact with Thumb pad ($P_4$).
* **Anatomical Definition**: Euclidean distance $\|P_4 - P_{\text{tip}, i}\| \le 0.35 \cdot d_{\text{ref}}$.
* **Recognition Criteria**: Sigmoidal confidence with threshold $d_{\text{pinch}} \le 0.38 \cdot d_{\text{ref}}$ and steepness $k = 18.0$.
* **Ambiguities**: False positive if index passes behind thumb in 2D perspective without actual 3D contact.

### F009 — Crossed
* **Description**: One finger positioned overlapping dorsally or laterally over an adjacent finger.
* **Anatomical Definition**: Vector intersection of digit central rays in hand plane.
* **Recognition Criteria**: Distance between lateral midpoints $< 0.12 \cdot d_{\text{ref}}$ with opposing lateral normals.
* **Ambiguities**: Heavy occlusion; requires temporal landmark tracking.

### F010 — Tapped
* **Description**: Transient contact event where a fingertip rapidly impacts a surface or another digit ($\le 80\,\text{ms}$).
* **Anatomical Definition**: Peak negative acceleration along approach vector followed by immediate release.
* **Recognition Criteria**: Velocity inflection $|a_{\text{approach}}| \ge 1.5\,\text{m/s}^2$ within contact sphere.
* **Ambiguities**: Tremor or micro-jitter.

### F011 — Opposed Thumb
* **Description**: Thumb rotated across the palm plane facing the palmar pads of digits 2–5.
* **Anatomical Definition**: Angle between thumb CMC-IP normal and palm plane normal $\ge 60^\circ$.
* **Recognition Criteria**: Dot product of palm normal and thumb normal $\le 0.50$.
* **Ambiguities**: Difficult to detect when hand is oriented edge-on to camera.

### F012 — Radial Abducted Thumb
* **Description**: Thumb extended outward away from index finger in the coronal plane ("hitchhiker" or "L-shape").
* **Anatomical Definition**: Angle between $(P_4 - P_2)$ and $(P_8 - P_5) \ge 65^\circ$.
* **Recognition Criteria**: Distance $\|P_4 - P_8\| \ge 0.75 \cdot d_{\text{ref}}$ while index is extended.
* **Ambiguities**: None; highly reliable geometry.

---

# Part IV: Level 1 — Static Hand Poses (H001–H099)

### H001 — Open Palm Neutral
* **Purpose**: Base spatial navigation, observation, and system rest.
* **Finger Anatomy**: Thumb: F001 (Extended) or F003 (Relaxed); Index: F001; Middle: F001; Ring: F001; Little: F001. All digits extended naturally without forced lateral spread.
* **Palm Orientation**: Facing display camera ($\mathbf{n}_{\text{palm}} \cdot \mathbf{v}_{\text{cam}} \le -0.55$).
* **Recognition Rules**: All digits $R_{\text{ext}} \ge 1.10$. Average adjacent finger spread between $10^\circ$ and $22^\circ$.
* **Confidence Requirements**: Composite confidence $\ge 0.48$.
* **Failure Conditions**: Abort if thumb-index distance indicates pinch contact ($< 0.35 \cdot d_{\text{ref}}$).

### H002 — Open Palm Spread
* **Purpose**: Maximum world expansion, zoom-in, primary attention signal.
* **Finger Anatomy**: Thumb: F012 (Radial Abducted); Index: F001 + F006; Middle: F001 + F006; Ring: F001 + F006; Little: F001 + F006.
* **Palm Orientation**: Direct facing camera ($\mathbf{n}_{\text{palm}} \cdot \mathbf{v}_{\text{cam}} \le -0.65$).
* **Recognition Rules**: All digits $R_{\text{ext}} \ge 1.22$. Average adjacent fingertip spread $\ge 25^\circ$.
* **Confidence Requirements**: Composite confidence $\ge 0.55$.
* **Failure Conditions**: Digits touching or curled.

### H003 — Closed Fist
* **Purpose**: World zoom-out, overview collapse, object grab anchor.
* **Finger Anatomy**: Thumb: F002 (Curled over fingers); Index: F002; Middle: F002; Ring: F002; Little: F002. All 5 digits balled tightly into palm.
* **Palm Orientation**: Any orientation (Forward, Down, Inward).
* **Recognition Rules**: All digits $R_{\text{ext}} \le 0.90$. Pinch confidence $< 0.40$ (thumb tip not isolated on index tip).
* **Confidence Requirements**: Composite confidence $\ge 0.55$.
* **Failure Conditions**: If index or thumb remains extended, do not classify as fist.

### H004 — Index Point
* **Purpose**: Precision raycasting, node targeting, spatial cursor targeting.
* **Finger Anatomy**: Thumb: F002 or F003 (Curled or resting); Index: F001 (Fully Extended, $R_{\text{ext}} \ge 1.25$); Middle: F002; Ring: F002; Little: F002.
* **Palm Orientation**: Facing screen or rotated inward 45° (pronation).
* **Recognition Rules**: $R_{\text{ext}}(\text{Index}) \ge 1.15$ AND $R_{\text{ext}}(\text{Index}) - \text{mean}(R_{\text{ext}}(\text{Mid, Ring, Pinky})) \ge 0.22$.
* **Confidence Requirements**: Confidence $\ge 0.45$.
* **Failure Conditions**: Abort if middle finger is also extended (transitions to H009 Peace).

### H005 — Precision Pinch
* **Purpose**: Fine object selection, node grabbing, parameter slider manipulation.
* **Finger Anatomy**: Thumb: F008 (Pad touching Index pad); Index: F008 (Pad touching Thumb pad); Middle: F002 or F003 (Relaxed or folded); Ring: F002; Little: F002.
* **Palm Orientation**: Coronal or semi-pronated ($30^\circ \le \theta_{\text{roll}} \le 80^\circ$).
* **Recognition Rules**: $\|P_4 - P_8\| \le 0.35 \cdot d_{\text{ref}}$.
* **Confidence Requirements**: Pinch metric confidence $\ge 0.45$.
* **Failure Conditions**: Must not trigger if all other digits are tightly balled into palm with thumb overlapping index nail (fist ambiguity).

### H006 — Lateral Pinch / Key Pinch
* **Purpose**: Secondary spatial interaction, continuous dial turning.
* **Finger Anatomy**: Thumb pad pressed against the lateral radial side of Index middle phalanx ($P_6$–$P_7$). Digits 3–5 curled.
* **Palm Orientation**: Inward facing (side).
* **Recognition Rules**: $\|P_4 - P_6\| \le 0.32 \cdot d_{\text{ref}}$ while Index is semi-flexed.
* **Confidence Requirements**: Confidence $\ge 0.50$.
* **Failure Conditions**: Confused with Closed Fist under poor lighting.

### H007 — Thumbs Up
* **Purpose**: Modal confirmation, affirmative semantic response, positive commit.
* **Finger Anatomy**: Thumb: F001 (Fully extended vertically, $R_{\text{ext}} \ge 1.20$); Index: F002; Middle: F002; Ring: F002; Little: F002.
* **Palm Orientation**: Edge-on to camera (lateral), thumb vector pointing upward ($\mathbf{v}_{\text{thumb}} \cdot \mathbf{u}_{\text{world}} \ge 0.70$).
* **Recognition Rules**: Thumb vector aligned with world $+Y$ axis within $25^\circ$. All other digits curled ($R_{\text{ext}} \le 0.90$).
* **Confidence Requirements**: Confidence $\ge 0.60$.
* **Failure Conditions**: Thumb tilted horizontally ($> 45^\circ$).

### H008 — Thumbs Down
* **Purpose**: Modal cancellation, negative semantic response, reject.
* **Finger Anatomy**: Thumb: F001 (Extended vertically downward); Digits 2–5: F002 (Curled).
* **Palm Orientation**: Edge-on to camera, thumb pointing downward ($\mathbf{v}_{\text{thumb}} \cdot \mathbf{u}_{\text{world}} \le -0.70$).
* **Recognition Rules**: Thumb vector aligned with world $-Y$ axis within $25^\circ$.
* **Confidence Requirements**: Confidence $\ge 0.60$.
* **Failure Conditions**: Hand moving dynamically down (ambiguity with M004 Swipe Down).

### H009 — Peace / Victory (V-Sign)
* **Purpose**: Secondary selection, view mode toggling, split screen trigger.
* **Finger Anatomy**: Thumb: F002 (Holding ring finger); Index: F001 + F006; Middle: F001 + F006; Ring: F002; Little: F002.
* **Palm Orientation**: Facing display ($\mathbf{n}_{\text{palm}} \cdot \mathbf{v}_{\text{cam}} \le -0.50$).
* **Recognition Rules**: Index and Middle extended ($R_{\text{ext}} \ge 1.15$), separation angle $\ge 15^\circ$. Ring and Little curled ($R_{\text{ext}} \le 0.92$).
* **Confidence Requirements**: Confidence $\ge 0.55$.
* **Failure Conditions**: Index and Middle held together (transitions to flat knife-edge).

### H010 — OK Ring
* **Purpose**: Parameter lock, calibration anchor, success state acknowledgment.
* **Finger Anatomy**: Thumb: F008 (Contact with Index tip); Index: F008 (Contact with Thumb tip); Middle: F001 (Extended); Ring: F001 (Extended); Little: F001 (Extended).
* **Palm Orientation**: Facing camera or tilted $30^\circ$.
* **Recognition Rules**: $\|P_4 - P_8\| \le 0.32 \cdot d_{\text{ref}}$ AND $R_{\text{ext}}(\text{Middle, Ring, Little}) \ge 1.10$.
* **Confidence Requirements**: Confidence $\ge 0.58$.
* **Failure Conditions**: Ring or Pinky curled (ambiguity with H005 Precision Pinch).

### H011 — Three-Finger Spread
* **Purpose**: Workspace desktop switcher, window tiling overview.
* **Finger Anatomy**: Thumb: F002; Index: F001; Middle: F001; Ring: F001; Little: F002.
* **Palm Orientation**: Facing screen.
* **Recognition Rules**: Index, Middle, Ring extended ($R_{\text{ext}} \ge 1.15$). Pinky and Thumb curled.
* **Confidence Requirements**: Confidence $\ge 0.55$.
* **Failure Conditions**: Pinky extended.

### H012 — Call Me (Shaka / Horns Variant)
* **Purpose**: Contextual shortcut, communication dock trigger.
* **Finger Anatomy**: Thumb: F001 (Extended); Index: F002; Middle: F002; Ring: F002; Little: F001 (Extended).
* **Palm Orientation**: Facing screen or tilted.
* **Recognition Rules**: Thumb and Pinky $R_{\text{ext}} \ge 1.15$. Digits 2–4 curled ($R_{\text{ext}} \le 0.90$).
* **Confidence Requirements**: Confidence $\ge 0.60$.
* **Failure Conditions**: Any central finger uncurling.

### H013 — Gun / L-Shape Point
* **Purpose**: Directional snapping, spatial orientation ruler.
* **Finger Anatomy**: Thumb: F001 (Extended upward); Index: F001 (Extended forward); Digits 3–5: F002 (Curled).
* **Palm Orientation**: Lateral edge-on.
* **Recognition Rules**: Angle between Thumb and Index vector $75^\circ \le \theta \le 105^\circ$.
* **Confidence Requirements**: Confidence $\ge 0.55$.
* **Failure Conditions**: Palm facing flat toward camera.

### H014 — Cupped Hand
* **Purpose**: Audio volume adjustment, gathering objects, virtual scoop.
* **Finger Anatomy**: All digits semi-flexed (F003/F005) with adduction (F007), forming a concave palmar bowl.
* **Palm Orientation**: Upward or toward user face.
* **Recognition Rules**: Palmar concavity index $> 0.25 \cdot d_{\text{ref}}$, fingertips coplanar.
* **Confidence Requirements**: Confidence $\ge 0.50$.
* **Failure Conditions**: Hand flat.

### H015 — Flat Hand Knife Edge
* **Purpose**: Spatial plane slicing, dividing canvas, boundary positioning.
* **Finger Anatomy**: Digits 1–5 fully extended (F001) and tightly adducted together (F007).
* **Palm Orientation**: Oriented perpendicular to display plane ($\mathbf{n}_{\text{palm}} \cdot \mathbf{v}_{\text{cam}} \approx 0$).
* **Recognition Rules**: Hand normal orthogonal to camera vector ($|\mathbf{n}_{\text{palm}} \cdot \mathbf{v}_{\text{cam}}| \le 0.20$).
* **Confidence Requirements**: Confidence $\ge 0.55$.
* **Failure Conditions**: Palm turning flat to camera.

### H016 — Finger Guns Double Point
* **Purpose**: Multi-object selection, coordinate axis alignment.
* **Finger Anatomy**: Index and Middle fingers extended tightly together (F001 + F007); Thumb extended upward (F001); Ring and Pinky curled (F002).
* **Palm Orientation**: Lateral.
* **Recognition Rules**: Index and Middle parallel ($< 6^\circ$ divergence), Thumb vertical.
* **Confidence Requirements**: Confidence $\ge 0.55$.
* **Failure Conditions**: Separation between index and middle digits.

---

# Part V: Level 2 — Motion Primitives (M001–M099)

Motion primitives represent pure spatial kinematics computed from the smoothed palm center velocity vector $\mathbf{v} = (v_x, v_y, v_z)$, net displacement $\mathbf{d} = (d_x, d_y, d_z)$, and trajectory path length $L$.

### M001 — Swipe Left (Linear Stroke)
* **Direction Vector**: $\mathbf{u} = (-1, 0, 0)$ (Normalized screen-space $X-$).
* **Velocity**: Peak speed $\|\mathbf{v}\| \ge 0.22\,\text{m/s}$ (or screen units/sec).
* **Duration**: $80\,\text{ms} \le \Delta t \le 380\,\text{ms}$.
* **Distance**: Displacement $\|\mathbf{d}\| \ge 0.045\,\text{screen units}$.
* **Temporal Consistency**: Linearity ratio $\|\mathbf{d}\| / L \ge 0.70$.
* **Confidence**: $C = \min\left(1.0, 0.70 + 0.30 \cdot \frac{\|\mathbf{d}\| / L - 0.70}{0.25}\right)$.

### M002 — Swipe Right (Linear Stroke)
* **Direction Vector**: $\mathbf{u} = (+1, 0, 0)$ (Normalized screen-space $X+$).
* **Velocity**: Peak speed $\|\mathbf{v}\| \ge 0.22\,\text{m/s}$.
* **Duration**: $80\,\text{ms} \le \Delta t \le 380\,\text{ms}$.
* **Distance**: Displacement $\|\mathbf{d}\| \ge 0.045\,\text{screen units}$.
* **Temporal Consistency**: Linearity ratio $\|\mathbf{d}\| / L \ge 0.70$.
* **Confidence**: Proportional to velocity and linearity consensus.

### M003 — Swipe Up (Linear Stroke)
* **Direction Vector**: $\mathbf{u} = (0, -1, 0)$ (MediaPipe inverted $Y-$ / World upward $+Y$).
* **Velocity**: Peak speed $\|\mathbf{v}\| \ge 0.20\,\text{m/s}$.
* **Duration**: $80\,\text{ms} \le \Delta t \le 380\,\text{ms}$.
* **Distance**: Displacement $\|\mathbf{d}\| \ge 0.040\,\text{screen units}$.
* **Temporal Consistency**: Linearity ratio $\|\mathbf{d}\| / L \ge 0.70$.
* **Confidence**: Gated on vertical dominance: $|d_y| \ge 1.4 \cdot |d_x|$.

### M004 — Swipe Down (Linear Stroke)
* **Direction Vector**: $\mathbf{u} = (0, +1, 0)$ (MediaPipe inverted $Y+$ / World downward $-Y$).
* **Velocity**: Peak speed $\|\mathbf{v}\| \ge 0.20\,\text{m/s}$.
* **Duration**: $80\,\text{ms} \le \Delta t \le 380\,\text{ms}$.
* **Distance**: Displacement $\|\mathbf{d}\| \ge 0.040\,\text{screen units}$.
* **Temporal Consistency**: Linearity ratio $\|\mathbf{d}\| / L \ge 0.70$.
* **Confidence**: Gated on vertical dominance: $|d_y| \ge 1.4 \cdot |d_x|$.

### M005 — Push Forward (Z+ Thrust)
* **Direction Vector**: $\mathbf{u} = (0, 0, -1)$ (Toward screen / camera optical center).
* **Velocity**: Rate of scale expansion $\frac{d}{dt}(d_{\text{ref}}) \ge 0.35\,\text{s}^{-1}$.
* **Duration**: $100\,\text{ms} \le \Delta t \le 400\,\text{ms}$.
* **Distance**: Hand bounding reference metric increases by $\ge 20\%$.
* **Temporal Consistency**: Continuous monotonic expansion across $\ge 3$ consecutive frames.
* **Confidence**: Confidence proportional to depth acceleration.

### M006 — Pull Back (Z- Retraction)
* **Direction Vector**: $\mathbf{u} = (0, 0, +1)$ (Away from screen toward user body).
* **Velocity**: Rate of scale contraction $\frac{d}{dt}(d_{\text{ref}}) \le -0.35\,\text{s}^{-1}$.
* **Duration**: $100\,\text{ms} \le \Delta t \le 400\,\text{ms}$.
* **Distance**: Hand bounding metric decreases by $\ge 20\%$.
* **Temporal Consistency**: Monotonic shrinking of $d_{\text{ref}}$.
* **Confidence**: Proportional to retreat velocity.

### M007 — Circle Clockwise (Axial Orbital)
* **Direction Vector**: Rotational trajectory in $XY$ plane: cumulative angular delta $\Delta \theta \ge +270^\circ$.
* **Velocity**: Angular speed $\omega \ge 2.5\,\text{rad/s}$.
* **Duration**: $250\,\text{ms} \le \Delta t \le 1200\,\text{ms}$.
* **Distance**: Orbit radius $r \ge 0.05\,\text{screen units}$.
* **Temporal Consistency**: Bounded radial variance $\sigma^2_r / \bar{r} \le 0.25$.
* **Confidence**: Integrated winding number over trajectory buffer.

### M008 — Circle Counter-Clockwise (Axial Orbital)
* **Direction Vector**: Rotational trajectory in $XY$ plane: cumulative angular delta $\Delta \theta \le -270^\circ$.
* **Velocity**: Angular speed $\omega \le -2.5\,\text{rad/s}$.
* **Duration**: $250\,\text{ms} \le \Delta t \le 1200\,\text{ms}$.
* **Distance**: Orbit radius $r \ge 0.05\,\text{screen units}$.
* **Temporal Consistency**: Bounded radial variance $\sigma^2_r / \bar{r} \le 0.25$.
* **Confidence**: Integrated negative winding number.

### M009 — Static Hold / Dwell
* **Direction Vector**: $\mathbf{u} = (0, 0, 0)$.
* **Velocity**: Palm velocity $\|\mathbf{v}\| \le 0.035\,\text{m/s}$ (jitter filter threshold).
* **Duration**: Sustained $\Delta t \ge 400\,\text{ms}$.
* **Distance**: Total displacement sphere radius $r \le 0.020\,\text{screen units}$.
* **Temporal Consistency**: Stationarity metric $S = 1.0 - \min(1.0, \|\mathbf{v}\| / 0.035)$.
* **Confidence**: Asymptotic rise toward $1.0$ as dwell time approaches $1000\,\text{ms}$.

### M010 — Directional Flick
* **Direction Vector**: Any linear vector in $XY$ plane.
* **Velocity**: Extreme acceleration stroke: $\|\mathbf{a}\| \ge 3.0\,\text{m/s}^2$, peak $\|\mathbf{v}\| \ge 0.45\,\text{m/s}$.
* **Duration**: Rapid impulse $40\,\text{ms} \le \Delta t \le 120\,\text{ms}$.
* **Distance**: Short displacement $0.025 \le \|\mathbf{d}\| \le 0.065\,\text{screen units}$.
* **Temporal Consistency**: High velocity followed by immediate deceleration.
* **Confidence**: Impulse detection score.

### M011 — Wrist Roll (Pronation / Supination)
* **Direction Vector**: Angular rotation around forearm longitudinal axis.
* **Velocity**: Angular roll speed $|\dot{\theta}_{\text{roll}}| \ge 2.0\,\text{rad/s}$.
* **Duration**: $100\,\text{ms} \le \Delta t \le 500\,\text{ms}$.
* **Distance**: Angular displacement $|\Delta \theta_{\text{roll}}| \ge 45^\circ$.
* **Temporal Consistency**: Normal vector rotation around palm $Y$ axis.
* **Confidence**: Computed from landmark plane normal rotation matrix.

### M012 — Spatial Wave / Oscillate
* **Direction Vector**: Alternating horizontal stroke ($X+ \to X- \to X+$).
* **Velocity**: Zero-crossing rate of $v_x \ge 2$ direction reversals within $600\,\text{ms}$.
* **Duration**: $300\,\text{ms} \le \Delta t \le 800\,\text{ms}$.
* **Distance**: Peak-to-peak amplitude $A \ge 0.06\,\text{screen units}$.
* **Temporal Consistency**: Periodic frequency analysis.
* **Confidence**: Sinusoidal fit score.

### M013 — Tilt Pitch
* **Direction Vector**: Hand angular rotation around lateral axis (fingers pointing up/down).
* **Velocity**: $|\dot{\theta}_{\text{pitch}}| \ge 1.8\,\text{rad/s}$.
* **Duration**: $120\,\text{ms} \le \Delta t \le 600\,\text{ms}$.
* **Distance**: Angular change $|\Delta \theta_{\text{pitch}}| \ge 35^\circ$.
* **Temporal Consistency**: Monotonic rotation of forearm-palm angle.
* **Confidence**: Euler pitch derivative continuity.

### M014 — Radial Drift
* **Direction Vector**: Slow, low-velocity translation below deliberate stroke threshold ($0.04 \le \|\mathbf{v}\| \le 0.12\,\text{m/s}$).
* **Velocity**: Unintentional background drift.
* **Duration**: Arbitrary.
* **Distance**: Unbounded.
* **Temporal Consistency**: Low linearity.
* **Confidence**: Negative confidence primitive used to suppress false activations.

---

# Part VI: Level 3 — Complete Gestures (G001–G999)

Every gesture in this section complies strictly with the official Gestura Gesture Anatomy Template.

---

## G001 — Swipe Left (Primary Slap)

**Category**  
Motion

**Purpose**  
Triggers rapid horizontal navigation to the next item, desktop, or carousel pane, or imparts an intensity-driven 360° counter-clockwise spin to the 3D globe.

**Finger Anatomy**  
* Thumb: F001 (Extended) or F003 (Relaxed)
* Index: F001 (Extended)
* Middle: F001 (Extended)
* Ring: F001 (Extended)
* Little: F001 (Extended)  
*Note: Hand must be open; digits may not be balled into a fist.*

**Palm Orientation**  
Forward facing camera ($\mathbf{n}_{\text{palm}} \cdot \mathbf{v}_{\text{cam}} \le -0.45$).

**Motion**  
M001 Swipe Left: Rapid horizontal stroke from screen right to screen left across the user's field of view.

**Recognition Criteria**  
* Minimum confidence: 0.65
* Velocity: Peak $\|\mathbf{v}\| \ge 0.22\,\text{m/s}$
* Duration: $80\,\text{ms} \le \Delta t \le 380\,\text{ms}$
* Stability: Linearity consistency ratio $\ge 0.68$
* Direction: Horizontal dominance $|d_x| \ge 1.4 \cdot |d_y|$ with $d_x < 0$

**State Machine**  
IDLE → OBSERVING → CANDIDATE (Stroke Initiation) → ACTIVE (Impulse Dispatched) → OBSERVING (Auto-return)

**Failure Conditions**  
* Closed fist (H003) detected (must not trigger swipe).
* Hand moving left as secondary modifier hand in two-hand mode (suppressed).
* Slow drifting hand movement ($\|\mathbf{v}\| < 0.20\,\text{m/s}$).
* Active refractory period ($220\,\text{ms}$) following preceding swipe.

**Developer Notes**  
The swipe gesture intensity directly dictates command output. For continuous 3D canvases, the impulse velocity $v_x$ is mapped to angular momentum ($2\pi \times \text{intensity}$) with smooth inertial deceleration.

---

## G002 — Swipe Right (Primary Slap)

**Category**  
Motion

**Purpose**  
Triggers rapid horizontal navigation to the previous item, desktop, or carousel pane, or imparts a 360° clockwise spin to the 3D globe.

**Finger Anatomy**  
* Thumb: F001 or F003
* Index: F001
* Middle: F001
* Ring: F001
* Little: F001

**Palm Orientation**  
Forward facing camera ($\mathbf{n}_{\text{palm}} \cdot \mathbf{v}_{\text{cam}} \le -0.45$).

**Motion**  
M002 Swipe Right: Rapid horizontal stroke from screen left to screen right.

**Recognition Criteria**  
* Minimum confidence: 0.65
* Velocity: Peak $\|\mathbf{v}\| \ge 0.22\,\text{m/s}$
* Duration: $80\,\text{ms} \le \Delta t \le 380\,\text{ms}$
* Stability: Linearity consistency ratio $\ge 0.68$
* Direction: Horizontal dominance $|d_x| \ge 1.4 \cdot |d_y|$ with $d_x > 0$

**State Machine**  
IDLE → OBSERVING → CANDIDATE → ACTIVE → OBSERVING

**Failure Conditions**  
* Hand in fist or pinch posture.
* Movement executed by modifier hand.
* Refractory period active ($220\,\text{ms}$).

**Developer Notes**  
Symmetric counterpart to G001. Requires temporal vetting to avoid mistaking arm-return repositioning for an intentional reverse swipe.

---

## G003 — Swipe Up (Vertical Slap)

**Category**  
Motion

**Purpose**  
Scrolls content upward, displays system app switcher / dock, or initiates upward 360° axial globe pitch.

**Finger Anatomy**  
* Thumb: F001 or F003
* Index: F001
* Middle: F001
* Ring: F001
* Little: F001

**Palm Orientation**  
Forward facing camera ($\mathbf{n}_{\text{palm}} \cdot \mathbf{v}_{\text{cam}} \le -0.45$).

**Motion**  
M003 Swipe Up: Rapid upward stroke (towards top of physical camera frame).

**Recognition Criteria**  
* Minimum confidence: 0.65
* Velocity: Peak speed $\ge 0.20\,\text{m/s}$
* Duration: $80\,\text{ms} \le \Delta t \le 380\,\text{ms}$
* Stability: Linearity ratio $\ge 0.68$
* Direction: Vertical dominance $|d_y| \ge 1.4 \cdot |d_x|$ with MediaPipe $d_y < 0$

**State Machine**  
IDLE → OBSERVING → CANDIDATE → ACTIVE → OBSERVING

**Failure Conditions**  
* Upward drift while maintaining static pointing posture.
* Hand leaving tracking camera frame upward.

**Developer Notes**  
Inverted coordinate space must be mapped cleanly: camera $Y=0$ is at the top of the sensor frame. Upward hand motion produces negative $\Delta y$ in normalized space.

---

## G004 — Swipe Down (Vertical Slap)

**Category**  
Motion

**Purpose**  
Scrolls content downward, minimizes active window, or initiates downward 360° axial globe pitch.

**Finger Anatomy**  
* Thumb: F001 or F003
* Index: F001
* Middle: F001
* Ring: F001
* Little: F001

**Palm Orientation**  
Forward facing camera ($\mathbf{n}_{\text{palm}} \cdot \mathbf{v}_{\text{cam}} \le -0.45$).

**Motion**  
M004 Swipe Down: Rapid downward stroke.

**Recognition Criteria**  
* Minimum confidence: 0.65
* Velocity: Peak speed $\ge 0.20\,\text{m/s}$
* Duration: $80\,\text{ms} \le \Delta t \le 380\,\text{ms}$
* Stability: Linearity ratio $\ge 0.68$
* Direction: Vertical dominance $|d_y| \ge 1.4 \cdot |d_x|$ with MediaPipe $d_y > 0$

**State Machine**  
IDLE → OBSERVING → CANDIDATE → ACTIVE → OBSERVING

**Failure Conditions**  
* Relaxed hand dropping out of interaction volume (must trigger `RELEASE`, not `SWIPE_DOWN`).
* Arm dropping due to fatigue.

**Developer Notes**  
To distinguish an intentional down-swipe from an arm drop, G004 requires an open palm with active forward normal vector ($\mathbf{n}_{\text{palm}} \cdot \mathbf{v}_{\text{cam}} \le -0.50$). Arm drops naturally pitch the palm downward ($\mathbf{n}_{\text{palm}} \cdot \mathbf{u}_{\text{floor}} \approx 1.0$), which suppresses G004.

---

## G005 — Point & Hover

**Category**  
Static

**Purpose**  
Drives screen cursor with 1:1 fidelity and magnetically highlights target interactive 3D nodes without selection.

**Finger Anatomy**  
* Thumb: F002 or F003 (Curled or relaxed against palm)
* Index: F001 (Extended, $R_{\text{ext}} \ge 1.15$)
* Middle: F002 (Curled, $R_{\text{ext}} \le 1.05$)
* Ring: F002 (Curled, $R_{\text{ext}} \le 1.05$)
* Little: F002 (Curled, $R_{\text{ext}} \le 1.05$)

**Palm Orientation**  
Facing screen or tilted inward up to 60° (natural pronation).

**Motion**  
M009 Static Hold / Smooth continuous drift.

**Recognition Criteria**  
* Minimum confidence: 0.45
* Velocity: Low to moderate pointing velocity ($\|\mathbf{v}\| \le 0.40\,\text{m/s}$)
* Duration: Sustained $\Delta t \ge 100\,\text{ms}$
* Stability: High differential between index extension and adjacent finger curl
* Direction: N/A

**State Machine**  
IDLE → OBSERVING → CANDIDATE → CONFIRMED → ACTIVE (Cursor tracking) → RELEASE

**Failure Conditions**  
* Middle finger uncurls (transitions to H009 Peace).
* Thumb pad touches index fingertip (transitions to G006 Pinch).
* Hand balled into complete fist (H003).

**Developer Notes**  
Pointing generates the `HOVER` command. It highlights items via screen-space magnetic projection ($d_{\text{screen}} \le 0.28\,\text{NDC}$), but NEVER executes destructive selection.

---

## G006 — Precision Pinch Select

**Category**  
Static / Self-Contact

**Purpose**  
Confirms item selection, expands hierarchical data clusters, or triggers primary click action at cursor coordinate.

**Finger Anatomy**  
* Thumb: F008 (Pad touching Index pad)
* Index: F008 (Pad touching Thumb pad)
* Middle: F002 or F003
* Ring: F002
* Little: F002

**Palm Orientation**  
Facing camera or semi-pronated ($30^\circ \le \theta_{\text{roll}} \le 80^\circ$).

**Motion**  
Static closure; index and thumb pads meet while hand remains steady ($\|\mathbf{v}\| \le 0.15\,\text{m/s}$).

**Recognition Criteria**  
* Minimum confidence: 0.48
* Velocity: Stationary or low drift ($\|\mathbf{v}\| \le 0.15\,\text{m/s}$)
* Duration: Sustained pinch $\Delta t \ge 80\,\text{ms}$
* Stability: Fingertip pad distance $\|P_4 - P_8\| \le 0.35 \cdot d_{\text{ref}}$
* Direction: N/A

**State Machine**  
OBSERVING → CANDIDATE → CONFIRMED → ACTIVE (`SELECT` / `FOCUS_NODE` emitted) → RELEASE

**Failure Conditions**  
* Digits balled into fist without index-thumb pad contact.
* Gross hand displacement during pinch initiation (treated as G007 Pinch Drag).

**Developer Notes**  
Pinch confirmation must register within $80\,\text{ms}$ to ensure zero perceptual latency, locking the selection coordinate to the initiation anchor.

---

## G007 — Pinch Drag & Hold

**Category**  
Motion / Sustained

**Purpose**  
Physically grabs and repositions any 3D node (parent or sub-node) across the spatial plane or globe surface.

**Finger Anatomy**  
* Thumb: F008 (Contact with Index)
* Index: F008 (Contact with Thumb)
* Middle: F002 or F003
* Ring: F002
* Little: F002

**Palm Orientation**  
Semi-pronated or facing screen.

**Motion**  
Hand moves through space ($\|\mathbf{d}\| > 0.015\,\text{NDC}$) while maintaining F008 pinch contact.

**Recognition Criteria**  
* Minimum confidence: 0.48
* Velocity: Tracking velocity ($0.02 \le \|\mathbf{v}\| \le 0.80\,\text{m/s}$)
* Duration: Continuous tracking until pinch release
* Stability: Continuous pad proximity $\|P_4 - P_8\| \le 0.38 \cdot d_{\text{ref}}$
* Direction: Unconstrained spatial translation vector $(dx, dy)$

**State Machine**  
ACTIVE (Pinch Confirmed on Node) → ACTIVE (Continuous `TRANSLATE_NODE`) → RELEASE (Anchor Node)

**Failure Conditions**  
* Fingers separate ($\|P_4 - P_8\| > 0.42 \cdot d_{\text{ref}}$).
* Second hand enters volume (may trigger bimanual scale).

**Developer Notes**  
Once `heldNode` is locked, the node strictly tracks hand displacement even if the physical cursor temporarily leaves the node radius. Moving open palm anchors the node.

---

## G008 — Closed Fist World Grab

**Category**  
Static / Sustained

**Purpose**  
Zooms the global camera out to maximum world overview distance when in empty space; locks spatial translation.

**Finger Anatomy**  
* Thumb: F002 (Folded over fingers)
* Index: F002 (Curled)
* Middle: F002 (Curled)
* Ring: F002 (Curled)
* Little: F002 (Curled)

**Palm Orientation**  
Any orientation.

**Motion**  
M009 Static Hold or deliberate inward fold.

**Recognition Criteria**  
* Minimum confidence: 0.55
* Velocity: Low displacement ($\|\mathbf{v}\| \le 0.12\,\text{m/s}$)
* Duration: Sustained $\ge 120\,\text{ms}$
* Stability: Extension ratios of all 5 digits $\le 0.90$
* Direction: N/A

**State Machine**  
OBSERVING → CANDIDATE → CONFIRMED → ACTIVE (`WORLD_ZOOM_MAX`) → RELEASE

**Failure Conditions**  
* Index finger extended.
* Executed while cursor is locked onto a target node (becomes Node Hold instead of World Zoom).

**Developer Notes**  
Clenched fist represents contraction/pulling back from the hologram, easing the camera to maximum overview distance ($d_{\text{cam}} = 11.2$).

---

## G009 — Fist-to-Spread World Expansion

**Category**  
Motion / Compound

**Purpose**  
Expands the globe/canvas back into close view when transitioning out of a zoomed-out overview state.

**Finger Anatomy**  
Transition sequence: H003 (Closed Fist) $\to$ H001/H002 (Open Palm / Spread).
* Thumb: F002 $\to$ F001/F012
* Digits 2–5: F002 $\to$ F001/F006

**Palm Orientation**  
Forward facing camera.

**Motion**  
Radial expansion of all digits outwards from palm center.

**Recognition Criteria**  
* Minimum confidence: 0.50
* Velocity: Digit expansion rate $\frac{d}{dt}(R_{\text{ext}}) \ge 1.8\,\text{s}^{-1}$
* Duration: Transition occurs over $80\,\text{ms} \le \Delta t \le 300\,\text{ms}$
* Stability: Monotonic increase from $R_{\text{ext}} \le 0.92$ to $\ge 1.15$
* Direction: Radial outward digit vector

**State Machine**  
ACTIVE (`isZoomedOut == true`) → CANDIDATE (Fist release) → ACTIVE (`WORLD_ZOOM_MIN` / Expand) → OBSERVING

**Failure Conditions**  
* Executed when system is not in zoomed-out state.
* Transition directly into pointing or pinch.

**Developer Notes**  
This gesture directly solves the ergonomic problem of returning from an overview zoom. The user simply opens or spreads their hand after making a fist.

---

## G010 — Open Palm Release & Anchor

**Category**  
Static / State Transition

**Purpose**  
Gracefully terminates active manipulation, anchors held nodes in their current spatial coordinate, and freezes globe motion with zero drift.

**Finger Anatomy**  
* Thumb: F001 or F003
* Index: F001
* Middle: F001
* Ring: F001
* Little: F001

**Palm Orientation**  
Facing display or relaxed forward.

**Motion**  
Digits open; palm velocity stabilizes below $0.05\,\text{m/s}$.

**Recognition Criteria**  
* Minimum confidence: 0.48
* Velocity: Low residual velocity ($\|\mathbf{v}\| \le 0.05\,\text{m/s}$)
* Duration: $\Delta t \ge 60\,\text{ms}$
* Stability: Open palm extension ratios $\ge 1.10$
* Direction: N/A

**State Machine**  
ACTIVE → RELEASE → IDLE / OBSERVING

**Failure Conditions**  
* Movement velocity remaining high (evaluated as Swipe).
* Pinch contact sustained.

**Developer Notes**  
The release gesture is the critical "hand brake" of Gestura. Upon entry, all velocity vectors snap to zero to prevent inertial drift when the user disengages.

---

## G011 — Air Tap

**Category**  
Motion

**Purpose**  
Rapid forward-and-back index finger poke to activate 2D/3D interface buttons without full pinch contact.

**Finger Anatomy**  
* Thumb: F003
* Index: F001 (Forward pointing)
* Middle: F002
* Ring: F002
* Little: F002

**Palm Orientation**  
Facing screen.

**Motion**  
Index tip $P_8$ moves rapidly forward along $Z-$ toward display by $\ge 0.04\,\text{m}$, then retracts within $150\,\text{ms}$.

**Recognition Criteria**  
* Minimum confidence: 0.55
* Velocity: Index tip peak velocity $v_z \le -0.25\,\text{m/s}$
* Duration: $60\,\text{ms} \le \Delta t \le 180\,\text{ms}$
* Stability: Palm center $P_{\text{palm}}$ remains stationary ($v_{\text{palm}} \le 0.08\,\text{m/s}$)
* Direction: $Z-$ alignment $\ge 80\%$

**State Machine**  
OBSERVING → CANDIDATE → CONFIRMED (`TAP_CLICK` emitted) → OBSERVING

**Failure Conditions**  
* Whole hand pushes forward (classified as M005 Push).
* Palm moves downward during tap.

**Developer Notes**  
Requires tracking isolation: index tip velocity must be measured relative to the palm center to filter out gross arm jitter.

---

## G012 — Push Dwell Confirm

**Category**  
Motion / Compound

**Purpose**  
Secondary confirmation for high-stakes actions (delete, submit, modal accept) without requiring pinch.

**Finger Anatomy**  
* Digits 1–5: H001 (Open Palm Neutral)

**Palm Orientation**  
Flat facing camera ($\mathbf{n}_{\text{palm}} \cdot \mathbf{v}_{\text{cam}} \le -0.70$).

**Motion**  
M005 Push Forward followed immediately by M009 Static Hold for $500\,\text{ms}$.

**Recognition Criteria**  
* Minimum confidence: 0.60
* Velocity: Initial forward push $v_z \le -0.15\,\text{m/s}$, followed by $\|\mathbf{v}\| \le 0.025\,\text{m/s}$
* Duration: $500\,\text{ms}$ stationary dwell
* Stability: Spatial jitter radius $< 0.015\,\text{screen units}$
* Direction: $Z-$ thrust

**State Machine**  
OBSERVING → CANDIDATE → ACTIVE (Progress indicator fills) → CONFIRMED → RELEASE

**Failure Conditions**  
* Hand pulls back before dwell timer expires.
* Fingers curl or drift off target.

**Developer Notes**  
Provides a tactile-feeling spatial confirmation. Visual feedback must render a circular progress bar filling during the 500 ms dwell.

---

## G013 — Thumbs Up Approval

**Category**  
Static

**Purpose**  
Confirms modal dialogs, accepts incoming voice transcription, or signals positive feedback.

**Finger Anatomy**  
H007 Thumbs Up pose.

**Palm Orientation**  
Lateral.

**Motion**  
Static hold; stationary in space.

**Recognition Criteria**  
* Minimum confidence: 0.60
* Velocity: $\|\mathbf{v}\| \le 0.10\,\text{m/s}$
* Duration: Sustained $\Delta t \ge 250\,\text{ms}$
* Stability: Thumb vector aligned with $+Y$
* Direction: Vertical upward thumb axis

**State Machine**  
OBSERVING → CANDIDATE → CONFIRMED (`ACTION_ACCEPT`) → RELEASE

**Failure Conditions**  
* Thumb tilted downward or horizontally.
* Dynamic movement during pose.

**Developer Notes**  
Non-spatial semantic gesture. Bypasses raycasting coordinates and targets the active modal overlay.

---

## G014 — Thumbs Down Dismissal

**Category**  
Static

**Purpose**  
Rejects modal dialogs, declines incoming calls, or cancels prompt operations.

**Finger Anatomy**  
H008 Thumbs Down pose.

**Palm Orientation**  
Lateral.

**Motion**  
Static hold.

**Recognition Criteria**  
* Minimum confidence: 0.60
* Velocity: $\|\mathbf{v}\| \le 0.10\,\text{m/s}$
* Duration: Sustained $\Delta t \ge 250\,\text{ms}$
* Stability: Thumb vector aligned with $-Y$
* Direction: Vertical downward thumb axis

**State Machine**  
OBSERVING → CANDIDATE → CONFIRMED (`ACTION_DECLINE`) → RELEASE

**Failure Conditions**  
* Hand moving rapidly down (Swipe Down ambiguity).
* Non-fist thumb posture.

**Developer Notes**  
Semantic rejection companion to G013.

---

## G015 — V-Sign Shortcut / Macro

**Category**  
Static

**Purpose**  
Contextual macro trigger: splits canvas, toggles wireframe rendering, or invokes telemetry heads-up display (HUD).

**Finger Anatomy**  
H009 Peace / V-Sign pose.

**Palm Orientation**  
Forward facing camera.

**Motion**  
Static dwell.

**Recognition Criteria**  
* Minimum confidence: 0.58
* Velocity: $\|\mathbf{v}\| \le 0.10\,\text{m/s}$
* Duration: Sustained $\ge 300\,\text{ms}$
* Stability: Clean separation between Index and Middle tips ($\ge 18^\circ$)
* Direction: N/A

**State Machine**  
OBSERVING → CANDIDATE → CONFIRMED (`TOGGLE_HUD` / `SPLIT_VIEW`) → RELEASE

**Failure Conditions**  
* Palm facing backward toward user face.
* Digits moving in scissors-cutting action.

**Developer Notes**  
Reserved for expert modal toggling. Easy to learn, highly distinctive geometric signature.

---

## G016 — OK Lock / Anchor

**Category**  
Static / Self-Contact

**Purpose**  
Freezes the current scene coordinates, locks node editing against accidental changes, or acknowledges system alerts.

**Finger Anatomy**  
H010 OK Ring pose.

**Palm Orientation**  
Facing camera.

**Motion**  
Static dwell.

**Recognition Criteria**  
* Minimum confidence: 0.60
* Velocity: $\|\mathbf{v}\| \le 0.08\,\text{m/s}$
* Duration: Sustained $\ge 350\,\text{ms}$
* Stability: Index-thumb ring closed; digits 3–5 extended
* Direction: N/A

**State Machine**  
OBSERVING → CANDIDATE → CONFIRMED (`LOCK_CANVAS`) → RELEASE

**Failure Conditions**  
* Digits 3–5 curled into palm.

**Developer Notes**  
Signals deliberate lock state. Cannot be triggered by involuntary hand transitions.

---

## G017 — Quick Flick Dismiss

**Category**  
Motion

**Purpose**  
Rapidly tosses floating cards, notifications, or modal dialogs off-screen.

**Finger Anatomy**  
* Digits 1–5: Open or semi-extended.

**Palm Orientation**  
Any.

**Motion**  
M010 Directional Flick: Extremely rapid lateral stroke with immediate wrist-snap deceleration.

**Recognition Criteria**  
* Minimum confidence: 0.65
* Velocity: Peak velocity $\|\mathbf{v}\| \ge 0.45\,\text{m/s}$, acceleration $\ge 3.0\,\text{m/s}^2$
* Duration: Ultra-short impulse $40\,\text{ms} \le \Delta t \le 120\,\text{ms}$
* Stability: Net displacement $0.025 \le \|\mathbf{d}\| \le 0.065\,\text{screen units}$
* Direction: Vector alignment with target dismiss direction

**State Machine**  
ACTIVE → CONFIRMED (`DISMISS_OBJECT`) → OBSERVING

**Failure Conditions**  
* Sustained stroke lasting $> 180\,\text{ms}$ (classified as standard G001/G002 Swipe).

**Developer Notes**  
Flick gestures are differentiated from standard swipes by their high acceleration spike and short displacement tail.

---

## G018 — Axial Clockwise Dial

**Category**  
Motion / Rotational

**Purpose**  
Incrementally increments continuous numerical parameters (audio volume, lighting intensity, time scrubbing).

**Finger Anatomy**  
* H004 Index Point or H005 Precision Pinch.

**Palm Orientation**  
Facing screen.

**Motion**  
M007 Circle Clockwise: Circular orbit around fixed center point.

**Recognition Criteria**  
* Minimum confidence: 0.55
* Velocity: $\omega \ge 2.2\,\text{rad/s}$
* Duration: Cumulative continuous rotation
* Stability: Orbit radius $0.04 \le r \le 0.12\,\text{screen units}$
* Direction: Clockwise ($+Z$ normal rotation)

**State Machine**  
CONFIRMED → ACTIVE (Streaming $\Delta \theta$ values) → RELEASE

**Failure Conditions**  
* Radial center point drifting faster than $0.06\,\text{m/s}$.

**Developer Notes**  
Streams continuous delta radians. Client applications map delta radians to parameter increments ($\Delta P = k \cdot \Delta \theta$).

---

# Part VII: Level 4 — Two-Hand Cooperative Gestures (T001–T099)

Two-hand interactions engage when two valid hand skeletons ($H_{\text{primary}}, H_{\text{modifier}}$) are tracked simultaneously.

### Rules of Engagement
1. **Dominant Hand Selection**: The hand with higher intent score $E_{\text{hand}}$ acts as Primary; the secondary hand acts as Modifier.
2. **Single-Hand Gating**: The Modifier hand **never** triggers unimanual swipes or selections while dual-hand mode is active.
3. **Midpoint & Intersite Metrics**:
   * Midpoint: $\mathbf{M} = \frac{P_{\text{palm}, 1} + P_{\text{palm}, 2}}{2}$
   * Distance: $D = \|P_{\text{palm}, 2} - P_{\text{palm}, 1}\|$
   * Connecting Angle: $\theta = \text{atan2}(y_2 - y_1, x_2 - x_1)$

---

## T001 — Bimanual Radial Expand (Apart)

**Category**  
Two-Hand Motion

**Purpose**  
Scales the global 3D world, zooms in on the spatial canvas, or increases object bounding volume.

**Finger Anatomy**  
Both hands in H001 (Open Palm Neutral), H002 (Spread), or H005 (Pinch).

**Palm Orientation**  
Both palms facing camera or angled symmetrically toward each other.

**Motion**  
Palms move apart laterally or radially: $\frac{dD}{dt} \ge 0.08\,\text{m/s}$.

**Recognition Criteria**  
* Minimum confidence: 0.60
* Rate of separation: $\Delta D \ge +0.020\,\text{screen units}$ per evaluation window
* Hand synchronization: Opposing velocity vectors ($\mathbf{v}_1 \cdot \mathbf{v}_2 \le -0.40$)
* Scale factor: $S = \min\left(1.25, 1.0 + |\Delta D| \cdot k_{\text{zoom}}\right)$

**State Machine**  
OBSERVING → ACTIVE (Streaming `delta_scale > 1.0`) → RELEASE

**Failure Conditions**  
* Both hands moving in the same direction (triggers T004 Translation).
* One hand stationary while the other moves casually.

**Developer Notes**  
Provides fluid continuous world zoom. The distance ratio $D_t / D_{t-1}$ directly scales the visualizer camera distance.

---

## T002 — Bimanual Radial Contract (Together)

**Category**  
Two-Hand Motion

**Purpose**  
Shrinks the global 3D world, zooms out on canvas, or contracts object bounding volume.

**Finger Anatomy**  
Both hands in H001, H002, or H005.

**Palm Orientation**  
Both palms facing camera or symmetrically angled.

**Motion**  
Palms move together: $\frac{dD}{dt} \le -0.08\,\text{m/s}$.

**Recognition Criteria**  
* Minimum confidence: 0.60
* Rate of contraction: $\Delta D \le -0.020\,\text{screen units}$
* Hand synchronization: Opposing inward vectors ($\mathbf{v}_1 \cdot \mathbf{v}_2 \le -0.40$)
* Scale factor: $S = \max\left(0.75, 1.0 - |\Delta D| \cdot k_{\text{zoom}}\right)$

**State Machine**  
OBSERVING → ACTIVE (Streaming `delta_scale < 1.0`) → RELEASE

**Failure Conditions**  
* Inter-palm distance $< 0.12\,\text{m}$ (collision / landmark confusion threshold).

**Developer Notes**  
Inverse operation of T001. Clamped at lower bound to prevent geometric inversion.

---

## T003 — Bimanual Bounded Axial Rotation

**Category**  
Two-Hand Motion

**Purpose**  
Rotates the 3D world or active object directly 1:1 with hand motion, bounded edge-to-edge horizontally and top-to-bottom vertically (NOT a 360° spin).

**Finger Anatomy**  
Both hands in H001 (Open Palm) or H003 (Fist) or H005 (Pinch).

**Palm Orientation**  
Both palms facing camera.

**Motion**  
Angle of line connecting Palm 1 and Palm 2 rotates ($\Delta \theta$), or hands translate differentially.

**Recognition Criteria**  
* Minimum confidence: 0.60
* Angular displacement: $|\Delta \theta| \ge 0.015\,\text{rad}$
* Bounded rotation limits:
  * Horizontal Yaw: Clamped edge-to-edge $[ -0.95\pi, +0.95\pi ]$
  * Vertical Pitch: Clamped top-to-bottom $[ -\pi / 2.2, +\pi / 2.2 ]$
  * Roll: Clamped $[ -\pi / 3.0, +\pi / 3.0 ]$

**State Machine**  
OBSERVING → ACTIVE (Streaming `applyBoundedHandRotation`) → RELEASE

**Failure Conditions**  
* Fast unimanual swipe detected (swipes handle 360° unbounded spin).

**Developer Notes**  
CRITICAL RULE: Two-hand rotation is strictly bounded edge-to-edge. It NEVER triggers an unbounded 360° spin. When the hands stop moving, the rotation stops instantly. 360° spinning is exclusively reserved for unimanual swipe slaps (G001–G004).

---

## T004 — Bimanual Spatial Midpoint Translation

**Category**  
Two-Hand Motion

**Purpose**  
Pans the entire 3D canvas or world coordinate frame in the $XY$ plane.

**Finger Anatomy**  
Both hands in open or steady posture.

**Palm Orientation**  
Both palms facing camera.

**Motion**  
Midpoint $\mathbf{M} = \frac{P_1 + P_2}{2}$ translates through screen space while inter-palm distance $D$ remains stable ($|\Delta D| \le 0.015$).

**Recognition Criteria**  
* Minimum confidence: 0.60
* Co-directional velocity: $\mathbf{v}_1 \cdot \mathbf{v}_2 \ge +0.70$
* Translation displacement: $\|\Delta \mathbf{M}\| \ge 0.008\,\text{screen units}$

**State Machine**  
OBSERVING → ACTIVE (Streaming `applyTranslationDelta`) → RELEASE

**Failure Conditions**  
* Distance $D$ changing significantly (evaluated as T001/T002 Scale).

**Developer Notes**  
Provides rock-steady bimanual panning. Hand tracking jitter is naturally halved by averaging the two palm positions.

---

## T005 — Dual Grab World Freeze

**Category**  
Two-Hand Static

**Purpose**  
Instantly freezes all simulation physics, particle motion, and object inertia; enters system emergency pause.

**Finger Anatomy**  
Both hands simultaneously clenched into H003 Closed Fist.

**Palm Orientation**  
Any.

**Motion**  
Static hold; both hands stationary.

**Recognition Criteria**  
* Minimum confidence: 0.65 on both hands concurrently
* Duration: Sustained $\ge 150\,\text{ms}$
* Extension ratios: All 10 digits $R_{\text{ext}} \le 0.90$

**State Machine**  
OBSERVING → ACTIVE (`FREEZE_ALL`) → RELEASE

**Failure Conditions**  
* Only one hand in fist posture.

**Developer Notes**  
The universal "halt" gesture. Immediately clears all angular and translation velocity accumulators.

---

## T006 — Two-Hand Frame / Crop

**Category**  
Two-Hand Compound

**Purpose**  
Creates a rectangular spatial selection box to capture screenshots, crop visual regions, or bound multi-object selections.

**Finger Anatomy**  
Both hands in H013 L-Shape Point, inverted relative to each other:
* Left Hand: L-shape forming upper-left corner (Thumb pointing right, Index pointing down).
* Right Hand: L-shape forming lower-right corner (Thumb pointing left, Index pointing up).

**Palm Orientation**  
Facing screen.

**Motion**  
Static hold after framing gesture.

**Recognition Criteria**  
* Geometric alignment: Index and thumb rays form diagonal corners of a rectangle.
* Confidence $\ge 0.65$ sustained over $\ge 250\,\text{ms}$.

**State Machine**  
CANDIDATE → ACTIVE (Bounding box visible) → CONFIRMED (`EMIT_BOUNDING_BOX`) → RELEASE

**Failure Conditions**  
* Non-planar orientation.

**Developer Notes**  
Classic cinematic framing gesture adapted for spatial user interfaces.

---

## T007 — Primary Point + Secondary Modifier Dial

**Category**  
Two-Hand Asymmetric

**Purpose**  
Dominant hand points at and holds a 3D node; non-dominant hand rotates around its wrist to adjust node parameters (gain, radius, thresholds).

**Finger Anatomy**  
* Primary Hand: H004 Index Point (targeting object).
* Modifier Hand: H005 Precision Pinch or H001 Open Palm rotating via M011 Wrist Roll.

**Palm Orientation**  
Primary: Facing target. Modifier: Lateral.

**Motion**  
Primary hand stationary; Modifier hand executes roll rotation.

**Recognition Criteria**  
* Primary hand $C_{\text{point}} \ge 0.50$ locked on target.
* Modifier hand $|\Delta \theta_{\text{roll}}| \ge 0.05\,\text{rad}$.

**State Machine**  
ACTIVE (`NODE_PARAM_MODULATE`) → RELEASE

**Failure Conditions**  
* Primary hand leaves target node.

**Developer Notes**  
Guiard asymmetric interaction exemplar: Left hand holds reference frame, Right hand adjusts value.

---

## T008 — Primary Pinch + Secondary Stabilize

**Category**  
Two-Hand Asymmetric

**Purpose**  
Stabilizes fine-precision spatial node translation in noisy webcam environments.

**Finger Anatomy**  
* Primary Hand: H005 Precision Pinch (holding node).
* Modifier Hand: H001 Open Palm held flat below the primary hand ($Y_{\text{mod}} > Y_{\text{prim}}$).

**Palm Orientation**  
Primary: Facing target. Modifier: Upward palmar tray ($\mathbf{n}_{\text{palm}} \cdot \mathbf{u}_{\text{floor}} \le -0.60$).

**Motion**  
Modifier hand acts as a stationary virtual platform; primary hand translates.

**Recognition Criteria**  
* Jitter damping factor increased by $3\times$ when stabilizing modifier palm is present below primary hand.

**State Machine**  
ACTIVE (`TRANSLATE_STABILIZED`) → RELEASE

**Failure Conditions**  
* Modifier hand leaves tracking volume.

**Developer Notes**  
Biomechanical dampening: humans naturally stabilize their working hand by positioning their non-dominant hand underneath as a psychological shelf.

---

## T009 — Dual Palm Push (Shield / Home)

**Category**  
Two-Hand Motion

**Purpose**  
Pushes away all active holographic windows; returns spatial desktop to clean Home state.

**Finger Anatomy**  
Both hands in H001 Open Palm Neutral.

**Palm Orientation**  
Both palms flat facing camera ($\mathbf{n} \cdot \mathbf{v}_{\text{cam}} \le -0.70$).

**Motion**  
Both hands simultaneously execute M005 Push Forward by $\ge 0.08\,\text{m}$.

**Recognition Criteria**  
* Concurrent push forward velocity $v_{z, 1} \le -0.22\,\text{m/s}$ and $v_{z, 2} \le -0.22\,\text{m/s}$.
* Hand synchronization: $Z$ velocity correlation $\ge 0.80$.

**State Machine**  
OBSERVING → CONFIRMED (`SYSTEM_GO_HOME`) → RELEASE

**Failure Conditions**  
* Asymmetric or single-hand push.

**Developer Notes**  
The universal "clear space" or "shield" push gesture.

---

## T010 — Clapped Hands Reset

**Category**  
Two-Hand Collision / Event

**Purpose**  
Recalibrates coordinate system, re-centers camera focus to origin, and clears all user error states.

**Finger Anatomy**  
Both hands open.

**Palm Orientation**  
Palms facing each other.

**Motion**  
Palms rapidly meet at spatial midpoint ($\|\mathbf{v}_{\text{rel}}\| \ge 0.45\,\text{m/s}$, final distance $D \le 0.04\,\text{m}$).

**Recognition Criteria**  
* Rapid approach velocity followed by sudden halt and contact geometry.
* Confidence $\ge 0.70$.

**State Machine**  
CONFIRMED (`CALIBRATE_SYSTEM_ORIGIN`) → RELEASE

**Failure Conditions**  
* Hands crossing without collision.

**Developer Notes**  
High physical certainty gesture. Almost impossible to trigger accidentally.

---

# Part VIII: Finger Micro-Gestures (FM001–FM099)

Finger micro-gestures operate at the sub-centimeter scale, allowing subtle interaction with minimal arm movement when the user’s hand is resting on a desk or armrest.

### FM001 — Thumb-to-Index Tap (Air Click)
* **Anatomical Definition**: Thumb tip $P_4$ taps Index PIP joint or index tip pad $P_8$ with contact duration $\le 90\,\text{ms}$, followed by immediate retraction.
* **Recognition Criteria**: Contact distance $\le 0.25 \cdot d_{\text{ref}}$, velocity reversal $\ge 1.2\,\text{m/s}^2$.
* **Emitted Command**: `MICRO_CLICK_PRIMARY`.

### FM002 — Thumb-to-Middle Tap (Secondary Click)
* **Anatomical Definition**: Thumb tip $P_4$ taps Middle fingertip pad $P_{12}$ while Index remains extended.
* **Recognition Criteria**: $\|P_4 - P_{12}\| \le 0.25 \cdot d_{\text{ref}}$ while $\|P_4 - P_8\| \ge 0.40 \cdot d_{\text{ref}}$.
* **Emitted Command**: `MICRO_CLICK_SECONDARY` (Context Menu).

### FM003 — Double Pinch Tap
* **Anatomical Definition**: Two successive FM001 thumb-index pinch contacts within $350\,\text{ms}$.
* **Recognition Criteria**: Dual contact pulses with inter-tap duration $80\,\text{ms} \le \Delta t_{\text{gap}} \le 280\,\text{ms}$.
* **Emitted Command**: `MICRO_DOUBLE_CLICK` / `EXPAND_CLUSTER`.

### FM004 — Index Knuckle Flexion (Trigger)
* **Anatomical Definition**: Index finger flexes at PIP joint while thumb remains static and isolated.
* **Recognition Criteria**: Angular rate $\dot{\theta}_{\text{PIP}}(\text{Index}) \ge 3.0\,\text{rad/s}$ while thumb velocity is zero.
* **Emitted Command**: `MICRO_TRIGGER_PRESS`.

### FM005 — Finger Fan / Progressive Spread
* **Anatomical Definition**: Sequential extension of fingers from pinky to index (guitar arpeggio motion).
* **Recognition Criteria**: Time-staggered extension: $t(\text{Pinky}) < t(\text{Ring}) < t(\text{Middle}) < t(\text{Index})$ across $180\,\text{ms}$.
* **Emitted Command**: `MICRO_REVEAL_LAYERS`.

### FM006 — Ring-Pinky Independent Tuck
* **Anatomical Definition**: Ring and little fingers curl tightly into the palm while index and middle remain fully extended.
* **Recognition Criteria**: $\text{mean}(R_{\text{ext}}(\text{Ring, Pinky})) \le 0.88$ while $\text{mean}(R_{\text{ext}}(\text{Index, Middle})) \ge 1.20$.
* **Emitted Command**: `MICRO_MODE_SHIFT`.

### FM007 — Thumb Swipe Across Index Edge
* **Anatomical Definition**: Thumb tip slides longitudinally along the lateral radial edge of the index finger from PIP to tip.
* **Recognition Criteria**: Continuous contact $\|P_4 - \text{Ray}(P_5 \to P_8)\| \le 0.15 \cdot d_{\text{ref}}$ with monotonic progression.
* **Emitted Command**: `MICRO_CONTINUOUS_SCROLL`.

### FM008 — Little Finger Accent / Pinky Out
* **Anatomical Definition**: Pinky finger abducted laterally outward by $\ge 30^\circ$ while other fingers hold an active pinch or point.
* **Recognition Criteria**: Pinky abduction angle $\ge 28^\circ$ relative to ring finger during H004/H005.
* **Emitted Command**: `MICRO_MODIFIER_SHIFT` (Equivalent to holding Shift/Alt key).

---

# Part IX: Spatial Interaction Principles

```
   ┌────────────────────────────────────────────────────────────┐
   │               GESTURA SPATIAL PIPELINE                     │
   └────────────────────────────────────────────────────────────┘
      │
      ├── 1. HOVER       ──► Magnetic Screen-Space Projection
      │
      ├── 2. SELECTION   ──► Contact Verification & Dwell Gating
      │
      ├── 3. GRAB        ──► 1:1 Kinematic Plane Coupling
      │
      ├── 4. RELEASE     ──► Velocity Decoupling & Motion Freeze
      │
      ├── 5. ZOOM        ──► Node vs World vs Dual-Hand Scaling
      │
      └── 6. ROTATION    ──► Bounded Hand Drag vs Unbounded Slap Spin
```

### 1. Spatial Hover Model
* **Magnetic Screen-Space Projection**: Standard 3D raycasting against tiny sub-nodes is ergonomically unusable over webcam feeds. Gestura projects target node 3D coordinates into 2D Normalized Device Coordinates (NDC). If the screen-space distance between cursor $(x_{\text{ndc}}, y_{\text{ndc}})$ and projected node $(u_{\text{ndc}}, v_{\text{ndc}})$ is within the capture radius ($r_{\text{mag}} = 0.28\,\text{NDC}$), the cursor magnetically snaps to the node.
* **Normal Occlusion Gating**: A node mounted on a 3D globe is only eligible for hover if its surface normal points toward the camera viewpoint ($\mathbf{n}_{\text{node}} \cdot \mathbf{v}_{\text{cam}} > 0.02$). Back-face nodes are completely invisible to raycasting.

### 2. Selection Mechanics
* **Zero-Drift Anchor**: When a user pinches (G006), the physical cursor coordinate is instantaneously frozen at the moment of contact. Hand displacement during the initial $60\,\text{ms}$ of contact is absorbed as click-settle to prevent moving the cursor off the target during a click.
* **Cluster Expansion**: Selecting a parent node on the 3D globe transitions the visualizer into Focus Mode, flying the camera forward to focal distance ($d = 4.2$) and expanding child sub-nodes outward along energy tether vectors.

### 3. Grab & Direct Manipulation
* **Plane Locking**: Grabbing an object binds its spatial anchor frame to the hand's NDC coordinates with a 1:1 translation mapping.
* **Sub-Node vs. Parent Node Kinematics**:
  * *Sub-Nodes*: Translated within the local 3D cluster plane ($dx \cdot 2.5, dy \cdot 2.5$). Energy line buffer geometry dynamically recalculates every frame to keep the connecting holographic link tethered from parent to child.
  * *Parent Nodes*: Translated along the spherical surface of the globe by updating latitude and longitude coordinates ($\Delta \text{lon} = dx \cdot 55^\circ, \Delta \text{lat} = dy \cdot 55^\circ$).

### 4. Release & Motion Freezing
* **Drift Suppression**: In traditional systems, releasing a hand leaks trailing velocity, causing objects to shoot off-screen. In Gestura, entering `RELEASE` or `OPEN_PALM` invokes an immediate motion freeze:
  $$\mathbf{v}_{\text{target}} \gets \mathbf{0}, \quad \omega_{\text{target}} \gets \mathbf{0}, \quad \mathbf{P}_{\text{target}} \gets \mathbf{P}_{\text{current}}$$
* **Elastic Settle**: Targets do not snap to historical frames; current positions are held instantaneously, stopping all lerp drift.

### 5. Zoom Hierarchy
Gestura defines three distinct, non-conflicting zoom modalities:
1. **Focus Node Zoom**: Triggered by pinching on a node. Smoothly flies the camera to focal distance ($d = 4.2$), orienting camera look-at to the node center.
2. **Unimanual Overview Zoom (Closed Fist / Spread)**:
   * Closed fist in empty space (G008) sets target camera distance to maximum overview ($d = 11.2$) and sets `isZoomedOut = true`.
   * Releasing the fist to open palm (G009 / G010) immediately expands the globe back to active view ($d = 5.2$).
3. **Bimanual Continuous Zoom (T001 / T002)**: Hands moving apart/together continuously scale camera distance between close boundary ($d = 3.8$) and overview boundary ($d = 11.5$).

### 6. Rotation Mechanics: Bounded vs. Unbounded
Gestura strictly segregates rotation physics:
* **Two-Hand Rotation (T003)**: Strictly **bounded** 1:1 direct tracking. The globe rotates only while the hands are moving, stopping cleanly at boundary stops ($|\theta_{\text{yaw}}| \le 0.95\pi, |\theta_{\text{pitch}}| \le \pi/2.2$). Two hands NEVER produce an unbounded continuous spin.
* **Unimanual Swiping (G001–G004)**: Velocity-proportional **unbounded 360° spin**. Swipe speed directly determines initial angular velocity ($\omega_0 = \Delta \theta \cdot 0.016$), which decelerates smoothly under inertial friction ($\omega_{t+1} = \omega_t \times 0.965$).

---

# Part X: Context Mapping Specification (C001–C999)

Gestura adheres to strict semantic decoupling. Gestures never directly execute operating-system-level actions; they emit abstract spatial tokens (`SpatialCommand`). Applications map tokens to local domain actions.

```
┌──────────────────┐       ┌──────────────────────┐       ┌────────────────────────┐
│ PHYSICAL GESTURE │ ────► │ ABSTRACT HMI COMMAND │ ────► │ DOMAIN APPLICATION     │
│ e.g., G001 Swipe │       │ e.g., ROTATE_OBJECT  │       │ e.g., Spin 3D Globe    │
└──────────────────┘       └──────────────────────┘       └────────────────────────┘
```

## Canonical Context Mapping Matrix

| ID | Abstract Spatial Command | Desktop Window Manager | Holographic 3D Globe | Media / Photo Gallery | CAD / Spatial Modeler |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C001** | `SPATIAL_SWIPE_LEFT` | Switch to Next Virtual Desktop | 360° Horizontal Spin Counter-Clockwise | Advance to Next Photo (Slide Left) | Orbit Camera 90° Clockwise |
| **C002** | `SPATIAL_SWIPE_RIGHT` | Switch to Previous Virtual Desktop | 360° Horizontal Spin Clockwise | Return to Previous Photo (Slide Right) | Orbit Camera 90° Counter-Clockwise |
| **C003** | `SPATIAL_SWIPE_UP` | Invoke Mission Control / App Exposé | 360° Vertical Pitch Upward | Open Info / Metadata Drawer | View Top Orthographic Projection |
| **C004** | `SPATIAL_SWIPE_DOWN` | Show Desktop (Minimize Windows) | 360° Vertical Pitch Downward | Dismiss Fullscreen / Return to Grid | View Bottom Orthographic Projection |
| **C005** | `SPATIAL_HOVER` | Cursor Tracking & Window Glow Focus | Cursor Raycast & Magnetic Node Hover | Thumbnail Highlight / Loupe Preview | Entity Highlighting & Snapping |
| **C006** | `SPATIAL_SELECT` | Primary Left Click / Activate | Focus Node & Expand Sub-Cluster | Open Photo Fullscreen | Select Vertex / Edge / Face |
| **C007** | `SPATIAL_HOLD_TRANSLATE` | Drag Active Window / Move File | Reposition Node Across Globe Surface | Pan Zoomed Photo Viewport | Translate Selected Geometry in Plane |
| **C008** | `WORLD_ZOOM_OVERVIEW` | Zoom Out to Multi-Monitor Grid | Easing Camera to World Distance (11.2) | Zoom Out to Month/Year Grid | Zoom to Extents / Model Bounds |
| **C009** | `WORLD_ZOOM_EXPAND` | Restore Window Workspace | Easing Camera to Active View (5.2) | Zoom into Selected Collection | Restore Working Camera Distance |
| **C010** | `BIMANUAL_NAV_ROTATE` | Window Roll / Tilted Canvas | Bounded Hand Rotation (Edge-to-Edge) | Rotate Image Canvas in $XY$ | Continuous Bounded 3D Orbit |
| **C011** | `BIMANUAL_NAV_SCALE` | Global UI Scaling / Zoom | Continuous World Camera Scaling | Continuous Image Pinch-Zoom | Scale Model Bounding Box |
| **C012** | `BIMANUAL_NAV_TRANSLATE`| Pan Multi-Monitor Canvas | Pan Globe Origin in Screen Space | Pan Image Canvas | Pan Model Workplane |
| **C013** | `INTERACTION_RELEASE` | Mouse Up / Drop File / Commit | Anchor Held Node & Freeze Inertia | Release Pan / Settle Canvas | Commit Geometric Transformation |
| **C014** | `SYSTEM_PAUSE_FREEZE` | Lock Screen / Pause Media | Freeze All Globe Particle Simulation | Pause Video Playback | Pause Physics Simulation |
| **C015** | `DISMISS_TRANSIENT` | Close Notification Toast / Popup | Collapse Sub-Node Cluster | Dismiss Photo Lightbox | Deselect All Geometry |
| **C016** | `MODAL_ACCEPT` | Confirm Dialog / Press OK | Commit Cluster Edits | Favorite / Star Active Media | Accept Extrude / Boolean Operation |
| **C017** | `MODAL_CANCEL` | Dismiss Dialog / Press Cancel | Revert Cluster Node Positions | Delete Active Media | Cancel Active Operation |
| **C018** | `TOGGLE_HUD_OVERLAY` | Toggle System Activity Monitor | Toggle Telemetry Debug Overlay (D-Key) | Toggle EXIF Data Inspector | Toggle Wireframe / Shading Mode |
| **C019** | `SYSTEM_GO_HOME` | Minimize All Windows to Desktop | Reset Globe Origin & Distance to 6.8 | Return to Root Album Library | Reset to Default Isometric View |
| **C020** | `SYSTEM_RECALIBRATE` | Recalibrate Mouse Driver | Recalibrate Webcam Coordinates & Plane | Reset Zoom & Rotation to Neutral | Re-center World Origin to Grid |

---

# Part XI: Contextual Voice Integration

Gestura implements a multimodal interaction paradigm where Vision and Voice form an interdependent symbiosis. Vision provides high-bandwidth **spatial precision** (where, what, coordinates), while Voice provides semantic **intent, identity, and modality switching** (who, why, command).

```
   ┌────────────────────────────────────────────────────────────┐
   │             MULTIMODAL FUSION ARCHITECTURE                 │
   └────────────────────────────────────────────────────────────┘
         │                                              │
         ▼                                              ▼
   [Vision Stream]                              [Audio Stream]
   • Spatial Coordinates                        • Intent Semantic
   • Pointing Vector                            • Target Selection
   • Active Focus Object                        • State Modifiers
         │                                              │
         └──────────────────────┬───────────────────────┘
                                │
                                ▼
                   [Multimodal Intent Arbiter]
                                │
                                ▼
                     Unified Spatial Action
```

### 1. Spatial Co-Referencing ("Deictic Fusion")
Human communication naturally combines pointing gestures with deictic pronouns ("this", "that", "there"):
* User points at Node Alpha (`G005 Point & Hover`) and speaks: *"Expand this cluster."*  
  $\to$ System fuses the active raycast target ID with the spoken verb, immediately executing `FOCUS_NODE` without requiring an explicit pinch gesture.
* User pinches a sub-node (`G007 Pinch Drag & Hold`) and speaks: *"Move that over here."*  
  $\to$ System maintains the node grip and drops it precisely at the spoken arrival timestamp coordinate.

### 2. Attention & Wake Philosophy
Gestura avoids intrusive global wake-words ("Hey Gestura") during active spatial engagement:
* **Gaze & Pose Wake**: If the user is actively pointing at the screen or has both hands within the interaction volume, the audio system enters hot-mic listening mode automatically.
* **Presence Declarations**: When entering the room or re-engaging tracking, natural spoken phrases re-establish spatial attention:
  * *"I'm over here."* $\to$ Directs tracking pipeline to optimize ROI bounding box toward user voice direction.
  * *"Wake up."* $\to$ Transitions system from `IDLE` sleep to `OBSERVING`.
  * *"Focus on me."* $\to$ In multi-person environments, locks dominant user tracking to the person speaking.

### 3. Voice Context Modifiers
Voice acts as an instantaneous state modifier for active gestures:
* Holding two-hand rotation (T003) + *"Snap to grid"* $\to$ Quantizes rotation angles to nearest $15^\circ$ increments.
* Swiping left (G001) + *"Fast"* $\to$ Triples the angular momentum multiplier.
* Holding node + *"Duplicate"* $\to$ Clones the held node and mounts a replica at the anchor coordinate.
* Open palm release (G010) + *"Discard changes"* $\to$ Reverts translated node to its original base coordinate.

### 4. Multimodal Conflict Arbitration
When vision and voice provide conflicting signals:
1. **Safety Dominance**: Explicit voice commands (*"Stop"*, *"Cancel"*, *"Freeze"*) unconditionally override all vision tracking streams.
2. **Spatial Dominance**: Vision coordinates unconditionally override spoken coordinates (spoken coordinates are inherently vague).
3. **Temporal Windowing**: Spoken deictic pronouns ("this") are matched against the visual hover trajectory within a $[-300\,\text{ms}, +200\,\text{ms}]$ temporal coincidence window.

---

# Appendix: Developer Integration Guide & Compliance Checklist

Any software library, driver, or visualizer integrating with Gestura must conform to the following compliance checklist:

- [ ] **Decoupling**: Visualizer consumes abstract `SpatialCommand` structures; it contains zero hardcoded landmark index checks.
- [ ] **No Single-Frame Commits**: All static gestures undergo temporal evidence accumulation through an FSM equivalent to Gestura's 6-state model.
- [ ] **Modifier Hand Isolation**: If two hands are visible, the non-dominant hand is blocked from emitting unimanual swipes or clicks.
- [ ] **Bounded Bimanual Navigation**: Dual-hand rotation must be clamped edge-to-edge; continuous spinning on two hands is rejected.
- [ ] **Momentum Safety**: When entering `RELEASE` or `OPEN_PALM`, target velocities must be zeroed to prevent runaway inertial drift.
- [ ] **Magnetic Screen Targeting**: 3D raycasting must incorporate 2D projected screen-space proximity ($r \ge 0.20\,\text{NDC}$) for sub-centimeter targets.
- [ ] **ID Permanence**: No gesture, pose, or motion ID defined in this Bible may be repurposed or altered in future revisions.

---
**END OF SPECIFICATION — GESTURA GESTURE BIBLE V1.0.0**
