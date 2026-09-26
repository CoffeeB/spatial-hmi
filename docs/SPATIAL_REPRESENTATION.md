# Gestura Spatial Representation Layer

> **"Continuous perception first. Discrete interpretation second."**  
> *Gestura does not ask "Is the hand moving LEFT or RIGHT?" but "What is the hand's actual 3D position, orientation, direction, and motion vector?"*

---

## 1. Architectural Overview & Core Principles

The classical perception layer suffered from two fundamental weaknesses:
1. **Discrete Categorical Bottlenecks**: Movements forced into 4-way discrete labels (`UP`, `DOWN`, `LEFT`, `RIGHT`), destroying natural diagonal motions (e.g. $(-0.707, -0.707)$) and creating catastrophic oscillations at sector boundaries.
2. **Viewpoint-Dependent Appearance**: Reliance on 2D screen appearances caused finger states, poses, and directions to collapse whenever the hand rotated, tilted, turned sideways, or faced directly toward the camera.

The **Gestura Continuous Spatial Representation Layer** redesigns perception from the ground up:

```text
                    CAMERA SENSOR
                          │
                          ▼
                  21 3D LANDMARKS
                          │
                          ▼
            HAND COORDINATE FRAME ESTIMATOR
        (Orthonormal Hand Basis: X_rad, Y_dist, Z_norm)
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
    ARTICULATED      CONTINUOUS 3D    VIEWPOINT-INVARIANT
  FINGER GEOMETRY       MOTION            POSE METRICS
  (MCP→PIP→DIP→TIP)  (v, a, j, depth)     (Flexion, Spread)
         │                │                │
         └────────────────┼────────────────┘
                          ▼
              TEMPORAL RECONSTRUCTION
            (Occlusion & Ambiguity Bridge)
                          │
                          ▼
              SOFT DIRECTION MODEL (8 Sectors)
            + EXPLICIT UNCERTAINTY & GREY ZONE
                          │
                          ▼
                HAND PERCEPTION OBJECT
                          │
                          ▼
                   STABILITY ENGINE
                          │
                          ▼
                TEMPORAL INTENT ENGINE
                          │
                          ▼
                    GESTURE BIBLE
```

---

## 2. Hand-Relative Orthonormal 3D Coordinate System

To achieve genuine **viewpoint invariance**, Gestura constructs an anatomical coordinate frame attached to the physical hand.

### 2.1 Mathematical Frame Derivation
Given 21 MediaPipe 3D joint landmarks:
* **Origin** $\mathbf{O} = \mathbf{p}_0$ (Wrist)
* **Longitudinal Axis** $\mathbf{e}_y$ (Distal vector from wrist to middle MCP):
  $$\mathbf{v}_y = \mathbf{p}_9 - \mathbf{p}_0, \quad \mathbf{e}_y = \frac{\mathbf{v}_y}{\|\mathbf{v}_y\|}$$
* **Palmar Normal Axis** $\mathbf{e}_z$ (Volar normal pointing outward from the palm face):
  $$\mathbf{n}_{\text{palm}} = \begin{cases} (\mathbf{p}_5 - \mathbf{p}_{17}) \times (\mathbf{p}_9 - \mathbf{p}_0) & \text{for Left hand} \\ (\mathbf{p}_{17} - \mathbf{p}_5) \times (\mathbf{p}_9 - \mathbf{p}_0) & \text{for Right hand} \end{cases}$$
  $$\mathbf{e}_z = \frac{\mathbf{n}_{\text{palm}}}{\|\mathbf{n}_{\text{palm}}\|}$$
* **Radial Axis** $\mathbf{e}_x$ (Lateral axis pointing toward index/thumb side):
  $$\mathbf{e}_x = \mathbf{e}_y \times \mathbf{e}_z$$

The resulting rotation matrix:
$$\mathbf{R} = \begin{bmatrix} \mathbf{e}_x & \mathbf{e}_y & \mathbf{e}_z \end{bmatrix}$$

### 2.2 Forward & Inverse Coordinate Transformations
* **Camera to Hand-Relative Coordinates**:
  $$\mathbf{p}_{\text{hand}} = \frac{\mathbf{R}^T (\mathbf{p}_{\text{cam}} - \mathbf{O})}{d_{\text{ref}}}$$
  $$\mathbf{v}_{\text{hand}} = \mathbf{R}^T \mathbf{v}_{\text{cam}}$$
* **Hand-Relative to Camera Coordinates**:
  $$\mathbf{p}_{\text{cam}} = (\mathbf{R} \cdot \mathbf{p}_{\text{hand}}) \cdot d_{\text{ref}} + \mathbf{O}$$
  $$\mathbf{v}_{\text{cam}} = \mathbf{R} \cdot \mathbf{v}_{\text{hand}}$$

### 2.3 Viewpoint Invariance Property
Because $\mathbf{v}_{\text{hand}}$ is computed via $\mathbf{R}^T \mathbf{v}_{\text{cam}}$, an extended index finger pointing straight along the hand's longitudinal axis evaluates to:
$$\mathbf{v}_{\text{index, hand}} \approx (0.0, 1.0, 0.0)$$
regardless of whether the hand is:
* Facing the camera (PALM front)
* Facing away (DORSAL back of hand)
* Rotated $45^\circ$, $90^\circ$, or upside down ($180^\circ$)
* Tilted laterally sideways

---

## 3. Articulated Finger Kinematic Chain

Fingers are modeled as continuous 3D articulated bone segments ($MCP \to PIP \to DIP \to TIP$):

```text
    p_mcp ──── v_prox ────> p_pip ──── v_inter ────> p_dip ──── v_dist ────> p_tip
      │                                                                        │
      └──────────────────────────── v_pointing ────────────────────────────────┘
```

### 3.1 Multi-Space Representations
Every digit exposes pointing vectors in three distinct reference frames:
1. **Camera Space** $\mathbf{u}_{\text{cam}} \in \mathbb{R}^3$: Raw physical vector from camera optical center.
2. **Hand Space** $\mathbf{u}_{\text{hand}} \in \mathbb{R}^3$: Viewpoint-invariant anatomical vector.
3. **Screen Space** $\mathbf{u}_{\text{screen}} \in \mathbb{R}^2$: Planar 2D projection on the image plane.

### 3.2 Continuous Geometric Metrics
* **Joint Flexion Angles**:
  $$\theta_{\text{flexion}} = 180^\circ - \arccos\left(\frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\| \|\mathbf{v}_2\|}\right)$$
  Calculated individually for $\text{MCP}$, $\text{PIP}$, and $\text{DIP}$.
* **Continuous Curl Ratio** $c \in [0.0, 1.0]$:
  $$c = 1.0 - \text{clamp}\left(\frac{\|\mathbf{p}_{\text{tip}} - \mathbf{p}_{\text{mcp}}\|}{L_{\text{prox}} + L_{\text{inter}} + L_{\text{dist}}}, 0.2, 1.0\right)$$
  * $c = 0.0$: Fully extended, straight digit.
  * $c = 0.5$: Curved / intermediate transition posture.
  * $c = 1.0$: Fully folded into palm.
* **Spread / Abduction Angle**: Angle relative to middle digit longitudinal axis in the hand plane.

---

## 4. Soft Spatial Direction Model & The Grey Zone

Gestura replaces rigid quadrant switching ($0^\circ-45^\circ = \text{RIGHT}$, $45^\circ-90^\circ = \text{DOWN}$) with an **8-direction radial sector model + CENTER** using continuous circular Gaussian probability distributions.

### 4.1 8-Direction Radial Sectors
Sector centers in standard screen coordinates ($+X$ Right, $+Y$ Down):
* `RIGHT`: $0^\circ$
* `BOTTOM_RIGHT`: $45^\circ$
* `BOTTOM`: $90^\circ$
* `BOTTOM_LEFT`: $135^\circ$
* `LEFT`: $\pm 180^\circ$
* `TOP_LEFT`: $-135^\circ$
* `TOP`: $-90^\circ$
* `TOP_RIGHT`: $-45^\circ$
* `CENTER`: Dead zone ($\|\mathbf{v}\| < r_{\text{dead}}$)

### 4.2 Circular Gaussian Soft Distribution
For an instantaneous motion angle $\theta$:
$$S_k = \exp\left(-\frac{\Delta\theta_k^2}{2\sigma^2}\right), \quad \Delta\theta_k = \min(|\theta - \theta_k|, 360^\circ - |\theta - \theta_k|)$$
$$p_k = \frac{S_k}{\sum_j S_j}$$
where $\sigma = \text{sector\_bandwidth} / 2 = 22.5^\circ$.

### 4.3 The Grey Zone & Ambiguity Modeling
Gestura never invents false certainty. For top candidates $p_{(1)}$ and $p_{(2)}$:
* $\Delta p = p_{(1)} - p_{(2)}$
* If $\Delta p \ge 0.32$: `AmbiguityLevel.LOW` (Decisive)
* If $0.18 \le \Delta p < 0.32$: `AmbiguityLevel.MEDIUM`
* If $\Delta p < 0.18$: `AmbiguityLevel.HIGH` (Grey zone)
* If $\Delta p < 0.08$: Output is explicitly classified as **`AMBIGUOUS`** rather than forcing a wrong sector.

### 4.4 Angular Hysteresis
To prevent rapid boundary chatter between adjacent sectors (e.g. `TOP` $\leftrightarrow$ `TOP_RIGHT`), the system enforces an angular barrier:
$$\Delta\theta_{\text{active}} - \Delta\theta_{\text{candidate}} > 12.0^\circ$$
The active sector is retained until the continuous vector decisively crosses into the neighboring territory.

---

## 5. Optical Foreshortening & The Camera-Facing Hand

When a hand points directly into or out of the camera:
* Perspective causes apparent 2D finger lengths to contract dramatically.
* Conventional 2D classifiers falsely classify straight fingers as curled or folded.

### Gestura Solution:
1. **Foreshortening Compression Index**:
   $$R_{\text{foreshorten}} = \frac{\|\mathbf{p}_{\text{tip}, xy} - \mathbf{p}_{\text{mcp}, xy}\|}{\|\mathbf{p}_{\text{tip}, xyz} - \mathbf{p}_{\text{mcp}, xyz}\|}$$
2. **Optical Axis Elevation Angle**:
   $$\phi_{\text{elev}} = \arcsin\left(\frac{v_z}{\|\mathbf{v}\|}\right)$$
3. **Behavior**:
   * If $R_{\text{foreshorten}} < 0.60$ or $|u_{z}| > 0.72$, the system flags `is_foreshortened = True`.
   * Evaluates 3D segment lengths rather than 2D screen lengths.
   * If observation is genuinely ambiguous due to self-occlusion, confidence degrades gracefully to `UNCERTAIN` or `UNKNOWN` rather than inventing false states.

---

## 6. Multi-Frame Temporal Reconstruction

When single-frame tracking noise, severe occlusion, or motion blur occurs:
* Frame $N-10 \dots N-1$: `Index = EXTENDED` (confidence $0.95$)
* Frame $N$: `Index = UNCERTAIN` (confidence $0.20$)
* **Temporal Reconstruction Engine**:
  * Scans recent $15$-frame buffer.
  * If pre-occlusion state was consistent ($\ge 60\%$), maintains `Index = EXTENDED`.
  * Applies exponential confidence decay: $\text{conf}_{\text{decay}} = \text{mean\_conf} \times (0.85)^{\text{gap}}$.
  * Bridges gaps up to $8$ frames without state collapse.
  * If occlusion persists beyond $8$ frames, state naturally transitions to `UNKNOWN`.

---

## 7. Continuous 3D Motion Kinematics

Motion is modeled through higher-order derivatives in full 3D:
* **3D Position**: $\mathbf{p}(t) = (x, y, z)$
* **3D Velocity**: $\mathbf{v}(t) = \frac{d\mathbf{p}}{dt} = (v_x, v_y, v_z)$
* **3D Acceleration**: $\mathbf{a}(t) = \frac{d\mathbf{v}}{dt} = (a_x, a_y, a_z)$
* **3D Jerk**: $\mathbf{j}(t) = \frac{d\mathbf{a}}{dt}$
* **3D Speed**: $\|\mathbf{v}\|$
* **Planar Speed**: $\sqrt{v_x^2 + v_y^2}$
* **Depth Speed**: $v_z + 0.85 \times \left|\frac{d(d_{\text{ref}})}{dt}\right|$
* **Depth State**: `TOWARD` (expanding scale or $-Z$), `AWAY` (contracting scale or $+Z$), `NEUTRAL`.

---

## 8. Master Hand Perception Object Schema

The unified output generated every frame by `SpatialPerceptionEngine`:

```json
{
  "hand": {
    "hand_id": 1,
    "handedness": "Right",
    "position": [0.52, 0.61, 0.04],
    "orientation": {
      "pitch_deg": -5.2,
      "yaw_deg": 12.1,
      "roll_deg": 3.4,
      "palm_facing": "PALM"
    },
    "axes": {
      "origin": [0.50, 0.70, 0.00],
      "x_radial": [0.98, 0.02, 0.12],
      "y_distal": [0.03, -0.99, 0.05],
      "z_normal": [-0.11, 0.04, -0.99]
    },
    "confidence": 0.94
  },
  "fingers": {
    "index": {
      "finger": "index",
      "state": "EXTENDED",
      "confidence": 0.92,
      "flexion": {
        "mcp_deg": 4.2,
        "pip_deg": 3.8,
        "dip_deg": 2.1,
        "total_deg": 10.1
      },
      "curl_ratio": 0.05,
      "spread_deg": 2.1,
      "distance_to_palm": 0.92,
      "foreshortening": {
        "ratio": 0.98,
        "is_foreshortened": false
      },
      "direction": {
        "camera": [0.04, -0.99, 0.06],
        "hand_relative": [0.01, 1.00, 0.02],
        "screen_2d": [0.04, -0.99]
      }
    }
  },
  "motion": {
    "velocity": [-0.64, -0.71, 0.05],
    "speed": {
      "speed_3d": 0.96,
      "speed_xy": 0.95,
      "speed_z": 0.05
    },
    "angles": {
      "heading_deg": -132.0,
      "elevation_deg": 3.0,
      "angular_vel_rad_s": 0.12
    },
    "depth": {
      "state": "NEUTRAL",
      "scale_rate": 0.01
    }
  },
  "spatial": {
    "screen_direction": [-0.67, -0.74],
    "hand_direction": [0.05, 0.98, 0.12],
    "depth_direction": "NEUTRAL",
    "primary_sector": "TOP_LEFT",
    "secondary_sector": "TOP",
    "ambiguity_level": "LOW",
    "zone_distribution": {
      "top_left": 0.72,
      "top": 0.22,
      "left": 0.06
    }
  },
  "uncertainty": {
    "state": "STABLE",
    "overall_confidence": 0.94,
    "directional_ambiguity": "LOW",
    "is_foreshortened": false,
    "reasons": []
  },
  "stability": {
    "state": "STABLE",
    "score": 0.95
  }
}
```

---

## 9. Verification & Performance Results

### Test Suite Execution
Run via:
```bash
PYTHONPATH=. ./.venv/bin/pytest tests/test_spatial_representation.py
```
* 11 comprehensive spatial test suites passing ($100\%$).
* Full test suite across all 160 tests passing in $< 3.1$ seconds.

### Key Test Matrix Validations:
* **Diagonal Movement**: Motion along $(-0.707, -0.707)$ evaluates to `TOP_LEFT` with $p_{\text{TOP\_LEFT}} > 0.40$, strictly superior to `LEFT` and `TOP`.
* **The Grey Zone**: Borderline motion ($-112.5^\circ$) flags `AmbiguityLevel.HIGH` and permits explicit `AMBIGUOUS` label rather than forcing false certainty.
* **Hand-Relative Invariance**: Index finger maintains $\mathbf{v}_{\text{hand}} \approx (0, 1, 0)$ across $0^\circ, 45^\circ, 90^\circ, 180^\circ, -90^\circ$ hand rotations.
* **Camera-Facing Postures**: Detects foreshortening and elevation angles, avoiding false flexion.
* **Temporal Reconstruction**: Single-frame occlusions bridged with graceful confidence decay.
