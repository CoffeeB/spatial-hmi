# Confidence-Aware Temporal Spatial Human–Machine Interface (HMI)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands_21_Landmarks-cyan.svg)](https://developers.google.com/mediapipe)
[![Three.js](https://img.shields.io/badge/WebGL-Three.js_3D_Globe-emerald.svg)](https://threejs.org/)
[![Tests](https://img.shields.io/badge/Tests-20%20Passing-brightgreen.svg)]()

A research-grade software system and evaluation framework for **gesture-driven spatial human–machine interfaces**. The interaction engine processes monocular RGB camera video into 21-point 3D hand landmarks, applies scale- and translation-invariant normalization, accumulates temporal intent evidence via a 6-state finite state machine, and applies speed-adaptive 1€ spatial filtering to smoothly manipulate a 3D WebGL interactive globe.

---

## 🌟 Core Research Contributions

1. **Resolution of the "Midas Touch" Problem**: Multi-stage temporal confirmation FSM ($\text{IDLE} \to \text{OBSERVING} \to \text{CANDIDATE} \to \text{CONFIRMED} \to \text{ACTIVE} \to \text{RELEASING}$) achieves a **$91.4\%$ to $100.0\%$ reduction in unintended activations** compared to instantaneous frame-by-frame classification.
2. **Adaptive 1€ Spatial Filtering**: Dynamic cutoff scaling ($f_c = f_{c,\min} + \beta |\hat{\dot{\mathbf{x}}}|$) achieves a **$69.7\%$ to $82.7\%$ reduction in stationary hand jitter** while maintaining sub-15 ms dynamic phase lag during ballistic targeting movements.
3. **Decoupled Architectural Pipeline**: Complete separation of Computer Vision Perception $\to$ Kinematic Features $\to$ Intent FSM $\to$ Spatial Commands $\to$ Three.js/WebGL Visualizer. The exact same vision engine can be reused to manipulate CAD objects, flight simulators, GIS maps, and robotics without code changes.

---

## 🏛 System Architecture

```text
Camera Stream (Threaded 30-60 FPS)
        ↓
MediaPipe Hand Detector (21 3D Joint Landmarks)
        ↓
Landmark Normalizer (Palm-centered, d_ref scale invariant)
        ↓
Kinematic Feature Extractor (Extension ratios, angles, sigmoid pinch confidence, palm velocity)
        ↓
Heuristic Gesture Classifier (Open Palm, Pinch, Point, Grab, Spread, Contraction, Rotation)
        ↓
Temporal Intent State Machine (IDLE -> OBSERVING -> CANDIDATE -> CONFIRMED -> ACTIVE -> RELEASING)
        ↓
Adaptive 1€ Spatial Filter & 5-Stage Coordinate Transformer
        ↓
Generic Spatial Interaction Engine (Mapping to High-Level 3D Commands)
        ↓
Asynchronous WebSocket Server (JSON Telemetry & Command Broadcast at 60 Hz)
        ↓
Three.js / WebGL Visualizer (Interactive 3D Globe with Spatial Cursor & Nodes)
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
Using `uv` (recommended) or standard `venv`:
```bash
# Clone the repository
git clone https://github.com/your-org/gesture-driven-spatial-hmi.git
cd gesture-driven-spatial-hmi

# Install dependencies using Python 3.12
~/.local/bin/uv venv --python 3.12 .venv
source .venv/bin/activate
~/.local/bin/uv pip install -r requirements.txt
```

### 2. Launching the Master HMI Server & 3D Web App
Run the single-command runner:
```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_hmi_server.py
```
This automatically starts:
- The camera frame capture & perception engine
- The asynchronous WebSocket server on `ws://127.0.0.1:8765`
- The static HTTP visualizer server on `http://localhost:8080/index.html`

Open your web browser at **`http://localhost:8080/index.html`**.

### Optional Developer Modes:
- **Developer Camera HUD**: Add `--dev` to open a local OpenCV window with real-time landmark overlays and state text:
  ```bash
  PYTHONPATH=. ./.venv/bin/python scripts/run_hmi_server.py --dev
  ```
- **Headless / Synthetic Mode** (for automated testing without webcam):
  ```bash
  PYTHONPATH=. ./.venv/bin/python scripts/run_hmi_server.py --synthetic
  ```

---

## 🖐 Spatial Gesture Interaction Primitives

| Gesture | Movement Context | Spatial Command | 3D Globe Behavior |
| :--- | :--- | :--- | :--- |
| **Point** | Stable pointing toward node | `SELECT` | Highlights and selects the spatial node |
| **Pinch** | Thumb & index together + drag | `TRANSLATE_NODE` | Moves the selected node across the globe surface |
| **Grab** | Closed fist + hand movement | `ROTATE_OBJECT` | Rotates the 3D globe in yaw and pitch |
| **2-Hand Spread** | Distance between hands expanding | `SCALE_OBJECT` | Smoothly zooms the camera in |
| **2-Hand Contract**| Distance between hands shrinking | `SCALE_OBJECT` | Smoothly zooms the camera out |
| **Open Palm** | All fingers extended | `RELEASE_OBJECT` | Cancels active interaction or releases node |
| **Hover** | Open hand browsing | `HOVER` | Moves spatial cursor and highlights hovered targets |

---

## 🧪 Running the Research Benchmark Suite

Run the full scientific evaluation pipeline across all 5 experiments:
```bash
PYTHONPATH=. ./.venv/bin/python experiments/run_all_benchmarks.py
```

Run individual experiments:
- **Experiment 1 (Static Recognition & ECE)**: `PYTHONPATH=. ./.venv/bin/python experiments/recognition/eval_static_gestures.py`
- **Experiment 2 (Temporal Intent vs Instantaneous)**: `PYTHONPATH=. ./.venv/bin/python experiments/temporal/eval_temporal_intent.py`
- **Experiment 3 (Movement Stability & 1€ Filter)**: `PYTHONPATH=. ./.venv/bin/python experiments/stability/eval_movement_stability.py`
- **Experiment 4 (User Scale & Variation)**: `PYTHONPATH=. ./.venv/bin/python experiments/user_variation/eval_user_variation.py`
- **Experiment 5 (Noise & Environmental Robustness)**: `PYTHONPATH=. ./.venv/bin/python experiments/robustness/eval_environmental_robustness.py`

---

## 🔬 Running Automated Unit Tests

```bash
PYTHONPATH=. ./.venv/bin/pytest tests/ -v
```

---

## 📁 Repository Structure

```text
gesture-interface/
├── research/
│   ├── RESEARCH.md               # Complete 21-section academic research paper
│   ├── literature_review.md      # Literature review of vision HMI, 1€ filter, Buxton model
│   ├── research_questions.md     # Primary/secondary RQs and formal operational metrics
│   ├── methodology.md            # Detailed mathematical formulations and proofs
│   ├── experiments.md            # Protocols for Experiments 1 through 5
│   ├── results.md                # Full tabular quantitative results and confusion matrices
│   ├── discussion.md             # In-depth HCI analysis and theoretical implications
│   ├── limitations.md            # Optical, occlusive, and hardware edge cases
│   └── references.md             # Academic bibliography
│
├── src/
│   ├── camera/                   # Threaded non-blocking video capture & auto-reconnection
│   ├── perception/               # MediaPipe Hands landmark extraction & debug drawing
│   ├── landmarks/                # Scale normalization, kinematics, and HandState
│   ├── gestures/                 # Heuristic classifier and continuous sigmoid confidence
│   ├── intent/                   # Leaky evidence accumulation & 6-state Intent FSM
│   ├── interaction/              # 1€ Filter, 5-stage coordinate transformer, spatial mapper
│   ├── communication/            # Asynchronous WebSocket server & HMIPacket protocol
│   └── utils/                    # Logging, Pydantic YAML config loader, scientific metrics
│
├── visualization/
│   ├── index.html                # Modern glassmorphic WebGL interface
│   ├── css/style.css             # Dark spatial computing styling
│   └── js/                       # Three.js 3D Globe, Spatial Nodes, Cursor, Debug HUD
│
├── experiments/                  # Standalone benchmark test harnesses for Exp 1 - Exp 5
├── tests/                        # Full pytest test suite (20 automated tests)
├── configs/                      # YAML configuration files
├── scripts/                      # Master server launch script
├── docs/                         # Technical guide, coordinate systems, API reference
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⌨️ Telemetry & Developer Controls

While in the WebGL visualizer (`http://localhost:8080/index.html`):
- **Press `[D]`**: Toggle the real-time **Research Telemetry HUD** displaying Pipeline FPS, End-to-End Latency, Interaction State Lifecycle, Active Gesture, and Intent Confidence Bar Gauge.
- **Fallback Mouse Mode**: If running without a webcam, click and drag with the mouse to rotate the globe or scroll the mouse wheel to zoom.
