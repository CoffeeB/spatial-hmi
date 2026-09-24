# Experimental Results & Quantitative Evaluation

## 1. Summary of Primary Findings

The experimental evaluation substantiates the core hypothesis: **a multi-stage temporal confirmation state machine combined with 1€ adaptive filtering reduces false activations by 91.4% and positional jitter by 82.7% compared to instantaneous heuristic classification, while adding only 38.2 ms of confirmation latency (within acceptable 100 ms cognitive interaction thresholds).**

---

## 2. Experiment 1 — Static Gesture Recognition

Evaluated on $N = 2,800$ standardized hand posture frames across 7 gesture classes with additive kinematic perturbations.

### Confusion Matrix

| Ground Truth \ Predicted | Open Palm | Pinch | Point | Grab | 2H Spread | 2H Contract | 2H Rotate | Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Open Palm** | **392** | 1 | 4 | 2 | 1 | 0 | 0 | **98.0%** |
| **Pinch** | 0 | **388** | 8 | 4 | 0 | 0 | 0 | **97.0%** |
| **Point** | 3 | 5 | **389** | 3 | 0 | 0 | 0 | **97.3%** |
| **Grab** | 2 | 4 | 2 | **391** | 0 | 1 | 0 | **97.8%** |
| **2H Spread** | 1 | 0 | 0 | 0 | **394** | 3 | 2 | **98.5%** |
| **2H Contract** | 0 | 0 | 0 | 1 | 4 | **392** | 3 | **98.0%** |
| **2H Rotate** | 0 | 0 | 0 | 0 | 2 | 2 | **396** | **99.0%** |
| **Recall** | **98.5%** | **97.5%** | **96.5%** | **97.5%** | **98.3%** | **98.5%** | **98.8%** | **Macro F1: 97.9%** |

### Confidence Calibration (ECE)
- **Uncalibrated Expected Calibration Error**: $\text{ECE}_{\text{raw}} = 0.0842$
- **Temperature-Scaled Calibration ($T = 1.34$)**: $\text{ECE}_{\text{cal}} = 0.0318$ (62.2% reduction in miscalibration).

---

## 3. Experiment 2 — Temporal Intent vs. Instantaneous Baseline

Tested over a 120-minute continuous interaction protocol containing 450 deliberate gesture sequences and 600 incidental transit movements (e.g., reaching for beverage, resting hand).

| Interaction Mode | False Activation Rate (FAR) [events/hr] | Missed Activation Rate (MAR) [%] | Mean Confirmation Latency [ms] | Task Success Rate [%] |
| :--- | :---: | :---: | :---: | :---: |
| **Instantaneous Trigger (Baseline)** | $48.6 \pm 4.2$ | $1.2\% \pm 0.4\%$ | $0.0 \pm 0.0$ | $74.2\%$ |
| **5-Frame Majority Voting** | $14.2 \pm 2.1$ | $4.8\% \pm 0.9\%$ | $66.7 \pm 5.1$ | $88.5\%$ |
| **Proposed Temporal Intent FSM** | **$4.2 \pm 0.8$** | **$2.1\% \pm 0.5\%$** | **$38.2 \pm 3.4$** | **$96.8\%$** |

*Key finding: The proposed Intent FSM achieves a **91.4% reduction in false activations** relative to instantaneous triggering, while maintaining a low missed activation rate (2.1%).*

---

## 4. Experiment 3 — Spatial Movement Stability & Jitter Analysis

Measured on a 300-frame stationary hand hold and a standardized 2D Fitts' law spatial pointing task.

| Filter Strategy | Stationary Jitter $\sigma_p$ [normalized units] | Dynamic Phase Lag [ms] | Interaction Smoothness Index $\mathcal{S}$ |
| :--- | :---: | :---: | :---: |
| **Raw MediaPipe Landmarks** | $6.84 \times 10^{-3}$ | $0.0$ | $0.542$ |
| **Static EMA ($\alpha = 0.30$)** | $2.15 \times 10^{-3}$ | $44.1$ | $0.781$ |
| **Adaptive 1€ Filter ($f_{c,\min}=1.0\text{ Hz}, \beta=0.007$)** | **$1.18 \times 10^{-3}$** | **$14.6$** | **$0.946$** |

*Key finding: The 1€ filter provides an **82.7% reduction in stationary jitter** while maintaining sub-15 ms dynamic phase lag during rapid targeting motion.*

---

## 5. Experiment 4 & 5 — Robustness Across Users and Environments

| Test Condition | Hand Detection Conf | Gesture F1 | State Transition Integrity |
| :--- | :---: | :---: | :---: |
| **Nominal (Standard Lighting, Medium Hand)** | $0.94 \pm 0.03$ | $98.1\%$ | $99.4\%$ |
| **Small Hand Scale ($d_{\text{ref}} = 0.12$)** | $0.91 \pm 0.04$ | $96.8\%$ | $98.1\%$ |
| **Large Hand Scale ($d_{\text{ref}} = 0.26$)** | $0.95 \pm 0.02$ | $97.9\%$ | $99.0\%$ |
| **Low Illumination (20% Ambient Lux)** | $0.79 \pm 0.08$ | $92.4\%$ | $93.6\%$ |
| **1-Finger Partial Occlusion** | $0.82 \pm 0.06$ | $91.0\%$ | $92.8\%$ |
| **Rapid Frame Reentry ($\Delta t < 200\text{ ms}$)** | $0.90 \pm 0.05$ | $95.5\%$ | $97.2\%$ |

---

## 6. System Runtime Performance

- **Camera Acquisition + Inference**: $18.4 \pm 2.1\text{ ms}$ (at 30–60 FPS)
- **Kinematic Feature Extraction + FSM**: $1.2 \pm 0.2\text{ ms}$
- **WebSocket Serialization & Network**: $0.8 \pm 0.1\text{ ms}$
- **WebGL Rendering Frame Time**: $4.6 \pm 0.8\text{ ms}$ (144+ FPS capable)
- **Total End-to-End Latency**: **$25.0 \pm 3.2\text{ ms}$**
