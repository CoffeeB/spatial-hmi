# Experimental Protocol & Evaluation Design

## 1. Overview of Controlled Experiments

The system is evaluated against five structured experiments designed to address the primary and secondary research questions.

```text
Experiment 1: Static Gesture Recognition Benchmark
   ├── Dataset: Synthetic + Real multi-user landmark distributions
   ├── Conditions: Open Palm, Pinch, Point, Grab, Spread, Contraction, Rotation
   └── Metrics: Confusion Matrix, Precision, Recall, F1, ECE

Experiment 2: Temporal Intent vs. Instantaneous Classification
   ├── Baseline: Instantaneous Frame Thresholding
   ├── Proposed: Multi-Stage Intent FSM + Temporal Evidence Accumulation
   └── Metrics: False Activation Rate (FAR), Missed Activation Rate (MAR), Transition Latency

Experiment 3: Spatial Movement Stability & Jitter Analysis
   ├── Baseline: Raw MediaPipe Landmarks
   ├── Alternative: Static Exponential Moving Average (EMA)
   ├── Proposed: Adaptive 1€ Filter (One-Euro)
   └── Metrics: Positional Standard Deviation (Jitter), Phase Lag, Frequency Spectrum

Experiment 4: Inter-User & Kinematic Variation
   ├── Variables: Hand Dimensions ($d_{\text{ref}} \in [0.10, 0.28]$), Velocity Profiles, Dominant Hand
   └── Metrics: Generalization Accuracy, Normalization Invariance, Failure Rate

Experiment 5: Environmental & Sensor Robustness
   ├── Variables: Illumination Scaling (Low: 20%, High: 150%), Gaussian Noise Injection, Occlusion Masking
   └── Metrics: Landmark Detection Dropout, Confidence Calibration Degradation, State Retention
```

---

## 2. Detailed Protocols

### Experiment 1: Static Gesture Recognition
- **Sample Size**: $N = 2,800$ evaluation frames across 7 canonical gestures.
- **Generation**: Uniform sampling across realistic anatomical joint distributions with additive Gaussian noise $\mathcal{N}(0, \sigma^2)$ reflecting monocular sensor uncertainty.
- **Evaluation Criteria**: Multi-class confusion matrix, Macro-averaged F1, and Expected Calibration Error (ECE) with 10 confidence bins.

### Experiment 2: Temporal Intent Verification
- **Input Stream**: Continuous simulated interaction sequences containing deliberate gestures interspersed with non-interaction transit movements (e.g., reaching, scratching, ambient fidgeting).
- **Comparative Regimes**:
  1. *Instantaneous*: Action triggered on frame $t$ if $c_t \ge 0.70$.
  2. *N-Frame Voting*: Majority vote over 5 frames.
  3. *Proposed FSM*: Leaky evidence accumulation ($\lambda = 0.82$) with hysteresis thresholds ($\theta_{\text{activate}} = 0.75, \theta_{\text{release}} = 0.35, N_{\text{confirm}} = 3$).

### Experiment 3: Movement Stability & Filter Dynamics
- **Static Test**: Hand held static for 300 frames. Measure Euclidean variance $\sigma_p = \sqrt{\mathbb{E}[\|p - \bar{p}\|^2]}$.
- **Dynamic Saccade Test**: Step impulse in hand position to measure rise time and 90% settling delay (latency penalty in milliseconds).

### Experiment 4: User Variation & Scale Invariance
- Hand scale variation from small ($d_{\text{ref}} = 0.12\text{ m}$) to large ($d_{\text{ref}} = 0.26\text{ m}$).
- Left hand vs. Right hand parity testing.
- Radial hand orientation rotations from $-45^\circ$ to $+45^\circ$.

### Experiment 5: Environmental Robustness Stress-Testing
- Synthetic degradation pipelines:
  - Contrast reduction ($\gamma \in [0.4, 1.8]$).
  - Landmark dropout simulation (simulating finger occlusion up to 2 fingers).
  - Rapid out-of-frame exit and reentry recovery time (measured in frames to stable tracking reacquisition).
