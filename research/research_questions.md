# Research Questions and Operational Definitions

## 1. Primary Research Question (RQ1)
> **RQ1**: How accurately, reliably, and with what latency can natural vision-based hand gestures replace conventional mouse and touch interactions for continuous spatial manipulation in 3D digital interfaces?

### Sub-Questions:
- **RQ1.1**: What is the baseline precision, recall, and F1-score of geometric/kinematic heuristic gesture classification across fundamental interaction primitives (Open Palm, Pinch, Point, Grab, Two-Hand Spread, Two-Hand Contraction, Two-Hand Rotation)?
- **RQ1.2**: What end-to-end latency (camera capture $\to$ inference $\to$ intent state $\to$ WebSocket dispatch $\to$ WebGL render) is achievable on commodity hardware without dedicated accelerators?
- **RQ1.3**: What degree of positional jitter reduction is achieved by adaptive filtering (1€ Filter) compared to raw landmarks and uniform exponential moving averages?

---

## 2. Secondary Research Question (RQ2)
> **RQ2**: To what extent does the incorporation of temporal evidence accumulation, contextual state machines, and confidence calibration reduce false activations and the "Midas Touch" problem compared to instantaneous frame-by-frame gesture classification?

### Sub-Questions:
- **RQ2.1**: By what percentage is the False Activation Rate (FAR) reduced when transitioning from an instantaneous trigger model to a multi-stage temporal confirmation state machine ($\text{IDLE} \to \text{OBSERVING} \to \text{CANDIDATE} \to \text{CONFIRMED} \to \text{ACTIVE}$)?
- **RQ2.2**: What is the associated latency trade-off (temporal latency penalty $\Delta t_{\text{confirm}}$) incurred to achieve a target False Activation Rate of $\le 1.0\%$ during non-interaction transit movements?
- **RQ2.3**: Does multi-modal confidence gating ($\mathcal{C}_{\text{detection}} \times \mathcal{C}_{\text{gesture}} \times \mathcal{C}_{\text{temporal}}$) maintain robustness against sudden occlusion, rapid hand entry/exit, and fluctuating illumination?

---

## 3. Operational Definitions & Formal Metrics

| Metric Term | Mathematical Definition | Operational Meaning | Target Benchmark |
| :--- | :--- | :--- | :--- |
| **Recognition Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall classification correctness over labeled landmark frames | $\ge 96.0\%$ |
| **False Activation Rate (FAR)** | $\frac{N_{\text{unintended activations}}}{T_{\text{idle tracking hours}}}$ | Rate of accidental interaction activations during idle or transit motion | $< 0.05 \text{ activations/min}$ |
| **Missed Activation Rate (MAR)** | $\frac{N_{\text{intended, failed activations}}}{N_{\text{total intended gestures}}}$ | Fraction of deliberate user gestures that failed to confirm interaction | $< 3.0\%$ |
| **End-to-End Latency** | $t_{\text{render\_frame}} - t_{\text{sensor\_capture}}$ | Wall-clock delay from physical hand movement to visual screen response | $< 35\text{ ms}$ (at 30–60 FPS) |
| **Positional Jitter** | $\sigma_{\Delta p} = \sqrt{\frac{1}{N}\sum_{i=1}^N \|p_i - \bar{p}\|^2}$ | Standard deviation of spatial cursor coordinates when hand is held stationary | $< 1.5 \times 10^{-3}$ norm units |
| **Expected Calibration Error (ECE)** | $\sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$ | Statistical divergence between predicted confidence and observed empirical accuracy | $\text{ECE} < 0.06$ |
| **Interaction Stability Index** | $\mathcal{S} = 1 - \frac{\int |\ddot{p}(t)| dt}{\int |\dot{p}(t)| dt + \epsilon}$ | Ratio of smooth velocity trajectory to high-frequency acceleration oscillations | $\mathcal{S} \ge 0.92$ |
