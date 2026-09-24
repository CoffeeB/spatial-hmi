# Literature Review: Gesture-Driven Spatial Human-Machine Interfaces

## 1. Vision-Based Hand Tracking and Landmark Estimation
Early spatial gesture systems relied on active markers or structured light sensors (e.g., Leap Motion, Microsoft Kinect). The introduction of real-time deep monocular landmark estimation architectures—notably MediaPipe Hands (Zhang et al., 2020)—enabled single RGB camera hand tracking with 21 3D landmarks at >30 FPS on consumer hardware.

MediaPipe operates as a two-stage pipeline:
1. **Palm Detector**: A Single Shot Multibox Detector (SSD) optimized for mobile real-time inference that predicts an oriented bounding box for the palm region across the entire image.
2. **Hand Landmark Model**: A convolutional regression network taking the cropped, rotated palm ROI and predicting 21 3D coordinates $(x_i, y_i, z_i)$ relative to the wrist in normalized coordinates, alongside a handedness flag and landmark visibility confidence.

While raw landmark regression provides high frame-rate spatial coordinates, raw outputs are subject to high-frequency sensor noise, quantization jitter, and temporary tracking dropouts due to rapid motion or self-occlusion.

---

## 2. Temporal Smoothing & Jitter Reduction in Spatial Interfaces
Directly mapping raw vision landmarks to cursor or 3D object manipulation produces unacceptably high motor jitter, leading to unintended selection and operator fatigue.

### 2.1 The One-Euro Filter (1€ Filter)
Casiez, Roussel, and Vogel (2012) introduced the 1€ filter, an adaptive low-pass filter with dynamic cutoff frequency tailored for human-computer interaction:
$$\hat{x}_k = \alpha x_k + (1 - \alpha) \hat{x}_{k-1}$$
where the smoothing coefficient $\alpha$ is computed as:
$$\alpha = \frac{1}{1 + \frac{\tau}{T_e}} = \frac{2\pi f_c T_e}{2\pi f_c T_e + 1}$$
The cutoff frequency $f_c$ scales dynamically with the estimated signal velocity $\dot{x}_k$:
$$f_c = f_{c,\min} + \beta |\dot{x}_k|$$
- When the hand is static or moving slowly ($|\dot{x}_k| \approx 0$), $f_c \to f_{c,\min}$, maximizing jitter suppression.
- When the hand moves rapidly ($|\dot{x}_k| \gg 0$), $f_c$ increases, drastically reducing phase lag and latency.

### 2.2 Double Exponential Smoothing (Holt's Linear Trend)
For trajectory prediction and velocity estimation under non-stationary dynamics, double exponential smoothing decouples level and trend estimation:
$$s_t = \alpha y_t + (1 - \alpha)(s_{t-1} + b_{t-1})$$
$$b_t = \gamma (s_t - s_{t-1}) + (1 - \gamma) b_{t-1}$$

---

## 3. The "Midas Touch" Problem and Intent Estimation
A classic open challenge in touchless spatial interfaces is the **Midas Touch problem** (Jacob, 1990)—the inability of the system to differentiate between communicative, explorative, or involuntary hand gestures and deliberate command activation.

Instantaneous frame-by-frame classifiers trigger false activations whenever finger geometry momentarily matches a gesture prototype (e.g., during transit between states or scratching the face).

### 3.1 Buxton's Three-State Model of Graphical Input
William Buxton (1990) formalized input device interaction into three fundamental states:
- **State 0 (Out of Range)**: Hand not tracked or outside interaction envelope.
- **State 1 (Tracking / Hover)**: Hand tracked, moving spatial cursor/ray, no action committed.
- **State 2 (Dragging / Engaged)**: Physical or gestural engagement (e.g., pinch-drag, rotation).

Extending Buxton's model to probabilistic computer-vision spatial interfaces requires intermediate state buffering:
$$\text{IDLE} \xrightarrow{\text{detection}} \text{OBSERVING} \xrightarrow{\text{evidence}} \text{CANDIDATE} \xrightarrow{\text{temporal confirm}} \text{CONFIRMED} \xrightarrow{\text{latch}} \text{ACTIVE} \xrightarrow{\text{release trigger}} \text{RELEASING} \to \text{IDLE}$$

### 3.2 Confidence Calibration & Temporal Evidence Accumulation
Rather than hard thresholding instantaneous classifier outputs, evidence accumulation over a sliding temporal window of duration $W$ or via exponential moving average (EMA) of class posterior $P(G_t \mid \mathbf{z}_t)$ produces robust hysteresis:
$$E_t(g) = \lambda E_{t-1}(g) + (1 - \lambda) P(g \mid \mathbf{z}_t)$$
An intent transition is triggered only when $E_t(g) \ge \theta_{\text{activate}}$ continuously for $N_{\text{confirm}}$ consecutive frames, and released when $E_t(g) < \theta_{\text{release}}$.

---

## 4. 3D Spatial Manipulation Techniques
Spatial manipulation of 3D objects from 2D monocular vision requires well-defined projective mapping:
1. **Ray-Casting and Spherical Target Selection**: Projecting normalized viewport coordinates $(u, v) \in [-1, 1]^2$ through the inverse camera projection matrix to define a picking ray in 3D scene coordinates.
2. **Two-Handed Bimanual Interaction (Guiard's Kinematic Chain)**: Guiard (1987) demonstrated that the non-dominant hand sets the spatial frame of reference (e.g., rotation/zoom) while the dominant hand performs fine manipulation (e.g., selection/dragging).
