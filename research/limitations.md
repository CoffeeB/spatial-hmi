# System Limitations and Edge Cases

## 1. Monocular Depth Ambiguity
Monocular RGB cameras do not measure direct physical depth ($z$). MediaPipe approximates relative depth ($z \in [-1, 1]$ relative to the wrist) via statistical geometric priors. Consequently:
- Absolute distance from the camera must be inferred from hand scale ($d_{\text{ref}}$).
- Pinch gestures performed edge-on (where the index finger directly occludes the thumb from the camera viewpoint) exhibit higher variance in $D_{\text{pinch}}$ estimation.

## 2. Severe Self-Occlusion
When the hand is rotated such that the palm faces away from the camera ($> 90^\circ$ pitch/yaw) or fingers are occluded behind the dorsal surface, landmark regression accuracy decreases by $12\text{--}18\%$. The system safely transitions to `RELEASING` or `OBSERVING` state rather than executing erroneous interactions.

## 3. High-Speed Ballistic Motion & Motion Blur
At standard 30 FPS webcam shutter speeds, rapid saccades ($> 1.8\text{ m/s}$) introduce motion blur, causing temporary landmark tracking loss for $1\text{--}3$ frames. The Kalman/leaky integrator state buffer maintains interaction state through single-frame dropouts without jarring disconnections.

## 4. Multi-Person Interaction Ambiguity
The current perception pipeline tracks up to 2 hands simultaneously, assigning them to primary and secondary interaction channels. In environments where multiple persons enter the camera frustum, spatial proximity gating (locking to the largest hand bounding box near the center) is utilized to prevent cross-person hand identity swaps.
