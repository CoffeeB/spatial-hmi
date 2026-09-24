# Mathematical Coordinate Transformations in Spatial HMI

This document details the exact 5-stage transformation pipeline converting a 2D camera observation into a 3D spatial interaction command.

```text
Camera Pixel Space (u_px, v_px)
        ↓  (Horizontal mirror & resolution division)
Normalized Image Space (u_norm, v_norm) ∈ [0, 1]²
        ↓  (Centering & Y-inversion)
Normalized Device Coordinates (x_ndc, y_ndc) ∈ [-1, 1]²
        ↓  (Pinhole inverse camera projection matrix)
3D Ray Casting (origin, direction) in Camera Space
        ↓  (Analytic quadratic ray-sphere intersection)
3D World Interaction Point P_surface on 3D Globe
```

---

## Stage 1: Camera Pixel Space $\to$ Normalized Image Space

A monocular webcam produces an image frame of resolution $W \times H$ pixels (e.g., $640 \times 480$).
Let $(u_{\text{px}}, v_{\text{px}}) \in [0, W] \times [0, H]$ be the pixel coordinates of the detected palm centroid or fingertip.

To achieve resolution independence and natural "mirror" interaction (where moving the physical right hand to the right translates the on-screen cursor to the right):
$$u_{\text{norm}} = 1.0 - \frac{u_{\text{px}}}{W}$$
$$v_{\text{norm}} = \frac{v_{\text{px}}}{H}$$
Both coordinates are bounded: $u_{\text{norm}}, v_{\text{norm}} \in [0.0, 1.0]$.

---

## Stage 2: Normalized Image Space $\to$ Normalized Device Coordinates (NDC)

In WebGL / Three.js, Normalized Device Coordinates range from $[-1.0, 1.0]$ across both axes, with $(0, 0)$ at the center of the viewport, $X$ positive to the right, and $Y$ positive upward.

Because image space originates at the top-left with $Y$ increasing downward:
$$x_{\text{ndc}} = 2 \cdot u_{\text{norm}} - 1.0$$
$$y_{\text{ndc}} = 1.0 - 2 \cdot v_{\text{norm}}$$

---

## Stage 3: Normalized Device Coordinates $\to$ 3D Ray in Camera Space

Using a pinhole perspective camera model with vertical field-of-view $\theta_{\text{fov}}$ and aspect ratio $A = \frac{W_{\text{viewport}}}{H_{\text{viewport}}}$:

1. **Ray Origin**: The optical center of the camera:
   $$\mathbf{r}_{\text{origin}} = \mathbf{C}_{\text{cam}} = [0, 0, z_{\text{cam}}]^\top$$
2. **Ray Direction**:
   $$\tan\left(\frac{\theta_{\text{fov}}}{2}\right)$$
   $$\mathbf{v}_{\text{ray}} = \begin{bmatrix} x_{\text{ndc}} \cdot A \cdot \tan(\theta_{\text{fov}}/2) \\ y_{\text{ndc}} \cdot \tan(\theta_{\text{fov}}/2) \\ -1.0 \end{bmatrix}$$
3. **Normalized Unit Ray Direction**:
   $$\mathbf{r}_{\text{dir}} = \frac{\mathbf{v}_{\text{ray}}}{\|\mathbf{v}_{\text{ray}}\|_2}$$

---

## Stage 4: 3D Ray-Sphere Intersection on Interactive Globe

For a 3D globe centered at $\mathbf{O} \in \mathbb{R}^3$ with radius $R$, points on the globe satisfy:
$$\|\mathbf{p} - \mathbf{O}\|^2 = R^2$$

Substituting the parametric ray equation $\mathbf{p}(t) = \mathbf{r}_{\text{origin}} + t \mathbf{r}_{\text{dir}}$ for $t > 0$:
$$\|\mathbf{r}_{\text{origin}} + t \mathbf{r}_{\text{dir}} - \mathbf{O}\|^2 = R^2$$

Expanding into quadratic form $a t^2 + b t + c = 0$:
- $a = \mathbf{r}_{\text{dir}} \cdot \mathbf{r}_{\text{dir}} = 1.0$ (since $\mathbf{r}_{\text{dir}}$ is a unit vector)
- $b = 2 \cdot (\mathbf{r}_{\text{origin}} - \mathbf{O}) \cdot \mathbf{r}_{\text{dir}}$
- $c = \|\mathbf{r}_{\text{origin}} - \mathbf{O}\|^2 - R^2$

The discriminant determines intersection:
$$\Delta = b^2 - 4ac$$

- If $\Delta \ge 0$: The ray intersects the front of the sphere at:
  $$t^* = \frac{-b - \sqrt{\Delta}}{2a}$$
  $$\mathbf{P}_{\text{surface}} = \mathbf{r}_{\text{origin}} + t^* \mathbf{r}_{\text{dir}}$$
- If $\Delta < 0$: The ray misses the sphere; the interaction projects onto the tangent interaction plane at $z = \mathbf{O}_z$.

---

## Stage 5: Spherical Surface Coordinate Mapping (Latitude / Longitude)

For an intersection point $\mathbf{P} = [x, y, z]^\top$ relative to globe center $\mathbf{O}$:
$$\text{Latitude } \phi = \arcsin\left(\frac{y}{R}\right) \times \frac{180}{\pi}$$
$$\text{Longitude } \lambda = \text{atan2}(z, -x) \times \frac{180}{\pi}$$

This allows direct spatial indexing into geographical nodes and digital markers mounted on the globe.
