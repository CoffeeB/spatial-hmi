/**
 * Spatial Cursor component managing 2D/3D cursor feedback and state transitions.
 * The cursor is clamped to the globe's projected screen-space circle so it
 * never escapes the hologram area regardless of hand position.
 */

class SpatialCursor {
  constructor() {
    this.cursorElem = document.getElementById("spatial-cursor");
    this.ringElem = this.cursorElem.querySelector(".cursor-ring");
    this.pinchRingElem = this.cursorElem.querySelector(".cursor-pinch-ring");

    this.currentX = window.innerWidth / 2;
    this.currentY = window.innerHeight / 2;
    this.targetX = this.currentX;
    this.targetY = this.currentY;

    // Globe screen-space bounds — updated each frame from app.js
    // {cx, cy, r} all in CSS pixels
    this.globeBounds = null;

    this.stateColors = {
      IDLE: "#6b7280",
      OBSERVING: "#00f0ff",
      CANDIDATE: "#f59e0b",
      CONFIRMED: "#a855f7",
      ACTIVE: "#10b981",
      RELEASING: "#f43f5e",
    };
  }

  /**
   * Called each frame from app.js with the globe's projected screen circle.
   * @param {{ cx: number, cy: number, r: number }} bounds  CSS-pixel center + radius
   */
  setGlobeBounds(bounds) {
    this.globeBounds = bounds;
  }

  /**
   * Clamps a point (px, py) to lie within a circle of center (cx, cy) and radius r.
   * Returns the clamped point.
   */
  _clampToCircle(px, py, cx, cy, r) {
    const dx = px - cx;
    const dy = py - cy;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist <= r) return { x: px, y: py };
    // Project onto circle boundary
    const scale = r / dist;
    return { x: cx + dx * scale, y: cy + dy * scale };
  }

  updateFromNDC(ndcX, ndcY, intentState, isPinchActive, pinchConfidence) {
    // Convert NDC [-1, 1] to screen pixel coordinates
    // NDC: x in [-1, 1], y in [-1, 1] (y is up in NDC, down in CSS)
    let rawX = ((ndcX + 1.0) / 2.0) * window.innerWidth;
    let rawY = ((1.0 - ndcY) / 2.0) * window.innerHeight;

    // Clamp to globe bounds circle so the cursor never escapes the hologram area
    if (this.globeBounds) {
      const clamped = this._clampToCircle(
        rawX, rawY,
        this.globeBounds.cx, this.globeBounds.cy, this.globeBounds.r
      );
      rawX = clamped.x;
      rawY = clamped.y;
    }

    this.targetX = rawX;
    this.targetY = rawY;

    // Smooth lerp toward clamped target
    this.currentX += (this.targetX - this.currentX) * 0.35;
    this.currentY += (this.targetY - this.currentY) * 0.35;

    this.cursorElem.style.transform = `translate(${this.currentX}px, ${this.currentY}px)`;

    // Color transition based on state
    const color = this.stateColors[intentState] || "#00f0ff";
    this.ringElem.style.borderColor = color;
    this.ringElem.style.boxShadow = `0 0 14px ${color}`;

    // Pinch visual indicator
    if (isPinchActive || pinchConfidence > 0.6) {
      this.pinchRingElem.style.opacity = pinchConfidence;
      const scale = 1.4 - pinchConfidence * 0.4;
      this.pinchRingElem.style.transform = `scale(${scale})`;
    } else {
      this.pinchRingElem.style.opacity = "0";
    }
  }

  hide() {
    this.cursorElem.style.opacity = "0";
  }

  show() {
    this.cursorElem.style.opacity = "1";
  }
}

