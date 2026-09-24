/**
 * Spatial Cursor component managing 2D/3D cursor feedback and state transitions.
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

    this.stateColors = {
      IDLE: "#6b7280",
      OBSERVING: "#00f0ff",
      CANDIDATE: "#f59e0b",
      CONFIRMED: "#a855f7",
      ACTIVE: "#10b981",
      RELEASING: "#f43f5e",
    };
  }

  updateFromNDC(ndcX, ndcY, intentState, isPinchActive, pinchConfidence) {
    // Convert NDC [-1, 1] to screen pixel coordinates
    // NDC: x in [-1, 1], y in [-1, 1] (y is up in NDC, down in CSS)
    this.targetX = ((ndcX + 1.0) / 2.0) * window.innerWidth;
    this.targetY = ((1.0 - ndcY) / 2.0) * window.innerHeight;

    // Smooth lerp
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
