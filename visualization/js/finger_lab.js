/**
 * Gestura Level 0 Finger State Laboratory
 * Core client script dedicated to interpreting and visualizing individual finger states.
 */

class FingerStateLab {
  constructor() {
    // DOM Elements - Status
    this.wsStatusDot = document.getElementById("ws-status-dot");
    this.wsStatusText = document.getElementById("ws-status-text");
    this.handCountChip = document.getElementById("hand-count-chip");
    this.fpsChip = document.getElementById("fps-chip");
    this.latencyChip = document.getElementById("latency-chip");
    this.dominantHandBadge = document.getElementById("dominant-hand-badge");
    this.trackingConfLabel = document.getElementById("tracking-conf-label");

    // DOM Elements - Sensor & Canvas
    this.cameraFeed = document.getElementById("camera-screen-feed");
    this.canvas = document.getElementById("hand-skeleton-canvas");
    this.ctx = this.canvas ? this.canvas.getContext("2d") : null;
    this.sensorPlaceholder = document.getElementById("sensor-placeholder");

    // Kinematic Values
    this.kPitchVal = document.getElementById("k-pitch-val");
    this.kYawVal = document.getElementById("k-yaw-val");
    this.kRollVal = document.getElementById("k-roll-val");
    this.kScaleVal = document.getElementById("k-scale-val");

    // Canonical Terminal & Synthesis
    this.canonicalSummary = document.getElementById("canonical-summary");
    this.synthesizedPoseLabel = document.getElementById("synthesized-pose-label");
    this.synthesizedPoseDesc = document.getElementById("synthesized-pose-desc");
    this.synthesisConf = document.getElementById("synthesis-conf");

    // Controls
    this.toggleAxesBtn = document.getElementById("toggle-axes-btn");
    this.toggleAnglesBtn = document.getElementById("toggle-angles-btn");
    this.recordBtn = document.getElementById("record-btn");
    this.recordBtnText = document.getElementById("record-btn-text");
    this.exportBtn = document.getElementById("export-btn");

    // State Flags
    this.showAxes = true;
    this.showAngles = true;
    this.isRecording = false;
    this.recordedPackets = [];

    // Digit References Map
    this.digitNames = ["Thumb", "Index", "Middle", "Ring", "Little"];
    this.digitElements = {};
    this.digitNames.forEach(d => {
      const lower = d.toLowerCase();
      this.digitElements[d] = {
        card: document.getElementById(`card-${lower}`),
        pill: document.getElementById(`pill-${lower}`),
        rext: document.getElementById(`rext-${lower}`),
        gaugeBar: document.getElementById(`gauge-bar-${lower}`),
        mcp: document.getElementById(`mcp-${lower}`),
        pip: document.getElementById(`pip-${lower}`),
        dip: document.getElementById(`dip-${lower}`),
        diag: document.getElementById(`diag-${lower}`),
      };
    });

    this._initCanvas();
    this._initHandlers();
    this._initWebSocket();
  }

  _initCanvas() {
    const resize = () => {
      if (this.canvas && this.canvas.parentElement) {
        this.canvas.width = this.canvas.parentElement.clientWidth;
        this.canvas.height = this.canvas.parentElement.clientHeight;
      }
    };
    window.addEventListener("resize", resize);
    resize();
  }

  _initHandlers() {
    if (this.toggleAxesBtn) {
      this.toggleAxesBtn.addEventListener("click", () => {
        this.showAxes = !this.showAxes;
        this.toggleAxesBtn.classList.toggle("active", this.showAxes);
      });
    }

    if (this.toggleAnglesBtn) {
      this.toggleAnglesBtn.addEventListener("click", () => {
        this.showAngles = !this.showAngles;
        this.toggleAnglesBtn.classList.toggle("active", this.showAngles);
      });
    }

    if (this.recordBtn) {
      this.recordBtn.addEventListener("click", () => this._toggleRecording());
    }

    if (this.exportBtn) {
      this.exportBtn.addEventListener("click", () => this._exportDataset());
    }
  }

  _toggleRecording() {
    this.isRecording = !this.isRecording;
    if (this.isRecording) {
      this.recordedPackets = [];
      this.recordBtn.classList.add("recording");
      this.recordBtnText.textContent = "STOP RECORDING";
      this.exportBtn.classList.add("hidden");
    } else {
      this.recordBtn.classList.remove("recording");
      this.recordBtnText.textContent = "RECORD DATASET";
      if (this.recordedPackets.length > 0) {
        this.exportBtn.classList.remove("hidden");
      }
    }
  }

  _exportDataset() {
    if (this.recordedPackets.length === 0) return;
    const blob = new Blob([JSON.stringify(this.recordedPackets, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `gestura_level0_dataset_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  _initWebSocket(host = "127.0.0.1", port = 8765) {
    const wsUrl = `ws://${host}:${port}`;
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      this.wsStatusDot.className = "chip-dot connected";
      this.wsStatusText.textContent = "CONNECTED (60Hz)";
    };

    this.ws.onmessage = (event) => {
      try {
        const packet = JSON.parse(event.data);
        this._processPacket(packet);
      } catch (err) {
        console.error("Error parsing packet:", err);
      }
    };

    this.ws.onclose = () => {
      this.wsStatusDot.className = "chip-dot disconnected";
      this.wsStatusText.textContent = "DISCONNECTED";
      setTimeout(() => this._initWebSocket(host, port), 2000);
    };

    this.ws.onerror = () => {
      this.ws.close();
    };
  }

  _processPacket(packet) {
    if (this.isRecording) {
      this.recordedPackets.push(packet);
    }

    // 1. Live Camera Stream
    if (packet.video_frame_b64) {
      this.cameraFeed.src = "data:image/jpeg;base64," + packet.video_frame_b64;
      this.cameraFeed.style.opacity = "1";
      if (this.sensorPlaceholder) this.sensorPlaceholder.classList.add("hidden");
    }

    // 2. Performance Stats
    if (this.fpsChip) this.fpsChip.textContent = `${packet.fps.toFixed(1)} FPS`;
    if (this.latencyChip) this.latencyChip.textContent = `${packet.latency_ms.toFixed(1)} MS`;

    const hands = packet.hands || [];
    const numHands = hands.length;
    if (this.handCountChip) {
      this.handCountChip.textContent = numHands === 0 ? "AWAITING HAND..." : (numHands === 1 ? "1 HAND DETECTED" : `${numHands} HANDS DETECTED`);
      this.handCountChip.classList.toggle("chip-muted", numHands === 0);
      this.handCountChip.classList.toggle("chip-active", numHands > 0);
    }

    // 3. Render 3D Hand Skeleton & Frame
    this._renderSkeleton(packet);

    if (numHands === 0) {
      this._setIdleStates();
      return;
    }

    const primary = hands[0];
    const handedness = primary.handedness || "Right";
    if (this.dominantHandBadge) {
      this.dominantHandBadge.textContent = `PRIMARY: ${handedness.toUpperCase()}`;
    }
    if (this.trackingConfLabel) {
      this.trackingConfLabel.textContent = `Conf: ${primary.detection_confidence.toFixed(2)}`;
    }

    // 4. Update Kinematics Orientation
    if (primary.orientation_angles) {
      const [pitch, yaw, roll] = primary.orientation_angles;
      const toDeg = (r) => (r * 180 / Math.PI).toFixed(1) + "°";
      if (this.kPitchVal) this.kPitchVal.textContent = toDeg(pitch);
      if (this.kYawVal) this.kYawVal.textContent = toDeg(yaw);
      if (this.kRollVal) this.kRollVal.textContent = toDeg(roll);
    }
    if (this.kScaleVal && primary.hand_scale_ref) {
      this.kScaleVal.textContent = primary.hand_scale_ref.toFixed(3);
    }

    // 5. Update All 5 Digit Interpretation Cards
    const details = primary.finger_details || {};
    const states = primary.finger_states || {};

    this.digitNames.forEach(d => {
      const el = this.digitElements[d];
      const data = details[d] || {};
      const state = states[d] || data.state || "uncertain";

      if (el.pill) {
        el.pill.textContent = state.toUpperCase();
        el.pill.className = `state-pill state-${state.toLowerCase()}`;
      }

      const rext = data.extension_ratio !== undefined ? data.extension_ratio : 1.0;
      if (el.rext) el.rext.textContent = rext.toFixed(2);
      if (el.gaugeBar) {
        if (state.toLowerCase() === "uncertain") {
          el.gaugeBar.style.width = "0%";
          el.gaugeBar.style.backgroundColor = "#ef4444";
        } else {
          // Clamp 0.6 to 1.8 ratio to 0% - 100%
          const pct = Math.min(100, Math.max(0, ((rext - 0.6) / 1.2) * 100));
          el.gaugeBar.style.width = `${pct}%`;
          el.gaugeBar.style.backgroundColor = "";
        }
      }

      if (el.mcp) el.mcp.textContent = data.mcp_deg !== undefined ? `${data.mcp_deg}°` : "--";
      if (el.pip) el.pip.textContent = data.pip_deg !== undefined ? `${data.pip_deg}°` : "--";
      if (el.dip) el.dip.textContent = data.dip_deg !== undefined ? `${data.dip_deg}°` : "--";

      if (el.diag) {
        const diags = data.diagnostics || [];
        if (diags.length > 0) {
          el.diag.textContent = diags.join(" | ");
        } else if (data.contact_target) {
          el.diag.textContent = `In contact with ${data.contact_target}`;
        } else {
          el.diag.textContent = `Geometric state: ${state}`;
        }
      }
    });

    // 6. Update Canonical Readout
    if (this.canonicalSummary) {
      const lines = [
        `Thumb:    ${(states["Thumb"] || "--").padEnd(10)}`,
        `Index:    ${(states["Index"] || "--").padEnd(10)}`,
        `Middle:   ${(states["Middle"] || "--").padEnd(10)}`,
        `Ring:     ${(states["Ring"] || "--").padEnd(10)}`,
        `Little:   ${(states["Little"] || "--").padEnd(10)}`,
      ];
      this.canonicalSummary.textContent = lines.join("\n");
    }

    // 7. Update Level 1 Pose Preview
    const activeGesture = packet.active_gesture || "NONE";
    const intentConf = packet.intent_confidence || 0.0;
    if (this.synthesizedPoseLabel) {
      this.synthesizedPoseLabel.textContent = activeGesture;
    }
    if (this.synthesisConf) {
      this.synthesisConf.textContent = `CONF: ${intentConf.toFixed(2)}`;
    }
    if (this.synthesizedPoseDesc) {
      const descMap = {
        "POINT": "Index extended, others curled into palm (Pointing / Raycast)",
        "PINCH": "Thumb and index tips touching in opposition (Pinch Hold)",
        "OPEN_PALM": "All 5 digits straightened open (Neutral Observation)",
        "GRAB": "All digits curled tightly into palm (Closed Fist)",
        "PEACE": "Index and Middle extended in V-formation",
        "SPREAD_FINGERS": "All fingers wide and abducted (Max Scale)",
        "NONE": "Transitioning posture or neutral resting state",
      };
      this.synthesizedPoseDesc.textContent = descMap[activeGesture] || `Composite Gesture: ${activeGesture}`;
    }
  }

  _renderSkeleton(packet) {
    if (!this.ctx || !this.canvas) return;
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);

    const hands = packet.hands || [];
    if (hands.length === 0) return;

    const hand = hands[0];
    const pts = hand.landmarks_normalized || [];
    if (pts.length < 21) return;

    // Convert normalized [x, y, z] to canvas coordinates
    // Landmarks are normalized [0, 1] relative to webcam image
    const toCanvas = (p) => [p[0] * w, p[1] * h];

    const fingerBones = [
      { name: "Thumb",  joints: [[0, 1], [1, 2], [2, 3], [3, 4]] },
      { name: "Index",  joints: [[0, 5], [5, 6], [6, 7], [7, 8]] },
      { name: "Middle", joints: [[0, 9], [9, 10], [10, 11], [11, 12]] },
      { name: "Ring",   joints: [[0, 13], [13, 14], [14, 15], [15, 16]] },
      { name: "Little", joints: [[0, 17], [17, 18], [18, 19], [19, 20]] },
    ];

    const knuckleBase = [[5, 9], [9, 13], [13, 17]];

    const stateColors = {
      "extended": "#10b981",
      "folded":   "#64748b",
      "curved":   "#a855f7",
      "relaxed":  "#38bdf8",
      "tucked":   "#475569",
      "hooked":   "#f59e0b",
      "touching": "#eab308",
      "pinching": "#00f0ff",
      "crossed":  "#ec4899",
      "uncertain":"#ef4444",
    };

    // Draw knuckle base
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
    knuckleBase.forEach(([i1, i2]) => {
      const [x1, y1] = toCanvas(pts[i1]);
      const [x2, y2] = toCanvas(pts[i2]);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    });

    // Draw finger bones colored by state
    const states = hand.finger_states || {};
    fingerBones.forEach(({ name, joints }) => {
      const state = (states[name] || "uncertain").toLowerCase();
      const color = stateColors[state] || "#94a3b8";

      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      joints.forEach(([i1, i2]) => {
        const [x1, y1] = toCanvas(pts[i1]);
        const [x2, y2] = toCanvas(pts[i2]);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      });
    });

    // Draw Joint Nodes
    pts.forEach((p, idx) => {
      const [px, py] = toCanvas(p);
      const isTip = [4, 8, 12, 16, 20].includes(idx);
      const isWrist = idx === 0;

      ctx.beginPath();
      ctx.arc(px, py, isTip ? 6 : (isWrist ? 7 : 4), 0, Math.PI * 2);
      ctx.fillStyle = isTip ? "#ffffff" : (isWrist ? "#00f0ff" : "#94a3b8");
      ctx.fill();
      ctx.strokeStyle = "#000000";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Show Joint Degrees if toggled
      if (this.showAngles && [6, 10, 14, 18].includes(idx)) {
        const fingerMap = { 6: "Index", 10: "Middle", 14: "Ring", 18: "Little" };
        const dName = fingerMap[idx];
        const pipAngle = hand.finger_details?.[dName]?.pip_deg;
        if (pipAngle !== undefined) {
          ctx.fillStyle = "#ffffff";
          ctx.font = "10px monospace";
          ctx.fillText(`${pipAngle}°`, px + 8, py - 4);
        }
      }
    });

    // Draw Hand-Local Orthonormal Axes at Wrist
    if (this.showAxes && pts.length >= 18) {
      const [wx, wy] = toCanvas(pts[0]);
      const [mx, my] = toCanvas(pts[9]);
      const [ix, iy] = toCanvas(pts[5]);
      const [px, py] = toCanvas(pts[17]);

      // Y Axis (Green: Longitudinal along middle finger)
      const vy_x = (mx - wx) * 0.45;
      const vy_y = (my - wy) * 0.45;

      // X Axis (Red: Lateral across knuckles)
      const vx_x = (px - ix) * 0.45;
      const vx_y = (py - iy) * 0.45;

      // Draw Y axis (Longitudinal)
      ctx.strokeStyle = "#22c55e";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(wx, wy);
      ctx.lineTo(wx + vy_x, wy + vy_y);
      ctx.stroke();

      // Draw X axis (Lateral)
      ctx.strokeStyle = "#ef4444";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(wx, wy);
      ctx.lineTo(wx + vx_x, wy + vx_y);
      ctx.stroke();

      // Labels
      ctx.fillStyle = "#22c55e";
      ctx.font = "bold 11px monospace";
      ctx.fillText("+Y", wx + vy_x + 4, wy + vy_y + 4);
      ctx.fillStyle = "#ef4444";
      ctx.fillText("+X", wx + vx_x + 4, wy + vx_y + 4);
    }
  }

  _setIdleStates() {
    this.digitNames.forEach(d => {
      const el = this.digitElements[d];
      if (el.pill) {
        el.pill.textContent = "WAITING";
        el.pill.className = "state-pill state-uncertain";
      }
      if (el.rext) el.rext.textContent = "0.00";
      if (el.gaugeBar) el.gaugeBar.style.width = "0%";
      if (el.mcp) el.mcp.textContent = "--";
      if (el.pip) el.pip.textContent = "--";
      if (el.dip) el.dip.textContent = "--";
      if (el.diag) el.diag.textContent = "Awaiting hand observation...";
    });

    if (this.canonicalSummary) {
      this.canonicalSummary.textContent = "Thumb:    --\nIndex:    --\nMiddle:   --\nRing:     --\nLittle:   --";
    }
    if (this.synthesizedPoseLabel) this.synthesizedPoseLabel.textContent = "IDLE";
    if (this.synthesisConf) this.synthesisConf.textContent = "CONF: 0.00";
    if (this.synthesizedPoseDesc) this.synthesizedPoseDesc.textContent = "Holding neutral hand posture";
  }
}

// Instantiate on page load
window.addEventListener("DOMContentLoaded", () => {
  window.fingerLab = new FingerStateLab();
});
