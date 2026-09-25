/**
 * Gestura v3 — Developer Telemetry HUD & Spatial Perception Visualizer.
 *
 * Displays:
 *   - Hand landmarks and skeleton connectivity
 *   - Palm centers
 *   - Velocity vectors
 *   - Dominant hand (Primary Hand)
 *   - Modifier hand
 *   - Gesture candidate
 *   - Intent confidence
 *   - Active gesture
 *   - Interaction state
 *   - FPS & latency
 *
 * Normal mode hides all debugging information. Toggle with 'D'.
 */

class DebugOverlay {
  constructor() {
    this.panel = document.getElementById("telemetry-panel");
    this.canvas = document.getElementById("debug-skeleton-canvas");
    this.ctx = this.canvas ? this.canvas.getContext("2d") : null;

    this.fpsValue     = document.getElementById("fps-value");
    this.latencyValue = document.getElementById("latency-value");
    this.stateValue   = document.getElementById("state-value");
    this.gestureValue = document.getElementById("gesture-value");
    this.commandValue = document.getElementById("command-value");
    this.cursorPosValue = document.getElementById("cursor-pos-value");

    // v3 telemetry elements
    this.primaryHandVal = document.getElementById("primary-hand-val");
    this.modifierHandVal = document.getElementById("modifier-hand-val");
    this.candidateGestureVal = document.getElementById("candidate-gesture-val");

    // Level 0 finger state telemetry elements
    this.fingerThumbVal = document.getElementById("finger-thumb-val");
    this.fingerIndexVal = document.getElementById("finger-index-val");
    this.fingerMiddleVal = document.getElementById("finger-middle-val");
    this.fingerRingVal = document.getElementById("finger-ring-val");
    this.fingerLittleVal = document.getElementById("finger-little-val");

    // Legacy confidence bar (intent)
    this.confFill  = document.getElementById("conf-bar-fill");
    this.confLabel = document.getElementById("conf-value-label");

    // v3 multi-confidence bars
    this._ensureV3Elements();

    this.visible = false;
    this._resizeCanvas();
    this._initKeybindings();
    window.addEventListener("resize", () => this._resizeCanvas());
  }

  _resizeCanvas() {
    if (this.canvas) {
      this.canvas.width = window.innerWidth;
      this.canvas.height = window.innerHeight;
    }
  }

  _ensureV3Elements() {
    if (!document.getElementById("conf-hand-fill")) {
      const tray = this.panel;
      if (!tray) return;

      const rows = [
        { id: "hand",     label: "HAND",     color: "#10b981" },
        { id: "gesture",  label: "GESTURE",  color: "#f59e0b" },
        { id: "intent",   label: "INTENT",   color: "#a855f7" },
        { id: "tracking", label: "TRACKING", color: "#00f0ff" },
      ];

      const section = document.createElement("div");
      section.id = "v3-confidence-section";
      section.style.cssText = "margin-top:10px; border-top:1px solid rgba(255,255,255,0.1); padding-top:8px;";
      section.innerHTML = `<div style="font-size:9px;color:#64748b;margin-bottom:6px;letter-spacing:1px;">CONFIDENCE MODEL</div>`;

      rows.forEach(r => {
        const row = document.createElement("div");
        row.style.cssText = "display:flex;align-items:center;gap:8px;margin-bottom:4px;";
        row.innerHTML = `
          <span style="font-size:9px;color:#94a3b8;width:52px;flex-shrink:0">${r.label}</span>
          <div style="flex:1;height:5px;background:rgba(255,255,255,0.07);border-radius:3px;overflow:hidden">
            <div id="conf-${r.id}-fill" style="height:100%;width:0%;background:${r.color};border-radius:3px;transition:width 0.1s"></div>
          </div>
          <span id="conf-${r.id}-label" style="font-size:9px;color:#e2e8f0;width:30px;text-align:right">0.00</span>
        `;
        section.appendChild(row);
      });

      // Feature contributions mini-table
      const featSection = document.createElement("div");
      featSection.id = "v3-features-section";
      featSection.style.cssText = "margin-top:8px; border-top:1px solid rgba(255,255,255,0.1); padding-top:6px;";
      featSection.innerHTML = `
        <div style="font-size:9px;color:#64748b;margin-bottom:4px;letter-spacing:1px;">MOTION & FEATURES</div>
        <div id="v3-features-body" style="font-size:9px;color:#94a3b8;line-height:1.8;font-family:monospace;"></div>
      `;

      tray.appendChild(section);
      tray.appendChild(featSection);
    }
  }

  _initKeybindings() {
    window.addEventListener("keydown", (e) => {
      if (e.key === "d" || e.key === "D") this.toggle();
    });
  }

  toggle() {
    this.visible = !this.visible;
    this.panel?.classList.toggle("hidden", !this.visible);
    this.canvas?.classList.toggle("hidden", !this.visible);
    if (!this.visible && this.ctx) {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  /**
   * Update all telemetry values & canvas overlay from a HMI packet.
   * @param {Object} packet Full WebSocket packet from Python backend.
   */
  update(packet) {
    if (!this.visible || !packet) return;

    // ── 1. Text & State Metrics ───────────────────────────────────────
    if (this.fpsValue)      this.fpsValue.textContent     = packet.fps ? packet.fps.toFixed(1) : "—";
    if (this.latencyValue)  this.latencyValue.textContent = packet.latency_ms ? `${packet.latency_ms.toFixed(1)} ms` : "—";
    if (this.stateValue)    this.stateValue.textContent   = packet.intent_state || "IDLE";
    if (this.gestureValue)  this.gestureValue.textContent = packet.active_gesture || "NONE";
    if (this.commandValue)  this.commandValue.textContent = packet.command?.command_type || "IDLE";

    const cmd = packet.command || {};
    if (this.primaryHandVal) {
      this.primaryHandVal.textContent = packet.dominant_hand || cmd.dominant_hand || cmd.handedness || "Right";
    }
    if (this.modifierHandVal) {
      this.modifierHandVal.textContent = packet.modifier_hand || cmd.modifier_hand || "None";
    }
    if (this.candidateGestureVal) {
      this.candidateGestureVal.textContent = packet.candidate_gesture || cmd.candidate_gesture || "NONE";
    }

    if (cmd.cursor_ndc) {
      const [nx, ny] = cmd.cursor_ndc;
      if (this.cursorPosValue) this.cursorPosValue.textContent = `(${nx.toFixed(2)}, ${ny.toFixed(2)})`;
    }

    // ── 2. Confidence Metrics ─────────────────────────────────────────
    const intentConf = cmd.intent_confidence ?? packet.intent_confidence ?? 0.0;
    if (this.confFill)  this.confFill.style.width = `${Math.round(intentConf * 100)}%`;
    if (this.confLabel) this.confLabel.textContent = intentConf.toFixed(2);

    const confs = {
      hand:     cmd.hand_confidence     ?? cmd.confidence_hand     ?? packet.hands?.[0]?.detection_confidence ?? 0,
      gesture:  cmd.gesture_confidence  ?? cmd.confidence_gesture  ?? intentConf,
      intent:   cmd.intent_confidence   ?? cmd.confidence_intent   ?? intentConf,
      tracking: cmd.tracking_confidence ?? cmd.confidence_tracking ?? 1.0,
    };

    for (const [key, val] of Object.entries(confs)) {
      const fill  = document.getElementById(`conf-${key}-fill`);
      const label = document.getElementById(`conf-${key}-label`);
      if (fill)  fill.style.width = `${Math.round(val * 100)}%`;
      if (label) label.textContent = val.toFixed(2);
    }

    // ── 2.5 Level 0 Finger States Readout ─────────────────────────────
    const fs = packet.hands?.[0]?.finger_states || {};
    if (this.fingerThumbVal) this.fingerThumbVal.textContent = fs["Thumb"] || "--";
    if (this.fingerIndexVal) this.fingerIndexVal.textContent = fs["Index"] || "--";
    if (this.fingerMiddleVal) this.fingerMiddleVal.textContent = fs["Middle"] || "--";
    if (this.fingerRingVal) this.fingerRingVal.textContent = fs["Ring"] || "--";
    if (this.fingerLittleVal) this.fingerLittleVal.textContent = fs["Little"] || "--";

    // ── 3. Feature Contributions Readout ──────────────────────────────
    const featBody = document.getElementById("v3-features-body");
    if (featBody && packet.hands?.[0]) {
      const hand = packet.hands[0];
      const fc = hand.feature_contributions || {};
      const lines = [];

      const featMap = {
        "pinch":  fc.pinch_score   ?? fc.s_pinch  ?? null,
        "grab":   fc.grab_score    ?? fc.s_grab   ?? null,
        "point":  fc.point_score   ?? fc.s_point  ?? null,
        "palm":   fc.open_palm_score ?? fc.s_palm ?? null,
        "speed":  fc.palm_speed    ?? null,
        "slap_c": fc.slap_consistency ?? null,
      };

      for (const [k, v] of Object.entries(featMap)) {
        if (v !== null) {
          const bar = "█".repeat(Math.min(8, Math.round(Math.abs(v) * 8))).padEnd(8, "░");
          lines.push(`${k.padEnd(6)} ${bar} ${v.toFixed(2)}`);
        }
      }
      featBody.textContent = lines.join("\n");
    }

    // ── 4. Render Hand Landmarks, Palm Centers & Velocity Vectors ─────
    this._renderPerceptionOverlay(packet);
  }

  _renderPerceptionOverlay(packet) {
    if (!this.ctx || !this.canvas) return;
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);

    const hands = packet.hands || [];
    if (hands.length === 0) return;

    // MediaPipe joint connectivity bones
    const bones = [
      [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
      [0, 5], [5, 6], [6, 7], [7, 8],       // Index
      [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
      [0, 13], [13, 14], [14, 15], [15, 16],// Ring
      [0, 17], [17, 18], [18, 19], [19, 20],// Pinky
      [5, 9], [9, 13], [13, 17],            // Palm knuckle base
    ];

    hands.forEach((hand, idx) => {
      const isPrimary = (hand.handedness === packet.dominant_hand) || (idx === 0 && !packet.modifier_hand);
      const mainColor = isPrimary ? "#00f0ff" : "#f59e0b";
      const landmarks = hand.landmarks_normalized || [];

      // Draw Bones
      ctx.strokeStyle = isPrimary ? "rgba(0, 240, 255, 0.45)" : "rgba(245, 158, 11, 0.45)";
      ctx.lineWidth = 2.0;

      bones.forEach(([i, j]) => {
        if (landmarks[i] && landmarks[j]) {
          const [x1, y1] = landmarks[i];
          const [x2, y2] = landmarks[j];
          ctx.beginPath();
          ctx.moveTo(x1 * w, y1 * h);
          ctx.lineTo(x2 * w, y2 * h);
          ctx.stroke();
        }
      });

      // Draw Joint Landmarks
      landmarks.forEach(([x, y]) => {
        ctx.fillStyle = mainColor;
        ctx.beginPath();
        ctx.arc(x * w, y * h, 3.5, 0, Math.PI * 2);
        ctx.fill();
      });

      // Draw Palm Center
      if (hand.palm_center) {
        const [px, py] = hand.palm_center;
        const screenPx = px * w;
        const screenPy = py * h;

        ctx.strokeStyle = mainColor;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(screenPx, screenPy, 14, 0, Math.PI * 2);
        ctx.stroke();

        ctx.fillStyle = mainColor;
        ctx.beginPath();
        ctx.arc(screenPx, screenPy, 5, 0, Math.PI * 2);
        ctx.fill();

        // Draw Velocity Vector
        if (hand.palm_velocity) {
          const [vx, vy] = hand.palm_velocity;
          const arrowLen = 120; // scale factor for visual clarity
          const endX = screenPx + vx * arrowLen;
          const endY = screenPy + vy * arrowLen;

          ctx.strokeStyle = "#10b981";
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.moveTo(screenPx, screenPy);
          ctx.lineTo(endX, endY);
          ctx.stroke();

          // Arrow head
          const angle = Math.atan2(vy, vx);
          ctx.fillStyle = "#10b981";
          ctx.beginPath();
          ctx.moveTo(endX, endY);
          ctx.lineTo(endX - 10 * Math.cos(angle - Math.PI / 6), endY - 10 * Math.sin(angle - Math.PI / 6));
          ctx.lineTo(endX - 10 * Math.cos(angle + Math.PI / 6), endY - 10 * Math.sin(angle + Math.PI / 6));
          ctx.closePath();
          ctx.fill();
        }

        // Draw Hand Role Label
        ctx.font = "bold 11px monospace";
        ctx.fillStyle = mainColor;
        const roleText = isPrimary ? `PRIMARY (${hand.handedness})` : `MODIFIER (${hand.handedness})`;
        ctx.fillText(roleText, screenPx - 30, screenPy - 20);
      }
    });
  }
}
