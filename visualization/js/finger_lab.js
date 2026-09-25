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

    // DOM Elements - Sensor & Overlay Canvas
    this.feedContainer = document.querySelector(".feed-container");
    this.cameraFeed = document.getElementById("camera-screen-feed");
    this.canvas = document.getElementById("hand-skeleton-canvas");
    this.ctx = this.canvas ? this.canvas.getContext("2d") : null;
    this.sensorPlaceholder = document.getElementById("sensor-placeholder");

    // Display Toggles (Bearing Axes & Joint Angles)
    this.toggleAxesBtn = document.getElementById("toggle-axes-btn");
    this.toggleAnglesBtn = document.getElementById("toggle-angles-btn");
    this.showAxes = true;
    this.showAngles = true;

    // Kinematic Values
    this.kPitchVal = document.getElementById("k-pitch-val");
    this.kYawVal = document.getElementById("k-yaw-val");
    this.kRollVal = document.getElementById("k-roll-val");
    this.kScaleVal = document.getElementById("k-scale-val");

    // Canonical Terminal & 3-Tier Derivation Hierarchy
    this.canonicalSummary = document.getElementById("canonical-summary");
    this.configSummaryText = document.getElementById("config-summary-text");
    this.configPredicatesList = document.getElementById("config-predicates-list");
    this.synthesizedPoseLabel = document.getElementById("synthesized-pose-label");
    this.synthesizedPoseId = document.getElementById("synthesized-pose-id");
    this.synthesizedPoseDesc = document.getElementById("synthesized-pose-desc");
    this.synthesisConf = document.getElementById("synthesis-conf");

    // Prominent Defined Pose Elements (Header, Viewport HUD & Hero Card)
    this.headerPoseName = document.getElementById("header-pose-name");
    this.headerPoseId = document.getElementById("header-pose-id");
    this.headerLeftHandChip = document.getElementById("header-left-hand-chip");
    this.headerRightHandChip = document.getElementById("header-right-hand-chip");
    this.headerLeftPose = document.getElementById("header-left-pose");
    this.headerRightPose = document.getElementById("header-right-pose");

    // Hand Selector Tabs
    this.tabHandLeft = document.getElementById("hand-tab-left");
    this.tabHandRight = document.getElementById("hand-tab-right");
    this.tabLeftPose = document.getElementById("tab-left-pose");
    this.tabRightPose = document.getElementById("tab-right-pose");
    this.selectedHandSide = "Right";
    this.lastPacket = null;

    this.hudPoseName = document.getElementById("hud-pose-name");
    this.hudPoseId = document.getElementById("hud-pose-id");
    this.hudPoseConf = document.getElementById("hud-pose-conf");
    this.hudPoseFormula = document.getElementById("hud-pose-formula");

    this.heroPoseName = document.getElementById("hero-pose-name");
    this.heroPoseId = document.getElementById("hero-pose-id");
    this.heroPoseConf = document.getElementById("hero-pose-conf");
    this.heroPoseDesc = document.getElementById("hero-pose-desc");
    this.heroConfigSummary = document.getElementById("hero-config-summary");
    this.heroPredicatesList = document.getElementById("hero-predicates-list");

    // Camera Auto-Zoom HUD & Badge
    this.cameraZoomBadge = document.getElementById("camera-zoom-badge");
    this.cameraZoomHud = document.getElementById("camera-zoom-hud");
    this.cameraZoomHudVal = document.getElementById("camera-zoom-hud-val");

    // Controls
    this.recordBtn = document.getElementById("record-btn");
    this.recordBtnText = document.getElementById("record-btn-text");
    this.exportBtn = document.getElementById("export-btn");

    // State Flags
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

    this._initHandlers();
    this._initWebSocket();
  }

  _initHandlers() {
    if (this.recordBtn) {
      this.recordBtn.addEventListener("click", () => this._toggleRecording());
    }

    if (this.exportBtn) {
      this.exportBtn.addEventListener("click", () => this._exportDataset());
    }

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

    if (this.tabHandLeft) {
      this.tabHandLeft.addEventListener("click", () => {
        this.selectedHandSide = "Left";
        this.tabHandLeft.classList.add("active");
        if (this.tabHandRight) this.tabHandRight.classList.remove("active");
        if (this.lastPacket) this._processPacket(this.lastPacket);
      });
    }

    if (this.tabHandRight) {
      this.tabHandRight.addEventListener("click", () => {
        this.selectedHandSide = "Right";
        this.tabHandRight.classList.add("active");
        if (this.tabHandLeft) this.tabHandLeft.classList.remove("active");
        if (this.lastPacket) this._processPacket(this.lastPacket);
      });
    }

    window.addEventListener("resize", () => this._syncCanvasToFeed());
    if (this.cameraFeed) {
      this.cameraFeed.addEventListener("load", () => this._syncCanvasToFeed());
    }
  }

  _syncCanvasToFeed() {
    if (!this.canvas || !this.feedContainer || !this.cameraFeed) return;
    const cw = this.feedContainer.clientWidth;
    const ch = this.feedContainer.clientHeight;
    if (cw <= 0 || ch <= 0) return;

    // Use feed's natural dimensions or fallback to standard 640x480
    const nw = this.cameraFeed.naturalWidth > 0 ? this.cameraFeed.naturalWidth : 640;
    const nh = this.cameraFeed.naturalHeight > 0 ? this.cameraFeed.naturalHeight : 480;

    const imgAspect = nw / nh;
    const containerAspect = cw / ch;

    let renderedW, renderedH, offsetX, offsetY;

    if (containerAspect > imgAspect) {
      // Container is wider than the image: letterbox left & right
      renderedH = ch;
      renderedW = ch * imgAspect;
      offsetX = (cw - renderedW) / 2;
      offsetY = 0;
    } else {
      // Container is taller than the image: letterbox top & bottom
      renderedW = cw;
      renderedH = cw / imgAspect;
      offsetX = 0;
      offsetY = (ch - renderedH) / 2;
    }

    // Set internal resolution of canvas to match camera feed
    if (this.canvas.width !== nw) this.canvas.width = nw;
    if (this.canvas.height !== nh) this.canvas.height = nh;

    // Align canvas position and size directly over rendered video
    this.canvas.style.position = "absolute";
    this.canvas.style.left = `${offsetX}px`;
    this.canvas.style.top = `${offsetY}px`;
    this.canvas.style.width = `${renderedW}px`;
    this.canvas.style.height = `${renderedH}px`;
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
      if (this.sensorPlaceholder) {
        this.sensorPlaceholder.classList.remove("hidden");
        this.sensorPlaceholder.style.display = "flex";
      }
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
      if (this.sensorPlaceholder) {
        this.sensorPlaceholder.classList.add("hidden");
        this.sensorPlaceholder.style.display = "none";
      }
    }

    // Sync canvas overlay to feed dimensions & render skeleton with bearing
    this._syncCanvasToFeed();
    this._renderSkeleton(packet);

    // 2. Performance Stats
    if (this.fpsChip) this.fpsChip.textContent = `${packet.fps.toFixed(1)} FPS`;
    if (this.latencyChip) this.latencyChip.textContent = `${packet.latency_ms.toFixed(1)} MS`;

    // Camera Dynamic Auto-Zoom & Auto-Focus State
    const zoomVal = packet.camera_zoom || 1.0;
    const isZoomTracking = Boolean(packet.camera_zoom_tracking);

    if (this.cameraZoomBadge) {
      if (zoomVal > 1.08) {
        this.cameraZoomBadge.textContent = `ZOOM: ${zoomVal.toFixed(1)}x [FOCUS LOCK]`;
        this.cameraZoomBadge.classList.add("zooming");
      } else {
        this.cameraZoomBadge.textContent = "ZOOM: 1.0x [WIDE]";
        this.cameraZoomBadge.classList.remove("zooming");
      }
    }

    if (this.cameraZoomHud) {
      if (zoomVal > 1.08 && isZoomTracking) {
        this.cameraZoomHud.classList.remove("hidden");
        if (this.cameraZoomHudVal) {
          this.cameraZoomHudVal.textContent = `${zoomVal.toFixed(1)}x`;
        }
      } else {
        this.cameraZoomHud.classList.add("hidden");
      }
    }

    const hands = packet.hands || [];
    const numHands = hands.length;
    if (this.handCountChip) {
      this.handCountChip.textContent = numHands === 0 ? "AWAITING HAND..." : (numHands === 1 ? "1 HAND DETECTED" : `${numHands} HANDS DETECTED`);
      this.handCountChip.classList.toggle("chip-muted", numHands === 0);
      this.handCountChip.classList.toggle("chip-active", numHands > 0);
    }

    // 3. Check for detected hands
    if (numHands === 0) {
      this._setIdleStates();
      return;
    }

    // Multi-hand identification
    const leftHand = hands.find(h => h.handedness === "Left");
    const rightHand = hands.find(h => h.handedness === "Right");

    // Header multi-pose badges
    if (this.headerLeftHandChip) {
      this.headerLeftHandChip.classList.toggle("hidden", !leftHand);
      if (leftHand && this.headerLeftPose) {
        this.headerLeftPose.textContent = leftHand.hand_pose_name || "NONE";
      }
    }
    if (this.headerRightHandChip) {
      this.headerRightHandChip.classList.toggle("hidden", !rightHand);
      if (rightHand && this.headerRightPose) {
        this.headerRightPose.textContent = rightHand.hand_pose_name || "NONE";
      }
    }

    // Update hand selector tabs in UI
    if (this.tabLeftPose) {
      this.tabLeftPose.textContent = leftHand ? (leftHand.hand_pose_name || "NONE") : "--";
    }
    if (this.tabRightPose) {
      this.tabRightPose.textContent = rightHand ? (rightHand.hand_pose_name || "NONE") : "--";
    }

    // Auto-switch selected hand tab if only one hand is in view
    if (leftHand && !rightHand && this.selectedHandSide !== "Left") {
      this.selectedHandSide = "Left";
    } else if (rightHand && !leftHand && this.selectedHandSide !== "Right") {
      this.selectedHandSide = "Right";
    }

    if (this.tabHandLeft) this.tabHandLeft.classList.toggle("active", this.selectedHandSide === "Left");
    if (this.tabHandRight) this.tabHandRight.classList.toggle("active", this.selectedHandSide === "Right");

    // Primary hand to display on digit cards
    const primary = hands.find(h => h.handedness === this.selectedHandSide) || hands[0];
    const handedness = primary.handedness || "Right";
    const facingTag = primary.palm_facing || "PALM";
    const facingDesc = facingTag === "DORSAL" ? "BACK OF HAND" : (facingTag === "PALM" ? "FRONT OF HAND" : "EDGE-ON");
    if (this.dominantHandBadge) {
      this.dominantHandBadge.textContent = `${handedness.toUpperCase()} HAND • ${facingDesc}`;
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

      // If finger is uncertain / not in view: do NOT assume position or fake metrics
      if (state.toLowerCase() === "uncertain") {
        if (el.pill) {
          el.pill.textContent = "NOT IN VIEW";
          el.pill.className = "state-pill state-uncertain";
        }
        if (el.rext) el.rext.textContent = "--";
        if (el.gaugeBar) {
          el.gaugeBar.style.width = "0%";
          el.gaugeBar.style.backgroundColor = "#ef4444";
        }
        if (el.mcp) el.mcp.textContent = "--";
        if (el.pip) el.pip.textContent = "--";
        if (el.dip) el.dip.textContent = "--";
        if (el.diag) el.diag.textContent = "Not in view / occluded from sensor";
      } else {
        if (el.pill) {
          el.pill.textContent = state.toUpperCase();
          el.pill.className = `state-pill state-${state.toLowerCase()}`;
        }
        const rext = data.extension_ratio !== undefined ? data.extension_ratio : 1.0;
        if (el.rext) el.rext.textContent = rext.toFixed(2);
        if (el.gaugeBar) {
          const pct = Math.min(100, Math.max(0, ((rext - 0.6) / 1.2) * 100));
          el.gaugeBar.style.width = `${pct}%`;
          el.gaugeBar.style.backgroundColor = "";
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
      }
    });

    // 6. Tier 1: Update Canonical Level 0 Readout
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

    // 7. Tier 2: Update Finger Configuration (Topology & Predicates)
    const configSummary = primary.finger_config_summary || "Configuring digits...";
    if (this.configSummaryText) {
      this.configSummaryText.textContent = configSummary;
    }
    if (this.configPredicatesList) {
      const predicates = primary.pose_predicates || [];
      if (predicates.length > 0) {
        this.configPredicatesList.innerHTML = predicates
          .map(p => `<li>${p}</li>`)
          .join("");
      } else {
        this.configPredicatesList.innerHTML = `<li>Awaiting clear topological pattern...</li>`;
      }
    }

    // 8. Tier 3: Update Derived Level 1 Hand Pose
    const poseId = primary.hand_pose_id || "UNKNOWN";
    const poseName = primary.hand_pose_name || packet.active_gesture || "NONE";
    const intentConf = packet.intent_confidence || 0.0;

    if (this.synthesizedPoseLabel) {
      this.synthesizedPoseLabel.textContent = poseName;
    }
    if (this.synthesizedPoseId) {
      this.synthesizedPoseId.textContent = poseId;
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
        "OK_RING": "Thumb and Index tips touching with outer digits extended",
        "GUN": "Thumb up + Index forward in L-formation (Directional Snap)",
        "THUMBS_UP": "Isolated vertical upward thumb with closed fist (Affirm)",
        "THUMBS_DOWN": "Isolated vertical downward thumb with closed fist (Dismiss)",
        "DOUBLE_POINT": "Index and Middle extended parallel with thumb up",
        "CALL_ME_SHAKA": "Thumb and Pinky extended outward (Shaka)",
        "THREE_FINGER": "Index, Middle, and Ring extended outward",
        "CUPPED_HAND": "All digits semi-flexed forming concave bowl",
        "KNIFE_EDGE": "Digits extended and tightly adducted, edge-on",
        "KEY_PINCH": "Thumb pad pressed against lateral side of index",
        "NONE": "Transitioning posture or neutral resting state",
      };
      this.synthesizedPoseDesc.textContent = descMap[poseName] || `Derived Pose: ${poseName}`;
    }

    // 9. Update Header Pose Chip
    if (this.headerPoseName) {
      this.headerPoseName.textContent = poseName;
    }
    if (this.headerPoseId) {
      this.headerPoseId.textContent = poseId;
    }

    // 10. Update Floating Camera Viewport HUD
    if (this.hudPoseName) {
      this.hudPoseName.textContent = poseName;
    }
    if (this.hudPoseId) {
      this.hudPoseId.textContent = poseId;
    }
    if (this.hudPoseConf) {
      this.hudPoseConf.textContent = `CONF: ${intentConf.toFixed(2)}`;
    }
    if (this.hudPoseFormula) {
      this.hudPoseFormula.textContent = configSummary;
    }

    // 11. Update Hero Card at Top of Right Panel
    if (this.heroPoseName) {
      this.heroPoseName.textContent = poseName;
    }
    if (this.heroPoseId) {
      this.heroPoseId.textContent = poseId;
    }
    if (this.heroPoseConf) {
      this.heroPoseConf.textContent = `CONF: ${intentConf.toFixed(2)}`;
    }
    if (this.heroPoseDesc) {
      this.heroPoseDesc.textContent = descMap[poseName] || `Derived Pose: ${poseName}`;
    }
    if (this.heroConfigSummary) {
      this.heroConfigSummary.textContent = configSummary;
    }
    if (this.heroPredicatesList) {
      const predicates = primary.pose_predicates || [];
      if (predicates.length > 0) {
        this.heroPredicatesList.innerHTML = predicates
          .map(p => `<span class="pred-tag">${p}</span>`)
          .join("");
      } else {
        this.heroPredicatesList.innerHTML = `<span class="pred-tag">Awaiting clear topological pattern...</span>`;
      }
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

    // Convert normalized [x, y, z] to canvas pixel coordinates
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

    // Render EVERY detected hand independently
    hands.forEach((hand) => {
      const pts = hand.landmarks_normalized || [];
      if (pts.length < 21) return;

      const isLeft = hand.handedness === "Left";
      const wristAccent = isLeft ? "#00f0ff" : "#10b981"; // Cyan for Left, Emerald for Right
      const poseName = hand.hand_pose_name || "NONE";
      const states = hand.finger_states || {};
      const details = hand.finger_details || {};

      // 1. Draw Knuckle Base (Transverse arch connecting MCP joints)
      ctx.save();
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.40)";
      ctx.setLineDash([]);
      knuckleBase.forEach(([i1, i2]) => {
        const [x1, y1] = toCanvas(pts[i1]);
        const [x2, y2] = toCanvas(pts[i2]);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      });
      ctx.restore();

      // 2. Draw Finger Bones
      // Zero-hallucination: If finger is UNCERTAIN (not in view / occluded),
      // DO NOT draw solid tracked bones. Draw faint dashed line so user sees it is unobserved.
      fingerBones.forEach(({ name, joints }) => {
        const state = (states[name] || "uncertain").toLowerCase();
        const isUncertain = state === "uncertain" || (details[name]?.confidence === 0.0);
        const color = stateColors[state] || "#94a3b8";

        ctx.save();
        if (isUncertain) {
          ctx.strokeStyle = "rgba(239, 68, 68, 0.25)";
          ctx.lineWidth = 2;
          ctx.setLineDash([2, 4]); // Dashed ghost line for out-of-view digit
        } else {
          ctx.strokeStyle = color;
          ctx.lineWidth = 4;
          ctx.lineCap = "round";
          ctx.lineJoin = "round";
          ctx.setLineDash([]);
        }

        joints.forEach(([i1, i2]) => {
          const [x1, y1] = toCanvas(pts[i1]);
          const [x2, y2] = toCanvas(pts[i2]);
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        });
        ctx.restore();
      });

      // 3. Draw Joint Nodes
      pts.forEach((p, idx) => {
        const [px, py] = toCanvas(p);
        const isTip = [4, 8, 12, 16, 20].includes(idx);
        const isWrist = idx === 0;

        // Check if joint belongs to an uncertain finger
        let isUncertainJoint = false;
        if ([1, 2, 3, 4].includes(idx)) isUncertainJoint = (states["Thumb"] || "").toLowerCase() === "uncertain";
        else if ([5, 6, 7, 8].includes(idx)) isUncertainJoint = (states["Index"] || "").toLowerCase() === "uncertain";
        else if ([9, 10, 11, 12].includes(idx)) isUncertainJoint = (states["Middle"] || "").toLowerCase() === "uncertain";
        else if ([13, 14, 15, 16].includes(idx)) isUncertainJoint = (states["Ring"] || "").toLowerCase() === "uncertain";
        else if ([17, 18, 19, 20].includes(idx)) isUncertainJoint = (states["Little"] || "").toLowerCase() === "uncertain";

        ctx.beginPath();
        const r = isTip ? 6 : (isWrist ? 7 : 4);
        ctx.arc(px, py, r, 0, Math.PI * 2);

        if (isUncertainJoint) {
          ctx.fillStyle = "rgba(239, 68, 68, 0.3)";
          ctx.strokeStyle = "rgba(0, 0, 0, 0.4)";
          ctx.lineWidth = 1;
        } else {
          ctx.fillStyle = isTip ? "#ffffff" : (isWrist ? wristAccent : "#94a3b8");
          ctx.strokeStyle = "#000000";
          ctx.lineWidth = 1.5;
        }
        ctx.fill();
        ctx.stroke();

        // Show Joint Degrees if toggled (only for observed digits)
        if (this.showAngles && !isUncertainJoint && [6, 10, 14, 18, 3].includes(idx)) {
          const fingerMap = { 3: "Thumb", 6: "Index", 10: "Middle", 14: "Ring", 18: "Little" };
          const dName = fingerMap[idx];
          const detail = details[dName];
          const angle = detail ? (idx === 3 ? (detail.pip_deg !== undefined ? detail.pip_deg : detail.mcp_deg) : detail.pip_deg) : undefined;
          if (angle !== undefined && angle !== null) {
            ctx.save();
            ctx.font = "bold 11px monospace";
            ctx.fillStyle = "#000000";
            ctx.fillText(`${angle}°`, px + 9, py - 3);
            ctx.fillStyle = "#ffffff";
            ctx.fillText(`${angle}°`, px + 8, py - 4);
            ctx.restore();
          }
        }
      });

      // 4. Draw Hand-Local Orthonormal Axes (Bearing) at Wrist
      if (this.showAxes && pts.length >= 18) {
        const [wx, wy] = toCanvas(pts[0]);
        const [mx, my] = toCanvas(pts[9]);
        const [ix, iy] = toCanvas(pts[5]);
        const [px, py] = toCanvas(pts[17]);

        // Y Axis (Green: Longitudinal along middle finger direction)
        const vy_x = (mx - wx) * 0.55;
        const vy_y = (my - wy) * 0.55;

        // X Axis (Red: Lateral across knuckles)
        const vx_x = (px - ix) * 0.55;
        const vx_y = (py - iy) * 0.55;

        // Draw Y axis (Longitudinal bearing)
        ctx.save();
        ctx.strokeStyle = "#22c55e";
        ctx.lineWidth = 3.5;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(wx, wy);
        ctx.lineTo(wx + vy_x, wy + vy_y);
        ctx.stroke();

        // Draw X axis (Lateral bearing)
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 3.5;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(wx, wy);
        ctx.lineTo(wx + vx_x, wy + vx_y);
        ctx.stroke();

        // Axis Labels
        ctx.font = "bold 12px monospace";
        ctx.fillStyle = "#22c55e";
        ctx.fillText("+Y", wx + vy_x + 5, wy + vy_y + 4);
        ctx.fillStyle = "#ef4444";
        ctx.fillText("+X", wx + vx_x + 5, wy + vx_y + 4);
        ctx.restore();
      }

      // 5. Floating Identity & Pose Badge at Wrist
      const [wx, wy] = toCanvas(pts[0]);
      const facing = hand.palm_facing ? ` • ${hand.palm_facing}` : "";
      const badgeText = `${hand.handedness ? hand.handedness.toUpperCase() : "HAND"}: ${poseName}${facing}`;
      ctx.save();
      ctx.font = "bold 12px 'SF Pro Display', -apple-system, sans-serif";
      const textMetrics = ctx.measureText(badgeText);
      const badgeW = textMetrics.width + 18;
      const badgeH = 22;
      const badgeX = Math.max(8, Math.min(w - badgeW - 8, wx - badgeW / 2));
      const badgeY = Math.max(8, Math.min(h - badgeH - 8, wy + 16));

      // Badge pill background
      ctx.fillStyle = "rgba(10, 15, 29, 0.85)";
      ctx.strokeStyle = wristAccent;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      const r = 6;
      if (ctx.roundRect) {
        ctx.roundRect(badgeX, badgeY, badgeW, badgeH, r);
      } else {
        ctx.rect(badgeX, badgeY, badgeW, badgeH);
      }
      ctx.fill();
      ctx.stroke();

      // Hand side dot
      ctx.beginPath();
      ctx.arc(badgeX + 9, badgeY + badgeH / 2, 4, 0, Math.PI * 2);
      ctx.fillStyle = wristAccent;
      ctx.fill();

      // Badge text
      ctx.fillStyle = "#ffffff";
      ctx.fillText(badgeText, badgeX + 17, badgeY + 15);
      ctx.restore();
    });
  }

  _setIdleStates() {
    if (this.ctx && this.canvas) {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
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
    if (this.synthesizedPoseId) this.synthesizedPoseId.textContent = "UNKNOWN";
    if (this.synthesisConf) this.synthesisConf.textContent = "CONF: 0.00";
    if (this.synthesizedPoseDesc) this.synthesizedPoseDesc.textContent = "Holding neutral hand posture";

    if (this.headerPoseName) this.headerPoseName.textContent = "NONE";
    if (this.headerPoseId) this.headerPoseId.textContent = "UNKNOWN";

    if (this.hudPoseName) this.hudPoseName.textContent = "AWAITING HAND";
    if (this.hudPoseId) this.hudPoseId.textContent = "UNKNOWN";
    if (this.hudPoseConf) this.hudPoseConf.textContent = "CONF: 0.00";
    if (this.hudPoseFormula) this.hudPoseFormula.textContent = "Finger States → Finger Configuration → Hand Pose";

    if (this.heroPoseName) this.heroPoseName.textContent = "AWAITING HAND...";
    if (this.heroPoseId) this.heroPoseId.textContent = "UNKNOWN";
    if (this.heroPoseConf) this.heroPoseConf.textContent = "CONF: 0.00";
    if (this.heroPoseDesc) this.heroPoseDesc.textContent = "Holding neutral hand posture";
    if (this.heroConfigSummary) this.heroConfigSummary.textContent = "T:-- | I:-- | M:-- | R:-- | L:--";
    if (this.heroPredicatesList) {
      this.heroPredicatesList.innerHTML = `<span class="pred-tag">Place hand in front of camera to derive static pose...</span>`;
    }
  }
}

// Instantiate on page load
window.addEventListener("DOMContentLoaded", () => {
  window.fingerLab = new FingerStateLab();
});
