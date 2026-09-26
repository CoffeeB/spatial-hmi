/**
 * Gestura Level 0 Finger State Laboratory
 * Core client script dedicated to interpreting and visualizing individual finger states.
 */

const POSE_DESCRIPTIONS = {
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
    this.heroPoseIntent = document.getElementById("hero-pose-intent");
    this.heroIntendedAction = document.getElementById("hero-intended-action");

    // Camera Auto-Zoom HUD & Badge
    this.cameraZoomBadge = document.getElementById("camera-zoom-badge");
    this.cameraZoomHud = document.getElementById("camera-zoom-hud");
    this.cameraZoomHudVal = document.getElementById("camera-zoom-hud-val");

    // Level 3 Complete Gesture Elements
    this.cgCard = document.getElementById("complete-gesture-card");
    this.cgId = document.getElementById("cg-gesture-id");
    this.cgName = document.getElementById("cg-gesture-name");
    this.cgPhase = document.getElementById("cg-phase-badge");
    this.cgConf = document.getElementById("cg-conf-badge");
    this.cgCat = document.getElementById("cg-category-badge");
    this.cgDisp = document.getElementById("cg-metric-disp");
    this.cgSpeed = document.getElementById("cg-metric-speed");
    this.cgDur = document.getElementById("cg-metric-dur");
    this.cgEventChain = document.getElementById("cg-event-chain");
    this.cgTaskIntent = document.getElementById("cg-task-intent");
    this.cgNextIntent = document.getElementById("cg-next-intent");

    // Level 2 Motion Primitives: Dual-Hand Separate Tracking
    this.motionLeft = {
      card: document.getElementById("motion-card-left"),
      pill: document.getElementById("motion-dyn-pill-left"),
      intent: document.getElementById("motion-intent-left"),
      icon: document.getElementById("motion-icon-left"),
      token: document.getElementById("motion-token-left"),
      speed: document.getElementById("motion-speed-left"),
      accel: document.getElementById("motion-accel-left"),
      dir: document.getElementById("motion-dir-left"),
      lin: document.getElementById("motion-lin-left"),
      dur: document.getElementById("motion-dur-left"),
      dorsal: document.getElementById("dorsal-tag-left"),
    };

    this.motionRight = {
      card: document.getElementById("motion-card-right"),
      pill: document.getElementById("motion-dyn-pill-right"),
      intent: document.getElementById("motion-intent-right"),
      icon: document.getElementById("motion-icon-right"),
      token: document.getElementById("motion-token-right"),
      speed: document.getElementById("motion-speed-right"),
      accel: document.getElementById("motion-accel-right"),
      dir: document.getElementById("motion-dir-right"),
      lin: document.getElementById("motion-lin-right"),
      dur: document.getElementById("motion-dur-right"),
      dorsal: document.getElementById("dorsal-tag-right"),
    };

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
        focalPill: document.getElementById(`focal-pill-${lower}`),
        intentPill: document.getElementById(`intent-pill-${lower}`),
        motionPill: document.getElementById(`motion-pill-${lower}`),
        rext: document.getElementById(`rext-${lower}`),
        gaugeBar: document.getElementById(`gauge-bar-${lower}`),
        mcp: document.getElementById(`mcp-${lower}`),
        pip: document.getElementById(`pip-${lower}`),
        dip: document.getElementById(`dip-${lower}`),
        diag: document.getElementById(`diag-${lower}`),
      };
    });

    // Developer Debug Overlay: Temporal Intent Engine
    this.temporalOverlay = document.getElementById("temporal-intent-overlay");
    this.toggleTemporalBtn = document.getElementById("toggle-temporal-overlay-btn");
    this.closeOverlayBtn = document.getElementById("close-overlay-btn");

    this.intentFsmPill = document.getElementById("intent-fsm-pill");
    this.intentLockPill = document.getElementById("intent-lock-pill");
    this.overlayFps = document.getElementById("overlay-perf-fps");
    this.overlayLat = document.getElementById("overlay-perf-lat");

    // Finger Layer
    this.ofFingerStatus = document.getElementById("overlay-finger-status");
    this.ofThumb = document.getElementById("of-thumb");
    this.ofIndex = document.getElementById("of-index");
    this.ofMiddle = document.getElementById("of-middle");
    this.ofRing = document.getElementById("of-ring");
    this.ofLittle = document.getElementById("of-little");
    this.ofConf = document.getElementById("of-conf");
    this.ofJitter = document.getElementById("of-jitter");

    // Pose Layer
    this.opPoseStatus = document.getElementById("overlay-pose-status");
    this.opPoseName = document.getElementById("op-pose-name");
    this.opPoseConf = document.getElementById("op-pose-conf");
    this.opCandidate = document.getElementById("op-candidate");
    this.opFrames = document.getElementById("op-frames");

    // Motion Layer
    this.omMotionStatus = document.getElementById("overlay-motion-status");
    this.omSpeed = document.getElementById("om-speed");
    this.omDisp = document.getElementById("om-disp");
    this.omLin = document.getElementById("om-lin");
    this.omCons = document.getElementById("om-cons");
    this.omDir = document.getElementById("om-dir");
    this.omDur = document.getElementById("om-dur");

    // Intent Layer
    this.oiPriorityBadge = document.getElementById("overlay-priority-badge");
    this.oiGestureName = document.getElementById("oi-gesture-name");
    this.oiEvidenceScore = document.getElementById("oi-evidence-score");
    this.oiLockStatus = document.getElementById("oi-lock-status");
    this.oiEasingVal = document.getElementById("oi-easing-val");

    // Reasoning / Explainability
    this.overlayExplainText = document.getElementById("overlay-explain-text");

    // Developer Debug Overlay: Stability Engine
    this.stabilityOverlay = document.getElementById("stability-engine-overlay");
    this.toggleStabilityBtn = document.getElementById("toggle-stability-panel-btn");
    this.closeStabilityBtn = document.getElementById("close-stability-btn");
    this.stabCategoryPill = document.getElementById("stab-category-pill");
    this.stabLockPill = document.getElementById("stab-lock-pill");
    this.stabEligiblePill = document.getElementById("stab-eligible-pill");
    this.stabScoreHeader = document.getElementById("stab-score-header");
    this.stabLatencyVal = document.getElementById("stab-latency-val");

    // Finger Layer
    this.stabFingerStatus = document.getElementById("stab-finger-status");
    ["Thumb", "Index", "Middle", "Ring", "Little"].forEach((d) => {
      const lower = d.toLowerCase();
      this[`sf${d}State`] = document.getElementById(`sf-${lower}-state`);
      this[`sf${d}Stab`] = document.getElementById(`sf-${lower}-stab`);
      this[`sf${d}Conf`] = document.getElementById(`sf-${lower}-conf`);
      this[`sf${d}Age`] = document.getElementById(`sf-${lower}-age`);
    });

    // Pose Layer
    this.stabPoseStatus = document.getElementById("stab-pose-status");
    this.spPoseName = document.getElementById("sp-pose-name");
    this.spPoseScore = document.getElementById("sp-pose-score");
    this.spObsTimer = document.getElementById("sp-obs-timer");
    this.spPoseAge = document.getElementById("sp-pose-age");
    this.spEnterTh = document.getElementById("sp-enter-th");
    this.spExitTh = document.getElementById("sp-exit-th");

    // Motion Layer
    this.stabDeadzoneBadge = document.getElementById("stab-deadzone-badge");
    this.smVel = document.getElementById("sm-vel");
    this.smDisp = document.getElementById("sm-disp");
    this.smRadius = document.getElementById("sm-radius");
    this.smRamp = document.getElementById("sm-ramp");
    this.smDir = document.getElementById("sm-dir");
    this.smStatus = document.getElementById("sm-status");

    // Stability Layer
    this.stabLifecycleBadge = document.getElementById("stab-lifecycle-badge");
    this.pillMicro = document.getElementById("pill-micro");
    this.pillTrans = document.getElementById("pill-trans");
    this.pillStable = document.getElementById("pill-stable");
    this.pillLocked = document.getElementById("pill-locked");
    this.slWindows = document.getElementById("sl-windows");
    this.slTransInfo = document.getElementById("sl-trans-info");
    this.slHoldInfo = document.getElementById("sl-hold-info");

    // Intent Layer
    this.stabEligibilityBadge = document.getElementById("stab-eligibility-badge");
    this.siConfirmedGesture = document.getElementById("si-confirmed-gesture");
    this.siStabilityScore = document.getElementById("si-stability-score");
    this.siCandidateGesture = document.getElementById("si-candidate-gesture");
    this.siThreshold = document.getElementById("si-threshold");
    this.siFormulaBreakdown = document.getElementById("si-formula-breakdown");
    this.stabilityExplainStrip = document.querySelector(".stability-rejection-strip");
    this.stabilityExplainText = document.getElementById("stability-explain-text");

    this._initHandlers();
    this._initWebSocket();
  }

  _initHandlers() {
    if (this.toggleStabilityBtn && this.stabilityOverlay) {
      this.toggleStabilityBtn.addEventListener("click", () => {
        const isHidden = this.stabilityOverlay.classList.contains("hidden");
        if (isHidden) {
          this.stabilityOverlay.classList.remove("hidden");
          this.toggleStabilityBtn.classList.add("active");
        } else {
          this.stabilityOverlay.classList.add("hidden");
          this.toggleStabilityBtn.classList.remove("active");
        }
      });
    }

    if (this.closeStabilityBtn && this.stabilityOverlay) {
      this.closeStabilityBtn.addEventListener("click", () => {
        this.stabilityOverlay.classList.toggle("minimized");
        this.closeStabilityBtn.textContent = this.stabilityOverlay.classList.contains("minimized") ? "+" : "─";
      });
    }

    if (this.toggleTemporalBtn && this.temporalOverlay) {
      this.toggleTemporalBtn.addEventListener("click", () => {
        const isHidden = this.temporalOverlay.classList.contains("hidden");
        if (isHidden) {
          this.temporalOverlay.classList.remove("hidden");
          this.toggleTemporalBtn.classList.add("active");
        } else {
          this.temporalOverlay.classList.add("hidden");
          this.toggleTemporalBtn.classList.remove("active");
        }
      });
    }

    if (this.closeOverlayBtn && this.temporalOverlay) {
      this.closeOverlayBtn.addEventListener("click", () => {
        this.temporalOverlay.classList.toggle("minimized");
        this.closeOverlayBtn.textContent = this.temporalOverlay.classList.contains("minimized") ? "+" : "─";
      });
    }
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

    // 4. Update Kinematics Orientation & Level 2 Motion
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

    // Level 2 Motion Primitives Telemetry (Tracked separately per hand)
    this._updateMotionCard(this.motionLeft, leftHand, "Left");
    this._updateMotionCard(this.motionRight, rightHand, "Right");

    // 5. Update All 5 Digit Interpretation Cards
    const details = primary.finger_details || {};
    const states = primary.finger_states || {};

    this.digitNames.forEach(d => {
      const el = this.digitElements[d];
      const data = details[d] || {};
      const state = states[d] || data.state || "uncertain";

      // If finger is uncertain / not in view: do NOT assume position or fake metrics
      if (state.toLowerCase() === "uncertain") {
        if (el.focalPill) el.focalPill.style.display = "none";
        if (el.intentPill) el.intentPill.style.display = "none";
        if (el.card) el.card.classList.remove("is-focal");
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
        if (el.focalPill) {
          if (data.is_focal) {
            el.focalPill.style.display = "inline-block";
            const pct = Math.round((data.focus_weight || 0.9) * 100);
            el.focalPill.textContent = `🎯 FOCAL (${pct}%)`;
            if (el.card) el.card.classList.add("is-focal");
          } else {
            el.focalPill.style.display = "none";
            if (el.card) el.card.classList.remove("is-focal");
          }
        }
        if (el.intentPill) {
          if (data.intention) {
            el.intentPill.style.display = "inline-block";
            el.intentPill.textContent = data.intention.replace(/_/g, " ");
          } else {
            el.intentPill.style.display = "none";
          }
        }
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

      // Level 2 Finger Motion Primitive update
      const fMotions = primary.finger_motions || {};
      const fMotion = fMotions[d];
      if (el.motionPill) {
        if (state.toLowerCase() === "uncertain") {
          el.motionPill.textContent = "--";
          el.motionPill.className = "finger-motion-pill";
        } else if (fMotion && fMotion.primitive && fMotion.primitive !== "STATIONARY") {
          const prim = fMotion.primitive;
          const lowerPrim = prim.toLowerCase();
          const rateVal = fMotion.extension_rate;
          const rateStr = (rateVal !== undefined && Math.abs(rateVal) > 0.05) 
            ? ` ${rateVal > 0 ? "+" : ""}${rateVal.toFixed(2)}` 
            : "";
          const iconMap = {
            "EXTENDING": "↗",
            "FLEXING": "↘",
            "TAPPING": "⚡",
            "SWIPING": "↔",
            "HOLD": "⏸",
          };
          const icon = iconMap[prim] || "•";
          el.motionPill.textContent = `${icon} ${prim}${rateStr}`;
          el.motionPill.className = `finger-motion-pill ${lowerPrim}`;
        } else {
          el.motionPill.textContent = "⏹ STILL";
          el.motionPill.className = "finger-motion-pill";
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
      this.synthesizedPoseDesc.textContent = POSE_DESCRIPTIONS[poseName] || `Derived Pose: ${poseName}`;
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
      this.heroPoseDesc.textContent = POSE_DESCRIPTIONS[poseName] || `Derived Pose: ${poseName}`;
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

    // Pose Intention
    if (this.heroPoseIntent) {
      this.heroPoseIntent.textContent = primary.pose_intention || "RESTING_PALM";
    }
    if (this.heroIntendedAction) {
      this.heroIntendedAction.textContent = primary.intended_action || "";
    }

    // 12. Level 3 Complete Gesture (Pose + Motion Sequential Event Synthesis)
    this._updateCompleteGesture(primary);

    // 13. Level 4 & 5 Temporal Intent Engine Developer Overlay
    this._updateTemporalIntentOverlay(packet, primary);

    // 14. Level 3.5 Stability Engine Developer Overlay
    this._updateStabilityEngineOverlay(packet, primary);
  }

  _updateCompleteGesture(primary) {
    if (!this.cgCard) return;

    try {
      const rawCgId = primary.complete_gesture_id || "NONE";
      const rawCgName = primary.complete_gesture_name || "AWAITING GESTURE";
      const rawPhase = primary.gesture_phase || "IDLE";
      const rawCat = primary.gesture_category || "COMPOSITE";
      const seq = primary.gesture_event_sequence || [];
      const metrics = primary.gesture_metrics || {};
      const isCompleted = Boolean(primary.is_stroke_completed);

      const hasDirectGesture = rawCgId !== "NONE" && rawCgName !== "NEUTRAL TRANSITION";
      const poseName = primary.hand_pose_name || "NONE";
      const motionPrim = primary.motion_primitive || "STATIONARY";
      const hasTrackedHand = Boolean(primary && primary.landmarks_normalized && primary.landmarks_normalized.length >= 21);

      let cgId = rawCgId;
      let cgName = rawCgName;
      let phase = rawPhase;
      let cat = rawCat;

      if (!hasDirectGesture && hasTrackedHand && poseName !== "NONE" && poseName !== "UNKNOWN") {
        // Fallback to active composition state so card is alive and indicating real-time changes
        cgId = poseName;
        cgName = `${poseName} • ${motionPrim}`;
        phase = motionPrim === "STATIONARY" || primary.is_holding ? "HOLDING" : "TRACKING";
        cat = "COMPOSITION";
      }

      const hasActive = (cgId !== "NONE" && phase !== "IDLE" && phase !== "NEUTRAL") || (hasTrackedHand && poseName !== "NONE");

      // Card status class
      if (isCompleted) {
        this.cgCard.className = "complete-gesture-hero-card stroke-complete";
      } else if (hasActive) {
        this.cgCard.className = "complete-gesture-hero-card active";
      } else {
        this.cgCard.className = "complete-gesture-hero-card awaiting";
      }

      // Header labels & pills
      if (this.cgId) this.cgId.textContent = cgId !== "NONE" ? cgId.split("_")[0] : "NONE";
      if (this.cgName) this.cgName.textContent = cgName;
      if (this.cgCat) this.cgCat.textContent = cat.toUpperCase();
      if (this.cgConf) {
        const conf = primary.detection_confidence || 0.95;
        this.cgConf.textContent = `CONF: ${conf.toFixed(2)}`;
      }

      if (this.cgTaskIntent) {
        this.cgTaskIntent.textContent = primary.task_intent || "IDLE_MONITORING";
      }
      if (this.cgNextIntent) {
        this.cgNextIntent.textContent = primary.predicted_next_intent || "NONE";
      }

      if (this.cgPhase) {
        this.cgPhase.textContent = phase.toUpperCase();
        const lowerPhase = phase.toLowerCase();
        this.cgPhase.className = `cg-phase-pill phase-${lowerPhase}`;
      }

      // Metrics
      if (this.cgDisp) {
        const disp = metrics.displacement !== undefined ? metrics.displacement : (metrics.total_displacement !== undefined ? metrics.total_displacement : primary.motion_displacement);
        this.cgDisp.textContent = disp !== undefined && disp !== null ? `${Number(disp).toFixed(2)}m` : "--";
      }
      if (this.cgSpeed) {
        const spd = metrics.peak_speed !== undefined ? metrics.peak_speed : (metrics.speed !== undefined ? metrics.speed : primary.motion_speed);
        this.cgSpeed.textContent = spd !== undefined && spd !== null ? `${Number(spd).toFixed(2)} u/s` : "--";
      }
      if (this.cgDur) {
        const dur = metrics.duration_ms !== undefined ? metrics.duration_ms : (metrics.contact_duration_ms !== undefined ? metrics.contact_duration_ms : (metrics.dwell_duration_ms !== undefined ? metrics.dwell_duration_ms : primary.stroke_duration_ms));
        this.cgDur.textContent = dur !== undefined && dur !== null ? `${Math.round(Number(dur))}ms` : "--";
      }

      // Event sequence breadcrumbs
      if (this.cgEventChain) {
        if (seq.length > 0) {
          const html = seq.map((step, idx) => {
            const isCurrent = idx === seq.length - 1;
            const cls = isCompleted ? "seq-step-item completed" : (isCurrent ? "seq-step-item active" : "seq-step-item");
            const arrow = idx < seq.length - 1 ? `<span class="seq-arrow">➔</span>` : "";
            return `<span class="${cls}">${step}</span>${arrow}`;
          }).join(" ");
          this.cgEventChain.innerHTML = html;
        } else if (hasTrackedHand && poseName !== "NONE") {
          const fallbackChain = [poseName, motionPrim, isCompleted ? "COMPLETED" : "TRACKING"];
          const html = fallbackChain.map((step, idx) => {
            const isCurrent = idx === fallbackChain.length - 1;
            const cls = isCurrent ? "seq-step-item active" : "seq-step-item";
            const arrow = idx < fallbackChain.length - 1 ? `<span class="seq-arrow">➔</span>` : "";
            return `<span class="${cls}">${step}</span>${arrow}`;
          }).join(" ");
          this.cgEventChain.innerHTML = html;
        } else {
          this.cgEventChain.innerHTML = `<span class="seq-step-item active">OBSERVING</span>`;
        }
      }
    } catch (err) {
      console.warn("Error updating complete gesture card:", err);
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

      // 0. Draw Level 2 Kinetic Motion Trajectory Ribbon
      const traj = hand.trajectory_points || [];
      if (traj.length >= 2) {
        ctx.save();
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        for (let i = 1; i < traj.length; i++) {
          const [x1, y1] = toCanvas(traj[i - 1]);
          const [x2, y2] = toCanvas(traj[i]);
          const progress = i / traj.length;
          const alpha = progress * 0.70;
          ctx.strokeStyle = isLeft ? `rgba(0, 240, 255, ${alpha.toFixed(2)})` : `rgba(16, 185, 129, ${alpha.toFixed(2)})`;
          ctx.lineWidth = Math.max(1.5, progress * 5.0);
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        }
        ctx.restore();
      }

      // 1. Draw Knuckle Base (Transverse arch connecting MCP joints)
      const isDorsal = Boolean(hand.is_dorsal || hand.palm_facing === "DORSAL");
      ctx.save();
      ctx.lineWidth = isDorsal ? 4 : 3;
      ctx.strokeStyle = isDorsal ? "rgba(245, 158, 11, 0.75)" : "rgba(255, 255, 255, 0.40)";
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

        // Draw fingertip motion halo / indicator
        if (isTip && !isUncertainJoint) {
          const tipMap = { 4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Little" };
          const tipName = tipMap[idx];
          const fMot = (hand.finger_motions || {})[tipName];
          if (fMot && fMot.primitive && fMot.primitive !== "STATIONARY") {
            ctx.save();
            ctx.beginPath();
            ctx.arc(px, py, 11, 0, Math.PI * 2);
            if (fMot.primitive === "TAPPING") {
              ctx.strokeStyle = "#f472b6";
              ctx.fillStyle = "rgba(244, 114, 182, 0.35)";
              ctx.lineWidth = 2.5;
            } else if (fMot.primitive === "EXTENDING") {
              ctx.strokeStyle = "#00f0ff";
              ctx.fillStyle = "rgba(0, 240, 255, 0.25)";
              ctx.lineWidth = 2;
            } else if (fMot.primitive === "FLEXING") {
              ctx.strokeStyle = "#f59e0b";
              ctx.fillStyle = "rgba(245, 158, 11, 0.25)";
              ctx.lineWidth = 2;
            } else {
              ctx.strokeStyle = "#10b981";
              ctx.fillStyle = "rgba(16, 185, 129, 0.20)";
              ctx.lineWidth = 1.5;
            }
            ctx.fill();
            ctx.stroke();
            ctx.restore();
          }
        }

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
      const facingStr = isDorsal ? " • DORSAL" : (hand.palm_facing ? ` • ${hand.palm_facing}` : "");
      const badgeText = `${hand.handedness ? hand.handedness.toUpperCase() : "HAND"}: ${poseName}${facingStr}`;
      ctx.save();
      ctx.font = "bold 12px 'SF Pro Display', -apple-system, sans-serif";
      const textMetrics = ctx.measureText(badgeText);
      const badgeW = textMetrics.width + 18;
      const badgeH = 22;
      const badgeX = Math.max(8, Math.min(w - badgeW - 8, wx - badgeW / 2));
      const badgeY = Math.max(8, Math.min(h - badgeH - 8, wy + 16));

      // Badge pill background
      ctx.fillStyle = "rgba(10, 15, 29, 0.85)";
      ctx.strokeStyle = isDorsal ? "#f59e0b" : wristAccent;
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
      ctx.fillStyle = isDorsal ? "#f59e0b" : wristAccent;
      ctx.fill();

      // Badge text
      ctx.fillStyle = "#ffffff";
      ctx.fillText(badgeText, badgeX + 17, badgeY + 15);
      ctx.restore();

      // 6. Instantaneous Velocity Vector Arrow (from Palm Center)
      if (hand.motion_velocity && hand.motion_speed && hand.motion_speed > 0.08) {
        const [vx, vy, vz] = hand.motion_velocity;
        const [cx, cy] = toCanvas(pts[9]); // Middle MCP (center of palm)
        const arrowScale = 110.0; // scale factor in pixels per unit/s
        const ex = cx + vx * arrowScale;
        const ey = cy + vy * arrowScale;

        ctx.save();
        ctx.strokeStyle = wristAccent;
        ctx.fillStyle = wristAccent;
        ctx.lineWidth = 2.5;
        ctx.shadowColor = wristAccent;
        ctx.shadowBlur = 8;

        // Shaft
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(ex, ey);
        ctx.stroke();

        // Arrowhead
        const angle = Math.atan2(ey - cy, ex - cx);
        const headLen = 10;
        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - headLen * Math.cos(angle - Math.PI / 6), ey - headLen * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(ex - headLen * Math.cos(angle + Math.PI / 6), ey - headLen * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      }

      // 7. On-Canvas Floating Level 2 Motion Primitive Badge
      const mPrim = hand.motion_primitive || "STATIONARY";
      const mSpeed = hand.motion_speed || 0.0;
      const mDyn = (hand.motion_dynamic_state || "STATIONARY").toUpperCase();
      const mIcon = {
        "MOVE_LEFT": "←", "MOVE_RIGHT": "→", "MOVE_UP": "↑", "MOVE_DOWN": "↓",
        "MOVE_TOWARD": "⊕", "MOVE_AWAY": "⊖", "ROTATE_CW": "↻", "ROTATE_CCW": "↺",
        "FLIP_TO_DORSAL": "⤾", "FLIP_TO_PALM": "⤿",
        "HOLD": "⏸", "RELEASE": "⏏", "STATIONARY": "⏹",
      }[mPrim] || "•";

      const mBadgeText = `${mIcon} ${mPrim.replace("MOVE_", "").replace("FLIP_TO_", "FLIP ")} [${mSpeed.toFixed(2)} u/s]`;
      const mBadgeY = badgeY + badgeH + 5;
      if (mBadgeY + 18 < h - 4) {
        ctx.save();
        ctx.font = "bold 10px monospace";
        const mMetrics = ctx.measureText(mBadgeText);
        const mBadgeW = mMetrics.width + 16;
        const mBadgeX = Math.max(8, Math.min(w - mBadgeW - 8, wx - mBadgeW / 2));

        ctx.fillStyle = "rgba(10, 15, 29, 0.88)";
        ctx.strokeStyle = mDyn === "ACCELERATING" ? "#00f0ff" : (mDyn === "DECELERATING" ? "#f59e0b" : (mDyn === "STEADY" ? "#10b981" : "rgba(255, 255, 255, 0.25)"));
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(mBadgeX, mBadgeY, mBadgeW, 18, 4);
        else ctx.rect(mBadgeX, mBadgeY, mBadgeW, 18);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = wristAccent;
        ctx.fillText(mBadgeText, mBadgeX + 8, mBadgeY + 13);
        ctx.restore();
      }

      // 8. On-Canvas Floating Level 3 Complete Gesture Hero Badge & Completion Flash
      const cgIdStr = hand.complete_gesture_id;
      const cgNameStr = hand.complete_gesture_name;
      const cgPhaseStr = hand.gesture_phase || "IDLE";
      const isCompleteStroke = Boolean(hand.is_stroke_completed);

      if (cgNameStr && cgNameStr !== "NONE" && cgPhaseStr !== "IDLE") {
        const idPrefix = cgIdStr ? cgIdStr.split("_")[0] : "L3";
        const cgBadgeText = `⚡ [${idPrefix}] ${cgNameStr} • ${cgPhaseStr}`;

        ctx.save();
        ctx.font = "bold 11px monospace";
        const cgMetrics = ctx.measureText(cgBadgeText);
        const cgBadgeW = cgMetrics.width + 18;
        const cgBadgeH = 21;
        const cgBadgeX = Math.max(8, Math.min(w - cgBadgeW - 8, wx - cgBadgeW / 2));
        // Draw above wrist badge if room, else beneath L2 badge
        const cgBadgeY = badgeY - cgBadgeH - 6 >= 8 ? badgeY - cgBadgeH - 6 : (mBadgeY + 22);

        ctx.fillStyle = isCompleteStroke ? "rgba(16, 185, 129, 0.95)" : "rgba(6, 182, 212, 0.90)";
        ctx.strokeStyle = isCompleteStroke ? "#34d399" : "#00f0ff";
        ctx.lineWidth = 1.6;
        ctx.shadowColor = isCompleteStroke ? "rgba(16, 185, 129, 0.6)" : "rgba(0, 240, 255, 0.5)";
        ctx.shadowBlur = 10;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(cgBadgeX, cgBadgeY, cgBadgeW, cgBadgeH, 5);
        else ctx.rect(cgBadgeX, cgBadgeY, cgBadgeW, cgBadgeH);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = "#ffffff";
        ctx.fillText(cgBadgeText, cgBadgeX + 9, cgBadgeY + 15);
        ctx.restore();

        // Celebratory ripple pulse on stroke completion
        if (isCompleteStroke) {
          const [cx, cy] = toCanvas(pts[9]);
          ctx.save();
          ctx.beginPath();
          ctx.arc(cx, cy, 38, 0, Math.PI * 2);
          ctx.strokeStyle = "#34d399";
          ctx.lineWidth = 3;
          ctx.shadowColor = "#10b981";
          ctx.shadowBlur = 16;
          ctx.stroke();
          ctx.restore();
        }
      }
    });
  }

  _updateMotionCard(elements, hand, sideName) {
    if (!elements || !elements.card) return;
    if (!hand) {
      elements.card.classList.add("awaiting");
      elements.card.classList.remove("active");
      if (elements.pill) {
        elements.pill.textContent = "AWAITING";
        elements.pill.className = "motion-dyn-pill";
      }
      if (elements.icon) elements.icon.textContent = "⏹";
      if (elements.token) elements.token.textContent = "--";
      if (elements.speed) elements.speed.innerHTML = `-- <small>u/s</small>`;
      if (elements.accel) elements.accel.innerHTML = `-- <small>u/s²</small>`;
      if (elements.dir) elements.dir.textContent = "--";
      if (elements.lin) elements.lin.textContent = "--";
      if (elements.dur) elements.dur.innerHTML = `-- <small>ms</small>`;
      if (elements.dorsal) {
        elements.dorsal.textContent = "PALM";
        elements.dorsal.className = "dorsal-pill palm";
      }
      if (elements.intent) {
        elements.intent.textContent = "STATIC";
        elements.intent.style.background = "rgba(59, 130, 246, 0.15)";
        elements.intent.style.borderColor = "#3b82f6";
        elements.intent.style.color = "#93c5fd";
      }
      return;
    }

    elements.card.classList.remove("awaiting");
    elements.card.classList.add("active");

    if (elements.intent) {
      const mIntent = hand.motion_intention || "STATIC_POSTURE";
      elements.intent.textContent = mIntent.replace("MOTION_", "").replace(/_/g, " ");
      if (hand.is_purposeful) {
        elements.intent.style.background = "rgba(16, 185, 129, 0.2)";
        elements.intent.style.borderColor = "#10b981";
        elements.intent.style.color = "#6ee7b7";
      } else {
        elements.intent.style.background = "rgba(59, 130, 246, 0.15)";
        elements.intent.style.borderColor = "#3b82f6";
        elements.intent.style.color = "#93c5fd";
      }
    }

    const isDorsal = Boolean(hand.is_dorsal || hand.palm_facing === "DORSAL");
    const facingTag = isDorsal ? "DORSAL" : "PALM";
    if (elements.dorsal) {
      elements.dorsal.textContent = facingTag;
      elements.dorsal.className = `dorsal-pill ${facingTag.toLowerCase()}`;
    }

    const prim = hand.motion_primitive || "STATIONARY";
    const dyn = (hand.motion_dynamic_state || "STATIONARY").toLowerCase();
    const spd = hand.motion_speed || 0.0;
    const acc = hand.motion_tangential_accel || 0.0;
    const dir = hand.motion_direction || "STATIONARY";
    const secDir = hand.motion_secondary_direction;
    const lin = hand.motion_linearity !== undefined ? hand.motion_linearity : 1.0;
    const dur = prim === "HOLD" ? (hand.dwell_duration_ms || 0.0) : (hand.stroke_duration_ms || 0.0);

    const iconMap = {
      "MOVE_LEFT": "←",
      "MOVE_RIGHT": "→",
      "MOVE_UP": "↑",
      "MOVE_DOWN": "↓",
      "MOVE_TOWARD": "⊕",
      "MOVE_AWAY": "⊖",
      "ROTATE_CW": "↻",
      "ROTATE_CCW": "↺",
      "FLIP_TO_DORSAL": "⤾",
      "FLIP_TO_PALM": "⤿",
      "HOLD": "⏸",
      "RELEASE": "⏏",
      "STATIONARY": "⏹",
    };

    if (elements.icon) elements.icon.textContent = iconMap[prim] || "•";
    if (elements.token) elements.token.textContent = prim.replace("MOVE_", "").replace("FLIP_TO_", "FLIP ");

    if (elements.pill) {
      if (prim === "FLIP_TO_DORSAL" || prim === "FLIP_TO_PALM") {
        elements.pill.textContent = prim === "FLIP_TO_DORSAL" ? "FLIP ➜ DORSAL" : "FLIP ➜ PALM";
        elements.pill.className = "motion-dyn-pill flip";
      } else if (prim === "HOLD") {
        elements.pill.textContent = "HOLD";
        elements.pill.className = "motion-dyn-pill hold";
      } else if (prim === "RELEASE") {
        elements.pill.textContent = "RELEASE";
        elements.pill.className = "motion-dyn-pill release";
      } else {
        elements.pill.textContent = dyn.toUpperCase();
        elements.pill.className = `motion-dyn-pill ${dyn}`;
      }
    }

    if (elements.speed) elements.speed.innerHTML = `${spd.toFixed(2)} <small>u/s</small>`;
    if (elements.accel) {
      const sign = acc > 0.05 ? "+" : "";
      elements.accel.innerHTML = `${sign}${acc.toFixed(2)} <small>u/s²</small>`;
    }
    if (elements.dir) {
      elements.dir.textContent = secDir ? `${dir}·${secDir}` : dir;
    }
    if (elements.lin) {
      elements.lin.textContent = lin.toFixed(2);
    }
    if (elements.dur) {
      elements.dur.innerHTML = `${Math.round(dur)} <small>ms</small>`;
    }
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
      if (el.motionPill) {
        el.motionPill.textContent = "⏹ STILL";
        el.motionPill.className = "finger-motion-pill";
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

    // Reset Level 2 Motion Cards to idle awaiting
    this._updateMotionCard(this.motionLeft, null, "Left");
    this._updateMotionCard(this.motionRight, null, "Right");

    // Reset Level 3 Complete Gesture Hero Card
    if (this.cgCard) this.cgCard.className = "complete-gesture-hero-card awaiting";
    if (this.cgId) this.cgId.textContent = "NONE";
    if (this.cgName) this.cgName.textContent = "AWAITING GESTURE";
    if (this.cgPhase) {
      this.cgPhase.textContent = "IDLE";
      this.cgPhase.className = "cg-phase-pill";
    }
    if (this.cgConf) this.cgConf.textContent = "CONF: 0.00";
    if (this.cgCat) this.cgCat.textContent = "COMPOSITE";
    if (this.cgDisp) this.cgDisp.textContent = "--";
    if (this.cgSpeed) this.cgSpeed.textContent = "--";
    if (this.cgDur) this.cgDur.textContent = "--";
    if (this.cgEventChain) {
      this.cgEventChain.innerHTML = `<span class="seq-step-item active">OBSERVING</span>`;
    }

    this._resetTemporalIntentOverlay();
    this._resetStabilityEngineOverlay();
  }

  _updateTemporalIntentOverlay(packet, primary) {
    if (!this.temporalOverlay) return;

    try {
      const telem = packet.temporal_intent || (primary && primary.temporal_telemetry) || {};
      const fLayer = telem.finger_layer || {};
      const pLayer = telem.pose_layer || {};
      const mLayer = telem.motion_layer || {};
      const iLayer = telem.intent_layer || {};
      const perf = telem.performance || {};
      const diags = telem.diagnostics || [];

      // Header State & Lock
      const currentState = iLayer.current_state || packet.intent_state || "IDLE";
      if (this.intentFsmPill) {
        this.intentFsmPill.textContent = currentState;
        this.intentFsmPill.className = `intent-state-pill state-${currentState.toLowerCase()}`;
      }

      const isLocked = Boolean(iLayer.is_locked);
      const lockedBy = iLayer.locked_by || "NONE";
      if (this.intentLockPill) {
        if (isLocked) {
          this.intentLockPill.textContent = `LOCKED (${lockedBy})`;
          this.intentLockPill.className = "intent-lock-pill locked";
        } else {
          this.intentLockPill.textContent = "UNLOCKED";
          this.intentLockPill.className = "intent-lock-pill unlocked";
        }
      }

      if (this.overlayFps) {
        const fpsVal = perf.fps !== undefined ? perf.fps : packet.fps;
        this.overlayFps.textContent = `${(Number(fpsVal) || 60).toFixed(1)} FPS`;
      }
      if (this.overlayLat) {
        const latVal = perf.latency_ms !== undefined ? perf.latency_ms : packet.latency_ms;
        this.overlayLat.textContent = `${(Number(latVal) || 0).toFixed(1)}ms`;
      }

      // 1. Finger Layer (1-3F)
      const isStableFingers = Boolean(fLayer.is_stable);
      if (this.ofFingerStatus) {
        this.ofFingerStatus.textContent = isStableFingers ? "STABLE" : "OBSERVING";
        this.ofFingerStatus.className = `layer-status-pill ${isStableFingers ? "stable" : "jitter"}`;
      }

      const fStates = fLayer.states || (primary && primary.finger_states) || {};
      const formatState = (s) => {
        if (!s || s === "uncertain") return "--";
        const str = typeof s === "string" ? s : (s.state || s.value || String(s));
        const abbrevMap = {
          "extended": "EXT",
          "folded": "FLD",
          "curved": "CRV",
          "relaxed": "RLX",
          "tucked": "TCK",
          "hooked": "HOK",
          "touching": "TCH",
          "pinching": "PNC",
          "crossed": "CRS",
        };
        return abbrevMap[str.toLowerCase()] || str.slice(0, 3).toUpperCase();
      };

      const getDigitState = (name) => {
        return fStates[name] || fStates[name.toLowerCase()] || fStates[name.toUpperCase()] || fStates[name.charAt(0).toUpperCase() + name.slice(1).toLowerCase()];
      };

      if (this.ofThumb) this.ofThumb.textContent = formatState(getDigitState("Thumb"));
      if (this.ofIndex) this.ofIndex.textContent = formatState(getDigitState("Index"));
      if (this.ofMiddle) this.ofMiddle.textContent = formatState(getDigitState("Middle"));
      if (this.ofRing) this.ofRing.textContent = formatState(getDigitState("Ring"));
      if (this.ofLittle) this.ofLittle.textContent = formatState(getDigitState("Little"));

      if (this.ofConf) {
        const c = fLayer.confidence !== undefined ? fLayer.confidence : (primary ? primary.detection_confidence : 1.0);
        this.ofConf.textContent = (Number(c) || 0.0).toFixed(2);
      }
      if (this.ofJitter) {
        this.ofJitter.textContent = isStableFingers ? "<0.002" : ">0.015";
      }

      // 2. Pose Layer (3-5F Hysteresis)
      const isConfirmedPose = Boolean(pLayer.is_confirmed);
      const activePose = pLayer.active_pose || (primary ? primary.hand_pose_name : "NONE") || "NONE";
      const candidatePose = pLayer.candidate_pose || activePose;
      const poseConf = pLayer.confidence !== undefined ? pLayer.confidence : (packet.intent_confidence || 0.0);
      const framesCons = pLayer.frames_consistent !== undefined ? pLayer.frames_consistent : (isConfirmedPose ? 5 : 1);

      if (this.opPoseStatus) {
        this.opPoseStatus.textContent = isConfirmedPose ? "CONFIRMED" : "OBSERVING";
        this.opPoseStatus.className = `layer-status-pill ${isConfirmedPose ? "confirmed" : "observing"}`;
      }
      if (this.opPoseName) this.opPoseName.textContent = activePose;
      if (this.opPoseConf) this.opPoseConf.textContent = (Number(poseConf) || 0.0).toFixed(2);
      if (this.opCandidate) this.opCandidate.textContent = candidatePose;
      if (this.opFrames) this.opFrames.textContent = `${framesCons}/5`;

      // 3. Motion Layer (5-10F)
      const isIntentionalMotion = Boolean(mLayer.is_intentional);
      const prim = mLayer.primitive || (primary ? primary.motion_primitive : "STATIONARY") || "STATIONARY";
      const speed = mLayer.speed !== undefined ? mLayer.speed : (primary ? primary.motion_speed : 0.0);
      const disp = mLayer.displacement !== undefined ? mLayer.displacement : (primary ? primary.motion_displacement : 0.0);
      const lin = mLayer.linearity !== undefined ? mLayer.linearity : (primary ? primary.motion_linearity : 1.0);
      const cons = mLayer.direction_consistency !== undefined ? mLayer.direction_consistency : 1.0;
      const dir = mLayer.direction || (primary ? primary.motion_direction : "STATIONARY") || "STILL";
      const dur = mLayer.duration_ms !== undefined ? mLayer.duration_ms : (primary ? primary.stroke_duration_ms : 0.0);

      if (this.omMotionStatus) {
        this.omMotionStatus.textContent = isIntentionalMotion ? "INTENTIONAL" : (prim !== "STATIONARY" ? "MOVING" : "STILL");
        this.omMotionStatus.className = `layer-status-pill ${isIntentionalMotion ? "moving" : "stable"}`;
      }
      if (this.omSpeed) this.omSpeed.textContent = (Number(speed) || 0.0).toFixed(2);
      if (this.omDisp) this.omDisp.textContent = (Number(disp) || 0.0).toFixed(3);
      if (this.omLin) this.omLin.textContent = (Number(lin) || 1.0).toFixed(2);
      if (this.omCons) this.omCons.textContent = (Number(cons) || 1.0).toFixed(2);
      if (this.omDir) this.omDir.textContent = dir;
      if (this.omDur) this.omDur.textContent = `${Math.round(Number(dur) || 0)}ms`;

      // 4. Intent Layer (8-15F & Priority & Lock)
      const priorityTier = iLayer.priority_tier || "TIER_5_IDLE";
      const confGesture = iLayer.confirmed_gesture || packet.active_gesture || "NONE";
      const candGesture = iLayer.candidate_gesture || packet.candidate_gesture || "NONE";
      const intentConf = iLayer.confidence !== undefined ? iLayer.confidence : (packet.intent_confidence || 0.0);
      const easing = iLayer.easing_factor !== undefined ? iLayer.easing_factor : 1.0;

      if (this.oiPriorityBadge) {
        const tierShort = priorityTier.replace("TIER_", "T").replace("_", " ");
        this.oiPriorityBadge.textContent = tierShort;
      }
      if (this.oiGestureName) {
        if (confGesture !== "NONE" && confGesture !== "IDLE") {
          this.oiGestureName.textContent = confGesture;
        } else if (candGesture !== "NONE" && candGesture !== "IDLE") {
          this.oiGestureName.textContent = `~${candGesture}`;
        } else {
          this.oiGestureName.textContent = "NONE";
        }
      }
      if (this.oiEvidenceScore) this.oiEvidenceScore.textContent = (Number(intentConf) || 0.0).toFixed(2);
      if (this.oiLockStatus) this.oiLockStatus.textContent = isLocked ? `LOCKED: ${lockedBy}` : "UNLOCKED";
      if (this.oiEasingVal) this.oiEasingVal.textContent = (Number(easing) || 1.0).toFixed(2);

      // 5. Reasoning / Explainability Strip
      if (this.overlayExplainText) {
        if (diags.length > 0) {
          this.overlayExplainText.textContent = diags.join(" | ");
        } else if (currentState === "ACTIVE") {
          this.overlayExplainText.textContent = `Active continuous interaction (${confGesture}). Secondary gestures suppressed by Priority Engine.`;
        } else if (currentState === "CONFIRMED") {
          this.overlayExplainText.textContent = `Confirmed ${confGesture} (confidence ${(Number(intentConf) || 0).toFixed(2)} exceeds threshold). Dispatching command.`;
        } else if (currentState === "CANDIDATE") {
          this.overlayExplainText.textContent = `Accumulating temporal consensus for ${candGesture} (window: 8-15 frames).`;
        } else if (currentState === "OBSERVING") {
          this.overlayExplainText.textContent = `Observing hand posture (${activePose}). Waiting for multi-frame pose confirmation (3-5F).`;
        } else if (currentState === "RELEASING") {
          this.overlayExplainText.textContent = `Gracefully releasing interaction with easing decay (${(Number(easing) || 1).toFixed(2)}). Never snap abruptly.`;
        } else {
          this.overlayExplainText.textContent = "Idle neutral observation. Temporal Intent Engine awaiting stable intent.";
        }
      }
    } catch (err) {
      console.warn("Error updating temporal intent overlay:", err);
    }
  }

  _resetTemporalIntentOverlay(packet) {
    if (!this.temporalOverlay) return;

    if (this.intentFsmPill) {
      this.intentFsmPill.textContent = "IDLE";
      this.intentFsmPill.className = "intent-state-pill state-idle";
    }
    if (this.intentLockPill) {
      this.intentLockPill.textContent = "UNLOCKED";
      this.intentLockPill.className = "intent-lock-pill unlocked";
    }
    if (this.overlayFps && packet) {
      this.overlayFps.textContent = `${(packet.fps || 60).toFixed(1)} FPS`;
    }
    if (this.overlayLat && packet) {
      this.overlayLat.textContent = `${(packet.latency_ms || 0).toFixed(1)}ms`;
    }

    if (this.ofFingerStatus) {
      this.ofFingerStatus.textContent = "IDLE";
      this.ofFingerStatus.className = "layer-status-pill stable";
    }
    ["ofThumb", "ofIndex", "ofMiddle", "ofRing", "ofLittle"].forEach(k => {
      if (this[k]) this[k].textContent = "--";
    });
    if (this.ofConf) this.ofConf.textContent = "0.00";
    if (this.ofJitter) this.ofJitter.textContent = "--";

    if (this.opPoseStatus) {
      this.opPoseStatus.textContent = "AWAITING";
      this.opPoseStatus.className = "layer-status-pill observing";
    }
    if (this.opPoseName) this.opPoseName.textContent = "NONE";
    if (this.opPoseConf) this.opPoseConf.textContent = "0.00";
    if (this.opCandidate) this.opCandidate.textContent = "NONE";
    if (this.opFrames) this.opFrames.textContent = "0/5";

    if (this.omMotionStatus) {
      this.omMotionStatus.textContent = "STILL";
      this.omMotionStatus.className = "layer-status-pill stable";
    }
    if (this.omSpeed) this.omSpeed.textContent = "0.00";
    if (this.omDisp) this.omDisp.textContent = "0.00";
    if (this.omLin) this.omLin.textContent = "1.00";
    if (this.omCons) this.omCons.textContent = "1.00";
    if (this.omDir) this.omDir.textContent = "STILL";
    if (this.omDur) this.omDur.textContent = "0ms";

    if (this.oiPriorityBadge) this.oiPriorityBadge.textContent = "TIER 5";
    if (this.oiGestureName) this.oiGestureName.textContent = "NONE";
    if (this.oiEvidenceScore) this.oiEvidenceScore.textContent = "0.00";
    if (this.oiLockStatus) this.oiLockStatus.textContent = "UNLOCKED";
    if (this.oiEasingVal) this.oiEasingVal.textContent = "1.00";

    if (this.overlayExplainText) {
      this.overlayExplainText.textContent = "Awaiting hand presence. Temporal Intent Engine initialized in IDLE state.";
    }
  }

  _updateStabilityEngineOverlay(packet, primary) {
    if (!this.stabilityOverlay) return;

    try {
      const stab = packet.stability || (primary && primary.stability_telemetry) || {};
      const fLayer = stab.finger_layer || {};
      const pLayer = stab.pose_layer || {};
      const mLayer = stab.motion_layer || {};
      const sLayer = stab.stability_layer || {};
      const iLayer = stab.intent_layer || {};
      const diag = stab.diagnostics || {};

      // 1. Header Badges
      const category = sLayer.category || "INTENTIONAL";
      const isLocked = Boolean(sLayer.is_locked);
      const lockedInteraction = sLayer.locked_interaction || "NONE";
      const isEligible = Boolean(iLayer.is_command_eligible);
      const score = Number(iLayer.stability_score || 0.0);
      const thresh = Number(iLayer.stability_threshold || 0.90);

      if (this.stabCategoryPill) {
        this.stabCategoryPill.textContent = category;
        if (category === "MICRO_ADJUSTMENT") {
          this.stabCategoryPill.className = "intent-state-pill stab-micro";
        } else if (category === "TRANSITION") {
          this.stabCategoryPill.className = "intent-state-pill stab-transition";
        } else {
          this.stabCategoryPill.className = "intent-state-pill stab-intentional";
        }
      }

      if (this.stabLockPill) {
        if (isLocked) {
          this.stabLockPill.textContent = `LOCKED (${lockedInteraction})`;
          this.stabLockPill.className = "intent-lock-pill locked";
        } else {
          this.stabLockPill.textContent = "UNLOCKED";
          this.stabLockPill.className = "intent-lock-pill unlocked";
        }
      }

      if (this.stabEligiblePill) {
        this.stabEligiblePill.textContent = isEligible ? "ELIGIBLE" : "SUPPRESSED";
        this.stabEligiblePill.className = `intent-state-pill ${isEligible ? "stab-eligible" : "stab-suppressed"}`;
      }

      if (this.stabScoreHeader) {
        this.stabScoreHeader.textContent = score.toFixed(2);
        this.stabScoreHeader.style.color = score >= thresh ? "#10b981" : "#f59e0b";
      }

      if (this.stabLatencyVal) {
        this.stabLatencyVal.textContent = `${(diag.latency_ms || 0.8).toFixed(1)}ms`;
      }

      // 2. Finger Layer (State, Stability, Confidence, Age)
      const fingerStates = fLayer.states || {};
      const digits = ["Thumb", "Index", "Middle", "Ring", "Little"];
      digits.forEach((d) => {
        const info = fingerStates[d] || {};
        const stateEl = this[`sf${d}State`];
        const stabEl = this[`sf${d}Stab`];
        const confEl = this[`sf${d}Conf`];
        const ageEl = this[`sf${d}Age`];
        if (stateEl) stateEl.textContent = (info.state || "EXT").slice(0, 3).toUpperCase();
        if (stabEl) stabEl.textContent = (info.stability !== undefined ? Number(info.stability) : 1.0).toFixed(2);
        if (confEl) confEl.textContent = (info.confidence !== undefined ? Number(info.confidence) : 1.0).toFixed(2);
        if (ageEl) ageEl.textContent = `${info.age || 0}f`;
      });
      if (this.stabFingerStatus) {
        const meanStab = Number(fLayer.mean_stability || 1.0);
        this.stabFingerStatus.textContent = meanStab >= 0.88 ? "STABLE" : "OBSERVING";
        this.stabFingerStatus.className = `layer-status-pill ${meanStab >= 0.88 ? "stable" : "observing"}`;
      }

      // 3. Pose Layer (Pose, Observation Timer, Stability Score, Hysteresis)
      const poseName = pLayer.pose || (primary ? primary.hand_pose_name : "NONE") || "NONE";
      const poseScore = Number(pLayer.stability_score || 0.0);
      const obsTimer = Number(pLayer.observation_timer_ms || 0.0);
      const poseAge = pLayer.age || 0;
      const enterTh = Number(pLayer.enter_threshold || 0.88);
      const exitTh = Number(pLayer.exit_threshold || 0.62);

      if (this.spPoseName) this.spPoseName.textContent = poseName;
      if (this.spPoseScore) this.spPoseScore.textContent = poseScore.toFixed(2);
      if (this.spObsTimer) this.spObsTimer.textContent = `${Math.round(obsTimer)}ms`;
      if (this.spPoseAge) this.spPoseAge.textContent = `${poseAge}f`;
      if (this.spEnterTh) this.spEnterTh.textContent = enterTh.toFixed(2);
      if (this.spExitTh) this.spExitTh.textContent = exitTh.toFixed(2);
      if (this.stabPoseStatus) {
        const isStablePose = poseScore >= exitTh && poseName !== "NONE" && poseName !== "UNKNOWN";
        this.stabPoseStatus.textContent = isStablePose ? "STABLE" : "OBSERVING";
        this.stabPoseStatus.className = `layer-status-pill ${isStablePose ? "confirmed" : "observing"}`;
      }

      // 4. Motion Layer (Velocity, Direction, Dead-Zone Indicator, Displacement)
      const inDeadzone = Boolean(mLayer.in_dead_zone);
      const dzRadius = Number(mLayer.effective_dead_zone_px || 18.0);
      const dzRamp = Number(mLayer.smooth_ramp_factor !== undefined ? mLayer.smooth_ramp_factor : 0.0);
      const vel = mLayer.filtered_velocity || [0, 0, 0];
      const disp = Number(mLayer.filtered_displacement || 0.0);
      const velMag = Math.sqrt(vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);

      if (this.stabDeadzoneBadge) {
        this.stabDeadzoneBadge.textContent = inDeadzone ? "DEAD ZONE" : "ACTIVE MOTION";
        this.stabDeadzoneBadge.className = `layer-status-pill ${inDeadzone ? "deadzone-in" : "deadzone-out"}`;
      }
      if (this.smVel) this.smVel.textContent = velMag.toFixed(2);
      if (this.smDisp) this.smDisp.textContent = disp.toFixed(3);
      if (this.smRadius) this.smRadius.textContent = `${Math.round(dzRadius)}px`;
      if (this.smRamp) this.smRamp.textContent = dzRamp.toFixed(2);
      if (this.smDir) this.smDir.textContent = inDeadzone ? "STATIONARY" : (primary ? primary.motion_direction : "MOTION");
      if (this.smStatus) this.smStatus.textContent = inDeadzone ? "DEAD ZONE" : (dzRamp < 1.0 ? "RAMPING OUT" : "ACTIVE");

      // 5. Stability Layer (Lifecycle, Windows, Transition, Hold)
      if (this.stabLifecycleBadge) {
        this.stabLifecycleBadge.textContent = category;
        this.stabLifecycleBadge.className = `layer-status-pill ${category === "INTENTIONAL" ? "stable" : (category === "MICRO_ADJUSTMENT" ? "micro" : "observing")}`;
      }
      // Update lifecycle 4-pills row
      if (this.pillMicro) this.pillMicro.className = `stab-indicator-pill ${category === "MICRO_ADJUSTMENT" ? "active-micro" : ""}`;
      if (this.pillTrans) this.pillTrans.className = `stab-indicator-pill ${category === "TRANSITION" ? "active-trans" : ""}`;
      if (this.pillStable) this.pillStable.className = `stab-indicator-pill ${category === "INTENTIONAL" && !isLocked ? "active-pill" : ""}`;
      if (this.pillLocked) this.pillLocked.className = `stab-indicator-pill ${isLocked ? "active-locked" : ""}`;

      const win = sLayer.observation_windows || {};
      if (this.slWindows) {
        const wf = win.finger ? win.finger.window_size : 3;
        const wp = win.pose ? win.pose.window_size : 5;
        const wm = win.motion ? win.motion.window_size : 8;
        const wg = win.gesture ? win.gesture.window_size : 12;
        this.slWindows.textContent = `F:${wf} P:${wp} M:${wm} G:${wg}`;
      }
      const trans = sLayer.transition_info || {};
      if (this.slTransInfo) {
        if (trans.is_transitioning) {
          const pct = Math.round((trans.progress || 0.0) * 100);
          this.slTransInfo.textContent = `${trans.from_pose} ➔ ${trans.to_pose} (${pct}%)`;
        } else {
          this.slTransInfo.textContent = "STABLE (0%)";
        }
      }
      if (this.slHoldInfo) {
        const holdDur = Math.round(Number(sLayer.hold_duration_ms || 0));
        const targetHold = Math.round(Number(sLayer.target_hold_ms || 0));
        this.slHoldInfo.textContent = `${holdDur}ms / ${targetHold}ms`;
      }

      // 6. Intent Layer (Candidate, Confirmed, Stability Score, Eligibility)
      const candGest = iLayer.candidate_gesture || packet.candidate_gesture || "NONE";
      const confGest = packet.active_gesture || "NONE";
      if (this.siCandidateGesture) this.siCandidateGesture.textContent = candGest;
      if (this.siConfirmedGesture) this.siConfirmedGesture.textContent = confGest !== "NONE" ? confGest : `~${candGest}`;
      if (this.siStabilityScore) this.siStabilityScore.textContent = score.toFixed(2);
      if (this.siThreshold) this.siThreshold.textContent = thresh.toFixed(2);
      if (this.stabEligibilityBadge) {
        this.stabEligibilityBadge.textContent = isEligible ? "ELIGIBLE" : "SUPPRESSED";
        this.stabEligibilityBadge.className = `layer-status-pill ${isEligible ? "confirmed" : "observing"}`;
      }
      const breakdown = iLayer.breakdown || {};
      if (this.siFormulaBreakdown && breakdown.weighted_total !== undefined) {
        this.siFormulaBreakdown.textContent = `R:${(breakdown.recognition_weighted||0).toFixed(2)} + T:${(breakdown.temporal_weighted||0).toFixed(2)} + M:${(breakdown.motion_weighted||0).toFixed(2)} + P:${(breakdown.persistence_weighted||0).toFixed(2)}`;
      }

      // 7. Rejection Explainability Strip
      const reasons = iLayer.suppression_reasons || [];
      if (this.stabilityExplainStrip && this.stabilityExplainText) {
        if (reasons.length > 0) {
          this.stabilityExplainStrip.classList.add("has-rejections");
          this.stabilityExplainText.textContent = `REJECTED: ${reasons.join(" | ")}`;
        } else if (isEligible) {
          this.stabilityExplainStrip.classList.remove("has-rejections");
          this.stabilityExplainText.textContent = `INTENT CONFIRMED: Stability score (${score.toFixed(2)}) meets threshold (${thresh.toFixed(2)}). Motion is stable & intentional.`;
        } else {
          this.stabilityExplainStrip.classList.remove("has-rejections");
          this.stabilityExplainText.textContent = "Observing hand stability. Waiting for stable hold duration.";
        }
      }
    } catch (err) {
      console.warn("Error updating stability engine overlay:", err);
    }
  }

  _resetStabilityEngineOverlay(packet) {
    if (!this.stabilityOverlay) return;
    if (this.stabCategoryPill) {
      this.stabCategoryPill.textContent = "IDLE";
      this.stabCategoryPill.className = "intent-state-pill stab-intentional";
    }
    if (this.stabLockPill) {
      this.stabLockPill.textContent = "UNLOCKED";
      this.stabLockPill.className = "intent-lock-pill unlocked";
    }
    if (this.stabEligiblePill) {
      this.stabEligiblePill.textContent = "IDLE";
      this.stabEligiblePill.className = "intent-state-pill stab-suppressed";
    }
    if (this.stabScoreHeader) this.stabScoreHeader.textContent = "0.00";
    if (this.spPoseName) this.spPoseName.textContent = "NONE";
    if (this.spPoseScore) this.spPoseScore.textContent = "0.00";
    if (this.siConfirmedGesture) this.siConfirmedGesture.textContent = "NONE";
    if (this.siCandidateGesture) this.siCandidateGesture.textContent = "NONE";
    if (this.siStabilityScore) this.siStabilityScore.textContent = "0.00";
    if (this.stabilityExplainStrip && this.stabilityExplainText) {
      this.stabilityExplainStrip.classList.remove("has-rejections");
      this.stabilityExplainText.textContent = "Waiting for hand detection. Stability Engine monitoring sensor feed.";
    }
  }
}

// Instantiate on page load
window.addEventListener("DOMContentLoaded", () => {
  window.fingerLab = new FingerStateLab();
});
